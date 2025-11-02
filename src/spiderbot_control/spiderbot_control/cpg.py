# spiderbot_control/cpg.py
import math
import numpy as np

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# =========================
# Hopf CPG (unchanged math)
# =========================
class HopfCPG:
    def __init__(self, num_envs: int, num_oscillators: int, dt: float, device: str):
        self.num_envs = num_envs
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.device = device

        self.x = torch.zeros(num_envs, num_oscillators, device=device)
        self.y = torch.zeros(num_envs, num_oscillators, device=device)

        self.alpha = 8.0
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids: torch.Tensor):
        # Avoid deprecated advanced indexing on expanded tensors
        env_ids = torch.as_tensor(env_ids, device=self.x.device, dtype=torch.long).view(-1)
        self.x.index_fill_(0, env_ids, 0.99)
        self.y.index_fill_(0, env_ids, 0.0)

    def step(self, frequency: torch.Tensor, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # frequency: (N,1), amplitude: (N,12), phase: (N,12)
        omega = 2.0 * math.pi * frequency.expand(-1, self.num_oscillators)  # (N,12)
        r2 = self.x * self.x + self.y * self.y
        mu = 1.0

        dx_dt = self.alpha * (mu - r2) * self.x - omega * self.y
        dy_dt = self.alpha * (mu - r2) * self.y + omega * self.x

        self.x = self.x + dx_dt * self.dt
        self.y = self.y + dy_dt * self.dt

        # Shared internal phase across joints (training-time behavior)
        x0 = self.x[:, :1]
        y0 = self.y[:, :1]
        self.x = x0.expand(-1, self.num_oscillators)
        self.y = y0.expand(-1, self.num_oscillators)

        # Phase-shift for each joint
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        x_shifted = self.x * cos_phase - self.y * sin_phase

        # Output joint deltas (radians)
        return amplitude * x_shifted


class SpiderCPG:
    def __init__(self, num_envs: int, dt: float, device: str):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        self.cpg = HopfCPG(num_envs, num_oscillators=12, dt=dt, device=device)
        # Trot template: LF & RR in phase; RF & LR pi shifted
        self.default_leg_phases = torch.tensor([0.0, math.pi, 0.0, math.pi], device=device)

    def reset(self, env_ids: torch.Tensor):
        self.cpg.reset(env_ids)

    def compute_joint_targets(self, frequency: torch.Tensor, amplitudes: torch.Tensor, leg_phase_offsets: torch.Tensor) -> torch.Tensor:
        # leg_phase_offsets: (N,4) -> expand each leg offset to its 3 joints
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1)  # (N,12)
        default_phases = self.default_leg_phases.repeat_interleave(3).unsqueeze(0)  # (1,12)
        joint_phases = joint_phases + default_phases
        return self.cpg.step(frequency, amplitudes, joint_phases)


# ============================
# Thin wrapper for the node
# ============================
_G = {
    "cpg": None,        # SpiderCPG instance (num_envs=1)
    "device": "cpu",
}

# Training-time action ranges (raw policy actions are in [-1, 1])
F_MIN, F_MAX = 0.0, 2.0      # Hz
A_MIN, A_MAX = 0.0, 0.3      # rad
P_MIN, P_MAX = -0.5, 0.5     # rad


def _lazy_init(state: dict, dt_default: float = 0.01):
    """Create the SpiderCPG once (N=1)."""
    if not TORCH_OK:
        raise RuntimeError("PyTorch not available for CPG.")
    device = _G["device"]
    if _G["cpg"] is None:
        _G["cpg"] = SpiderCPG(num_envs=1, dt=dt_default, device=device)
        state["need_reset"] = True
        state.setdefault("last_t", None)


def _linmap(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map from [-1,1] to [lo,hi] elementwise."""
    return lo + 0.5 * (u + 1.0) * (hi - lo)


def actions_to_angles(actions17: np.ndarray, t: float, state: dict) -> np.ndarray:
    """
    Args:
      actions17: np.ndarray shape (17,) — raw policy actions in [-1,1]
      t:         current time (sec) from node
      state:     dict with persistent fields: 'last_t', 'need_reset'

    Returns:
      angles12: np.ndarray shape (12,) — joint *deltas* in radians
                (the controller should add DEFAULT_Q before publishing)
    """
    if not TORCH_OK:
        return np.zeros(12, dtype=np.float32)

    # Init once
    _lazy_init(state, dt_default=0.01)
    cpg = _G["cpg"]

    # Variable dt from wall/sim time
    last_t = state.get("last_t", None)
    if last_t is None:
        dt = 0.01
    else:
        dt = float(t - last_t)
        # Clamp to keep integrator stable on hiccups
        if not (1e-4 <= dt <= 0.1):
            dt = 0.01
            state["need_reset"] = True
    state["last_t"] = float(t)
    cpg.cpg.dt = dt

    # Reset if requested (first tick or large time jump)
    if state.get("need_reset", False):
        cpg.reset(torch.tensor([0], device=_G["device"]))
        state["need_reset"] = False

    # ---- Parse & MAP raw actions ([-1,1]) -> training ranges ----
    a = np.asarray(actions17, dtype=np.float32).reshape(-1)
    if a.shape[0] < 17:
        a = np.pad(a, (0, 17 - a.shape[0]))
    a = a[:17]

    freq_raw  = a[0:1]      # shape (1,)
    amp_raw   = a[1:13]     # shape (12,)
    phase_raw = a[13:17]    # shape (4,)

    freq  = _linmap(freq_raw,  F_MIN, F_MAX)    # -> (1,)
    amps  = _linmap(amp_raw,  A_MIN, A_MAX)     # -> (12,)
    phases= _linmap(phase_raw,P_MIN, P_MAX)     # -> (4,)

    # Safety bounds (idempotent with mapping)
    freq  = np.clip(freq,   F_MIN, F_MAX)
    amps  = np.clip(amps,   A_MIN, A_MAX)
    phases= np.clip(phases, P_MIN, P_MAX)

    # ---- To tensors (N=1) ----
    device = _G["device"]
    F = torch.tensor(freq,   dtype=torch.float32, device=device).view(1, 1)
    A = torch.tensor(amps,   dtype=torch.float32, device=device).view(1, 12)
    P = torch.tensor(phases, dtype=torch.float32, device=device).view(1, 4)

    with torch.no_grad():
        jt = cpg.compute_joint_targets(F, A, P)  # (1,12)

    return jt.squeeze(0).cpu().numpy().astype(np.float32)
