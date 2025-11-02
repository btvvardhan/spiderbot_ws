# spiderbot_control/cpg.py
import math
import numpy as np

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ===== Your training-time CPG (unchanged math) =====
# Source: user's uploaded training code
# - HopfCPG with shared internal phase across 12 joints
# - SpiderCPG that expands 4 leg phase offsets to 12 joints and adds trot template
# (kept verbatim, only trimmed comments)
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
        self.x[env_ids] = 0.99
        self.y[env_ids] = 0.0

    def step(self, frequency: torch.Tensor, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        omega = 2.0 * math.pi * frequency.expand(-1, self.num_oscillators)
        r2 = self.x * self.x + self.y * self.y
        mu = 1.0
        dx_dt = self.alpha * (mu - r2) * self.x - omega * self.y
        dy_dt = self.alpha * (mu - r2) * self.y + omega * self.x

        self.x = self.x + dx_dt * self.dt
        self.y = self.y + dy_dt * self.dt

        # shared internal phase
        x0 = self.x[:, :1]
        y0 = self.y[:, :1]
        self.x = x0.expand(-1, self.num_oscillators)
        self.y = y0.expand(-1, self.num_oscillators)

        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        x_shifted = self.x * cos_phase - self.y * sin_phase
        return amplitude * x_shifted


class SpiderCPG:
    def __init__(self, num_envs: int, dt: float, device: str):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        self.cpg = HopfCPG(num_envs, num_oscillators=12, dt=dt, device=device)
        self.default_leg_phases = torch.tensor([0.0, math.pi, 0.0, math.pi], device=device)

    def reset(self, env_ids: torch.Tensor):
        self.cpg.reset(env_ids)

    def compute_joint_targets(self, frequency: torch.Tensor, amplitudes: torch.Tensor, leg_phase_offsets: torch.Tensor) -> torch.Tensor:
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1)
        default_phases = self.default_leg_phases.repeat_interleave(3).unsqueeze(0)
        joint_phases = joint_phases + default_phases  # (N,12)
        return self.cpg.step(frequency, amplitudes, joint_phases)


# ===== Thin wrapper expected by cpgrl_controller =====
_G = {
    "cpg": None,       # SpiderCPG instance (num_envs=1)
    "device": "cpu",
    "last_t": None,    # last sim/wall time we stepped
}

def _lazy_init(state: dict, dt_default: float = 0.01):
    """Create the SpiderCPG once (N=1)."""
    if not TORCH_OK:
        raise RuntimeError("PyTorch not available for CPG.")

    device = _G["device"]
    if _G["cpg"] is None:
        _G["cpg"] = SpiderCPG(num_envs=1, dt=dt_default, device=device)
        # allow external reset flag
        state["need_reset"] = True

def actions_to_angles(actions17: np.ndarray, t: float, state: dict) -> np.ndarray:
    """
    Args:
      actions17: np.ndarray shape (17,)
      t:         current time (sec) from node
      state:     dict preserved across calls (we use last_t, need_reset)

    Returns:
      angles12: np.ndarray shape (12,) in radians
    """
    if not TORCH_OK:
        # safe fallback (zeros)
        return np.zeros(12, dtype=np.float32)

    # init (dt ~= 10 ms default; node runs ~100 Hz)
    _lazy_init(state, dt_default=0.01)
    cpg = _G["cpg"]

    # compute variable dt from time (t might be sim time)
    last_t = state.get("last_t", None)
    if last_t is None:
        dt = 0.01
    else:
        dt = float(t - last_t)
        # clamp to avoid instability on hiccups
        if not (1e-4 <= dt <= 0.1):
            dt = 0.01
            state["need_reset"] = True
    state["last_t"] = float(t)

    # update integrator dt on the fly (okay for this simple integrator)
    cpg.cpg.dt = dt

    # reset if requested (first tick or large time jump)
    if state.get("need_reset", False):
        cpg.reset(torch.tensor([0], device=_G["device"]))
        state["need_reset"] = False

    # parse actions -> freq, amps, leg phases
    a = np.asarray(actions17, dtype=np.float32).reshape(-1)
    if a.shape[0] < 17:
        a = np.pad(a, (0, 17 - a.shape[0]))

    freq = float(a[0])
    amps = a[1:13].astype(np.float32)       # 12
    legp = a[13:17].astype(np.float32)      # 4

    # safety clamps (tune to match your training bounds)
    freq = float(np.clip(freq, 0.2, 4.0))                 # Hz
    amps = np.clip(amps, 0.0, 1.5)                        # rad
    legp = np.clip(legp, -math.pi, math.pi)               # rad

    # to tensors with batch dim N=1
    F = torch.tensor([[freq]], dtype=torch.float32)
    A = torch.tensor(amps, dtype=torch.float32).unsqueeze(0)      # (1,12)
    P = torch.tensor(legp, dtype=torch.float32).unsqueeze(0)      # (1,4)

    with torch.no_grad():
        jt = cpg.compute_joint_targets(F, A, P)  # (1,12) tensor
    return jt.squeeze(0).cpu().numpy().astype(np.float32)
