# SPDX-License-Identifier: BSD-3-Clause
# /home/teja/spiderbot/source/spiderbot/spiderbot/tasks/direct/spiderbot/cpg.py
from __future__ import annotations
import math
import torch

def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

class SmoothOpenLoopCPG:
    """Convert 12 amplitude actions -> 12 joint targets (open-loop, smooth, stop-on-zero)."""
    def __init__(self, num_envs: int, device: str, dt: float, cfg):
        self.num_envs = int(num_envs)
        self.device = device
        self.dt = float(dt)

        # timing & frequency from commands
        self.base_hz = float(cfg.phase_base_hz)
        self.k_v = float(cfg.phase_k_v)
        self.yaw_to_speed = float(cfg.cpg_yaw_to_speed_gain)

        # gating / envelopes / smoothing
        self.stop_deadband = float(cfg.cpg_stop_speed_deadband)
        self.full_stride = float(cfg.cpg_full_stride_speed)
        self.tau_env_move = float(cfg.cpg_envelope_tau_s)
        self.tau_env_stop = float(cfg.cpg_envelope_tau_stop_s)
        self.tau_out = float(cfg.cpg_output_tau_s)
        self.joint_limit = float(cfg.joint_target_limit_rad)

        coxa_max  = float(cfg.cpg_amp_coxa_max)
        femur_max = float(cfg.cpg_amp_femur_max)
        tibia_max = float(cfg.cpg_amp_tibia_max)
        self.amp_limits = torch.tensor([coxa_max, femur_max, tibia_max]*4,
                                       device=self.device).view(1, 12)

        j0, j1, j2 = cfg.cpg_joint_phase_offsets
        leg_bases = torch.tensor([0.0, math.pi, math.pi, 0.0],
                                 device=self.device).repeat_interleave(3)
        intra = torch.tensor([j0, j1, j2]*4, device=self.device)
        self.phase_template = (leg_bases + intra).view(1, 12)

        self.phi = torch.zeros(self.num_envs, 1, device=self.device)
        self.amp_env = torch.zeros(self.num_envs, 12, device=self.device)
        self.targets_smooth = torch.zeros(self.num_envs, 12, device=self.device)

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor, q0: torch.Tensor | None = None):
        self.phi[env_ids] = 0.0
        self.amp_env[env_ids] = 0.0
        if q0 is not None:
            self.targets_smooth[env_ids] = q0[env_ids]
        else:
            self.targets_smooth[env_ids] = 0.0

    @torch.no_grad()
    def phase_features(self):
        return torch.sin(self.phi), torch.cos(self.phi)

    @torch.no_grad()
    def step(self, commands: torch.Tensor, actions: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
        # frequency from command magnitude
        vxvy = torch.linalg.norm(commands[:, :2], dim=1, keepdim=True)
        wz = torch.abs(commands[:, 2:3]) * self.yaw_to_speed
        speed_eq = vxvy + wz
        freq = self.base_hz + self.k_v * speed_eq

        # phase
        self.phi = (self.phi + (2.0 * math.pi) * freq * self.dt) % (2.0 * math.pi)

        # gate (exact stop when command ~0)
        gate = _smoothstep01((speed_eq - self.stop_deadband) / max(1e-6, (self.full_stride - self.stop_deadband)))

        # actions -> amplitude targets
        amp_target = torch.tanh(actions) * self.amp_limits
        amp_target = amp_target * gate

        # smooth envelope (faster decay when stopping)
        use_stop = (gate[:, 0] < 0.25).float().view(-1, 1)
        tau = use_stop * self.tau_env_stop + (1.0 - use_stop) * self.tau_env_move
        alpha = 1.0 - torch.exp(-self.dt / torch.clamp(tau, min=1e-6))
        self.amp_env = self.amp_env + alpha * (amp_target - self.amp_env)

        # oscillator
        phi_joint = self.phi + self.phase_template  # (N,12) via broadcast
        wave = torch.sin(phi_joint)
        targets = q0 + self.amp_env * wave

        # output smoothing + clamp
        alpha_out = 1.0 - math.exp(-self.dt / max(1e-6, self.tau_out))
        self.targets_smooth = self.targets_smooth + alpha_out * (targets - self.targets_smooth)
        return torch.clamp(self.targets_smooth, -self.joint_limit, self.joint_limit)
