#!/usr/bin/env python3
"""
HopfCPG for ROS2 Deployment

This is the CORRECT CPG that matches training.
Port this to your spiderbot_control package.

Training used:
- HopfCPG with limit cycle dynamics (α=8.0)
- Diagonal coupling: k_phase=0.7, k_amp=1.0
- Trot gait template
- Joint-type scaling

Usage in policy_cpg_node.py:
    from spiderbot_control.cpg_hopf import SpiderCPG
    
    self.cpg = SpiderCPG(
        num_envs=1,
        dt=1.0/50.0,  # 50Hz
        device="cpu",
        k_phase=0.7,
        k_amp=1.0
    )
"""

import torch
import math


class HopfCPG:
    """
    Hopf oscillator with limit cycle dynamics.
    
    Dynamics:
        dx/dt = α(μ - r²)x - ωy
        dy/dt = α(μ - r²)y + ωx
    
    where r² = x² + y², converges to radius √μ = 1.0
    """
    
    def __init__(self, num_envs: int, dt: float, device: str, freq_floor_hz: float = 1e-3):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.freq_floor = freq_floor_hz
        
        # Oscillator state
        self.x = torch.zeros(num_envs, 1, device=device)
        self.y = torch.zeros(num_envs, 1, device=device)
        
        # Dynamics parameters
        self.alpha = 8.0  # Convergence rate to limit cycle
        
        # Initialize
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids: torch.Tensor):
        """Reset oscillator to near-cycle initial condition."""
        self.x[env_ids] = 0.99
        self.y[env_ids] = 0.0

    @torch.no_grad()
    def step(
        self,
        frequency: torch.Tensor,   # (N, 1) or (N, 12) in Hz
        amplitude: torch.Tensor,   # (N, 12) in radians
        phase: torch.Tensor        # (N, 12) in radians
    ) -> torch.Tensor:
        """
        Step the Hopf oscillator and generate joint deltas.
        
        Args:
            frequency: Oscillation frequency in Hz
            amplitude: Output amplitude per joint in radians
            phase: Phase offset per joint in radians
            
        Returns:
            Joint position deltas (N, 12) in radians
        """
        # Angular frequency
        omega = 2.0 * math.pi * torch.clamp(frequency, min=self.freq_floor)
        
        # Hopf dynamics
        r2 = self.x * self.x + self.y * self.y
        mu = 1.0  # Target radius
        
        dx_dt = self.alpha * (mu - r2) * self.x - omega * self.y
        dy_dt = self.alpha * (mu - r2) * self.y + omega * self.x
        
        # Euler integration
        self.x += dx_dt * self.dt
        self.y += dy_dt * self.dt
        
        # Apply phase shift and amplitude
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        x_shifted = self.x * cos_phase - self.y * sin_phase
        
        return amplitude * x_shifted

    @torch.no_grad()
    def phase_angle(self) -> torch.Tensor:
        """
        Return current oscillator phase angle φ in [-π, π].
        
        Returns:
            Phase angle (N, 1) in radians
        """
        return torch.atan2(self.y, self.x)


class SpiderCPG:
    """
    Spider robot CPG with diagonal coupling for trot gait.
    
    Features:
    - Hopf oscillator base
    - Trot phase template: FL+RR at 0°, FR+RL at 180°
    - Diagonal coupling: enforces FL↔RR and FR↔RL similarity
    - Joint-type scaling: coxa=1.0, femur=0.3, tibia=0.2
    - Intra-leg phase offsets: femur/tibia lead coxa by 90°
    """
    
    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: str,
        k_phase: float = 0.7,
        k_amp: float = 1.0
    ):
        """
        Args:
            num_envs: Number of parallel environments
            dt: Timestep in seconds
            device: 'cpu' or 'cuda'
            k_phase: Phase coupling strength [0,1]. 1.0 = full coupling
            k_amp: Amplitude coupling strength [0,1]. 1.0 = full coupling
        """
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.k_phase = float(k_phase)
        self.k_amp = float(k_amp)
        
        # Hopf oscillator
        self.cpg = HopfCPG(num_envs, dt=dt, device=device)

        # Trot phase template for legs: [FL, FR, RL, RR]
        # FL+RR = diagonal 0 (0°), FR+RL = diagonal 1 (180°)
        leg_phases = torch.tensor([0.0, math.pi, math.pi, 0.0], device=device)

        # Intra-leg phase offsets [coxa, femur, tibia]
        # Femur and tibia lead coxa by 90° for foot lift during swing
        intra = torch.tensor([
            0.0,  +0.5*math.pi,  +0.5*math.pi,  # FL
            0.0,  +0.5*math.pi,  +0.5*math.pi,  # FR
            0.0,  +0.5*math.pi,  +0.5*math.pi,  # RL
            0.0,  +0.5*math.pi,  +0.5*math.pi,  # RR
        ], device=device).unsqueeze(0)

        # Combine leg and intra-leg phases
        self.default_joint_phases = (
            leg_phases.repeat_interleave(3).unsqueeze(0) + intra
        )  # (1, 12)

        # Joint-type scaling [coxa, femur, tibia] × 4 legs
        # Coxa has largest range, tibia smallest (biological)
        self.joint_type_scales = torch.tensor([
            1.0, 0.3, 0.2,  # FL
            1.0, 0.3, 0.2,  # FR
            1.0, 0.3, 0.2,  # RL
            1.0, 0.3, 0.2,  # RR
        ], device=device).unsqueeze(0)  # (1, 12)

    def reset(self, env_ids: torch.Tensor):
        """Reset CPG state for given environments."""
        self.cpg.reset(env_ids)

    @torch.no_grad()
    def _couple_diagonal_phases(self, leg_phase_offsets: torch.Tensor) -> torch.Tensor:
        """
        Couple phase offsets across diagonal leg pairs.
        
        Args:
            leg_phase_offsets: (N, 4) [FL, FR, RL, RR]
            
        Returns:
            Coupled phase offsets (N, 4)
        """
        phi = leg_phase_offsets  # (N, 4)
        
        # Average diagonal pairs
        d0 = 0.5 * (phi[:, 0:1] + phi[:, 3:4])  # FL & RR
        d1 = 0.5 * (phi[:, 1:2] + phi[:, 2:3])  # FR & RL
        
        # Rebuild with averages
        coupled = torch.cat([d0, d1, d1, d0], dim=1)
        
        # Blend original and coupled
        return (1.0 - self.k_phase) * phi + self.k_phase * coupled

    @torch.no_grad()
    def _couple_diagonal_amplitudes(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """
        Couple amplitudes across diagonal leg pairs.
        
        Args:
            amplitudes: (N, 12) in [FL, FR, RL, RR] order
                        Each leg has 3 joints: [coxa, femur, tibia]
                        
        Returns:
            Coupled amplitudes (N, 12)
        """
        N = amplitudes.shape[0]
        amps = amplitudes.view(N, 4, 3)  # (N, 4legs, 3joints)
        
        # Average diagonal pairs
        # diagonal 0: legs 0 (FL) & 3 (RR)
        # diagonal 1: legs 1 (FR) & 2 (RL)
        avg0 = 0.5 * (amps[:, 0, :] + amps[:, 3, :])  # (N, 3)
        avg1 = 0.5 * (amps[:, 1, :] + amps[:, 2, :])  # (N, 3)
        
        k = self.k_amp
        
        # Blend each leg toward its diagonal average
        amps[:, 0, :] = (1 - k) * amps[:, 0, :] + k * avg0
        amps[:, 3, :] = (1 - k) * amps[:, 3, :] + k * avg0
        amps[:, 1, :] = (1 - k) * amps[:, 1, :] + k * avg1
        amps[:, 2, :] = (1 - k) * amps[:, 2, :] + k * avg1
        
        return amps.reshape(N, 12)

    def compute_joint_targets(
        self,
        frequency: torch.Tensor,        # (N, 1) Hz
        amplitudes: torch.Tensor,       # (N, 12) radians
        leg_phase_offsets: torch.Tensor # (N, 4) radians
    ) -> torch.Tensor:
        """
        Compute joint position deltas using CPG.
        
        Args:
            frequency: Oscillation frequency in Hz (N, 1)
            amplitudes: Joint amplitudes in radians (N, 12)
            leg_phase_offsets: Per-leg phase offsets in radians (N, 4)
            
        Returns:
            Joint position deltas in radians (N, 12)
            Add to default joint positions to get absolute targets.
        """
        # 1) Couple diagonal phases (FL↔RR, FR↔RL)
        leg_phase_offsets = self._couple_diagonal_phases(leg_phase_offsets)

        # 2) Expand to joint phases and add trot template
        joint_phases = (
            leg_phase_offsets.repeat_interleave(3, dim=1) +
            self.default_joint_phases
        )  # (N, 12)

        # 3) Couple diagonal amplitudes, then apply joint-type scaling
        amplitudes = self._couple_diagonal_amplitudes(amplitudes)
        scaled_amplitudes = amplitudes * self.joint_type_scales

        # 4) Hopf oscillator step → joint deltas
        return self.cpg.step(frequency, scaled_amplitudes, joint_phases)


# ============================================================================
# Example usage for testing
# ============================================================================
if __name__ == "__main__":
    print("Testing HopfCPG...")
    
    cpg = SpiderCPG(
        num_envs=1,
        dt=1.0/50.0,  # 50Hz
        device="cpu",
        k_phase=0.7,
        k_amp=1.0
    )
    
    # Test parameters (example from training)
    frequency = torch.tensor([[2.0]])  # 2 Hz
    amplitudes = torch.ones(1, 12) * 0.3  # 0.3 rad all joints
    leg_phases = torch.zeros(1, 4)  # No additional phase offset
    
    # Run a few steps
    for i in range(10):
        deltas = cpg.compute_joint_targets(frequency, amplitudes, leg_phases)
        phase = cpg.cpg.phase_angle()
        
        if i % 5 == 0:
            print(f"Step {i}:")
            print(f"  Phase: {phase[0, 0]:.3f} rad")
            print(f"  FL coxa delta: {deltas[0, 0]:.3f} rad")
            print(f"  FR coxa delta: {deltas[0, 3]:.3f} rad")
            print(f"  RL coxa delta: {deltas[0, 6]:.3f} rad")
            print(f"  RR coxa delta: {deltas[0, 9]:.3f} rad")
    
    print("\nTest complete! HopfCPG is working.")
    print("\nTo use in policy_cpg_node.py:")
    print("  from spiderbot_control.cpg_hopf import SpiderCPG")
    print("  self.cpg = SpiderCPG(num_envs=1, dt=self.dt, device='cpu')")