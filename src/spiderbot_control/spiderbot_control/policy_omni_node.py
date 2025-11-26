#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Node with Dynamic Coxa Sign Control for Omnidirectional Motion

New approach: Dynamically adjust coxa joint signs based on commanded motion
to achieve forward/back/left/right/yaw without retraining policy!

Key insight: By changing coxa rotation directions, we can convert between:
- Walking straight (alternating signs: +-+-)
- Yawing (same signs: ++++ or ----)
- Lateral motion (different patterns)
"""

import os
import sys
import math
from typing import Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpg import SpiderCPG


DEFAULT_JOINT_ORDER = [
    "fl_coxa_joint", "fl_femur_joint", "fl_tibia_joint",
    "fr_coxa_joint", "fr_femur_joint", "fr_tibia_joint",
    "rl_coxa_joint", "rl_femur_joint", "rl_tibia_joint",
    "rr_coxa_joint", "rr_femur_joint", "rr_tibia_joint",
]


class ActorMLP(nn.Module):
    """Actor network with named layers (f1, f2, f3)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(32, 32), activation="elu"):
        super().__init__()
        self.f1 = nn.Linear(obs_dim, hidden[0])
        self.f2 = nn.Linear(hidden[0], hidden[1])
        self.f3 = nn.Linear(hidden[1], act_dim)
        self.act = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.f1(x))
        x = self.act(self.f2(x))
        return self.f3(x)


class ActorSeq(nn.Module):
    """Actor network with Sequential-style layers (0, 2, 4)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(32, 32), activation="elu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.Identity(),
            nn.Linear(hidden[0], hidden[1]),
            nn.Identity(),
            nn.Linear(hidden[1], act_dim),
        )
        self.act_fn = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0](x)
        x = self.act_fn(x)
        x = self.net[2](x)
        x = self.act_fn(x)
        x = self.net[4](x)
        return x


class PolicyCPGNode(Node):
    def __init__(self):
        super().__init__("policy_cpg_node")

        # Parameters
        self.declare_parameter("policy_pt", "/home/teja/spiderbot_ws/src/spiderbot_control/models/policy.pt")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("controller_command_topic", "/position_controller/commands")
        self.declare_parameter("rate_hz", 50.0)
        
        # Default joint positions (radians) - neutral stance
        self.declare_parameter("q0", [0.0] * 12)
        
        # NEW: Enable dynamic coxa sign control
        self.declare_parameter("enable_dynamic_signs", True)
        
        # Base sign pattern for straight walking (from your discovery!)
        # This is the pattern that makes it walk straight forward
        self.declare_parameter("base_joint_signs", [
            1.0, 1.0, 1.0,   # FL
           -1.0, 1.0, 1.0,   # FR (coxa flipped)
            1.0, 1.0, 1.0,   # RL
           -1.0, 1.0, 1.0    # RR (coxa flipped)
        ])
        
        # Zero command threshold
        self.declare_parameter("cmd_zero_threshold", 0.01)
        self.declare_parameter("standstill_on_zero_cmd", True)
        
        # CRITICAL: Must match training exactly!
        self.H = 5
        self.obs_dim = 37
        self.act_dim = 17

        self.device = torch.device("cpu")
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / self.rate_hz

        # State buffers
        self.cmd_now = torch.zeros(1, 3)
        self.cmd_hist = torch.zeros(1, self.H, 3)
        self.prev_actions = torch.zeros(1, self.act_dim)

        # Load policy
        self.actor = self._load_actor()
        
        # Initialize HopfCPG
        self.cpg = SpiderCPG(
            num_envs=1,
            dt=self.dt,
            device="cpu",
            k_phase=0.7,
            k_amp=1.0
        )
        
        # Default joint positions
        self.q0 = torch.tensor(
            self.get_parameter("q0").value,
            dtype=torch.float32
        ).view(1, 12)
        
        # Base sign pattern (for straight walking)
        self.base_signs = torch.tensor(
            self.get_parameter("base_joint_signs").value,
            dtype=torch.float32
        ).view(1, 12)
        
        self.cmd_zero_threshold = float(self.get_parameter("cmd_zero_threshold").value)
        self.standstill_enabled = bool(self.get_parameter("standstill_on_zero_cmd").value)
        self.dynamic_signs_enabled = bool(self.get_parameter("enable_dynamic_signs").value)
        
        # ROS I/O
        self.cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        self.joint_topic = self.get_parameter("joint_state_topic").get_parameter_value().string_value
        self.ctrl_topic = self.get_parameter("controller_command_topic").get_parameter_value().string_value
        
        self.sub = self.create_subscription(Twist, self.cmd_topic, self._cmd_cb, 10)
        self.pub = self.create_publisher(JointState, self.joint_topic, 10)
        self.ctrl_pub = self.create_publisher(Float64MultiArray, self.ctrl_topic, 10)
        self.timer = self.create_timer(self.dt, self._tick)
        
        # Track state
        self.is_standstill = False

        self.get_logger().info(
            f"✅ Policy+CPG node ready!\n"
            f"  📊 Observation: 37D = cmd(3) + hist(15) + phase(2) + prev_act(17)\n"
            f"  🎮 Action: 17D = freq(1) + amp(12) + phase(4)\n"
            f"  🔄 CPG: HopfCPG with diagonal coupling (k_phase=0.7, k_amp=1.0)\n"
            f"  🎯 Dynamic coxa signs: {self.dynamic_signs_enabled}\n"
            f"  ⏸️  Standstill on zero cmd: {self.standstill_enabled}\n"
            f"  ⚡ Rate: {self.rate_hz}Hz\n"
            f"  📁 Policy: {self.get_parameter('policy_pt').value}"
        )

    def _load_actor(self) -> nn.Module:
        """Load actor network from .pt file, handling both naming styles."""
        pt_path = self.get_parameter("policy_pt").get_parameter_value().string_value
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(f"❌ Policy file not found: {pt_path}")

        self.get_logger().info(f"📦 Loading policy from {pt_path}...")
        payload = torch.load(pt_path, map_location="cpu")
        
        state_dict = payload["actor_state_dict"]
        
        has_sequential = any(k.startswith(("0.", "2.", "4.")) for k in state_dict.keys())
        has_named = any(k.startswith(("f1.", "f2.", "f3.")) for k in state_dict.keys())
        
        if has_named:
            self.get_logger().info("  Detected named parameter style (f1, f2, f3)")
            actor = ActorMLP(obs_dim=37, act_dim=17, hidden=(32, 32), activation="elu").eval()
            actor.load_state_dict(state_dict, strict=True)
        elif has_sequential:
            self.get_logger().info("  Detected Sequential parameter style (0, 2, 4)")
            actor = ActorSeq(obs_dim=37, act_dim=17, hidden=(32, 32), activation="elu").eval()
            state_dict_prefixed = {f"net.{k}": v for k, v in state_dict.items()}
            actor.load_state_dict(state_dict_prefixed, strict=True)
        else:
            raise RuntimeError(f"❌ Unknown parameter naming style!")
        
        self.get_logger().info("✅ Policy loaded successfully!")
        return actor

    def _cmd_cb(self, msg: Twist):
        """Velocity command callback."""
        self.cmd_now[0, 0] = float(msg.linear.x)
        self.cmd_now[0, 1] = float(msg.linear.y)
        self.cmd_now[0, 2] = float(msg.angular.z)

    def _is_zero_command(self) -> bool:
        """Check if current command is effectively zero."""
        cmd_magnitude = torch.sqrt(torch.sum(self.cmd_now ** 2)).item()
        return cmd_magnitude < self.cmd_zero_threshold

    def _compute_dynamic_signs(self) -> torch.Tensor:
        """
        Dynamically compute joint signs based on commanded motion.
        
        Key insight:
        - Straight walking: alternating coxa signs [+1, -1, +1, -1]
        - CW yaw: all coxas same direction [+1, +1, +1, +1]
        - CCW yaw: all coxas opposite direction [-1, -1, -1, -1]
        - Blend between for curved motion
        
        Returns:
            Joint signs (1, 12) tensor
        """
        if not self.dynamic_signs_enabled:
            return self.base_signs.clone()
        
        # Extract commands
        vx = self.cmd_now[0, 0].item()  # forward/back
        vy = self.cmd_now[0, 1].item()  # left/right
        wz = self.cmd_now[0, 2].item()  # yaw rate
        
        # Start with base pattern (straight walking)
        signs = self.base_signs.clone()
        
        # Coxa indices: 0 (FL), 3 (FR), 6 (RL), 9 (RR)
        fl_coxa_sign = signs[0, 0].item()
        fr_coxa_sign = signs[0, 3].item()
        rl_coxa_sign = signs[0, 6].item()
        rr_coxa_sign = signs[0, 9].item()
        
        # Handle backward motion: coxa pattern should be -+-+
        if vx < -0.05:  # Backward threshold
            signs[0, 0] = -1.0  # FL
            signs[0, 3] = 1.0   # FR
            signs[0, 6] = -1.0  # RL
            signs[0, 9] = 1.0   # RR
            fl_coxa_sign = signs[0, 0].item()
            fr_coxa_sign = signs[0, 3].item()
            rl_coxa_sign = signs[0, 6].item()
            rr_coxa_sign = signs[0, 9].item()
        
        # If significant yaw command, blend toward uniform signs
        if abs(wz) > 0.05:  # threshold for yaw activation
            # Compute blend factor (0 = straight, 1 = pure yaw)
            linear_mag = math.sqrt(vx**2 + vy**2)
            yaw_mag = abs(wz)
            
            # When yaw is strong relative to linear, blend toward spinning
            if linear_mag > 0.01:
                yaw_ratio = yaw_mag / (yaw_mag + linear_mag)
            else:
                yaw_ratio = 1.0  # Pure yaw
            
            # Target signs for pure yaw
            if wz > 0:  # CCW yaw
                target_sign = -1.0
            else:  # CW yaw
                target_sign = 1.0
            
            # Blend coxa signs
            signs[0, 0] = fl_coxa_sign * (1 - yaw_ratio) + target_sign * yaw_ratio  # FL
            signs[0, 3] = fr_coxa_sign * (1 - yaw_ratio) + target_sign * yaw_ratio  # FR
            signs[0, 6] = rl_coxa_sign * (1 - yaw_ratio) + target_sign * yaw_ratio  # RL
            signs[0, 9] = rr_coxa_sign * (1 - yaw_ratio) + target_sign * yaw_ratio  # RR
        
        # Handle lateral motion (left/right)
        if abs(vy) > 0.05:
            # For lateral motion, front and rear need different patterns
            # Left: FL+RR forward, FR+RL backward
            # Right: opposite
            lateral_mag = abs(vy)
            linear_mag = math.sqrt(vx**2)
            
            if linear_mag > 0.01:
                lateral_ratio = lateral_mag / (lateral_mag + linear_mag)
            else:
                lateral_ratio = 1.0
            
            if vy > 0:  # Left
                # FL and RR should be same, FR and RL opposite
                signs[0, 0] = fl_coxa_sign * (1 - lateral_ratio) + 1.0 * lateral_ratio   # FL
                signs[0, 3] = fr_coxa_sign * (1 - lateral_ratio) + (-1.0) * lateral_ratio # FR
                signs[0, 6] = rl_coxa_sign * (1 - lateral_ratio) + (-1.0) * lateral_ratio # RL
                signs[0, 9] = rr_coxa_sign * (1 - lateral_ratio) + 1.0 * lateral_ratio   # RR
            else:  # Right
                signs[0, 0] = fl_coxa_sign * (1 - lateral_ratio) + (-1.0) * lateral_ratio # FL
                signs[0, 3] = fr_coxa_sign * (1 - lateral_ratio) + 1.0 * lateral_ratio    # FR
                signs[0, 6] = rl_coxa_sign * (1 - lateral_ratio) + 1.0 * lateral_ratio    # RL
                signs[0, 9] = rr_coxa_sign * (1 - lateral_ratio) + (-1.0) * lateral_ratio # RR
        
        return signs

    def _build_obs(self) -> torch.Tensor:
        """Build 37D observation vector matching training."""
        cpg_phase = self.cpg.cpg.phase_angle()
        sin_phase = torch.sin(cpg_phase)
        cos_phase = torch.cos(cpg_phase)
        
        obs = torch.cat([
            self.cmd_now,
            self.cmd_hist.reshape(1, -1),
            sin_phase,
            cos_phase,
            self.prev_actions,
        ], dim=-1)
        
        assert obs.shape[1] == 37, f"❌ Expected 37D obs, got {obs.shape[1]}D!"
        return obs

    def _tick(self):
        """Main control loop - runs at rate_hz."""
        
        # Check if we should be in standstill mode
        if self.standstill_enabled and self._is_zero_command():
            if not self.is_standstill:
                self.get_logger().info("⏸️  Zero command detected - entering standstill mode")
                self.is_standstill = True
            
            positions = [float(x) for x in self.q0.view(-1).tolist()]
            
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(DEFAULT_JOINT_ORDER)
            msg.position = positions
            self.pub.publish(msg)
            
            ctrl_msg = Float64MultiArray()
            ctrl_msg.data = positions
            self.ctrl_pub.publish(ctrl_msg)
            
            return
        
        # Exiting standstill mode
        if self.is_standstill:
            self.get_logger().info("▶️  Non-zero command detected - resuming motion")
            self.is_standstill = False
        
        # Normal operation: policy + CPG + dynamic signs
        # 1) Build 37D observations
        obs = self._build_obs()
        
        # 2) Policy forward pass → 17D actions
        with torch.no_grad():
            actions = self.actor(obs)
        
        # 3) Store for next observation
        self.prev_actions = actions.clone()
        
        # 4) Split actions into CPG parameters
        freq_raw = actions[:, 0:1]
        amp_raw = actions[:, 1:13]
        phase_raw = actions[:, 13:17]
        
        # 5) Denormalize to CPG ranges
        freq_hz = 0.5 + torch.sigmoid(freq_raw) * (3.0 - 0.5)
        amp_rad = 0.0 + torch.sigmoid(amp_raw) * (0.6 - 0.0)
        phase_rad = -1.2 + torch.sigmoid(phase_raw) * (1.2 - (-1.2))
        
        # 6) CPG computes joint deltas
        q_deltas = self.cpg.compute_joint_targets(
            frequency=freq_hz,
            amplitudes=amp_rad,
            leg_phase_offsets=phase_rad
        )
        
        # 7) Apply DYNAMIC sign corrections based on command
        joint_signs = self._compute_dynamic_signs()
        q_deltas = q_deltas * joint_signs
        
        # 8) Add default pose to get absolute targets
        q_targets = q_deltas + self.q0
        
        # 9) Convert to list for publishing
        positions = [float(x) for x in q_targets.view(-1).tolist()]
        
        # 10) Publish JointState
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(DEFAULT_JOINT_ORDER)
        msg.position = positions
        self.pub.publish(msg)
        
        # 11) Publish to controller
        ctrl_msg = Float64MultiArray()
        ctrl_msg.data = positions
        self.ctrl_pub.publish(ctrl_msg)
        
        # 12) Roll command history
        self.cmd_hist = torch.roll(self.cmd_hist, shifts=1, dims=1)
        self.cmd_hist[:, 0, :] = self.cmd_now


def main():
    rclpy.init()
    node = PolicyCPGNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Shutting down policy node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()