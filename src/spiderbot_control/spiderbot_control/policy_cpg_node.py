#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 node: teleop (/cmd_vel) -> policy (.pt) -> CPG
Publishes:
  - JointState (policy order [FL, FR, RL, RR]) for debugging/visualization
  - Float64MultiArray to ros2_control controller (e.g., /position_controller/commands) to MOVE robot in Gazebo

Obs (32 dims): [cmd_now(3), cmd_history(9*3), sin(phi), cos(phi)]
Loads exported actor-only .pt (supports Sequential keys "0.*" and named keys "f1.*").
Uses local SmoothOpenLoopCPG (place cpg.py in spiderbot_control/).

Run:
  ros2 run spiderbot_control policy_cpg_node --ros-args \
    -p policy_pt:=/home/teja/spiderbot/export/policy_actor_03000.pt \
    -p controller_command_topic:=/position_controller/commands \
    -p rate_hz:=50.0 -p apply_training_remap:=true
"""

import os
import sys
import math
import time
from typing import Tuple

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray  # <-- added

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Import the CPG locally (vendor file: spiderbot_control/cpg.py)
# ---------------------------------------------------------------------
try:
    from spiderbot_control.cpg import SmoothOpenLoopCPG
except Exception as e:
    raise ImportError(
        "Could not import SmoothOpenLoopCPG. Place cpg.py in your package as 'spiderbot_control/cpg.py'. "
        f"Import error: {e}"
    )

# ---------------------------------------------------------------------
# Joint order constants
# Policy/CPG & URDF order: [FL, FR, RL, RR]
# ---------------------------------------------------------------------
DEFAULT_JOINT_ORDER = [
    "fl_coxa_joint", "fl_femur_joint", "fl_tibia_joint",
    "fr_coxa_joint", "fr_femur_joint", "fr_tibia_joint",
    "rl_coxa_joint", "rl_femur_joint", "rl_tibia_joint",
    "rr_coxa_joint", "rr_femur_joint", "rr_tibia_joint",
]

# ---------------------------------------------------------------------
# Actor heads (both naming styles)
# ---------------------------------------------------------------------
class ActorMLP(nn.Module):
    """Actor with named layers f1/f2/f3."""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(32, 32), activation="elu", tanh_out=True):
        super().__init__()
        self.f1 = nn.Linear(obs_dim, hidden[0])
        self.f2 = nn.Linear(hidden[0], hidden[1])
        self.f3 = nn.Linear(hidden[1], act_dim)
        self.act = getattr(F, activation)
        self.tanh_out = tanh_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.f1(x))
        x = self.act(self.f2(x))
        x = self.f3(x)
        return torch.tanh(x) if self.tanh_out else x


class ActorSeq(nn.Module):
    """Actor with nn.Sequential-style param names: 0,2,4."""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(32, 32), activation="elu", tanh_out=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),  # 0.*
            nn.Identity(),                  # 1 (activation placeholder)
            nn.Linear(hidden[0], hidden[1]),# 2.*
            nn.Identity(),                  # 3
            nn.Linear(hidden[1], act_dim),  # 4.*
        )
        self.act_fn = getattr(F, activation)
        self.tanh_out = tanh_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0](x); x = self.act_fn(x)
        x = self.net[2](x); x = self.act_fn(x)
        x = self.net[4](x)
        return torch.tanh(x) if self.tanh_out else x


class PolicyCPGNode(Node):
    def __init__(self):
        super().__init__("policy_cpg_node")

        # -------- Parameters --------
        self.declare_parameter("policy_pt", "/home/teja/spiderbot/export/policy.pt")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("controller_command_topic", "/position_controller/commands")  # <-- added
        self.declare_parameter("rate_hz", 50.0)

        # Observation construction
        self.declare_parameter("history_len", 9)          # => 3*(9+1) + 2 = 32
        self.declare_parameter("use_phase", True)
        self.declare_parameter("apply_training_remap", True)  # vx'=-vy, vy'=-vx, wz'=wz

        # Actor meta (fallbacks if not in .pt)
        self.declare_parameter("obs_dim", 32)
        self.declare_parameter("act_dim", 12)
        self.declare_parameter("hidden", [32, 32])
        self.declare_parameter("activation", "elu")
        self.declare_parameter("actor_outputs_tanh", True)

        # CPG params (must match training)
        self.declare_parameter("phase_base_hz", 1.5)
        self.declare_parameter("phase_k_v", 1.0)
        self.declare_parameter("cpg_yaw_to_speed_gain", 0.30)

        self.declare_parameter("cpg_stop_speed_deadband", 0.02)
        self.declare_parameter("cpg_full_stride_speed", 0.35)
        self.declare_parameter("cpg_envelope_tau_s", 0.15)
        self.declare_parameter("cpg_envelope_tau_stop_s", 0.08)
        self.declare_parameter("cpg_output_tau_s", 0.03)
        self.declare_parameter("joint_target_limit_rad", 1.40)

        self.declare_parameter("cpg_amp_coxa_max", 0.60)
        self.declare_parameter("cpg_amp_femur_max", 0.35)
        self.declare_parameter("cpg_amp_tibia_max", 0.35)
        self.declare_parameter("cpg_joint_phase_offsets", [0.0, 0.5 * math.pi, 0.5 * math.pi])

        # default joint center pose (radians)
        self.declare_parameter("q0", [0.0] * 12)

        # -------- Internal state --------
        self.device = torch.device("cpu")
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / max(1e-6, self.rate_hz)

        self.H = int(self.get_parameter("history_len").value)
        self.use_phase = bool(self.get_parameter("use_phase").value)
        self.apply_remap = bool(self.get_parameter("apply_training_remap").value)

        self.cmd_now = torch.zeros(1, 3)
        self.cmd_hist = torch.zeros(1, self.H, 3)

        # dims (may be overridden by payload meta)
        self.obs_dim = int(self.get_parameter("obs_dim").value)
        self.act_dim = int(self.get_parameter("act_dim").value)

        # Load actor
        self.actor, self.obs_mean, self.obs_var = self._load_actor()

        # Build CPG cfg (simple proxy object)
        class CFG: pass
        cfg = CFG()
        for name in (
            "phase_base_hz", "phase_k_v", "cpg_yaw_to_speed_gain",
            "cpg_stop_speed_deadband", "cpg_full_stride_speed",
            "cpg_envelope_tau_s", "cpg_envelope_tau_stop_s", "cpg_output_tau_s",
            "joint_target_limit_rad",
            "cpg_amp_coxa_max", "cpg_amp_femur_max", "cpg_amp_tibia_max",
        ):
            setattr(cfg, name, float(self.get_parameter(name).value))
        offs = self.get_parameter("cpg_joint_phase_offsets").value
        setattr(cfg, "cpg_joint_phase_offsets", (float(offs[0]), float(offs[1]), float(offs[2])))

        self.cpg = SmoothOpenLoopCPG(num_envs=1, device="cpu", dt=self.dt, cfg=cfg)
        self.q0 = torch.tensor(self.get_parameter("q0").value, dtype=torch.float32).view(1, 12)

        # ROS I/O
        self.cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        self.joint_topic = self.get_parameter("joint_state_topic").get_parameter_value().string_value
        self.ctrl_topic = self.get_parameter("controller_command_topic").get_parameter_value().string_value  # <-- added
        self.sub = self.create_subscription(Twist, self.cmd_topic, self._cmd_cb, 10)
        self.pub = self.create_publisher(JointState, self.joint_topic, 10)
        self.ctrl_pub = self.create_publisher(Float64MultiArray, self.ctrl_topic, 10)  # <-- added
        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            f"Policy+CPG node up. rate={self.rate_hz:.1f}Hz, H={self.H}, use_phase={self.use_phase}, "
            f"apply_remap={self.apply_remap}, obs_dim={self.obs_dim}, act_dim={self.act_dim}\n"
            f"policy_pt={self.get_parameter('policy_pt').value}\n"
            f"Publishing JointState to {self.joint_topic} and controller commands to {self.ctrl_topic}."
        )

        self.add_on_set_parameters_callback(self._on_set_params)

    # -------------------- utils --------------------
    def _on_set_params(self, params):
        for p in params:
            if p.name == "apply_training_remap":
                self.apply_remap = bool(p.value)
            elif p.name == "rate_hz":
                self.rate_hz = float(p.value)
                self.dt = 1.0 / max(1e-6, self.rate_hz)
        return SetParametersResult(successful=True)

    def _load_actor(self) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        pt_path = self.get_parameter("policy_pt").get_parameter_value().string_value
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(f"policy_pt not found: {pt_path}")

        payload = torch.load(pt_path, map_location="cpu")
        meta = payload.get("meta", {}) or {}
        hidden = meta.get("hidden", self.get_parameter("hidden").value)
        activation = meta.get("activation", self.get_parameter("activation").value)
        tanh_head = bool(meta.get("expects_tanh", self.get_parameter("actor_outputs_tanh").value))

        # prefer dims from payload if present
        self.obs_dim = int(meta.get("obs_dim", self.obs_dim))
        self.act_dim = int(meta.get("act_dim", self.act_dim))

        sd = dict(payload["actor_state_dict"])  # copy
        has_sequential_top = any(k.startswith(("0.", "2.", "4.")) for k in sd.keys())
        has_named = any(k.startswith(("f1.", "f2.", "f3.")) for k in sd.keys())

        if has_named:
            actor = ActorMLP(self.obs_dim, self.act_dim, tuple(hidden), activation, tanh_out=tanh_head).eval()
            actor.load_state_dict(sd, strict=True)
        elif has_sequential_top:
            sd_prefixed = {f"net.{k}": v for k, v in sd.items()}
            actor = ActorSeq(self.obs_dim, self.act_dim, tuple(hidden), activation, tanh_out=tanh_head).eval()
            actor.load_state_dict(sd_prefixed, strict=True)
        else:
            actor = nn.Sequential(
                nn.Linear(self.obs_dim, hidden[0]),
                getattr(nn, activation.upper())() if activation in ("relu", "elu", "selu") else nn.ELU(),
                nn.Linear(hidden[0], hidden[1]),
                getattr(nn, activation.upper())() if activation in ("relu", "elu", "selu") else nn.ELU(),
                nn.Linear(hidden[1], self.act_dim),
                nn.Tanh() if tanh_head else nn.Identity(),
            ).eval()
            actor.load_state_dict(sd, strict=False)

        # Optional obs normalization
        obs_mean = payload.get("obs_mean", None)
        obs_var = payload.get("obs_var", None)
        if obs_mean is not None:
            obs_mean = torch.as_tensor(obs_mean, dtype=torch.float32)
        if obs_var is not None:
            obs_var = torch.as_tensor(obs_var, dtype=torch.float32)
        return actor, obs_mean, obs_var

    def _norm_obs(self, o: torch.Tensor) -> torch.Tensor:
        if self.obs_mean is None or self.obs_var is None:
            return o
        eps = 1e-6
        return (o - self.obs_mean) / torch.sqrt(self.obs_var + eps)

    @staticmethod
    def _remap_for_training(v: torch.Tensor) -> torch.Tensor:
        # training remap: vx'=-vy, vy'=-vx, wz'=wz
        vx, vy, wz = v[..., 0:1], v[..., 1:2], v[..., 2:3]
        return torch.cat([vy, vx, wz], dim=-1)

    # -------------------- ROS callbacks --------------------
    def _cmd_cb(self, msg: Twist):
        self.cmd_now[0, 0] = float(msg.linear.x)
        self.cmd_now[0, 1] = float(msg.linear.y)
        self.cmd_now[0, 2] = float(msg.angular.z)

    def _build_obs(self, cmd_for_policy: torch.Tensor) -> torch.Tensor:
        parts = [cmd_for_policy]
        if self.H > 0:
            parts.append(self.cmd_hist.reshape(1, -1))
        if self.use_phase:
            sphi, cphi = self.cpg.phase_features()
            parts += [sphi, cphi]
        return torch.cat(parts, dim=-1)

    # -------------------- main loop --------------------
    def _tick(self):
        # 1) command (remap if needed)
        cmd = self.cmd_now.clone()
        cmd_for_policy = self._remap_for_training(cmd) if self.apply_remap else cmd

        # 2) obs -> actor -> actions (12 amplitudes in [-1,1])
        obs = self._norm_obs(self._build_obs(cmd_for_policy))
        with torch.no_grad():
            actions = self.actor(obs)

        # 3) CPG -> joint targets (rad)
        q_targets = self.cpg.step(commands=cmd_for_policy, actions=actions, q0=self.q0)  # (1,12)
        positions = [float(x) for x in q_targets.view(-1).tolist()]

        # 4) publish JointState (for debug / serial bridge)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(DEFAULT_JOINT_ORDER)  # [FL, FR, RL, RR]
        msg.position = positions
        self.pub.publish(msg)

        # 4b) ALSO publish to ros2_control position controller so Gazebo moves
        ctrl_msg = Float64MultiArray()
        ctrl_msg.data = positions  # must match controller YAML 'joints:' order
        self.ctrl_pub.publish(ctrl_msg)

        # 5) roll history (front = most recent)
        if self.H > 0:
            self.cmd_hist = torch.roll(self.cmd_hist, shifts=1, dims=1)
            self.cmd_hist[:, 0, :] = cmd_for_policy


def main():
    rclpy.init()
    node = PolicyCPGNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
