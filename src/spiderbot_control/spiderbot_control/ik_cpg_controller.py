#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file IK+CPG controller for a 12-DOF spider:
- Subscribes: /cmd_vel (geometry_msgs/Twist)
- Generates trot gait with yaw shaping
- Computes per-leg analytic IK (coxa yaw, femur pitch, tibia pitch)
- Publishes 12 joint angles (Float64MultiArray) to /position_controller/commands
Order: [FL, FR, RL, RR] × (coxa, femur, tibia)
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


# ---------- Geometry (inline, from your URDF + measurements) ----------

# Hip (coxa joint) positions in BODY frame (meters) for [FL, FR, RL, RR]
HIPS = np.array([
    [ 0.0802, -0.1049, -0.0717],  # FL
    [-0.1202, -0.1037, -0.0723],  # FR
    [ 0.0802,  0.0649, -0.0717],  # RL
    [-0.1202,  0.0649, -0.0717],  # RR
], dtype=np.float32)

# Per-leg link lengths (meters)
LENGTHS = {
    "fl": (0.0600454, 0.0799992, 0.15),
    "fr": (0.0600454, 0.0800133, 0.15),
    "rl": (0.0600454, 0.0799992, 0.15),
    "rr": (0.0596356, 0.0799992, 0.15),
}

# Neutral feet (meters), safe stance (outboard >= Lcoxa+0.02, z≈-0.22)
NEUTRAL_FEET = np.array([
    [ 0.2002, -0.0249, -0.2200],  # FL
    [-0.0002, -0.1837, -0.2200],  # FR
    [-0.0398,  0.1449, -0.2200],  # RL
    [-0.2402, -0.0147, -0.2200],  # RR
], dtype=np.float32)

# Per-leg coxa zero offsets (radians) to align “forward” across all legs:
# Front legs: 0, 0  |  Rear legs: RL -pi/2, RR +pi/2
COXA_ZERO = np.array([0.0, 0.0, -math.pi/2, +math.pi/2], dtype=np.float32)

# Joint limits (radians): [coxa, femur, tibia]
QMIN = np.array([-math.pi/2, -math.radians(100), -math.radians(150)], dtype=np.float32)
QMAX = np.array([ math.pi/2,  math.radians(100),  math.radians(150)], dtype=np.float32)


# ---------- Math helpers ----------

def wrap_to_pi(a: float) -> float:
    return ((a + math.pi) % (2.0*math.pi)) - math.pi

def ik_leg_3dof(foot_body, hip_body, Lc, Lf, Lt):
    """Analytic IK for 3-DoF leg (coxa yaw, femur pitch, tibia pitch)."""
    px, py, pz = (np.asarray(foot_body) - np.asarray(hip_body)).astype(float)
    q_coxa = math.atan2(py, px)

    dxy = math.hypot(px, py)
    x1  = max(dxy - float(Lc), 1e-6)
    z1  = -pz

    D = (x1**2 + z1**2 - Lf**2 - Lt**2) / (2.0*Lf*Lt)
    D = max(-1.0, min(1.0, D))
    q_tibia = math.acos(D) - math.pi

    phi = math.atan2(z1, x1)
    psi = math.atan2(Lt*math.sin(q_tibia+math.pi), Lf + Lt*math.cos(q_tibia+math.pi))
    q_femur = phi - psi

    q = np.array([q_coxa, q_femur, q_tibia], dtype=np.float32)
    return np.minimum(np.maximum(q, QMIN), QMAX)  # clamp per-joint

def compute_q12(feet_body_4x3, hips_body_4x3, lengths_dict, coxa_zero_offsets=None):
    """Return 12 angles in controller order [FL,FR,RL,RR]×(coxa,femur,tibia)."""
    order = ["fl", "fr", "rl", "rr"]
    feet = np.asarray(feet_body_4x3, dtype=float)
    hips = np.asarray(hips_body_4x3, dtype=float)
    out = []
    for i, leg in enumerate(order):
        Lc, Lf, Lt = lengths_dict[leg]
        q = ik_leg_3dof(feet[i], hips[i], Lc, Lf, Lt)
        if coxa_zero_offsets is not None:
            q[0] = wrap_to_pi(q[0] + float(coxa_zero_offsets[i]))
            q = np.minimum(np.maximum(q, QMIN), QMAX)
        out.append(q)
    return np.concatenate(out, axis=0).astype(np.float32)


# ---------- CPG ----------

class CPG:
    def __init__(self, freq_hz=1.6):
        self.freq_hz = float(freq_hz)
        self.phase = 0.0

    def step(self, dt: float):
        self.phase = (self.phase + 2.0*math.pi*self.freq_hz*dt) % (2.0*math.pi)

    def leg_phases(self):
        # [FL, FR, RL, RR] with diagonal pairing (FL/RR in phase, FR/RL π-shifted)
        ph = np.array([self.phase, self.phase+math.pi, self.phase+math.pi, self.phase], dtype=np.float32)
        return np.mod(ph, 2.0*math.pi)


# ---------- ROS 2 node ----------

class IkCpgController(Node):
    def __init__(self):
        super().__init__('ik_cpg_controller')

        # Topics/loop
        self.declare_parameter('out_topic', '/position_controller/commands')
        self.declare_parameter('rate_hz', 100.0)
        self.out_topic = self.get_parameter('out_topic').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.dt = 1.0 / max(1.0, self.rate_hz)

        # Gait shaping
        self.declare_parameter('freq_hz', 1.6)
        self.declare_parameter('swing_h', 0.05)
        self.declare_parameter('stance_ratio', 0.6)
        self.declare_parameter('step_len_gain', 0.25)
        self.declare_parameter('lateral_gain', 0.20)
        self.declare_parameter('yaw_lat_gain', 0.06)
        self.declare_parameter('yaw_x_gain',   0.02)

        # Hold-still gating
        self.declare_parameter('hold_still_if_no_cmd', True)
        self.declare_parameter('cmd_deadband', 1e-3)
        self.declare_parameter('reset_phase_on_stop', True)

        # Optional mapping (defaults match controller order)
        self.declare_parameter('perm', list(range(12)))
        self.declare_parameter('signs', [1.0]*12)
        self.declare_parameter('default_q', [0.0]*12)
        self.declare_parameter('max_joint_rad', 1.2)

        self.cpg = CPG(float(self.get_parameter('freq_hz').value))
        self.swing_h   = float(self.get_parameter('swing_h').value)
        self.stance_r  = float(self.get_parameter('stance_ratio').value)
        self.step_len_gain = float(self.get_parameter('step_len_gain').value)
        self.lat_gain  = float(self.get_parameter('lateral_gain').value)
        self.yaw_lat_gain = float(self.get_parameter('yaw_lat_gain').value)
        self.yaw_x_gain   = float(self.get_parameter('yaw_x_gain').value)

        self.perm  = np.array(self.get_parameter('perm').value,  dtype=np.int64)
        self.signs = np.array(self.get_parameter('signs').value, dtype=np.float32)
        self.default_q = np.array(self.get_parameter('default_q').value, dtype=np.float32)
        self.max_joint = float(self.get_parameter('max_joint_rad').value)

        self.hold_still  = bool(self.get_parameter('hold_still_if_no_cmd').value)
        self.cmd_deadband= float(self.get_parameter('cmd_deadband').value)
        self.reset_phase = bool(self.get_parameter('reset_phase_on_stop').value)

        self.feet_neutral = NEUTRAL_FEET.copy()
        self.hips = HIPS.copy()
        self.coxa_zero = COXA_ZERO.copy()

        self.cmd = np.zeros(3, dtype=np.float32)  # [vx, vy, wz]

        self.pub = self.create_publisher(Float64MultiArray, self.out_topic, 10)
        self.sub = self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)
        self.timer = self.create_timer(self.dt, self._on_timer)

        self.get_logger().info(
            f'ik_cpg_controller @ {self.rate_hz:.1f} Hz -> {self.out_topic} | inline IK + hold-still'
        )

    def _on_cmd(self, msg: Twist):
        self.cmd[:] = (msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_timer(self):
        # live frequency tuning if param adjusted
        self.cpg.freq_hz = float(self.get_parameter('freq_hz').value)

        vx, vy, wz = self.cmd.tolist()
        cmd_mag = abs(vx) + abs(vy) + 0.5*abs(wz)
        if self.hold_still and cmd_mag < self.cmd_deadband:
            if self.reset_phase: self.cpg.phase = 0.0
            targets = np.clip(self.default_q.copy(), -self.max_joint, self.max_joint)
            self._publish(targets)
            return

        # advance gait, build feet targets
        self.cpg.step(self.dt)
        ph = self.cpg.leg_phases()
        feet = self._feet_traj(ph, vx, vy, wz)

        # IK -> 12 angles (controller order)
        q12 = compute_q12(feet, self.hips, LENGTHS, coxa_zero_offsets=self.coxa_zero)

        # map, add neutral offsets, clamp
        q12 = (q12[self.perm]) * self.signs
        q12 = np.clip(self.default_q + q12, -self.max_joint, self.max_joint)

        self._publish(q12)

    def _feet_traj(self, ph: np.ndarray, vx: float, vy: float, wz: float) -> np.ndarray:
        """
        Foot trajectory around NEUTRAL_FEET with diagonal trot.
        Adds per-leg yaw shaping so Q/E turns in place.
        """
        feet = self.feet_neutral.copy()
        stance_th = math.pi * self.stance_r
        step_len = self.step_len_gain * vx

        side  = np.array([+1, -1, +1, -1], dtype=np.float32)  # +y left legs
        front = np.array([+1, +1, -1, -1], dtype=np.float32)  # +x front legs

        for i in range(4):
            p = ph[i]
            in_stance = p > stance_th
            if in_stance:
                s = (p - stance_th) / (2.0*math.pi - stance_th)
                x_off = (1.0 - 2.0*s) * (0.5 * step_len)
                z_off = 0.0
            else:
                s = p / stance_th
                x_off = (-1.0 + 2.0*s) * (0.5 * step_len)
                z_off = self.swing_h * math.sin(math.pi * s)

            # lateral from vy + yaw shaping (leg-specific), small fore/aft yaw bias
            y_off = self.lat_gain * vy + self.yaw_lat_gain * wz * side[i]
            x_off += self.yaw_x_gain * wz * front[i]

            feet[i, 0] += x_off
            feet[i, 1] += y_off
            feet[i, 2] += z_off

        return feet

    def _publish(self, q12: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q12]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = IkCpgController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
