#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IK+CPG teleop controller (no RL) for a 12-DOF spider:
- Subscribes: /cmd_vel (from teleop keyboard)
- Generates diagonal-trot foot trajectories with a CPG
- Runs analytic 3-DoF leg IK (coxa/femur/tibia)
- Publishes 12 absolute joint angles to a JointGroupPositionController topic

Now auto-loads link lengths from /mnt/data/ik_dims.json by default,
so you don't have to pass a path each run. You can still override with
--ros-args -p ik_dims_path:=<path> if needed.
"""
import json, math
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

NUM_JOINTS = 12

# ---- Simple CPG (shared phase, diagonal trot) ----
class CPG:
    def __init__(self, freq_hz=1.5):
        self.freq_hz = float(freq_hz)
        self.phase = 0.0
    def step(self, dt: float):
        self.phase = (self.phase + 2.0*math.pi*self.freq_hz*dt) % (2.0*math.pi)
    def leg_phases(self) -> np.ndarray:
        ph = np.array([self.phase, self.phase+math.pi, self.phase+math.pi, self.phase])
        return np.mod(ph, 2.0*math.pi)

# ---- Analytic 3-DoF leg IK ----
class LegIK:
    def __init__(self, Lcoxa: float, Lfemur: float, Ltibia: float):
        self.Lc = float(Lcoxa); self.Lf = float(Lfemur); self.Lt = float(Ltibia)
        self.qmin = np.array([-math.pi/2, -math.radians(100), -math.radians(150)])
        self.qmax = np.array([ math.pi/2,  math.radians(100),  math.radians(150)])
    def solve(self, foot_xyz_body: np.ndarray, hip_offset: Tuple[float,float,float]) -> np.ndarray:
        hx, hy, hz = hip_offset
        px, py, pz = foot_xyz_body[0]-hx, foot_xyz_body[1]-hy, foot_xyz_body[2]-hz
        q_coxa = math.atan2(py, px)
        dxy = max(1e-6, math.hypot(px, py) - self.Lc)
        z1 = -pz
        D = (dxy**2 + z1**2 - self.Lf**2 - self.Lt**2) / (2*self.Lf*self.Lt)
        D = max(-1.0, min(1.0, D))
        q_tibia = math.acos(D) - math.pi
        phi = math.atan2(z1, dxy)
        psi = math.atan2(self.Lt*math.sin(q_tibia+math.pi), self.Lf + self.Lt*math.cos(q_tibia+math.pi))
        q_femur = phi - psi
        q = np.array([q_coxa, q_femur, q_tibia], dtype=np.float32)
        return np.clip(q, self.qmin, self.qmax)

# ---- Inline IK dimensions (baked from ik_dims.json) ----
IK_DIMS_INLINE = {
    "fl": {"Lcoxa": 0.06004539929919694, "Lfemur": 0.07999923279632123, "Ltibia": 0.15},
    "fr": {"Lcoxa": 0.06004539929919694, "Lfemur": 0.08001330320640437, "Ltibia": 0.15},
    "rl": {"Lcoxa": 0.06004539929919694, "Lfemur": 0.07999923279632123, "Ltibia": 0.15},
    "rr": {"Lcoxa": 0.05963555966870773, "Lfemur": 0.07999923279632123, "Ltibia": 0.15},
}

class IkCpgController(Node):
    def __init__(self):
        super().__init__('ik_cpg_controller')
        # Topics & timings
        self.declare_parameter('out_topic', '/position_controller/commands')
        self.declare_parameter('rate_hz', 100.0)
        # Gait parameters
        self.declare_parameter('freq_hz', 1.5)
        self.declare_parameter('swing_h', 0.05)
        self.declare_parameter('stance_ratio', 0.6)
        self.declare_parameter('step_len_gain', 0.25)
        self.declare_parameter('lateral_gain', 0.20)
        self.declare_parameter('yaw_gain', 0.03)
        # Hold-still + gating
        self.declare_parameter('hold_still_if_no_cmd', True)
        self.declare_parameter('cmd_deadband', 1e-3)
        self.declare_parameter('reset_phase_on_stop', True)
        # Mapping & pose
        self.declare_parameter('perm', list(range(NUM_JOINTS)))
        self.declare_parameter('signs', [1.0]*NUM_JOINTS)
        self.declare_parameter('default_q', [0.0]*NUM_JOINTS)
        self.declare_parameter('max_joint_rad', 1.2)
        # IK dims and geometry (auto-default to your uploaded file)
        # (ik_dims_path removed — using IK_DIMS_INLINE)
        # Hip locations in body frame (meters); tune to your URDF base frame
        self.declare_parameter('hip_offsets', [  # FL, FR, RL, RR
            0.20,  0.12, 0.0,
            0.20, -0.12, 0.0,
           -0.20,  0.12, 0.0,
           -0.20, -0.12, 0.0,
        ])
        # Neutral feet (body frame)
        self.declare_parameter('neutral_feet', [
            0.20,  0.12, -0.22,   # FL
            0.20, -0.12, -0.22,   # FR
           -0.20,  0.12, -0.22,   # RL
           -0.20, -0.12, -0.22,   # RR
        ])
        # Read params
        self.out_topic = self.get_parameter('out_topic').value
        self.rate_hz   = float(self.get_parameter('rate_hz').value)
        self.freq_hz   = float(self.get_parameter('freq_hz').value)
        self.swing_h   = float(self.get_parameter('swing_h').value)
        self.stance_r  = float(self.get_parameter('stance_ratio').value)
        self.step_len_gain = float(self.get_parameter('step_len_gain').value)
        self.lat_gain  = float(self.get_parameter('lateral_gain').value)
        self.yaw_gain  = float(self.get_parameter('yaw_gain').value)
        self.perm  = np.array(self.get_parameter('perm').value,  dtype=np.int64)
        self.signs = np.array(self.get_parameter('signs').value, dtype=np.float32)
        self.default_q = np.array(self.get_parameter('default_q').value, dtype=np.float32)
        self.max_joint = float(self.get_parameter('max_joint_rad').value)
        # Load IK dimensions once from the default path
        # Build IK solvers from inline dims (no file I/O)
        self.leg_ik = {
            "fl": LegIK(IK_DIMS_INLINE["fl"]["Lcoxa"], IK_DIMS_INLINE["fl"]["Lfemur"], IK_DIMS_INLINE["fl"]["Ltibia"]),
            "fr": LegIK(IK_DIMS_INLINE["fr"]["Lcoxa"], IK_DIMS_INLINE["fr"]["Lfemur"], IK_DIMS_INLINE["fr"]["Ltibia"]),
            "rl": LegIK(IK_DIMS_INLINE["rl"]["Lcoxa"], IK_DIMS_INLINE["rl"]["Lfemur"], IK_DIMS_INLINE["rl"]["Ltibia"]),
            "rr": LegIK(IK_DIMS_INLINE["rr"]["Lcoxa"], IK_DIMS_INLINE["rr"]["Lfemur"], IK_DIMS_INLINE["rr"]["Ltibia"]),
        }
        # Hip offsets and neutral feet
        hip = np.array(self.get_parameter('hip_offsets').value, dtype=np.float32).reshape(4,3)
        self.hip_offsets = {
            "fl": tuple(hip[0]), "fr": tuple(hip[1]), "rl": tuple(hip[2]), "rr": tuple(hip[3])
        }
        # Build neutral feet from hip offsets so feet are outboard of Lcoxa (avoid IK degeneracy)
        Lcoxa_by_leg = {"fl": self.leg_ik["fl"].Lc, "fr": self.leg_ik["fr"].Lc, "rl": self.leg_ik["rl"].Lc, "rr": self.leg_ik["rr"].Lc}
        side  = np.array([+1, -1, +1, -1], dtype=np.float32)  # +y for left legs
        front = np.array([+1, +1, -1, -1], dtype=np.float32)  # +x front legs
        feet = np.zeros((4,3), dtype=np.float32)
        for i, leg in enumerate(["fl","fr","rl","rr"]):
            hx, hy, hz = self.hip_offsets[leg]
            y = hy + side[i] * (Lcoxa_by_leg[leg] + 0.02)
            x = hx + front[i] * 0.12
            feet[i] = [x, y, -0.22]
        self.neutral_feet = feet
        # CPG & I/O
        self.cpg = CPG(self.freq_hz)
        self.pub = self.create_publisher(Float64MultiArray, self.out_topic, 10)
        self.sub = self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.dt = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(self.dt, self._on_timer)
        self.get_logger().info(f'ik_cpg_controller @ {self.rate_hz:.1f} Hz -> {self.out_topic} (inline IK dims + hold-still)')
    # ---- Utilities ----
    def _load_ik_dims(self, path: str):
        try:
            with open(path, 'r') as f:
                d = json.load(f)
            return {
                "fl": LegIK(d["fl"]["Lcoxa"], d["fl"]["Lfemur"], d["fl"]["Ltibia"]),
                "fr": LegIK(d["fr"]["Lcoxa"], d["fr"]["Lfemur"], d["fr"]["Ltibia"]),
                "rl": LegIK(d["rl"]["Lcoxa"], d["rl"]["Lfemur"], d["rl"]["Ltibia"]),
                "rr": LegIK(d["rr"]["Lcoxa"], d["rr"]["Lfemur"], d["rr"]["Ltibia"]),
            }
        except Exception as e:
            self.get_logger().warn(f'Could not read IK dims from {path}: {e}; using fallback lengths.')
            return {
                "fl": LegIK(0.06, 0.08, 0.15),
                "fr": LegIK(0.06, 0.08, 0.15),
                "rl": LegIK(0.06, 0.08, 0.15),
                "rr": LegIK(0.06, 0.08, 0.15),
            }
    # ---- ROS callbacks ----
    def _on_cmd(self, msg: Twist):
        self.cmd[:] = (msg.linear.x, msg.linear.y, msg.angular.z)
    # ---- Core loop ----
    def _on_timer(self):
        # 1) phase
        self.cpg.freq_hz = float(self.get_parameter('freq_hz').value)
        vx, vy, wz = self.cmd.tolist()
        # Hold-still gate
        cmd_mag = abs(vx) + abs(vy) + 0.5*abs(wz)
        if bool(self.get_parameter('hold_still_if_no_cmd').value) and cmd_mag < float(self.get_parameter('cmd_deadband').value):
            if bool(self.get_parameter('reset_phase_on_stop').value):
                self.cpg.phase = 0.0
            targets = np.clip(self.default_q.copy(), -self.max_joint, self.max_joint)
            self.pub.publish(Float64MultiArray(data=[float(v) for v in targets]))
            return
        # Advance CPG and compute feet
        self.cpg.step(self.dt)
        ph = self.cpg.leg_phases()
        # 2) foot targets
        ph = self.cpg.leg_phases()
        feet = self._foot_traj(ph, vx, vy, wz)
        # 3) IK per leg (FL, FR, RL, RR)
        q = []
        q.extend(self.leg_ik["fl"].solve(feet[0], self.hip_offsets["fl"]))
        q.extend(self.leg_ik["fr"].solve(feet[1], self.hip_offsets["fr"]))
        q.extend(self.leg_ik["rl"].solve(feet[2], self.hip_offsets["rl"]))
        q.extend(self.leg_ik["rr"].solve(feet[3], self.hip_offsets["rr"]))
        q = np.array(q, dtype=np.float32)
        # 4) map -> controller order, add neutral joint pose, clamp
        q = (q[self.get_parameter('perm').value]) * np.array(self.get_parameter('signs').value, dtype=np.float32)
        targets = self.default_q + q
        targets = np.clip(targets, -self.max_joint, self.max_joint)
        # 5) publish
        self.pub.publish(Float64MultiArray(data=[float(v) for v in targets]))
    # ---- Helpers ----
    def _foot_traj(self, ph: np.ndarray, vx: float, vy: float, wz: float) -> np.ndarray:
        """Return (4,3) feet targets in body frame given phases and command."""
        step_len = self.step_len_gain * vx
        feet = self.neutral_feet.copy()
        stance_th = math.pi * self.stance_r
        side  = np.array([+1, -1, +1, -1], dtype=np.float32)  # left +, right -
        front = np.array([+1, +1, -1, -1], dtype=np.float32)  # front +, rear -
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
            # lateral from vy plus yaw shaping; tiny fore/aft yaw bias
            y_off = self.lat_gain * vy + 0.06 * wz * side[i]
            x_off += 0.02 * wz * front[i]
            feet[i,0] += x_off
            feet[i,1] += y_off
            feet[i,2] += z_off
        return feet

# ---- main ----
def main():
    rclpy.init()
    node = IkCpgController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
