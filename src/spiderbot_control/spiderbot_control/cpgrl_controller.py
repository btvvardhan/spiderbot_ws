#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
from typing import List, Dict, Optional

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry

# CPG converter must expose: actions_to_angles(actions17, t, state) -> 12 angles
from . import cpg

try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

try:
    from ament_index_python.packages import get_package_share_directory
    _AMENT_OK = True
except Exception:
    _AMENT_OK = False


JOINT_ORDER: List[str] = [
    "Revolute 77","Revolute 78","Revolute 79","Revolute 80",
    "Revolute 90","Revolute 91","Revolute 92","Revolute 93",
    "Revolute 102","Revolute 103","Revolute 104","Revolute 105",
]
NUM_ACTIONS = 17
NUM_JOINTS  = 12
OBS_DIM     = 53


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    """Rotation matrix (world_from_body) from quaternion (x,y,z,w)."""
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx+zz),   2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),     1 - 2*(xx+yy)],
    ], dtype=np.float32)
    return R


class CpgrlController(Node):
    """
    ROS2 bridge that reproduces the exact obs your Isaac policy expects (53-D),
    runs the policy (TorchScript or torch.load), converts 17-D actions with your
    CPG to 12 joint angles, and publishes to the ForwardCommandController.
    """

    def __init__(self):
        super().__init__('cpgrl_controller')

        # ---------------- Parameters ----------------
        self.declare_parameter('use_model', False)
        self.declare_parameter('model_path', '')
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('action_scale', 1.0)
        self.declare_parameter('max_joint_rad', 1.2)

        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('out_topic', '/position_controller/commands')
        self.declare_parameter('joint_order', JOINT_ORDER)

        # Sensing topics & assumptions
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic',  '/imu/data')
        # If odom.twist is already in the robot (child) frame, keep True.
        # If odom.twist is in world frame, set False and we'll rotate into body.
        self.declare_parameter('odom_twist_is_body', True)

        # Default joint positions used during training (usually zeros).
        self.declare_parameter('default_joint_rad', [0.0]*NUM_JOINTS)

        # Safe default for sim clocks (but don't redeclare if launch passes it)
        try:
            self.declare_parameter('use_sim_time', True)
        except Exception:
            pass

        self.use_model      = bool(self.get_parameter('use_model').value)
        self.rate_hz        = float(self.get_parameter('rate_hz').value)
        self.action_scale   = float(self.get_parameter('action_scale').value)
        self.max_joint      = float(self.get_parameter('max_joint_rad').value)
        self.cmd_topic      = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.out_topic      = self.get_parameter('out_topic').get_parameter_value().string_value
        self.odom_topic     = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.imu_topic      = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.odom_is_body   = bool(self.get_parameter('odom_twist_is_body').value)

        joint_param = self.get_parameter('joint_order').get_parameter_value().string_array_value
        self.joint_order: List[str] = list(joint_param) if joint_param else list(JOINT_ORDER)
        dj = self.get_parameter('default_joint_rad').get_parameter_value().double_array_value
        self.default_joint = np.array(dj[:NUM_JOINTS] if dj else [0.0]*NUM_JOINTS, dtype=np.float32)

        # ---------------- State ----------------
        self._last_cmd = np.zeros(3, dtype=np.float32)  # [vx, vy, wz]
        self._joint_state: JointState = JointState()
        self._have_js = False

        # Body-frame signals for obs
        self._lin_vel_b = np.zeros(3, dtype=np.float32)
        self._ang_vel_b = np.zeros(3, dtype=np.float32)
        self._gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._have_odom = False
        self._have_imu  = False

        # For deriving joint velocity if not provided
        self._prev_q = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._prev_q_t = None

        # Previous 17 actions (for the last block in the obs)
        self._prev_actions = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # CPG internal state container
        self._state: Dict = {}

        # ---------------- Pub/Sub ----------------
        self.pub_cmd = self.create_publisher(Float64MultiArray, self.out_topic, 10)
        self.sub_cmd = self.create_subscription(Twist, self.cmd_topic, self._on_cmd, 10)
        self.sub_js  = self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.sub_odom= self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self._on_imu, 10)

        # ---------------- Model loading ----------------
        self.model = None
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if self.use_model and not model_path and _AMENT_OK:
            try:
                share = get_package_share_directory('spiderbot_control')
                default_path = os.path.join(share, 'models', 'policy.pt')
                if os.path.exists(default_path):
                    model_path = default_path
                    self.get_logger().info(f'No model_path param set; using {default_path}')
            except Exception:
                pass

        if self.use_model:
            self.model = self._load_model(model_path)

        # ---------------- Timer ----------------
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._step)
        self.get_logger().info(f'CPG-RL controller @ {self.rate_hz:.1f} Hz -> {self.out_topic} | use_model={self.use_model}')

    # ---------------- Model I/O ----------------

    def _load_model(self, path: str):
        if not _TORCH_OK:
            self.get_logger().warn('Torch not available; running without model.')
            return None
        if not path or not os.path.exists(path):
            self.get_logger().error(f'Model path not found: "{path}". Running without model.')
            return None
        try:
            mdl = torch.jit.load(path, map_location='cpu')
            mdl.eval()
            self.get_logger().info(f'Loaded TorchScript policy: {path}')
            return mdl
        except Exception as e:
            self.get_logger().warn(f'jit.load failed ({e}); trying torch.load.')
            try:
                mdl = torch.load(path, map_location='cpu')
                if hasattr(mdl, 'eval'):
                    mdl.eval()
                self.get_logger().info(f'Loaded policy via torch.load: {path}')
                return mdl
            except Exception as e2:
                self.get_logger().error(f'Could not load model: {e2}. Running without model.')
                return None

    # ---------------- Callbacks ----------------

    def _on_cmd(self, msg: Twist):
        self._last_cmd[:] = (msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_js(self, msg: JointState):
        self._joint_state = msg
        self._have_js = True

    def _on_odom(self, msg: Odometry):
        # Get linear & angular velocity; rotate into body if needed
        v = np.array([msg.twist.twist.linear.x,
                      msg.twist.twist.linear.y,
                      msg.twist.twist.linear.z], dtype=np.float32)
        w = np.array([msg.twist.twist.angular.x,
                      msg.twist.twist.angular.y,
                      msg.twist.twist.angular.z], dtype=np.float32)

        if self.odom_is_body:
            self._lin_vel_b[:] = v
            self._ang_vel_b[:] = w
        else:
            # Rotate world->body using pose orientation
            q = msg.pose.pose.orientation
            R_wb = quat_to_rot(q.x, q.y, q.z, q.w)  # world_from_body
            self._lin_vel_b[:] = R_wb.T @ v
            self._ang_vel_b[:] = R_wb.T @ w

        self._have_odom = True

    def _on_imu(self, msg: Imu):
        # Body-from-world gravity projection: g_b = R^T * [0,0,-1]
        q = msg.orientation
        R_wb = quat_to_rot(q.x, q.y, q.z, q.w)  # world_from_body
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._gravity_b[:] = R_wb.T @ g_world

        # Prefer gyro from IMU for angular vel if available
        self._ang_vel_b[:] = np.array([msg.angular_velocity.x,
                                       msg.angular_velocity.y,
                                       msg.angular_velocity.z], dtype=np.float32)
        self._have_imu = True

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds * 1e-9)

    # ---------------- Observation & Policy ----------------

    def _gather_joint_positions(self) -> np.ndarray:
        q = np.zeros(NUM_JOINTS, dtype=np.float32)
        if self._joint_state.name and self._joint_state.position:
            name_to_pos = dict(zip(self._joint_state.name, self._joint_state.position))
            for i, jn in enumerate(self.joint_order[:NUM_JOINTS]):
                if jn in name_to_pos:
                    q[i] = float(name_to_pos[jn])
        return q

    def _gather_joint_velocities(self) -> np.ndarray:
        # Prefer provided velocities; otherwise finite diff
        v = np.zeros(NUM_JOINTS, dtype=np.float32)
        if self._joint_state.name:
            if self._joint_state.velocity and len(self._joint_state.velocity) >= len(self._joint_state.name):
                name_to_vel = dict(zip(self._joint_state.name, self._joint_state.velocity))
                for i, jn in enumerate(self.joint_order[:NUM_JOINTS]):
                    if jn in name_to_vel:
                        v[i] = float(name_to_vel[jn])
                return v

            # FD fallback
            q = self._gather_joint_positions()
            t = self._now_sec()
            if self._prev_q_t is not None:
                dt = max(1e-3, t - self._prev_q_t)
                v = (q - self._prev_q) / dt
            self._prev_q[:] = q
            self._prev_q_t = t
        return v

    def _build_observation(self) -> np.ndarray:
        """
        Reproduce Isaac training obs order & size (53):
          [ root_lin_vel_b(3), root_ang_vel_b(3), projected_gravity_b(3),
            commands(3), joint_pos - default(12), joint_vel(12), previous_actions(17) ]
        """
        vlin_b = self._lin_vel_b if (self._have_odom or self._have_imu) else np.zeros(3, dtype=np.float32)
        vang_b = self._ang_vel_b if (self._have_odom or self._have_imu) else np.zeros(3, dtype=np.float32)
        gproj  = self._gravity_b

        qpos = self._gather_joint_positions()
        qvel = self._gather_joint_velocities()
        dpos = qpos - self.default_joint

        obs = np.concatenate([
            vlin_b.astype(np.float32),         # 3
            vang_b.astype(np.float32),         # 3
            gproj.astype(np.float32),          # 3
            self._last_cmd.astype(np.float32), # 3
            dpos.astype(np.float32),           # 12
            qvel.astype(np.float32),           # 12
            self._prev_actions.astype(np.float32),  # 17
        ], axis=0)

        if obs.shape[0] != OBS_DIM:
            self.get_logger().warn(f'Obs size {obs.shape[0]} != {OBS_DIM}; padding/clipping.')
            if obs.shape[0] < OBS_DIM:
                obs = np.pad(obs, (0, OBS_DIM - obs.shape[0]))
            else:
                obs = obs[:OBS_DIM]
        return obs

    def _heuristic_actions_from_cmd(self) -> np.ndarray:
        vx, vy, wz = self._last_cmd
        a = np.zeros(NUM_ACTIONS, dtype=np.float32)
        base_freq = 1.5 + 2.0 * np.clip(abs(vx) + 0.3 * abs(vy) + 0.2 * abs(wz), 0.0, 1.0)
        a[0] = float(np.clip(base_freq, 0.5, 3.0))                 # frequency
        a[1:1+NUM_JOINTS] = float(np.clip(0.6 * self.action_scale, 0.0, 1.2))  # amplitudes
        # a[13:17] phases/extras left 0.0 unless your CPG uses them
        return a

    def _policy_actions(self, obs: np.ndarray) -> np.ndarray:
        if self.model is None:
            return self._heuristic_actions_from_cmd()
        try:
            with torch.no_grad():
                x = torch.from_numpy(obs).float().unsqueeze(0)
                y = self.model(x)
                if isinstance(y, (list, tuple)):
                    y = y[0]
                y = y.squeeze(0).cpu().numpy().astype(np.float32).reshape(-1)
            if y.shape[0] < NUM_ACTIONS:
                self.get_logger().warn(f'Policy produced {y.shape[0]} dims; expected {NUM_ACTIONS}. Padding zeros.')
                y = np.pad(y, (0, NUM_ACTIONS - y.shape[0]))
            return y[:NUM_ACTIONS]
        except Exception as e:
            self.get_logger().error(f'Policy inference error: {e}. Using heuristic actions.')
            return self._heuristic_actions_from_cmd()

    # ---------------- Main loop ----------------

    def _step(self):
        if not self._have_js:
            # Hold controller with neutral pose until joint_states arrive
            msg = Float64MultiArray()
            msg.data = [0.0]*NUM_JOINTS
            self.pub_cmd.publish(msg)
            return

        t = self._now_sec()
        obs = self._build_observation()
        actions17 = self._policy_actions(obs)

###########





        # Convert 17-D actions -> 12 joint angles with your CPG
        try:
            angles12 = cpg.actions_to_angles(actions17, t, self._state)
        except Exception as e:
            self.get_logger().error(f'CPG conversion failed: {e}. Sending zeros.')
            angles12 = np.zeros(NUM_JOINTS, dtype=np.float32)


        # Publish ABSOLUTE targets: default pose + deltas (exactly like training)
        targets12 = DEFAULT_Q + np.asarray(angles12, dtype=np.float32)

        # Safety clamp
        targets12 = np.clip(targets12, -self.max_joint, self.max_joint)

        msg = Float64MultiArray()
        msg.data = [float(v) for v in targets12]
        self.pub_cmd.publish(msg)




        # Save actions for next observation
        self._prev_actions[:] = actions17.astype(np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = CpgrlController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
