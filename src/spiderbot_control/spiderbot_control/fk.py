#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardcoded Gait Node - Drop-in replacement for policy_omni_node.py

Replaces RL policy with hardcoded gait tables.
Architecture remains identical - publishes to /joint_states.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


DEFAULT_JOINT_ORDER = [
    "fl_coxa_joint", "fl_femur_joint", "fl_tibia_joint",
    "fr_coxa_joint", "fr_femur_joint", "fr_tibia_joint",
    "rl_coxa_joint", "rl_femur_joint", "rl_tibia_joint",
    "rr_coxa_joint", "rr_femur_joint", "rr_tibia_joint",
]

# ===================== GAIT TABLES (DEGREES) =====================
# Format: [FL, FR, RL, RR] x [coxa/hip, femur/knee, tibia/ankle]

GAIT_FORWARD = [
    [[145, 20, 35], [90, 55, 45], [35, 55, 45], [90, 55, 45]],
    [[90, 20, 35], [90, 55, 45], [35, 55, 45], [90, 55, 45]],
    [[90, 45, 35], [90, 45, 35], [35, 45, 35], [90, 45, 35]],
    [[90, 45, 35], [35, 45, 35], [90, 45, 35], [145, 20, 35]],
    [[90, 55, 45], [35, 20, 35], [90, 55, 45], [145, 55, 45]],
    [[90, 55, 45], [90, 20, 35], [90, 55, 45], [145, 55, 45]],
    [[90, 45, 35], [90, 45, 35], [90, 45, 35], [145, 45, 35]],
    [[145, 45, 35], [90, 45, 35], [35, 20, 35], [90, 45, 35]],
]

GAIT_BACKWARD = [
    [[145, 45, 35], [90, 45, 35], [35, 20, 35], [90, 45, 35]],
    [[90, 45, 35], [90, 45, 35], [90, 45, 35], [145, 45, 35]],
    [[90, 55, 45], [90, 20, 35], [90, 55, 45], [145, 55, 45]],
    [[90, 55, 45], [35, 20, 35], [90, 55, 45], [145, 55, 45]],
    [[90, 45, 35], [35, 45, 35], [90, 45, 35], [145, 20, 35]],
    [[90, 45, 35], [90, 45, 35], [35, 45, 35], [90, 45, 35]],
    [[90, 20, 35], [90, 55, 45], [35, 55, 45], [90, 55, 45]],
    [[145, 20, 35], [90, 55, 45], [35, 55, 45], [90, 55, 45]],
]

GAIT_RIGHT = [
    [[35, 55, 45], [145, 20, 35], [90, 55, 45], [90, 55, 45]],
    [[35, 55, 45], [90, 20, 35], [90, 55, 45], [90, 55, 45]],
    [[35, 45, 35], [90, 45, 35], [90, 45, 35], [90, 45, 35]],
    [[90, 45, 35], [90, 45, 35], [145, 20, 35], [35, 45, 35]],
    [[90, 55, 45], [90, 55, 45], [145, 55, 45], [35, 20, 35]],
    [[90, 55, 45], [90, 55, 45], [145, 55, 45], [90, 20, 35]],
    [[90, 45, 35], [90, 45, 35], [145, 45, 35], [90, 45, 35]],
    [[35, 20, 35], [145, 45, 35], [90, 45, 35], [90, 45, 35]],
]

GAIT_LEFT = [
    [[35, 20, 35], [145, 45, 35], [90, 45, 35], [90, 45, 35]],
    [[90, 45, 35], [90, 45, 35], [145, 45, 35], [90, 45, 35]],
    [[90, 55, 45], [90, 55, 45], [145, 55, 45], [90, 20, 35]],
    [[90, 55, 45], [90, 55, 45], [145, 55, 45], [35, 20, 35]],
    [[90, 45, 35], [90, 45, 35], [145, 20, 35], [35, 45, 35]],
    [[35, 45, 35], [90, 45, 35], [90, 45, 35], [90, 45, 35]],
    [[35, 55, 45], [90, 20, 35], [90, 55, 45], [90, 55, 45]],
    [[35, 55, 45], [145, 20, 35], [90, 55, 45], [90, 55, 45]],
]

GAIT_YAW_RIGHT = [
    [[145, 20, 35], [90, 45, 35], [35, 55, 45], [90, 45, 35]],
    [[90, 20, 35], [90, 45, 35], [35, 55, 45], [145, 45, 35]],
    [[90, 45, 35], [90, 20, 35], [35, 45, 35], [145, 55, 45]],
    [[90, 45, 35], [35, 20, 35], [90, 45, 35], [145, 55, 45]],
    [[90, 55, 45], [35, 45, 35], [90, 55, 45], [145, 20, 35]],
    [[90, 55, 45], [90, 45, 35], [90, 55, 45], [90, 45, 35]],
    [[90, 45, 35], [90, 55, 45], [90, 45, 35], [90, 55, 45]],
    [[145, 45, 35], [90, 55, 45], [35, 20, 35], [90, 55, 45]],
]

GAIT_YAW_LEFT = [
    [[145, 45, 35], [90, 55, 45], [35, 20, 35], [90, 55, 45]],
    [[90, 45, 35], [90, 55, 45], [90, 45, 35], [90, 55, 45]],
    [[90, 55, 45], [90, 45, 35], [90, 55, 45], [90, 45, 35]],
    [[90, 55, 45], [35, 45, 35], [90, 55, 45], [145, 20, 35]],
    [[90, 45, 35], [35, 20, 35], [90, 45, 35], [145, 55, 45]],
    [[90, 45, 35], [90, 20, 35], [35, 45, 35], [145, 55, 45]],
    [[90, 20, 35], [90, 45, 35], [35, 55, 45], [145, 45, 35]],
    [[145, 20, 35], [90, 45, 35], [35, 55, 45], [90, 45, 35]],
]

# Home pose (neutral stance)
HOME_POSE = [[90, 35, 35], [90, 35, 35], [90, 35, 35], [90, 35, 35]]


class HardcodedGaitNode(Node):
    def __init__(self):
        super().__init__("fk")  # Keep same node name for drop-in replacement

        # Parameters (match policy_omni_node interface)
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("controller_command_topic", "/position_controller/commands")
        self.declare_parameter("rate_hz", 50.0)
        
        # Gait parameters
        self.declare_parameter("step_delay", 0.15)  # Time per gait step (seconds)
        self.declare_parameter("smooth_steps", 10)  # Interpolation frames
        self.declare_parameter("cmd_zero_threshold", 0.01)
        self.declare_parameter("standstill_on_zero_cmd", True)
        
        # Default joint positions (radians) - home pose
        self.declare_parameter("q0", [0.0] * 12)
        
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / self.rate_hz
        self.step_delay = float(self.get_parameter("step_delay").value)
        self.smooth_steps = int(self.get_parameter("smooth_steps").value)
        self.cmd_zero_threshold = float(self.get_parameter("cmd_zero_threshold").value)
        self.standstill_enabled = bool(self.get_parameter("standstill_on_zero_cmd").value)
        
        # Gait state
        self.gaits = {
            'forward': GAIT_FORWARD,
            'backward': GAIT_BACKWARD,
            'right': GAIT_RIGHT,
            'left': GAIT_LEFT,
            'yaw_right': GAIT_YAW_RIGHT,
            'yaw_left': GAIT_YAW_LEFT,
        }
        
        self.current_gait = None
        self.current_step = 0
        self.current_frame = 0
        self.is_walking = False
        self.is_standstill = False
        
        # Command state
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        
        # Home pose in radians
        self.q0 = self.get_parameter("q0").value
        
        # ROS I/O (match policy_omni_node interface)
        self.cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        self.joint_topic = self.get_parameter("joint_state_topic").get_parameter_value().string_value
        self.ctrl_topic = self.get_parameter("controller_command_topic").get_parameter_value().string_value
        
        self.sub = self.create_subscription(Twist, self.cmd_topic, self._cmd_cb, 10)
        self.pub = self.create_publisher(JointState, self.joint_topic, 10)
        self.ctrl_pub = self.create_publisher(Float64MultiArray, self.ctrl_topic, 10)
        self.timer = self.create_timer(self.dt, self._tick)
        
        self.get_logger().info(
            f"✅ Hardcoded Gait node ready!\n"
            f"  🎮 Gaits: 6 hardcoded patterns (forward, back, left, right, yaw L/R)\n"
            f"  🔄 Smooth interpolation: {self.smooth_steps} frames\n"
            f"  ⏸️  Standstill on zero cmd: {self.standstill_enabled}\n"
            f"  ⚡ Rate: {self.rate_hz}Hz\n"
            f"  📊 Step delay: {self.step_delay}s per step"
        )
    
    def _cmd_cb(self, msg: Twist):
        """Velocity command callback."""
        self.cmd_vx = float(msg.linear.x)
        self.cmd_vy = float(msg.linear.y)
        self.cmd_wz = float(msg.angular.z)
        
        # Select gait based on command
        self._select_gait()
    
    def _is_zero_command(self) -> bool:
        """Check if current command is effectively zero."""
        cmd_magnitude = math.sqrt(self.cmd_vx**2 + self.cmd_vy**2 + self.cmd_wz**2)
        return cmd_magnitude < self.cmd_zero_threshold
    
    def _select_gait(self):
        """Select gait based on cmd_vel (priority: rotation > forward/back > strafe)."""
        if abs(self.cmd_wz) > 0.1:
            gait = 'yaw_left' if self.cmd_wz > 0 else 'yaw_right'
        elif abs(self.cmd_vx) > 0.1:
            gait = 'forward' if self.cmd_vx > 0 else 'backward'
        elif abs(self.cmd_vy) > 0.1:
            gait = 'left' if self.cmd_vy > 0 else 'right'
        else:
            gait = None
        
        if gait != self.current_gait:
            self._set_gait(gait)
    
    def _set_gait(self, gait_name):
        """Change active gait."""
        if gait_name is None:
            if self.is_walking:
                self.get_logger().info('⏸️  Stopping')
                self.is_walking = False
                self.current_gait = None
        elif gait_name in self.gaits:
            if gait_name != self.current_gait:
                self.get_logger().info(f'🚶 Starting: {gait_name}')
                self.current_gait = gait_name
                self.current_step = 0
                self.current_frame = 0
                self.is_walking = True
    
    def _deg_to_rad(self, degrees):
        """Convert degrees to radians (centered at 90°)."""
        # Arduino: 90° = neutral
        # ROS: 0 rad = neutral
        # So: angle_rad = (angle_deg - 90) * pi/180
        return (degrees - 90.0) * math.pi / 180.0
    
    def _pose_to_radians(self, pose_deg):
        """
        Convert pose from degrees to radians.
        pose_deg: [FL, FR, RL, RR] x [coxa, femur, tibia] in degrees
        Returns: list of 12 joint positions in radians (FL, FR, RL, RR order)
        """
        positions = []
        for leg_idx, leg in enumerate(pose_deg):
            for joint_idx, joint_deg in enumerate(leg):
                rad = self._deg_to_rad(joint_deg)
                # Invert tibia joints (index 2 in each leg)
                if joint_idx == 2:
                    rad = -rad
                positions.append(rad)
        return positions
    
    def _interpolate_poses(self, from_pose, to_pose, alpha):
        """Linear interpolation between two poses (in degrees)."""
        result = []
        for leg_idx in range(4):
            leg_angles = []
            for joint_idx in range(3):
                from_angle = from_pose[leg_idx][joint_idx]
                to_angle = to_pose[leg_idx][joint_idx]
                interp_angle = from_angle + alpha * (to_angle - from_angle)
                leg_angles.append(interp_angle)
            result.append(leg_angles)
        return result
    
    def _publish_pose(self, pose_deg):
        """Publish pose to /joint_states and /position_controller/commands."""
        # Convert to radians
        positions = self._pose_to_radians(pose_deg)
        
        # Publish JointState
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(DEFAULT_JOINT_ORDER)
        msg.position = positions
        self.pub.publish(msg)
        
        # Publish to controller
        ctrl_msg = Float64MultiArray()
        ctrl_msg.data = positions
        self.ctrl_pub.publish(ctrl_msg)
    
    def _tick(self):
        """Main control loop - runs at rate_hz."""
        
        # Check if we should be in standstill mode
        if self.standstill_enabled and self._is_zero_command():
            if not self.is_standstill:
                self.get_logger().info("⏸️  Zero command detected - entering standstill mode")
                self.is_standstill = True
            
            # Use q0 if set, otherwise HOME_POSE
            if any(q != 0.0 for q in self.q0):
                # q0 is already in radians
                positions = list(self.q0)
            else:
                # Convert HOME_POSE to radians
                positions = self._pose_to_radians(HOME_POSE)
            
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
        
        # If not walking, hold home pose
        if not self.is_walking or self.current_gait is None:
            self._publish_pose(HOME_POSE)
            return
        
        # Execute gait with smooth interpolation
        gait_sequence = self.gaits[self.current_gait]
        
        # Calculate interpolation
        current_step_pose = gait_sequence[self.current_step]
        next_step = (self.current_step + 1) % len(gait_sequence)
        next_step_pose = gait_sequence[next_step]
        
        if self.smooth_steps == 0:
            # No smoothing
            self._publish_pose(current_step_pose)
            self.current_step = next_step
        else:
            # Smooth interpolation
            alpha = self.current_frame / (self.smooth_steps + 1)
            interpolated_pose = self._interpolate_poses(
                current_step_pose, next_step_pose, alpha
            )
            self._publish_pose(interpolated_pose)
            
            # Advance frame
            self.current_frame += 1
            if self.current_frame > self.smooth_steps:
                self.current_step = next_step
                self.current_frame = 0


def main():
    rclpy.init()
    node = HardcodedGaitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Shutting down gait node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()