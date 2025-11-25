#!/usr/bin/env python3
"""
Joint Mapping Diagnostic Tool

Tests if joint commands from policy match expected robot motion.
Run this in Gazebo to verify joint order and sign correctness.

Usage:
    ros2 run spiderbot_control diagnose_joint_mapping
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import time
import math


class JointDiagnostic(Node):
    def __init__(self):
        super().__init__("joint_diagnostic")
        
        self.joint_order = [
            "fl_coxa_joint", "fl_femur_joint", "fl_tibia_joint",
            "fr_coxa_joint", "fr_femur_joint", "fr_tibia_joint",
            "rl_coxa_joint", "rl_femur_joint", "rl_tibia_joint",
            "rr_coxa_joint", "rr_femur_joint", "rr_tibia_joint",
        ]
        
        # Publisher
        self.ctrl_pub = self.create_publisher(
            Float64MultiArray,
            "/position_controller/commands",
            10
        )
        
        # Subscriber to verify what's actually happening
        self.state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.state_callback,
            10
        )
        
        self.current_positions = {}
        
        self.get_logger().info("🔍 Joint Mapping Diagnostic Tool Ready")
        self.get_logger().info("=" * 70)
        
        # Wait for first state
        time.sleep(1.0)
        
        # Run tests
        self.run_tests()

    def state_callback(self, msg: JointState):
        """Store current joint states."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]

    def send_command(self, positions):
        """Send joint position command."""
        msg = Float64MultiArray()
        msg.data = positions
        self.ctrl_pub.publish(msg)
        
    def wait_and_check(self, test_name, expected_leg, duration=2.0):
        """Wait and report which joints actually moved."""
        self.get_logger().info(f"\n{'='*70}")
        self.get_logger().info(f"TEST: {test_name}")
        self.get_logger().info(f"Expected to move: {expected_leg}")
        self.get_logger().info(f"{'='*70}")
        
        # Record initial positions
        initial = self.current_positions.copy()
        
        # Wait for motion
        time.sleep(duration)
        
        # Check what moved
        moved_joints = []
        for joint in self.joint_order:
            if joint in initial and joint in self.current_positions:
                delta = abs(self.current_positions[joint] - initial[joint])
                if delta > 0.05:  # 0.05 rad threshold
                    moved_joints.append((joint, delta))
        
        if moved_joints:
            self.get_logger().info("✓ Joints that moved:")
            for joint, delta in moved_joints:
                leg = joint.split('_')[0].upper()
                self.get_logger().info(f"  • {joint:20s} (delta: {delta:+.3f} rad) [{leg}]")
        else:
            self.get_logger().warn("✗ No joints moved! Check controller.")
        
        # Return to neutral
        neutral = [0.0] * 12
        self.send_command(neutral)
        time.sleep(1.0)

    def run_tests(self):
        """Run a series of diagnostic tests."""
        
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("STARTING JOINT MAPPING DIAGNOSTICS")
        self.get_logger().info("="*70)
        self.get_logger().info("This will test each leg individually to verify mapping.")
        self.get_logger().info("Watch the robot in Gazebo and confirm motion matches expected leg.")
        self.get_logger().info("="*70)
        
        time.sleep(2.0)
        
        # Test 1: FL leg only
        cmd = [0.3, 0.0, 0.0,   # FL: move coxa
               0.0, 0.0, 0.0,   # FR
               0.0, 0.0, 0.0,   # RL
               0.0, 0.0, 0.0]   # RR
        self.send_command(cmd)
        self.wait_and_check("FL Coxa Test", "Front Left leg (green)")
        
        # Test 2: FR leg only  
        cmd = [0.0, 0.0, 0.0,   # FL
               0.3, 0.0, 0.0,   # FR: move coxa
               0.0, 0.0, 0.0,   # RL
               0.0, 0.0, 0.0]   # RR
        self.send_command(cmd)
        self.wait_and_check("FR Coxa Test", "Front Right leg (green)")
        
        # Test 3: RL leg only
        cmd = [0.0, 0.0, 0.0,   # FL
               0.0, 0.0, 0.0,   # FR
               0.3, 0.0, 0.0,   # RL: move coxa
               0.0, 0.0, 0.0]   # RR
        self.send_command(cmd)
        self.wait_and_check("RL Coxa Test", "Rear Left leg (red)")
        
        # Test 4: RR leg only
        cmd = [0.0, 0.0, 0.0,   # FL
               0.0, 0.0, 0.0,   # FR
               0.0, 0.0, 0.0,   # RL
               0.3, 0.0, 0.0]   # RR: move coxa
        self.send_command(cmd)
        self.wait_and_check("RR Coxa Test", "Rear Right leg (red)")
        
        # Test 5: Diagonal pair 1 (should be FL + RR for trot)
        cmd = [0.3, 0.0, 0.0,   # FL
               0.0, 0.0, 0.0,   # FR
               0.0, 0.0, 0.0,   # RL
               0.3, 0.0, 0.0]   # RR
        self.send_command(cmd)
        self.wait_and_check("Diagonal 1 Test (FL+RR)", "Front Left + Rear Right")
        
        # Test 6: Diagonal pair 2 (should be FR + RL for trot)
        cmd = [0.0, 0.0, 0.0,   # FL
               0.3, 0.0, 0.0,   # FR
               0.3, 0.0, 0.0,   # RL
               0.0, 0.0, 0.0]   # RR
        self.send_command(cmd)
        self.wait_and_check("Diagonal 2 Test (FR+RL)", "Front Right + Rear Left")
        
        # Final summary
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("DIAGNOSTIC COMPLETE")
        self.get_logger().info("="*70)
        self.get_logger().info("\nVERIFY THE FOLLOWING:")
        self.get_logger().info("1. Each test moved ONLY the expected leg(s)")
        self.get_logger().info("2. FL and RR form a diagonal pair (both green or both red)")
        self.get_logger().info("3. FR and RL form a diagonal pair")
        self.get_logger().info("\nIf mismatches found, you need to:")
        self.get_logger().info("• Fix URDF axis definitions")
        self.get_logger().info("• Add sign corrections to policy node")
        self.get_logger().info("• Verify joint order in controller.yaml")
        self.get_logger().info("="*70)


def main():
    rclpy.init()
    node = JointDiagnostic()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()