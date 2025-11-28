#!/usr/bin/env python3
"""
Minimal IMU Odometry for Quadruped Robot

Integrates IMU angular velocity to provide odometry for SLAM.
No position estimation - just rotation tracking.
SLAM will handle position via scan matching.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math


class MinimalImuOdom(Node):
    def __init__(self):
        super().__init__('minimal_imu_odom')
        
        # State: only track yaw (rotation around Z)
        self.yaw = 0.0
        self.last_time = None
        
        # Subscribe to IMU
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10
        )
        
        # Publish odometry
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        
        # Publish TF
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info('Minimal IMU Odometry Started')
        self.get_logger().info('  Integrating yaw rate from IMU')
        self.get_logger().info('  SLAM will handle position via scan matching')
    
    def imu_callback(self, msg: Imu):
        """Integrate IMU angular velocity to track yaw"""
        
        current_time = self.get_clock().now()
        
        # Initialize on first message
        if self.last_time is None:
            self.last_time = current_time
            return
        
        # Calculate dt
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        # Integrate yaw rate (angular velocity around Z)
        wz = msg.angular_velocity.z
        self.yaw += wz * dt
        
        # Normalize to [-pi, pi]
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        
        # Publish odometry
        self.publish_odometry(current_time)
    
    def publish_odometry(self, stamp):
        """Publish odometry message and TF"""
        
        # Create quaternion from yaw
        qz = math.sin(self.yaw / 2.0)
        qw = math.cos(self.yaw / 2.0)
        
        # Publish TF: odom → base_link
        t = TransformStamped()
        t.header.stamp = stamp.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        
        # Position: (0, 0, 0) - SLAM will handle this via scan matching
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        # Orientation: only yaw changes
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        self.tf_broadcaster.sendTransform(t)
        
        # Publish Odometry message
        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Position: (0, 0, 0)
        odom.pose.pose.position.x = 0.0
        odom.pose.pose.position.y = 0.0
        odom.pose.pose.position.z = 0.0
        
        # Orientation: yaw only
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        
        # Covariance: high uncertainty in position, low in orientation
        odom.pose.covariance[0] = 999999.0  # x (unknown)
        odom.pose.covariance[7] = 999999.0  # y (unknown)
        odom.pose.covariance[35] = 0.1      # yaw (known from IMU)
        
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = MinimalImuOdom()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
