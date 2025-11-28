#!/usr/bin/env python3
"""
Simplified SLAM Pipeline for Physical Robot
- LiDAR: YDLidar X2  
- IMU: MPU9250 (only for rotation tracking)
- Minimal odometry: Integrates IMU yaw, SLAM handles position
"""

import os
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    # Package directories
    slam_pkg = get_package_share_directory('spiderbot_slam')
    
    # Config files
    slam_config = PathJoinSubstitution([slam_pkg, 'config', 'slam_toolbox.yaml'])
    rviz_config = PathJoinSubstitution([slam_pkg, 'rviz', 'slam.rviz'])
    
    # URDF path
    urdf_file = '/home/teja/spiderbot_ws/src/spiderbot_description/urdf/spidy.urdf'
    
    return LaunchDescription([
        
        # ============================================================
        # 1. ROBOT STATE PUBLISHER
        # ============================================================
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'robot_description': open(urdf_file).read()
            }]
        ),
        
        # ============================================================
        # 2. STATIC TRANSFORMS
        # ============================================================
        
        # base_link → laser_frame (LiDAR mounting)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_laser_tf',
            arguments=['--frame-id', 'base_link', '--child-frame-id', 'laser_frame',
                      '--x', '0', '--y', '0', '--z', '0.1',
                      '--roll', '0', '--pitch', '0', '--yaw', '0']
        ),
        
        # base_link → imu_link (IMU mounting)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu_tf',
            arguments=['--frame-id', 'base_link', '--child-frame-id', 'imu_link',
                      '--x', '0', '--y', '0', '--z', '0.05',
                      '--roll', '0', '--pitch', '0', '--yaw', '0']
        ),
        
        # ============================================================
        # 3. MINIMAL IMU ODOMETRY
        # ============================================================
        # Integrates IMU yaw rate, publishes odom→base_link TF
        # Position is (0,0) - SLAM handles position via scan matching
        Node(
            package='spiderbot_slam',
            executable='minimal_imu_odom',
            name='minimal_imu_odom',
            output='screen',
            parameters=[{'use_sim_time': False}]
        ),
        
        # ============================================================
        # 4. SLAM TOOLBOX
        # ============================================================
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                slam_config,
                {'use_sim_time': False}
            ]
        ),
        
        # ============================================================
        # 5. RVIZ VISUALIZATION
        # ============================================================
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
            parameters=[{'use_sim_time': False}]
        ),
    ])
