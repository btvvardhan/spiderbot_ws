#!/usr/bin/env python3
"""
Complete SLAM Pipeline for Physical Robot
- LiDAR: YDLidar X2
- IMU: MPU9250
- EKF: Fuses IMU data for odometry
- SLAM Toolbox: Creates map using LiDAR + fused odometry
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    # Package directories
    slam_pkg = get_package_share_directory('spiderbot_slam')
    
    # Config files
    slam_config = PathJoinSubstitution([slam_pkg, 'config', 'slam_toolbox.yaml'])
    ekf_config = PathJoinSubstitution([slam_pkg, 'config', 'ekf_config.yaml'])
    rviz_config = PathJoinSubstitution([slam_pkg, 'rviz', 'slam.rviz'])
    
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
                'robot_description': open('/home/teja/spiderbot_ws/src/spiderbot_description/urdf/spidy.urdf').read()
            }]
        ),
        
        # ============================================================
        # 2. STATIC TRANSFORMS
        # ============================================================
        
        # base_link → laser_frame (LiDAR mounting position)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_laser_tf',
            arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser_frame']
        ),
        
        # base_link → imu_link (IMU mounting position)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu_tf',
            arguments=['0', '0', '0.05', '0', '0', '0', 'base_link', 'imu_link']
        ),
        
        # ============================================================
        # 3. EKF NODE (Sensor Fusion)
        # ============================================================
        # Fuses IMU data to create odometry estimate
        # Publishes: /odometry/filtered and odom→base_link TF
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[
                ekf_config,
                {'use_sim_time': False}
            ],
            remappings=[
                ('odometry/filtered', 'odometry/filtered')
            ]
        ),
        
        # ============================================================
        # 4. SLAM TOOLBOX
        # ============================================================
        # Creates map using LiDAR scans and EKF odometry
        # Publishes: /map and map→odom TF
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