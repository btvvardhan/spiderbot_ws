#!/usr/bin/env python3
"""
Complete SLAM Pipeline for Physical Robot (MAIN COMPUTER)
- Subscribes to /scan from Pi's LiDAR
- Subscribes to /imu from Pi's Arduino
- Subscribes to /camera/image_raw/compressed from Pi's camera
- Processes IMU data for odometry
- Runs SLAM for mapping
- Visualizes everything in RViz

Location: ~/spiderbot_ws/src/spiderbot_slam/launch/slam_minimal.launch.py
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
        
        # base_link → camera_link (Camera mounting - adjust x,y,z to match your robot)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera_tf',
            arguments=['--frame-id', 'base_link', '--child-frame-id', 'camera_link',
                      '--x', '0.1', '--y', '0', '--z', '0.15',  # Adjust these!
                      '--roll', '0', '--pitch', '0', '--yaw', '0']
        ),
        
        # ============================================================
        # 3. MINIMAL IMU ODOMETRY
        # ============================================================
        # Subscribes to /imu from Pi
        # Publishes odom→base_link TF with yaw from IMU
        Node(
            package='spiderbot_slam',
            executable='minimal_imu_odom.py',
            name='minimal_imu_odom',
            output='screen',
            parameters=[{'use_sim_time': False}]
        ),
        
        # ============================================================
        # 4. SLAM TOOLBOX
        # ============================================================
        # Subscribes to /scan from Pi
        # Uses odom→base_link TF from minimal_imu_odom
        # Publishes /map and map→odom TF
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
        # Displays /map, /scan, /camera/image_raw/compressed from Pi
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
            parameters=[{'use_sim_time': False}]
        ),
    ])