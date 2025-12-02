#!/usr/bin/env python3
"""
Complete VLM Navigation System Launch File

Starts:
1. SLAM system
2. VLM scene planner
3. VLM dashboard
4. Robot control
5. Teleop

Usage: ros2 launch spiderbot_vlm vlm_full.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    
    # Get package directories
    slam_pkg = get_package_share_directory('spiderbot_slam')
    
    # Config files
    slam_config = PathJoinSubstitution([slam_pkg, 'config', 'slam_toolbox.yaml'])
    rviz_config = PathJoinSubstitution([slam_pkg, 'rviz', 'slam.rviz'])
    
    # URDF
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
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base_link',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link']
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_laser_tf',
            arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser_frame']
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu_tf',
            arguments=['0', '0', '0.05', '0', '0', '0', 'base_link', 'imu_link']
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera_tf',
            arguments=['0.1', '0', '0.15', '0', '0', '0', 'base_link', 'camera_link']
        ),
        
        # ============================================================
        # 3. MINIMAL IMU ODOMETRY
        # ============================================================
        Node(
            package='spiderbot_slam',
            executable='minimal_imu_odom.py',
            name='minimal_imu_odom',
            output='screen'
        ),
        
        # ============================================================
        # 4. SLAM TOOLBOX
        # ============================================================
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_config]
        ),
        
        # ============================================================
        # 5. VLM SCENE PLANNER
        # ============================================================
        Node(
            package='spiderbot_vlm',
            executable='vlm_scene_planner_node',
            name='vlm_scene_planner',
            output='screen',
            parameters=[{
                'camera_topic': '/camera/image_raw',
                'goal_topic': '/vlm_goal',
                'cmd_topic': '/cmd',
                'model_name': 'gemini-2.5-flash'
            }]
        ),
        
        # ============================================================
        # 6. VLM DASHBOARD
        # ============================================================
        Node(
            package='spiderbot_vlm',
            executable='vlm_dashboard_node',
            name='vlm_dashboard',
            output='screen',
            parameters=[{
                'camera_topic': '/camera/image_raw',
                'port': 5000
            }]
        ),
        
        # ============================================================
        # 7. ROBOT CONTROL (Policy Node)
        # ============================================================
        # Node(
        #     package='spiderbot_control',
        #     executable='policy_omni_node',
        #     name='policy_omni_node',
        #     output='screen',
        #     prefix='gnome-terminal --'
        # ),
        
        # ============================================================
        # 8. KEYBOARD TELEOP (Backup Control)
        # ============================================================
        Node(
            package='spiderbot_control',
            executable='control',
            name='control',
            output='screen',
            prefix='gnome-terminal --'
        ),
        
        # ============================================================
        # 9. RVIZ VISUALIZATION
        # ============================================================
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),
        
        # ============================================================
        # 10. BROWSER LAUNCHER (Optional)
        # ============================================================
        ExecuteProcess(
            cmd=['xdg-open', 'http://localhost:5000'],
            shell=False,
            output='screen'
        ),
    ])