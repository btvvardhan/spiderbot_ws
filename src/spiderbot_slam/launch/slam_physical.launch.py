#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Get package shares
    desc_share = FindPackageShare('spiderbot_description')
    slam_share = FindPackageShare('spiderbot_slam')
    
    # Paths
    urdf_path = PathJoinSubstitution([desc_share, 'urdf', 'spidy.urdf'])
    slam_config = PathJoinSubstitution([slam_share, 'config', 'slam_toolbox.yaml'])
    
    # Robot description
    robot_description = ParameterValue(
        Command(['cat ', urdf_path]),
        value_type=str
    )
    
    return LaunchDescription([
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen'
        ),
        
        # LIDAR Driver (adjust for your LIDAR model)
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar_node',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'serial_baudrate': 115200,
                'frame_id': 'laser_link',
                'angle_compensate': True,
            }],
            output='screen'
        ),
        
        # SLAM Toolbox
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                slam_config,
                {'use_sim_time': False}
            ],
        ),
        
        # Policy Controller
        Node(
            package='spiderbot_control',
            executable='policy_omni_node',
            name='policy_omni_node',
            output='screen',
        ),
        
        # Serial Bridge to Arduino
        Node(
            package='spiderbot_control',
            executable='serial_bridge_jointstate',
            name='serial_bridge',
            output='screen',
        ),
        
        # Teleop
        Node(
            package='spiderbot_control',
            executable='teleop_keyboard',
            name='teleop_keyboard',
            output='screen',
            prefix='xterm -e',
        ),
    ])