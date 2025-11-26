#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, 
    DeclareLaunchArgument, 
    ExecuteProcess, 
    TimerAction,
    SetEnvironmentVariable
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Paths
    slam_share = FindPackageShare('spiderbot_slam')
    desc_share = get_package_share_directory('spiderbot_description')
    
    slam_config = PathJoinSubstitution([slam_share, 'config', 'slam_toolbox.yaml'])
    rviz_config = PathJoinSubstitution([slam_share, 'config', 'slam_view.rviz'])
    world_file = PathJoinSubstitution([slam_share, 'worlds', 'test_world.sdf'])
    xacro_file = os.path.join(desc_share, 'urdf', 'spidy.xacro')
    
    pkg_share_parent = os.path.dirname(desc_share)
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Environment variables for Gazebo
    set_ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            EnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', default_value=''),
            ':',
            pkg_share_parent
        ]
    )
    
    set_ign_plugin_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_SYSTEM_PLUGIN_PATH',
        value=[
            EnvironmentVariable('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', default_value=''),
            ':',
            '/opt/ros/humble/lib'
        ]
    )
    
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )
    
    # Gazebo
    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world_file],
        output='screen'
    )
    
    # === BRIDGES: Gazebo ↔ ROS2 ===
    
    # Clock bridge (for time synchronization)
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen'
    )
    
    # LIDAR bridge (Gazebo sensor → ROS2 /scan)
    lidar_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='lidar_bridge',
        arguments=['/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan'],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # IMU bridge (Gazebo sensor → ROS2 /imu)
    imu_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='imu_bridge',
        arguments=['/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU'],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # Robot State Publisher
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
            'publish_frequency': 50.0
        }]
    )
    
    # Spawn robot
    spawn_robot = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-world', 'slam_test',
                    '-topic', 'robot_description',
                    '-name', 'spiderbot',
                    '-x', '0',
                    '-y', '0',
                    '-z', '0.5'
                ],
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        ]
    )
    
    # Controllers
    load_joint_state_broadcaster = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        ]
    )
    
    load_position_controller = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['position_controller', '-c', '/controller_manager'],
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        ]
    )
    
    # SLAM Toolbox
    slam_node = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[
                    slam_config,
                    {'use_sim_time': True}
                ],
            )
        ]
    )
    
    # Policy controller
    policy_node = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='spiderbot_control',
                executable='policy_omni_node',
                name='policy_omni_node',
                output='screen',
                parameters=[{'use_sim_time': True}],
                prefix='gnome-terminal --'
            )
        ]
    )
    
    # Teleop
    teleop_node = Node(
        package='spiderbot_control',
        executable='teleop_keyboard',
        name='teleop_keyboard',
        output='screen',
        prefix='gnome-terminal --',
    )
    
    # RViz
    rviz_node = TimerAction(
        period=15.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config],
                parameters=[{'use_sim_time': True}],
                output='screen'
            )
        ]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        set_ign_resource_path,
        set_ign_plugin_path,
        gazebo,
        clock_bridge,    # Time sync
        lidar_bridge,    # /scan topic
        imu_bridge,      # /imu topic
        robot_state_pub,
        spawn_robot,
        load_joint_state_broadcaster,
        load_position_controller,
        slam_node,
        policy_node,
        teleop_node,
        rviz_node,
    ])