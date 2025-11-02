#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('spiderbot_description')
    xacro_file = os.path.join(pkg_share, 'urdf', 'spiderbot.urdf.xacro')
    
    # Process xacro to URDF - this resolves $(find ...) substitutions
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )

    # 1. robot_state_publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': True}
        ],
        output='screen'
    )

    # 2. Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    # 3. Spawn robot - use robot_description topic
    spawn_robot = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-world', 'empty',
                    '-topic', 'robot_description',  # Use the description from topic
                    '-name', 'spiderbot',
                    '-z', '0.5'
                ],
                output='screen'
            )
        ]
    )

    # 4. Spawn controllers
    load_joint_state_broadcaster = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
                output='screen'
            )
        ]
    )

    load_position_controller = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['position_controller', '-c', '/controller_manager'],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        robot_state_publisher,
        gazebo,
        spawn_robot,
        load_joint_state_broadcaster,
        load_position_controller,
    ])