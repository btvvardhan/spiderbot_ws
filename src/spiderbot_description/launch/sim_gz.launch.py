#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, PathJoinSubstitution
from launch.actions import SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable, PathJoinSubstitution
from launch.actions import TimerAction

pkg_share_parent = PathJoinSubstitution([FindPackageShare('spiderbot_description'), '..'])
set_ign_path = SetEnvironmentVariable(
    name='IGN_GAZEBO_RESOURCE_PATH',
    value=[EnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', default_value=''), ':', pkg_share_parent]
)
# Then include `set_ign_path` in the LaunchDescription list (before gz)





def generate_launch_description():
    pkg = FindPackageShare('spiderbot_description')
    urdf = PathJoinSubstitution([pkg, 'urdf', 'spiderbot.urdf'])
    controllers = PathJoinSubstitution([pkg, 'config', 'controllers.yaml'])

    # Publish URDF for TF (RViz etc.)
    robot_description = ParameterValue(Command(['cat ', urdf]), value_type=str)
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    # Start Gazebo (GZ) with an explicit world
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('ros_gz_sim'), '/launch/gz_sim.launch.py']
        ),
        launch_arguments={'gz_args': '-r -v 3 empty.sdf'}.items()
    )

    # Spawn the robot into that world after GZ is up
    spawn = TimerAction(period=5.0, actions=[
        Node(
            package='ros_gz_sim', executable='create', output='screen',
            arguments=['-world', 'empty', '-file', urdf, '-name', 'spiderbot']
        )
    ])


    jsb = TimerAction(period=6.0, actions=[
        Node(package='controller_manager', executable='spawner', output='screen',
            arguments=[
                'joint_state_broadcaster',
                '--controller-manager', '/controller_manager',
                '--activate'  # optional; ensures it tries to go ACTIVE
            ])
    ])

    pos = TimerAction(period=7.2, actions=[
        Node(package='controller_manager', executable='spawner', output='screen',
            arguments=[
                'position_controller',
                '--controller-manager', '/controller_manager',
                '--activate'
            ])
    ])




    return LaunchDescription([set_ign_path, rsp, gz, spawn, jsb, pos])
