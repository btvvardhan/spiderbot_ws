#!/usr/bin/env python3
"""
Launch only the Pi bridge node (no LIDAR)
Useful for testing Arduino communication separately
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    arduino_port_arg = DeclareLaunchArgument(
        'arduino_port',
        default_value='/dev/ttyACM0',
        description='Arduino serial port'
    )

    pi_bridge_node = Node(
        package='spiderbot_bridge',
        executable='pi_bridge_node',
        name='pi_bridge_node',
        output='screen',
        parameters=[{
            'arduino_port': LaunchConfiguration('arduino_port'),
            'baudrate': 115200,
            'deg_offset': 90.0,
            'servo_min_deg': 0.0,
            'servo_max_deg': 180.0,
        }]
    )

    return LaunchDescription([
        arduino_port_arg,
        pi_bridge_node,
    ])