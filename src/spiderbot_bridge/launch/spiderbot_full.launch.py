#!/usr/bin/env python3
"""
Complete Spiderbot Launch File for Raspberry Pi
- Starts YDLidar driver → publishes /scan
- Starts Pi hard node (Arduino + IMU) → publishes /imu, /joint_states
- Starts USB camera → publishes /camera/image_raw
- Compresses camera images → publishes /camera/image_raw/compressed

Location: ~/spiderbot_pi_ws/src/spiderbot_hard/launch/spiderbot_full.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Arguments
    arduino_port_arg = DeclareLaunchArgument(
        'arduino_port',
        default_value='/dev/ttyACM0',
        description='Arduino serial port'
    )
    
    lidar_port_arg = DeclareLaunchArgument(
        'lidar_port',
        default_value='/dev/ttyUSB0',
        description='YDLidar serial port'
    )
    
    camera_device_arg = DeclareLaunchArgument(
        'camera_device',
        default_value='/dev/video0',
        description='Camera device'
    )

    # YDLidar launch file (publishes /scan)
    ydlidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ydlidar_ros2_driver'),
                'launch',
                'ydlidar_launch.py'
            ])
        ])
    )

    # Pi hard Node (publishes /imu and /joint_states)
    pi_hard_node = Node(
        package='spiderbot_bridge',
        executable='pi_hard_node',
        name='pi_hard_node',
        output='screen',
        parameters=[{
            'arduino_port': LaunchConfiguration('arduino_port'),
            'baudrate': 115200,
            'deg_offset': 90.0,
            'servo_min_deg': 0.0,
            'servo_max_deg': 180.0,
        }],
        respawn=True,
        respawn_delay=2.0
    )
    
    # USB Camera Node (publishes /camera/image_raw)
    camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration('camera_device'),
            'framerate': 30.0,
            'image_width': 640,
            'image_height': 480,
            'pixel_format': 'mjpeg2rgb',  # Logitech Brio supports MJPEG
            'camera_frame_id': 'camera_link',
            'io_method': 'mmap',
        }],
        remappings=[
            ('image_raw', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
        ],
        respawn=True,
        respawn_delay=2.0
    )
    
    # Image Compressor (publishes /camera/image_raw/compressed)
    # This saves ~90% bandwidth over WiFi!
    image_compressor = Node(
        package='image_transport',
        executable='republish',
        name='image_compressor',
        output='screen',
        arguments=[
            'raw',
            'compressed',
            '--ros-args',
            '--remap', 'in:=/camera/image_raw',
            '--remap', 'out/compressed:=/camera/image_raw/compressed',
            '--param', 'compressed.jpeg_quality:=70'
        ],
        respawn=True,
        respawn_delay=2.0
    )

    return LaunchDescription([
        arduino_port_arg,
        lidar_port_arg,
        camera_device_arg,
        ydlidar_launch,
        pi_hard_node,
        camera_node,
        image_compressor,
    ])