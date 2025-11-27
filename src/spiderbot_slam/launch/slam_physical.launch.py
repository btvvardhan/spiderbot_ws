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
    rviz_config = PathJoinSubstitution([slam_share, 'config', 'slam_view.rviz'])
    
    # Robot description
    robot_description = ParameterValue(
        Command(['cat ', urdf_path]),
        value_type=str
    )
    
    return LaunchDescription([
        # ============================================
        # RASPBERRY PI: Launch on the Pi first!
        # ssh vijay@192.168.1.210
        # source ~/spiderbot_pi_ws/install/setup.bash
        # export ROS_DOMAIN_ID=7
        # ros2 launch spiderbot_bridge spiderbot_full.launch.py
        # ============================================
        
        # Robot State Publisher (publishes joint transforms)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen'
        ),
        
        # TF: odom -> base_link (static for now, no drift)
        # Later we'll replace this with proper odometry from wheel encoders or EKF
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base_link',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
            output='screen'
        ),
        
        # TF: base_link -> laser_frame
        # Adjust x, y, z to match your LiDAR mounting position
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_laser',
            arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser_frame'],
            output='screen'
        ),
        
        # SLAM Toolbox (creates map -> odom transform)
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
            prefix='gnome-terminal --',
        ),
        
        # Teleop
        Node(
            package='spiderbot_control',
            executable='teleop_keyboard',
            name='teleop_keyboard',
            output='screen',
            prefix='gnome-terminal --',
        ),
        
        # RViz for Visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),
    ])