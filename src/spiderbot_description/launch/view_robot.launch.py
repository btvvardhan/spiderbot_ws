from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg = FindPackageShare('spiderbot_description')
    urdf_path = PathJoinSubstitution([pkg, 'urdf', 'spidy.urdf'])
    robot_description = ParameterValue(Command(['cat ', urdf_path]), value_type=str)
    rviz_cfg = PathJoinSubstitution([pkg, 'rviz', 'robot.rviz'])

    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
     #    Node(package='joint_state_publisher', executable='joint_state_publisher',
     #         parameters=[{'source_list': ['/anim_joint_states'], 'rate': 50.0}]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui'),
        #Node(package='spiderbot_description', executable='animate_spiderbot.py'),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_cfg]),
    ])
