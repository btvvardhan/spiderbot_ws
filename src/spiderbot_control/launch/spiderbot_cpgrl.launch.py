from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='spiderbot_control',
            executable='cpgrl_controller',
            output='screen',
            parameters=[{
                'model_path': '',             # set to your .pt when ready
                'use_model': False,           # True when your TorchScript is in place
                'rate_hz': 100.0,
                'cmd_topic': '/cmd_vel',
                'out_topic': '/position_controller/commands',
                'action_scale': 1.0,
                'max_joint_rad': 1.2,
                'joint_order': [
                    "Revolute 77","Revolute 78","Revolute 79","Revolute 80",
                    "Revolute 90","Revolute 91","Revolute 92","Revolute 93",
                    "Revolute 102","Revolute 103","Revolute 104","Revolute 105",
                ],
            }],
        ),
    ])
