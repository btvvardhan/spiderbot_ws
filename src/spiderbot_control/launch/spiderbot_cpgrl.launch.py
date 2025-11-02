# spiderbot_control/launch/spiderbot_cpgrl.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    use_model  = LaunchConfiguration('use_model')
    model_path = LaunchConfiguration('model_path')
    rate_hz    = LaunchConfiguration('rate_hz')
    cmd_topic  = LaunchConfiguration('cmd_topic')
    out_topic  = LaunchConfiguration('out_topic')

    return LaunchDescription([
        # ---- CLI args (with sensible defaults) ----
        DeclareLaunchArgument('use_model',  default_value='false',  description='Enable RL policy'),
        DeclareLaunchArgument('model_path', default_value='',       description='Path to TorchScript .pt'),
        DeclareLaunchArgument('rate_hz',    default_value='100.0',  description='Control loop rate'),
        DeclareLaunchArgument('cmd_topic',  default_value='/cmd_vel'),
        DeclareLaunchArgument('out_topic',  default_value='/position_controller/commands'),

        # ---- Node ----
        Node(
            package='spiderbot_control',
            executable='cpgrl_controller',
            name='cpgrl_controller',
            output='screen',
            parameters=[{
                'use_model':   ParameterValue(use_model, value_type=bool),
                'model_path':  model_path,
                'rate_hz':     ParameterValue(rate_hz, value_type=float),
                'cmd_topic':   cmd_topic,
                'out_topic':   out_topic,
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
