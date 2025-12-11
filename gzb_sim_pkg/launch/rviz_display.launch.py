from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg_share = FindPackageShare('gzb_sim_pkg')

    # Dùng xacro load URDF
    robot_file = PathJoinSubstitution([pkg_share, 'urdf', 'sim_Assem.SLDASM.urdf'])
    robot_description = ParameterValue(Command(['xacro ', robot_file]), value_type=str)

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }],
        output='screen'
    )

    # Mở RViz2 default
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen'
    )

    return LaunchDescription([rsp, rviz])
