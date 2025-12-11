from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg_share = FindPackageShare('gzb_sim_pkg')
    world_file = PathJoinSubstitution([pkg_share, 'worlds', 'building_sim_Assem.sdf'])

    # Gazebo (Ignition / ros_gz_sim) cơ bản
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={'gz_args': ['-r ', world_file]}.items(),
    )

    # Robot description: xacro/URDF -> string
    robot_file = PathJoinSubstitution([pkg_share, 'urdf', 'sim_Assem.SLDASM.urdf'])
    robot_description_cmd = Command(['xacro ', robot_file])
    robot_description = ParameterValue(robot_description_cmd, value_type=str)

    # Chỉ publish TF + spawn robot vào Gazebo, không ros2_control
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }]
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', robot_description_cmd,
            '-name', 'assem_robot',
            '-allow_renaming', 'true'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        rsp,
        spawn,
    ])
