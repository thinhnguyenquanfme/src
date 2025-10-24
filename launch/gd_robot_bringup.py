
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    grab_frame_node = Node(
        package='camera_worker_pkg',
        executable='basler_snapshot_server'
    )

    camera_undistort_node = Node(
        package='camera_worker_pkg',
        executable='undistort_img_node'
    )

    canny_edge_node = Node(
        package='camera_worker_pkg',
        executable='canny_edge_node'
    )

    ld.add_action(grab_frame_node)
    ld.add_action(camera_undistort_node)
    ld.add_action(canny_edge_node)

    return ld