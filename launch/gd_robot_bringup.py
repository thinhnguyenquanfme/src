
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
    
    object_segment_node = Node(
        package='camera_worker_pkg',
        executable='object_segment_node'
    )

    trajectory_planning_node = Node(
        package='geometry_calculate',
        executable='trajectory_planning_node'
    )

    trajectory_graph_node = Node(
        package='geometry_calculate',
        executable='trajectory_graph_node'
    )

    plc_communicate_node = Node(
        package='plc_worker_pkg',
        executable='plc_communicate_node'
    )

    plc_monitor_node = Node(
        package='plc_worker_pkg',
        executable='plc_monitor_node'
    )

    ld.add_action(grab_frame_node)
    ld.add_action(camera_undistort_node)
    ld.add_action(object_segment_node)
    ld.add_action(trajectory_planning_node)
    ld.add_action(trajectory_graph_node)
    ld.add_action(plc_communicate_node)
    ld.add_action(plc_monitor_node)

    return ld