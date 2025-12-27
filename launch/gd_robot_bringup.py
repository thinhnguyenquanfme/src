
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

    plc_motion_node = Node(
        package='plc_worker_pkg',
        executable='plc_motion_node'
    )

    object_spawn_node = Node(
        package='system_sim_pkg',
        executable='object_spawn_node'
    )

    robot_state_sim_node = Node(
        package='system_sim_pkg',
        executable='robot_state_sim_node'
    )

    point_sim = Node(
        package='system_sim_pkg',
        executable='point_sim'
    )

    trajectory_plotter = Node(
        package='system_sim_pkg',
        executable='trajectory_plotter'
    )

    ld.add_action(grab_frame_node)
    ld.add_action(camera_undistort_node)
    ld.add_action(object_segment_node)
    ld.add_action(trajectory_planning_node)
    ld.add_action(trajectory_graph_node)
    ld.add_action(plc_communicate_node)
    ld.add_action(plc_monitor_node)
    ld.add_action(object_spawn_node)
    ld.add_action(robot_state_sim_node)
    ld.add_action(point_sim)
    ld.add_action(trajectory_plotter)
    ld.add_action(plc_motion_node)

    return ld