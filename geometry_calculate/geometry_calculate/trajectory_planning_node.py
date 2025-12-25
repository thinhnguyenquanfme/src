import numpy as np
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.time import Time
from rclpy.clock import Clock

from robot_interfaces.srv import PoseListSrv
from robot_interfaces.msg import PoseListMsg, PoseStampedConveyor
from geometry_msgs.msg import Transform, Vector3, Twist
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from std_srvs.srv import SetBool

PI = np.pi

class TrajectoryPlanning(Node):
    def __init__(self):
        super().__init__('trajectory_planning_node')

        # =============== Set parameter ===============
        self.declare_parameter('conveyor_speed_x', -20.0)           # Conveyor speed in robot X axis (mm/s)
        self.declare_parameter('conveyor_speed_y', 0.0)             # Conveyor speed in robot Y axis (mm/s)
        self.declare_parameter('dispensing_speed', 100.0)           # Robot speed when dispensing glue (mm/s)
        self.declare_parameter('radius', 25.0)                      # Glue circle radius (mm)
        self.declare_parameter('n_segment', 40)                     # Divide glue circle into 20 segment
        self.declare_parameter('rest_point_x', 300.0)               # Waiting position of robot
        self.declare_parameter('rest_point_y', 100.0)               # //
        self.declare_parameter('rest_point_z', 0.0)               # //
        self.declare_parameter('dispensing_z', 30.0)             
        self.declare_parameter('cycle_time', 4.0)                   # Minimum time between 2 dispensing (s)
        self.declare_parameter('working_space_lower', 400.0)        # Lower limit of working space (mm)
        self.declare_parameter('working_space_upper', 10.0)         # Upper limit of working space (mm)

        self.conveyor_speed_x = self.get_parameter('conveyor_speed_x').value
        self.conveyor_speed_y = self.get_parameter('conveyor_speed_y').value
        self.dispensing_speed = self.get_parameter('dispensing_speed').value
        self.radius = self.get_parameter('radius').value
        self.n_segment = self.get_parameter('n_segment').value
        self.rest_pnt_x = self.get_parameter('rest_point_x').value
        self.rest_pnt_y = self.get_parameter('rest_point_y').value
        self.rest_pnt_z = self.get_parameter('rest_point_z').value
        self.dispensing_z = self.get_parameter('dispensing_z').value
        self.cycle_time = self.get_parameter('cycle_time').value
        self.ws_lower = self.get_parameter('working_space_lower').value
        self.ws_upper = self.get_parameter('working_space_upper').value
        
        self.prev_task_finish_time = 0.0
        
        
        # =============== Service define ===============
        self.start_command = self.create_service(SetBool, '/geometry/set_trajectory_start', self.handle_start_cmd)
        self.start_status = False
        self.idle_status_srv = self.create_service(SetBool, '/geometry/idle_status', self.handle_set_idle_status)
        self.idle_status = True
        self.timer_idle = None

        # =============== Publisher define ===============
        self.trajectory_transfrom_pub = self.create_publisher(PoseListMsg, '/geometry/trajectory_data', 10)
        # self.trajectory_numpy_pub = self.create_publisher()

        # =============== Subscriber define ===============
        self.object_center_sub = self.create_subscription(PoseStampedConveyor, '/geometry/camera_coord/object_center', self.handle_new_job, 1)

        # =============== Jobs queue ===============
        self.job_queue = []

         # =============== Define timer ===============
        self.timer_trajectory = self.create_timer(0.1, self.handle_trajectory)
        


    def log_trajectory(self, pose_list: PoseListMsg):
        tfs = pose_list.trajectory.transforms
        stamps = pose_list.stamp
        n = len(tfs)

        if n == 0:
            self.get_logger().info("Trajectory: no waypoints")
            return

        # Use first time as reference so you see 0.000, 0.100, ...
        t0 = float(stamps[0]) if len(stamps) > 0 else 0.0

        self.get_logger().info("Idx |   Time(s) |         X |         Y |         Z")
        self.get_logger().info("---------------------------------------------------")

        for i in range(n):
            t = float(stamps[i]) - t0 if len(stamps) > 0 else 0.0
            x = tfs[i].translation.x
            y = tfs[i].translation.y
            z = tfs[i].translation.z
            self.get_logger().info(
                f"{i:3d} | {t:9.3f} | {x:9.3f} | {y:9.3f} | {z:9.3f}"
            )
        

    def handle_new_job(self, msg: PoseStampedConveyor):
        if not self.start_status:
            return
        
        T = self.cycle_time

        x_r = self.rest_pnt_x
        y_r = self.rest_pnt_y
        z_r = self.rest_pnt_z

        z_D = self.dispensing_z

        v_cx = self.conveyor_speed_x
        v_cy = self.conveyor_speed_y
        
        v_D = self.dispensing_speed

        R = self.radius

        n = self.n_segment

        x_wmin = self.ws_lower
        x_wmax = self.ws_upper

        # ================ Collect data ================
        a = msg.pose.position.x
        b = msg.pose.position.y
        obj_id = float(msg.pose.position.z)
        t_do = Time.from_msg(msg.header.stamp) 
        t_do_s = float(t_do.nanoseconds) * 1e-9
        conv_pose = float(msg.conv_pose)
        # Calculate when obj go to workspace
        t_in = t_do_s + (x_wmin - a)/v_cx
        t_out = t_do_s + (x_wmax - a)/v_cx
        # Create new job
        new_job = Job()
        new_job.a = a
        new_job.b = b
        new_job.obj_id = obj_id
        new_job.t_in = t_in
        new_job.t_out = t_out
        new_job.t_do = t_do_s
        new_job.conv_pose = conv_pose
        # Add job to queue 
        self.job_queue.append(new_job)
        self.get_logger().info('add new job to queue')


    def handle_trajectory(self):
        if not self.idle_status:
            return
        if not self.job_queue:
            return

        R = self.radius
        v_D = self.dispensing_speed
        T = self.cycle_time

        DELTA = 0.1
        T_ding = 2 * PI * R / v_D
        T_rr = (T - T_ding) / 2
        T_lr = T_rr

        t_now = Time.from_msg(self.get_clock().now().to_msg())
        t_now_s = float(t_now.nanoseconds) * 1e-9

        for job in self.job_queue:
            job.ES = max(t_now_s, job.t_in - T_lr)
            job.LF = job.t_out - T_lr - T_ding - DELTA
            job.slack = job.LF - job.ES

        filtered_queue = []
        for job in self.job_queue:
            if t_now_s > job.t_out:
                continue
            elif job.ES > job.LF:
                continue
            else:
                filtered_queue.append(job)
        self.job_queue = filtered_queue

        if not filtered_queue:
            return
        
        chosen_job = min(self.job_queue, key=lambda j: j.slack)

        self.generate_trajectory(chosen_job)
        self.idle_status = False

        # Timer reset idle_status. use in simulation
        if self.timer_idle is None:
            self.timer_idle = self.create_timer(0.1, lambda: self.handle_idle(self.prev_task_finish_time))

        # Loại bỏ job đã xử lý để không lặp lại
        try:
            self.job_queue.remove(chosen_job)
        except ValueError:
            pass

    def generate_trajectory(self, job):
        """
        Generate one dispensing cycle for the selected job and publish as PoseListMsg.

        Phases:
        0) Rest pose at ES (earliest start)
        1) Move from rest to start of circle at time t_sd = ES + T_lr
        2) Follow moving circle (object on conveyor) during T_ding
        3) Move from end of circle back to rest at time t_end = ES + T
        """
        # -------------- Parameters --------------
        T = self.cycle_time          # total cycle time
        R = self.radius              # glue circle radius
        v_D = self.dispensing_speed  # dispensing speed along circle

        x_r = self.rest_pnt_x
        y_r = self.rest_pnt_y
        z_r = self.rest_pnt_z
        z_D = self.dispensing_z

        v_cx = self.conveyor_speed_x
        v_cy = self.conveyor_speed_y

        n = self.n_segment

        # -------------- Time decomposition --------------
        # Time to complete the circle
        T_ding = 2.0 * PI * R / v_D
        # Remaining time for approach + retreat
        T_rr = (T - T_ding) / 2.0
        T_lr = T_rr  # symmetric approach / retreat

        # Absolute times
        t_es = job.ES           # earliest start time of the task
        t_sd = t_es + T_lr      # time when nozzle reaches start point of circle
        t_end = t_es + T        # time when robot returns to rest

        # -------------- Prepare message --------------
        robot_trajectory = MultiDOFJointTrajectoryPoint()
        robot_trajectory.transforms = []
        robot_trajectory.velocities = []

        robot_pose_list = PoseListMsg()
        robot_pose_list.trajectory = robot_trajectory
        robot_pose_list.stamp = []
        robot_pose_list.conv_pose = 0.0

        # -------------- Phase 0: rest pose at ES --------------
        tf0 = Transform()
        tf0.translation.x = float(x_r)
        tf0.translation.y = float(y_r)
        tf0.translation.z = float(z_r)

        tw0 = Twist()
        tw0.linear.x = 0.0

        robot_trajectory.transforms.append(tf0)
        robot_trajectory.velocities.append(tw0)
        robot_pose_list.stamp.append(float(t_es))

        # -------------- Phase 1: rest -> start of circle --------------
        # Object center at time t_sd
        x_c0 = job.a + v_cx * (t_sd - job.t_do)
        y_c0 = job.b + v_cy * (t_sd - job.t_do)
        v_c = float(np.hypot(v_cx, v_cy))
        conv_pose_start = job.conv_pose + v_c * (t_sd - job.t_do)
        robot_pose_list.conv_pose = float(conv_pose_start)

        # Start point on circle at angle 0
        x_s = x_c0 + R * np.cos(0.0)
        y_s = y_c0 + R * np.sin(0.0)

        p_r = np.array([x_r, y_r, z_r], dtype=float)
        p_s = np.array([x_s, y_s, z_D], dtype=float)

        L_lr = np.linalg.norm(p_s - p_r)          # approach distance
        T_lr_safe = max(T_lr, 1e-6)               # avoid division by zero
        v_lr = L_lr / T_lr_safe                   # average speed for approach

        tf1 = Transform()
        tf1.translation.x = float(x_s)
        tf1.translation.y = float(y_s)
        tf1.translation.z = float(z_D)

        tw1 = Twist()
        tw1.linear.x = float(v_lr)

        robot_trajectory.transforms.append(tf1)
        robot_trajectory.velocities.append(tw1)
        robot_pose_list.stamp.append(float(t_sd))

        # -------------- Phase 2: follow moving circle --------------
        # Circle duration and time step per segment
        T_ding = 2.0 * PI * R / v_D
        dt = T_ding / float(n)
        dt_safe = max(dt, 1e-6)

        p_prev = p_s.copy()
        p_last_circle = p_s.copy()

        for i in range(n):
            # Absolute time of this waypoint
            t_i = t_sd + (i + 1) * dt_safe

            # Circle angle
            theta = (i + 1) * 2.0 * PI / float(n)

            # Object center at time t_i
            x_c_i = job.a + v_cx * (t_i - job.t_do)
            y_c_i = job.b + v_cy * (t_i - job.t_do)

            # Nozzle position following circle attached to moving object
            x_i = x_c_i + R * np.cos(theta)
            y_i = y_c_i + R * np.sin(theta)

            p_i = np.array([x_i, y_i, z_D], dtype=float)
            d_i = np.linalg.norm(p_i - p_prev)
            v_i = d_i / dt_safe

            p_prev = p_i
            p_last_circle = p_i

            tf_i = Transform()
            tf_i.translation.x = float(x_i)
            tf_i.translation.y = float(y_i)
            tf_i.translation.z = float(z_D)

            tw_i = Twist()
            tw_i.linear.x = float(v_i)

            robot_trajectory.transforms.append(tf_i)
            robot_trajectory.velocities.append(tw_i)
            robot_pose_list.stamp.append(float(t_i))

        # -------------- Phase 3: circle end -> rest --------------
        # End of circle time (for reference)
        t_circle_end = t_sd + T_ding

        p_e = p_last_circle
        L_rr = np.linalg.norm(p_r - p_e)      # retreat distance
        T_rr_safe = max(T_rr, 1e-6)
        v_rr = L_rr / T_rr_safe               # average retreat speed

        tf_end = Transform()
        tf_end.translation.x = float(x_r)
        tf_end.translation.y = float(y_r)
        tf_end.translation.z = float(z_r)

        tw_end = Twist()
        tw_end.linear.x = float(v_rr)

        robot_trajectory.transforms.append(tf_end)
        robot_trajectory.velocities.append(tw_end)
        robot_pose_list.stamp.append(float(t_end))

        # -------------- Publish --------------
        obj_id = getattr(job, "obj_id", 0.0)
        sec = int(obj_id)
        nanosec = int((obj_id - sec) * 1e9)
        robot_trajectory.time_from_start = DurationMsg(sec=sec, nanosec=nanosec)
        self.trajectory_transfrom_pub.publish(robot_pose_list)
        self.get_logger().info('Published trajectory with %d waypoints' %
                            (len(robot_trajectory.transforms)))

        # Save finish time for idle check
        self.prev_task_finish_time = t_end



    def handle_idle(self, t_e):
        t_now = Time.from_msg(self.get_clock().now().to_msg())
        t_now_s = float(t_now.nanoseconds) * 1e-9
        if t_now_s >= t_e:
            self.idle_status = True
            self.destroy_timer(self.timer_idle)
            self.timer_idle = None

    def handle_start_cmd(self, req, resp):
        self.start_status = bool(req.data)

        if req.data:
            resp.success = True
            resp.message = 'Set trajectory_planning_node to START'
            self.get_logger().info(resp.message)
            
        else: 
            resp.success = True
            resp.message = 'Set trajectory_planning_node to STOP'
            self.get_logger().info(resp.message)
        return resp  

    def handle_set_idle_status(self, req, resp):
        self.idle_status = bool(req.data)

        if req.data:
            resp.success = True
            resp.message = 'Set IDLE status to True'
            self.get_logger().info(resp.message)
            
        else: 
            resp.success = True
            resp.message = 'Set IDLE status to False'
            self.get_logger().info(resp.message)
        return resp  

@dataclass
class Job():
    a: float = 0.0
    b: float = 0.0
    obj_id: float = 0.0
    t_in: float = 0.0
    t_out: float = 0.0
    t_do: float = 0.0       #in second
    conv_pose: float = 0.0
    ES: float = 0.0
    LF: float = 0.0
    slack: float = 0.0



def main(args=None):
    rclpy.init(args=args)
    my_node = TrajectoryPlanning()
    rclpy.spin(my_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
