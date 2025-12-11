import numpy as np
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.clock import Clock

from robot_interfaces.srv import PoseListSrv
from robot_interfaces.msg import PoseListMsg
from geometry_msgs.msg import Transform, Vector3, PoseStamped, Twist
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
        self.declare_parameter('rest_point_y', 500.0)               # //
        self.declare_parameter('rest_point_z', 150.0)               # //
        self.declare_parameter('dispensing_z', 180.0)             
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
        self.object_center_sub = self.create_subscription(PoseStamped, '/geometry/camera_coord/object_center', self.handle_new_job, 1)

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
        

    def handle_new_job(self, msg: PoseStamped):
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
        t_do = Time.from_msg(msg.header.stamp) 
        t_do_s = float(t_do.nanoseconds) * 1e-9
        # Calculate when obj go to workspace
        t_in = t_do_s + (x_wmin - a)/v_cx
        t_out = t_do_s + (x_wmax - a)/v_cx
        # Create new job
        new_job = Job()
        new_job.a = a
        new_job.b = b
        new_job.t_in = t_in
        new_job.t_out = t_out
        new_job.t_do = t_do_s
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
        T = self.cycle_time
        R = self.radius
        v_D = self.dispensing_speed

        DELTA = 0.1
        T_ding = 2 * PI * R / v_D
        T_rr = (T - T_ding) / 2
        T_lr = T_rr

        x_r = self.rest_pnt_x
        y_r = self.rest_pnt_y
        z_r = self.rest_pnt_z

        z_D = self.dispensing_z

        v_cx = self.conveyor_speed_x
        v_cy = self.conveyor_speed_y
        
        v_D = self.dispensing_speed

        R = self.radius

        n = self.n_segment


        # ================ Publish data prepare ================
        robot_trajectory = MultiDOFJointTrajectoryPoint()  
        robot_trajectory.transforms = []

        robot_pose_list = PoseListMsg()
        robot_pose_list.trajectory = robot_trajectory
        robot_pose_list.stamp = []

        # Assign value for rest position
        tf = Transform()
        tf.translation.x = float(x_r)
        tf.translation.y = float(y_r)
        tf.translation.z = float(z_r)
        tw = Twist()
        tw.linear.x = float(0)

        robot_trajectory.transforms.append(tf)
        robot_trajectory.velocities.append(tw)
        robot_pose_list.stamp.append(job.ES)

        # ================ Calculate ================
        C = 2 * PI * R      # Gluing circuit

        x_c = job.a + v_cx * (job.ES - job.t_do)
        y_c = job.b + v_cy * (job.ES - job.t_do)

        # -------------- Phase 1 --------------
        x_s  = x_c + R * np.cos(0)
        y_s  = y_c + R * np.sin(0)

        p_s = np.array([x_s, y_s, z_D])
        p_r = np.array([x_r, y_r, z_r])

        L_lr = np.linalg.norm(p_s - p_r)

        T_a = 0.1
        T_r = T_rr - 2 * T_a
        v_lr = L_lr / (T_a + T_r)
        t_sd = job.ES + T_lr

        tf = Transform()
        tf.translation.x = float(x_s)
        tf.translation.y = float(y_s)
        tf.translation.z = float(z_D)
        tw = Twist()
        tw.linear.x = float(v_lr)

        robot_trajectory.transforms.append(tf)
        robot_trajectory.velocities.append(tw)
        robot_pose_list.stamp.append(float(t_sd))

        # -------------- Phase 2 --------------
        dt = 2 * PI * R / (v_D * n)
        p_prev = p_s
        for i in range(n):
            theta = (i + 1) * 2 * PI / n
            x_i = x_c + R * np.cos(theta) + (i + 1) * dt * v_cx
            y_i = y_c + R * np.sin(theta) + (i + 1) * dt * v_cy
            
            p_i = np.array([x_i, y_i, z_D])
            d_i = np.linalg.norm(p_i - p_prev)
            p_prev = p_i

            v_i = d_i / dt
            t_i = t_sd + (i + 1) * dt

            tf = Transform()
            tf.translation.x = float(x_i)
            tf.translation.y = float(y_i)
            tf.translation.z = float(z_D)
            tw = Twist()
            tw.linear.x = float(v_i)

            robot_trajectory.transforms.append(tf)
            robot_trajectory.velocities.append(tw)
            robot_pose_list.stamp.append(float(t_i))

        # -------------- Phase 3 --------------
        x_e = x_c + R * np.cos(0) + n * dt * v_cx
        y_e = y_c + R * np.sin(0) + n * dt * v_cy
        p_e = np.array([x_e, y_e, z_D])

        L_rr = np.linalg.norm(p_r - p_e)
        v_rr = L_rr / (T_a + T_r)
        t_e = job.ES + T

        tf = Transform()
        tf.translation.x = float(x_r)
        tf.translation.y = float(y_r)
        tf.translation.z = float(z_r)
        tw = Twist()
        tw.linear.x = float(v_rr)

        robot_trajectory.transforms.append(tf)
        robot_trajectory.velocities.append(tw)
        robot_pose_list.stamp.append(float(t_e))
            
        self.trajectory_transfrom_pub.publish(robot_pose_list)
        self.get_logger().info('Published trajectory with %d waypoints' % (n + 3))

        # self.log_trajectory(robot_pose_list)
        self.prev_task_finish_time = t_e


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
    t_in: float = 0.0
    t_out: float = 0.0
    t_do: float = 0.0       #in second
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
