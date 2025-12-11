import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class DemoTrajectory(Node):
    def __init__(self):
        super().__init__('demo_trajectory')
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.timer = self.create_timer(1.0, self.send_cmd)

    def send_cmd(self):
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3']

        p = JointTrajectoryPoint()
        p.positions = [0.10, 0.05, 0.20]
        p.time_from_start.sec = 3

        msg.points.append(p)
        self.pub.publish(msg)
        self.get_logger().info('Sent demo trajectory point.')


def main():
    rclpy.init()
    node = DemoTrajectory()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
