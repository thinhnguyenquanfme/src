import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from robot_interfaces.msg import PlcReadWord, PlcState
from geometry_msgs.msg import Transform, Vector3, PoseStamped, Twist
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

class PlcMotion(Node):
    def __init__(self):
        super().__init__('plc_motion_node')

def main(args=None):
    rclpy.init(args=args)
    my_node = PlcMotion()
    rclpy.spin(my_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()