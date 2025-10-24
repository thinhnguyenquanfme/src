#!/src/bin/env python3
import rclpy
from rclpy.node import Node
# from std_msgs.msg import Float32
from turtlesim.srv import TeleportAbsolute

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.client = self.create_client(TeleportAbsolute, "/turtle1/teleport_absolute")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for response...")
        self.req = TeleportAbsolute.Request()
        
    def absolute_pos(self):
        req = TeleportAbsolute.Request()
        req.x = float(input("Nhap x: "))
        req.y = float(input("Nhap y: "))
        req.theta = float(input ("Nhap theta: "))

        future = self.client.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f"X: {req.x}, Y: {req.y}, Theta: {req.theta}")

        

def main(args=None):
    rclpy.init(args=args)
    my_node = MinimalService()
    my_node.absolute_pos()

    

    rclpy.shutdown()

if __name__ == '__main__':
    main()
