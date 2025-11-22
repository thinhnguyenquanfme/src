import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from robot_interfaces.msg import PlcReadWord, PlcState

class PlcMonitor(Node):
    def __init__(self):
        super().__init__('plc_monitor_node')

        # =============== Create Subscriber ===============
        self.read_word_sub = self.create_subscription(PlcReadWord, '/plc/plc_read_word', self.read_word_cb,10)

        # =============== Create Publisher ===============
        self.plc_monitor_pub = self.create_publisher(PlcState, 'plc/plc_state', 10)

        self.pos_scale = 0.001   # pulse -> mm
        self.vel_scale = 0.001   # pulse/s -> mm/s

    # ------------ Helper ------------
    def words_to_int32(self, lo16, hi16):
        lo = lo16      # đã là uint16
        hi = hi16      # đã là uint16
        raw = (hi << 16) | lo
        if raw & 0x80000000:
            raw -= 0x100000000
        return raw


    def read_word_cb(self, msg: PlcReadWord):
        data = msg.data
        if len(data) < 16:
            self.get_logger().warn(f"Not enough data, expected >=16 words, got {len(data)}")
            return

        # decode 32-bit
        x_pos_raw = self.words_to_int32(data[0], data[1])
        y_pos_raw = self.words_to_int32(data[2], data[3])
        z_pos_raw = self.words_to_int32(data[4], data[5])
        x_vel_raw = self.words_to_int32(data[6], data[7])
        y_vel_raw = self.words_to_int32(data[8], data[9])
        z_vel_raw = self.words_to_int32(data[10], data[11])
        

        # scale sang float64
        x_pos = x_pos_raw * self.pos_scale
        y_pos = y_pos_raw * self.pos_scale
        z_pos = z_pos_raw * self.pos_scale

        x_vel = x_vel_raw * self.vel_scale
        y_vel = y_vel_raw * self.vel_scale
        z_vel = z_vel_raw * self.vel_scale

        out = PlcMonitor()
        out.stamp = msg.stamp  # dùng luôn stamp từ node PLC

        out.axis_x_pos_fb = x_pos
        out.axis_y_pos_fb = y_pos
        out.axis_z_pos_fb = z_pos

        out.axis_x_vel_fb = x_vel
        out.axis_y_vel_fb = y_vel
        out.axis_z_vel_fb = z_vel

        self.plc_monitor_pub.publish(out)
        

def main(args=None):
    rclpy.init(args=args)
    my_node = PlcMonitor()
    rclpy.spin(my_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()