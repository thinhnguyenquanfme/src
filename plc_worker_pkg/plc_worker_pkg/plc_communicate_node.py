import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from robot_interfaces.msg import PlcCommand, PlcReadWord, PlcReadBit

import pymcprotocol as pymc

PLC_IP = "192.168.1.70"
PLC_PORT = 1025


class PlcCommunicate(Node):
    def __init__(self):
        self.plc = None
        super().__init__('plc_communicate_node')
        
        # =============== Create Service ===============
        self.connect_plc = self.create_service(SetBool, '/plc/set_connect', self.init_pymc)

        # =============== Create Publisher ===============
        self.plc_read_pub_word = self.create_publisher(PlcReadWord, '/plc/plc_read_word', 10)
        self.plc_read_pub_bit = self.create_publisher(PlcReadWord, '/plc/plc_read_bit', 10)
        

        # =============== Create Subscriber ===============
        self.plc_cmd_sub = self.create_subscription(PlcCommand,'/plc/plc_cmd', self.plc_cmd_cb, 10)

    def plc_cmd_cb(self, msg):
        if self.plc is None:
            self.get_logger().error('PLC do not connect yet!')
            return
        now = self.get_clock().now().to_msg()
        try: 
            match msg.op:
                case 'READ_WORD':
                    register = f"{msg.device}{msg.start_addr}"
                    read_count = msg.read_count
                    wordunits_values = self.plc.batchread_wordunits(headdevice=register, readsize=read_count)

                    plc_read_word = PlcReadWord()
                    plc_read_word.data = wordunits_values
                    plc_read_word.stamp = now
                    self.plc_read_pub_word.publish(plc_read_word)

                case 'WRITE_WORD':
                    register = f"{msg.device}{msg.start_addr}"
                    write_data = msg.write_data
                    self.plc.batchwrite_wordunits(headdevice=register, values=write_data)
                    
                case 'READ_BIT':
                    register = f"{msg.device}{msg.start_addr}"
                    read_count = msg.read_count
                    bitunits_values = self.plc.batchread_bitunits(headdevice=register, readsize=read_count)
                    bool_values = [bool(v) for v in bitunits_values]

                    plc_read_bit = PlcReadBit()
                    plc_read_bit.data = bool_values
                    plc_read_bit.stamp = now
                    self.plc_read_pub_bit.publish(plc_read_bit)

                case 'WRITE_BIT':
                    register = f"{msg.device}{msg.start_addr}"
                    write_data = msg.write_data
                    self.plc.batchwrite_wordunits(headdevice=register, values=write_data)

                case _:
                    self.get_logger().error('No cmd found!')
        except Exception as e:
            self.get_logger().error('PLC command error!')

    def init_pymc(self, req, resp):
        if req.data:
            try:
                self.plc = pymc.Type3E()
                self.plc.connect(PLC_IP, PLC_PORT)

                resp.success = True
                resp.message = 'Connect PLC successfully'
                self.get_logger().info(resp.message)
            
            except Exception as e:
                self.get_logger().error()
                resp.success = True
                resp.message = f'Connect PLC error: {e}'
                self.get_logger().error(resp.message)
                self.plc = None
        else: 
            if self.plc is not None:
                try:
                    self.plc.close()
                except Exception:
                    pass
            self.plc = None
            resp.success = True
            resp.message = 'Disconnected from PLC'
        return resp

class Tag():
    def __init__(self, name, cfg):
        self.name = name
        self.device = cfg['device']
        self.address = cfg['address']
        self.type = cfg['type']
        self.count = cfg['count']




def main(args=None):
    rclpy.init(args=args)
    my_node = PlcCommunicate()

    rclpy.spin(my_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()