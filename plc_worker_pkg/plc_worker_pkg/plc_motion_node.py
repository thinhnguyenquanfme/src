import rclpy
from rclpy.node import Node

from std_srvs.srv import SetBool
from robot_interfaces.msg import PlcReadWord, PlcState, PoseListMsg, PlcCommand
from geometry_msgs.msg import Transform, Vector3, PoseStamped, Twist
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

from dataclasses import dataclass, field
from enum import IntEnum
from rclpy.time import Time

MAX_POINTS = 50
YZ_WORDS_PER_POINT = 2
X_WORDS_PER_POINT = 10

X_OFFSET = 0
Y_OFFSET = X_OFFSET + (MAX_POINTS * X_WORDS_PER_POINT)
Z_OFFSET = Y_OFFSET + (MAX_POINTS * YZ_WORDS_PER_POINT)
CONV_OFFSET = Z_OFFSET + (MAX_POINTS * YZ_WORDS_PER_POINT)
TOTAL_WORDS = CONV_OFFSET + 2

HANDSHAKE = 'M100'
START_TRAJ_ADDR = 'D1000'

@dataclass
class TrajData():
    x_list: list[float] = field(default_factory=list)
    y_list: list[float] = field(default_factory=list)
    z_list: list[float] = field(default_factory=list)
    vel_list: list[float] = field(default_factory=list)
    time_list: list[Time] = field(default_factory=list)
    conv_pose: float = 0.0

# ================ Helper ================
def int32_to_words(v: int) -> tuple[int, int]:
    # two's complement -> uint32
    u = v & 0xFFFFFFFF
    lo = u & 0xFFFF
    hi = (u >> 16) & 0xFFFF
    return lo, hi

ADDRESS_OFFSET = {
    "pos_id": 0,  # WORD: pack (Da2, Da4, Da3, Da5, Da1)
    "mcode": 1,  # WORD: nghia tuy bien (VD: M1=OFF, M2=ON cho glue)
    "dwell_ms": 2,  # WORD: thoi gian dwell ms (one-shot)
    "pos_opt": 3,  # WORD: truong du phong
    "speed_lo": 4,  # DWORD lo: lenh speed
    "speed_hi": 5,  # DWORD hi (implicit)
    "pos_lo": 6,  # DWORD lo: vi tri dich
    "pos_hi": 7,  # DWORD hi (implicit)
    "arc_lo": 8,  # DWORD lo: tam cung (X/Y) tuyet doi
    "arc_hi": 9,  # DWORD hi (implicit)
}

ROW_WORDS_PER_AXIS = 10

# ================ Positioning Identifier class ================
class OprPattern(IntEnum):
    POS_CPLT = 0b00
    CONT_POS = 0b01
    CONT_PATH = 0b11
class ControlSys(IntEnum):
    ABS3 = 0x15
    VF1 = 0x04
    VR1 = 0x05
class AccTime(IntEnum):
    SLOW_ACC = 0b00
    FAST_ACC = 0b01
class DecTime(IntEnum):
    SLOW_DEC = 0b00
    FAST_DEC = 0b01
class AxisInter(IntEnum):
    AX1 = 0b00

class PosId():
    def __init__(
        self,
        control_sys: ControlSys = ControlSys.ABS3,
        dec_time: DecTime = DecTime.SLOW_DEC,
        acc_time: AccTime = AccTime.SLOW_ACC,
        axis_inter: AxisInter = AxisInter.AX1,
        opr_pattern: OprPattern = OprPattern.POS_CPLT,
    ):
        self.control_sys = control_sys
        self.dec_time = dec_time
        self.acc_time = acc_time
        self.axis_inter = axis_inter
        self.opr_pattern = opr_pattern

    def to_word(self) -> int:
        control_sys = int(self.control_sys) & 0xFF
        dec_time = int(self.dec_time) & 0x03
        acc_time = int(self.acc_time) & 0x03
        axis_inter = int(self.axis_inter) & 0x03
        opr_pattern = int(self.opr_pattern) & 0x03
        return (
            (control_sys << 8)
            | (dec_time << 6)
            | (acc_time << 4)
            | (axis_inter << 2)
            | opr_pattern
        )

class Mcode(IntEnum): # Int16
    GLUE = 2
    NOGLUE = 1

class Dwell(IntEnum): # Int16
    DEFAULT = 0

class PosOpt(IntEnum): # Int16
    DEFAULT = 0

class ArcAddr(IntEnum): # Int32
    DEFAULT = 0

# ========================== END ========================


class PlcMotion(Node):
    def __init__(self):
        super().__init__('plc_motion_node')

        # --------------- Subscriber ---------------
        self.traj_sub = self.create_subscription(PoseListMsg, '/geometry/trajectory_data', self.traj_sub_cb, 10)
        self.traj_data = TrajData()
        self.plc_cmd_pub = self.create_publisher(PlcCommand, '/plc/plc_cmd', 10)

    def traj_sub_cb(self, msg):
        traj = msg.trajectory
        self.traj_data.x_list.clear()
        self.traj_data.y_list.clear()
        self.traj_data.z_list.clear()
        self.traj_data.vel_list.clear()
        self.traj_data.time_list.clear()

        for i, tf in enumerate(traj.transforms):
            self.traj_data.x_list.append(float(tf.translation.x))
            self.traj_data.y_list.append(float(tf.translation.y))
            self.traj_data.z_list.append(float(tf.translation.z))

            if i < len(traj.velocities):
                self.traj_data.vel_list.append(float(traj.velocities[i].linear.x))
            else:
                self.traj_data.vel_list.append(0.0)

            if i < len(msg.stamp):
                t = float(msg.stamp[i])
                sec = int(t)
                nanosec = int((t - sec) * 1e9)
                self.traj_data.time_list.append(Time(seconds=sec, nanoseconds=nanosec))

        self.traj_data.conv_pose = float(msg.conv_pose)

        buf_all = self.convert_motion_data()
        self.send_traj_to_plc(buf_all)

    def send_traj_to_plc(self, buf_all: list[int]) -> None:
        cmd_handshake = PlcCommand()
        cmd_handshake.op = 'WRITE_BIT'
        cmd_handshake.device = HANDSHAKE[0]
        cmd_handshake.start_addr = HANDSHAKE[1:]
        cmd_handshake.read_count = 0
        cmd_handshake.write_data = [1]
        self.plc_cmd_pub.publish(cmd_handshake)

        cmd_traj = PlcCommand()
        cmd_traj.op = 'WRITE_WORD'
        cmd_traj.device = START_TRAJ_ADDR[0]
        cmd_traj.start_addr = START_TRAJ_ADDR[1:]
        cmd_traj.read_count = 0
        cmd_traj.write_data = buf_all
        self.plc_cmd_pub.publish(cmd_traj)

    def build_posid_for_index(self, i: int, n: int) -> PosId:
        if i == 0:
            return PosId(opr_pattern=OprPattern.CONT_POS,
                        acc_time=AccTime.SLOW_ACC,
                        dec_time=DecTime.SLOW_DEC)
        if i == n - 1:
            return PosId(opr_pattern=OprPattern.POS_CPLT,
                        acc_time=AccTime.SLOW_ACC,
                        dec_time=DecTime.SLOW_DEC)
        return PosId(opr_pattern=OprPattern.CONT_PATH,
                    acc_time=AccTime.FAST_ACC,
                    dec_time=DecTime.FAST_DEC)
    

    def build_mcode_for_index(self, i: int, n: int) -> int:
        if i == 0 or i == n - 1:
            return int(Mcode.NOGLUE)
        return int(Mcode.GLUE)
    
    def convert_motion_data(self):
        buf_all = [0] * TOTAL_WORDS

        # ---- scale config (placeholder) ----
        scale_x = 10000.0
        scale_y = 10000.0
        scale_z = 10000.0
        scale_v = 6000.0
        scale_conv = 10000.0

        n = min(len(self.traj_data.x_list), MAX_POINTS)

        # 1) init buffers = zeros
        buf_x = [0] * (MAX_POINTS * X_WORDS_PER_POINT)
        buf_y = [0] * (MAX_POINTS * YZ_WORDS_PER_POINT)
        buf_z = [0] * (MAX_POINTS * YZ_WORDS_PER_POINT)
        buf_conv = [0] * 2

        # 2) pack trajectory points
        for i in range(n):
            x_mm = self.traj_data.x_list[i]
            y_mm = self.traj_data.y_list[i]
            z_mm = self.traj_data.z_list[i]

            # hiện bạn chỉ lấy vel linear.x -> vx
            vx_mm_s = self.traj_data.vel_list[i] if i < len(self.traj_data.vel_list) else 0.0

            # convert to counts (int32)
            x_cnt = int(round(x_mm * scale_x))
            y_cnt = int(round(y_mm * scale_y))
            z_cnt = int(round(z_mm * scale_z))
            v_cnt = int(round(vx_mm_s * scale_v))

            # --- X record 10 words ---
            posid = self.build_posid_for_index(i, n)
            mcode =  self.build_mcode_for_index(i, n)
            dwell = int(Dwell.DEFAULT)
            posopt = int(PosOpt.DEFAULT)
            arc = int(ArcAddr.DEFAULT)

            v_lo, v_hi = int32_to_words(v_cnt)
            x_lo, x_hi = int32_to_words(x_cnt)
            arc_lo, arc_hi = int32_to_words(arc)

            base = i * X_WORDS_PER_POINT
            buf_x[base + ADDRESS_OFFSET["pos_id"]] = posid.to_word() & 0xFFFF
            buf_x[base + ADDRESS_OFFSET["mcode"]] = mcode & 0xFFFF
            buf_x[base + ADDRESS_OFFSET["dwell_ms"]] = dwell & 0xFFFF
            buf_x[base + ADDRESS_OFFSET["pos_opt"]] = posopt & 0xFFFF
            buf_x[base + ADDRESS_OFFSET["speed_lo"]] = v_lo
            buf_x[base + ADDRESS_OFFSET["speed_hi"]] = v_hi
            buf_x[base + ADDRESS_OFFSET["pos_lo"]] = x_lo
            buf_x[base + ADDRESS_OFFSET["pos_hi"]] = x_hi
            buf_x[base + ADDRESS_OFFSET["arc_lo"]] = arc_lo
            buf_x[base + ADDRESS_OFFSET["arc_hi"]] = arc_hi

            # --- Y record: speed + pos (2 words) ---
            y_lo, y_hi = int32_to_words(y_cnt)

            base_y = i * YZ_WORDS_PER_POINT
            buf_y[base_y + 0] = y_lo
            buf_y[base_y + 1] = y_hi

            # --- Z record: speed + pos (2 words) ---
            z_lo, z_hi = int32_to_words(z_cnt)

            base_z = i * YZ_WORDS_PER_POINT
            buf_z[base_z + 0] = z_lo
            buf_z[base_z + 1] = z_hi

        # 3) conv_pose dword
        conv_cnt = int(round(float(self.traj_data.conv_pose) * scale_conv))
        conv_lo, conv_hi = int32_to_words(conv_cnt)
        buf_conv[0] = conv_lo
        buf_conv[1] = conv_hi

        buf_all = buf_x + buf_y + buf_z + buf_conv
        return buf_all


def main(args=None):
    rclpy.init(args=args)
    my_node = PlcMotion()
    rclpy.spin(my_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
