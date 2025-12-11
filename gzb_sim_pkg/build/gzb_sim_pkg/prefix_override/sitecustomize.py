import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/thinh/ros2_ws/src/gzb_sim_pkg/install/gzb_sim_pkg'
