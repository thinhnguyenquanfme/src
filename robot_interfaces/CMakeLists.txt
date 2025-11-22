cmake_minimum_required(VERSION 3.8)
project(robot_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PoseListSrv.srv"
  "msg/PoseListMsg.msg"
  "msg/PlcValue.msg"
  "msg/PlcState.msg"
  "msg/PlcCommand.msg"
  "msg/PlcReadWord.msg"
  "msg/PlcReadBit.msg"
  DEPENDENCIES geometry_msgs trajectory_msgs builtin_interfaces
)
 
ament_export_dependencies(rosidl_default_runtime)

ament_package()
