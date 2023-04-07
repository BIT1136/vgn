cmake_minimum_required(VERSION 3.0.2)
project(vgn)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  GraspCandidate.msg
)

add_service_files(
  FILES
  PredictGrasps.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
