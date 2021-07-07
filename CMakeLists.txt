cmake_minimum_required(VERSION 3.0.0)
project(roma_confident_scount_ros)
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs geometry_msgs message_generation visualization_msgs image_transport)#quadrotor_msgs)
find_package(OpenCV REQUIRED)



## Declare a catkin package
catkin_package(
LIBRARIES
CATKIN_DEPENDS
)


include_directories(include ${catkin_INCLUDE_DIRS})




catkin_install_python(PROGRAMS src/SCOUNT_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
