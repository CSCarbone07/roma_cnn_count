cmake_minimum_required(VERSION 3.0.0)
project(roma_cnn_count)

#set(Torch_DIR ".local/lib/python3.6/site-packages/torch/share/cmake/Torch")
#set(Torch_DIR "git/pytorch")

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs geometry_msgs message_generation visualization_msgs image_transport roslib)#quadrotor_msgs)
find_package(OpenCV REQUIRED)
find_package(roma_quad_ai)
#find_package(Torch REQUIRED)
#find_package(OpenCV 3 REQUIRED)

## Generate added messages and services
#add_message_files(FILES Utility.msg)

catkin_python_setup()

## Declare a catkin package
catkin_package(
LIBRARIES
CATKIN_DEPENDS

)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
  #${Torch_INSTALL_INCLUDE}
  #${Torch_DIR}
)



#catkin_install_python(PROGRAMS script/SCOUNT_node.py script/dataset/fruit_count_dataset.py script/engines/SCOUNT_Engine.py  script/models/SCOUNT.py 
#catkin_install_python(PROGRAMS src/SCOUNT_node.py src/dataset/fruit_count_dataset.py src/engines/SCOUNT_Engine.py  src/models/SCOUNT.py 
#catkin_install_python(PROGRAMS src/SCOUNT_node.py script/dataset/fruit_count_dataset.py
#catkin_install_python(PROGRAMS scripts/SCOUNT_node.py scripts/dataset/fruit_count_dataset.py
#catkin_install_python(PROGRAMS src/SCOUNT_node.py src/configs.py #scripts/dataset/fruit_count_dataset.py #
catkin_install_python(PROGRAMS bin/SCOUNT_node.py 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS bin/SCOUNT_node_gpu_sim.py 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)



#link_directories(
#  ${Torch_INSTALL_LIB}
#)
