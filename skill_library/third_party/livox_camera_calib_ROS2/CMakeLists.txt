cmake_minimum_required(VERSION 3.8)
project(livox_camera_calib)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Ceres REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Create executables
add_executable(lidar_camera_calib src/lidar_camera_calib.cpp include/lidar_camera_calib.hpp)
add_executable(lidar_camera_multi_calib src/lidar_camera_multi_calib.cpp include/lidar_camera_calib.hpp)
add_executable(bag_to_pcd src/bag_to_pcd.cpp)

# Dependencies for lidar_camera_calib
ament_target_dependencies(lidar_camera_calib
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
  livox_ros_driver2
  rosbag2_cpp
  rosbag2_storage
)

# Dependencies for lidar_camera_multi_calib
ament_target_dependencies(lidar_camera_multi_calib
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
  livox_ros_driver2
  rosbag2_cpp
  rosbag2_storage
)

# Dependencies for bag_to_pcd
ament_target_dependencies(bag_to_pcd
  rclcpp
  sensor_msgs
  pcl_conversions
  livox_ros_driver2
  rosbag2_cpp
  rosbag2_storage
)

# Link libraries
target_link_libraries(lidar_camera_calib 
  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
)

target_link_libraries(lidar_camera_multi_calib 
  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
)

target_link_libraries(bag_to_pcd 
  ${PCL_LIBRARIES}
)

# Install executables
install(TARGETS
  lidar_camera_calib
  lidar_camera_multi_calib
  bag_to_pcd
  DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY
  launch
  config
  rviz_cfg
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()