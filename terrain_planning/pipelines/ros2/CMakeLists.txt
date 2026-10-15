cmake_minimum_required(VERSION 3.16...3.24)
cmake_policy(SET CMP0074 NEW)
project(trg_planner_ros VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "Configuring pipelines/ros2")

get_filename_component(TRG_ROS_DIR ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
add_definitions(-DTRG_ROS_DIR="${TRG_ROS_DIR}")
message(STATUS "TRG_ROS_DIR: ${TRG_ROS_DIR}")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(trg_planner REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
)

add_executable(trg_ros2_node
    ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ros2_node.cpp
)
ament_target_dependencies(trg_ros2_node
    rclcpp
    geometry_msgs
    sensor_msgs
    std_msgs
    pcl_conversions
    nav_msgs
    tf2
    tf2_ros
    visualization_msgs
    trg_planner
)

target_include_directories(trg_ros2_node
    PUBLIC
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(trg_ros2_node
    trg_planner::trg_planner_core
    ${PCL_LIBRARIES}
)
install(TARGETS
    trg_ros2_node
    DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
    ${CMAKE_CURRENT_LIST_DIR}/scripts/fake_pose_pub.py
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    launch
    config
    rviz
    DESTINATION share/${PROJECT_NAME}
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
)

install(FILES
    ${CMAKE_CURRENT_LIST_DIR}/../../config/robocup_default.yaml
    DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
