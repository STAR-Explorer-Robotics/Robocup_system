cmake_minimum_required(VERSION 3.8)
project(trg_path_follower)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(rpp_follower_node
  src/main.cpp
  src/rpp_follower_node.cpp
)
target_include_directories(rpp_follower_node PRIVATE include)
ament_target_dependencies(rpp_follower_node
  geometry_msgs
  nav_msgs
  rclcpp
  std_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
)
target_compile_features(rpp_follower_node PRIVATE cxx_std_17)

add_executable(rpp_offline_sim_node
  src/rpp_offline_sim_node.cpp
)
target_include_directories(rpp_offline_sim_node PRIVATE include)
ament_target_dependencies(rpp_offline_sim_node
  geometry_msgs
  nav_msgs
  rclcpp
  std_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
)
target_compile_features(rpp_offline_sim_node PRIVATE cxx_std_17)

install(TARGETS rpp_follower_node rpp_offline_sim_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_rpp_geometry
    test/test_rpp_geometry.cpp
  )
  target_include_directories(test_rpp_geometry PRIVATE include)
  target_compile_features(test_rpp_geometry PRIVATE cxx_std_17)
endif()

ament_package()
