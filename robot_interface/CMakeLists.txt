cmake_minimum_required(VERSION 3.8)
project(unitree_adapter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(ROBOCUP_UNITREE_USE_SDK2 "Build the hardware backend against official unitree_sdk2" OFF)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetBodyHeight.srv"
)

add_library(unitree_adapter_core
  src/command_limiter.cpp
)
target_include_directories(unitree_adapter_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(unitree_adapter_core PUBLIC cxx_std_17)

set(UNITREE_ADAPTER_NODE_SOURCES
  src/unitree_adapter_node.cpp
  src/sport_client_backend.cpp
)

add_executable(unitree_adapter_node
  ${UNITREE_ADAPTER_NODE_SOURCES}
)
target_include_directories(unitree_adapter_node PRIVATE include)
target_link_libraries(unitree_adapter_node unitree_adapter_core)
ament_target_dependencies(unitree_adapter_node
  geometry_msgs
  rclcpp
  std_msgs
  std_srvs
)
target_compile_features(unitree_adapter_node PRIVATE cxx_std_17)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(unitree_adapter_node "${cpp_typesupport_target}")

add_executable(a2_cmd_vel_bridge
  src/a2_cmd_vel_bridge.cpp
)
target_include_directories(a2_cmd_vel_bridge PRIVATE include)
target_link_libraries(a2_cmd_vel_bridge unitree_adapter_core "${cpp_typesupport_target}")
ament_target_dependencies(a2_cmd_vel_bridge
  geometry_msgs
  rclcpp
  std_msgs
  std_srvs
)
target_compile_features(a2_cmd_vel_bridge PRIVATE cxx_std_17)

set(UNITREE_ADAPTER_EXECUTABLES
  unitree_adapter_node
  a2_cmd_vel_bridge
)

if(ROBOCUP_UNITREE_USE_SDK2)
  find_package(unitree_sdk2 REQUIRED)
  target_compile_definitions(unitree_adapter_node PRIVATE ROBOCUP_UNITREE_USE_SDK2=1)
  target_link_libraries(unitree_adapter_node unitree_sdk2)

  add_executable(a2_sdk2_sport_worker
    src/a2_sdk2_sport_worker.cpp
  )
  target_link_libraries(a2_sdk2_sport_worker unitree_sdk2)
  target_compile_features(a2_sdk2_sport_worker PRIVATE cxx_std_17)
  list(APPEND UNITREE_ADAPTER_EXECUTABLES a2_sdk2_sport_worker)

  get_target_property(UNITREE_SDK2_DDSC_LOCATION ddsc IMPORTED_LOCATION)
  if(UNITREE_SDK2_DDSC_LOCATION)
    get_filename_component(UNITREE_SDK2_LIBRARY_DIR "${UNITREE_SDK2_DDSC_LOCATION}" DIRECTORY)
    set_target_properties(unitree_adapter_node PROPERTIES
      BUILD_RPATH "${UNITREE_SDK2_LIBRARY_DIR}"
      INSTALL_RPATH "${UNITREE_SDK2_LIBRARY_DIR}"
    )
    set_target_properties(a2_sdk2_sport_worker PROPERTIES
      BUILD_RPATH "${UNITREE_SDK2_LIBRARY_DIR}"
      INSTALL_RPATH "${UNITREE_SDK2_LIBRARY_DIR}"
    )
  endif()
else()
  target_compile_definitions(unitree_adapter_node PRIVATE ROBOCUP_UNITREE_USE_SDK2=0)
endif()

install(TARGETS unitree_adapter_core
  EXPORT export_unitree_adapter_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS ${UNITREE_ADAPTER_EXECUTABLES}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_targets(export_unitree_adapter_core HAS_LIBRARY_TARGET)
ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  ament_add_gtest(test_command_limiter
    test/test_command_limiter.cpp
  )
  target_link_libraries(test_command_limiter unitree_adapter_core)
  target_include_directories(test_command_limiter PRIVATE include)
  target_compile_features(test_command_limiter PRIVATE cxx_std_17)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
