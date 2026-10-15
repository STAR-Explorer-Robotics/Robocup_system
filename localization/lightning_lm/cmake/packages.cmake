find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
if (ROBOCUP_LIGHTNING_WITH_PANGOLIN)
    find_package(Pangolin REQUIRED)
    find_package(OpenGL REQUIRED)
endif ()
find_package(pcl_conversions REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
if (ROBOCUP_LIGHTNING_BUILD_OFFLINE_TOOLS)
    find_package(rosbag2_cpp REQUIRED)
endif ()
find_package(rosidl_default_generators REQUIRED)

if (TARGET yaml-cpp::yaml-cpp)
    set(LIGHTNING_YAML_CPP_LIB yaml-cpp::yaml-cpp)
elseif (TARGET yaml-cpp)
    set(LIGHTNING_YAML_CPP_LIB yaml-cpp)
else ()
    set(LIGHTNING_YAML_CPP_LIB ${YAML_CPP_LIBRARIES} ${yaml-cpp_LIBRARIES})
endif ()

# OMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

if (BUILD_WITH_MARCH_NATIVE)
    add_compile_options(-march=native)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
    add_compile_options(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
else ()
    message(STATUS "Skipping x86 SSE flags for ${CMAKE_SYSTEM_PROCESSOR}")
endif ()

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${tf2_INCLUDE_DIRS}
        ${pcl_conversions_INCLUDR_DIRS}
        ${rclcpp_INCLUDE_DIRS}
        ${nav_msgs_INCLUDE_DIRS}
)

if (ROBOCUP_LIGHTNING_WITH_PANGOLIN)
    include_directories(
            ${Pangolin_INCLUDE_DIRS}
            ${GLEW_INCLUDE_DIRS}
    )
    add_compile_definitions(ROBOCUP_LIGHTNING_WITH_PANGOLIN=1)
else ()
    add_compile_definitions(ROBOCUP_LIGHTNING_WITH_PANGOLIN=0)
endif ()

if (ROBOCUP_LIGHTNING_BUILD_OFFLINE_TOOLS)
    include_directories(
            ${rosbag2_cpp_INCLUDE_DIRS}
    )
    add_compile_definitions(ROBOCUP_LIGHTNING_BUILD_OFFLINE_TOOLS=1)
else ()
    add_compile_definitions(ROBOCUP_LIGHTNING_BUILD_OFFLINE_TOOLS=0)
endif ()

include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/livox_ros_driver/rosidl_generator_cpp
)

include_directories(
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/thirdparty
)


set(third_party_libs
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        glog gflags
        ${LIGHTNING_YAML_CPP_LIB}
        ${pcl_conversions_LIBRARIES}
        tbb
)

if (ROBOCUP_LIGHTNING_WITH_PANGOLIN)
    list(APPEND third_party_libs
            ${Pangolin_LIBRARIES}
    )
endif ()

if (ROBOCUP_LIGHTNING_BUILD_OFFLINE_TOOLS)
    list(APPEND third_party_libs
            ${rosbag2_cpp_LIBRARIES}
    )
endif ()
