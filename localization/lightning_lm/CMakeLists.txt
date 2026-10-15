cmake_minimum_required(VERSION 3.16)

project(lightning)

option(ROBOCUP_LIGHTNING_WITH_PANGOLIN "Build Pangolin-based visualization UI" ON)
option(ROBOCUP_LIGHTNING_BUILD_OFFLINE_TOOLS "Build rosbag/offline utility executables" ON)
option(ROBOCUP_LIGHTNING_WITH_DEBUG_SYMBOLS "Build Lightning with debug symbols" ON)
option(ROBOCUP_LIGHTNING_LOW_MEMORY_BUILD "Use lower-memory compile options for large translation units" OFF)

# Default to C++17
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

# set to release by default
set(CMAKE_BUILD_TYPE Release)

# -------------------------------------------------------------
# 导入第三方库，同时做一些全局编译选项配置
add_definitions("-DPCL_NO_PRECOMPILE")

set(LIGHTNING_BASE_DEBUG_FLAGS "")
set(LIGHTNING_RELEASE_DEBUG_FLAGS "")
if (ROBOCUP_LIGHTNING_WITH_DEBUG_SYMBOLS)
    set(LIGHTNING_BASE_DEBUG_FLAGS "-g")
    set(LIGHTNING_RELEASE_DEBUG_FLAGS "-g -ggdb")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fopenmp -fPIC ${LIGHTNING_BASE_DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -fopenmp -fPIC ${LIGHTNING_RELEASE_DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -ggdb -g -fopenmp -fPIC -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC ${LIGHTNING_BASE_DEBUG_FLAGS}")


if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/libs)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/libs)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/packages.cmake)

# msg 和 srv
set(MY_SRV_FILES "")
set(SRC_DIR ${PROJECT_SOURCE_DIR}/srv)
file(GLOB_RECURSE SRC_FILES ${SRC_DIR}/*.srv)
foreach (SRC_FILE ${SRC_FILES})
    file(RELATIVE_PATH REL_PATH ${SRC_DIR} ${SRC_FILE})
    list(APPEND MY_SRV_FILES ${PROJECT_SOURCE_DIR}:srv/${REL_PATH})
endforeach ()

rosidl_generate_interfaces(${PROJECT_NAME}
        ${MY_SRV_FILES}
        ${MY_MSG_FILES}
        ${MY_ACTION_FILES}
        DEPENDENCIES std_msgs geometry_msgs sensor_msgs nav_msgs
)

include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
)

install(DIRECTORY config
        DESTINATION lib/${PROJECT_NAME}/
)

add_subdirectory(thirdparty/livox_ros_driver)

add_subdirectory(src)

ament_package()
