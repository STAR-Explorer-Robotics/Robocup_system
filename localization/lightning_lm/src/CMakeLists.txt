set(LIGHTNING_LIB_SOURCES
        common/nav_state.cc
        common/constant.h
        common/options.cc
        common/params.cc

        io/yaml_io.cc
        io/file_io.cc

        core/lio/eskf.cc
        core/lio/laser_mapping.cc
        core/lio/pointcloud_preprocess.cc
        core/loop_closing/loop_closing.cc

        core/g2p5/g2p5_map.cc
        core/g2p5/g2p5.cc
        core/g2p5/g2p5_subgrid.cc

        core/maps/tiled_map.cc
        core/maps/tiled_map_chunk.cc

        core/localization/localization_result.cc
        core/localization/lidar_loc/lidar_loc.cc
        core/localization/lidar_loc/pclomp/ndt_omp.cpp
        core/localization/lidar_loc/pclomp/voxel_grid_covariance_omp.cpp

        core/localization/pose_graph/pgo.cc
        core/localization/pose_graph/pgo_impl.cc
        core/localization/pose_graph/pose_extrapolator.cc
        core/localization/localization.cpp

        utils/timer.cc
        utils/pointcloud_utils.cc

        ui/pangolin_window.cc

        core/system/slam.cc
        core/system/loc_system.cc
)

if (ROBOCUP_LIGHTNING_WITH_PANGOLIN)
    list(APPEND LIGHTNING_LIB_SOURCES
            ui/pangolin_window_impl.cc
            ui/ui_car.cc
            ui/ui_cloud.cc
            ui/ui_trajectory.cc
    )
endif ()

if (ROBOCUP_LIGHTNING_BUILD_OFFLINE_TOOLS)
    list(APPEND LIGHTNING_LIB_SOURCES
            wrapper/bag_io.cc
    )
endif ()

add_library(${PROJECT_NAME}.libs SHARED
        ${LIGHTNING_LIB_SOURCES}
)

if (ROBOCUP_LIGHTNING_LOW_MEMORY_BUILD)
    message(STATUS "Using low-memory compile options for Lightning large translation units")
    set_source_files_properties(core/lio/laser_mapping.cc PROPERTIES COMPILE_OPTIONS "-O1")
endif ()

ament_target_dependencies(
        ${PROJECT_NAME}.libs
        "rclcpp"
        "std_msgs"
        "geometry_msgs"
        "sensor_msgs"
        "nav_msgs"
        "std_srvs"
        "message_filters"
        "pcl_conversions"
        "tf2_ros"
        "tf2"
)

add_dependencies(${PROJECT_NAME}.libs
        livox_interfaces2__rosidl_typesupport_cpp
        ${PROJECT_NAME}__rosidl_typesupport_cpp
)

include_directories(${PROJECT_SOURCE_DIR}/src/core/miao)
add_subdirectory(core/miao)

target_link_libraries(${PROJECT_NAME}.libs
        ${third_party_libs}
        ${PCL_LIBRARIES}
        miao.core
        miao.utils
        livox_interfaces2__rosidl_typesupport_cpp
        ${PROJECT_NAME}__rosidl_typesupport_cpp
)

install(TARGETS ${PROJECT_NAME}.libs miao.core miao.utils
        DESTINATION lib)

add_subdirectory(app)
