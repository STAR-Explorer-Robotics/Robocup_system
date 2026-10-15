import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
import launch_ros

def get_share_file(package_name, file_name):
    return os.path.join(get_package_share_directory(package_name), file_name)

def generate_launch_description():
    config_path = get_share_file(
        package_name="super_odometry",
        file_name="config/vlp_16.yaml")
    calib_path = get_share_file(
        package_name="super_odometry",
        file_name="config/velodyne/vlp_16_calibration.yaml"
    )
    paper_repro_switches_path = get_share_file(
        package_name="super_odometry",
        file_name="config/paper_reproduction_switches.yaml"
    )
    home_directory = os.path.expanduser("~")
    
    config_path_arg = DeclareLaunchArgument(
        "config_file",
        default_value=config_path,
        description="Path to config file for super_odometry"
    )
    calib_path_arg = DeclareLaunchArgument(
        "calibration_file",
        default_value=calib_path,
    )
    paper_repro_switches_arg = DeclareLaunchArgument(
        "paper_repro_switches_file",
        default_value=paper_repro_switches_path,
        description="Path to paper reproduction switch config file for laser_mapping_node"
    )
    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="integrated_to_init"
    )
    world_frame_arg = DeclareLaunchArgument(
        "world_frame",
        default_value="map",
    )
    world_frame_rot_arg = DeclareLaunchArgument(
        "world_frame_rot",
        default_value="map_rot",
    )
    sensor_frame_arg = DeclareLaunchArgument(
        "sensor_frame",
        default_value="sensor",
    )
    sensor_frame_rot_arg = DeclareLaunchArgument(
        "sensor_frame_rot",
        default_value="sensor_rot",
    )
    map_dir_arg = DeclareLaunchArgument(
        "map_dir",
        default_value=os.path.join(home_directory, "/path/to/your/pcd"),
        description="Path to prior map PCD for localization mode"
    )

    feature_extraction_node = Node(
        package="super_odometry",
        executable="feature_extraction_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[LaunchConfiguration("config_file"),
            { "calibration_file": LaunchConfiguration("calibration_file"),
        }],
    )

    laser_mapping_node = Node(
        package="super_odometry",
        executable="laser_mapping_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[LaunchConfiguration("config_file"),
            LaunchConfiguration("paper_repro_switches_file"),
            { "calibration_file": LaunchConfiguration("calibration_file"),
             "map_dir": LaunchConfiguration("map_dir"),
        }],
        remappings=[
            ("laser_odom_to_init", LaunchConfiguration("odom_topic")),
        ]
    )

    imu_preintegration_node = Node(
        package="super_odometry",
        executable="imu_preintegration_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[LaunchConfiguration("config_file"),
            { "calibration_file": LaunchConfiguration("calibration_file")
        }],
    )

    
    return LaunchDescription([
        launch_ros.actions.SetParameter(name='use_sim_time', value='false'),
        config_path_arg,
        calib_path_arg,
        paper_repro_switches_arg,
        odom_topic_arg,
        world_frame_arg,
        world_frame_rot_arg,
        sensor_frame_arg,
        sensor_frame_rot_arg,
        map_dir_arg,
        feature_extraction_node,
        laser_mapping_node,
        imu_preintegration_node,
    ])
