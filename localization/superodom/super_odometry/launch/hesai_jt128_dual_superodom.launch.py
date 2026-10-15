import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import launch_ros


def get_share_file(package_name, file_name):
    return os.path.join(get_package_share_directory(package_name), file_name)


def generate_launch_description():
    superodom_config_path = get_share_file(
        package_name="super_odometry",
        file_name="config/hesai_jt128_dual_superodom.yaml")
    fusion_config_path = get_share_file(
        package_name="super_odometry",
        file_name="config/hesai_jt128_dual_fusion.yaml")
    calib_path = get_share_file(
        package_name="super_odometry",
        file_name="config/hesai/jt128_calibration.yaml")
    paper_repro_switches_path = get_share_file(
        package_name="super_odometry",
        file_name="config/paper_reproduction_switches.yaml")

    superodom_config_arg = DeclareLaunchArgument(
        "superodom_config_file",
        default_value=superodom_config_path,
        description="Path to dual JT128 SuperOdom config file")
    fusion_config_arg = DeclareLaunchArgument(
        "fusion_config_file",
        default_value=fusion_config_path,
        description="Path to dual JT128 pre-fusion config file")
    calib_path_arg = DeclareLaunchArgument(
        "calibration_file",
        default_value=calib_path,
        description="Path to virtual JT128 sensor-to-IMU calibration file")
    paper_repro_switches_arg = DeclareLaunchArgument(
        "paper_repro_switches_file",
        default_value=paper_repro_switches_path,
        description="Path to SuperLoc paper reproduction switch config file")
    map_dir_arg = DeclareLaunchArgument(
        "map_dir",
        default_value="/path/to/your/prior_map.pcd",
        description="Path to prior map PCD for localization mode")
    body_odom_input_topic_arg = DeclareLaunchArgument(
        "body_odom_input_topic",
        default_value="/laser_odometry",
        description="SuperOdom sensor-frame odometry topic to transform for the planner")
    body_odom_output_topic_arg = DeclareLaunchArgument(
        "body_odom_output_topic",
        default_value="/body_odometry",
        description="Planner-facing body-frame odometry topic")
    body_frame_arg = DeclareLaunchArgument(
        "body_frame",
        default_value="base_link",
        description="Planner-facing robot body frame")
    publish_body_tf_arg = DeclareLaunchArgument(
        "publish_body_tf",
        default_value="true",
        description="Publish world_frame -> body_frame TF from body odometry")

    dual_lidar_fusion_node = Node(
        package="super_odometry",
        executable="dual_lidar_fusion_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[LaunchConfiguration("fusion_config_file")],
    )

    feature_extraction_node = Node(
        package="super_odometry",
        executable="feature_extraction_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[
            LaunchConfiguration("superodom_config_file"),
            {"calibration_file": LaunchConfiguration("calibration_file")},
        ],
    )

    laser_mapping_node = Node(
        package="super_odometry",
        executable="laser_mapping_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[
            LaunchConfiguration("superodom_config_file"),
            LaunchConfiguration("paper_repro_switches_file"),
            {
                "calibration_file": LaunchConfiguration("calibration_file"),
                "map_dir": LaunchConfiguration("map_dir"),
            },
        ],
    )

    imu_preintegration_node = Node(
        package="super_odometry",
        executable="imu_preintegration_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[
            LaunchConfiguration("superodom_config_file"),
            {"calibration_file": LaunchConfiguration("calibration_file")},
        ],
    )

    odom_frame_transform_node = Node(
        package="super_odometry",
        executable="odom_frame_transform_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[
            LaunchConfiguration("superodom_config_file"),
            {
                "input_odom_topic": LaunchConfiguration("body_odom_input_topic"),
                "output_odom_topic": LaunchConfiguration("body_odom_output_topic"),
                "output_child_frame": LaunchConfiguration("body_frame"),
                "publish_tf": ParameterValue(
                    LaunchConfiguration("publish_body_tf"), value_type=bool),
            },
        ],
    )

    return LaunchDescription([
        launch_ros.actions.SetParameter(name="use_sim_time", value=False),
        superodom_config_arg,
        fusion_config_arg,
        calib_path_arg,
        paper_repro_switches_arg,
        map_dir_arg,
        body_odom_input_topic_arg,
        body_odom_output_topic_arg,
        body_frame_arg,
        publish_body_tf_arg,
        dual_lidar_fusion_node,
        feature_extraction_node,
        laser_mapping_node,
        imu_preintegration_node,
        odom_frame_transform_node,
    ])
