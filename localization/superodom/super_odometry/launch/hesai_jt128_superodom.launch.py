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
    config_path = get_share_file(
        package_name="super_odometry",
        file_name="config/hesai_jt128.yaml")
    calib_path = get_share_file(
        package_name="super_odometry",
        file_name="config/hesai/jt128_calibration.yaml")
    paper_repro_switches_path = get_share_file(
        package_name="super_odometry",
        file_name="config/paper_reproduction_switches.yaml")

    config_path_arg = DeclareLaunchArgument(
        "config_file",
        default_value=config_path,
        description="Path to JT128 SuperOdom config file")
    calib_path_arg = DeclareLaunchArgument(
        "calibration_file",
        default_value=calib_path,
        description="Path to JT128 calibration file")
    paper_repro_switches_arg = DeclareLaunchArgument(
        "paper_repro_switches_file",
        default_value=paper_repro_switches_path,
        description="Path to SuperLoc paper reproduction switch config file")
    map_dir_arg = DeclareLaunchArgument(
        "map_dir",
        default_value="/path/to/your/prior_map.pcd",
        description="Path to prior map PCD for localization mode")
    input_point_cloud_topic_arg = DeclareLaunchArgument(
        "input_point_cloud_topic",
        default_value="/hesai/jt128/points",
        description="Input JT128 PointCloud2 topic. For Unitree A2 fused mode, set this to the robot fused point cloud topic")
    output_point_cloud_topic_arg = DeclareLaunchArgument(
        "output_point_cloud_topic",
        default_value="/hesai/jt128/points_superodom",
        description="SuperOdom-compatible PointCloud2 topic published by hesai_to_superodom_node")
    imu_topic_arg = DeclareLaunchArgument(
        "imu_topic",
        default_value="/hesai/jt128/imu",
        description="Primary IMU topic used by SuperOdom")
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

    hesai_adapter_node = Node(
        package="super_odometry",
        executable="hesai_to_superodom_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[
            LaunchConfiguration("config_file"),
            {
                "input_point_cloud_topic": LaunchConfiguration("input_point_cloud_topic"),
                "output_point_cloud_topic": LaunchConfiguration("output_point_cloud_topic"),
            },
        ],
    )

    feature_extraction_node = Node(
        package="super_odometry",
        executable="feature_extraction_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[
            LaunchConfiguration("config_file"),
            {
                "calibration_file": LaunchConfiguration("calibration_file"),
                "laser_topic": LaunchConfiguration("output_point_cloud_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
            },
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
            LaunchConfiguration("config_file"),
            LaunchConfiguration("paper_repro_switches_file"),
            {
                "calibration_file": LaunchConfiguration("calibration_file"),
                "laser_topic": LaunchConfiguration("output_point_cloud_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
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
            LaunchConfiguration("config_file"),
            {
                "calibration_file": LaunchConfiguration("calibration_file"),
                "laser_topic": LaunchConfiguration("output_point_cloud_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
            },
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
            LaunchConfiguration("config_file"),
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
        config_path_arg,
        calib_path_arg,
        paper_repro_switches_arg,
        map_dir_arg,
        input_point_cloud_topic_arg,
        output_point_cloud_topic_arg,
        imu_topic_arg,
        body_odom_input_topic_arg,
        body_odom_output_topic_arg,
        body_frame_arg,
        publish_body_tf_arg,
        hesai_adapter_node,
        feature_extraction_node,
        laser_mapping_node,
        imu_preintegration_node,
        odom_frame_transform_node,
    ])
