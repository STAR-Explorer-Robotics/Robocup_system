import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def get_share_file(package_name, file_name):
    return os.path.join(get_package_share_directory(package_name), file_name)


def generate_launch_description():
    config_path = get_share_file(
        package_name="super_odometry",
        file_name="config/dual_lidar_fusion.yaml")

    config_path_arg = DeclareLaunchArgument(
        "config_file",
        default_value=config_path,
        description="Path to dual LiDAR fusion config file")

    dual_lidar_fusion_node = Node(
        package="super_odometry",
        executable="dual_lidar_fusion_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[LaunchConfiguration("config_file")],
    )

    return LaunchDescription([
        config_path_arg,
        dual_lidar_fusion_node,
    ])
