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
        file_name="config/hesai_driver/jt128_driver.yaml")

    config_path_arg = DeclareLaunchArgument(
        "driver_config_file",
        default_value=config_path,
        description="Path to HesaiLidar_ROS_2.0 JT128 driver config file")

    hesai_driver_node = Node(
        package="hesai_ros_driver",
        executable="hesai_ros_driver_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[{"config_path": LaunchConfiguration("driver_config_file")}],
    )

    return LaunchDescription([
        config_path_arg,
        hesai_driver_node,
    ])
