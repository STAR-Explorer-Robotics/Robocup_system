from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("trg_path_follower"), "config", "rpp_follower.yaml"]
                ),
                description="Regulated pure pursuit follower parameter file.",
            ),
            Node(
                package="trg_path_follower",
                executable="rpp_follower_node",
                name="rpp_follower_node",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
