from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    package_share = get_package_share_directory("trg_path_follower")
    default_follower_params = os.path.join(package_share, "config", "rpp_follower.yaml")
    default_sim_params = os.path.join(package_share, "config", "offline_sim.yaml")

    follower_params_file = LaunchConfiguration("follower_params_file")
    sim_params_file = LaunchConfiguration("sim_params_file")

    follower_node = Node(
        package="trg_path_follower",
        executable="rpp_follower_node",
        name="rpp_follower_node",
        output="screen",
        parameters=[
            follower_params_file,
            {"control.enabled_on_start": False},
        ],
    )

    sim_node = Node(
        package="trg_path_follower",
        executable="rpp_offline_sim_node",
        name="rpp_offline_sim_node",
        output="screen",
        parameters=[sim_params_file],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "follower_params_file",
            default_value=default_follower_params,
            description="Parameter file for rpp_follower_node.",
        ),
        DeclareLaunchArgument(
            "sim_params_file",
            default_value=default_sim_params,
            description="Parameter file for rpp_offline_sim_node.",
        ),
        follower_node,
        sim_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=sim_node,
                on_exit=[EmitEvent(event=Shutdown(reason="offline sim finished"))],
            )
        ),
    ])
