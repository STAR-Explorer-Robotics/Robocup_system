# Copyright 2026 bichy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    backend_type = LaunchConfiguration("backend_type")
    robot_model = LaunchConfiguration("robot_model")
    network_interface = LaunchConfiguration("network_interface")
    input_cmd_topic = LaunchConfiguration("input_cmd_topic")
    enabled_on_start = LaunchConfiguration("enabled_on_start")
    sdk_timeout_sec = LaunchConfiguration("sdk_timeout_sec")
    cmd_timeout_sec = LaunchConfiguration("cmd_timeout_sec")
    yaw_rate_sign = LaunchConfiguration("yaw_rate_sign")
    body_height_enabled = LaunchConfiguration("body_height_enabled")
    body_height_min_m = LaunchConfiguration("body_height_min_m")
    body_height_max_m = LaunchConfiguration("body_height_max_m")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("unitree_adapter"),
                        "config",
                        "unitree_adapter.yaml",
                    ]
                ),
                description="Unitree adapter parameter file.",
            ),
            DeclareLaunchArgument(
                "backend_type",
                default_value="mock",
                description="Backend type: mock or sdk2.",
            ),
            DeclareLaunchArgument(
                "robot_model",
                default_value="a2",
                description="Unitree SDK2 robot model: a2, b2, go2, go2w, g2-w alias.",
            ),
            DeclareLaunchArgument(
                "network_interface",
                default_value="",
                description="Ethernet interface for SDK2 backend, for example enp3s0.",
            ),
            DeclareLaunchArgument(
                "input_cmd_topic",
                default_value="/cmd_vel",
                description="Guarded velocity command topic.",
            ),
            DeclareLaunchArgument(
                "enabled_on_start",
                default_value="true",
                description="Start SDK command streaming enabled.",
            ),
            DeclareLaunchArgument(
                "sdk_timeout_sec",
                default_value="2.0",
                description="Unitree SDK2 API call timeout in seconds.",
            ),
            DeclareLaunchArgument(
                "cmd_timeout_sec",
                default_value="0.25",
                description="Stop if the latest input Twist is older than this many seconds.",
            ),
            DeclareLaunchArgument(
                "yaw_rate_sign",
                default_value="1.0",
                description="Multiplier applied to input cmd_vel.angular.z before Unitree Move.",
            ),
            DeclareLaunchArgument(
                "body_height_enabled",
                default_value="false",
                description=(
                    "Enable the explicit BodyHeight service. Keep false unless "
                    "whole-body height adjustment is intentionally tested."
                ),
            ),
            DeclareLaunchArgument(
                "body_height_min_m",
                default_value="0.30",
                description="Minimum allowed absolute Unitree BodyHeight target in meters.",
            ),
            DeclareLaunchArgument(
                "body_height_max_m",
                default_value="0.50",
                description="Maximum allowed absolute Unitree BodyHeight target in meters.",
            ),
            Node(
                package="unitree_adapter",
                executable="unitree_adapter_node",
                name="unitree_adapter_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "backend.type": backend_type,
                        "backend.robot_model": robot_model,
                        "backend.network_interface": network_interface,
                        "backend.sdk_timeout_sec": sdk_timeout_sec,
                        "topics.input_cmd": input_cmd_topic,
                        "control.enabled_on_start": enabled_on_start,
                        "control.cmd_timeout_sec": cmd_timeout_sec,
                        "control.yaw_rate_sign": yaw_rate_sign,
                        "body_height.enabled": body_height_enabled,
                        "body_height.min_m": body_height_min_m,
                        "body_height.max_m": body_height_max_m,
                    },
                ],
            ),
        ]
    )
