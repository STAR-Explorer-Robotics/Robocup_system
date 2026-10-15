#!/usr/bin/env python3
"""
mission.launch.py — launch all mission-layer nodes.

Usage:
  ros2 launch mission_control mission.launch.py heading_deg:=30
  (0° = robot forward at start, positive = right, negative = left)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    heading_deg = LaunchConfiguration("heading_deg")
    config_file = LaunchConfiguration("config_file")
    require_start_signal = LaunchConfiguration("require_start_signal")
    start_direction_topic = LaunchConfiguration("start_direction_topic")
    min_start_direction_distance = LaunchConfiguration("min_start_direction_distance")
    default_config = os.path.join(
        get_package_share_directory("mission_control"),
        "config",
        "mission.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "heading_deg",
                default_value="0.0",
                description="Forward direction in degrees relative to robot start yaw.",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Mission parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "require_start_signal",
                default_value="false",
                description="Wait for a mission start direction point before executing.",
            ),
            DeclareLaunchArgument(
                "start_direction_topic",
                default_value="/mission/start_direction_point",
                description="PointStamped topic used as horizontal mission direction.",
            ),
            DeclareLaunchArgument(
                "min_start_direction_distance",
                default_value="0.5",
                description="Minimum horizontal distance from start to accept a direction point.",
            ),
            Node(
                package="mission_control",
                executable="goal_selector",
                name="goal_selector",
                output="screen",
                parameters=[
                    config_file,
                    {"heading_deg": ParameterValue(heading_deg, value_type=float)},
                    {"start_direction_topic": start_direction_topic},
                    {
                        "min_start_direction_distance": ParameterValue(
                            min_start_direction_distance, value_type=float
                        )
                    },
                ],
            ),
            Node(
                package="mission_control",
                executable="end_detector",
                name="end_detector",
                output="screen",
                parameters=[
                    config_file,
                    {"heading_deg": ParameterValue(heading_deg, value_type=float)},
                    {"start_direction_topic": start_direction_topic},
                    {
                        "min_start_direction_distance": ParameterValue(
                            min_start_direction_distance, value_type=float
                        )
                    },
                ],
            ),
            Node(
                package="mission_control",
                executable="recorded_path_manager",
                name="recorded_path_manager",
                output="screen",
                parameters=[config_file],
            ),
            Node(
                package="mission_control",
                executable="path_source_mux",
                name="path_source_mux",
                output="screen",
                parameters=[config_file],
            ),
            Node(
                package="mission_control",
                executable="mission_fsm",
                name="mission_fsm",
                output="screen",
                parameters=[
                    config_file,
                    {"heading_deg": ParameterValue(heading_deg, value_type=float)},
                    {
                        "require_start_signal": ParameterValue(
                            require_start_signal, value_type=bool
                        )
                    },
                    {"start_direction_topic": start_direction_topic},
                    {
                        "min_start_direction_distance": ParameterValue(
                            min_start_direction_distance, value_type=float
                        )
                    },
                ],
            ),
        ]
    )
