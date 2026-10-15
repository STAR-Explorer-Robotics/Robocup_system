from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_arg = DeclareLaunchArgument(
        "config",
        default_value=PathJoinSubstitution([FindPackageShare("qrc_ros2_perception"), "configs", "result_panel.yaml"]),
        description="Path to qrc_ros2_perception result panel YAML config.",
    )
    latest_confirmed_topic_arg = DeclareLaunchArgument(
        "latest_confirmed_topic",
        default_value="",
        description="Optional latest confirmed std_msgs/String JSON topic override.",
    )
    latest_confirmed_overlay_topic_arg = DeclareLaunchArgument(
        "latest_confirmed_overlay_topic",
        default_value="",
        description="Optional latest confirmed evidence sensor_msgs/Image topic override.",
    )
    status_topic_arg = DeclareLaunchArgument(
        "status_topic",
        default_value="",
        description="Optional status std_msgs/String JSON topic override.",
    )
    panel_image_topic_arg = DeclareLaunchArgument(
        "panel_image_topic",
        default_value="",
        description="Optional output result panel sensor_msgs/Image topic override.",
    )
    publish_panel_image_arg = DeclareLaunchArgument(
        "publish_panel_image",
        default_value="true",
        description="Publish rendered panel image topic.",
    )
    show_window_arg = DeclareLaunchArgument(
        "show_window",
        default_value="true",
        description="Show local OpenCV result panel window.",
    )
    refresh_fps_arg = DeclareLaunchArgument(
        "refresh_fps",
        default_value="2.0",
        description="Result panel refresh rate in Hz.",
    )
    stale_threshold_sec_arg = DeclareLaunchArgument(
        "stale_threshold_sec",
        default_value="10.0",
        description="Seconds before latest confirmed result is marked STALE in the UI.",
    )

    panel_node = Node(
        package="qrc_ros2_perception",
        executable="qrc-result-panel",
        name="qrc_result_panel",
        output="screen",
        parameters=[
            {
                "config": LaunchConfiguration("config"),
                "latest_confirmed_topic": LaunchConfiguration("latest_confirmed_topic"),
                "latest_confirmed_overlay_topic": LaunchConfiguration("latest_confirmed_overlay_topic"),
                "status_topic": LaunchConfiguration("status_topic"),
                "panel_image_topic": LaunchConfiguration("panel_image_topic"),
                "publish_panel_image": ParameterValue(LaunchConfiguration("publish_panel_image"), value_type=bool),
                "show_window": ParameterValue(LaunchConfiguration("show_window"), value_type=bool),
                "refresh_fps": ParameterValue(LaunchConfiguration("refresh_fps"), value_type=float),
                "stale_threshold_sec": ParameterValue(LaunchConfiguration("stale_threshold_sec"), value_type=float),
            }
        ],
    )

    return LaunchDescription(
        [
            config_arg,
            latest_confirmed_topic_arg,
            latest_confirmed_overlay_topic_arg,
            status_topic_arg,
            panel_image_topic_arg,
            publish_panel_image_arg,
            show_window_arg,
            refresh_fps_arg,
            stale_threshold_sec_arg,
            panel_node,
        ]
    )
