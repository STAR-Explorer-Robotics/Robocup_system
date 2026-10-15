from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _launch_value(context, name: str) -> str:
    return str(LaunchConfiguration(name).perform(context)).strip()


def _launch_bool(context, name: str) -> bool:
    value = _launch_value(context, name).lower()
    return value in {"1", "true", "yes", "on"}


def _optional_launch_bool(context, name: str) -> bool | None:
    value = _launch_value(context, name).lower()
    if not value:
        return None
    return value in {"1", "true", "yes", "on"}


def _create_sensing_node(context):
    params = {
        "config": _launch_value(context, "config"),
        "use_sim_time": _launch_bool(context, "use_sim_time"),
        "save_outputs": _launch_bool(context, "save_outputs"),
        "output_root": _launch_value(context, "output_root"),
        "max_frames": int(_launch_value(context, "max_frames") or "0"),
    }
    for name in (
        "input_mode",
        "image_topic",
        "overlay_topic",
        "result_topic",
        "latest_confirmed_topic",
        "latest_confirmed_overlay_topic",
        "status_topic",
        "inspection_config",
        "pyrealsense_serial",
        "pyrealsense_format",
        "camera_passthrough_topic",
        "camera_passthrough_frame_id",
        "camera_passthrough_encoding",
    ):
        value = _launch_value(context, name)
        if value:
            params[name] = value

    for name in ("pyrealsense_width", "pyrealsense_height", "pyrealsense_fps"):
        value = int(_launch_value(context, name) or "0")
        if value > 0:
            params[name] = value

    save_overlay_video = _launch_value(context, "save_overlay_video")
    if save_overlay_video:
        params["save_overlay_video"] = save_overlay_video.lower() in {"1", "true", "yes", "on"}

    for name in ("publish_overlay", "enable_local_overlay_window", "publish_camera_passthrough"):
        value = _optional_launch_bool(context, name)
        if value is not None:
            params[name] = value

    for name in ("overlay_publish_fps", "local_overlay_window_fps", "camera_passthrough_fps"):
        value = _launch_value(context, name)
        if value:
            params[name] = float(value)

    local_overlay_window_name = _launch_value(context, "local_overlay_window_name")
    if local_overlay_window_name:
        params["local_overlay_window_name"] = local_overlay_window_name

    local_overlay_wait_key_ms = int(_launch_value(context, "local_overlay_wait_key_ms") or "0")
    if local_overlay_wait_key_ms > 0:
        params["local_overlay_wait_key_ms"] = local_overlay_wait_key_ms

    sensing_node = Node(
        package="qrc_ros2_perception",
        executable="qrc-sensing-node",
        name="qrc_sensing_node",
        output="screen",
        parameters=[params],
    )
    shutdown_when_sensing_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=sensing_node,
            on_exit=[EmitEvent(event=Shutdown(reason="qrc_sensing_node exited"))],
        )
    )
    return [sensing_node, shutdown_when_sensing_exits]


def generate_launch_description() -> LaunchDescription:
    config_arg = DeclareLaunchArgument(
        "config",
        default_value=PathJoinSubstitution([FindPackageShare("qrc_ros2_perception"), "configs", "realsense.yaml"]),
        description="Path to qrc_ros2_perception YAML config.",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="",
        description="Optional input sensor_msgs/Image topic override.",
    )
    input_mode_arg = DeclareLaunchArgument(
        "input_mode",
        default_value="",
        description="Optional input mode override: ros_image_topic or pyrealsense2.",
    )
    pyrealsense_width_arg = DeclareLaunchArgument(
        "pyrealsense_width",
        default_value="0",
        description="Optional pyrealsense2 color stream width override; 0 keeps YAML.",
    )
    pyrealsense_height_arg = DeclareLaunchArgument(
        "pyrealsense_height",
        default_value="0",
        description="Optional pyrealsense2 color stream height override; 0 keeps YAML.",
    )
    pyrealsense_fps_arg = DeclareLaunchArgument(
        "pyrealsense_fps",
        default_value="0",
        description="Optional pyrealsense2 color stream FPS override; 0 keeps YAML.",
    )
    pyrealsense_serial_arg = DeclareLaunchArgument(
        "pyrealsense_serial",
        default_value="",
        description="Optional pyrealsense2 camera serial override.",
    )
    pyrealsense_format_arg = DeclareLaunchArgument(
        "pyrealsense_format",
        default_value="",
        description="Optional pyrealsense2 color format override.",
    )
    overlay_topic_arg = DeclareLaunchArgument(
        "overlay_topic",
        default_value="",
        description="Optional output overlay sensor_msgs/Image topic override.",
    )
    publish_overlay_arg = DeclareLaunchArgument(
        "publish_overlay",
        default_value="",
        description="Optional live overlay topic publish override; empty keeps YAML.",
    )
    overlay_publish_fps_arg = DeclareLaunchArgument(
        "overlay_publish_fps",
        default_value="",
        description="Optional live overlay topic FPS override; empty keeps YAML.",
    )
    enable_local_overlay_window_arg = DeclareLaunchArgument(
        "enable_local_overlay_window",
        default_value="",
        description="Optional local OpenCV live overlay window override; empty keeps YAML.",
    )
    local_overlay_window_fps_arg = DeclareLaunchArgument(
        "local_overlay_window_fps",
        default_value="",
        description="Optional local OpenCV live overlay window FPS override; empty keeps YAML.",
    )
    local_overlay_window_name_arg = DeclareLaunchArgument(
        "local_overlay_window_name",
        default_value="",
        description="Optional local OpenCV live overlay window name override; empty keeps YAML.",
    )
    local_overlay_wait_key_ms_arg = DeclareLaunchArgument(
        "local_overlay_wait_key_ms",
        default_value="0",
        description="Optional local OpenCV waitKey delay in milliseconds; 0 keeps YAML.",
    )
    publish_camera_passthrough_arg = DeclareLaunchArgument(
        "publish_camera_passthrough",
        default_value="",
        description="Optional raw camera passthrough topic publish override; empty keeps YAML.",
    )
    camera_passthrough_topic_arg = DeclareLaunchArgument(
        "camera_passthrough_topic",
        default_value="",
        description="Optional raw camera passthrough sensor_msgs/Image topic override.",
    )
    camera_passthrough_fps_arg = DeclareLaunchArgument(
        "camera_passthrough_fps",
        default_value="",
        description="Optional raw camera passthrough FPS override; empty keeps YAML, <=0 publishes every frame.",
    )
    camera_passthrough_frame_id_arg = DeclareLaunchArgument(
        "camera_passthrough_frame_id",
        default_value="",
        description="Optional raw camera passthrough frame_id override.",
    )
    camera_passthrough_encoding_arg = DeclareLaunchArgument(
        "camera_passthrough_encoding",
        default_value="",
        description="Optional raw camera passthrough encoding override; currently bgr8.",
    )
    result_topic_arg = DeclareLaunchArgument(
        "result_topic",
        default_value="",
        description="Optional output std_msgs/String JSON result topic override.",
    )
    latest_confirmed_topic_arg = DeclareLaunchArgument(
        "latest_confirmed_topic",
        default_value="",
        description="Optional output std_msgs/String latest confirmed JSON topic override.",
    )
    latest_confirmed_overlay_topic_arg = DeclareLaunchArgument(
        "latest_confirmed_overlay_topic",
        default_value="",
        description="Optional output sensor_msgs/Image latest confirmed evidence overlay topic override.",
    )
    status_topic_arg = DeclareLaunchArgument(
        "status_topic",
        default_value="",
        description="Optional output std_msgs/String node status topic override.",
    )
    enable_result_panel_arg = DeclareLaunchArgument(
        "enable_result_panel",
        default_value="false",
        description="Start the latest confirmed result panel node with the sensing node.",
    )
    result_panel_config_arg = DeclareLaunchArgument(
        "result_panel_config",
        default_value=PathJoinSubstitution([FindPackageShare("qrc_ros2_perception"), "configs", "result_panel.yaml"]),
        description="Path to qrc_ros2_perception result panel YAML config.",
    )
    result_panel_image_topic_arg = DeclareLaunchArgument(
        "result_panel_image_topic",
        default_value="/qrc/sensing/result_panel/image_raw",
        description="Output sensor_msgs/Image topic for the rendered result panel.",
    )
    show_result_panel_window_arg = DeclareLaunchArgument(
        "show_result_panel_window",
        default_value="true",
        description="Show the local OpenCV result panel window when panel is enabled.",
    )
    publish_result_panel_image_arg = DeclareLaunchArgument(
        "publish_result_panel_image",
        default_value="true",
        description="Publish the rendered result panel image topic when panel is enabled.",
    )
    inspection_config_arg = DeclareLaunchArgument(
        "inspection_config",
        default_value="",
        description="Optional path to the inspection sensing config override.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulated time for rosbag playback.",
    )
    save_outputs_arg = DeclareLaunchArgument(
        "save_outputs",
        default_value="true",
        description="Write inspection-like run artifacts under output_root.",
    )
    save_overlay_video_arg = DeclareLaunchArgument(
        "save_overlay_video",
        default_value="",
        description="Optional overlay video writer override; empty keeps YAML.",
    )
    output_root_arg = DeclareLaunchArgument(
        "output_root",
        default_value="outputs",
        description="qrc_ros2_perception-relative output root for run artifacts.",
    )
    max_frames_arg = DeclareLaunchArgument(
        "max_frames",
        default_value="0",
        description="Stop after this many processed frames; 0 means run until interrupted.",
    )

    sensing_node = OpaqueFunction(function=_create_sensing_node)

    panel_node = Node(
        package="qrc_ros2_perception",
        executable="qrc-result-panel",
        name="qrc_result_panel",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_result_panel")),
        parameters=[
            {
                "config": LaunchConfiguration("result_panel_config"),
                "latest_confirmed_topic": LaunchConfiguration("latest_confirmed_topic"),
                "latest_confirmed_overlay_topic": LaunchConfiguration("latest_confirmed_overlay_topic"),
                "status_topic": LaunchConfiguration("status_topic"),
                "panel_image_topic": LaunchConfiguration("result_panel_image_topic"),
                "publish_panel_image": ParameterValue(LaunchConfiguration("publish_result_panel_image"), value_type=bool),
                "show_window": ParameterValue(LaunchConfiguration("show_result_panel_window"), value_type=bool),
            }
        ],
    )

    return LaunchDescription(
        [
            config_arg,
            input_mode_arg,
            image_topic_arg,
            pyrealsense_width_arg,
            pyrealsense_height_arg,
            pyrealsense_fps_arg,
            pyrealsense_serial_arg,
            pyrealsense_format_arg,
            overlay_topic_arg,
            publish_overlay_arg,
            overlay_publish_fps_arg,
            enable_local_overlay_window_arg,
            local_overlay_window_fps_arg,
            local_overlay_window_name_arg,
            local_overlay_wait_key_ms_arg,
            publish_camera_passthrough_arg,
            camera_passthrough_topic_arg,
            camera_passthrough_fps_arg,
            camera_passthrough_frame_id_arg,
            camera_passthrough_encoding_arg,
            result_topic_arg,
            latest_confirmed_topic_arg,
            latest_confirmed_overlay_topic_arg,
            status_topic_arg,
            enable_result_panel_arg,
            result_panel_config_arg,
            result_panel_image_topic_arg,
            show_result_panel_window_arg,
            publish_result_panel_image_arg,
            inspection_config_arg,
            use_sim_time_arg,
            save_outputs_arg,
            save_overlay_video_arg,
            output_root_arg,
            max_frames_arg,
            sensing_node,
            panel_node,
        ]
    )
