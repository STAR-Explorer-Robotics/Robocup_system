"""ROS parameter loading for qrc_sensing_node."""

from __future__ import annotations

from pathlib import Path

from rclpy.node import Node

from qrc_ros2_perception.config import (
    PerceptionBridgeConfig,
    apply_parameter_overrides,
    load_perception_config,
)


def declare_and_load_parameters(node: Node) -> PerceptionBridgeConfig:
    """Declare qrc_sensing_node parameters and merge ROS overrides into YAML config."""

    node.declare_parameter("config", "")
    config_path = str(node.get_parameter("config").value).strip()
    base_config = load_perception_config(Path(config_path) if config_path else None)
    _declare_runtime_parameters(node, base_config)
    return _read_runtime_parameters(node, base_config)


def _declare_runtime_parameters(node: Node, config: PerceptionBridgeConfig) -> None:
    node.declare_parameter("input_mode", config.input_mode)
    node.declare_parameter("image_topic", config.image_topic)
    node.declare_parameter("camera_info_topic", config.camera_info_topic or "")
    node.declare_parameter("input_qos", config.input_qos)
    node.declare_parameter("pyrealsense_width", config.pyrealsense_width)
    node.declare_parameter("pyrealsense_height", config.pyrealsense_height)
    node.declare_parameter("pyrealsense_fps", config.pyrealsense_fps)
    node.declare_parameter("pyrealsense_serial", config.pyrealsense_serial or "")
    node.declare_parameter("pyrealsense_format", config.pyrealsense_format)
    node.declare_parameter("inspection_config", str(config.inspection_config))
    node.declare_parameter("publish_overlay", config.publish_overlay)
    node.declare_parameter("overlay_publish_fps", config.overlay_publish_fps)
    node.declare_parameter("enable_local_overlay_window", config.enable_local_overlay_window)
    node.declare_parameter("local_overlay_window_fps", config.local_overlay_window_fps)
    node.declare_parameter("local_overlay_window_name", config.local_overlay_window_name)
    node.declare_parameter("local_overlay_wait_key_ms", config.local_overlay_wait_key_ms)
    node.declare_parameter("publish_camera_passthrough", config.publish_camera_passthrough)
    node.declare_parameter("camera_passthrough_topic", config.camera_passthrough_topic)
    node.declare_parameter("camera_passthrough_fps", config.camera_passthrough_fps)
    node.declare_parameter("camera_passthrough_frame_id", config.camera_passthrough_frame_id)
    node.declare_parameter("camera_passthrough_encoding", config.camera_passthrough_encoding)
    node.declare_parameter("publish_result_json", config.publish_result_json)
    node.declare_parameter("publish_latest_confirmed_json", config.publish_latest_confirmed_json)
    node.declare_parameter("max_processing_fps", config.max_processing_fps or 0.0)
    node.declare_parameter("max_frames", config.max_frames or 0)
    node.declare_parameter("drop_frames_when_busy", config.drop_frames_when_busy)
    node.declare_parameter("overlay_topic", config.overlay_topic)
    node.declare_parameter("result_topic", config.result_topic)
    node.declare_parameter("latest_confirmed_topic", config.latest_confirmed_topic)
    node.declare_parameter("latest_confirmed_overlay_topic", config.latest_confirmed_overlay_topic)
    node.declare_parameter("status_topic", config.status_topic)
    node.declare_parameter("status_period_sec", config.status_period_sec)
    node.declare_parameter("save_outputs", config.save_outputs)
    node.declare_parameter("output_root", str(config.output_root))
    node.declare_parameter("save_overlay_video", config.save_overlay_video)
    node.declare_parameter("save_confirmed_overlays", config.save_confirmed_overlays)
    node.declare_parameter("latest_result_write_period_sec", config.latest_result_write_period_sec)
    node.declare_parameter("output_video_fps", config.output_video_fps)


def _read_runtime_parameters(node: Node, config: PerceptionBridgeConfig) -> PerceptionBridgeConfig:
    max_fps = float(node.get_parameter("max_processing_fps").value)
    max_frames = int(node.get_parameter("max_frames").value)
    input_mode = _parameter_text(node, "input_mode", config.input_mode)
    image_topic = _parameter_text(node, "image_topic", config.image_topic)
    camera_info_topic = _parameter_text(node, "camera_info_topic", config.camera_info_topic or "")
    pyrealsense_serial = _parameter_text(node, "pyrealsense_serial", config.pyrealsense_serial or "")
    pyrealsense_format = _parameter_text(node, "pyrealsense_format", config.pyrealsense_format)
    inspection_config = _parameter_text(node, "inspection_config", str(config.inspection_config))
    overlay_topic = _parameter_text(node, "overlay_topic", config.overlay_topic)
    result_topic = _parameter_text(node, "result_topic", config.result_topic)
    latest_confirmed_topic = _parameter_text(node, "latest_confirmed_topic", config.latest_confirmed_topic)
    latest_confirmed_overlay_topic = _parameter_text(
        node,
        "latest_confirmed_overlay_topic",
        config.latest_confirmed_overlay_topic,
    )
    status_topic = _parameter_text(node, "status_topic", config.status_topic)
    output_root = _parameter_text(node, "output_root", str(config.output_root))
    local_overlay_window_name = _parameter_text(
        node,
        "local_overlay_window_name",
        config.local_overlay_window_name,
    )
    camera_passthrough_topic = _parameter_text(
        node,
        "camera_passthrough_topic",
        config.camera_passthrough_topic,
    )
    camera_passthrough_frame_id = _parameter_text(
        node,
        "camera_passthrough_frame_id",
        config.camera_passthrough_frame_id,
    )
    camera_passthrough_encoding = _parameter_text(
        node,
        "camera_passthrough_encoding",
        config.camera_passthrough_encoding,
    )
    merged = PerceptionBridgeConfig(
        input_mode=input_mode,
        image_topic=image_topic,
        camera_info_topic=camera_info_topic or None,
        input_qos=str(node.get_parameter("input_qos").value),
        pyrealsense_width=int(node.get_parameter("pyrealsense_width").value),
        pyrealsense_height=int(node.get_parameter("pyrealsense_height").value),
        pyrealsense_fps=int(node.get_parameter("pyrealsense_fps").value),
        pyrealsense_serial=pyrealsense_serial or None,
        pyrealsense_format=pyrealsense_format,
        inspection_config=Path(inspection_config),
        publish_overlay=bool(node.get_parameter("publish_overlay").value),
        overlay_publish_fps=float(node.get_parameter("overlay_publish_fps").value),
        enable_local_overlay_window=bool(node.get_parameter("enable_local_overlay_window").value),
        local_overlay_window_fps=float(node.get_parameter("local_overlay_window_fps").value),
        local_overlay_window_name=local_overlay_window_name,
        local_overlay_wait_key_ms=int(node.get_parameter("local_overlay_wait_key_ms").value),
        publish_camera_passthrough=bool(node.get_parameter("publish_camera_passthrough").value),
        camera_passthrough_topic=camera_passthrough_topic,
        camera_passthrough_fps=float(node.get_parameter("camera_passthrough_fps").value),
        camera_passthrough_frame_id=camera_passthrough_frame_id,
        camera_passthrough_encoding=camera_passthrough_encoding,
        publish_result_json=bool(node.get_parameter("publish_result_json").value),
        publish_latest_confirmed_json=bool(node.get_parameter("publish_latest_confirmed_json").value),
        max_processing_fps=max_fps if max_fps > 0.0 else None,
        max_frames=max_frames if max_frames > 0 else None,
        drop_frames_when_busy=bool(node.get_parameter("drop_frames_when_busy").value),
        overlay_topic=overlay_topic,
        result_topic=result_topic,
        latest_confirmed_topic=latest_confirmed_topic,
        latest_confirmed_overlay_topic=latest_confirmed_overlay_topic,
        status_topic=status_topic,
        status_period_sec=float(node.get_parameter("status_period_sec").value),
        save_outputs=bool(node.get_parameter("save_outputs").value),
        output_root=Path(output_root),
        save_overlay_video=bool(node.get_parameter("save_overlay_video").value),
        save_confirmed_overlays=bool(node.get_parameter("save_confirmed_overlays").value),
        latest_result_write_period_sec=float(node.get_parameter("latest_result_write_period_sec").value),
        output_video_fps=float(node.get_parameter("output_video_fps").value),
        log_level=config.log_level,
    )
    return apply_parameter_overrides(
        merged,
        image_topic=image_topic,
        overlay_topic=overlay_topic,
        result_topic=result_topic,
        latest_confirmed_topic=latest_confirmed_topic,
        latest_confirmed_overlay_topic=latest_confirmed_overlay_topic,
        input_mode=input_mode,
        pyrealsense_width=int(node.get_parameter("pyrealsense_width").value),
        pyrealsense_height=int(node.get_parameter("pyrealsense_height").value),
        pyrealsense_fps=int(node.get_parameter("pyrealsense_fps").value),
        pyrealsense_serial=pyrealsense_serial,
        pyrealsense_format=pyrealsense_format,
        publish_camera_passthrough=bool(node.get_parameter("publish_camera_passthrough").value),
        camera_passthrough_topic=camera_passthrough_topic,
        camera_passthrough_fps=float(node.get_parameter("camera_passthrough_fps").value),
        camera_passthrough_frame_id=camera_passthrough_frame_id,
        camera_passthrough_encoding=camera_passthrough_encoding,
        inspection_config=inspection_config,
        output_root=output_root,
        use_sim_time=None,
    )


def _parameter_text(node: Node, name: str, default: str) -> str:
    text = str(node.get_parameter(name).value).strip()
    return text or default
