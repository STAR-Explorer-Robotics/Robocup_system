"""ROS2 node for the QRC inspection sensing bridge."""

from __future__ import annotations

import json
import logging

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from qrc_ros2_perception.bridge import QrcSensingBridge
from qrc_ros2_perception.input_manager import QrcInputManager, resolve_source_topic
from qrc_ros2_perception.local_overlay_window import LocalOverlayWindow
from qrc_ros2_perception.node_parameters import declare_and_load_parameters
from qrc_ros2_perception.passthrough import CameraPassthrough
from qrc_ros2_perception.processing import FrameProcessor
from qrc_ros2_perception.publishers import QrcSensingPublishers
from qrc_ros2_perception.run_outputs import Ros2RunOutputWriter
from qrc_ros2_perception.status import SensingNodeStatus


class QrcSensingNode(Node):
    """Subscribe to ROS2 images, run inspection sensing, and publish overlay/results."""

    def __init__(self) -> None:
        super().__init__("qrc_sensing_node")
        self.config = declare_and_load_parameters(self)

        self.bridge = QrcSensingBridge(
            self.config.inspection_config,
            logger=logging.getLogger("qrc_ros2_perception.inspection"),
        )
        self._source_topic = resolve_source_topic(self.config)
        self._status = SensingNodeStatus(config=self.config, source_topic=self._source_topic)
        self._local_overlay_window = (
            LocalOverlayWindow(
                self.config.local_overlay_window_name,
                wait_key_ms=self.config.local_overlay_wait_key_ms,
                logger=self.get_logger(),
            )
            if self.config.enable_local_overlay_window
            else None
        )
        self._outputs = (
            Ros2RunOutputWriter(
                output_root=self.config.output_root,
                image_topic=self._source_topic,
                overlay_topic=self.config.overlay_topic,
                result_topic=self.config.result_topic,
                status_topic=self.config.status_topic,
                inspection_config=self.config.inspection_config,
                save_overlay_video=self.config.save_overlay_video,
                output_video_fps=self.config.output_video_fps,
                latest_result_write_period_sec=self.config.latest_result_write_period_sec,
                save_confirmed_overlays=self.config.save_confirmed_overlays,
            )
            if self.config.save_outputs
            else None
        )
        self._publishers = QrcSensingPublishers(node=self, config=self.config)
        self._camera_passthrough = CameraPassthrough(node=self, config=self.config)
        self._processor = FrameProcessor(
            config=self.config,
            bridge=self.bridge,
            publishers=self._publishers,
            status=self._status,
            outputs=self._outputs,
            local_overlay_window=self._local_overlay_window,
            request_shutdown_if_done=self._request_shutdown_if_done,
        )
        self._input_manager = QrcInputManager(
            node=self,
            config=self.config,
            status=self._status,
            processor=self._processor,
            passthrough=self._camera_passthrough,
            source_topic=self._source_topic,
        )
        self._status_timer = self.create_timer(self.config.status_period_sec, self._publish_status)
        self._input_manager.start()

        self.get_logger().info(
            "QRC sensing bridge ready: "
            f"input_mode={self.config.input_mode} "
            f"image_topic={self.config.image_topic} "
            f"overlay_topic={self.config.overlay_topic} "
            f"result_topic={self.config.result_topic} "
            f"latest_confirmed_topic={self.config.latest_confirmed_topic} "
            f"latest_confirmed_overlay_topic={self.config.latest_confirmed_overlay_topic} "
            f"status_topic={self.config.status_topic} "
            f"inspection_config={self.config.inspection_config} "
            f"input_qos={self.config.input_qos} "
            f"publish_overlay={self.config.publish_overlay} "
            f"overlay_publish_fps={self.config.overlay_publish_fps} "
            f"enable_local_overlay_window={self.config.enable_local_overlay_window} "
            f"local_overlay_window_fps={self.config.local_overlay_window_fps} "
            f"publish_camera_passthrough={self.config.publish_camera_passthrough} "
            f"camera_passthrough_topic={self.config.camera_passthrough_topic} "
            f"camera_passthrough_fps={self.config.camera_passthrough_fps} "
            f"camera_passthrough_frame_id={self.config.camera_passthrough_frame_id} "
            f"camera_passthrough_encoding={self.config.camera_passthrough_encoding} "
            f"output_dir={self._outputs.run_dir if self._outputs is not None else 'disabled'}"
        )

    def _publish_status(self) -> None:
        if not rclpy.ok():
            return
        status = self._build_status()
        status_json = json.dumps(status, ensure_ascii=False, separators=(",", ":"))
        self._publishers.publish_status(status_json)
        if self._outputs is not None:
            self._outputs.write_status(status_json)

    def _build_status(self) -> dict[str, object]:
        return self._status.build_payload(
            camera_passthrough_frames_published=self._camera_passthrough.frames_published,
            camera_passthrough_frames_skipped_no_subscriber=(
                self._camera_passthrough.frames_skipped_no_subscriber
            ),
            camera_passthrough_subscription_count=self._camera_passthrough.subscription_count(),
        )

    def finalize_outputs(self) -> None:
        if self._outputs is None:
            return
        try:
            self._outputs.finalize(node_summary=self._build_status())
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            self.get_logger().warning(f"Failed to finalize ROS2 sensing outputs: {exc}")

    def stop_inputs(self) -> None:
        if self._local_overlay_window is not None:
            self._local_overlay_window.close()
        self._input_manager.stop()

    def _request_shutdown_if_done(self) -> None:
        if self.config.max_frames is None:
            return
        if self._status.shutdown_requested or self._status.frames_processed < self.config.max_frames:
            return
        self._status.shutdown_requested = True
        self.get_logger().info(
            f"Reached max_frames={self.config.max_frames}; finalizing ROS2 sensing outputs before shutdown."
        )
        if self._outputs is not None:
            status_json = json.dumps(self._build_status(), ensure_ascii=False, separators=(",", ":"))
            self._outputs.write_status(status_json)
        raise KeyboardInterrupt


def main(args: list[str] | None = None) -> None:
    """ROS2 entry point."""

    node: QrcSensingNode | None = None
    rclpy.init(args=args)
    try:
        node = QrcSensingNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.stop_inputs()
            node.finalize_outputs()
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":  # pragma: no cover
    main()
