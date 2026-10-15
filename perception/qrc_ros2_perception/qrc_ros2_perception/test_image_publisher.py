"""Small ROS2 raw image publisher for bridge smoke tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

from qrc_ros2_perception.config import DEFAULT_IMAGE_TOPIC
from qrc_ros2_perception.image_codec import bgr_to_image_msg


class StaticImagePublisher(Node):
    """Publish one local image repeatedly as sensor_msgs/Image."""

    def __init__(self) -> None:
        super().__init__("qrc_test_image_publisher")
        self.declare_parameter("image_path", "../inspection/samples/qr_sample.png")
        self.declare_parameter("image_topic", DEFAULT_IMAGE_TOPIC)
        self.declare_parameter("fps", 2.0)
        self.declare_parameter("frame_id", "qrc_test_camera")

        self.image_path = Path(str(self.get_parameter("image_path").value)).resolve()
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        fps = max(float(self.get_parameter("fps").value), 0.1)

        image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read test image: {self.image_path}")
        self.image_bgr = image
        self.publisher = self.create_publisher(Image, self.image_topic, 10)
        self.timer = self.create_timer(1.0 / fps, self._publish)
        self.get_logger().info(f"Publishing {self.image_path} to {self.image_topic} at {fps:.2f} FPS")

    def _publish(self) -> None:
        header = self.get_clock().now().to_msg()
        message = bgr_to_image_msg(self.image_bgr)
        message.header.stamp = header
        message.header.frame_id = self.frame_id
        self.publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: StaticImagePublisher | None = None
    try:
        node = StaticImagePublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
