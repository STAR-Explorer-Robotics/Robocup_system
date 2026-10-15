#!/usr/bin/env python3
"""Shift a localization odometry pose from a head/LiDAR point to robot center.

The offset is a horizontal body-frame vector.  Positive x is forward and
positive y is left.  Only yaw is used to rotate the offset into the map frame,
so a tilted LiDAR mount does not tilt the horizontal planning reference point.
"""

from __future__ import annotations

import argparse
import math

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RobotCenterOdom(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("robot_center_odom")
        self.input_topic = args.input_topic
        self.output_topic = args.output_topic
        self.offset_x = float(args.offset_x)
        self.offset_y = float(args.offset_y)
        self.offset_z = float(args.offset_z)
        self.child_frame_id = args.child_frame_id
        self.publish_tf = bool(args.publish_tf)
        self.body_marker_topic = args.body_marker_topic
        self.publish_body_marker = not args.no_body_marker
        self.body_length = float(args.body_length)
        self.body_width = float(args.body_width)
        self.body_height = float(args.body_height)
        self.body_z_offset = float(args.body_z_offset)

        self.pub = self.create_publisher(Odometry, self.output_topic, 20)
        self.marker_pub = (
            self.create_publisher(Marker, self.body_marker_topic, 10)
            if self.publish_body_marker
            else None
        )
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.sub = self.create_subscription(Odometry, self.input_topic, self.on_odom, 20)
        self.get_logger().info(
            "Robot center odom started: "
            f"{self.input_topic} -> {self.output_topic}, "
            f"offset_body=({self.offset_x:.3f}, {self.offset_y:.3f}, {self.offset_z:.3f})m, "
            f"body_marker={self.body_marker_topic if self.publish_body_marker else 'disabled'} "
            f"size=({self.body_length:.2f}, {self.body_width:.2f}, {self.body_height:.2f})m"
        )

    def on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        yaw = yaw_from_quaternion(pose.orientation)
        c = math.cos(yaw)
        s = math.sin(yaw)
        dx_map = c * self.offset_x - s * self.offset_y
        dy_map = s * self.offset_x + c * self.offset_y

        out = Odometry()
        out.header = msg.header
        out.child_frame_id = self.child_frame_id
        out.pose.pose.position.x = float(pose.position.x) + dx_map
        out.pose.pose.position.y = float(pose.position.y) + dy_map
        out.pose.pose.position.z = float(pose.position.z) + self.offset_z
        out.pose.pose.orientation = pose.orientation
        out.pose.covariance = msg.pose.covariance

        out.twist.twist = msg.twist.twist
        wz = float(msg.twist.twist.angular.z)
        out.twist.twist.linear.x = float(msg.twist.twist.linear.x) - wz * self.offset_y
        out.twist.twist.linear.y = float(msg.twist.twist.linear.y) + wz * self.offset_x
        out.twist.covariance = msg.twist.covariance

        self.pub.publish(out)
        if self.marker_pub is not None:
            self.publish_body_marker_msg(out)
        if self.tf_broadcaster is not None:
            self.publish_tf_msg(out)

    def publish_body_marker_msg(self, odom: Odometry) -> None:
        marker = Marker()
        marker.header = odom.header
        marker.ns = "robot_center_body"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = odom.pose.pose
        marker.pose.position.z = float(marker.pose.position.z) + self.body_z_offset
        marker.scale.x = self.body_length
        marker.scale.y = self.body_width
        marker.scale.z = self.body_height
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 0.35
        self.marker_pub.publish(marker)

    def publish_tf_msg(self, odom: Odometry) -> None:
        tf_msg = TransformStamped()
        tf_msg.header = odom.header
        tf_msg.child_frame_id = odom.child_frame_id
        tf_msg.transform.translation.x = float(odom.pose.pose.position.x)
        tf_msg.transform.translation.y = float(odom.pose.pose.position.y)
        tf_msg.transform.translation.z = float(odom.pose.pose.position.z)
        tf_msg.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-topic", default="/lightning/lio_odom")
    parser.add_argument("--output-topic", default="/robot_center/odom")
    parser.add_argument("--offset-x", type=float, default=-0.25)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--offset-z", type=float, default=0.0)
    parser.add_argument("--child-frame-id", default="robot_center")
    parser.add_argument("--publish-tf", action="store_true")
    parser.add_argument("--body-marker-topic", default="/robot_center/body_marker")
    parser.add_argument("--body-length", type=float, default=0.60)
    parser.add_argument("--body-width", type=float, default=0.40)
    parser.add_argument("--body-height", type=float, default=0.20)
    parser.add_argument("--body-z-offset", type=float, default=0.0)
    parser.add_argument("--no-body-marker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = RobotCenterOdom(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
