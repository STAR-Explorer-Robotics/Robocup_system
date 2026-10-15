#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class PosePublisher(Node):

    def __init__(self):
        super().__init__('trg_rviz_pose_bridge')

        self.frame_id = self.declare_parameter('frame_id', 'map').value
        self.child_frame_id = self.declare_parameter('child_frame_id',
                                                     'base_link').value
        initialpose_topic = self.declare_parameter('initialpose_topic',
                                                   '/initialpose').value
        odom_topic = self.declare_parameter('odom_topic',
                                            '/laser_odometry').value
        goal_in_topic = self.declare_parameter('goal_in_topic',
                                               '/goal_pose').value
        goal_out_topic = self.declare_parameter('goal_out_topic', '').value

        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self.goal_pub = None
        if goal_out_topic:
            self.goal_pub = self.create_publisher(PoseStamped, goal_out_topic, 10)

        self.create_subscription(PoseWithCovarianceStamped, initialpose_topic,
                                 self.initial_pose_callback, 10)
        if goal_in_topic and self.goal_pub is not None:
            self.create_subscription(PoseStamped, goal_in_topic,
                                     self.goal_callback, 10)

        self.get_logger().info(
            f'RViz initial pose bridge: {initialpose_topic} -> {odom_topic}')

    def initial_pose_callback(self, msg):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = self.frame_id
        odom_msg.child_frame_id = self.child_frame_id
        odom_msg.pose = msg.pose

        self.odom_pub.publish(odom_msg)
        p = msg.pose.pose.position
        self.get_logger().info(f'initial pose -> odom: {p.x}, {p.y}')

    def goal_callback(self, msg):
        if self.goal_pub is None:
            return
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = self.frame_id
        goal_msg.pose = msg.pose

        self.goal_pub.publish(goal_msg)
        self.get_logger().info(
            f'goal pose: {msg.pose.position.x}, {msg.pose.position.y}')

    def run(self):
        rclpy.spin(self)


def main(args=None):
    rclpy.init(args=args)
    pose_publisher = PosePublisher()

    try:
        pose_publisher.run()
    except KeyboardInterrupt:
        pass
    finally:
        pose_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
