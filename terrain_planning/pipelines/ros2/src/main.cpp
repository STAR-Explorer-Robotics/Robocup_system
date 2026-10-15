#include <rclcpp/rclcpp.hpp>

#include <iostream>

#include "ros2_node.h"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  try {
    auto     node = std::make_shared<rclcpp::Node>("trg_ros2_node");
    ROS2Node trg_ros2_node(node);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Failed to start TRG ROS2 node: " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
}
