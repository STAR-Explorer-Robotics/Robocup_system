#include "rclcpp/rclcpp.hpp"
#include "trg_path_follower/rpp_follower_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<trg_path_follower::RegulatedPurePursuitNode>());
  rclcpp::shutdown();
  return 0;
}
