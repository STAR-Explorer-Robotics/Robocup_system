#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_srvs/srv/set_bool.hpp"

#include "trg_path_follower/rpp_geometry.hpp"

namespace trg_path_follower
{

class RegulatedPurePursuitNode final : public rclcpp::Node
{
public:
  explicit RegulatedPurePursuitNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  struct Params
  {
    std::string path_topic{"/trg/output/path"};
    std::string odom_topic{"/laser_odometry"};
    std::string cmd_vel_topic{"/cmd_vel"};
    std::string lookahead_topic{"/rpp/lookahead_point"};
    std::string debug_topic{"/rpp/debug"};
    std::string state_topic{"/rpp/state"};
    std::string set_enabled_service{"/rpp/set_enabled"};

    double control_rate_hz{30.0};
    bool enabled_on_start{true};
    bool holonomic{true};
    bool publish_debug{true};
    bool publish_zero_on_stop{true};

    double desired_speed{0.45};
    double min_tracking_speed{0.05};
    double max_forward_speed{0.6};
    double max_lateral_speed{0.3};
    double max_yaw_rate{0.7};
    double accel_limit{0.4};
    double lateral_accel_limit{0.4};
    double yaw_accel_limit{0.8};

    double lookahead_distance{1.0};
    double lookahead_time{0.5};
    double min_lookahead_distance{0.5};
    double max_lookahead_distance{1.8};

    double yaw_gain{1.2};
    double lateral_gain{1.0};
    std::string tracking_mode{"vector_field"};
    double path_tube_gain{1.5};
    double path_tube_max_correction_speed{0.4};
    double path_tube_deadband{0.03};
    double heading_error_slowdown_angle{1.2};
    double rotate_in_place_angle{2.2};

    double regulated_min_turning_radius{0.8};
    double regulated_min_speed_scale{0.2};
    double approach_slowdown_distance{1.2};
    double deviation_slowdown_distance{0.8};
    double deviation_stop_distance{1.5};
    double goal_tolerance{0.4};
    bool use_final_yaw{false};
    double final_yaw_tolerance{0.12};
    double final_yaw_gain{1.2};

    double odom_timeout_sec{0.5};
    double path_timeout_sec{0.0};
  };

  enum class TrackerState : int
  {
    kIdle = 0,
    kTracking = 1,
    kGoalReached = 2,
    kInputStale = 3,
    kPathDeviation = 4,
  };

  struct PathProjection
  {
    std::size_t segment_index{0U};
    double segment_ratio{0.0};
    Point2D point;
    Point2D tangent{1.0, 0.0};
    Point2D normal{0.0, 1.0};
    double signed_error{0.0};
    double distance{0.0};
  };

  void declare_and_load_params();
  void on_path(const nav_msgs::msg::Path::SharedPtr msg);
  void on_odom(const nav_msgs::msg::Odometry::SharedPtr msg);
  void on_control_timer();
  void on_set_enabled(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);

  bool has_valid_inputs(const rclcpp::Time & now, std::string & reason) const;
  std::size_t find_nearest_index(const Pose2D & robot, const nav_msgs::msg::Path & path) const;
  std::size_t find_lookahead_index(
    const nav_msgs::msg::Path & path,
    std::size_t start_index,
    double lookahead_distance) const;
  PathProjection project_to_path(const Pose2D & robot, const nav_msgs::msg::Path & path) const;
  std::size_t find_lookahead_index_from_projection(
    const nav_msgs::msg::Path & path,
    const PathProjection & projection,
    double lookahead_distance) const;
  double remaining_path_distance_from_projection(
    const nav_msgs::msg::Path & path,
    const PathProjection & projection) const;
  double remaining_path_distance(
    const Pose2D & robot,
    const nav_msgs::msg::Path & path,
    std::size_t nearest_index) const;
  double path_heading_at(const nav_msgs::msg::Path & path, std::size_t index) const;
  geometry_msgs::msg::Twist apply_limits(
    const geometry_msgs::msg::Twist & target,
    double dt);
  void publish_zero(TrackerState state, const std::string & reason);
  void publish_state(TrackerState state);
  void publish_debug(
    TrackerState state,
    const nav_msgs::msg::Path & path,
    std::size_t nearest_index,
    std::size_t lookahead_index,
    double lookahead_distance,
    double deviation,
    double remaining_distance,
    double curvature,
    double speed_scale,
    const geometry_msgs::msg::Twist & cmd);

  Params params_;
  mutable std::mutex mutex_;
  nav_msgs::msg::Path::SharedPtr latest_path_;
  nav_msgs::msg::Odometry::SharedPtr latest_odom_;
  rclcpp::Time last_path_time_;
  rclcpp::Time last_odom_time_;
  rclcpp::Time last_control_time_;
  geometry_msgs::msg::Twist last_cmd_;
  bool enabled_{true};
  TrackerState current_state_{TrackerState::kIdle};

  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr lookahead_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr debug_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_enabled_srv_;
  rclcpp::TimerBase::SharedPtr control_timer_;
};

}  // namespace trg_path_follower
