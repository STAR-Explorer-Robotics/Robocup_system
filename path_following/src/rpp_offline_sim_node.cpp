#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace trg_path_follower
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

double clamp(const double value, const double min_value, const double max_value)
{
  return std::max(min_value, std::min(value, max_value));
}

double normalize_angle(double angle)
{
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

double distance2d(
  const geometry_msgs::msg::Point & a,
  const geometry_msgs::msg::Point & b)
{
  return std::hypot(a.x - b.x, a.y - b.y);
}

geometry_msgs::msg::Quaternion yaw_to_quaternion(const double yaw)
{
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  return tf2::toMsg(q);
}

double quaternion_to_yaw(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

}  // namespace

class RppOfflineSimNode final : public rclcpp::Node
{
public:
  RppOfflineSimNode()
  : Node("rpp_offline_sim_node")
  {
    load_params();
    if (path_source_ == "generated") {
      make_path();
    } else if (path_source_ != "external") {
      throw std::runtime_error("path.source must be 'generated' or 'external'");
    } else {
      path_.header.frame_id = frame_id_;
    }
    open_outputs();

    if (path_source_ == "generated") {
      path_pub_ = create_publisher<nav_msgs::msg::Path>(path_topic_, rclcpp::QoS(1).reliable());
    } else {
      path_sub_ = create_subscription<nav_msgs::msg::Path>(
        path_topic_, rclcpp::QoS(1).reliable(),
        [this](const nav_msgs::msg::Path::SharedPtr msg) {
          if (msg->poses.size() < 2U) {
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000, "Ignoring external path with fewer than 2 poses");
            return;
          }
          path_ = *msg;
          tracker_reported_goal_ = false;
        });
    }
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, rclcpp::QoS(20).reliable());
    trajectory_pub_ = create_publisher<nav_msgs::msg::Path>(
      trajectory_topic_, rclcpp::QoS(1).reliable().transient_local());
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, rclcpp::QoS(10),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        commanded_cmd_ = *msg;
        last_cmd_time_ = now();
      });
    initialpose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      initialpose_topic_, rclcpp::QoS(10),
      std::bind(&RppOfflineSimNode::on_initialpose, this, std::placeholders::_1));
    state_sub_ = create_subscription<std_msgs::msg::Int32>(
      state_topic_, rclcpp::QoS(10),
      [this](const std_msgs::msg::Int32::SharedPtr msg) {
        tracker_state_ = msg->data;
        if (tracker_state_ == 2) {
          tracker_reported_goal_ = true;
        }
      });
    debug_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      debug_topic_, rclcpp::QoS(10),
      [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        if (msg->data.size() >= 11U) {
          last_debug_deviation_ = msg->data[4];
          last_debug_remaining_ = msg->data[5];
          last_debug_speed_scale_ = msg->data[7];
        }
      });
    set_enabled_client_ = create_client<std_srvs::srv::SetBool>(set_enabled_service_);

    x_ = initial_x_;
    y_ = initial_y_;
    yaw_ = initial_yaw_;
    odom_x_ = x_;
    odom_y_ = y_;
    odom_yaw_ = yaw_;
    rng_.seed(static_cast<std::mt19937::result_type>(noise_seed_));
    last_cmd_time_ = now();
    last_path_publish_time_ = -std::numeric_limits<double>::infinity();
    last_trajectory_publish_time_ = -std::numeric_limits<double>::infinity();
    trajectory_path_.header.frame_id = frame_id_;

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, rate_hz_));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&RppOfflineSimNode::on_timer, this));

    RCLCPP_INFO(
      get_logger(),
      "RPP offline sim started: path_source=%s scenario=%s output_dir=%s duration=%.1fs",
      path_source_.c_str(), scenario_.c_str(), output_dir_.c_str(), max_duration_sec_);
  }

private:
  void load_params()
  {
    path_topic_ = declare_parameter<std::string>("topics.path", path_topic_);
    odom_topic_ = declare_parameter<std::string>("topics.odom", odom_topic_);
    cmd_vel_topic_ = declare_parameter<std::string>("topics.cmd_vel", cmd_vel_topic_);
    trajectory_topic_ = declare_parameter<std::string>("topics.trajectory", trajectory_topic_);
    initialpose_topic_ = declare_parameter<std::string>("topics.initialpose", initialpose_topic_);
    state_topic_ = declare_parameter<std::string>("topics.state", state_topic_);
    debug_topic_ = declare_parameter<std::string>("topics.debug", debug_topic_);
    set_enabled_service_ =
      declare_parameter<std::string>("services.set_enabled", set_enabled_service_);

    rate_hz_ = declare_parameter<double>("sim.rate_hz", rate_hz_);
    max_duration_sec_ = declare_parameter<double>("sim.max_duration_sec", max_duration_sec_);
    output_dir_ = declare_parameter<std::string>("sim.output_dir", output_dir_);
    stop_on_goal_ = declare_parameter<bool>("sim.stop_on_goal", stop_on_goal_);
    path_publish_period_sec_ =
      declare_parameter<double>("sim.path_publish_period_sec", path_publish_period_sec_);
    trajectory_publish_period_sec_ = declare_parameter<double>(
      "sim.trajectory_publish_period_sec", trajectory_publish_period_sec_);
    enable_follower_ = declare_parameter<bool>("sim.enable_follower", enable_follower_);
    command_timeout_sec_ =
      declare_parameter<double>("sim.command_timeout_sec", command_timeout_sec_);

    initial_x_ = declare_parameter<double>("initial_pose.x", initial_x_);
    initial_y_ = declare_parameter<double>("initial_pose.y", initial_y_);
    initial_yaw_ = declare_parameter<double>("initial_pose.yaw", initial_yaw_);

    path_source_ = declare_parameter<std::string>("path.source", path_source_);
    scenario_ = declare_parameter<std::string>("path.scenario", scenario_);
    frame_id_ = declare_parameter<std::string>("path.frame_id", frame_id_);
    point_spacing_ = declare_parameter<double>("path.point_spacing", point_spacing_);
    path_length_ = declare_parameter<double>("path.length", path_length_);
    path_amplitude_ = declare_parameter<double>("path.amplitude", path_amplitude_);
    path_wavelength_ = declare_parameter<double>("path.wavelength", path_wavelength_);
    waypoints_xy_ = declare_parameter<std::vector<double>>("path.waypoints_xy", waypoints_xy_);

    response_time_constant_sec_ = declare_parameter<double>(
      "actuation.response_time_constant_sec", response_time_constant_sec_);
    max_forward_ = declare_parameter<double>("actuation.max_forward", max_forward_);
    max_lateral_ = declare_parameter<double>("actuation.max_lateral", max_lateral_);
    max_yaw_rate_ = declare_parameter<double>("actuation.max_yaw_rate", max_yaw_rate_);
    forward_scale_ = declare_parameter<double>("actuation.forward_scale", forward_scale_);
    lateral_scale_ = declare_parameter<double>("actuation.lateral_scale", lateral_scale_);
    yaw_rate_scale_ = declare_parameter<double>("actuation.yaw_rate_scale", yaw_rate_scale_);

    noise_enabled_ = declare_parameter<bool>("noise.enabled", noise_enabled_);
    noise_seed_ = declare_parameter<int>("noise.seed", noise_seed_);
    command_scale_stddev_ =
      declare_parameter<double>("noise.command_scale_stddev", command_scale_stddev_);
    body_velocity_stddev_x_ =
      declare_parameter<double>("noise.body_velocity_stddev_x", body_velocity_stddev_x_);
    body_velocity_stddev_y_ =
      declare_parameter<double>("noise.body_velocity_stddev_y", body_velocity_stddev_y_);
    yaw_rate_stddev_ = declare_parameter<double>("noise.yaw_rate_stddev", yaw_rate_stddev_);
    odom_position_stddev_ =
      declare_parameter<double>("noise.odom_position_stddev", odom_position_stddev_);
    odom_yaw_stddev_ = declare_parameter<double>("noise.odom_yaw_stddev", odom_yaw_stddev_);
    odom_velocity_stddev_ =
      declare_parameter<double>("noise.odom_velocity_stddev", odom_velocity_stddev_);

    terrain_enabled_ = declare_parameter<bool>("terrain.enabled", terrain_enabled_);
    terrain_world_drift_x_ =
      declare_parameter<double>("terrain.world_drift_x", terrain_world_drift_x_);
    terrain_world_drift_y_ =
      declare_parameter<double>("terrain.world_drift_y", terrain_world_drift_y_);
    terrain_yaw_drift_rate_ =
      declare_parameter<double>("terrain.yaw_drift_rate", terrain_yaw_drift_rate_);
    terrain_min_x_ = declare_parameter<double>("terrain.min_x", terrain_min_x_);
    terrain_max_x_ = declare_parameter<double>("terrain.max_x", terrain_max_x_);
    terrain_min_y_ = declare_parameter<double>("terrain.min_y", terrain_min_y_);
    terrain_max_y_ = declare_parameter<double>("terrain.max_y", terrain_max_y_);

    goal_tolerance_m_ = declare_parameter<double>("goal.tolerance_m", goal_tolerance_m_);
  }

  void on_initialpose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    reset_sim_pose(
      msg->pose.pose.position.x,
      msg->pose.pose.position.y,
      quaternion_to_yaw(msg->pose.pose.orientation),
      true);
  }

  void reset_sim_pose(const double x, const double y, const double yaw, const bool reset_run)
  {
    x_ = x;
    y_ = y;
    yaw_ = normalize_angle(yaw);
    odom_x_ = x_;
    odom_y_ = y_;
    odom_yaw_ = yaw_;
    commanded_cmd_ = geometry_msgs::msg::Twist();
    actual_cmd_ = geometry_msgs::msg::Twist();
    integrated_cmd_ = geometry_msgs::msg::Twist();
    odom_twist_ = geometry_msgs::msg::Twist();
    last_world_disturbance_x_ = 0.0;
    last_world_disturbance_y_ = 0.0;
    last_yaw_disturbance_ = 0.0;
    last_cmd_time_ = now();
    tracker_reported_goal_ = false;

    if (reset_run) {
      sim_time_ = 0.0;
      last_path_publish_time_ = -std::numeric_limits<double>::infinity();
      last_trajectory_publish_time_ = -std::numeric_limits<double>::infinity();
      trajectory_path_.poses.clear();
      max_deviation_ = 0.0;
      sum_deviation_ = 0.0;
      sample_count_ = 0U;
      max_abs_cmd_x_ = 0.0;
      max_abs_cmd_y_ = 0.0;
      max_abs_cmd_yaw_ = 0.0;
    }

    publish_odom();
    publish_trajectory();
    RCLCPP_INFO(
      get_logger(),
      "Offline sim pose reset from %s: x=%.3f y=%.3f yaw=%.3f",
      initialpose_topic_.c_str(), x_, y_, yaw_);
  }

  void make_path()
  {
    path_.header.frame_id = frame_id_;

    std::vector<std::pair<double, double>> points;
    if (waypoints_xy_.size() >= 4U && waypoints_xy_.size() % 2U == 0U) {
      for (std::size_t i = 0U; i < waypoints_xy_.size(); i += 2U) {
        points.emplace_back(waypoints_xy_[i], waypoints_xy_[i + 1U]);
      }
      make_piecewise_linear_path(points);
      return;
    }

    if (scenario_ == "straight") {
      points = {{0.0, 0.0}, {path_length_, 0.0}};
      make_piecewise_linear_path(points);
    } else if (scenario_ == "turn") {
      points = {{0.0, 0.0}, {0.5 * path_length_, 0.0}, {0.5 * path_length_, 0.5 * path_length_}};
      make_piecewise_linear_path(points);
    } else if (scenario_ == "short") {
      points = {{0.0, 0.0}, {1.0, 0.0}};
      make_piecewise_linear_path(points);
    } else {
      make_s_curve_path();
    }

    if (path_.poses.size() < 2U) {
      throw std::runtime_error("offline sim path requires at least two poses");
    }
  }

  void make_piecewise_linear_path(const std::vector<std::pair<double, double>> & points)
  {
    path_.poses.clear();
    const double spacing = std::max(0.05, point_spacing_);
    for (std::size_t i = 1U; i < points.size(); ++i) {
      const auto [x0, y0] = points[i - 1U];
      const auto [x1, y1] = points[i];
      const double dx = x1 - x0;
      const double dy = y1 - y0;
      const double length = std::hypot(dx, dy);
      const int steps = std::max(1, static_cast<int>(std::ceil(length / spacing)));
      for (int step = 0; step <= steps; ++step) {
        if (i > 1U && step == 0) {
          continue;
        }
        const double ratio = static_cast<double>(step) / static_cast<double>(steps);
        append_path_pose(x0 + ratio * dx, y0 + ratio * dy, std::atan2(dy, dx));
      }
    }
  }

  void make_s_curve_path()
  {
    path_.poses.clear();
    const double spacing = std::max(0.05, point_spacing_);
    const int steps = std::max(2, static_cast<int>(std::ceil(path_length_ / spacing)));
    const double wavelength = std::max(0.5, path_wavelength_);
    for (int i = 0; i <= steps; ++i) {
      const double x = path_length_ * static_cast<double>(i) / static_cast<double>(steps);
      const double y = path_amplitude_ * std::sin(2.0 * kPi * x / wavelength);
      const double dydx =
        path_amplitude_ * (2.0 * kPi / wavelength) * std::cos(2.0 * kPi * x / wavelength);
      append_path_pose(x, y, std::atan2(dydx, 1.0));
    }
  }

  void append_path_pose(const double x, const double y, const double yaw)
  {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = frame_id_;
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.orientation = yaw_to_quaternion(yaw);
    path_.poses.push_back(pose);
  }

  void open_outputs()
  {
    std::filesystem::create_directories(output_dir_);
    trajectory_.open(std::filesystem::path(output_dir_) / "trajectory.csv", std::ios::trunc);
    trajectory_ <<
      "sim_time,x,y,yaw,odom_x,odom_y,odom_yaw,cmd_x,cmd_y,cmd_yaw,"
      "actual_x,actual_y,actual_yaw,integrated_x,integrated_y,integrated_yaw,"
      "disturbance_x,disturbance_y,disturbance_yaw,"
      "tracker_state,path_deviation,debug_deviation,debug_remaining,debug_speed_scale\n";
  }

  void on_timer()
  {
    const double dt = 1.0 / std::max(1.0, rate_hz_);
    sim_time_ += dt;

    if (enable_follower_ && !enable_request_sent_ && sim_time_ >= 0.25) {
      enable_request_sent_ = true;
      if (set_enabled_client_->wait_for_service(std::chrono::milliseconds(20))) {
        auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
        request->data = true;
        set_enabled_client_->async_send_request(request);
      } else {
        RCLCPP_WARN(get_logger(), "Follower enable service not available yet");
      }
    }

    if (path_source_ == "generated" &&
      sim_time_ - last_path_publish_time_ >= path_publish_period_sec_)
    {
      path_.header.stamp = now();
      for (auto & pose : path_.poses) {
        pose.header.stamp = path_.header.stamp;
      }
      path_pub_->publish(path_);
      last_path_publish_time_ = sim_time_;
    }

    if ((now() - last_cmd_time_).seconds() > command_timeout_sec_) {
      commanded_cmd_ = geometry_msgs::msg::Twist();
    }

    commanded_cmd_.linear.x = clamp(commanded_cmd_.linear.x, -max_forward_, max_forward_);
    commanded_cmd_.linear.y = clamp(commanded_cmd_.linear.y, -max_lateral_, max_lateral_);
    commanded_cmd_.angular.z = clamp(commanded_cmd_.angular.z, -max_yaw_rate_, max_yaw_rate_);

    geometry_msgs::msg::Twist target_cmd = commanded_cmd_;
    target_cmd.linear.x = clamp(
      target_cmd.linear.x * command_axis_scale(forward_scale_),
      -max_forward_, max_forward_);
    target_cmd.linear.y = clamp(
      target_cmd.linear.y * command_axis_scale(lateral_scale_),
      -max_lateral_, max_lateral_);
    target_cmd.angular.z = clamp(
      target_cmd.angular.z * command_axis_scale(yaw_rate_scale_),
      -max_yaw_rate_, max_yaw_rate_);

    const double tau = std::max(0.0, response_time_constant_sec_);
    const double alpha = tau <= 1e-6 ? 1.0 : clamp(dt / tau, 0.0, 1.0);
    actual_cmd_.linear.x += alpha * (target_cmd.linear.x - actual_cmd_.linear.x);
    actual_cmd_.linear.y += alpha * (target_cmd.linear.y - actual_cmd_.linear.y);
    actual_cmd_.angular.z += alpha * (target_cmd.angular.z - actual_cmd_.angular.z);

    integrated_cmd_ = actual_cmd_;
    integrated_cmd_.linear.x += sample_noise(body_velocity_stddev_x_);
    integrated_cmd_.linear.y += sample_noise(body_velocity_stddev_y_);
    integrated_cmd_.angular.z += sample_noise(yaw_rate_stddev_);

    last_world_disturbance_x_ = 0.0;
    last_world_disturbance_y_ = 0.0;
    last_yaw_disturbance_ = 0.0;
    if (terrain_is_active()) {
      last_world_disturbance_x_ = terrain_world_drift_x_;
      last_world_disturbance_y_ = terrain_world_drift_y_;
      last_yaw_disturbance_ = terrain_yaw_drift_rate_;
    }

    const double cos_yaw = std::cos(yaw_);
    const double sin_yaw = std::sin(yaw_);
    x_ += (
      cos_yaw * integrated_cmd_.linear.x - sin_yaw * integrated_cmd_.linear.y +
      last_world_disturbance_x_) * dt;
    y_ += (
      sin_yaw * integrated_cmd_.linear.x + cos_yaw * integrated_cmd_.linear.y +
      last_world_disturbance_y_) * dt;
    yaw_ = normalize_angle(
      yaw_ + (integrated_cmd_.angular.z + last_yaw_disturbance_) * dt);

    publish_odom();
    publish_trajectory();
    update_metrics();
    write_trajectory_row();

    const std::string completion_reason = get_completion_reason();
    const bool complete = completion_reason != "running";
    if (complete) {
      const bool goal_reached = completion_reason != "timeout";
      write_summary(goal_reached, completion_reason);
      RCLCPP_INFO(
        get_logger(),
        "Offline sim finished: result=%s reason=%s duration=%.2fs max_deviation=%.3fm output=%s",
        goal_reached ? "goal_reached" : "timeout", completion_reason.c_str(), sim_time_,
        max_deviation_, output_dir_.c_str());
      rclcpp::shutdown();
    }
  }

  void publish_odom()
  {
    odom_x_ = x_ + sample_noise(odom_position_stddev_);
    odom_y_ = y_ + sample_noise(odom_position_stddev_);
    odom_yaw_ = normalize_angle(yaw_ + sample_noise(odom_yaw_stddev_));
    odom_twist_ = actual_cmd_;
    odom_twist_.linear.x += sample_noise(odom_velocity_stddev_);
    odom_twist_.linear.y += sample_noise(odom_velocity_stddev_);
    odom_twist_.angular.z += sample_noise(odom_velocity_stddev_);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = now();
    odom.header.frame_id = frame_id_;
    odom.child_frame_id = "base_link";
    odom.pose.pose.position.x = odom_x_;
    odom.pose.pose.position.y = odom_y_;
    odom.pose.pose.orientation = yaw_to_quaternion(odom_yaw_);
    odom.twist.twist = odom_twist_;
    if (noise_enabled_) {
      const double position_var = odom_position_stddev_ * odom_position_stddev_;
      const double yaw_var = odom_yaw_stddev_ * odom_yaw_stddev_;
      odom.pose.covariance[0] = position_var;
      odom.pose.covariance[7] = position_var;
      odom.pose.covariance[35] = yaw_var;
      const double velocity_var = odom_velocity_stddev_ * odom_velocity_stddev_;
      odom.twist.covariance[0] = velocity_var;
      odom.twist.covariance[7] = velocity_var;
      odom.twist.covariance[35] = velocity_var;
    }
    odom_pub_->publish(odom);
  }

  void publish_trajectory()
  {
    if (sim_time_ - last_trajectory_publish_time_ < trajectory_publish_period_sec_) {
      return;
    }

    trajectory_path_.header.stamp = now();
    trajectory_path_.header.frame_id = frame_id_;

    geometry_msgs::msg::PoseStamped pose;
    pose.header = trajectory_path_.header;
    pose.pose.position.x = x_;
    pose.pose.position.y = y_;
    pose.pose.orientation = yaw_to_quaternion(yaw_);
    trajectory_path_.poses.push_back(pose);

    trajectory_pub_->publish(trajectory_path_);
    last_trajectory_publish_time_ = sim_time_;
  }

  double distance_to_path() const
  {
    if (path_.poses.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    geometry_msgs::msg::Point robot;
    robot.x = x_;
    robot.y = y_;

    double best = std::numeric_limits<double>::infinity();
    for (const auto & pose : path_.poses) {
      best = std::min(best, distance2d(robot, pose.pose.position));
    }
    return best;
  }

  double distance_to_final_path_pose() const
  {
    if (path_.poses.empty()) {
      return std::numeric_limits<double>::infinity();
    }

    geometry_msgs::msg::Point robot;
    robot.x = x_;
    robot.y = y_;
    return distance2d(robot, path_.poses.back().pose.position);
  }

  void update_metrics()
  {
    const double deviation = distance_to_path();
    if (std::isfinite(deviation)) {
      max_deviation_ = std::max(max_deviation_, deviation);
      sum_deviation_ += deviation;
      ++sample_count_;
    }
    max_abs_cmd_x_ = std::max(max_abs_cmd_x_, std::abs(commanded_cmd_.linear.x));
    max_abs_cmd_y_ = std::max(max_abs_cmd_y_, std::abs(commanded_cmd_.linear.y));
    max_abs_cmd_yaw_ = std::max(max_abs_cmd_yaw_, std::abs(commanded_cmd_.angular.z));
  }

  void write_trajectory_row()
  {
    if (!trajectory_.is_open()) {
      return;
    }
    trajectory_ << sim_time_ << ',' << x_ << ',' << y_ << ',' << yaw_ << ','
                << odom_x_ << ',' << odom_y_ << ',' << odom_yaw_ << ','
                << commanded_cmd_.linear.x << ',' << commanded_cmd_.linear.y << ','
                << commanded_cmd_.angular.z << ',' << actual_cmd_.linear.x << ','
                << actual_cmd_.linear.y << ',' << actual_cmd_.angular.z << ','
                << integrated_cmd_.linear.x << ',' << integrated_cmd_.linear.y << ','
                << integrated_cmd_.angular.z << ',' << last_world_disturbance_x_ << ','
                << last_world_disturbance_y_ << ',' << last_yaw_disturbance_ << ','
                << tracker_state_ << ',' << distance_to_path() << ','
                << last_debug_deviation_ << ',' << last_debug_remaining_ << ','
                << last_debug_speed_scale_ << '\n';
  }

  std::string get_completion_reason() const
  {
    if (stop_on_goal_ && tracker_reported_goal_) {
      return "tracker_goal";
    }
    if (stop_on_goal_ && distance_to_final_path_pose() <= goal_tolerance_m_) {
      return "final_tolerance";
    }
    if (sim_time_ >= max_duration_sec_) {
      return "timeout";
    }
    return "running";
  }

  void write_summary(const bool goal_reached, const std::string & completion_reason)
  {
    if (summary_written_) {
      return;
    }
    summary_written_ = true;
    if (trajectory_.is_open()) {
      trajectory_.flush();
    }
    std::ofstream summary(std::filesystem::path(output_dir_) / "summary.txt", std::ios::trunc);
    const double mean_deviation = sample_count_ == 0U ? 0.0 : sum_deviation_ / sample_count_;
    summary << "result=" << (goal_reached ? "goal_reached" : "timeout") << '\n'
            << "completion_reason=" << completion_reason << '\n'
            << "path_source=" << path_source_ << '\n'
            << "path_available=" << (!path_.poses.empty() ? "true" : "false") << '\n'
            << "trajectory_topic=" << trajectory_topic_ << '\n'
            << "trajectory_points=" << trajectory_path_.poses.size() << '\n'
            << "scenario=" << scenario_ << '\n'
            << "duration_sec=" << sim_time_ << '\n'
            << "path_points=" << path_.poses.size() << '\n'
            << "final_distance_m=" << distance_to_final_path_pose() << '\n'
            << "max_deviation_m=" << max_deviation_ << '\n'
            << "mean_deviation_m=" << mean_deviation << '\n'
            << "max_abs_cmd_x=" << max_abs_cmd_x_ << '\n'
            << "max_abs_cmd_y=" << max_abs_cmd_y_ << '\n'
            << "max_abs_cmd_yaw=" << max_abs_cmd_yaw_ << '\n'
            << "actuation_forward_scale=" << forward_scale_ << '\n'
            << "actuation_lateral_scale=" << lateral_scale_ << '\n'
            << "actuation_yaw_rate_scale=" << yaw_rate_scale_ << '\n'
            << "noise_enabled=" << (noise_enabled_ ? "true" : "false") << '\n'
            << "noise_seed=" << noise_seed_ << '\n'
            << "noise_command_scale_stddev=" << command_scale_stddev_ << '\n'
            << "noise_body_velocity_stddev_x=" << body_velocity_stddev_x_ << '\n'
            << "noise_body_velocity_stddev_y=" << body_velocity_stddev_y_ << '\n'
            << "noise_yaw_rate_stddev=" << yaw_rate_stddev_ << '\n'
            << "noise_odom_position_stddev=" << odom_position_stddev_ << '\n'
            << "noise_odom_yaw_stddev=" << odom_yaw_stddev_ << '\n'
            << "noise_odom_velocity_stddev=" << odom_velocity_stddev_ << '\n'
            << "terrain_enabled=" << (terrain_enabled_ ? "true" : "false") << '\n'
            << "terrain_world_drift_x=" << terrain_world_drift_x_ << '\n'
            << "terrain_world_drift_y=" << terrain_world_drift_y_ << '\n'
            << "terrain_yaw_drift_rate=" << terrain_yaw_drift_rate_ << '\n'
            << "tracker_reported_goal=" << (tracker_reported_goal_ ? "true" : "false") << '\n'
            << "tracker_state=" << tracker_state_ << '\n';
  }

  double sample_noise(const double stddev)
  {
    if (!noise_enabled_ || stddev <= 0.0) {
      return 0.0;
    }
    return normal_dist_(rng_) * stddev;
  }

  double command_axis_scale(const double nominal_scale)
  {
    if (!noise_enabled_ || command_scale_stddev_ <= 0.0) {
      return nominal_scale;
    }
    return std::max(0.0, nominal_scale + sample_noise(command_scale_stddev_));
  }

  bool terrain_is_active() const
  {
    if (!terrain_enabled_) {
      return false;
    }
    return x_ >= terrain_min_x_ && x_ <= terrain_max_x_ &&
           y_ >= terrain_min_y_ && y_ <= terrain_max_y_;
  }

  std::string path_topic_{"/trg/output/path"};
  std::string odom_topic_{"/laser_odometry"};
  std::string cmd_vel_topic_{"/cmd_vel"};
  std::string trajectory_topic_{"/rpp/offline_sim/trajectory"};
  std::string initialpose_topic_{"/initialpose"};
  std::string state_topic_{"/rpp/state"};
  std::string debug_topic_{"/rpp/debug"};
  std::string set_enabled_service_{"/rpp/set_enabled"};
  std::string output_dir_{"/tmp/trg_path_follower_offline_sim"};
  std::string path_source_{"generated"};
  std::string scenario_{"s_curve"};
  std::string frame_id_{"map"};

  double rate_hz_{50.0};
  double max_duration_sec_{45.0};
  double path_publish_period_sec_{0.5};
  double trajectory_publish_period_sec_{0.2};
  double command_timeout_sec_{0.5};
  double initial_x_{0.0};
  double initial_y_{-0.3};
  double initial_yaw_{0.0};
  double point_spacing_{0.15};
  double path_length_{8.0};
  double path_amplitude_{0.8};
  double path_wavelength_{4.0};
  double response_time_constant_sec_{0.12};
  double max_forward_{0.8};
  double max_lateral_{0.5};
  double max_yaw_rate_{1.0};
  double forward_scale_{1.0};
  double lateral_scale_{1.0};
  double yaw_rate_scale_{1.0};
  double command_scale_stddev_{0.0};
  double body_velocity_stddev_x_{0.0};
  double body_velocity_stddev_y_{0.0};
  double yaw_rate_stddev_{0.0};
  double odom_position_stddev_{0.0};
  double odom_yaw_stddev_{0.0};
  double odom_velocity_stddev_{0.0};
  double terrain_world_drift_x_{0.0};
  double terrain_world_drift_y_{0.0};
  double terrain_yaw_drift_rate_{0.0};
  double terrain_min_x_{-1.0e9};
  double terrain_max_x_{1.0e9};
  double terrain_min_y_{-1.0e9};
  double terrain_max_y_{1.0e9};
  double goal_tolerance_m_{0.25};
  bool stop_on_goal_{true};
  bool enable_follower_{true};
  bool noise_enabled_{false};
  bool terrain_enabled_{false};
  int noise_seed_{7};
  std::vector<double> waypoints_xy_;

  nav_msgs::msg::Path path_;
  geometry_msgs::msg::Twist commanded_cmd_;
  geometry_msgs::msg::Twist actual_cmd_;
  geometry_msgs::msg::Twist integrated_cmd_;
  geometry_msgs::msg::Twist odom_twist_;
  double x_{0.0};
  double y_{0.0};
  double yaw_{0.0};
  double odom_x_{0.0};
  double odom_y_{0.0};
  double odom_yaw_{0.0};
  double last_world_disturbance_x_{0.0};
  double last_world_disturbance_y_{0.0};
  double last_yaw_disturbance_{0.0};
  double sim_time_{0.0};
  double last_path_publish_time_{0.0};
  double last_trajectory_publish_time_{0.0};
  rclcpp::Time last_cmd_time_;
  bool enable_request_sent_{false};
  bool tracker_reported_goal_{false};
  bool summary_written_{false};
  int tracker_state_{0};
  double last_debug_deviation_{0.0};
  double last_debug_remaining_{0.0};
  double last_debug_speed_scale_{0.0};
  double max_deviation_{0.0};
  double sum_deviation_{0.0};
  std::size_t sample_count_{0U};
  double max_abs_cmd_x_{0.0};
  double max_abs_cmd_y_{0.0};
  double max_abs_cmd_yaw_{0.0};
  std::ofstream trajectory_;
  std::mt19937 rng_;
  std::normal_distribution<double> normal_dist_{0.0, 1.0};

  nav_msgs::msg::Path trajectory_path_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr state_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr debug_sub_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr set_enabled_client_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace trg_path_follower

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<trg_path_follower::RppOfflineSimNode>());
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
