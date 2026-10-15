#include "trg_path_follower/rpp_follower_node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace trg_path_follower
{
namespace
{

Point2D point_from_pose(const geometry_msgs::msg::PoseStamped & pose)
{
  return Point2D{pose.pose.position.x, pose.pose.position.y};
}

Pose2D pose_from_odom(const nav_msgs::msg::Odometry & odom)
{
  return Pose2D{
    odom.pose.pose.position.x,
    odom.pose.pose.position.y,
    tf2::getYaw(odom.pose.pose.orientation)};
}

double seconds_since(const rclcpp::Time & now, const rclcpp::Time & stamp)
{
  if (stamp.nanoseconds() == 0) {
    return std::numeric_limits<double>::infinity();
  }
  return (now - stamp).seconds();
}

}  // namespace

RegulatedPurePursuitNode::RegulatedPurePursuitNode(const rclcpp::NodeOptions & options)
: Node("rpp_follower_node", options)
{
  declare_and_load_params();
  enabled_ = params_.enabled_on_start;

  path_sub_ = create_subscription<nav_msgs::msg::Path>(
    params_.path_topic, rclcpp::QoS(1).reliable(),
    std::bind(&RegulatedPurePursuitNode::on_path, this, std::placeholders::_1));
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    params_.odom_topic, rclcpp::QoS(20).reliable(),
    std::bind(&RegulatedPurePursuitNode::on_odom, this, std::placeholders::_1));
  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(params_.cmd_vel_topic, rclcpp::QoS(10));
  lookahead_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
    params_.lookahead_topic, rclcpp::QoS(10));
  debug_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
    params_.debug_topic, rclcpp::QoS(10));
  state_pub_ = create_publisher<std_msgs::msg::Int32>(params_.state_topic, rclcpp::QoS(10));
  set_enabled_srv_ = create_service<std_srvs::srv::SetBool>(
    params_.set_enabled_service,
    std::bind(
      &RegulatedPurePursuitNode::on_set_enabled, this,
      std::placeholders::_1, std::placeholders::_2));

  const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, params_.control_rate_hz));
  control_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&RegulatedPurePursuitNode::on_control_timer, this));

  RCLCPP_INFO(
    get_logger(),
    "RPP follower started: path=%s odom=%s cmd=%s holonomic=%s mode=%s enabled=%s rate=%.1f Hz",
    params_.path_topic.c_str(), params_.odom_topic.c_str(), params_.cmd_vel_topic.c_str(),
    params_.holonomic ? "true" : "false", params_.tracking_mode.c_str(),
    enabled_ ? "true" : "false",
    params_.control_rate_hz);
}

void RegulatedPurePursuitNode::declare_and_load_params()
{
  auto declare_string = [this](const std::string & name, const std::string & value) {
    return declare_parameter<std::string>(name, value);
  };
  auto declare_double = [this](const std::string & name, const double value) {
    return declare_parameter<double>(name, value);
  };
  auto declare_bool = [this](const std::string & name, const bool value) {
    return declare_parameter<bool>(name, value);
  };

  params_.path_topic = declare_string("topics.path", params_.path_topic);
  params_.odom_topic = declare_string("topics.odom", params_.odom_topic);
  params_.cmd_vel_topic = declare_string("topics.cmd_vel", params_.cmd_vel_topic);
  params_.lookahead_topic = declare_string("topics.lookahead", params_.lookahead_topic);
  params_.debug_topic = declare_string("topics.debug", params_.debug_topic);
  params_.state_topic = declare_string("topics.state", params_.state_topic);
  params_.set_enabled_service =
    declare_string("services.set_enabled", params_.set_enabled_service);

  params_.control_rate_hz = declare_double("control.rate_hz", params_.control_rate_hz);
  params_.enabled_on_start = declare_bool("control.enabled_on_start", params_.enabled_on_start);
  params_.holonomic = declare_bool("control.holonomic", params_.holonomic);
  params_.publish_debug = declare_bool("control.publish_debug", params_.publish_debug);
  params_.publish_zero_on_stop =
    declare_bool("control.publish_zero_on_stop", params_.publish_zero_on_stop);

  params_.desired_speed = declare_double("speed.desired", params_.desired_speed);
  params_.min_tracking_speed = declare_double("speed.min_tracking", params_.min_tracking_speed);
  params_.max_forward_speed = declare_double("speed.max_forward", params_.max_forward_speed);
  params_.max_lateral_speed = declare_double("speed.max_lateral", params_.max_lateral_speed);
  params_.max_yaw_rate = declare_double("speed.max_yaw_rate", params_.max_yaw_rate);
  params_.accel_limit = declare_double("speed.accel_limit", params_.accel_limit);
  params_.lateral_accel_limit =
    declare_double("speed.lateral_accel_limit", params_.lateral_accel_limit);
  params_.yaw_accel_limit = declare_double("speed.yaw_accel_limit", params_.yaw_accel_limit);

  params_.lookahead_distance = declare_double("lookahead.distance", params_.lookahead_distance);
  params_.lookahead_time = declare_double("lookahead.time", params_.lookahead_time);
  params_.min_lookahead_distance =
    declare_double("lookahead.min_distance", params_.min_lookahead_distance);
  params_.max_lookahead_distance =
    declare_double("lookahead.max_distance", params_.max_lookahead_distance);

  params_.yaw_gain = declare_double("tracking.yaw_gain", params_.yaw_gain);
  params_.lateral_gain = declare_double("tracking.lateral_gain", params_.lateral_gain);
  params_.tracking_mode = declare_string("tracking.mode", params_.tracking_mode);
  params_.path_tube_gain = declare_double("tracking.path_tube_gain", params_.path_tube_gain);
  params_.path_tube_max_correction_speed = declare_double(
    "tracking.path_tube_max_correction_speed",
    params_.path_tube_max_correction_speed);
  params_.path_tube_deadband =
    declare_double("tracking.path_tube_deadband", params_.path_tube_deadband);
  params_.heading_error_slowdown_angle =
    declare_double("tracking.heading_error_slowdown_angle", params_.heading_error_slowdown_angle);
  params_.rotate_in_place_angle =
    declare_double("tracking.rotate_in_place_angle", params_.rotate_in_place_angle);

  if (params_.tracking_mode != "vector_field" && params_.tracking_mode != "pure_pursuit") {
    RCLCPP_WARN(
      get_logger(),
      "Unknown tracking.mode '%s'; falling back to vector_field",
      params_.tracking_mode.c_str());
    params_.tracking_mode = "vector_field";
  }

  params_.regulated_min_turning_radius =
    declare_double("regulation.min_turning_radius", params_.regulated_min_turning_radius);
  params_.regulated_min_speed_scale =
    declare_double("regulation.min_speed_scale", params_.regulated_min_speed_scale);
  params_.approach_slowdown_distance =
    declare_double("regulation.approach_slowdown_distance", params_.approach_slowdown_distance);
  params_.deviation_slowdown_distance =
    declare_double("regulation.deviation_slowdown_distance", params_.deviation_slowdown_distance);
  params_.deviation_stop_distance =
    declare_double("regulation.deviation_stop_distance", params_.deviation_stop_distance);
  params_.goal_tolerance = declare_double("regulation.goal_tolerance", params_.goal_tolerance);
  params_.use_final_yaw = declare_bool("goal.use_final_yaw", params_.use_final_yaw);
  params_.final_yaw_tolerance =
    declare_double("goal.yaw_tolerance", params_.final_yaw_tolerance);
  params_.final_yaw_gain = declare_double("goal.yaw_gain", params_.final_yaw_gain);

  params_.odom_timeout_sec = declare_double("watchdog.odom_timeout_sec", params_.odom_timeout_sec);
  params_.path_timeout_sec = declare_double("watchdog.path_timeout_sec", params_.path_timeout_sec);
}

void RegulatedPurePursuitNode::on_path(const nav_msgs::msg::Path::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_path_ = msg;
  last_path_time_ = now();
}

void RegulatedPurePursuitNode::on_odom(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_odom_ = msg;
  last_odom_time_ = now();
}

void RegulatedPurePursuitNode::on_set_enabled(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = request->data;
  }

  if (!request->data) {
    publish_zero(TrackerState::kIdle, "disabled by service");
  }
  response->success = true;
  response->message = request->data ? "RPP follower enabled" : "RPP follower disabled";
  RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
}

bool RegulatedPurePursuitNode::has_valid_inputs(
  const rclcpp::Time & now_time,
  std::string & reason) const
{
  if (!latest_odom_) {
    reason = "waiting for odometry";
    return false;
  }
  if (params_.odom_timeout_sec > 0.0 &&
    seconds_since(now_time, last_odom_time_) > params_.odom_timeout_sec)
  {
    reason = "odometry timeout";
    return false;
  }
  if (!latest_path_ || latest_path_->poses.size() < 2U) {
    reason = "waiting for path";
    return false;
  }
  if (params_.path_timeout_sec > 0.0 &&
    seconds_since(now_time, last_path_time_) > params_.path_timeout_sec)
  {
    reason = "path timeout";
    return false;
  }
  return true;
}

void RegulatedPurePursuitNode::on_control_timer()
{
  nav_msgs::msg::Path::SharedPtr path;
  nav_msgs::msg::Odometry::SharedPtr odom;
  rclcpp::Time current_time;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_time = now();
    if (!enabled_) {
      publish_zero(TrackerState::kIdle, "disabled");
      return;
    }
    std::string reason;
    if (!has_valid_inputs(current_time, reason)) {
      publish_zero(TrackerState::kInputStale, reason);
      return;
    }
    path = latest_path_;
    odom = latest_odom_;
  }

  const Pose2D robot = pose_from_odom(*odom);
  const PathProjection projection = project_to_path(robot, *path);
  const std::size_t nearest_index = projection.segment_index;
  const double deviation = projection.distance;

  if (deviation > params_.deviation_stop_distance) {
    publish_zero(TrackerState::kPathDeviation, "path deviation exceeded");
    publish_debug(
      TrackerState::kPathDeviation, *path, nearest_index, nearest_index, 0.0, deviation, 0.0,
      0.0, 0.0, last_cmd_);
    return;
  }

  const double remaining_along_path = remaining_path_distance_from_projection(*path, projection);
  const double remaining = std::hypot(remaining_along_path, deviation);
  if (remaining <= params_.goal_tolerance) {
    if (params_.use_final_yaw) {
      const double final_yaw = tf2::getYaw(path->poses.back().pose.orientation);
      const double final_yaw_error = normalize_angle(final_yaw - robot.yaw);
      if (std::abs(final_yaw_error) > params_.final_yaw_tolerance) {
        const double dt = last_control_time_.nanoseconds() == 0 ?
          1.0 / std::max(1.0, params_.control_rate_hz) :
          std::max(1e-3, (current_time - last_control_time_).seconds());
        geometry_msgs::msg::Twist limited_cmd;
        limited_cmd.angular.z = limit_rate(
          clamp(
            params_.final_yaw_gain * final_yaw_error,
            -params_.max_yaw_rate,
            params_.max_yaw_rate),
          last_cmd_.angular.z,
          params_.yaw_accel_limit,
          dt);
        cmd_pub_->publish(limited_cmd);
        publish_state(TrackerState::kTracking);
        last_cmd_ = limited_cmd;
        last_control_time_ = current_time;

        geometry_msgs::msg::PointStamped lookahead_msg;
        lookahead_msg.header = path->header;
        lookahead_msg.point.x = path->poses.back().pose.position.x;
        lookahead_msg.point.y = path->poses.back().pose.position.y;
        lookahead_msg.point.z = path->poses.back().pose.position.z;
        lookahead_pub_->publish(lookahead_msg);

        publish_debug(
          TrackerState::kTracking, *path, nearest_index, path->poses.size() - 1U, 0.0,
          deviation, remaining, 0.0, 0.0, limited_cmd);
        return;
      }
    }
    publish_zero(TrackerState::kGoalReached, "goal reached");
    publish_debug(
      TrackerState::kGoalReached, *path, nearest_index, path->poses.size() - 1U, 0.0, deviation,
      remaining, 0.0, 0.0, last_cmd_);
    return;
  }

  const double measured_speed = std::hypot(
    odom->twist.twist.linear.x, odom->twist.twist.linear.y);
  const double lookahead_distance = clamp(
    params_.lookahead_distance + params_.lookahead_time * measured_speed,
    params_.min_lookahead_distance, params_.max_lookahead_distance);
  const std::size_t lookahead_index = find_lookahead_index_from_projection(
    *path, projection, lookahead_distance);
  const Point2D target_world = point_from_pose(path->poses[lookahead_index]);
  const Point2D target_body = transform_to_body(robot, target_world);
  const double curvature = pure_pursuit_curvature(target_body);
  const bool use_vector_field = params_.holonomic && params_.tracking_mode == "vector_field";

  double speed_scale = 1.0;
  const double abs_curvature = std::abs(curvature);
  if (abs_curvature > 1e-4 && params_.regulated_min_turning_radius > 0.0) {
    const double radius = 1.0 / abs_curvature;
    if (radius < params_.regulated_min_turning_radius) {
      speed_scale *= clamp(
        radius / params_.regulated_min_turning_radius,
        params_.regulated_min_speed_scale, 1.0);
    }
  }

  if (params_.approach_slowdown_distance > 0.0 &&
    remaining < params_.approach_slowdown_distance)
  {
    speed_scale *= clamp(
      remaining / params_.approach_slowdown_distance,
      params_.regulated_min_speed_scale, 1.0);
  }

  if (deviation > params_.deviation_slowdown_distance) {
    const double range = std::max(
      1e-3, params_.deviation_stop_distance - params_.deviation_slowdown_distance);
    speed_scale *= clamp(
      (params_.deviation_stop_distance - deviation) / range,
      params_.regulated_min_speed_scale, 1.0);
  }

  const double heading = use_vector_field ?
    std::atan2(projection.tangent.y, projection.tangent.x) :
    path_heading_at(*path, lookahead_index);
  const double yaw_error = normalize_angle(heading - robot.yaw);
  if (std::abs(yaw_error) > params_.heading_error_slowdown_angle) {
    const double range = std::max(1e-3, params_.rotate_in_place_angle -
      params_.heading_error_slowdown_angle);
    speed_scale *= clamp(
      1.0 - (std::abs(yaw_error) - params_.heading_error_slowdown_angle) / range,
      params_.regulated_min_speed_scale, 1.0);
  }

  const double regulated_speed = clamp(
    params_.desired_speed * speed_scale,
    params_.min_tracking_speed,
    params_.max_forward_speed);
  const double target_distance = std::max(1e-3, std::hypot(target_body.x, target_body.y));
  const double bearing = std::atan2(target_body.y, target_body.x);

  geometry_msgs::msg::Twist target_cmd;
  if (use_vector_field) {
    double path_error = projection.signed_error;
    if (std::abs(path_error) <= params_.path_tube_deadband) {
      path_error = 0.0;
    } else {
      path_error -= std::copysign(params_.path_tube_deadband, path_error);
    }
    const double correction_speed = clamp(
      -params_.path_tube_gain * path_error,
      -params_.path_tube_max_correction_speed,
      params_.path_tube_max_correction_speed);
    const Point2D cmd_world{
      regulated_speed * projection.tangent.x + correction_speed * projection.normal.x,
      regulated_speed * projection.tangent.y + correction_speed * projection.normal.y};
    const Point2D cmd_body = transform_vector_to_body(robot, cmd_world);
    target_cmd.linear.x = cmd_body.x;
    target_cmd.linear.y = cmd_body.y;
  } else if (params_.holonomic) {
    const double forward_ratio = clamp(std::cos(bearing), 0.0, 1.0);
    target_cmd.linear.x = regulated_speed * forward_ratio;
    target_cmd.linear.y = clamp(
      params_.lateral_gain * regulated_speed * target_body.y / target_distance,
      -params_.max_lateral_speed, params_.max_lateral_speed);
  } else {
    target_cmd.linear.x = regulated_speed;
    target_cmd.linear.y = 0.0;
  }

  if (std::abs(yaw_error) > params_.rotate_in_place_angle) {
    target_cmd.linear.x = 0.0;
    target_cmd.linear.y = 0.0;
  }

  if (!params_.holonomic) {
    target_cmd.angular.z = clamp(
      target_cmd.linear.x * curvature, -params_.max_yaw_rate, params_.max_yaw_rate);
  } else {
    target_cmd.angular.z = clamp(
      params_.yaw_gain * yaw_error, -params_.max_yaw_rate, params_.max_yaw_rate);
  }

  const double dt = last_control_time_.nanoseconds() == 0 ?
    1.0 / std::max(1.0, params_.control_rate_hz) :
    std::max(1e-3, (current_time - last_control_time_).seconds());
  geometry_msgs::msg::Twist limited_cmd = apply_limits(target_cmd, dt);
  cmd_pub_->publish(limited_cmd);
  publish_state(TrackerState::kTracking);
  last_cmd_ = limited_cmd;
  last_control_time_ = current_time;

  geometry_msgs::msg::PointStamped lookahead_msg;
  lookahead_msg.header = path->header;
  lookahead_msg.point.x = target_world.x;
  lookahead_msg.point.y = target_world.y;
  lookahead_msg.point.z = path->poses[lookahead_index].pose.position.z;
  lookahead_pub_->publish(lookahead_msg);

  publish_debug(
    TrackerState::kTracking, *path, nearest_index, lookahead_index, lookahead_distance,
    deviation, remaining, curvature, speed_scale, limited_cmd);
}

std::size_t RegulatedPurePursuitNode::find_nearest_index(
  const Pose2D & robot,
  const nav_msgs::msg::Path & path) const
{
  std::size_t best_index = 0U;
  double best_distance = std::numeric_limits<double>::infinity();
  const Point2D robot_point{robot.x, robot.y};
  for (std::size_t i = 0U; i < path.poses.size(); ++i) {
    const double dist = distance2d(robot_point, point_from_pose(path.poses[i]));
    if (dist < best_distance) {
      best_distance = dist;
      best_index = i;
    }
  }
  return best_index;
}

std::size_t RegulatedPurePursuitNode::find_lookahead_index(
  const nav_msgs::msg::Path & path,
  const std::size_t start_index,
  const double lookahead_distance) const
{
  if (path.poses.empty()) {
    return 0U;
  }
  double accumulated = 0.0;
  for (std::size_t i = start_index + 1U; i < path.poses.size(); ++i) {
    accumulated += distance2d(point_from_pose(path.poses[i - 1U]), point_from_pose(path.poses[i]));
    if (accumulated >= lookahead_distance) {
      return i;
    }
  }
  return path.poses.size() - 1U;
}

RegulatedPurePursuitNode::PathProjection RegulatedPurePursuitNode::project_to_path(
  const Pose2D & robot,
  const nav_msgs::msg::Path & path) const
{
  PathProjection best;
  best.distance = std::numeric_limits<double>::infinity();
  const Point2D robot_point{robot.x, robot.y};

  for (std::size_t i = 0U; i + 1U < path.poses.size(); ++i) {
    const SegmentProjection projection = project_point_to_segment(
      robot_point,
      point_from_pose(path.poses[i]),
      point_from_pose(path.poses[i + 1U]));
    if (projection.distance < best.distance) {
      best.segment_index = i;
      best.segment_ratio = projection.ratio;
      best.point = projection.point;
      best.tangent = projection.tangent;
      best.normal = projection.normal;
      best.signed_error = projection.signed_error;
      best.distance = projection.distance;
    }
  }

  if (!std::isfinite(best.distance)) {
    best.point = point_from_pose(path.poses.front());
    best.distance = distance2d(robot_point, best.point);
  }
  return best;
}

std::size_t RegulatedPurePursuitNode::find_lookahead_index_from_projection(
  const nav_msgs::msg::Path & path,
  const PathProjection & projection,
  const double lookahead_distance) const
{
  if (path.poses.empty()) {
    return 0U;
  }
  if (projection.segment_index + 1U >= path.poses.size()) {
    return path.poses.size() - 1U;
  }

  double accumulated = distance2d(
    projection.point,
    point_from_pose(path.poses[projection.segment_index + 1U]));
  if (accumulated >= lookahead_distance) {
    return projection.segment_index + 1U;
  }
  for (std::size_t i = projection.segment_index + 2U; i < path.poses.size(); ++i) {
    accumulated += distance2d(point_from_pose(path.poses[i - 1U]), point_from_pose(path.poses[i]));
    if (accumulated >= lookahead_distance) {
      return i;
    }
  }
  return path.poses.size() - 1U;
}

double RegulatedPurePursuitNode::remaining_path_distance_from_projection(
  const nav_msgs::msg::Path & path,
  const PathProjection & projection) const
{
  if (path.poses.empty()) {
    return 0.0;
  }
  if (projection.segment_index + 1U >= path.poses.size()) {
    return distance2d(projection.point, point_from_pose(path.poses.back()));
  }

  double remaining = distance2d(
    projection.point,
    point_from_pose(path.poses[projection.segment_index + 1U]));
  for (std::size_t i = projection.segment_index + 2U; i < path.poses.size(); ++i) {
    remaining += distance2d(point_from_pose(path.poses[i - 1U]), point_from_pose(path.poses[i]));
  }
  return remaining;
}

double RegulatedPurePursuitNode::remaining_path_distance(
  const Pose2D & robot,
  const nav_msgs::msg::Path & path,
  const std::size_t nearest_index) const
{
  if (path.poses.empty()) {
    return 0.0;
  }
  double remaining = distance2d(
    Point2D{robot.x, robot.y}, point_from_pose(path.poses[nearest_index]));
  for (std::size_t i = nearest_index + 1U; i < path.poses.size(); ++i) {
    remaining += distance2d(point_from_pose(path.poses[i - 1U]), point_from_pose(path.poses[i]));
  }
  return remaining;
}

double RegulatedPurePursuitNode::path_heading_at(
  const nav_msgs::msg::Path & path,
  const std::size_t index) const
{
  if (path.poses.size() < 2U) {
    return 0.0;
  }
  const std::size_t i0 = index + 1U < path.poses.size() ? index : index - 1U;
  const std::size_t i1 = std::min(i0 + 1U, path.poses.size() - 1U);
  const Point2D p0 = point_from_pose(path.poses[i0]);
  const Point2D p1 = point_from_pose(path.poses[i1]);
  return std::atan2(p1.y - p0.y, p1.x - p0.x);
}

geometry_msgs::msg::Twist RegulatedPurePursuitNode::apply_limits(
  const geometry_msgs::msg::Twist & target,
  const double dt)
{
  geometry_msgs::msg::Twist limited;
  limited.linear.x = limit_rate(
    clamp(target.linear.x, 0.0, params_.max_forward_speed),
    last_cmd_.linear.x, params_.accel_limit, dt);
  limited.linear.y = limit_rate(
    clamp(target.linear.y, -params_.max_lateral_speed, params_.max_lateral_speed),
    last_cmd_.linear.y, params_.lateral_accel_limit, dt);
  limited.angular.z = limit_rate(
    clamp(target.angular.z, -params_.max_yaw_rate, params_.max_yaw_rate),
    last_cmd_.angular.z, params_.yaw_accel_limit, dt);
  return limited;
}

void RegulatedPurePursuitNode::publish_zero(
  const TrackerState state,
  const std::string & reason)
{
  if (params_.publish_zero_on_stop) {
    geometry_msgs::msg::Twist zero;
    cmd_pub_->publish(zero);
    last_cmd_ = zero;
  }
  last_control_time_ = now();
  publish_state(state);
  RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "RPP stopped: %s", reason.c_str());

  if (params_.publish_debug) {
    std_msgs::msg::Float32MultiArray msg;
    msg.data = {
      static_cast<float>(static_cast<int>(state)),
      0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    debug_pub_->publish(msg);
  }
}

void RegulatedPurePursuitNode::publish_state(const TrackerState state)
{
  current_state_ = state;
  std_msgs::msg::Int32 msg;
  msg.data = static_cast<int>(state);
  state_pub_->publish(msg);
}

void RegulatedPurePursuitNode::publish_debug(
  const TrackerState state,
  const nav_msgs::msg::Path &,
  const std::size_t nearest_index,
  const std::size_t lookahead_index,
  const double lookahead_distance,
  const double deviation,
  const double remaining_distance,
  const double curvature,
  const double speed_scale,
  const geometry_msgs::msg::Twist & cmd)
{
  if (!params_.publish_debug) {
    return;
  }
  std_msgs::msg::Float32MultiArray msg;
  msg.data = {
    static_cast<float>(static_cast<int>(state)),
    static_cast<float>(nearest_index),
    static_cast<float>(lookahead_index),
    static_cast<float>(lookahead_distance),
    static_cast<float>(deviation),
    static_cast<float>(remaining_distance),
    static_cast<float>(curvature),
    static_cast<float>(speed_scale),
    static_cast<float>(cmd.linear.x),
    static_cast<float>(cmd.linear.y),
    static_cast<float>(cmd.angular.z)};
  debug_pub_->publish(msg);
}

}  // namespace trg_path_follower
