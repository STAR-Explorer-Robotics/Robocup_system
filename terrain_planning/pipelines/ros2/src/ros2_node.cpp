#include "ros2_node.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <numeric>

namespace {
constexpr float kMarkerZ = 0.02f;
constexpr float kEarEps  = 1e-5f;

float signedPolygonArea(const std::vector<Eigen::Vector2f> &points) {
  float area = 0.0f;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector2f &a = points[i];
    const Eigen::Vector2f &b = points[(i + 1) % points.size()];
    area += a.x() * b.y() - b.x() * a.y();
  }
  return 0.5f * area;
}

float cross2d(const Eigen::Vector2f &a, const Eigen::Vector2f &b, const Eigen::Vector2f &c) {
  const Eigen::Vector2f ab = b - a;
  const Eigen::Vector2f ac = c - a;
  return ab.x() * ac.y() - ab.y() * ac.x();
}

bool pointInTriangle(const Eigen::Vector2f &p,
                     const Eigen::Vector2f &a,
                     const Eigen::Vector2f &b,
                     const Eigen::Vector2f &c) {
  const float c1 = cross2d(a, b, p);
  const float c2 = cross2d(b, c, p);
  const float c3 = cross2d(c, a, p);
  const bool  has_neg = c1 < -kEarEps || c2 < -kEarEps || c3 < -kEarEps;
  const bool  has_pos = c1 > kEarEps || c2 > kEarEps || c3 > kEarEps;
  return !(has_neg && has_pos);
}

std::vector<std::array<int, 3>> triangulatePolygon(const std::vector<Eigen::Vector2f> &points) {
  std::vector<std::array<int, 3>> triangles;
  if (points.size() < 3) {
    return triangles;
  }
  if (points.size() == 3) {
    triangles.push_back({0, 1, 2});
    return triangles;
  }

  std::vector<int> indices(points.size());
  std::iota(indices.begin(), indices.end(), 0);
  const bool ccw = signedPolygonArea(points) >= 0.0f;

  while (indices.size() > 3) {
    bool clipped = false;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      const int prev = indices[(i + indices.size() - 1) % indices.size()];
      const int curr = indices[i];
      const int next = indices[(i + 1) % indices.size()];

      const float turn = cross2d(points[prev], points[curr], points[next]);
      if ((ccw && turn <= kEarEps) || (!ccw && turn >= -kEarEps)) {
        continue;
      }

      bool contains_other = false;
      for (const int candidate : indices) {
        if (candidate == prev || candidate == curr || candidate == next) {
          continue;
        }
        if (pointInTriangle(points[candidate], points[prev], points[curr], points[next])) {
          contains_other = true;
          break;
        }
      }
      if (contains_other) {
        continue;
      }

      triangles.push_back({prev, curr, next});
      indices.erase(indices.begin() + static_cast<std::ptrdiff_t>(i));
      clipped = true;
      break;
    }

    if (!clipped) {
      triangles.clear();
      for (std::size_t i = 1; i + 1 < points.size(); ++i) {
        triangles.push_back({0, static_cast<int>(i), static_cast<int>(i + 1)});
      }
      return triangles;
    }
  }

  triangles.push_back({indices[0], indices[1], indices[2]});
  return triangles;
}

geometry_msgs::msg::Point makePoint(float x, float y, float z) {
  geometry_msgs::msg::Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  return point;
}
}  // namespace

ROS2Node::ROS2Node(const rclcpp::Node::SharedPtr &node) : n_(node) {
  getParams(n_);

  tf_cache.tfBuffer   = std::make_shared<tf2_ros::Buffer>(n_->get_clock());
  tf_cache.tfListener = std::make_shared<tf2_ros::TransformListener>(*tf_cache.tfBuffer);

  sub.ego_pose_ = n_->create_subscription<ROS2Types::Pose>(
      topics_["egoPose"], qos.for_reli, std::bind(&ROS2Node::cbPose, this, std::placeholders::_1));
  sub.ego_odom_ = n_->create_subscription<ROS2Types::Odom>(
      topics_["egoOdom"], qos.for_reli, std::bind(&ROS2Node::cbOdom, this, std::placeholders::_1));
  sub.obs_cloud_ = n_->create_subscription<ROS2Types::PointCloud>(
      topics_["obsCloud"],
      qos.for_reli,
      std::bind(&ROS2Node::cbCloud, this, std::placeholders::_1));
  sub.goal_ = n_->create_subscription<ROS2Types::Pose>(
      topics_["goal"], qos.for_reli, std::bind(&ROS2Node::cbGoal, this, std::placeholders::_1));

  pub.pre_map_ = n_->create_publisher<ROS2Types::PointCloud>(topics_["preMap"], 1);
  pub.goal_    = n_->create_publisher<ROS2Types::PointCloud>(topics_["outGoal"], 1);
  pub.path_    = n_->create_publisher<ROS2Types::Path>(topics_["path"], 1);

  debug.global_trg_   = n_->create_publisher<ROS2Types::MarkerArray>(topics_["globalTRG"], 1);
  debug.local_trg_    = n_->create_publisher<ROS2Types::MarkerArray>(topics_["localTRG"], 1);
  debug.historical_trg_ = n_->create_publisher<ROS2Types::MarkerArray>(
      topics_["historicalTRG"], rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
  debug.allowed_area_ = n_->create_publisher<ROS2Types::MarkerArray>(
      topics_["allowedArea"], rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
  debug.obs_map_   = n_->create_publisher<ROS2Types::PointCloud>(topics_["obsMap"], 1);
  debug.path_info_ = n_->create_publisher<ROS2Types::FloatArray>(topics_["pathInfo"], 1);

  TRGPlanner::init();
  if (!topics_["allowedAreaInput"].empty()) {
    sub.allowed_area_ = n_->create_subscription<ROS2Types::PolygonStamped>(
        topics_["allowedAreaInput"],
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
        std::bind(&ROS2Node::cbAllowedArea, this, std::placeholders::_1));
  }
  publishAllowedAreaMarker();
  print("TRG Planner ROS2 initialized", TRGPlanner::param_.isVerbose);

  if (param_.replan_enabled && param_.replan_check_rate > 0.0f) {
    const auto period_ms =
        std::chrono::milliseconds(static_cast<int>(1000.0f / param_.replan_check_rate));
    replan_timer_ = n_->create_wall_timer(period_ms, std::bind(&ROS2Node::replanTimer, this));
  }

  //// Threads
  thd.publish = std::thread(&ROS2Node::publishTimer, this);
  if (param_.isDebug) {
    thd.debug = std::thread(&ROS2Node::debugTimer, this);
  }
}

ROS2Node::~ROS2Node() {
  is_running.store(false);
  if (thd.publish.joinable()) {
    thd.publish.join();
  }
  if (thd.debug.joinable()) {
    thd.debug.join();
  }
}

void ROS2Node::getParams(const rclcpp::Node::SharedPtr &n_) {
  n_->declare_parameter<bool>("ros2.isDebug", true);
  n_->declare_parameter<std::string>("ros2.frameId", "map");
  n_->declare_parameter<float>("ros2.publishRate", 1.0f);
  n_->declare_parameter<float>("ros2.debugRate", 1.0f);
  n_->declare_parameter<bool>("ros2.replan.enabled", true);
  n_->declare_parameter<float>("ros2.replan.checkRate", 2.0f);
  n_->declare_parameter<float>("ros2.replan.minInterval", 2.0f);
  n_->declare_parameter<float>("ros2.replan.minTranslation", 0.8f);
  n_->declare_parameter<float>("ros2.replan.minYaw", 0.35f);

  n_->declare_parameter<std::string>("ros2.topic.input.egoPose",
                                     "/trg_ros2_node/input/default_ego_pose");
  n_->declare_parameter<std::string>("ros2.topic.input.egoOdom",
                                     "/trg_ros2_node/input/default_ego_odom");
  n_->declare_parameter<std::string>("ros2.topic.input.obsCloud",
                                     "/trg_ros2_node/input/default_obs_cloud");
  n_->declare_parameter<std::string>("ros2.topic.input.goal", "/trg_ros2_node/input/default_goal");
  n_->declare_parameter<std::string>("ros2.topic.input.allowedArea",
                                     "/navigation/detected_allowed_area");
  n_->declare_parameter<std::string>("ros2.topic.output.preMap",
                                     "/trg_ros2_node/output/default_preMap");
  n_->declare_parameter<std::string>("ros2.topic.output.goal",
                                     "/trg_ros2_node/output/default_goal");
  n_->declare_parameter<std::string>("ros2.topic.output.path",
                                     "/trg_ros2_node/output/default_path");
  n_->declare_parameter<std::string>("ros2.topic.debug.globalTRG",
                                     "/trg_ros2_node/debug/globalTRG");
  n_->declare_parameter<std::string>("ros2.topic.debug.historicalTRG",
                                     "/trg/debug/historical_trg");
  n_->declare_parameter<std::string>("ros2.topic.debug.localTRG", "/trg_ros2_node/debug/localTRG");
  n_->declare_parameter<std::string>("ros2.topic.debug.allowedArea",
                                     "/navigation/allowed_area_marker");
  n_->declare_parameter<std::string>("ros2.topic.debug.obsMap",
                                     "/trg_ros2_node/debug/default_obsMap");
  n_->declare_parameter<std::string>("ros2.topic.debug.pathInfo",
                                     "/trg_ros2_node/debug/default_pathInfo");

  n_->declare_parameter<std::string>("mapConfig", "robocup_default");
  n_->declare_parameter<std::string>("mapConfigPath", "");
  n_->declare_parameter<std::string>("prebuiltMapPath", "");
  n_->declare_parameter<std::string>("boundaryFile", "");
  n_->declare_parameter<std::string>("boundaryKeepoutMargin", "");

  //// Get parameters
  n_->get_parameter("ros2.isDebug", param_.isDebug);
  n_->get_parameter("ros2.frameId", param_.frame_id);
  n_->get_parameter("ros2.publishRate", param_.publish_rate);
  n_->get_parameter("ros2.debugRate", param_.debug_rate);
  n_->get_parameter("ros2.replan.enabled", param_.replan_enabled);
  n_->get_parameter("ros2.replan.checkRate", param_.replan_check_rate);
  n_->get_parameter("ros2.replan.minInterval", param_.replan_min_interval);
  n_->get_parameter("ros2.replan.minTranslation", param_.replan_min_translation);
  n_->get_parameter("ros2.replan.minYaw", param_.replan_min_yaw);

  n_->get_parameter("ros2.topic.input.egoPose", topics_["egoPose"]);
  n_->get_parameter("ros2.topic.input.egoOdom", topics_["egoOdom"]);
  n_->get_parameter("ros2.topic.input.obsCloud", topics_["obsCloud"]);
  n_->get_parameter("ros2.topic.input.goal", topics_["goal"]);
  n_->get_parameter("ros2.topic.input.allowedArea", topics_["allowedAreaInput"]);
  n_->get_parameter("ros2.topic.output.preMap", topics_["preMap"]);
  n_->get_parameter("ros2.topic.output.goal", topics_["outGoal"]);
  n_->get_parameter("ros2.topic.output.path", topics_["path"]);
  n_->get_parameter("ros2.topic.debug.globalTRG", topics_["globalTRG"]);
  n_->get_parameter("ros2.topic.debug.historicalTRG", topics_["historicalTRG"]);
  n_->get_parameter("ros2.topic.debug.localTRG", topics_["localTRG"]);
  n_->get_parameter("ros2.topic.debug.allowedArea", topics_["allowedArea"]);
  n_->get_parameter("ros2.topic.debug.obsMap", topics_["obsMap"]);
  n_->get_parameter("ros2.topic.debug.pathInfo", topics_["pathInfo"]);

  std::string map_config_name;
  std::string map_config_path;
  std::string prebuilt_map_path;
  std::string boundary_file;
  std::string boundary_keepout_margin;
  n_->get_parameter("mapConfig", map_config_name);
  n_->get_parameter("mapConfigPath", map_config_path);
  n_->get_parameter("prebuiltMapPath", prebuilt_map_path);
  n_->get_parameter("boundaryFile", boundary_file);
  n_->get_parameter("boundaryKeepoutMargin", boundary_keepout_margin);
  if (map_config_path.empty()) {
    map_config_path = std::string(TRG_ROS_DIR) + "/../config/" + map_config_name + ".yaml";
  }
  if (!std::filesystem::exists(map_config_path)) {
    print_error("Map config file does not exist: " + map_config_path);
    exit(1);
  }
  TRGPlanner::setParams(map_config_path);
  if (!prebuilt_map_path.empty()) {
    TRGPlanner::param_.preMapPath = prebuilt_map_path;
  }
  if (!boundary_file.empty()) {
    std::filesystem::path boundary_path(boundary_file);
    if (!std::filesystem::exists(boundary_path)) {
      print_error("Allowed area file does not exist: " + boundary_path.string());
      exit(1);
    }
    const bool override_keepout = !boundary_keepout_margin.empty();
    float      keepout_margin  = TRGPlanner::param_.boundaryKeepoutMargin;
    if (override_keepout) {
      keepout_margin = std::stof(boundary_keepout_margin);
    }
    TRGPlanner::param_.boundaryEnabled = true;
    TRGPlanner::param_.boundaryPath    = boundary_path.string();
    TRGPlanner::loadAllowedArea();
    if (override_keepout) {
      TRGPlanner::param_.boundaryKeepoutMargin = keepout_margin;
    }
  }
}

void ROS2Node::cbPose(const std::shared_ptr<const ROS2Types::Pose> &msg) {
  {
    std::lock_guard<std::mutex> lock(TRGPlanner::mtx.odom);
    TRGPlanner::state_.frame_id = msg->header.frame_id;
    TRGPlanner::state_.pose3d =
        Eigen::Vector3f(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    TRGPlanner::state_.pose2d = Eigen::Vector2f(msg->pose.position.x, msg->pose.position.y);
    TRGPlanner::state_.quat   = Eigen::Vector4f(msg->pose.orientation.w,
                                              msg->pose.orientation.x,
                                              msg->pose.orientation.y,
                                              msg->pose.orientation.z);
    Eigen::Quaternionf q(TRGPlanner::state_.quat(0),
                         TRGPlanner::state_.quat(1),
                         TRGPlanner::state_.quat(2),
                         TRGPlanner::state_.quat(3));
    TRGPlanner::state_.T_B2M.block<3, 3>(0, 0) = q.toRotationMatrix();
    TRGPlanner::state_.T_B2M.block<3, 1>(0, 3) = TRGPlanner::state_.pose3d;
    TRGPlanner::flag_.poseIn                   = true;
  }
}

void ROS2Node::cbOdom(const std::shared_ptr<const ROS2Types::Odom> &msg) {
  {
    std::lock_guard<std::mutex> lock(mtx.odom);
    TRGPlanner::state_.frame_id = msg->header.frame_id;
    TRGPlanner::state_.pose3d   = Eigen::Vector3f(
        msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    TRGPlanner::state_.pose2d =
        Eigen::Vector2f(msg->pose.pose.position.x, msg->pose.pose.position.y);
    TRGPlanner::state_.quat = Eigen::Vector4f(msg->pose.pose.orientation.w,
                                              msg->pose.pose.orientation.x,
                                              msg->pose.pose.orientation.y,
                                              msg->pose.pose.orientation.z);
    Eigen::Quaternionf q(TRGPlanner::state_.quat(0),
                         TRGPlanner::state_.quat(1),
                         TRGPlanner::state_.quat(2),
                         TRGPlanner::state_.quat(3));
    TRGPlanner::state_.T_B2M.block<3, 3>(0, 0) = q.toRotationMatrix();
    TRGPlanner::state_.T_B2M.block<3, 1>(0, 3) = TRGPlanner::state_.pose3d;
    TRGPlanner::flag_.poseIn                   = true;
  }
}

void ROS2Node::cbCloud(const std::shared_ptr<const ROS2Types::PointCloud> &msg) {
  if (!TRGPlanner::flag_.poseIn) {
    return;
  }
  {
    std::lock_guard<std::mutex>      lock(TRGPlanner::mtx.obs);
    pcl::PointCloud<PtsDefault>::Ptr cloud_in(new pcl::PointCloud<PtsDefault>());
    pcl::fromROSMsg(*msg, *cloud_in);
    if (msg->header.frame_id == TRGPlanner::state_.frame_id) {
      pcl::transformPointCloud(*cloud_in, *TRGPlanner::cs_.obsPtr, Eigen::Matrix4f::Identity());
    } else {
      try {
        tf_cache.tfStamped = tf_cache.tfBuffer->lookupTransform(
            TRGPlanner::state_.frame_id, msg->header.frame_id, rclcpp::Time(0));
        tf_cache.isTFCached = true;
      } catch (tf2::TransformException &ex) {
        print_error(ex.what());
        return;
      }

      // transform from sensor frame to map frame
      Eigen::Matrix4f    T_S2M = Eigen::Matrix4f::Identity();
      Eigen::Quaternionf q(tf_cache.tfStamped.transform.rotation.w,
                           tf_cache.tfStamped.transform.rotation.x,
                           tf_cache.tfStamped.transform.rotation.y,
                           tf_cache.tfStamped.transform.rotation.z);
      T_S2M.block<3, 3>(0, 0) = q.toRotationMatrix();
      T_S2M.block<3, 1>(0, 3) = Eigen::Vector3f(tf_cache.tfStamped.transform.translation.x,
                                                tf_cache.tfStamped.transform.translation.y,
                                                tf_cache.tfStamped.transform.translation.z);
      pcl::transformPointCloud(*cloud_in, *TRGPlanner::cs_.obsPtr, T_S2M);
    }
    TRGPlanner::flag_.obsIn = true;
  }
}

void ROS2Node::cbGoal(const std::shared_ptr<const ROS2Types::Pose> &msg) {
  Eigen::Vector2f goal2d(msg->pose.position.x, msg->pose.position.y);
  if (TRGPlanner::trg_ != nullptr && !TRGPlanner::trg_->isWithinAllowedArea(goal2d)) {
    print_error("Goal is outside allowed area");
    return;
  }
  const bool graph_ready = TRGPlanner::flag_.graphInit;
  {
    std::lock_guard<std::mutex> lock(TRGPlanner::mtx.goal);
    TRGPlanner::goal_state_.pose =
        Eigen::Vector3f(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    TRGPlanner::goal_state_.quat = Eigen::Vector4f(msg->pose.orientation.w,
                                                   msg->pose.orientation.x,
                                                   msg->pose.orientation.y,
                                                   msg->pose.orientation.z);
    TRGPlanner::goal_state_.init = true;
    TRGPlanner::flag_.goalIn     = graph_ready;
  }
  has_goal_ = true;
  queued_goal_ = !graph_ready;
  Eigen::Vector3f curr_pose;
  Eigen::Vector4f curr_quat;
  bool            pose_ready = false;
  {
    std::lock_guard<std::mutex> odom_lock(TRGPlanner::mtx.odom);
    pose_ready = TRGPlanner::flag_.poseIn;
    curr_pose  = TRGPlanner::state_.pose3d;
    curr_quat  = TRGPlanner::state_.quat;
  }
  {
    std::lock_guard<std::mutex> replan_lock(replan_mtx_);
    if (pose_ready) {
      setReplanReferenceLocked(curr_pose, curr_quat);
    } else {
      replan_reference_set_ = false;
    }
    last_replan_request_time_ = n_->now();
  }
  if (graph_ready) {
    print("Goal received", TRGPlanner::param_.isVerbose);
  } else {
    print_warning("Goal queued until TRG graph is initialized");
  }
}

void ROS2Node::cbAllowedArea(const std::shared_ptr<const ROS2Types::PolygonStamped> &msg) {
  std::vector<Eigen::Vector2f> polygon;
  polygon.reserve(msg->polygon.points.size());
  for (const auto &point : msg->polygon.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
      continue;
    }
    polygon.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
  }

  if (polygon.size() < 3) {
    print_warning("Ignoring dynamic allowed area with fewer than 3 finite vertices",
                  TRGPlanner::param_.isVerbose);
    return;
  }

  float keepout_margin = 0.0f;
  {
    std::lock_guard<std::mutex> lock(boundary_mtx_);
    keepout_margin = TRGPlanner::param_.boundaryKeepoutMargin;
    TRGPlanner::param_.boundaryEnabled = true;
    TRGPlanner::param_.boundaryPath.clear();
    TRGPlanner::param_.boundaryPolygon = polygon;
  }

  if (TRGPlanner::trg_ != nullptr) {
    TRGPlanner::trg_->setAllowedArea(polygon, keepout_margin, true);
    TRGPlanner::trg_->pruneOutsideAllowedArea();
  }
  publishAllowedAreaMarker();
  print("Dynamic allowed area updated from " + msg->header.frame_id + ": " +
            std::to_string(polygon.size()) + " vertices",
        TRGPlanner::param_.isVerbose);
}

void ROS2Node::replanTimer() {
  if (!param_.replan_enabled || !TRGPlanner::flag_.graphInit || !has_goal_) {
    return;
  }

  const rclcpp::Time now = n_->now();
  {
    std::lock_guard<std::mutex> replan_lock(replan_mtx_);
    if (last_replan_request_time_.nanoseconds() > 0 &&
        (now - last_replan_request_time_).seconds() < param_.replan_min_interval) {
      return;
    }
  }

  Eigen::Vector3f curr_pose;
  Eigen::Vector4f curr_quat;
  {
    std::lock_guard<std::mutex> lock(TRGPlanner::mtx.odom);
    if (!TRGPlanner::flag_.poseIn) {
      return;
    }
    curr_pose = TRGPlanner::state_.pose3d;
    curr_quat = TRGPlanner::state_.quat;
  }

  std::lock_guard<std::mutex> replan_lock(replan_mtx_);
  if (!replan_reference_set_) {
    setReplanReferenceLocked(curr_pose, curr_quat);
    last_replan_request_time_ = now;
    return;
  }

  const float translation = (curr_pose.head(2) - replan_reference_pose_.head(2)).norm();
  const float yaw_delta =
      std::abs(shortestYawDiff(yawFromQuat(curr_quat), yawFromQuat(replan_reference_quat_)));

  if (translation < param_.replan_min_translation && yaw_delta < param_.replan_min_yaw) {
    return;
  }

  {
    std::lock_guard<std::mutex> goal_lock(TRGPlanner::mtx.goal);
    if (TRGPlanner::flag_.goalIn) {
      return;
    }
    TRGPlanner::flag_.goalIn = true;
  }
  setReplanReferenceLocked(curr_pose, curr_quat);
  last_replan_request_time_ = now;
  print_warning("Periodic replan requested by odometry delta: translation=" +
                    std::to_string(translation) + " m, yaw=" + std::to_string(yaw_delta) + " rad",
                TRGPlanner::param_.isVerbose);
}

void ROS2Node::publishTimer() {
  while (is_running.load()) {
    auto start_loop = tic();
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));

    if (TRGPlanner::param_.isPreMap) {
      publishCloud(n_, TRGPlanner::state_.frame_id, TRGPlanner::cs_.preMapPtr, pub.pre_map_);
    }
    if (queued_goal_.load() && TRGPlanner::flag_.graphInit) {
      {
        std::lock_guard<std::mutex> lock(TRGPlanner::mtx.goal);
        TRGPlanner::flag_.goalIn = true;
      }
      queued_goal_ = false;
      print_warning("Queued goal activated after TRG graph initialization");
    }
    if (TRGPlanner::flag_.pathFound) {
      TRGPlanner::flag_.pathFound = false;
      std::vector<Eigen::Vector3f> path_to_publish = TRGPlanner::path_.smooth;
      if (path_to_publish.size() < 2U) {
        path_to_publish.clear();
        path_to_publish.push_back(TRGPlanner::state_.pose3d);
        path_to_publish.push_back(TRGPlanner::goal_state_.pose);
        print_warning("Publishing two-point fallback path for short TRG path");
      }
      publishPath(n_, TRGPlanner::state_.frame_id, path_to_publish, pub.path_);
      ROS2Types::FloatArray path_info_msg;
      path_info_msg.data.push_back(TRGPlanner::path_.direct_dist);
      path_info_msg.data.push_back(TRGPlanner::path_.raw_path_length);
      path_info_msg.data.push_back(TRGPlanner::path_.smooth_path_length);
      path_info_msg.data.push_back(TRGPlanner::path_.planning_time);
      path_info_msg.data.push_back(TRGPlanner::path_.avg_risk);
      debug.path_info_->publish(path_info_msg);
    }
    if (TRGPlanner::goal_state_.init) {
      PointCloudPtr goal_cloud(new pcl::PointCloud<PtsDefault>());
      PtsDefault    pt;
      pt.x = TRGPlanner::goal_state_.pose.x();
      pt.y = TRGPlanner::goal_state_.pose.y();
      pt.z = TRGPlanner::goal_state_.pose.z();
      goal_cloud->push_back(pt);
      publishCloud(n_, TRGPlanner::state_.frame_id, goal_cloud, pub.goal_);
      TRGPlanner::goal_state_.init = false;
    }
    float loop_time   = toc(start_loop, "ms");
    int   remain_time = 1000 / param_.publish_rate - loop_time;
    if (remain_time > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(remain_time));
    }
    thd.hz["publish"] = std::round(1000 / toc(start_loop, "ms") * 100) / 100;
  }
}

void ROS2Node::debugTimer() {
  while (is_running.load()) {
    auto start_loop = tic();
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    if (TRGPlanner::flag_.graphInit) {
      vizGraph("global", debug.global_trg_);
    }
    if (TRGPlanner::flag_.graphInit && TRGPlanner::param_.isUpdate) {
      vizGraph("local", debug.local_trg_);
    }
    if (TRGPlanner::flag_.obsIn) {
      publishCloud(n_, TRGPlanner::state_.frame_id, TRGPlanner::cs_.obsPtr, debug.obs_map_);
    }
    publishHistoricalTrg();
    publishAllowedAreaMarker();
    float loop_time   = toc(start_loop, "ms");
    int   remain_time = 1000 / param_.debug_rate - loop_time;
    if (remain_time > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(remain_time));
    }
    thd.hz["debug"] = std::round(1000 / toc(start_loop, "ms") * 100) / 100;
  }
}

void ROS2Node::publishAllowedAreaMarker() {
  ROS2Types::MarkerArray marker_array;
  ROS2Types::Marker      delete_marker;
  delete_marker.header.frame_id = param_.frame_id;
  delete_marker.header.stamp    = n_->now();
  delete_marker.action          = ROS2Types::Marker::DELETEALL;
  marker_array.markers.push_back(delete_marker);

  bool                         boundary_enabled = false;
  std::vector<Eigen::Vector2f> polygon;
  {
    std::lock_guard<std::mutex> lock(boundary_mtx_);
    boundary_enabled = TRGPlanner::param_.boundaryEnabled;
    polygon          = TRGPlanner::param_.boundaryPolygon;
  }
  if (!boundary_enabled || polygon.size() < 3) {
    debug.allowed_area_->publish(marker_array);
    return;
  }

  ROS2Types::Marker fill_marker;
  fill_marker.header.frame_id = param_.frame_id;
  fill_marker.header.stamp    = n_->now();
  fill_marker.ns              = "allowed_area_fill";
  fill_marker.id              = 1;
  fill_marker.type            = ROS2Types::Marker::TRIANGLE_LIST;
  fill_marker.action          = ROS2Types::Marker::ADD;
  fill_marker.pose.orientation.w = 1.0;
  fill_marker.scale.x = fill_marker.scale.y = fill_marker.scale.z = 1.0;
  fill_marker.color.r = 0.1f;
  fill_marker.color.g = 0.35f;
  fill_marker.color.b = 1.0f;
  fill_marker.color.a = 0.25f;

  const auto triangles = triangulatePolygon(polygon);
  for (const auto &triangle : triangles) {
    for (const int index : triangle) {
      const Eigen::Vector2f &pt = polygon[static_cast<std::size_t>(index)];
      fill_marker.points.push_back(makePoint(pt.x(), pt.y(), kMarkerZ));
    }
  }

  ROS2Types::Marker line_marker;
  line_marker.header.frame_id = param_.frame_id;
  line_marker.header.stamp    = n_->now();
  line_marker.ns              = "allowed_area_outline";
  line_marker.id              = 2;
  line_marker.type            = ROS2Types::Marker::LINE_STRIP;
  line_marker.action          = ROS2Types::Marker::ADD;
  line_marker.pose.orientation.w = 1.0;
  line_marker.scale.x = 0.06;
  line_marker.color.r = 0.0f;
  line_marker.color.g = 0.45f;
  line_marker.color.b = 1.0f;
  line_marker.color.a = 0.95f;
  for (const Eigen::Vector2f &pt : polygon) {
    line_marker.points.push_back(makePoint(pt.x(), pt.y(), kMarkerZ + 0.01f));
  }
  line_marker.points.push_back(makePoint(polygon.front().x(), polygon.front().y(), kMarkerZ + 0.01f));

  ROS2Types::Marker vertex_marker;
  vertex_marker.header.frame_id = param_.frame_id;
  vertex_marker.header.stamp    = n_->now();
  vertex_marker.ns              = "allowed_area_vertices";
  vertex_marker.id              = 3;
  vertex_marker.type            = ROS2Types::Marker::SPHERE_LIST;
  vertex_marker.action          = ROS2Types::Marker::ADD;
  vertex_marker.pose.orientation.w = 1.0;
  vertex_marker.scale.x = 0.18;
  vertex_marker.scale.y = 0.18;
  vertex_marker.scale.z = 0.18;
  vertex_marker.color.r = 0.0f;
  vertex_marker.color.g = 0.75f;
  vertex_marker.color.b = 1.0f;
  vertex_marker.color.a = 1.0f;
  for (const Eigen::Vector2f &pt : polygon) {
    vertex_marker.points.push_back(makePoint(pt.x(), pt.y(), kMarkerZ + 0.04f));
  }

  marker_array.markers.push_back(fill_marker);
  marker_array.markers.push_back(line_marker);
  marker_array.markers.push_back(vertex_marker);
  debug.allowed_area_->publish(marker_array);
}

void ROS2Node::publishHistoricalTrg() {
  // Snapshot the global graph whenever it is non-empty. When the graph is
  // later cleared (robot leaves boundary), keep publishing the last snapshot
  // so the user can still see the accumulated graph as a historical overlay.
  bool has_nodes = false;
  TRGPlanner::trg_->lockGraph();
  auto nodes = TRGPlanner::trg_->getGraph("global");
  for (const auto &n : nodes) {
    if (n.second != nullptr && n.second->state_ != TRG::NodeState::Invalid) {
      has_nodes = true;
      break;
    }
  }
  TRGPlanner::trg_->unlockGraph();

  if (has_nodes) {
    // Graph is alive: update the frozen snapshot (but don't publish yet).
    historical_frozen_ = false;
  } else if (!historical_frozen_ && historical_markers_cache_.markers.size() > 1) {
    // Graph just became empty — freeze the last snapshot and publish it.
    historical_frozen_ = true;
    // Replace DELETEALL with ADD so the historical overlay persists.
    if (!historical_markers_cache_.markers.empty() &&
        historical_markers_cache_.markers[0].action == ROS2Types::Marker::DELETEALL) {
      historical_markers_cache_.markers[0].action = ROS2Types::Marker::ADD;
    }
    debug.historical_trg_->publish(historical_markers_cache_);
  }
  // When frozen, do NOT update the cache — keep the peak snapshot.
}

void ROS2Node::vizGraph(std::string                                          type,
                        rclcpp::Publisher<ROS2Types::MarkerArray>::SharedPtr pub) {
  ROS2Types::MarkerArray graph_marker;
  ROS2Types::Marker      delete_marker;
  delete_marker.id     = -1;
  delete_marker.action = ROS2Types::Marker::DELETEALL;
  graph_marker.markers.push_back(delete_marker);

  TRGPlanner::trg_->lockGraph();
  std::unordered_map<int, TRG::Node *> nodes = TRGPlanner::trg_->getGraph(type);

  if (!nodes.empty()) {
    ROS2Types::Marker n_marker;
    n_marker.header.frame_id = param_.frame_id;
    n_marker.header.stamp    = n_->now();
    n_marker.type            = ROS2Types::Marker::SPHERE_LIST;
    n_marker.id              = 0;
    n_marker.action          = ROS2Types::Marker::ADD;
    n_marker.scale.x = n_marker.scale.y = n_marker.scale.z = 0.1;
    n_marker.pose.orientation.w                            = 1.0;

    ROS2Types::Marker edge_marker;
    edge_marker.header.frame_id = param_.frame_id;
    edge_marker.header.stamp    = n_->now();
    edge_marker.type            = ROS2Types::Marker::LINE_LIST;
    edge_marker.id              = 1;
    edge_marker.action          = ROS2Types::Marker::ADD;
    edge_marker.scale.x = edge_marker.scale.y = edge_marker.scale.z = 0.02;
    edge_marker.pose.orientation.w                                  = 1.0;

    float z_offset = 0.15;

    for (const auto &node : nodes) {
      if (node.second == nullptr || node.second->state_ == TRG::NodeState::Invalid) {
        continue;
      }
      geometry_msgs::msg::Point p;
      p.x = node.second->pos_.x();
      p.y = node.second->pos_.y();
      p.z = node.second->pos_.z() + z_offset;
      n_marker.points.push_back(p);
      std_msgs::msg::ColorRGBA color;
      if (node.second->state_ == TRG::NodeState::Frontier) {
        color.r = 1.0;
        color.g = 0.0;
        color.b = 0.0;
        color.a = 1.0;
      } else {
        color.r = 0.0;
        color.g = 1.0;
        color.b = 0.0;
        color.a = 1.0;
      }
      n_marker.colors.push_back(color);

      for (const auto &edge : node.second->edges_) {
        const auto dst_it = nodes.find(edge->dst_id_);
        if (dst_it == nodes.end() || dst_it->second == nullptr) {
          continue;
        }
        if (dst_it->second->state_ == TRG::NodeState::Invalid) {
          continue;
        }
        geometry_msgs::msg::Point p1, p2;
        p1.x = node.second->pos_.x();
        p1.y = node.second->pos_.y();
        p1.z = node.second->pos_.z() + z_offset;
        p2.x = dst_it->second->pos_.x();
        p2.y = dst_it->second->pos_.y();
        p2.z = dst_it->second->pos_.z() + z_offset;
        edge_marker.points.push_back(p1);
        edge_marker.points.push_back(p2);
        std_msgs::msg::ColorRGBA color;
        color.r = 1.0;
        color.g = 1.0 * (1 - edge->weight_);
        color.b = 1.0 * (1 - edge->weight_);
        color.a = 1.0;
        edge_marker.colors.push_back(color);
        edge_marker.colors.push_back(color);
      }
    }
    graph_marker.markers.push_back(n_marker);
    graph_marker.markers.push_back(edge_marker);
  }
  // Cache global graph snapshot for historical overlay.
  if (type == "global") {
    historical_markers_cache_ = graph_marker;
  }
  pub->publish(graph_marker);
  TRGPlanner::trg_->unlockGraph();
}

float ROS2Node::yawFromQuat(const Eigen::Vector4f &quat) const {
  Eigen::Quaternionf q(quat[0], quat[1], quat[2], quat[3]);
  const Eigen::Matrix3f rot = q.normalized().toRotationMatrix();
  return std::atan2(rot(1, 0), rot(0, 0));
}

float ROS2Node::shortestYawDiff(float lhs, float rhs) const {
  float diff = lhs - rhs;
  while (diff > static_cast<float>(M_PI)) {
    diff -= static_cast<float>(2.0 * M_PI);
  }
  while (diff < static_cast<float>(-M_PI)) {
    diff += static_cast<float>(2.0 * M_PI);
  }
  return diff;
}

void ROS2Node::setReplanReferenceLocked(const Eigen::Vector3f &pose,
                                        const Eigen::Vector4f &quat) {
  replan_reference_pose_ = pose;
  replan_reference_quat_ = quat;
  replan_reference_set_  = true;
}
