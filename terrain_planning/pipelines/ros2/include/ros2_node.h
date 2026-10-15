#ifndef PIPELINES_ROS2_INCLUDE_ROS2_NODE_H_
#define PIPELINES_ROS2_INCLUDE_ROS2_NODE_H_

#include "trg_planner/include/planner/trg_planner.h"
#include "trg_planner/include/utils/common.h"

#include "ros2_utils.hpp"

class ROS2Node : public TRGPlanner {
 public:
  explicit ROS2Node(const rclcpp::Node::SharedPtr &node);
  virtual ~ROS2Node();

  void getParams(const rclcpp::Node::SharedPtr &node);
  void cbPose(const std::shared_ptr<const ROS2Types::Pose> &msg);
  void cbOdom(const std::shared_ptr<const ROS2Types::Odom> &msg);
  void cbCloud(const std::shared_ptr<const ROS2Types::PointCloud> &msg);
  void cbGoal(const std::shared_ptr<const ROS2Types::Pose> &msg);
  void cbAllowedArea(const std::shared_ptr<const ROS2Types::PolygonStamped> &msg);

  void publishTimer();
  void debugTimer();
  void replanTimer();

 protected:
  //// ROS2 Node
  rclcpp::Node::SharedPtr n_;
  //// ROS2 Subscribers
  struct Subscribers {
    rclcpp::Subscription<ROS2Types::Pose>::SharedPtr           ego_pose_;
    rclcpp::Subscription<ROS2Types::Odom>::SharedPtr           ego_odom_;
    rclcpp::Subscription<ROS2Types::PointCloud>::SharedPtr     obs_cloud_;
    rclcpp::Subscription<ROS2Types::Pose>::SharedPtr           goal_;
    rclcpp::Subscription<ROS2Types::PolygonStamped>::SharedPtr allowed_area_;
  } sub;
  //// ROS2 Publishers
  struct Publishers {
    rclcpp::Publisher<ROS2Types::PointCloud>::SharedPtr pre_map_;
    rclcpp::Publisher<ROS2Types::PointCloud>::SharedPtr goal_;
    rclcpp::Publisher<ROS2Types::Path>::SharedPtr       path_;
  } pub;
  struct Debug {
    rclcpp::Publisher<ROS2Types::MarkerArray>::SharedPtr global_trg_;
    rclcpp::Publisher<ROS2Types::MarkerArray>::SharedPtr local_trg_;
    rclcpp::Publisher<ROS2Types::MarkerArray>::SharedPtr historical_trg_;
    rclcpp::Publisher<ROS2Types::MarkerArray>::SharedPtr allowed_area_;
    rclcpp::Publisher<ROS2Types::PointCloud>::SharedPtr  obs_map_;
    rclcpp::Publisher<ROS2Types::FloatArray>::SharedPtr  path_info_;
  } debug;
  struct QoS {
    rclcpp::QoS for_reli = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    rclcpp::QoS for_viz  = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  } qos;

  //// Parameters
  struct Parameters {
    bool        isDebug;
    std::string frame_id;
    float       publish_rate;
    float       debug_rate;
    bool        replan_enabled;
    float       replan_check_rate;
    float       replan_min_interval;
    float       replan_min_translation;
    float       replan_min_yaw;
  } param_;

  std::unordered_map<std::string, std::string> topics_;

  //// Threads
  struct Threads {
    std::thread                            publish;
    std::thread                            debug;
    std::unordered_map<std::string, float> hz;
  } thd;

  //// TF
  struct TFCache {
    std::shared_ptr<tf2_ros::Buffer>            tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    geometry_msgs::msg::TransformStamped        tfStamped;
    bool                                        isTFCached = false;
  } tf_cache;

  void vizGraph(std::string type, rclcpp::Publisher<ROS2Types::MarkerArray>::SharedPtr pub);
  void publishAllowedAreaMarker();
  void publishHistoricalTrg();
  float yawFromQuat(const Eigen::Vector4f &quat) const;
  float shortestYawDiff(float lhs, float rhs) const;
  void  setReplanReferenceLocked(const Eigen::Vector3f &pose, const Eigen::Vector4f &quat);

  rclcpp::TimerBase::SharedPtr replan_timer_;
  std::mutex                   replan_mtx_;
  std::mutex                   boundary_mtx_;
  std::atomic<bool>            has_goal_{false};
  std::atomic<bool>            queued_goal_{false};
  bool                         replan_reference_set_ = false;
  Eigen::Vector3f              replan_reference_pose_ = Eigen::Vector3f::Zero();
  Eigen::Vector4f              replan_reference_quat_ = Eigen::Vector4f(1, 0, 0, 0);
  rclcpp::Time                 last_replan_request_time_{0, 0, RCL_ROS_TIME};

  // Historical TRG: freeze the last non-empty graph when robot leaves boundary.
  ROS2Types::MarkerArray historical_markers_cache_;
  bool                   historical_frozen_ = false;
};

#endif  // PIPELINES_ROS2_INCLUDE_ROS2_NODE_H_
