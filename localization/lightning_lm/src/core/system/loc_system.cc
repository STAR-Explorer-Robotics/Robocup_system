//
// Created by xiang on 25-9-12.
//

#include "core/system/loc_system.h"
#include "core/localization/localization.h"
#include "core/lightning_math.hpp"
#include "io/yaml_io.h"
#include "wrapper/ros_utils.h"

#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <vector>

namespace lightning {
namespace {

template <typename T>
T GetYamlValueOr(const YAML::Node& yaml, const std::string& section, const std::string& key, const T& default_value) {
    const auto section_node = yaml[section];
    if (!section_node || !section_node[key]) {
        return default_value;
    }
    return section_node[key].as<T>();
}

std::string NormalizeFrameName(std::string frame) {
    while (!frame.empty() && frame.front() == '/') {
        frame.erase(frame.begin());
    }

    std::transform(frame.begin(), frame.end(), frame.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return frame;
}

SE3 XyzRpyToSE3(const std::vector<double>& xyzrpy, bool rpy_degrees) {
    if (xyzrpy.size() != 6) {
        LOG(WARNING) << "rviz_initial_pose_lidar_to_body_xyzrpy must contain [x, y, z, roll, pitch, yaw], use identity";
        return SE3();
    }

    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    const double angle_scale = rpy_degrees ? kDegToRad : 1.0;
    PoseRPYD pose(xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3] * angle_scale, xyzrpy[4] * angle_scale,
                  xyzrpy[5] * angle_scale);
    return math::XYZRPYToSE3(pose);
}

SE3 TransformMsgToSE3(const geometry_msgs::msg::TransformStamped& msg) {
    Quatd q(msg.transform.rotation.w, msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z);
    if (q.norm() < 1e-6) {
        LOG(WARNING) << "received invalid localization transform quaternion, use identity rotation";
        q = Quatd::Identity();
    } else {
        q.normalize();
    }

    return SE3(q, Vec3d(msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z));
}

}  // namespace

LocSystem::LocSystem(LocSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

LocSystem::~LocSystem() { loc_->Finish(); }

bool LocSystem::Init(const std::string &yaml_path) {
    loc::Localization::Options opt;
    opt.online_mode_ = true;
    loc_ = std::make_shared<loc::Localization>(opt);

    YAML::Node yaml_node = YAML::LoadFile(yaml_path);
    YAML_IO yaml(yaml_path);

    map_path_ = GetYamlValueOr<std::string>(yaml_node, "system", "map_path", "./data/new_map/");
    options_.pub_tf_ = GetYamlValueOr<bool>(yaml_node, "system", "pub_tf", options_.pub_tf_);
    options_.enable_rviz_ =
        GetYamlValueOr<bool>(yaml_node, "system", "enable_lidar_loc_rviz", options_.enable_rviz_);
    options_.auto_start_from_identity_ =
        GetYamlValueOr<bool>(yaml_node, "system", "auto_start_from_identity", options_.auto_start_from_identity_);
    options_.initialpose_topic_ =
        GetYamlValueOr<std::string>(yaml_node, "system", "rviz_initialpose_topic", options_.initialpose_topic_);
    options_.map_topic_ = GetYamlValueOr<std::string>(yaml_node, "system", "rviz_map_topic", options_.map_topic_);
    options_.odom_topic_ = GetYamlValueOr<std::string>(yaml_node, "system", "rviz_odom_topic", options_.odom_topic_);
    options_.registered_scan_topic_ =
        GetYamlValueOr<std::string>(yaml_node, "system", "rviz_registered_scan_topic", options_.registered_scan_topic_);
    options_.obs_cloud_topic_ =
        GetYamlValueOr<std::string>(yaml_node, "system", "rviz_obs_cloud_topic", options_.obs_cloud_topic_);
    options_.map_frame_id_ = GetYamlValueOr<std::string>(yaml_node, "system", "map_frame_id", options_.map_frame_id_);
    options_.output_child_frame_id_ =
        GetYamlValueOr<std::string>(yaml_node, "system", "output_child_frame_id", options_.output_child_frame_id_);
    options_.rviz_initial_pose_z_offset_m_ =
        GetYamlValueOr<double>(yaml_node, "system", "rviz_initial_pose_z_offset_m",
                               options_.rviz_initial_pose_z_offset_m_);
    options_.rviz_initial_pose_frame_ = NormalizeFrameName(
        GetYamlValueOr<std::string>(yaml_node, "system", "rviz_initial_pose_frame", options_.rviz_initial_pose_frame_));

    if (options_.rviz_initial_pose_frame_ == "base_link" || options_.rviz_initial_pose_frame_ == "body") {
        options_.rviz_initial_pose_is_body_frame_ = true;
    } else if (options_.rviz_initial_pose_frame_ == "sensor" || options_.rviz_initial_pose_frame_ == "lidar") {
        options_.rviz_initial_pose_is_body_frame_ = false;
    } else {
        LOG(WARNING) << "unknown rviz_initial_pose_frame: " << options_.rviz_initial_pose_frame_
                     << ", interpret RViz initial pose as sensor frame";
        options_.rviz_initial_pose_frame_ = "sensor";
        options_.rviz_initial_pose_is_body_frame_ = false;
    }

    const auto lidar_to_body_xyzrpy = GetYamlValueOr<std::vector<double>>(
        yaml_node, "system", "rviz_initial_pose_lidar_to_body_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0});
    const bool extrinsic_rpy_degrees =
        GetYamlValueOr<bool>(yaml_node, "system", "rviz_initial_pose_extrinsic_rpy_degrees", false);
    T_body_lidar_ = XyzRpyToSE3(lidar_to_body_xyzrpy, extrinsic_rpy_degrees);

    LOG(INFO) << "RViz initial pose frame: " << options_.rviz_initial_pose_frame_
              << ", output child frame: " << options_.output_child_frame_id_
              << ", z offset: " << options_.rviz_initial_pose_z_offset_m_ << " m";

    LOG(INFO) << "online mode, creating ros2 node ... ";

    /// subscribers
    node_ = std::make_shared<rclcpp::Node>("lightning_slam");

    imu_topic_ = yaml.GetValue<std::string>("common", "imu_topic");
    cloud_topic_ = yaml.GetValue<std::string>("common", "lidar_topic");
    livox_topic_ = yaml.GetValue<std::string>("common", "livox_lidar_topic");

    rclcpp::QoS qos(10);
    rclcpp::SensorDataQoS sensor_qos;

    imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, sensor_qos, [this](sensor_msgs::msg::Imu::SharedPtr msg) {
            IMUPtr imu = std::make_shared<IMU>();
            imu->timestamp = ToSec(msg->header.stamp);
            imu->linear_acceleration =
                Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
            imu->angular_velocity = Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

            ProcessIMU(imu);
        });

    cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, sensor_qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
        });

    livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        livox_topic_, sensor_qos, [this](livox_ros_driver2::msg::CustomMsg ::SharedPtr cloud) {
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
        });

    if (options_.pub_tf_ || options_.enable_rviz_) {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
        loc_->SetTFCallback([this](const geometry_msgs::msg::TransformStamped& pose) {
            const auto output_pose = MakeOutputTransform(TransformMsgToSE3(pose), pose.header.stamp);
            if (options_.pub_tf_) {
                tf_broadcaster_->sendTransform(output_pose);
            }
            PublishOdom(output_pose);
        });
    }

    if (options_.enable_rviz_) {
        initialpose_sub_ = node_->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            options_.initialpose_topic_, qos,
            [this](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) { HandleInitialPose(msg); });

        rclcpp::QoS map_qos(1);
        map_qos.reliable().transient_local();
        map_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(options_.map_topic_, map_qos);
        odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>(options_.odom_topic_, qos);
        registered_scan_pub_ =
            node_->create_publisher<sensor_msgs::msg::PointCloud2>(options_.registered_scan_topic_, qos);
        obs_cloud_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(options_.obs_cloud_topic_, qos);
        loc_->SetPointcloudWorldCallback(
            [this](const sensor_msgs::msg::PointCloud2& cloud) { PublishRvizCloudAliases(cloud); });
        map_pub_timer_ = node_->create_wall_timer(std::chrono::seconds(2), [this]() { PublishRvizMap(); });
    }

    bool ret = loc_->Init(yaml_path, map_path_);
    if (ret) {
        LOG(INFO) << "online loc node has been created, sensor QoS subscriptions: imu=" << imu_topic_
                  << ", pointcloud2=" << cloud_topic_ << ", livox=" << livox_topic_;
        if (options_.enable_rviz_) {
            PublishRvizMap();
        }
    }

    return ret;
}

void LocSystem::SetInitPose(const SE3 &pose) {
    LOG(INFO) << "set init pose: " << pose.translation().transpose() << ", "
              << pose.unit_quaternion().coeffs().transpose();

    loc_->SetExternalPose(pose.unit_quaternion(), pose.translation());
    loc_started_ = true;
    PublishPoseVisualization(pose);
}

void LocSystem::HandleInitialPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr& msg) {
    if (msg == nullptr) {
        return;
    }

    if (!msg->header.frame_id.empty() && msg->header.frame_id != options_.map_frame_id_) {
        LOG(WARNING) << "initial pose frame_id is " << msg->header.frame_id << ", expected " << options_.map_frame_id_;
    }

    const auto& p = msg->pose.pose.position;
    const auto& q_msg = msg->pose.pose.orientation;
    Quatd q(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
    if (q.norm() < 1e-6) {
        LOG(ERROR) << "ignore invalid RViz initial pose quaternion";
        return;
    }
    q.normalize();

    const double z = p.z + options_.rviz_initial_pose_z_offset_m_;
    const SE3 rviz_pose(q, Vec3d(p.x, p.y, z));
    const SE3 internal_pose = RvizInitialPoseToInternalPose(rviz_pose);

    SetInitPose(internal_pose);
    LOG(INFO) << "accepted RViz initial pose from topic: " << options_.initialpose_topic_
              << ", input z: " << p.z << ", adjusted z: " << z;
}

SE3 LocSystem::RvizInitialPoseToInternalPose(const SE3& pose) const {
    if (!options_.rviz_initial_pose_is_body_frame_) {
        return pose;
    }

    return pose * T_body_lidar_;
}

SE3 LocSystem::InternalPoseToOutputPose(const SE3& pose) const { return pose * T_body_lidar_.inverse(); }

geometry_msgs::msg::TransformStamped LocSystem::MakeOutputTransform(const SE3& pose,
                                                                    const builtin_interfaces::msg::Time& stamp) const {
    const SE3 output_pose = InternalPoseToOutputPose(pose);
    geometry_msgs::msg::TransformStamped msg;
    msg.header.frame_id = options_.map_frame_id_;
    msg.header.stamp = stamp;
    msg.child_frame_id = options_.output_child_frame_id_;
    msg.transform.translation.x = output_pose.translation().x();
    msg.transform.translation.y = output_pose.translation().y();
    msg.transform.translation.z = output_pose.translation().z();
    msg.transform.rotation.x = output_pose.unit_quaternion().x();
    msg.transform.rotation.y = output_pose.unit_quaternion().y();
    msg.transform.rotation.z = output_pose.unit_quaternion().z();
    msg.transform.rotation.w = output_pose.unit_quaternion().w();
    return msg;
}

void LocSystem::PublishPoseVisualization(const SE3& pose) {
    if (node_ == nullptr) {
        return;
    }

    const auto msg = MakeOutputTransform(pose, node_->now());

    if (options_.pub_tf_ && tf_broadcaster_ != nullptr) {
        tf_broadcaster_->sendTransform(msg);
    }
    PublishOdom(msg);
}

void LocSystem::PublishOdom(const geometry_msgs::msg::TransformStamped& pose) {
    if (odom_pub_ == nullptr) {
        return;
    }

    nav_msgs::msg::Odometry odom;
    odom.header = pose.header;
    odom.header.frame_id = options_.map_frame_id_;
    odom.child_frame_id = pose.child_frame_id.empty() ? "base_link" : pose.child_frame_id;
    odom.pose.pose.position.x = pose.transform.translation.x;
    odom.pose.pose.position.y = pose.transform.translation.y;
    odom.pose.pose.position.z = pose.transform.translation.z;
    odom.pose.pose.orientation = pose.transform.rotation;
    odom_pub_->publish(odom);
}

void LocSystem::PublishRvizCloudAliases(const sensor_msgs::msg::PointCloud2& cloud) {
    if (registered_scan_pub_ == nullptr && obs_cloud_pub_ == nullptr) {
        return;
    }

    sensor_msgs::msg::PointCloud2 msg = cloud;
    msg.header.frame_id = options_.map_frame_id_;

    if (registered_scan_pub_ != nullptr) {
        registered_scan_pub_->publish(msg);
    }
    if (obs_cloud_pub_ != nullptr) {
        obs_cloud_pub_->publish(msg);
    }
}

void LocSystem::PublishRvizMap() {
    if (map_pub_ == nullptr) {
        return;
    }

    if (!rviz_map_msg_loaded_) {
        const std::filesystem::path global_map_path = std::filesystem::path(map_path_) / "global.pcd";
        if (!std::filesystem::exists(global_map_path)) {
            if (!rviz_map_missing_warned_) {
                LOG(WARNING) << "RViz map preview not published because global.pcd does not exist: "
                             << global_map_path;
                rviz_map_missing_warned_ = true;
            }
            return;
        }

        pcl::PCLPointCloud2 pcl_cloud;
        if (pcl::io::loadPCDFile(global_map_path.string(), pcl_cloud) != 0) {
            LOG(ERROR) << "failed to load RViz map preview: " << global_map_path;
            return;
        }

        pcl_conversions::fromPCL(pcl_cloud, rviz_map_msg_);
        rviz_map_msg_.header.frame_id = options_.map_frame_id_;
        rviz_map_msg_loaded_ = true;
        LOG(INFO) << "loaded RViz map preview: " << global_map_path << " -> " << options_.map_topic_;
    }

    rviz_map_msg_.header.stamp = node_->now();
    map_pub_->publish(rviz_map_msg_);
}

void LocSystem::ProcessIMU(const IMUPtr &imu) {
    if (loc_started_) {
        loc_->ProcessIMUMsg(imu);
    }
}

void LocSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr &cloud) {
    if (loc_started_) {
        loc_->ProcessLidarMsg(cloud);
    }
}

void LocSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr &cloud) {
    if (loc_started_) {
        loc_->ProcessLivoxLidarMsg(cloud);
    }
}

void LocSystem::Spin() {
    if (node_ != nullptr) {
        spin(node_);
    }
}

}  // namespace lightning
