#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include "super_odometry/sensor_data/pointcloud/point_os.h"

namespace super_odometry {
namespace {

constexpr int64_t kNanosecondsPerSecond = 1000000000LL;

int64_t stampToNanoseconds(const builtin_interfaces::msg::Time &stamp) {
    return static_cast<int64_t>(stamp.sec) * kNanosecondsPerSecond +
           static_cast<int64_t>(stamp.nanosec);
}

builtin_interfaces::msg::Time nanosecondsToStamp(int64_t stamp_ns) {
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(stamp_ns / kNanosecondsPerSecond);
    stamp.nanosec = static_cast<uint32_t>(stamp_ns % kNanosecondsPerSecond);
    return stamp;
}

Eigen::Isometry3d makeTransform(const std::vector<double> &xyzrpy, bool rpy_degrees) {
    const double scale = rpy_degrees ? M_PI / 180.0 : 1.0;
    const double roll = xyzrpy[3] * scale;
    const double pitch = xyzrpy[4] * scale;
    const double yaw = xyzrpy[5] * scale;

    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = Eigen::Vector3d(xyzrpy[0], xyzrpy[1], xyzrpy[2]);
    transform.linear() =
        (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    return transform;
}

bool isIdentity(const Eigen::Isometry3d &transform) {
    return transform.translation().norm() < 1e-9 &&
           (transform.linear() - Eigen::Matrix3d::Identity()).norm() < 1e-9;
}

struct FieldInfo {
    int offset = -1;
    uint8_t datatype = 0;
    uint32_t count = 0;

    bool valid() const {
        return offset >= 0 && count > 0;
    }
};

FieldInfo findField(const sensor_msgs::msg::PointCloud2 &cloud,
                    const std::vector<std::string> &names) {
    for (const auto &name : names) {
        for (const auto &field : cloud.fields) {
            if (field.name == name) {
                return FieldInfo{static_cast<int>(field.offset), field.datatype, field.count};
            }
        }
    }
    return FieldInfo{};
}

bool readScalar(const sensor_msgs::msg::PointCloud2 &cloud,
                const std::size_t point_base,
                const FieldInfo &field,
                double &value) {
    if (!field.valid()) {
        return false;
    }
    const std::size_t offset = point_base + static_cast<std::size_t>(field.offset);
    if (offset >= cloud.data.size()) {
        return false;
    }

    switch (field.datatype) {
        case sensor_msgs::msg::PointField::INT8: {
            int8_t tmp;
            std::memcpy(&tmp, &cloud.data[offset], sizeof(tmp));
            value = tmp;
            return true;
        }
        case sensor_msgs::msg::PointField::UINT8: {
            uint8_t tmp;
            std::memcpy(&tmp, &cloud.data[offset], sizeof(tmp));
            value = tmp;
            return true;
        }
        case sensor_msgs::msg::PointField::INT16: {
            int16_t tmp;
            std::memcpy(&tmp, &cloud.data[offset], sizeof(tmp));
            value = tmp;
            return true;
        }
        case sensor_msgs::msg::PointField::UINT16: {
            uint16_t tmp;
            std::memcpy(&tmp, &cloud.data[offset], sizeof(tmp));
            value = tmp;
            return true;
        }
        case sensor_msgs::msg::PointField::INT32: {
            int32_t tmp;
            std::memcpy(&tmp, &cloud.data[offset], sizeof(tmp));
            value = tmp;
            return true;
        }
        case sensor_msgs::msg::PointField::UINT32: {
            uint32_t tmp;
            std::memcpy(&tmp, &cloud.data[offset], sizeof(tmp));
            value = tmp;
            return true;
        }
        case sensor_msgs::msg::PointField::FLOAT32: {
            float tmp;
            std::memcpy(&tmp, &cloud.data[offset], sizeof(tmp));
            value = tmp;
            return true;
        }
        case sensor_msgs::msg::PointField::FLOAT64: {
            double tmp;
            std::memcpy(&tmp, &cloud.data[offset], sizeof(tmp));
            value = tmp;
            return true;
        }
        default:
            return false;
    }
}

std::size_t pointBaseOffset(const sensor_msgs::msg::PointCloud2 &cloud, std::size_t point_index) {
    const std::size_t row = point_index / cloud.width;
    const std::size_t col = point_index % cloud.width;
    return row * cloud.row_step + col * cloud.point_step;
}

float normalizePointTime(double raw_time, const std::string &field_name, int64_t cloud_stamp_ns) {
    if (!std::isfinite(raw_time)) {
        return 0.0f;
    }

    double relative_sec = raw_time;
    if (field_name == "t" || field_name == "timestamp_ns" || field_name == "offset_time") {
        relative_sec = raw_time * 1e-9;
    } else if (field_name == "timestamp" || field_name == "timeSecond") {
        if (raw_time > 1e12) {
            relative_sec = raw_time * 1e-9 - static_cast<double>(cloud_stamp_ns) * 1e-9;
        } else if (raw_time > 1e5) {
            relative_sec = raw_time - static_cast<double>(cloud_stamp_ns) * 1e-9;
        }
    }

    if (relative_sec < -1.0 || relative_sec > 10.0) {
        return 0.0f;
    }
    return static_cast<float>(std::max(0.0, relative_sec));
}

}  // namespace

class DualLidarFusionNode : public rclcpp::Node {
public:
    DualLidarFusionNode() : Node("dual_lidar_fusion_node") {
        front_topic_ = declare_parameter<std::string>("front_lidar_topic", "/front/lidar_points");
        rear_topic_ = declare_parameter<std::string>("rear_lidar_topic", "/rear/lidar_points");
        merged_topic_ = declare_parameter<std::string>("merged_lidar_topic", "/dual_lidar/points");
        output_frame_ = declare_parameter<std::string>("output_frame", "sensor");
        sync_tolerance_ns_ = static_cast<int64_t>(
            declare_parameter<double>("sync_tolerance_ms", 2.0) * 1e6);
        max_queue_size_ = static_cast<std::size_t>(declare_parameter<int>("max_queue_size", 10));
        sort_by_time_ = declare_parameter<bool>("sort_by_time", true);
        enable_min_range_filter_ = declare_parameter<bool>("enable_min_range_filter", false);
        min_range_ = std::max(0.0, declare_parameter<double>("min_range", 0.0));
        const bool rpy_degrees = declare_parameter<bool>("extrinsic_rpy_degrees", false);
        scan_line_ = declare_parameter<int>("scan_line", 128);
        front_ring_offset_ = declare_parameter<int>("front_ring_offset", 0);
        rear_ring_offset_ = declare_parameter<int>("rear_ring_offset", 0);

        const auto front_xyzrpy = readXyzRpyParameter("front_lidar_to_output_xyzrpy");
        const auto rear_xyzrpy = readXyzRpyParameter("rear_lidar_to_output_xyzrpy");
        front_to_output_ = makeTransform(front_xyzrpy, rpy_degrees);
        rear_to_output_ = makeTransform(rear_xyzrpy, rpy_degrees);

        rclcpp::QoS qos = rclcpp::SensorDataQoS().keep_last(max_queue_size_);
        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(merged_topic_, qos);
        front_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            front_topic_, qos,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                handleCloud(std::move(msg), front_queue_);
            });
        rear_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            rear_topic_, qos,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                handleCloud(std::move(msg), rear_queue_);
            });

        if (isIdentity(front_to_output_) && isIdentity(rear_to_output_)) {
            RCLCPP_WARN(get_logger(),
                        "Both LiDAR extrinsics are identity. This is usable as a placeholder, "
                        "but the merged cloud is geometrically meaningful only after calibration.");
        }

        RCLCPP_INFO(get_logger(),
                    "Dual LiDAR fusion: front=%s rear=%s merged=%s output_frame=%s tolerance=%.3f ms enable_min_range_filter=%s min_range=%.3f",
                    front_topic_.c_str(), rear_topic_.c_str(), merged_topic_.c_str(),
                    output_frame_.c_str(), sync_tolerance_ns_ / 1e6,
                    enable_min_range_filter_ ? "true" : "false", min_range_);
    }

private:
    std::vector<double> readXyzRpyParameter(const std::string &name) {
        auto value = declare_parameter<std::vector<double>>(
            name, std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        if (value.size() != 6) {
            RCLCPP_WARN(get_logger(),
                        "%s must contain [x, y, z, roll, pitch, yaw]. Falling back to identity.",
                        name.c_str());
            value = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        return value;
    }

    void handleCloud(sensor_msgs::msg::PointCloud2::SharedPtr msg,
                     std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> &queue) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue.push_back(std::move(msg));
        while (queue.size() > max_queue_size_) {
            queue.pop_front();
        }
        synchronizeAndPublish();
    }

    void synchronizeAndPublish() {
        while (!front_queue_.empty() && !rear_queue_.empty()) {
            const auto front_stamp_ns = stampToNanoseconds(front_queue_.front()->header.stamp);
            const auto best_rear = findBestMatch(rear_queue_, front_stamp_ns);

            if (best_rear < rear_queue_.size()) {
                const auto rear_stamp_ns = stampToNanoseconds(rear_queue_[best_rear]->header.stamp);
                if (std::llabs(front_stamp_ns - rear_stamp_ns) <= sync_tolerance_ns_) {
                    const auto front = front_queue_.front();
                    const auto rear = rear_queue_[best_rear];
                    front_queue_.pop_front();
                    rear_queue_.erase(rear_queue_.begin() + static_cast<std::ptrdiff_t>(best_rear));
                    publishMergedCloud(*front, *rear);
                    continue;
                }
            }

            const auto rear_front_stamp_ns = stampToNanoseconds(rear_queue_.front()->header.stamp);
            if (front_stamp_ns + sync_tolerance_ns_ < rear_front_stamp_ns) {
                front_queue_.pop_front();
            } else if (rear_front_stamp_ns + sync_tolerance_ns_ < front_stamp_ns) {
                rear_queue_.pop_front();
            } else {
                break;
            }
        }
    }

    std::size_t findBestMatch(
        const std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> &queue,
        int64_t target_stamp_ns) const {
        std::size_t best = queue.size();
        int64_t best_delta = std::numeric_limits<int64_t>::max();
        for (std::size_t i = 0; i < queue.size(); ++i) {
            const int64_t delta =
                std::llabs(stampToNanoseconds(queue[i]->header.stamp) - target_stamp_ns);
            if (delta < best_delta) {
                best_delta = delta;
                best = i;
            }
        }
        return best;
    }

    void publishMergedCloud(const sensor_msgs::msg::PointCloud2 &front,
                            const sensor_msgs::msg::PointCloud2 &rear) {
        const int64_t output_stamp_ns = std::min(stampToNanoseconds(front.header.stamp),
                                                 stampToNanoseconds(rear.header.stamp));
        pcl::PointCloud<point_os::PointcloudXYZITR> merged;
        merged.reserve(static_cast<std::size_t>(front.width) * front.height +
                       static_cast<std::size_t>(rear.width) * rear.height);

        appendTransformedCloud(front, front_to_output_, front_ring_offset_, output_stamp_ns, merged);
        appendTransformedCloud(rear, rear_to_output_, rear_ring_offset_, output_stamp_ns, merged);

        if (merged.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Merged cloud is empty after field validation/filtering.");
            return;
        }

        if (sort_by_time_) {
            std::stable_sort(merged.begin(), merged.end(),
                             [](const auto &lhs, const auto &rhs) {
                                 return lhs.time < rhs.time;
                             });
        }

        merged.width = static_cast<uint32_t>(merged.size());
        merged.height = 1;
        merged.is_dense = false;

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(merged, output);
        output.header.stamp = nanosecondsToStamp(output_stamp_ns);
        output.header.frame_id = output_frame_;
        pub_->publish(output);
    }

    void appendTransformedCloud(const sensor_msgs::msg::PointCloud2 &cloud,
                                const Eigen::Isometry3d &source_to_output,
                                int ring_offset,
                                int64_t output_stamp_ns,
                                pcl::PointCloud<point_os::PointcloudXYZITR> &merged) {
        if (cloud.width == 0 || cloud.height == 0 || cloud.point_step == 0) {
            return;
        }

        const FieldInfo x = findField(cloud, {"x"});
        const FieldInfo y = findField(cloud, {"y"});
        const FieldInfo z = findField(cloud, {"z"});
        if (!x.valid() || !y.valid() || !z.valid()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Dropping cloud from frame '%s': missing x/y/z fields.",
                                 cloud.header.frame_id.c_str());
            return;
        }

        const FieldInfo intensity = findField(cloud, {"intensity", "reflectivity"});
        const auto time_field = selectTimeField(cloud);
        const FieldInfo ring = findField(cloud, {"ring", "laser_id", "channel", "line"});

        const double stamp_offset_sec =
            static_cast<double>(stampToNanoseconds(cloud.header.stamp) - output_stamp_ns) * 1e-9;
        const int64_t cloud_stamp_ns = stampToNanoseconds(cloud.header.stamp);
        const std::size_t point_count = static_cast<std::size_t>(cloud.width) * cloud.height;

        for (std::size_t i = 0; i < point_count; ++i) {
            const std::size_t base = pointBaseOffset(cloud, i);

            double raw_x = 0.0;
            double raw_y = 0.0;
            double raw_z = 0.0;
            if (!readScalar(cloud, base, x, raw_x) ||
                !readScalar(cloud, base, y, raw_y) ||
                !readScalar(cloud, base, z, raw_z) ||
                !std::isfinite(raw_x) ||
                !std::isfinite(raw_y) ||
                !std::isfinite(raw_z)) {
                continue;
            }

            if (enable_min_range_filter_ && min_range_ > 0.0) {
                const double range_sq = raw_x * raw_x + raw_y * raw_y + raw_z * raw_z;
                if (range_sq <= min_range_ * min_range_) {
                    continue;
                }
            }

            const Eigen::Vector3d transformed =
                source_to_output * Eigen::Vector3d(raw_x, raw_y, raw_z);

            double raw_intensity = 0.0;
            readScalar(cloud, base, intensity, raw_intensity);

            double relative_time = 0.0;
            if (time_field.info.valid()) {
                readScalar(cloud, base, time_field.info, relative_time);
                relative_time = normalizePointTime(relative_time, time_field.name, cloud_stamp_ns);
            }
            relative_time += stamp_offset_sec;

            double raw_ring = 0.0;
            int adjusted_ring = 0;
            if (readScalar(cloud, base, ring, raw_ring)) {
                adjusted_ring = static_cast<int>(std::llround(raw_ring)) + ring_offset;
            } else if (scan_line_ > 0) {
                adjusted_ring = static_cast<int>(i % static_cast<std::size_t>(scan_line_)) + ring_offset;
            }
            adjusted_ring = std::max(0, std::min(65535, adjusted_ring));

            merged.push_back(point_os::PointcloudXYZITR::make(
                static_cast<float>(transformed.x()),
                static_cast<float>(transformed.y()),
                static_cast<float>(transformed.z()),
                static_cast<float>(raw_intensity),
                static_cast<float>(relative_time),
                static_cast<uint16_t>(adjusted_ring)));
        }
    }

    struct NamedField {
        std::string name;
        FieldInfo info;
    };

    NamedField selectTimeField(const sensor_msgs::msg::PointCloud2 &cloud) const {
        const std::vector<std::string> names = {
            "time", "t", "offset_time", "timestamp", "timestamp_ns", "timeSecond"};
        for (const auto &name : names) {
            FieldInfo field = findField(cloud, {name});
            if (field.valid()) {
                return NamedField{name, field};
            }
        }
        return NamedField{"", FieldInfo{}};
    }

    std::string front_topic_;
    std::string rear_topic_;
    std::string merged_topic_;
    std::string output_frame_;
    int64_t sync_tolerance_ns_;
    std::size_t max_queue_size_;
    bool sort_by_time_;
    bool enable_min_range_filter_;
    double min_range_;
    int scan_line_;
    int front_ring_offset_;
    int rear_ring_offset_;
    Eigen::Isometry3d front_to_output_ = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d rear_to_output_ = Eigen::Isometry3d::Identity();

    std::mutex mutex_;
    std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> front_queue_;
    std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> rear_queue_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr front_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr rear_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

}  // namespace super_odometry

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<super_odometry::DualLidarFusionNode>());
    rclcpp::shutdown();
    return 0;
}
