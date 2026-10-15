#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
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

struct NamedField {
    std::string name;
    FieldInfo info;
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

NamedField findNamedField(const sensor_msgs::msg::PointCloud2 &cloud,
                          const std::vector<std::string> &names) {
    for (const auto &name : names) {
        FieldInfo field = findField(cloud, {name});
        if (field.valid()) {
            return NamedField{name, field};
        }
    }
    return NamedField{"", FieldInfo{}};
}

std::size_t scalarSize(uint8_t datatype) {
    switch (datatype) {
        case sensor_msgs::msg::PointField::INT8:
        case sensor_msgs::msg::PointField::UINT8:
            return 1;
        case sensor_msgs::msg::PointField::INT16:
        case sensor_msgs::msg::PointField::UINT16:
            return 2;
        case sensor_msgs::msg::PointField::INT32:
        case sensor_msgs::msg::PointField::UINT32:
        case sensor_msgs::msg::PointField::FLOAT32:
            return 4;
        case sensor_msgs::msg::PointField::FLOAT64:
            return 8;
        default:
            return 0;
    }
}

bool readScalar(const sensor_msgs::msg::PointCloud2 &cloud,
                std::size_t point_base,
                const FieldInfo &field,
                double &value) {
    if (!field.valid()) {
        return false;
    }
    const std::size_t offset = point_base + static_cast<std::size_t>(field.offset);
    const std::size_t size = scalarSize(field.datatype);
    if (size == 0 || offset + size > cloud.data.size()) {
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

bool normalizePointTime(double raw_time,
                        const std::string &field_name,
                        int64_t cloud_stamp_ns,
                        float &relative_time) {
    if (!std::isfinite(raw_time)) {
        return false;
    }

    double relative_sec = raw_time;
    if (field_name == "t" || field_name == "offset_time") {
        relative_sec = raw_time * 1e-9;
    } else if (field_name == "timestamp_ns") {
        if (raw_time > 1e12) {
            relative_sec = raw_time * 1e-9 - static_cast<double>(cloud_stamp_ns) * 1e-9;
        } else {
            relative_sec = raw_time * 1e-9;
        }
    } else if (field_name == "timestamp" || field_name == "time" || field_name == "timeSecond") {
        if (raw_time > 1e12) {
            relative_sec = raw_time * 1e-9 - static_cast<double>(cloud_stamp_ns) * 1e-9;
        } else if (raw_time > 1e5) {
            relative_sec = raw_time - static_cast<double>(cloud_stamp_ns) * 1e-9;
        }
    }

    if (relative_sec < -1e-3 || relative_sec > 10.0) {
        return false;
    }
    relative_time = static_cast<float>(std::max(0.0, relative_sec));
    return true;
}

struct CloudStats {
    std::size_t input_points = 0;
    std::size_t output_points = 0;
    std::size_t skipped_xyz = 0;
    std::size_t skipped_range = 0;
    std::size_t skipped_time = 0;
    std::size_t skipped_ring = 0;
    int ring_min = std::numeric_limits<int>::max();
    int ring_max = std::numeric_limits<int>::min();
    std::size_t unique_rings = 0;
    std::vector<bool> ring_seen = std::vector<bool>(65536, false);
    float time_min = std::numeric_limits<float>::max();
    float time_max = std::numeric_limits<float>::lowest();
    std::size_t valid_times = 0;

    void addRing(int ring) {
        ring_min = std::min(ring_min, ring);
        ring_max = std::max(ring_max, ring);
        if (ring >= 0 && ring < static_cast<int>(ring_seen.size()) &&
            !ring_seen[static_cast<std::size_t>(ring)]) {
            ring_seen[static_cast<std::size_t>(ring)] = true;
            ++unique_rings;
        }
    }

    void addTime(float time) {
        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        ++valid_times;
    }

    double duration() const {
        if (valid_times == 0) {
            return 0.0;
        }
        return static_cast<double>(time_max - time_min);
    }
};

std::string fieldList(const sensor_msgs::msg::PointCloud2 &cloud) {
    std::ostringstream stream;
    for (std::size_t i = 0; i < cloud.fields.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        stream << cloud.fields[i].name;
    }
    return stream.str();
}

}  // namespace

class HesaiToSuperOdomNode : public rclcpp::Node {
public:
    HesaiToSuperOdomNode() : Node("hesai_to_superodom_node") {
        input_topic_ = declare_parameter<std::string>("input_point_cloud_topic", "/hesai/jt128/points");
        output_topic_ = declare_parameter<std::string>("output_point_cloud_topic",
                                                       "/hesai/jt128/points_superodom");
        output_frame_ = declare_parameter<std::string>("output_frame", "sensor");
        scan_line_ = declare_parameter<int>("scan_line", 128);
        ring_offset_ = declare_parameter<int>("ring_offset", 0);
        sort_by_time_ = declare_parameter<bool>("sort_by_time", true);
        require_point_time_ = declare_parameter<bool>("require_point_time", true);
        require_ring_ = declare_parameter<bool>("require_ring", true);
        allow_ring_fallback_ = declare_parameter<bool>("allow_ring_fallback", false);
        max_point_time_sec_ = declare_parameter<double>("max_point_time_sec", 0.5);
        enable_min_range_filter_ = declare_parameter<bool>("enable_min_range_filter", false);
        min_range_ = std::max(0.0, declare_parameter<double>("min_range", 0.0));
        const bool rpy_degrees = declare_parameter<bool>("extrinsic_rpy_degrees", false);
        const auto xyzrpy = readXyzRpyParameter("lidar_to_output_xyzrpy");
        lidar_to_output_ = makeTransform(xyzrpy, rpy_degrees);

        rclcpp::QoS qos = rclcpp::SensorDataQoS().keep_last(5);
        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, qos);
        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, qos, std::bind(&HesaiToSuperOdomNode::cloudHandler, this,
                                         std::placeholders::_1));

        if (isIdentity(lidar_to_output_)) {
            RCLCPP_INFO(get_logger(),
                        "JT128 lidar_to_output_xyzrpy is identity; publishing raw LiDAR-frame points. "
                        "Feature extraction applies IMU-derived gravity alignment after IMU init.");
        }
        RCLCPP_INFO(get_logger(), "Hesai JT128 adapter: %s -> %s, output_frame=%s",
                    input_topic_.c_str(), output_topic_.c_str(), output_frame_.c_str());
        RCLCPP_INFO(get_logger(),
                    "Hesai JT128 contract: scan_line=%d require_point_time=%s require_ring=%s allow_ring_fallback=%s max_point_time_sec=%.3f enable_min_range_filter=%s min_range=%.3f",
                    scan_line_,
                    require_point_time_ ? "true" : "false",
                    require_ring_ ? "true" : "false",
                    allow_ring_fallback_ ? "true" : "false",
                    max_point_time_sec_,
                    enable_min_range_filter_ ? "true" : "false",
                    min_range_);
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

    void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (msg->width == 0 || msg->height == 0 || msg->point_step == 0) {
            return;
        }

        const FieldInfo x = findField(*msg, {"x"});
        const FieldInfo y = findField(*msg, {"y"});
        const FieldInfo z = findField(*msg, {"z"});
        if (!x.valid() || !y.valid() || !z.valid()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Dropping JT128 cloud: missing x/y/z fields.");
            return;
        }

        const FieldInfo intensity = findField(*msg, {"intensity", "reflectivity"});
        const auto ring_field = selectRingField(*msg);
        const auto time_field = selectTimeField(*msg);
        if (require_point_time_ && !time_field.info.valid()) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "Dropping JT128 cloud: missing required per-point time field. Available fields: %s",
                fieldList(*msg).c_str());
            return;
        }
        if (!require_point_time_ && !time_field.info.valid()) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "JT128 cloud has no recognized time field; publishing zero point time because require_point_time=false.");
        }
        if (require_ring_ && !ring_field.info.valid() && !allow_ring_fallback_) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "Dropping JT128 cloud: missing required ring/channel field. Available fields: %s",
                fieldList(*msg).c_str());
            return;
        }
        if (!ring_field.info.valid() && allow_ring_fallback_) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "JT128 cloud has no recognized ring field; using explicit fallback point_index %% scan_line.");
        }

        pcl::PointCloud<point_os::PointcloudXYZITR> output_cloud;
        const std::size_t point_count = static_cast<std::size_t>(msg->width) * msg->height;
        output_cloud.reserve(point_count);
        const int64_t stamp_ns = stampToNanoseconds(msg->header.stamp);
        CloudStats stats;
        stats.input_points = point_count;

        for (std::size_t i = 0; i < point_count; ++i) {
            const std::size_t base = pointBaseOffset(*msg, i);
            double raw_x = 0.0;
            double raw_y = 0.0;
            double raw_z = 0.0;
            if (!readScalar(*msg, base, x, raw_x) ||
                !readScalar(*msg, base, y, raw_y) ||
                !readScalar(*msg, base, z, raw_z) ||
                !std::isfinite(raw_x) ||
                !std::isfinite(raw_y) ||
                !std::isfinite(raw_z)) {
                ++stats.skipped_xyz;
                continue;
            }

            if (enable_min_range_filter_ && min_range_ > 0.0) {
                const double range_sq = raw_x * raw_x + raw_y * raw_y + raw_z * raw_z;
                if (range_sq <= min_range_ * min_range_) {
                    ++stats.skipped_range;
                    continue;
                }
            }

            const Eigen::Vector3d transformed =
                lidar_to_output_ * Eigen::Vector3d(raw_x, raw_y, raw_z);

            double raw_intensity = 0.0;
            readScalar(*msg, base, intensity, raw_intensity);

            float point_time = 0.0f;
            if (time_field.info.valid()) {
                double raw_time = 0.0;
                if (!readScalar(*msg, base, time_field.info, raw_time) ||
                    !normalizePointTime(raw_time, time_field.name, stamp_ns, point_time)) {
                    ++stats.skipped_time;
                    if (require_point_time_) {
                        continue;
                    }
                    point_time = 0.0f;
                }
            }
            if (time_field.info.valid()) {
                stats.addTime(point_time);
            }

            double raw_ring = 0.0;
            int adjusted_ring = 0;
            if (readScalar(*msg, base, ring_field.info, raw_ring) && std::isfinite(raw_ring)) {
                adjusted_ring = static_cast<int>(std::llround(raw_ring)) + ring_offset_;
            } else if (allow_ring_fallback_ && scan_line_ > 0) {
                adjusted_ring = static_cast<int>(i % static_cast<std::size_t>(scan_line_));
            } else {
                ++stats.skipped_ring;
                if (require_ring_) {
                    continue;
                }
            }
            adjusted_ring = std::max(0, std::min(65535, adjusted_ring));
            stats.addRing(adjusted_ring);

            output_cloud.push_back(point_os::PointcloudXYZITR::make(
                static_cast<float>(transformed.x()),
                static_cast<float>(transformed.y()),
                static_cast<float>(transformed.z()),
                static_cast<float>(raw_intensity),
                point_time,
                static_cast<uint16_t>(adjusted_ring)));
        }
        stats.output_points = output_cloud.size();

        if (output_cloud.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Converted JT128 cloud is empty.");
            return;
        }
        if (require_point_time_ && stats.valid_times > 1) {
            const double duration = stats.duration();
            if (duration <= 1e-6 || duration > max_point_time_sec_) {
                RCLCPP_ERROR_THROTTLE(
                    get_logger(), *get_clock(), 5000,
                    "Dropping JT128 cloud: invalid relative point-time span %.9f s from field '%s'. Expected nonzero span <= %.3f s.",
                    duration, time_field.name.c_str(), max_point_time_sec_);
                return;
            }
        }

        if (sort_by_time_) {
            std::stable_sort(output_cloud.begin(), output_cloud.end(),
                             [](const auto &lhs, const auto &rhs) {
                                 return lhs.time < rhs.time;
                             });
        }

        if (!logged_first_cloud_) {
            logged_first_cloud_ = true;
            RCLCPP_INFO(get_logger(),
                        "Converted first JT128 cloud: input_frame=%s output_frame=%s input_points=%zu output_points=%zu fields=[%s]",
                        msg->header.frame_id.c_str(), output_frame_.c_str(),
                        stats.input_points, stats.output_points, fieldList(*msg).c_str());
            RCLCPP_INFO(get_logger(),
                        "JT128 first cloud contract: time_field=%s relative_time=[%.9f, %.9f] duration=%.9f ring_field=%s ring_range=[%d, %d] unique_rings=%zu scan_line=%d enable_min_range_filter=%s min_range=%.3f skipped_xyz=%zu skipped_range=%zu skipped_time=%zu skipped_ring=%zu",
                        time_field.name.empty() ? "<none>" : time_field.name.c_str(),
                        stats.valid_times == 0 ? 0.0 : static_cast<double>(stats.time_min),
                        stats.valid_times == 0 ? 0.0 : static_cast<double>(stats.time_max),
                        stats.duration(),
                        ring_field.name.empty() ? "<fallback>" : ring_field.name.c_str(),
                        stats.ring_min == std::numeric_limits<int>::max() ? -1 : stats.ring_min,
                        stats.ring_max == std::numeric_limits<int>::min() ? -1 : stats.ring_max,
                        stats.unique_rings,
                        scan_line_,
                        enable_min_range_filter_ ? "true" : "false",
                        min_range_,
                        stats.skipped_xyz,
                        stats.skipped_range,
                        stats.skipped_time,
                        stats.skipped_ring);
        }

        output_cloud.width = static_cast<uint32_t>(output_cloud.size());
        output_cloud.height = 1;
        output_cloud.is_dense = false;

        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(output_cloud, output_msg);
        output_msg.header.stamp = msg->header.stamp;
        output_msg.header.frame_id = output_frame_;
        pub_->publish(output_msg);
    }

    NamedField selectTimeField(const sensor_msgs::msg::PointCloud2 &cloud) const {
        return findNamedField(
            cloud, {"time", "t", "offset_time", "timestamp", "timestamp_ns", "timeSecond"});
    }

    NamedField selectRingField(const sensor_msgs::msg::PointCloud2 &cloud) const {
        return findNamedField(cloud, {"ring", "laser_id", "channel", "line"});
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string output_frame_;
    int scan_line_;
    int ring_offset_;
    bool sort_by_time_;
    bool require_point_time_;
    bool require_ring_;
    bool allow_ring_fallback_;
    bool enable_min_range_filter_;
    double max_point_time_sec_;
    double min_range_;
    bool logged_first_cloud_ = false;
    Eigen::Isometry3d lidar_to_output_ = Eigen::Isometry3d::Identity();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

}  // namespace super_odometry

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<super_odometry::HesaiToSuperOdomNode>());
    rclcpp::shutdown();
    return 0;
}
