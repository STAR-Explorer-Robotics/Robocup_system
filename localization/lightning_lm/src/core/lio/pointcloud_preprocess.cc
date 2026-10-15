#include "pointcloud_preprocess.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <execution>

#include <glog/logging.h>

namespace lightning {
namespace {

const sensor_msgs::msg::PointField *FindField(const sensor_msgs::msg::PointCloud2 &msg, const std::string &name) {
    const auto iter = std::find_if(msg.fields.begin(), msg.fields.end(),
                                   [&name](const sensor_msgs::msg::PointField &field) { return field.name == name; });
    return iter == msg.fields.end() ? nullptr : &(*iter);
}

template <typename T>
T ReadUnaligned(const std::uint8_t *data) {
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

bool ReadNumericField(const std::uint8_t *point, const sensor_msgs::msg::PointField &field, double &value) {
    const auto *data = point + field.offset;
    switch (field.datatype) {
        case sensor_msgs::msg::PointField::INT8:
            value = ReadUnaligned<std::int8_t>(data);
            return true;
        case sensor_msgs::msg::PointField::UINT8:
            value = ReadUnaligned<std::uint8_t>(data);
            return true;
        case sensor_msgs::msg::PointField::INT16:
            value = ReadUnaligned<std::int16_t>(data);
            return true;
        case sensor_msgs::msg::PointField::UINT16:
            value = ReadUnaligned<std::uint16_t>(data);
            return true;
        case sensor_msgs::msg::PointField::INT32:
            value = ReadUnaligned<std::int32_t>(data);
            return true;
        case sensor_msgs::msg::PointField::UINT32:
            value = ReadUnaligned<std::uint32_t>(data);
            return true;
        case sensor_msgs::msg::PointField::FLOAT32:
            value = ReadUnaligned<float>(data);
            return true;
        case sensor_msgs::msg::PointField::FLOAT64:
            value = ReadUnaligned<double>(data);
            return true;
        default:
            return false;
    }
}

bool ReadRingField(const std::uint8_t *point, const sensor_msgs::msg::PointField &field, std::uint16_t &value) {
    double numeric_value = 0.0;
    if (!ReadNumericField(point, field, numeric_value) || numeric_value < 0.0 ||
        numeric_value > std::numeric_limits<std::uint16_t>::max()) {
        return false;
    }

    value = static_cast<std::uint16_t>(numeric_value);
    return true;
}

}  // namespace

void PointCloudPreprocess::Set(LidarType lid_type, double bld, int pfilt_num) {
    lidar_type_ = lid_type;
    blind_ = bld;
    point_filter_num_ = pfilt_num;
}

void PointCloudPreprocess::Process(const sensor_msgs::msg::PointCloud2 ::SharedPtr &msg, PointCloudType::Ptr &pcl_out) {
    switch (lidar_type_) {
        case LidarType::AVIA:
            LivoxPointCloud2Handler(msg);
            break;

        case LidarType::OUST64:
            Oust64Handler(msg);
            break;

        case LidarType::VELO32:
            VelodyneHandler(msg);
            break;

        case LidarType::ROBOSENSE:
            RoboSenseHandler(msg);
            break;

        case LidarType::JT128:
            JT128Handler(msg);
            break;

        default:
            LOG(ERROR) << "Error LiDAR Type";
            break;
    }
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::Process(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg,
                                   PointCloudType::Ptr &pcl_out) {
    cloud_out_.clear();
    cloud_full_.clear();

    int plsize = msg->point_num;

    cloud_out_.reserve(plsize);
    cloud_full_.resize(plsize);

    std::vector<char> is_valid_pt(plsize, 0);
    std::vector<uint> index(plsize - 1);
    for (uint i = 0; i < plsize - 1; ++i) {
        index[i] = i + 1;  // 从1开始
    }

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const uint &i) {
        // if ((msg->points[i].line < num_scans_) &&
        // ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
        if (i % point_filter_num_ != 0) {
            return;
        }

        cloud_full_[i].x = msg->points[i].x;
        cloud_full_[i].y = msg->points[i].y;
        cloud_full_[i].z = msg->points[i].z;
        cloud_full_[i].intensity = msg->points[i].reflectivity;

        // use curvature as time of each laser points, curvature unit: ms
        cloud_full_[i].time = msg->points[i].offset_time / double(1000000);

        if (cloud_full_[i].z < height_min_ || cloud_full_[i].z > height_max_) {
            return;
        }

        if ((abs(cloud_full_[i].x - cloud_full_[i - 1].x) > 1e-7) ||
            (abs(cloud_full_[i].y - cloud_full_[i - 1].y) > 1e-7) ||
            (abs(cloud_full_[i].z - cloud_full_[i - 1].z) > 1e-7) &&
                (cloud_full_[i].x * cloud_full_[i].x + cloud_full_[i].y * cloud_full_[i].y +
                     cloud_full_[i].z * cloud_full_[i].z >
                 (blind_ * blind_))) {
            is_valid_pt[i] = 1;
        }

        // }
    });

    for (uint i = 1; i < plsize; i++) {
        if (is_valid_pt[i]) {
            cloud_out_.points.push_back(cloud_full_[i]);
        }
    }

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::LivoxPointCloud2Handler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    if (msg->is_bigendian) {
        LOG(ERROR) << "Livox PointCloud2 big-endian format is not supported";
        return;
    }

    const auto *x_field = FindField(*msg, "x");
    const auto *y_field = FindField(*msg, "y");
    const auto *z_field = FindField(*msg, "z");
    const auto *intensity_field = FindField(*msg, "intensity");
    const auto *timestamp_field = FindField(*msg, "timestamp");
    const auto *time_field = FindField(*msg, "time");

    if (x_field == nullptr || y_field == nullptr || z_field == nullptr || intensity_field == nullptr) {
        LOG(ERROR) << "Livox PointCloud2 requires x/y/z/intensity fields";
        return;
    }

    const double blind_sq = blind_ * blind_;
    const double range_max_sq = static_cast<double>(range_max_) * range_max_;
    const double head_time = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;
    const std::size_t point_count = static_cast<std::size_t>(msg->width) * msg->height;
    cloud_out_.reserve(point_count);

    bool has_prev = false;
    double prev_x = 0.0;
    double prev_y = 0.0;
    double prev_z = 0.0;

    std::size_t point_index = 0;
    for (std::uint32_t row = 0; row < msg->height; ++row) {
        const auto *row_data = msg->data.data() + static_cast<std::size_t>(row) * msg->row_step;
        for (std::uint32_t col = 0; col < msg->width; ++col, ++point_index) {
            const auto *point = row_data + static_cast<std::size_t>(col) * msg->point_step;

            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            double intensity = 0.0;
            double point_time = 0.0;

            const bool readable = ReadNumericField(point, *x_field, x) && ReadNumericField(point, *y_field, y) &&
                                  ReadNumericField(point, *z_field, z) &&
                                  ReadNumericField(point, *intensity_field, intensity);
            if (!readable) {
                continue;
            }
            if (timestamp_field != nullptr) {
                if (!ReadNumericField(point, *timestamp_field, point_time)) {
                    continue;
                }
            } else if (time_field != nullptr) {
                if (!ReadNumericField(point, *time_field, point_time)) {
                    continue;
                }
            }

            if (point_filter_num_ > 1 && point_index % point_filter_num_ != 0) {
                has_prev = true;
                prev_x = x;
                prev_y = y;
                prev_z = z;
                continue;
            }

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                continue;
            }

            const double range = x * x + y * y + z * z;
            if (range <= blind_sq || range > range_max_sq) {
                continue;
            }

            if (z < height_min_ || z > height_max_) {
                continue;
            }

            if (has_prev) {
                const bool point_changed =
                    (std::abs(x - prev_x) > 1e-7) || (std::abs(y - prev_y) > 1e-7) || (std::abs(z - prev_z) > 1e-7);
                if (!point_changed) {
                    continue;
                }
            }

            has_prev = true;
            prev_x = x;
            prev_y = y;
            prev_z = z;

            PointType added_pt;
            added_pt.x = x;
            added_pt.y = y;
            added_pt.z = z;
            added_pt.intensity = intensity;

            if (timestamp_field != nullptr || point_time > 1e6) {
                added_pt.time = (point_time - head_time) * 1e3;
            } else {
                added_pt.time = point_time * 1e3;
            }
            if (!std::isfinite(added_pt.time) || added_pt.time < 0.0) {
                continue;
            }

            cloud_out_.points.push_back(added_pt);
        }
    }

    std::sort(cloud_out_.points.begin(), cloud_out_.points.end(),
              [](const PointType &lhs, const PointType &rhs) { return lhs.time < rhs.time; });

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
}

void PointCloudPreprocess::Oust64Handler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    cloud_out_.reserve(plsize);

    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num_ != 0) {
            continue;
        }

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (blind_ * blind_)) {
            continue;
        }

        if (pl_orig.points[i].z < height_min_ || pl_orig.points[i].z > height_max_) {
            continue;
        }

        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;

        added_pt.time = pl_orig.points[i].t / 1e6;
        cloud_out_.points.push_back(added_pt);
    }

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
}

void PointCloudPreprocess::RoboSenseHandler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<PointRobotSense> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);

    int plsize = pl_orig.size();
    cloud_out_.reserve(plsize);

    double head_time = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;

    /// RoboSense的时间戳是double, 均为linux时间且单位为秒，这里减去header time并乘以1000得到毫秒为单位的时间戳

    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num_ != 0) {
            continue;
        }

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (blind_ * blind_)) {
            continue;
        }

        if (pl_orig.points[i].z < height_min_ || pl_orig.points[i].z > height_max_) {
            continue;
        }

        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;

        added_pt.time = (pl_orig.points[i].timestamp - head_time) * 1e3;  //  / 1e6;  // curvature unit: ms

        cloud_out_.points.push_back(added_pt);
    }

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
}

void PointCloudPreprocess::VelodyneHandler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    cloud_out_.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 3.61;  // scan angular velocity
    std::vector<bool> is_first(num_scans_, true);
    std::vector<double> yaw_fp(num_scans_, 0.0);    // yaw of first scan point
    std::vector<float> yaw_last(num_scans_, 0.0);   // yaw of last scan point
    std::vector<float> time_last(num_scans_, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0) {
        given_offset_time_ = true;
    } else {
        given_offset_time_ = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--) {
            if (pl_orig.points[i].ring == layer_first) {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++) {
        PointType added_pt;

        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.time = pl_orig.points[i].time * time_scale_;  // curvature unit: ms

        if (!given_offset_time_) {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer]) {
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.time = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.time;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer]) {
                added_pt.time = (yaw_fp[layer] - yaw_angle) / omega_l;
            } else {
                added_pt.time = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.time < time_last[layer]) {
                added_pt.time += 360.0 / omega_l;
            }

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.time;
        }

        if (i % point_filter_num_ == 0) {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind_ * blind_)) {
                cloud_out_.points.push_back(added_pt);
            }
        }
    }

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
}

void PointCloudPreprocess::JT128Handler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    if (msg->is_bigendian) {
        LOG(ERROR) << "JT128 big-endian PointCloud2 is not supported";
        return;
    }

    const auto *x_field = FindField(*msg, "x");
    const auto *y_field = FindField(*msg, "y");
    const auto *z_field = FindField(*msg, "z");
    const auto *intensity_field = FindField(*msg, "intensity");
    const auto *ring_field = FindField(*msg, "ring");
    const auto *time_field = FindField(*msg, "time");
    const auto *timestamp_field = FindField(*msg, "timestamp");

    if (x_field == nullptr || y_field == nullptr || z_field == nullptr || intensity_field == nullptr ||
        ring_field == nullptr || (time_field == nullptr && timestamp_field == nullptr)) {
        LOG(ERROR) << "JT128 PointCloud2 requires x/y/z/intensity/ring plus time or timestamp fields";
        return;
    }

    const double blind_sq = blind_ * blind_;
    const double range_max_sq = static_cast<double>(range_max_) * range_max_;
    const double head_time = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;
    const std::size_t point_count = static_cast<std::size_t>(msg->width) * msg->height;
    cloud_out_.reserve(point_count);

    bool has_prev = false;
    double prev_x = 0.0;
    double prev_y = 0.0;
    double prev_z = 0.0;

    std::size_t point_index = 0;
    for (std::uint32_t row = 0; row < msg->height; ++row) {
        const auto *row_data = msg->data.data() + static_cast<std::size_t>(row) * msg->row_step;
        for (std::uint32_t col = 0; col < msg->width; ++col, ++point_index) {
            const auto *point = row_data + static_cast<std::size_t>(col) * msg->point_step;

            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            double intensity = 0.0;
            double point_time = 0.0;
            std::uint16_t ring = 0;

            const bool readable = ReadNumericField(point, *x_field, x) && ReadNumericField(point, *y_field, y) &&
                                  ReadNumericField(point, *z_field, z) &&
                                  ReadNumericField(point, *intensity_field, intensity) &&
                                  ReadRingField(point, *ring_field, ring) &&
                                  ReadNumericField(point, timestamp_field != nullptr ? *timestamp_field : *time_field,
                                                   point_time);
            if (!readable) {
                continue;
            }

            if (point_filter_num_ > 1 && point_index % point_filter_num_ != 0) {
                has_prev = true;
                prev_x = x;
                prev_y = y;
                prev_z = z;
                continue;
            }

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(point_time)) {
                continue;
            }

            if (num_scans_ > 0 && ring >= static_cast<std::uint16_t>(num_scans_)) {
                continue;
            }

            const double range = x * x + y * y + z * z;
            if (range <= blind_sq || range > range_max_sq) {
                continue;
            }

            if (z < height_min_ || z > height_max_) {
                continue;
            }

            if (has_prev) {
                const bool point_changed =
                    (std::abs(x - prev_x) > 1e-7) || (std::abs(y - prev_y) > 1e-7) || (std::abs(z - prev_z) > 1e-7);
                if (!point_changed) {
                    continue;
                }
            }

            has_prev = true;
            prev_x = x;
            prev_y = y;
            prev_z = z;

            PointType added_pt;
            added_pt.x = x;
            added_pt.y = y;
            added_pt.z = z;
            added_pt.intensity = intensity;

            // SuperOdom-compatible adapters use relative "time" in seconds.
            // The native JT128 bag uses absolute "timestamp" in seconds.
            added_pt.time = timestamp_field != nullptr || point_time > 1e6 ? (point_time - head_time) * 1e3
                                                                           : point_time * 1e3;
            if (added_pt.time < 0.0) {
                continue;
            }

            cloud_out_.points.push_back(added_pt);
        }
    }

    std::sort(cloud_out_.points.begin(), cloud_out_.points.end(),
              [](const PointType &lhs, const PointType &rhs) { return lhs.time < rhs.time; });

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
}

}  // namespace lightning
