#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>

#include "core/maps/tiled_map.h"

DEFINE_string(input_pcd, "", "Input PCD file");
DEFINE_string(output_map, "./data/external_map", "Output Lightning-LM tiled map directory");
DEFINE_bool(overwrite, false, "Remove output_map first when it already exists");
DEFINE_bool(save_global, true, "Save global.pcd in output_map for inspection");
DEFINE_double(global_voxel_size, 0.0, "Optional voxel size for global.pcd; <=0 keeps filtered input density");
DEFINE_double(chunk_size, 100.0, "Lightning-LM map chunk size in meters");
DEFINE_double(voxel_size, 0.1, "Voxel size used inside each saved chunk");
DEFINE_double(min_z, -1.0e9, "Minimum z kept from input PCD");
DEFINE_double(max_z, 1.0e9, "Maximum z kept from input PCD");
DEFINE_double(range_min, 0.0, "Minimum horizontal range kept from input PCD");
DEFINE_double(range_max, 1.0e9, "Maximum horizontal range kept from input PCD");
DEFINE_double(default_intensity, 0.0, "Intensity value used when input PCD has no intensity field");
DEFINE_double(start_x, 0.0, "Initial localization pose x in map frame");
DEFINE_double(start_y, 0.0, "Initial localization pose y in map frame");
DEFINE_double(start_z, 0.0, "Initial localization pose z in map frame");
DEFINE_double(start_roll_deg, 0.0, "Initial localization pose roll in degrees");
DEFINE_double(start_pitch_deg, 0.0, "Initial localization pose pitch in degrees");
DEFINE_double(start_yaw_deg, 0.0, "Initial localization pose yaw in degrees");

namespace {

const pcl::PCLPointField* FindField(const pcl::PCLPointCloud2& cloud, const std::string& name) {
    const auto iter = std::find_if(cloud.fields.begin(), cloud.fields.end(),
                                   [&name](const pcl::PCLPointField& field) { return field.name == name; });
    return iter == cloud.fields.end() ? nullptr : &(*iter);
}

template <typename T>
T ReadUnaligned(const std::uint8_t* data) {
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

bool ReadNumericField(const std::uint8_t* point, const pcl::PCLPointField& field, double& value) {
    const auto* data = point + field.offset;
    switch (field.datatype) {
        case pcl::PCLPointField::INT8:
            value = ReadUnaligned<std::int8_t>(data);
            return true;
        case pcl::PCLPointField::UINT8:
            value = ReadUnaligned<std::uint8_t>(data);
            return true;
        case pcl::PCLPointField::INT16:
            value = ReadUnaligned<std::int16_t>(data);
            return true;
        case pcl::PCLPointField::UINT16:
            value = ReadUnaligned<std::uint16_t>(data);
            return true;
        case pcl::PCLPointField::INT32:
            value = ReadUnaligned<std::int32_t>(data);
            return true;
        case pcl::PCLPointField::UINT32:
            value = ReadUnaligned<std::uint32_t>(data);
            return true;
        case pcl::PCLPointField::FLOAT32:
            value = ReadUnaligned<float>(data);
            return true;
        case pcl::PCLPointField::FLOAT64:
            value = ReadUnaligned<double>(data);
            return true;
        default:
            return false;
    }
}

lightning::SE3 MakeStartPose() {
    constexpr double kDegToRad = M_PI / 180.0;
    const lightning::AngAxisd roll(FLAGS_start_roll_deg * kDegToRad, lightning::Vec3d::UnitX());
    const lightning::AngAxisd pitch(FLAGS_start_pitch_deg * kDegToRad, lightning::Vec3d::UnitY());
    const lightning::AngAxisd yaw(FLAGS_start_yaw_deg * kDegToRad, lightning::Vec3d::UnitZ());
    const lightning::Quatd q = yaw * pitch * roll;
    return lightning::SE3(q.normalized(), lightning::Vec3d(FLAGS_start_x, FLAGS_start_y, FLAGS_start_z));
}

lightning::CloudPtr LoadPCDAsLightningCloud(const std::string& path) {
    pcl::PCLPointCloud2 raw_cloud;
    if (pcl::io::loadPCDFile(path, raw_cloud) != 0) {
        LOG(ERROR) << "failed to load PCD: " << path;
        return nullptr;
    }

    if (raw_cloud.is_bigendian) {
        LOG(ERROR) << "big-endian PCD is not supported: " << path;
        return nullptr;
    }

    const auto* x_field = FindField(raw_cloud, "x");
    const auto* y_field = FindField(raw_cloud, "y");
    const auto* z_field = FindField(raw_cloud, "z");
    const auto* intensity_field = FindField(raw_cloud, "intensity");
    if (x_field == nullptr || y_field == nullptr || z_field == nullptr) {
        LOG(ERROR) << "input PCD must contain x/y/z fields";
        return nullptr;
    }

    const double range_min_sq = FLAGS_range_min * FLAGS_range_min;
    const double range_max_sq = FLAGS_range_max * FLAGS_range_max;
    const std::size_t point_count = static_cast<std::size_t>(raw_cloud.width) * raw_cloud.height;

    auto cloud = std::make_shared<lightning::PointCloudType>();
    cloud->reserve(point_count);

    for (std::uint32_t row = 0; row < raw_cloud.height; ++row) {
        const auto* row_data = raw_cloud.data.data() + static_cast<std::size_t>(row) * raw_cloud.row_step;
        for (std::uint32_t col = 0; col < raw_cloud.width; ++col) {
            const auto* point = row_data + static_cast<std::size_t>(col) * raw_cloud.point_step;

            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            double intensity = FLAGS_default_intensity;
            if (!ReadNumericField(point, *x_field, x) || !ReadNumericField(point, *y_field, y) ||
                !ReadNumericField(point, *z_field, z)) {
                continue;
            }

            if (intensity_field != nullptr && !ReadNumericField(point, *intensity_field, intensity)) {
                intensity = FLAGS_default_intensity;
            }

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z < FLAGS_min_z ||
                z > FLAGS_max_z) {
                continue;
            }

            const double range_sq = x * x + y * y;
            if (range_sq < range_min_sq || range_sq > range_max_sq) {
                continue;
            }

            lightning::PointType output;
            output.x = static_cast<float>(x);
            output.y = static_cast<float>(y);
            output.z = static_cast<float>(z);
            output.intensity = static_cast<float>(intensity);
            output.time = 0.0;
            cloud->push_back(output);
        }
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

lightning::CloudPtr Voxelize(lightning::CloudPtr cloud, double voxel_size) {
    if (voxel_size <= 0.0) {
        return cloud;
    }

    auto filtered = std::make_shared<lightning::PointCloudType>();
    pcl::VoxelGrid<lightning::PointType> voxel;
    voxel.setLeafSize(static_cast<float>(voxel_size), static_cast<float>(voxel_size), static_cast<float>(voxel_size));
    voxel.setInputCloud(cloud);
    voxel.filter(*filtered);
    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;
    return filtered;
}

bool PrepareOutputDirectory(const std::filesystem::path& output_dir) {
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
        return true;
    }

    if (!std::filesystem::is_directory(output_dir)) {
        LOG(ERROR) << "output_map exists but is not a directory: " << output_dir;
        return false;
    }

    if (std::filesystem::is_empty(output_dir)) {
        return true;
    }

    if (!FLAGS_overwrite) {
        LOG(ERROR) << "output_map is not empty: " << output_dir << " (pass --overwrite to replace it)";
        return false;
    }

    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = google::INFO;
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_input_pcd.empty()) {
        LOG(ERROR) << "--input_pcd is required";
        return -1;
    }

    if (FLAGS_chunk_size <= 0.0 || FLAGS_voxel_size <= 0.0 || FLAGS_range_min < 0.0 ||
        FLAGS_range_max <= FLAGS_range_min || FLAGS_max_z <= FLAGS_min_z) {
        LOG(ERROR) << "invalid map conversion parameters";
        return -1;
    }

    const std::filesystem::path output_dir = FLAGS_output_map;
    if (!PrepareOutputDirectory(output_dir)) {
        return -1;
    }

    auto cloud = LoadPCDAsLightningCloud(FLAGS_input_pcd);
    if (cloud == nullptr || cloud->empty()) {
        LOG(ERROR) << "input PCD produced an empty map cloud";
        return -1;
    }

    LOG(INFO) << "loaded points: " << cloud->size();

    lightning::TiledMap::Options options;
    options.map_path_ = FLAGS_output_map;
    options.chunk_size_ = static_cast<float>(FLAGS_chunk_size);
    options.inv_chunk_size_ = 1.0f / options.chunk_size_;
    options.voxel_size_in_chunk_ = static_cast<float>(FLAGS_voxel_size);

    lightning::TiledMap map(options);
    if (!map.ConvertFromFullPCD(cloud, MakeStartPose(), FLAGS_output_map)) {
        LOG(ERROR) << "failed to convert PCD into Lightning-LM tiled map";
        return -1;
    }

    if (FLAGS_save_global) {
        auto global = Voxelize(cloud, FLAGS_global_voxel_size);
        const std::filesystem::path global_path = output_dir / "global.pcd";
        pcl::io::savePCDFileBinaryCompressed(global_path.string(), *global);
        LOG(INFO) << "global preview saved: " << global_path << ", points: " << global->size();
    }

    LOG(INFO) << "Lightning-LM map saved to: " << output_dir;
    LOG(INFO) << "Use it with: ./scripts/run_jt128.sh loc-bag --bag BAG --map " << output_dir;
    return 0;
}
