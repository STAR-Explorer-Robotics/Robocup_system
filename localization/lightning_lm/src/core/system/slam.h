//
// Created by xiang on 25-5-6.
//

#ifndef LIGHTNING_SLAM_H
#define LIGHTNING_SLAM_H

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

#include "lightning/srv/save_map.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"

namespace lightning {

class LaserMapping;  //  lio 前端
class LoopClosing;   // 回环检测

namespace ui {
class PangolinWindow;
}

namespace g2p5 {
class G2P5;
}

/**
 * SLAM 系统调用接口
 */
class SlamSystem {
   public:
    struct Options {
        Options() {}

        bool online_mode_ = true;  // 在线模式，在线模式下会起一些子线程来做异步处理

        bool with_cc_ = true;               // 是否需要带交叉验证
        bool with_gridmap_ = true;          // 是否需要2D栅格
        bool with_loop_closing_ = true;     // 是否需要回环检测
        bool with_visualization_ = true;    // 是否需要可视化UI
        bool with_2dvisualization_ = true;  // 是否需要2D可视化UI

        bool step_on_kf_ = true;  // 是否在关键帧处暂停p
    };

    using SaveMapService = srv::SaveMap;

    SlamSystem(Options options);
    ~SlamSystem();

    /// 初始化
    bool Init(const std::string& yaml_path);

    /// 对外部交互接口
    /// 开始建图，输入地图名称
    void StartSLAM(std::string map_name);

    /// 保存地图，默认保存至./data/地图名/ 下方
    void SaveMap(const std::string& path = "");

    /// 处理IMU
    void ProcessIMU(const lightning::IMUPtr& imu);

    /// 处理点云
    void ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);

    /// 实时模式下的spin
    void Spin();

   private:
    /// ros端保存地图的实现
    void SaveMap(const SaveMapService::Request::SharedPtr request, SaveMapService::Response::SharedPtr response);

    /// 在线建图时，每个新关键帧发布位姿(odom)与map系下的配准点云(registered scan)
    /// 纯接口层：不影响LIO/回环/匹配，供下游(如TRG增量管线)实时订阅
    void PublishKeyframe(const Keyframe::Ptr& kf);

    /// 在线建图时，每个雷达帧发布前端实时位姿(高频odom)。
    /// 关键帧odom每~1m才更新一次，无法支撑下游对原始稠密点云做逐帧位姿变换；
    /// 该接口按雷达帧率(~10Hz)发布LIO前端位姿(未经回环优化、平滑无跳变)，
    /// 供原始/lidar_points的高程建图实时对齐到map系。纯接口层叠加。
    void PublishLioOdom();

    /// 在线建图时，每个雷达帧发布"去畸变稠密点云在map系下的配准结果"。
    /// 由桥侧自行变换原始/lidar_points存在三重缺陷：点云与位姿异步(50~150ms错配)、
    /// 无帧内运动去畸变、错位被全场永久累积放大(大角度旋转时高程图糊掉)。
    /// 此处复用LIO已算好的去畸变点云(GetScanUndist)与同帧精确位姿(GetState)，
    /// 在SLAM内一次性变换到map系发布，下游零变换直接消费，三缺陷一并消除，
    /// 且保持稠密原始点设计(区别于仅关键帧的registered_scan)。
    void PublishDenseRegisteredScan();

    Options options_;
    std::atomic_bool running_ = false;

    rclcpp::Service<SaveMapService>::SharedPtr savemap_service_ = nullptr;

    std::string map_name_;  // 地图名

    std::shared_ptr<LaserMapping> lio_ = nullptr;       // lio 前端
    std::shared_ptr<LoopClosing> lc_ = nullptr;         // 回环检测
    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;  // ui
    std::shared_ptr<g2p5::G2P5> g2p5_ = nullptr;        // 栅格地图

    Keyframe::Ptr cur_kf_ = nullptr;

    /// 实时模式下的ros2 node, subscribers
    rclcpp::Node::SharedPtr node_;
    std::string imu_topic_;
    std::string cloud_topic_;
    std::string livox_topic_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_ = nullptr;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_ = nullptr;

    /// 在线建图输出发布器 (interface layer)
    std::string map_frame_id_ = "map";
    std::string odom_topic_ = "/lightning/slam_odom";
    std::string registered_scan_topic_ = "/lightning/registered_scan";
    std::string lio_odom_topic_ = "/lightning/lio_odom";
    std::string dense_registered_scan_topic_ = "/lightning/dense_registered_scan";
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_ = nullptr;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr registered_scan_pub_ = nullptr;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr lio_odom_pub_ = nullptr;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dense_registered_scan_pub_ = nullptr;
};
}  // namespace lightning

#endif  // LIGHTNING_SLAM_H
