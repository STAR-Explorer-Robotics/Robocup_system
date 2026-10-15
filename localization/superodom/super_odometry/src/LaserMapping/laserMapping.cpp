//
// Created by shiboz on 2021-10-18.
//

#include "super_odometry/LaserMapping/laserMapping.h"

#include <algorithm>
#include <cctype>

double parameters[7] = {0, 0, 0, 0, 0, 0, 1};
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters);
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters+3);

Eigen::Vector3d vel_b;
Eigen::Vector3d ang_vel_b;

namespace {

std::string normalizeFrameName(std::string frame_name) {
    std::transform(frame_name.begin(), frame_name.end(), frame_name.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return frame_name;
}

Eigen::Quaterniond quaternionFromRpy(double roll, double pitch, double yaw) {
    Eigen::Quaterniond q =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    q.normalize();
    return q;
}

Transformd transformFromXyzRpy(const std::vector<double>& xyzrpy, bool rpy_degrees) {
    const double scale = rpy_degrees ? M_PI / 180.0 : 1.0;
    return Transformd(
        quaternionFromRpy(xyzrpy[3] * scale, xyzrpy[4] * scale, xyzrpy[5] * scale),
        Eigen::Vector3d(xyzrpy[0], xyzrpy[1], xyzrpy[2]));
}

void getRpy(const Eigen::Quaterniond& q, double& roll, double& pitch, double& yaw) {
    Eigen::Quaterniond normalized = q.normalized();
    tf2::Quaternion orientation(normalized.x(), normalized.y(), normalized.z(), normalized.w());
    tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
}

}  // namespace

namespace super_odometry {

    laserMapping::laserMapping(const rclcpp::NodeOptions & options)
    : Node("laser_mapping_node", options) {
    this->get_logger().set_level(rclcpp::Logger::Level::Debug);
    }

    void laserMapping::initInterface() {
        //! Callback Groups
        cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = cb_group_;

        if(!readGlobalparam(shared_from_this()))
        {
            RCLCPP_ERROR(this->get_logger(), "[SuperOdometry::laserMapping] Could not read calibration. Exiting...");
            rclcpp::shutdown();
        }

        if (!readParameters())
        {
            RCLCPP_ERROR(this->get_logger(), "[SuperOdometry::laserMapping] Could not read parameters. Exiting...");
            rclcpp::shutdown();
        }

        if (!readCalibration(shared_from_this()))
        {
            RCLCPP_ERROR(this->get_logger(), "[AriseSlam::laserMapping] Could not read parameters. Exiting...");
            rclcpp::shutdown();
        }

        RCLCPP_INFO(this->get_logger(), "DEBUG VIEW: %d", config_.debug_view_enabled);
        RCLCPP_INFO(this->get_logger(), "ENABLE OUSTER DATA: %d", config_.enable_ouster_data);
        RCLCPP_INFO(this->get_logger(), "line resolution %f plane resolution %f vision_laser_time_offset %f",
                config_.lineRes, config_.planeRes, vision_laser_time_offset);

        downSizeFilterCorner.setLeafSize(config_.lineRes, config_.lineRes, config_.lineRes);
        downSizeFilterSurf.setLeafSize(config_.planeRes, config_.planeRes, config_.planeRes);


        subLaserFeatureInfo = this->create_subscription<super_odometry_msgs::msg::LaserFeature>(
            ProjectName+"/feature_info", 2,
            std::bind(&laserMapping::laserFeatureInfoHandler, this,
                        std::placeholders::_1), sub_options);
                        

        pubLaserCloudSurround = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/laser_cloud_surround", 2);

        pubLaserCloudMap = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/laser_cloud_map", 2);

        pubLaserCloudPrior = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/overall_map", 2);

        pubLaserCloudFullRes = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            ProjectName+"/registered_scan", 2);


        pubOdomAftMapped = this->create_publisher<nav_msgs::msg::Odometry>(
            ProjectName+"/laser_odometry", 1);

        pubLaserOdometryIncremental = this->create_publisher<nav_msgs::msg::Odometry>(
            ProjectName+"/aft_mapped_to_init_incremental", 1);


        pubVIOPrediction=  this->create_publisher<nav_msgs::msg::Odometry>(
            ProjectName+"/vio_prediction", 1);

        pubLIOPrediction= this->create_publisher<nav_msgs::msg::Odometry>(
            ProjectName+"/lio_prediction", 1);


        pubLaserAfterMappedPath = this->create_publisher<nav_msgs::msg::Path>(
            ProjectName+"/laser_odom_path", 1);

        pubOptimizationStats = this->create_publisher<super_odometry_msgs::msg::OptimizationStats>(
            ProjectName+"/super_odometry_stats", 1);

  

        pubprediction_source = this->create_publisher<std_msgs::msg::String>(
            ProjectName+"/prediction_source", 1);

        if (config_.localization_mode && config_.use_rviz_initial_pose) {
            subInitialPose = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
                "/initialpose", 1,
                std::bind(&laserMapping::initialPoseHandler, this, std::placeholders::_1), sub_options);
            RCLCPP_INFO(this->get_logger(),
                        "Localization will wait for manual initial pose on /initialpose.");
        }

        process_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(100.)),
            std::bind(&laserMapping::process, this));

        slam.initROSInterface(shared_from_this());
        slam.localMap.lineRes_ = config_.lineRes;
        slam.localMap.planeRes_ = config_.planeRes;
        slam.Visual_confidence_factor=config_.visual_confidence_factor;
        slam.Pos_degeneracy_threshold=config_.pos_degeneracy_threshold;
        slam.Ori_degeneracy_threshold=config_.ori_degeneracy_threshold;
        slam.LocalizationICPMaxIter=config_.max_iterations;
        slam.LocalizationPlaneDistanceNbrNeighbors=static_cast<size_t>(config_.plane_knn_neighbors);
        slam.OptSet.debug_view_enabled=config_.debug_view_enabled;
        slam.OptSet.velocity_failure_threshold=config_.velocity_failure_threshold;
        slam.OptSet.max_surface_features=config_.max_surface_features;
        slam.OptSet.lio_diagnostics_enabled=config_.lio_diagnostics_enabled;
        slam.OptSet.lio_diagnostics_period=config_.lio_diagnostics_period;
        slam.OptSet.plane_neighbor_distance_factor=config_.plane_neighbor_distance_factor;
        slam.OptSet.plane_pca_min_ratio=config_.plane_pca_min_ratio;
        slam.OptSet.plane_max_point_distance_factor=config_.plane_max_point_distance_factor;
        slam.OptSet.plane_loss_distance_factor=config_.plane_loss_distance_factor;
        slam.OptSet.yaw_ratio=yaw_ratio;
        slam.paper_repro.enable_prediction_source_switching =
            config_.paper_repro_enable_prediction_source_switching;
        slam.paper_repro.enable_active_degeneracy_absolute_pose_constraint =
            config_.paper_repro_enable_active_degeneracy_absolute_pose_constraint;
        slam.paper_repro.enable_degeneracy_state_from_uncertainty_gate =
            config_.paper_repro_enable_degeneracy_state_from_uncertainty_gate;
        slam.paper_repro.enable_degeneracy_state_from_histogram_gate =
            config_.paper_repro_enable_degeneracy_state_from_histogram_gate;
        slam.map_dir=config_.map_dir;
        slam.localization_mode=config_.localization_mode;
        slam.init_x=config_.init_x;
        slam.init_y=config_.init_y;
        slam.init_z=config_.init_z;
        slam.init_roll=config_.init_roll;
        slam.init_pitch=config_.init_pitch;
        slam.init_yaw=config_.init_yaw;

        prediction_source = PredictionSource::IMU_ORIENTATION;
        timeLatestImuOdometry = rclcpp::Time(0,0,RCL_ROS_TIME);

        initializationParam();

    }

    void laserMapping::initializationParam() {

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());

        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurround.reset(new pcl::PointCloud<PointType>());
        laserCloudFullRes.reset(new pcl::PointCloud<PointType>());
        laserCloudFullRes_rot.reset(new pcl::PointCloud<PointType>());
        laserCloudRawRes.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerStack.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfStack.reset(new pcl::PointCloud<PointType>());
        laserCloudRealsense.reset(new pcl::PointCloud<PointType>());
        laserCloudPriorOrg.reset(new pcl::PointCloud<PointType>());
        laserCloudPrior.reset(new pcl::PointCloud<PointType>());

        Eigen::Quaterniond q_wmap_wodom_(1, 0, 0, 0);
        Eigen::Vector3d t_wmap_wodom_(0, 0, 0);
        Eigen::Quaterniond q_wodom_curr_(1, 0, 0, 0);
        Eigen::Vector3d t_wodom_curr_(0, 0, 0);
        Eigen::Quaterniond q_wodom_pre_(1, 0, 0, 0);
        Eigen::Vector3d t_wodom_pre_(0, 0, 0);

        q_wmap_wodom = q_wmap_wodom_;
        t_wmap_wodom = t_wmap_wodom_;
        q_wodom_curr = q_wodom_curr_;
        t_wodom_curr = t_wodom_curr_;
        q_wodom_pre = q_wodom_pre_;
        t_wodom_pre = t_wodom_pre_;

        imu_odom_buf.allocate(5000);
        visual_odom_buf.allocate(5000);
        
        slam.localMap.setOrigin(Eigen::Vector3d(slam.init_x, slam.init_y, slam.init_z));

        if (slam.localization_mode) {
            RCLCPP_INFO(this->get_logger(), "\033[1;32m Loading GT Map now.... Please wait for 10 sec before running rosbag.\033[0m");
            if(utils::readPointCloud(config_.map_dir, laserCloudPrior)) {
                slam.localMap.addSurfPointCloud(*laserCloudPrior);
                pcl::toROSMsg(*laserCloudPrior, priorCloudMsg);
                priorCloudMsg.header.stamp = this->now();
                priorCloudMsg.header.frame_id = WORLD_FRAME;
                pubLaserCloudPrior->publish(priorCloudMsg);
                RCLCPP_INFO(this->get_logger(), "\033[1;32m Loading GT Map Succesfully. Localization mode is Ready.\033[0m");
            } else {
                slam.localization_mode = false;
                RCLCPP_INFO(this->get_logger(), "\033[1;32mCannot read map file, switch to mapping mode.\033[0m");
            }
        } else {
            RCLCPP_INFO(this->get_logger(), "\033[1;32mStart SLAM in mapping mode.\033[0m");
        }
    }

    bool laserMapping::readParameters()
    {
        // Declare with default values
        this->declare_parameter("laser_mapping_node.mapping_line_resolution", 0.1);
        this->declare_parameter("laser_mapping_node.mapping_plane_resolution", 0.2);
        this->declare_parameter("laser_mapping_node.max_iterations", 4);
        this->declare_parameter("laser_mapping_node.debug_view", false);
        this->declare_parameter("laser_mapping_node.enable_ouster_data", false);
        this->declare_parameter("laser_mapping_node.publish_only_feature_points", false);
        this->declare_parameter("laser_mapping_node.use_imu_roll_pitch", false);
        this->declare_parameter("laser_mapping_node.lio_diagnostics_enabled", false);
        this->declare_parameter("laser_mapping_node.max_surface_features", 2000);
        this->declare_parameter("laser_mapping_node.lio_diagnostics_period", 20);
        this->declare_parameter("laser_mapping_node.plane_knn_neighbors", 5);
        this->declare_parameter("laser_mapping_node.plane_neighbor_distance_factor", 3.0);
        this->declare_parameter("laser_mapping_node.plane_pca_min_ratio", 0.1);
        this->declare_parameter("laser_mapping_node.plane_max_point_distance_factor", 0.5);
        this->declare_parameter("laser_mapping_node.plane_loss_distance_factor", 3.0);
        this->declare_parameter("laser_mapping_node.velocity_failure_threshold", 30.0);
        this->declare_parameter("laser_mapping_node.auto_voxel_size", true);
        this->declare_parameter("laser_mapping_node.forget_far_chunks", false);
        this->declare_parameter("laser_mapping_node.visual_confidence_factor", 1.0);
        // Source code original state: ON. determinePredictionSource() already switches
        // predictors based on degeneracy state when this flag is true.
        this->declare_parameter("paper_repro.enable_prediction_source_switching", true);
        // Source code original state: ON. The absolute pose prior path already exists
        // in the public source but only activates for VIO_ODOM + degeneracy.
        this->declare_parameter("paper_repro.enable_active_degeneracy_absolute_pose_constraint", true);
        // Source code original state: OFF. This exact gate exists as commented logic
        // in LidarSlam.cpp and is opt-in here.
        this->declare_parameter("paper_repro.enable_degeneracy_state_from_uncertainty_gate", false);
        // Source code original state: OFF. This exact gate exists as commented logic
        // in LidarSlam.cpp and is opt-in here.
        this->declare_parameter("paper_repro.enable_degeneracy_state_from_histogram_gate", false);
        this->declare_parameter("laser_mapping_node.localization_mode", false); // Add default value!
        this->declare_parameter("laser_mapping_node.read_pose_file", false);
        this->declare_parameter("laser_mapping_node.use_rviz_initial_pose", false);
        this->declare_parameter("laser_mapping_node.rviz_initial_pose_xy_yaw_only", false);
        this->declare_parameter("laser_mapping_node.rviz_initial_pose_frame", "sensor");
        this->declare_parameter<std::vector<double>>(
            "laser_mapping_node.rviz_initial_pose_lidar_to_body_xyzrpy",
            std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        this->declare_parameter("laser_mapping_node.rviz_initial_pose_extrinsic_rpy_degrees", false);
        this->declare_parameter("laser_mapping_node.init_x", 0.0);
        this->declare_parameter("laser_mapping_node.init_y", 0.0);
        this->declare_parameter("laser_mapping_node.init_z", 0.0);
        this->declare_parameter("laser_mapping_node.init_roll", 0.0);
        this->declare_parameter("laser_mapping_node.init_pitch", 0.0);
        this->declare_parameter("laser_mapping_node.init_yaw", 0.0);
        this->declare_parameter("map_dir", "pointcloud_local.pcd");


        // Get parameters
        config_.lineRes = this->get_parameter("laser_mapping_node.mapping_line_resolution").as_double();
        config_.planeRes = this->get_parameter("laser_mapping_node.mapping_plane_resolution").as_double();
        config_.max_iterations = this->get_parameter("laser_mapping_node.max_iterations").as_int();
        config_.debug_view_enabled = this->get_parameter("laser_mapping_node.debug_view").as_bool();
        config_.enable_ouster_data = this->get_parameter("laser_mapping_node.enable_ouster_data").as_bool();
        config_.publish_only_feature_points = this->get_parameter("laser_mapping_node.publish_only_feature_points").as_bool();
        // config_.use_imu_roll_pitch = this->get_parameter("laser_mapping_node.use_imu_roll_pitch").as_bool();
        config_.lio_diagnostics_enabled = this->get_parameter("laser_mapping_node.lio_diagnostics_enabled").as_bool();
        config_.max_surface_features = this->get_parameter("laser_mapping_node.max_surface_features").as_int();
        config_.lio_diagnostics_period = this->get_parameter("laser_mapping_node.lio_diagnostics_period").as_int();
        if (config_.lio_diagnostics_period < 1) {
            config_.lio_diagnostics_period = 1;
        }
        config_.plane_knn_neighbors = this->get_parameter("laser_mapping_node.plane_knn_neighbors").as_int();
        if (config_.plane_knn_neighbors < 3) {
            RCLCPP_WARN(this->get_logger(),
                        "laser_mapping_node.plane_knn_neighbors must be >= 3, clamping %d to 3",
                        config_.plane_knn_neighbors);
            config_.plane_knn_neighbors = 3;
        }
        config_.plane_neighbor_distance_factor = this->get_parameter("laser_mapping_node.plane_neighbor_distance_factor").as_double();
        config_.plane_pca_min_ratio = this->get_parameter("laser_mapping_node.plane_pca_min_ratio").as_double();
        config_.plane_max_point_distance_factor = this->get_parameter("laser_mapping_node.plane_max_point_distance_factor").as_double();
        config_.plane_loss_distance_factor = this->get_parameter("laser_mapping_node.plane_loss_distance_factor").as_double();
        if (config_.plane_neighbor_distance_factor <= 0.0) {
            config_.plane_neighbor_distance_factor = 3.0;
        }
        if (config_.plane_pca_min_ratio < 0.0) {
            config_.plane_pca_min_ratio = 0.0;
        }
        if (config_.plane_max_point_distance_factor <= 0.0) {
            config_.plane_max_point_distance_factor = 0.5;
        }
        if (config_.plane_loss_distance_factor <= 0.0) {
            config_.plane_loss_distance_factor = 3.0;
        }
        config_.velocity_failure_threshold = this->get_parameter("laser_mapping_node.velocity_failure_threshold").as_double();
        config_.auto_voxel_size = this->get_parameter("laser_mapping_node.auto_voxel_size").as_bool();
        config_.forget_far_chunks = this->get_parameter("laser_mapping_node.forget_far_chunks").as_bool();
        config_.visual_confidence_factor = this->get_parameter("laser_mapping_node.visual_confidence_factor").as_double();
        config_.paper_repro_enable_prediction_source_switching =
            this->get_parameter("paper_repro.enable_prediction_source_switching").as_bool();
        config_.paper_repro_enable_active_degeneracy_absolute_pose_constraint =
            this->get_parameter("paper_repro.enable_active_degeneracy_absolute_pose_constraint").as_bool();
        config_.paper_repro_enable_degeneracy_state_from_uncertainty_gate =
            this->get_parameter("paper_repro.enable_degeneracy_state_from_uncertainty_gate").as_bool();
        config_.paper_repro_enable_degeneracy_state_from_histogram_gate =
            this->get_parameter("paper_repro.enable_degeneracy_state_from_histogram_gate").as_bool();
        config_.map_dir = this->get_parameter("map_dir").as_string(); 
        config_.localization_mode = this->get_parameter("laser_mapping_node.localization_mode").as_bool();
        config_.read_pose_file = this->get_parameter("laser_mapping_node.read_pose_file").as_bool();
        config_.use_rviz_initial_pose = this->get_parameter("laser_mapping_node.use_rviz_initial_pose").as_bool();
        config_.rviz_initial_pose_xy_yaw_only =
            this->get_parameter("laser_mapping_node.rviz_initial_pose_xy_yaw_only").as_bool();
        config_.rviz_initial_pose_frame = normalizeFrameName(
            this->get_parameter("laser_mapping_node.rviz_initial_pose_frame").as_string());
        if (config_.rviz_initial_pose_frame == "base_link" ||
            config_.rviz_initial_pose_frame == "body") {
            config_.rviz_initial_pose_is_body_frame = true;
        } else if (config_.rviz_initial_pose_frame == "sensor" ||
                   config_.rviz_initial_pose_frame == "lidar") {
            config_.rviz_initial_pose_is_body_frame = false;
        } else {
            RCLCPP_WARN(this->get_logger(),
                        "Unknown laser_mapping_node.rviz_initial_pose_frame '%s'. "
                        "Falling back to sensor-frame interpretation.",
                        config_.rviz_initial_pose_frame.c_str());
            config_.rviz_initial_pose_frame = "sensor";
            config_.rviz_initial_pose_is_body_frame = false;
        }
        config_.rviz_initial_pose_extrinsic_rpy_degrees =
            this->get_parameter("laser_mapping_node.rviz_initial_pose_extrinsic_rpy_degrees").as_bool();
        auto initial_pose_lidar_to_body =
            this->get_parameter("laser_mapping_node.rviz_initial_pose_lidar_to_body_xyzrpy").as_double_array();
        if (initial_pose_lidar_to_body.size() != 6) {
            RCLCPP_WARN(this->get_logger(),
                        "laser_mapping_node.rviz_initial_pose_lidar_to_body_xyzrpy must contain "
                        "[x, y, z, roll, pitch, yaw]. Falling back to identity.");
            initial_pose_lidar_to_body = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        rviz_initial_pose_lidar_to_body_ = transformFromXyzRpy(
            initial_pose_lidar_to_body, config_.rviz_initial_pose_extrinsic_rpy_degrees);
        config_.use_imu_roll_pitch = USE_IMU_ROLL_PITCH;

        if(config_.read_pose_file)
        {   
            std::vector<utils::OdometryData> odometryResults;
            utils::readLocalizationPose(config_.map_dir, odometryResults);
            config_.init_x= odometryResults[0].x;
            config_.init_y= odometryResults[0].y;
            config_.init_z= odometryResults[0].z;
            config_.init_roll= odometryResults[0].roll;
            config_.init_pitch= odometryResults[0].pitch;
            config_.init_yaw= odometryResults[0].yaw;
        }
        else
        {  
            config_.init_x = get_parameter("laser_mapping_node.init_x").as_double(); 
            config_.init_y = get_parameter("laser_mapping_node.init_y").as_double(); 
            config_.init_z = get_parameter("laser_mapping_node.init_z").as_double(); 
            config_.init_roll = get_parameter("laser_mapping_node.init_roll").as_double();
            config_.init_pitch = get_parameter("laser_mapping_node.init_pitch").as_double();
            config_.init_yaw = get_parameter("laser_mapping_node.init_yaw").as_double(); 
        }

        if (config_.localization_mode && config_.use_rviz_initial_pose) {
            if (config_.rviz_initial_pose_xy_yaw_only) {
                RCLCPP_INFO(this->get_logger(),
                            "RViz initial pose mode: x/y/yaw only in %s frame; "
                            "z/roll/pitch stay at configured values "
                            "(z=%.3f, roll=%.3f, pitch=%.3f).",
                            config_.rviz_initial_pose_frame.c_str(),
                            config_.init_z, config_.init_roll, config_.init_pitch);
            } else {
                RCLCPP_INFO(this->get_logger(),
                            "RViz initial pose mode: full pose from /initialpose in %s frame.",
                            config_.rviz_initial_pose_frame.c_str());
            }
        } else if (config_.localization_mode) {
            RCLCPP_INFO(this->get_logger(),
                        "RViz initial pose mode: disabled; using configured init pose or pose file.");
        }

        return true;
    }
    
   

    void laserMapping::laserFeatureInfoHandler(const super_odometry_msgs::msg::LaserFeature::SharedPtr msgIn) {
       
        mBuf.lock();
        cornerLastBuf.push(msgIn->cloud_corner);
        surfLastBuf.push(msgIn->cloud_surface);
        realsenseBuf.push(msgIn->cloud_realsense);
        fullResBuf.push(msgIn->cloud_nodistortion);
        Eigen::Quaterniond imuprediction_tmp(msgIn->initial_quaternion_w, msgIn->initial_quaternion_x,
                                             msgIn->initial_quaternion_y, msgIn->initial_quaternion_z);

        IMUPredictionBuf.push(imuprediction_tmp);
        mBuf.unlock();
    }


void laserMapping::setInitialGuess()
{
  //Case1: First Frame Initialization 
  if(!initialization){
    initializeFirstFrame();
    return;
  }
  //Case2: Startup period -continue using IMU for stability 
  if(startupCount>0){
    initializeWithIMU();
    startupCount--;
    return; 
  }
  //Case3: Normal operation -select prediction source 
  selectPosePrediction();
}

void laserMapping::initializeFirstFrame(){

    // Use IMU roll/pitch only when the configuration explicitly enables it.
    if(config_.use_imu_roll_pitch && sensorMeas.imuPrediction.w()!=0){   //Have IMU data
        //Extract roll and pitch, zero out yaw 
        tf2::Quaternion initial_orientation=utils::extractRollPitch(sensorMeas.imuPrediction);
        q_w_curr=Eigen::Quaterniond(initial_orientation.w(), initial_orientation.x(),
              initial_orientation.y(), initial_orientation.z());
        auto q_extrinsic=Eigen::Quaterniond(imu_laser_R);
        q_extrinsic.normalize();
        q_w_curr=q_extrinsic.inverse()*q_w_curr;
        
        
    }else{

        q_w_curr=Eigen::Quaterniond(1,0,0,0); //If no IMU data, use identity rotation 

    }

    //initialize position 
    T_w_lidar.rot=q_w_curr;
    T_w_lidar.pos=Eigen::Vector3d::Zero();

    //Overide with predefined pose if localization mode 
    if(slam.localization_mode){
        T_w_lidar.pos=Eigen::Vector3d(slam.init_x,slam.init_y,slam.init_z);
        tf2::Quaternion localization_pose;
        localization_pose.setRPY(slam.init_roll, slam.init_pitch,slam.init_yaw);
        T_w_lidar.rot=Eigen::Quaterniond(localization_pose.w(), localization_pose.x(),
                                        localization_pose.y(), localization_pose.z());
        slam.last_T_w_lidar=T_w_lidar;
    }

    q_w_curr = T_w_lidar.rot;

    if (slam.localization_mode && config_.use_rviz_initial_pose &&
        manual_initial_pose_received_ && sensorMeas.imuPrediction.w() != 0) {
        q_w_imu_pre = sensorMeas.imuPrediction.normalized();
        q_wodom_pre = q_w_imu_pre;
    } else {
        q_wodom_pre = q_w_curr;
        q_w_imu_pre = Eigen::Quaterniond(1, 0, 0, 0);
    }

}

void laserMapping::initializeWithIMU(){
    if(config_.use_imu_roll_pitch && sensorMeas.imuPrediction.w()!=0){  //Have IMU data
    // During localization with a manual prior-map alignment, keep the
    // user-chosen world alignment and only apply relative IMU rotation.
    Eigen::Quaterniond curr_imu = sensorMeas.imuPrediction.normalized();

    //Keep position from last frame 
    t_w_curr=last_T_w_lidar.pos;
    T_w_lidar.pos=t_w_curr;

    if (slam.localization_mode && config_.use_rviz_initial_pose && manual_initial_pose_received_) {
        Eigen::Quaterniond delta_q = q_w_imu_pre.inverse() * curr_imu;
        delta_q.normalize();
        q_w_curr = last_T_w_lidar.rot * delta_q;
        q_w_curr.normalize();
        T_w_lidar.rot = q_w_curr;
        q_w_imu_pre = curr_imu;
        q_wodom_pre = curr_imu;
    } else {
        // Original source behavior: use IMU absolute orientation directly
        // during startup for a few frames.
        q_w_curr = curr_imu;
        q_w_curr.normalize();
        T_w_lidar.rot=q_w_curr;
    }
    }else
    {
      //No IMU data, use last rotation 
      q_w_curr=last_T_w_lidar.rot;
      t_w_curr=last_T_w_lidar.pos;
      T_w_lidar=last_T_w_lidar;

    } 
}

Transformd laserMapping::convertRvizInitialPoseToSensorFrame(const Transformd& pose) const {
    if (!config_.rviz_initial_pose_is_body_frame) {
        return pose;
    }
    return pose * rviz_initial_pose_lidar_to_body_;
}

void laserMapping::initialPoseHandler(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    Transformd rviz_pose;
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    Eigen::Quaterniond input_rot(
        msg->pose.pose.orientation.w,
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z);
    input_rot.normalize();
    tf2::Quaternion orientation(input_rot.x(), input_rot.y(), input_rot.z(), input_rot.w());
    tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    rviz_pose.pos = Eigen::Vector3d(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        config_.rviz_initial_pose_xy_yaw_only ? config_.init_z : msg->pose.pose.position.z);

    if (config_.rviz_initial_pose_xy_yaw_only) {
        tf2::Quaternion planar_orientation;
        planar_orientation.setRPY(config_.init_roll, config_.init_pitch, yaw);
        rviz_pose.rot = Eigen::Quaterniond(
            planar_orientation.w(),
            planar_orientation.x(),
            planar_orientation.y(),
            planar_orientation.z());
        roll = config_.init_roll;
        pitch = config_.init_pitch;
    } else {
        rviz_pose.rot = input_rot;
    }
    rviz_pose.rot.normalize();

    if (!msg->header.frame_id.empty() && msg->header.frame_id != WORLD_FRAME) {
        RCLCPP_WARN(this->get_logger(),
                    "Received /initialpose in frame '%s', but SuperOdom world frame is '%s'. "
                    "The pose will still be interpreted in the configured world frame.",
                    msg->header.frame_id.c_str(), WORLD_FRAME.c_str());
    }

    Transformd pose = convertRvizInitialPoseToSensorFrame(rviz_pose);
    pose.rot.normalize();

    {
        std::lock_guard<std::mutex> lock(initial_pose_mutex_);
        manual_initial_pose_ = pose;
        manual_initial_pose_received_ = true;
        pending_manual_initial_pose_ = true;
    }

    double sensor_roll = 0.0;
    double sensor_pitch = 0.0;
    double sensor_yaw = 0.0;
    getRpy(pose.rot, sensor_roll, sensor_pitch, sensor_yaw);
    RCLCPP_INFO(this->get_logger(),
                "Received manual initial pose in %s frame: xyz=(%.3f, %.3f, %.3f) "
                "rpy=(%.3f, %.3f, %.3f). Applied sensor-frame pose: "
                "xyz=(%.3f, %.3f, %.3f) rpy=(%.3f, %.3f, %.3f)",
                config_.rviz_initial_pose_frame.c_str(),
                rviz_pose.pos.x(), rviz_pose.pos.y(), rviz_pose.pos.z(),
                roll, pitch, yaw,
                pose.pos.x(), pose.pos.y(), pose.pos.z(),
                sensor_roll, sensor_pitch, sensor_yaw);
}

bool laserMapping::manualInitialPoseReady() {
    std::lock_guard<std::mutex> lock(initial_pose_mutex_);
    return manual_initial_pose_received_;
}

bool laserMapping::applyPendingManualInitialPose() {
    Transformd pose;
    {
        std::lock_guard<std::mutex> lock(initial_pose_mutex_);
        if (!pending_manual_initial_pose_) {
            return false;
        }
        pose = manual_initial_pose_;
        pending_manual_initial_pose_ = false;
    }

    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    tf2::Quaternion orientation(pose.rot.x(), pose.rot.y(), pose.rot.z(), pose.rot.w());
    tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    config_.init_x = pose.pos.x();
    config_.init_y = pose.pos.y();
    config_.init_z = pose.pos.z();
    config_.init_roll = roll;
    config_.init_pitch = pitch;
    config_.init_yaw = yaw;

    slam.init_x = config_.init_x;
    slam.init_y = config_.init_y;
    slam.init_z = config_.init_z;
    slam.init_roll = config_.init_roll;
    slam.init_pitch = config_.init_pitch;
    slam.init_yaw = config_.init_yaw;

    slam.localMap = LocalMap();
    slam.localMap.lineRes_ = config_.lineRes;
    slam.localMap.planeRes_ = config_.planeRes;
    slam.localMap.setOrigin(Eigen::Vector3d(slam.init_x, slam.init_y, slam.init_z));
    if (slam.localization_mode && laserCloudPrior && !laserCloudPrior->empty()) {
        slam.localMap.addSurfPointCloud(*laserCloudPrior);
        pcl::toROSMsg(*laserCloudPrior, priorCloudMsg);
        priorCloudMsg.header.frame_id = WORLD_FRAME;
    }

    Eigen::Quaterniond imu_baseline(1, 0, 0, 0);
    bool have_imu_baseline = false;
    {
        std::lock_guard<std::mutex> lock(mBuf);
        if (!IMUPredictionBuf.empty()) {
            imu_baseline = IMUPredictionBuf.front().normalized();
            have_imu_baseline = true;
        }
        clearSensorData();
    }

    prediction_source = PredictionSource::IMU_ORIENTATION;
    startupCount = 10;
    slam.startupCount = 10;
    initialization = false;
    q_w_curr = pose.rot;
    t_w_curr = pose.pos;
    T_w_lidar = pose;
    last_T_w_lidar = pose;
    slam.T_w_lidar = pose;
    slam.last_T_w_lidar = pose;
    slam.T_w_initial_guess = pose;
    if (have_imu_baseline) {
        q_w_imu_pre = imu_baseline;
        q_wodom_pre = q_w_imu_pre;
        q_wodom_curr = q_w_imu_pre;
    } else {
        q_w_imu_pre = Eigen::Quaterniond(1, 0, 0, 0);
        q_wodom_pre = pose.rot;
        q_wodom_curr = pose.rot;
    }
    laserAfterMappedPath.poses.clear();
    slam.stats.iterations.clear();
    timeLaserOdometryPrev = 0.0;

    std::vector<utils::OdometryData> saved_pose_records;
    if (utils::saveLocalizationPose(0.0, pose, slam.map_dir, saved_pose_records)) {
        RCLCPP_INFO(this->get_logger(), "Saved manual initial pose to %s",
                    utils::getLocalizationPosePath(slam.map_dir).c_str());
    } else {
        RCLCPP_WARN(this->get_logger(), "Failed to save manual initial pose for prior map %s",
                    slam.map_dir.c_str());
    }

    RCLCPP_INFO(this->get_logger(), "Applied manual initial pose and reset localization state.");
    return true;
}

void laserMapping::selectPosePrediction(){

// Step1: Decide prediction source based on system state 
prediction_source=determinePredictionSource();

//Step2: Get prediction from selected source 
switch(prediction_source){
    case PredictionSource::LIO_ODOM:{
    T_w_lidar= T_w_lidar*sensorMeas.lioPrediction;
    break;
    } 
   
    case PredictionSource::VIO_ODOM:{
    T_w_lidar= T_w_lidar*sensorMeas.vioPrediction;
    break; 
    } 

    case PredictionSource::NEURAL_IMU_ODOM:{
    T_w_lidar= T_w_lidar*sensorMeas.nioPrediction;
    break; 
    } 
    case PredictionSource::IMU_ORIENTATION:{
    Eigen::Quaterniond q_w_predict=q_w_curr*q_wodom_pre.inverse()*q_wodom_curr;
    q_w_predict.normalize();
    T_w_lidar.rot=q_w_predict;
    q_wodom_pre=q_wodom_curr;
    break;
    } 
   
    case PredictionSource::CONSTANT_VELOCITY:{  
    Transformd relative_pose=last_T_w_lidar.inverse()*T_w_lidar;
    T_w_lidar=T_w_lidar*relative_pose;
    break; 
    }
}

//Step4: Update current pose 
q_w_curr=T_w_lidar.rot;
t_w_curr=T_w_lidar.pos;

}

laserMapping::PredictionSource laserMapping::determinePredictionSource(){
// If system is degerenate, prefer VIO or learning imu odom

if(!slam.paper_repro.enable_prediction_source_switching){
    if(sensorMeas.lio_prediction_status){
        return PredictionSource::LIO_ODOM;
    }
    sensorMeas.imu_orientation_status =
        config_.use_imu_roll_pitch && useIMUPrediction(sensorMeas.imuPrediction);
    if(sensorMeas.imu_orientation_status){
        return PredictionSource::IMU_ORIENTATION;
    }
    return PredictionSource::CONSTANT_VELOCITY;
}

if(slam.isDegenerate){
    if(sensorMeas.vio_prediction_status){
        return PredictionSource::VIO_ODOM;
    }
    if(sensorMeas.nio_prediction_status){
        return PredictionSource::NEURAL_IMU_ODOM;
    }

}else{
    // If system is not degenerate, use IMU orientation 
    if(sensorMeas.lio_prediction_status){
        return PredictionSource::LIO_ODOM;
    }
    sensorMeas.imu_orientation_status =
        config_.use_imu_roll_pitch && useIMUPrediction(sensorMeas.imuPrediction);
    if(sensorMeas.imu_orientation_status){
        return PredictionSource::IMU_ORIENTATION;
    }
   
}



// If no prediction source is available, use constant velocity
return PredictionSource::CONSTANT_VELOCITY;

}

    void laserMapping::publishTopic(){

        TicToc t_pub;
        std_msgs::msg::String prediction_source_msg;
        switch (prediction_source) {
            case PredictionSource::IMU_ORIENTATION :
                prediction_source_msg.data = "IMU Only Orientation Prediction";
                break;
            case PredictionSource::LIO_ODOM :
                prediction_source_msg.data = "Using Laser-Inertial Odometry (LIO)";
                break;
            case PredictionSource::VIO_ODOM :
                prediction_source_msg.data = "Using Visual-Inertial Odometry (VIO)";
                break;
            case PredictionSource::NEURAL_IMU_ODOM :
                prediction_source_msg.data = "Using Neural-Inertial Odometry (Neural-IMU)";
                break;
            case PredictionSource::CONSTANT_VELOCITY :
                prediction_source_msg.data = "Using Constant Velocity Prediction";
                break;
        }
        pubprediction_source->publish(prediction_source_msg);

        if (frameCount % 5 == 0 && config_.debug_view_enabled) {
            laserCloudSurround->clear();
            *laserCloudSurround = slam.localMap.get5x5LocalMap(slam.pos_in_localmap);
            sensor_msgs::msg::PointCloud2 laserCloudSurround3;
            pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
            laserCloudSurround3.header.stamp =
                    rclcpp::Time(timeLaserOdometry*1e9);
            laserCloudSurround3.header.frame_id = WORLD_FRAME;
            pubLaserCloudSurround->publish(laserCloudSurround3);
        }

        if (frameCount % 20 == 0) {
            pcl::PointCloud<PointType> laserCloudMap;
            laserCloudMap = slam.localMap.getAllLocalMap();
            sensor_msgs::msg::PointCloud2 laserCloudMsg;
            pcl::toROSMsg(laserCloudMap, laserCloudMsg);
            laserCloudMsg.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);
            laserCloudMsg.header.frame_id = WORLD_FRAME;
            pubLaserCloudMap->publish(laserCloudMsg);
            
            if (slam.localization_mode) {
                priorCloudMsg.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);
                pubLaserCloudPrior->publish(priorCloudMsg);
            }
        }

        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
            PointType const *const &pi = &laserCloudFullRes->points[i];
            if (pi->x* pi->x+ pi->y * pi->y + pi->z* pi->z < 0.01)
            {
                continue;
            }

            utils::pointAssociateToMap(&laserCloudFullRes->points[i],
                                &laserCloudFullRes->points[i],
                                q_w_curr,
                                t_w_curr);
        }

        pcl::PointCloud<pcl::PointXYZI> laserCloudFullResCvt, laserCloudFullResClean;
        sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        pcl::fromROSMsg(laserCloudFullRes3, laserCloudFullResCvt);
        for (int i = 0; i < laserCloudFullResNum; i++) {
          PointType const *const &pi = &laserCloudFullResCvt.points[i];
          if (pi->x* pi->x+ pi->y * pi->y + pi->z* pi->z > 0.01)
          {
             laserCloudFullResClean.push_back(*pi);
          }
        }
        pcl::toROSMsg(laserCloudFullResClean, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);
        laserCloudFullRes3.header.frame_id = WORLD_FRAME;
        pubLaserCloudFullRes->publish(laserCloudFullRes3);

        laserCloudFullResCvt.clear();
        laserCloudFullResClean.clear();
        laserCloudFullRes_rot->clear();
        laserCloudFullRes_rot->resize(laserCloudFullResNum);

        for (int i = 0; i < laserCloudFullResNum; i++) {
            laserCloudFullRes_rot->points[i].x = laserCloudFullRes->points[i].y;
            laserCloudFullRes_rot->points[i].y = laserCloudFullRes->points[i].z;
            laserCloudFullRes_rot->points[i].z = laserCloudFullRes->points[i].x;
            laserCloudFullRes_rot->points[i].intensity = laserCloudFullRes->points[i].intensity;
        }

        nav_msgs::msg::Odometry odomAftMapped;
        odomAftMapped.header.frame_id = WORLD_FRAME;
        odomAftMapped.child_frame_id = SENSOR_FRAME;
        odomAftMapped.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);

        odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
        odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
        odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
        odomAftMapped.pose.pose.orientation.w = q_w_curr.w();

        odomAftMapped.pose.pose.position.x = t_w_curr.x();
        odomAftMapped.pose.pose.position.y = t_w_curr.y();
        odomAftMapped.pose.pose.position.z = t_w_curr.z();

        odomAftMapped.twist.twist.linear.x = vel_b.x();
        odomAftMapped.twist.twist.linear.y = vel_b.y();
        odomAftMapped.twist.twist.linear.z = vel_b.z();

        odomAftMapped.twist.twist.angular.x = ang_vel_b.x();
        odomAftMapped.twist.twist.angular.y = ang_vel_b.y();
        odomAftMapped.twist.twist.angular.z = ang_vel_b.z();

        nav_msgs::msg::Odometry laserOdomIncremental;

        if (initialization == false)
        {
            laserOdomIncremental.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);
            laserOdomIncremental.header.frame_id = WORLD_FRAME;
            laserOdomIncremental.child_frame_id =  SENSOR_FRAME;
            laserOdomIncremental.pose.pose.position.x = t_w_curr.x();
            laserOdomIncremental.pose.pose.position.y = t_w_curr.y();
            laserOdomIncremental.pose.pose.position.z = t_w_curr.z();
            laserOdomIncremental.pose.pose.orientation.x = q_w_curr.x();
            laserOdomIncremental.pose.pose.orientation.y = q_w_curr.y();
            laserOdomIncremental.pose.pose.orientation.z = q_w_curr.z();
            laserOdomIncremental.pose.pose.orientation.w = q_w_curr.w();
        }
        else
        {

            laser_incremental_T = T_w_lidar;
            laser_incremental_T.rot.normalized();

            laserOdomIncremental.header.stamp = rclcpp::Time(timeLaserOdometry*1e9);
            laserOdomIncremental.header.frame_id = WORLD_FRAME;
            laserOdomIncremental.child_frame_id =  SENSOR_FRAME;
            laserOdomIncremental.pose.pose.position.x = laser_incremental_T.pos.x();
            laserOdomIncremental.pose.pose.position.y = laser_incremental_T.pos.y();
            laserOdomIncremental.pose.pose.position.z = laser_incremental_T.pos.z();
            laserOdomIncremental.pose.pose.orientation.x = laser_incremental_T.rot.x();
            laserOdomIncremental.pose.pose.orientation.y = laser_incremental_T.rot.y();
            laserOdomIncremental.pose.pose.orientation.z = laser_incremental_T.rot.z();
            laserOdomIncremental.pose.pose.orientation.w = laser_incremental_T.rot.w();
        }

        pubLaserOdometryIncremental->publish(laserOdomIncremental);


        if (slam.isDegenerate || !slam.lastMotionAccepted) {
            odomAftMapped.pose.covariance[0] = 1;
        } else {
            odomAftMapped.pose.covariance[0] = 0;
        }

        rclcpp::Time pub_time = rclcpp::Clock{RCL_ROS_TIME}.now(); //PARV_TODO - find how to syncrynoise this with rosbag time
        pubOdomAftMapped->publish(odomAftMapped);

        geometry_msgs::msg::PoseStamped laserAfterMappedPose;
        laserAfterMappedPose.header = odomAftMapped.header;
        laserAfterMappedPose.pose = odomAftMapped.pose.pose;
        laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
        laserAfterMappedPath.header.frame_id = WORLD_FRAME;
        laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
        pubLaserAfterMappedPath->publish(laserAfterMappedPath);


        slam.stats.header = odomAftMapped.header;
        if (timeLatestImuOdometry.seconds() < 1.0)
        {
            timeLatestImuOdometry = pub_time;
        }
        rclcpp::Duration latency = timeLatestImuOdometry - pub_time;  
        slam.stats.latency = latency.seconds() * 1000;
        slam.stats.n_iterations = slam.stats.iterations.size();
        // Avoid breaking rqt_multiplot
        while (slam.stats.iterations.size() < 4)
        {
            slam.stats.iterations.push_back(super_odometry_msgs::msg::IterationStats());
        }

        pubOptimizationStats->publish(slam.stats);
        slam.stats.iterations.clear();
    }


    void laserMapping::adjustVoxelSize(){

        // Calculate cloud statistics
        bool increase_blind_radius = false;
        if(config_.auto_voxel_size)
        {
            Eigen::Vector3f average(0,0,0);
            int count_far_points = 0;
            for (auto &point : *laserCloudSurfLast)
            {
                average(0) += fabs(point.x);
                average(1) += fabs(point.y);
                average(2) += fabs(point.z);
                if(point.x*point.x + point.y*point.y + point.z*point.z>9){
                    count_far_points++;
                }
            }
            if (count_far_points > 3000)
            {
                increase_blind_radius = true;
            }

            average /= laserCloudSurfLast->points.size();
            slam.stats.average_distance = average(0)*average(1)*average(2);
            if (slam.stats.average_distance < 25)
            {
                config_.lineRes = 0.1;
                config_.planeRes = 0.2;
            }
            else if (slam.stats.average_distance > 65)
            {
                config_.lineRes = 0.4;
                config_.planeRes = 0.8;
            }
            downSizeFilterSurf.setLeafSize(config_.planeRes , config_.planeRes , config_.planeRes );
            downSizeFilterCorner.setLeafSize(config_.lineRes , config_.lineRes , config_.lineRes );
        }

        laserCloudCornerStack->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerStack);


        laserCloudSurfStack->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfStack);
      

        slam.localMap.lineRes_ = config_.lineRes;
        slam.localMap.planeRes_ = config_.planeRes;

    }

    
    bool  laserMapping::checkDataAvailable() const{
        return !cornerLastBuf.empty() && !surfLastBuf.empty() 
               && !fullResBuf.empty() && !IMUPredictionBuf.empty();       
        //Note: in pure laser odometry, IMU Prediction will be identy. 
    }

    laserMapping::SensorData laserMapping::extractSensorData(){
        
        SensorData data;
        //1. Extract timestamp
        data.timestamp=secs(&fullResBuf.front());
        timeLaserOdometry=data.timestamp;

        //2. Extract point cloud data 
        pcl::fromROSMsg(cornerLastBuf.front(), *laserCloudCornerLast);
        cornerLastBuf.pop();
        pcl::fromROSMsg(surfLastBuf.front(), *laserCloudSurfLast);
        surfLastBuf.pop();
        pcl::fromROSMsg(fullResBuf.front(), *laserCloudFullRes);
        fullResBuf.pop();

        //3. Extract IMU prediction 
        data.imuPrediction=IMUPredictionBuf.front();
        data.imuPrediction.normalize();
        IMUPredictionBuf.pop();

        //4 set status for prediction source (TODO: didn't release code other prediction source yet) 
        data.vio_prediction_status=false;
        data.lio_prediction_status=false;
        data.nio_prediction_status=false;
        data.imu_orientation_status=false;

        return data;
    }

    void laserMapping::clearSensorData(){
        auto clearBuffer=[](auto&buffer){
            while(!buffer.empty()){
                buffer.pop();
            }
        };
        clearBuffer(cornerLastBuf);
        clearBuffer(surfLastBuf);
        clearBuffer(fullResBuf);
        clearBuffer(IMUPredictionBuf);
    }


    void laserMapping::performSLAMOptimization(){
        tf2::Quaternion imu_roll_pitch;
        if(config_.use_imu_roll_pitch){  // TODO: Livox mid360 not use roll pitch angle
            slam.OptSet.use_imu_roll_pitch=true;
            imu_roll_pitch=utils::extractRollPitch(sensorMeas.imuPrediction);
            slam.OptSet.imu_roll_pitch=imu_roll_pitch;
        }else{
            slam.OptSet.use_imu_roll_pitch=false;
            slam.OptSet.imu_roll_pitch=tf2::Quaternion(0,0,0,1);
        }

        slam.Localization(initialization, static_cast<LidarSLAM::PredictionSource>(prediction_source), T_w_lidar,
                 laserCloudCornerStack, laserCloudSurfStack, timeLaserOdometry);
    }


    bool laserMapping::useIMUPrediction(const Eigen::Quaterniond& imuPrediction){
        if (imuPrediction.w()!=0)
        {
            q_wodom_curr=imuPrediction;
            q_wodom_curr.normalize();
            return true;
        }
        else
        {
            return false;
        }
    }

    void laserMapping::updatePoseAndPublish(){

        //1. Update pose 
        q_w_curr=slam.T_w_lidar.rot;
        t_w_curr=slam.T_w_lidar.pos;
        T_w_lidar.rot=slam.T_w_lidar.rot;
        T_w_lidar.pos=slam.T_w_lidar.pos;
        startupCount=slam.startupCount;
        frameCount++;
        slam.frame_count=frameCount;
        slam.laser_imu_sync=laser_imu_sync;
        initialization = true;

        // Calculate linear and angular velocity
        double dt = timeLaserOdometry - timeLaserOdometryPrev;

        if (dt > 1e-6) {  
            Eigen::Vector3d vel_w = (t_w_curr - last_T_w_lidar.pos) / dt;
            vel_b = q_w_curr.inverse() * vel_w;     

            Eigen::Quaterniond dq = q_w_curr * last_T_w_lidar.rot.inverse();
            Eigen::AngleAxisd angle_axis(dq);
            Eigen::Vector3d ang_vel_w = angle_axis.axis() * angle_axis.angle() / dt;
            ang_vel_b = q_w_curr.inverse() * ang_vel_w;
        } else {
            vel_b = Eigen::Vector3d::Zero();
            ang_vel_b = Eigen::Vector3d::Zero();
        }

        //2. Publish results 
        publishTopic();

        //3. Store current pose and time for next iteration
        last_T_w_lidar = slam.T_w_lidar;
        timeLaserOdometryPrev = timeLaserOdometry;
    }

    void laserMapping::process() {

        while (rclcpp::ok()) {
            if (slam.localization_mode && config_.use_rviz_initial_pose && !manualInitialPoseReady()) {
                if (laserCloudPrior && !laserCloudPrior->empty()) {
                    priorCloudMsg.header.stamp = this->now();
                    priorCloudMsg.header.frame_id = WORLD_FRAME;
                    pubLaserCloudPrior->publish(priorCloudMsg);
                }
                if (checkDataAvailable()) {
                    std::lock_guard<std::mutex> lock(mBuf);
                    clearSensorData();
                }
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "Waiting for manual initial pose on /initialpose before processing localization data.");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            if(!checkDataAvailable()){
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            try{
                utils::ScopedTimer timer("Frame Processing");
                if (applyPendingManualInitialPose()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    continue;
                }
                mBuf.lock(); 
                sensorMeas=extractSensorData();
                clearSensorData();
                mBuf.unlock();
                setInitialGuess();
                adjustVoxelSize();
                performSLAMOptimization();
                updatePoseAndPublish();
               
                //updateStatsAndDebugInfo();

            }catch(const std::exception&e){
                RCLCPP_ERROR(this->get_logger(), "Error in frame processing: %s", e.what());
            }
        }

    }


#pragma clang diagnostic pop

} // namespace super_odometry
