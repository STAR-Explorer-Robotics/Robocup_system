
// LOCAL
#include "super_odometry/LidarProcess/LidarSlam.h"

#include <algorithm>

//TODO: add to header file
double pose_parameters[7] = {0, 0, 0, 0, 0, 0, 1};
Eigen::Map<Eigen::Vector3d> T_w_curr(pose_parameters);
Eigen::Map<Eigen::Quaterniond> Q_w_curr(pose_parameters + 3);

namespace super_odometry {


    LidarSLAM::LidarSLAM() {
        EdgesPoints.reset(new PointCloud());
        PlanarsPoints.reset(new PointCloud());
        WorldEdgesPoints.reset(new PointCloud());
        WorldPlanarsPoints.reset(new PointCloud());
        pcl_to_save.reset(new pcl::PointCloud<pcl::PointXYZI>());
    }
    void LidarSLAM::initROSInterface(rclcpp::Node::SharedPtr node) {
        node_ = node;
        pubUncertaintyX=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_X", 1);
        pubUncertaintyY=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_Y", 1);
        pubUncertaintyZ=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_Z", 1);
        pubUncertaintyRoll=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_roll", 1);
        pubUncertaintyPitch=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_pitch", 1);
        pubUncertaintyYaw=node_->create_publisher<std_msgs::msg::Float32>(ProjectName+"uncertainty_yaw", 1);
    }

    void LidarSLAM::Localization(
        bool initialization,
        PredictionSource predictodom,
        Transformd position,
        pcl::PointCloud<Point>::Ptr edge_point,
        pcl::PointCloud<Point>::Ptr planner_point,
        double timeLaserOdometry){  
       
       //Initialize state with current position 
       initializeState(initialization, position);

       //ProcessInputClouds (we remove the edge points in optimization step)
       processInputClouds(edge_point, planner_point);

       //Intialize and perform localization and mapping
       if(!initialization){
        initializeMapping(timeLaserOdometry);
       } else{
        EstimateLidarUncertainty();
        performLocalizationAndMapping(predictodom, timeLaserOdometry); 
       }
    }
     
    void LidarSLAM::initializeState(bool initialization, const Transformd&position){
        T_w_lidar=position;
        T_w_initial_guess=position;
        last_T_w_lidar=T_w_lidar;
        lastMotionAccepted = true;
        lastMotionStatus = MOTION_ACCEPTED;
        last_motion_dt = 0.0;
        last_motion_speed = 0.0;
    }
    

    void LidarSLAM::transformAndAddToMap(const pcl::PointCloud<Point>::Ptr&source_cloud, 
        pcl::PointCloud<Point>::Ptr&world_cloud, bool is_edge){

         //prepare point cloud 
         world_cloud->clear();
         world_cloud->points.reserve(source_cloud->size());
         world_cloud->header=source_cloud->header;

         //Transform points to world frame 
         for (const Point&p: *source_cloud){
            world_cloud->push_back(utils::TransformPointd(p,T_w_lidar));
         }

         //Add to local map 
         if(is_edge){
            localMap.addEdgePointCloud(*world_cloud);
         }else{
            localMap.addSurfPointCloud(*world_cloud);
         }

         // Save world cloud to .ply file
         if (SAVE_PLY) {
            *pcl_to_save += *world_cloud;
            utils::savePly(pcl_to_save, node_);
         }

    }
    

    void LidarSLAM::initializeMapping(double timeLaserOdometry){
        //clear map and reset statistics 
        
        //set origin for local map 
        localMap.setOrigin(T_w_lidar.pos);

        //Transform and add feature points to map 
        transformAndAddToMap(EdgesPoints, WorldEdgesPoints, true);
        transformAndAddToMap(PlanarsPoints, WorldPlanarsPoints, false);

        lasttimeLaserOdometry=timeLaserOdometry;
    }
    

    void LidarSLAM::processInputClouds(const pcl::PointCloud<Point>::Ptr&edge_point, const pcl::PointCloud<Point>::Ptr&planner_point){
        //clear and reserve space for efficiency 
        EdgesPoints->clear();
        PlanarsPoints->clear();
        EdgesPoints->reserve(edge_point->size());
        PlanarsPoints->reserve(planner_point->size());
        *EdgesPoints=*edge_point;
        *PlanarsPoints=*planner_point;
    }    
    
     void LidarSLAM::performLocalizationAndMapping(PredictionSource predictodom, double timeLaserOdometry)
    {  
        //intialize optimization state 
        prepareOptimizationState();

        //Check if we have enough features for optimization 
        if(!hasEnoughFeatures()){
            ResetDistanceParameters();
            last_surface_sampled_num = 0;
            last_corner_sampled_num = 0;
            last_surface_sampling_rate = 1.0;
            lastMotionAccepted = false;
            lastMotionStatus = MOTION_NOT_ENOUGH_MAP_FEATURES;
            updateDiagnosticStats(stats);
            maybeLogLioDiagnostics(timeLaserOdometry, stats);
            RCLCPP_WARN(node_->get_logger(), "Not enough features for optimization");
            return;
        }
        //Perform ICP iteration 
        TicToc t_opt;
        for (size_t icp_iter=0; icp_iter<LocalizationICPMaxIter; ++icp_iter){
        //Extract features 
        int edge_num=0; int planner_num=0;
        ResetDistanceParameters();
        super_odometry_msgs::msg::IterationStats iter_stats;

        tbb::concurrent_vector<OptimizationParameter> feature_corres;
        extractFeaturesConstraints(feature_corres, edge_num, planner_num);
        
        //Setup and solve the optimization problem 
        Transformd previous_T(T_w_lidar);
      
        auto problem=setupOptimizationProblem(feature_corres, predictodom, T_w_initial_guess);
        auto summary=solveOptimizationProblem(problem);
        
        //Update pose estimates 
        T_w_lidar.pos=T_w_curr;
        T_w_lidar.rot=Q_w_curr;
        //Record iteration statistics 
        recordIterationStats(iter_stats, planner_num, edge_num, previous_T, T_w_lidar);
        //Check for convergence 
        
        if ((summary.num_successful_steps == 1) ||(icp_iter == this->LocalizationICPMaxIter - 1)) {
            this->LocalizationUncertainty =
                    EstimateRegistrationError(problem, 100);
            break;
        
        }

      }
      
      //post-optimization processing 
      performPostOptimizationProcessing(timeLaserOdometry, t_opt, stats);
    }

    
    void LidarSLAM::performPostOptimizationProcessing(double timeLaserOdometry, TicToc &t_opt, super_odometry_msgs::msg::OptimizationStats &stats) {
        // Apply manual yaw correction
        MannualYawCorrection();
        
        // Update statistics
        updateOptimizationStats(t_opt, stats);
        
        // Check motion thresholds and update map
        lastMotionAccepted = checkMotionThresholds(timeLaserOdometry, stats);
        updateDiagnosticStats(stats);
        maybeLogLioDiagnostics(timeLaserOdometry, stats);
        if (lastMotionAccepted) {
            // Transform and add new features to map
            transformAndAddToMap(EdgesPoints, WorldEdgesPoints, true);
            transformAndAddToMap(PlanarsPoints, WorldPlanarsPoints, false);
            last_T_w_lidar = T_w_lidar;
        }
        
        // Update timing
        lasttimeLaserOdometry = timeLaserOdometry;
    }

    bool LidarSLAM::checkMotionThresholds(double timeLaserOdometry, super_odometry_msgs::msg::OptimizationStats &stats) {
    
        double delta_t = timeLaserOdometry - lasttimeLaserOdometry;
        last_motion_dt = delta_t;
        last_motion_speed = 0.0;
        bool acceptResult = true;

        if (delta_t <= 1e-6) {
            T_w_lidar = last_T_w_lidar;
            lastMotionStatus = MOTION_INVALID_DT;
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                "invalid lidar odometry dt, not accumulating. %f", delta_t);
            return false;
        }
        
        // Check velocity threshold
        last_motion_speed = stats.translation_from_last/delta_t;
        if (last_motion_speed > OptSet.velocity_failure_threshold) {
            T_w_lidar = last_T_w_lidar;
            startupCount = 5;
            acceptResult = false;
            lastMotionStatus = MOTION_TOO_LARGE;
            RCLCPP_WARN(node_->get_logger(), "large motion detected, ignoring predictor for a while");
        }
        
        // Check small motion threshold
        if (acceptResult &&
            stats.translation_from_last < 0.02 && stats.rotation_from_last < 0.005) {
            acceptResult = false;
            T_w_lidar = last_T_w_lidar;
            lastMotionStatus = MOTION_TOO_SMALL;
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                "very small motion, not accumulating. %f", stats.translation_from_last);
        }

        if (acceptResult) {
            lastMotionStatus = MOTION_ACCEPTED;
        }

        return acceptResult;
}


    void LidarSLAM::updateOptimizationStats(TicToc &t_opt, super_odometry_msgs::msg::OptimizationStats &stats){
        double time_duration = t_opt.toc();
        stats.time_elapsed = time_duration;
        Transformd total_incremental_T;
        total_incremental_T = T_w_initial_guess.inverse() * T_w_lidar;
        stats.total_translation = (total_incremental_T).pos.norm();
        stats.total_rotation = 2 * atan2(total_incremental_T.rot.vec().norm(), total_incremental_T.rot.w());
        Transformd diff_from_last_T = last_T_w_lidar.inverse() * T_w_lidar;

        stats.translation_from_last = diff_from_last_T.pos.norm();
        stats.rotation_from_last = 2 * atan2(diff_from_last_T.rot.vec().norm(), diff_from_last_T.rot.w());
    }

    void LidarSLAM::updateDiagnosticStats(super_odometry_msgs::msg::OptimizationStats &stats) {
        auto match_count = [](const auto &hist, MatchingResult result) -> int {
            return hist.at(static_cast<size_t>(result)).load();
        };
        auto obs_count = [this](Feature_observability result) -> int {
            return PlaneFeatureHistogramObs.at(static_cast<size_t>(result)).load();
        };

        stats.plane_match_success = match_count(MatchRejectionHistogramPlane, SUCCESS);
        stats.plane_no_enough_neighbor = match_count(MatchRejectionHistogramPlane, NOT_ENOUGH_NEIGHBORS);
        stats.plane_neighbor_too_far = match_count(MatchRejectionHistogramPlane, NEIGHBORS_TOO_FAR);
        stats.plane_badpca_structure = match_count(MatchRejectionHistogramPlane, BAD_PCA_STRUCTURE);
        stats.plane_invalid_numerical = match_count(MatchRejectionHistogramPlane, INVAVLID_NUMERICAL);
        stats.plane_mse_too_large = match_count(MatchRejectionHistogramPlane, MSE_TOO_LARGE);
        stats.plane_unknown = match_count(MatchRejectionHistogramPlane, UNKNON);

        stats.line_match_success = match_count(MatchRejectionHistogramLine, SUCCESS);
        stats.line_no_enough_neighbor = match_count(MatchRejectionHistogramLine, NOT_ENOUGH_NEIGHBORS);
        stats.line_neighbor_too_far = match_count(MatchRejectionHistogramLine, NEIGHBORS_TOO_FAR);
        stats.line_badpca_structure = match_count(MatchRejectionHistogramLine, BAD_PCA_STRUCTURE);
        stats.line_invalid_numerical = match_count(MatchRejectionHistogramLine, INVAVLID_NUMERICAL);
        stats.line_mse_too_large = match_count(MatchRejectionHistogramLine, MSE_TOO_LARGE);
        stats.line_unknown = match_count(MatchRejectionHistogramLine, UNKNON);

        stats.observability_rx = obs_count(rx_cross) + obs_count(neg_rx_cross);
        stats.observability_ry = obs_count(ry_cross) + obs_count(neg_ry_cross);
        stats.observability_rz = obs_count(rz_cross) + obs_count(neg_rz_cross);
        stats.observability_tx = obs_count(tx_dot);
        stats.observability_ty = obs_count(ty_dot);
        stats.observability_tz = obs_count(tz_dot);

        stats.motion_status = static_cast<int>(lastMotionStatus);
        stats.motion_dt = last_motion_dt;
        stats.motion_speed = last_motion_speed;
        stats.surface_sampling_rate = last_surface_sampling_rate;
        stats.surface_sampled_num = last_surface_sampled_num;
        stats.corner_sampled_num = last_corner_sampled_num;
    }

    const char *LidarSLAM::motionStatusName(MotionStatus status) const {
        switch (status) {
            case MOTION_ACCEPTED:
                return "accepted";
            case MOTION_INVALID_DT:
                return "invalid_dt";
            case MOTION_TOO_LARGE:
                return "too_large";
            case MOTION_TOO_SMALL:
                return "too_small";
            case MOTION_NOT_ENOUGH_MAP_FEATURES:
                return "not_enough_map_features";
        }
        return "unknown";
    }

    void LidarSLAM::maybeLogLioDiagnostics(
        double timeLaserOdometry,
        const super_odometry_msgs::msg::OptimizationStats &stats) {
        if (!OptSet.lio_diagnostics_enabled || node_ == nullptr) {
            return;
        }
        const int period = std::max(1, OptSet.lio_diagnostics_period);
        if (frame_count % period != 0 && lastMotionStatus == MOTION_ACCEPTED) {
            return;
        }

        const int plane_total =
            stats.plane_match_success + stats.plane_no_enough_neighbor +
            stats.plane_neighbor_too_far + stats.plane_badpca_structure +
            stats.plane_invalid_numerical + stats.plane_mse_too_large +
            stats.plane_unknown;
        const int line_total =
            stats.line_match_success + stats.line_no_enough_neighbor +
            stats.line_neighbor_too_far + stats.line_badpca_structure +
            stats.line_invalid_numerical + stats.line_mse_too_large +
            stats.line_unknown;
        const Eigen::Vector3d pose_rpy = T_w_lidar.rot.toRotationMatrix().eulerAngles(0, 1, 2);

        RCLCPP_INFO(node_->get_logger(),
                    "[LIO_DIAG] frame=%d t=%.3f motion=%s accepted=%s degenerate=%s dt=%.4f speed=%.3f dT=%.4f dR=%.5f pose_xyz=[%.3f,%.3f,%.3f] pose_rpy=[%.4f,%.4f,%.4f] surf_scan=%d surf_map=%d surf_sampled=%d surf_rate=%.3f plane_ok=%d/%d plane_rej=[nn:%d far:%d pca:%d num:%d mse:%d unk:%d] line_ok=%d/%d line_rej=[nn:%d far:%d pca:%d num:%d mse:%d unk:%d] obs_rpy=[%d,%d,%d] obs_xyz=[%d,%d,%d] unc_xyz=[%.2f,%.2f,%.2f] unc_rpy=[%.2f,%.2f,%.2f]",
                    frame_count,
                    timeLaserOdometry,
                    motionStatusName(lastMotionStatus),
                    lastMotionAccepted ? "true" : "false",
                    isDegenerate ? "true" : "false",
                    stats.motion_dt,
                    stats.motion_speed,
                    stats.translation_from_last,
                    stats.rotation_from_last,
                    T_w_lidar.pos.x(),
                    T_w_lidar.pos.y(),
                    T_w_lidar.pos.z(),
                    pose_rpy.x(),
                    pose_rpy.y(),
                    pose_rpy.z(),
                    stats.laser_cloud_surf_stack_num,
                    stats.laser_cloud_surf_from_map_num,
                    stats.surface_sampled_num,
                    stats.surface_sampling_rate,
                    stats.plane_match_success,
                    plane_total,
                    stats.plane_no_enough_neighbor,
                    stats.plane_neighbor_too_far,
                    stats.plane_badpca_structure,
                    stats.plane_invalid_numerical,
                    stats.plane_mse_too_large,
                    stats.plane_unknown,
                    stats.line_match_success,
                    line_total,
                    stats.line_no_enough_neighbor,
                    stats.line_neighbor_too_far,
                    stats.line_badpca_structure,
                    stats.line_invalid_numerical,
                    stats.line_mse_too_large,
                    stats.line_unknown,
                    stats.observability_rx,
                    stats.observability_ry,
                    stats.observability_rz,
                    stats.observability_tx,
                    stats.observability_ty,
                    stats.observability_tz,
                    stats.uncertainty_x,
                    stats.uncertainty_y,
                    stats.uncertainty_z,
                    stats.uncertainty_roll,
                    stats.uncertainty_pitch,
                    stats.uncertainty_yaw);
    }


    ceres::Problem LidarSLAM::setupOptimizationProblem(const tbb::concurrent_vector<OptimizationParameter>&features_corres, 
                                                       PredictionSource predictsource, const Transformd&position){
        ceres::Problem::Options problem_options; 
        ceres::Problem problem(problem_options);
        problem.AddParameterBlock(pose_parameters, 7, new PoseLocalParameterization());

        //Add feature constraints
        addFeatureConstraints(problem, features_corres);

       
        //Add absolute pose constraints if needed 
        if(shouldAddAbsolutePoseConstraints(predictsource)){
            addAbsolutePoseConstraints(problem,position, features_corres.size());
        }
        return problem;
    }

    ceres::Solver::Summary LidarSLAM::solveOptimizationProblem(ceres::Problem&problem){
        ceres::Solver::Options options;
        options.max_num_iterations=4;
        options.linear_solver_type=ceres::DENSE_QR;
        options.minimizer_progress_to_stdout=false;
        options.check_gradients=false;
        options.gradient_check_relative_precision=1e-4;
        ceres::Solver::Summary summary; 
        ceres::Solve(options, &problem, &summary);
        return summary;
    }

    void LidarSLAM::recordIterationStats(super_odometry_msgs::msg::IterationStats& iter_stats,
                                        int surf_num, int edge_num, Transformd&previous_T, Transformd&current_T){
        //Record iteration statistics 
        iter_stats.num_surf_from_scan=surf_num;
        iter_stats.num_corner_from_scan=edge_num;
        Transformd incremental_T=previous_T.inverse()*current_T;
        iter_stats.translation_norm=incremental_T.pos.norm();
        iter_stats.rotation_norm=2*atan2(incremental_T.rot.vec().norm(), incremental_T.rot.w());
        stats.iterations.push_back(iter_stats);
    }
    

    void LidarSLAM::addFeatureConstraints(ceres::Problem&problem, const tbb::concurrent_vector<OptimizationParameter>&features_corres){
        //Add edge constraints 
       int edge_num=0;
       int planner_num=0;
        for(const auto&constraint: features_corres){
            if(constraint.feature_type==FeatureType::EdgeFeature){
            ceres::CostFunction*cost_function=new EdgeAnalyticCostFunction
            (constraint.Xvalue, constraint.corres.first, constraint.corres.second);
            // Use a robustifier to limit the outlier contribution 
            auto *loss_function=new ceres::TukeyLoss(std::sqrt(3*localMap.lineRes_));
            // Weight the contribution of the given match by its reliability 
            auto *weight_function=new ceres::ScaledLoss(loss_function, constraint.residualCoefficient, ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost_function, weight_function, pose_parameters);
            edge_num++;
            }else if(constraint.feature_type==FeatureType::PlaneFeature){
                ceres::CostFunction*cost_function=new SurfNormAnalyticCostFunction(constraint.Xvalue, constraint.NormDir, constraint.negative_OA_dot_norm);
                // Use a robustifier to limit the outlier contribution 
                auto *loss_function = new ceres::TukeyLoss(
                    std::sqrt(OptSet.plane_loss_distance_factor * localMap.planeRes_));
                // Weight the contribution of the given match by its reliability 
                auto *weight_function=new ceres::ScaledLoss(loss_function, constraint.residualCoefficient, ceres::TAKE_OWNERSHIP);
                problem.AddResidualBlock(cost_function, weight_function, pose_parameters);
                planner_num++;
            }
        }
        stats.prediction_source=0;
    }
    
    bool LidarSLAM::shouldAddAbsolutePoseConstraints(PredictionSource predictodom){
        if (!paper_repro.enable_active_degeneracy_absolute_pose_constraint) {
            return false;
        }
        return predictodom==PredictionSource::VIO_ODOM and isDegenerate==true and Visual_confidence_factor!=0;
    }

    void LidarSLAM::addAbsolutePoseConstraints(ceres::Problem&problem, const Transformd&position, int good_feature_num){
        //Add absolute pose constraint 
       Eigen::Matrix<double, 6, 6, Eigen::RowMajor> information;
       information.setIdentity();
       information(0, 0) =(1 - lidarOdomUncer.uncertainty_x) * std::max(50, int(good_feature_num*0.1))* Visual_confidence_factor;
       information(1, 1) =(1 - lidarOdomUncer.uncertainty_y) * std::max(50, int(good_feature_num*0.1))* Visual_confidence_factor;
       information(2, 2) =(1 - lidarOdomUncer.uncertainty_z) * std::max(50, int(good_feature_num*0.1))* Visual_confidence_factor;
       information(3, 3) = std::max(10, int(good_feature_num*0.01)) * Visual_confidence_factor;
       information(4, 4) = std::max(10, int(good_feature_num*0.01)) * Visual_confidence_factor;
       information(5, 5) = std::max(5, int(good_feature_num*0.001)) * 0;     
       SE3AbsolutatePoseFactor *absolutatePoseFactor=new SE3AbsolutatePoseFactor(position, information);
       problem.AddResidualBlock(absolutatePoseFactor, nullptr, pose_parameters);
       stats.prediction_source=1;
    }
    void LidarSLAM::extractFeaturesConstraints(
        tbb::concurrent_vector<LidarSLAM::OptimizationParameter>&feature_corres,
        int &edge_num, int &planner_num){

        //Process edge features 
        processEdgeFeatures(feature_corres, edge_num);

        //Process planner features 
        processPlannerFeatures(feature_corres,planner_num);
    }

    void LidarSLAM::processEdgeFeatures(tbb::concurrent_vector<OptimizationParameter>&features_corres, int &edge_num){
        edge_num=0;
        last_corner_sampled_num = static_cast<int>(EdgesPoints->size());
        if(EdgesPoints->empty()) return;
        for(const auto&p: *EdgesPoints){
            auto constraint=ComputeLineDistanceParameters(localMap, p);
            if(constraint.match_result==MatchingResult::SUCCESS){
                features_corres.push_back(constraint);
                edge_num++;
            }
        MatchRejectionHistogramLine[constraint.match_result]++;
        }
    }

    void LidarSLAM::processPlannerFeatures(tbb::concurrent_vector<OptimizationParameter>&features_corres, int &planner_num){
        last_surface_sampled_num = 0;
        last_surface_sampling_rate = 1.0;
        planner_num=0;
        if(PlanarsPoints->empty()) return;

        double sampling_rate=calculateSamplingRate(PlanarsPoints->size());
        last_surface_sampling_rate = sampling_rate < 0.0 ? 1.0 : sampling_rate;
        for(size_t i=0; i<PlanarsPoints->size(); ++i){
            if(!shouldProcessPoint(i,sampling_rate)) continue;
            last_surface_sampled_num++;
            const Point&p=PlanarsPoints->points[i];
            auto constraint=ComputePlaneDistanceParameters(localMap, p);
            if(constraint.match_result==MatchingResult::SUCCESS){
                features_corres.push_back(constraint);
                planner_num++;
                //update observability histogram
                const auto&obs=constraint.feature.observability;
                PlaneFeatureHistogramObs[obs[0]]++;
                PlaneFeatureHistogramObs[obs[1]]++;
                PlaneFeatureHistogramObs[obs[2]]++;
            }
            MatchRejectionHistogramPlane[constraint.match_result]++;
        }
       
    } 

    double LidarSLAM::calculateSamplingRate(size_t num_points){
        if(num_points>OptSet.max_surface_features){   
            return 1.0*OptSet.max_surface_features/num_points;
        }
        return -1.0;
    }

    bool LidarSLAM::shouldProcessPoint(size_t index, double sampling_rate){
        if(sampling_rate<0.0) return true;
        double remainder = fmod(index*sampling_rate, 1.0);
        if (remainder + 0.001 > sampling_rate)
            return false;
        return true;
    }

    void LidarSLAM::prepareOptimizationState(){
        
        pos_in_localmap=localMap.shiftMap(T_w_lidar.pos); 
        T_w_curr=T_w_lidar.pos;
        Q_w_curr=T_w_lidar.rot; 

        auto [edge_count, planner_count]=localMap.get5x5LocalMapFeatureSize(pos_in_localmap);
        updateFeatureStats(edge_count, planner_count);
    } 

    void LidarSLAM::updateFeatureStats(size_t edge_count, size_t planner_count){
        stats.laser_cloud_corner_from_map_num = edge_count;
        stats.laser_cloud_surf_from_map_num = planner_count;
        stats.laser_cloud_corner_stack_num = EdgesPoints->size();
        stats.laser_cloud_surf_stack_num = PlanarsPoints->size();
        stats.iterations.clear();
    }
    
    bool LidarSLAM::hasEnoughFeatures(){
         return stats.laser_cloud_surf_from_map_num>50;
    }
    void LidarSLAM::ComputePointInitAndFinalPose(
            LidarSLAM::MatchingMode matchingMode, const LidarSLAM::Point &p,
            Eigen::Vector3d &pInit, Eigen::Vector3d &pFinal) {
      
        const bool is_local_lization_step =
                matchingMode == MatchingMode::LOCALIZATION;
        const Eigen::Vector3d pos = p.getVector3fMap().cast<double>();

        if (this->Undistortion == UndistortionMode::OPTIMIZED and
            is_local_lization_step) {

        } else if (this->Undistortion == UndistortionMode::APPROXIMATED and
                   is_local_lization_step) {

        } else {
            pInit = pos;
            pFinal = this->T_w_lidar * pos;
        }
    }

LidarSLAM::OptimizationParameter LidarSLAM::ComputeLineDistanceParameters(
            LocalMap &local_map, const LidarSLAM::Point &p) {
    // 1. Initialize point
    Eigen::Vector3d pInit, pFinal;
    OptimizationParameter result;
    Eigen::Vector3d mean;
    Eigen::Vector3d eigenvalues;
    Eigen::Matrix3d eigenvectors;
    if (!initializeAndTransformPoint(p, pInit, pFinal)) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return result;
    }

    // 2. Find neighbors using line-specific search
    std::vector<Point> nearest_pts;
    std::vector<float> nearest_dist;
    Point query{pFinal.x(), pFinal.y(), pFinal.z()};
    bool found = local_map.nearestKSearchSpecificEdgePoint(
                query, nearest_pts, nearest_dist, LocalizationLineDistanceNbrNeighbors,
                static_cast<float>(this->LocalizationLineMaxDistInlier));
 
    if (!validateNeighborSearch(found, nearest_pts, nearest_dist, result)) {
        return result;
    }

    // 3. Compute and validate PCA
    if (!computePCAForFeature(nearest_pts, mean, eigenvalues, eigenvectors, result, FeatureType::EdgeFeature)) {
        return result;
    }

    // 4. Process results using shared components
    result=processLineResults(pInit, mean, eigenvalues, eigenvectors, nearest_pts, 3*local_map.lineRes_);
    return result;
}


LidarSLAM::OptimizationParameter LidarSLAM::processLineResults(
                                  const Eigen::Vector3d &pInit,
                                  const Eigen::Vector3d &mean,
                                  const Eigen::Vector3d &eigenvalues,
                                  const Eigen::Matrix3d &eigenvectors,
                                  const std::vector<Point> &nearest_pts,
                                  double square_max_dist) {
    OptimizationParameter result;
  // 1. Get line direction (principal component)
    Eigen::Vector3d line_direction = eigenvectors.col(2);
    line_direction.normalize();

    // 2. Compute projection matrix for point-to-line distance
    Eigen::Matrix3d projection_matrix = Eigen::Matrix3d::Identity() - 
    line_direction * line_direction.transpose();
    
    // 3. Validate projection matrix
    if (!projection_matrix.allFinite()) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return result;
    }

    // 4. Compute quality metrics
    double meanSquareDist = 0.0;
    for (const auto &pt : nearest_pts) {
        Eigen::Vector3d point_vec(pt.x, pt.y, pt.z);
        double squareDist = (point_vec - mean).transpose() * 
                           projection_matrix * (point_vec - mean);

        if (squareDist > 3*localMap.lineRes_) {
            result.match_result = MatchingResult::MSE_TOO_LARGE;
            return result;
        }
        meanSquareDist += squareDist;
    }
    meanSquareDist /= static_cast<double>(nearest_pts.size());

    // 5. Compute quality coefficient
    double fitQualityCoeff = 1.0 - std::sqrt(meanSquareDist / (3*localMap.lineRes_));

    // 6. Compute line endpoints for correspondence
    const double line_segment_length = 0.1; // 10cm line segment
    Eigen::Vector3d point_a = line_segment_length * line_direction + mean;
    Eigen::Vector3d point_b = -line_segment_length * line_direction + mean;

    // 7. Set result parameters
    result.feature_type = FeatureType::EdgeFeature;
    result.match_result = MatchingResult::SUCCESS;
    result.Avalue = projection_matrix;
    result.Pvalue = mean;
    result.Xvalue = pInit;
    result.corres = std::make_pair(point_a, point_b);
    result.TimeValue = 1.0;  // TODO: should be point cloud time
    result.residualCoefficient = fitQualityCoeff;
    return result;
}   

bool LidarSLAM::validateNeighborSearch(
        bool found,
        const std::vector<Point> &nearest_pts,
        const std::vector<float> &nearest_dist,
        OptimizationParameter &result) {
    
    if (!found || nearest_pts.size() < LocalizationMinmumLineNeighborRejection) {
        result.match_result = MatchingResult::NOT_ENOUGH_NEIGHBORS;
        return false;
    }

    if (nearest_dist.back() > 3*localMap.lineRes_) {
        result.match_result = MatchingResult::NEIGHBORS_TOO_FAR;
        return false;
    }

    return true;
}

LidarSLAM::OptimizationParameter LidarSLAM::ComputePlaneDistanceParameters(
            LocalMap &local_map, const Point &p) {
        OptimizationParameter result;
    // 1. Initialize and transform point
  
    Eigen::Vector3d pInit, pFinal;
    if (!initializeAndTransformPoint(p, pInit, pFinal)) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return result;
    }
    // 2. Set search parameters
    const size_t requiredNearest = std::max<size_t>(3, LocalizationPlaneDistanceNbrNeighbors);
    const double square_max_dist = OptSet.plane_neighbor_distance_factor * local_map.planeRes_;

    // 3. Find nearest neighbors
    std::vector<Point> nearest_pts;
    std::vector<float> nearest_dist;
    if (!findNearestNeighbors(local_map, pFinal, nearest_pts, nearest_dist, 
                             requiredNearest, requiredNearest, square_max_dist, result)) {
        return result;
    }

    // 4. Perform PCA analysis
    Eigen::Vector3d mean;
    Eigen::Vector3d eigenvalues;
    Eigen::Matrix3d eigenvectors;
    Eigen::Vector3d plane_normal;
    double negative_OA_dot_norm;
    if (!computePCAForFeature(nearest_pts, mean, eigenvalues, eigenvectors, result, FeatureType::PlaneFeature)) {
        return result;
    }

    // 5. Validate and compute quality metrics
    double meanSquareDist = computePlaneQualityMetrics(nearest_pts, plane_normal, 
                                                      negative_OA_dot_norm, result);
    if (result.match_result != MatchingResult::SUCCESS) {
        return result;
    }
    
    // 6. Check normal direction
    Eigen::Vector3d correct_normal;
    Eigen::Vector3d curr_point(pFinal.x(), pFinal.y(), pFinal.z());
    Eigen::Vector3d viewpoint_direction = curr_point;
    Eigen::Vector3d normal=eigenvectors.col(0);
    double dot_product = viewpoint_direction.dot(normal);
    correct_normal=normal;
    if (dot_product < 0)
        correct_normal = -correct_normal;

    // 7. Compute feature observability
    pcaFeature feature;
    FeatureObservabilityAnalysis(
                feature, pFinal, eigenvalues, correct_normal, eigenvectors.col(2));

    double fitQualityCoeff = 1.0 - sqrt(meanSquareDist / square_max_dist);
    // 8. Set result parameters
    setPlaneResults(result, mean, pInit, plane_normal, negative_OA_dot_norm, feature, fitQualityCoeff);
    return result;
}

void LidarSLAM::FeatureObservabilityAnalysis(pcaFeature &feature, const Eigen::Vector3d &pFinal, 
                                          const Eigen::Vector3d &eigenvalues, 
                                          const Eigen::Vector3d &normal_direction, 
                                          const Eigen::Vector3d &principal_direction) {


    // 1. Initialize feature point and directions
    feature.pt.x = pFinal.x();
    feature.pt.y = pFinal.y();
    feature.pt.z = pFinal.z();
    normal_direction.normalized();
    principal_direction.normalized();
    feature.vectors.principalDirection = principal_direction.cast<float>();
    feature.vectors.normalDirection = normal_direction.cast<float>();
    
    // 2. Compute eigenvalues and geometric properties
    computeEigenProperties(feature, eigenvalues);
    
    // 3. Compute rotation axes and cross products
    auto rotated_axes = computeRotatedAxes();
    computeCrossProducts(feature, rotated_axes);
    
    // 4. Compute translation observability
    computeTranslationObservability(feature, rotated_axes);
    
    // 5. Analyze feature quality and observability
    analyzeFeatureObservability(feature);
}

void LidarSLAM::computeEigenProperties(pcaFeature &feature, const Eigen::Vector3d &eigenvalues) {
    // Compute square roots of eigenvalues
    feature.values.lamada1 = std::sqrt(eigenvalues(2));
    feature.values.lamada2 = std::sqrt(eigenvalues(1));
    feature.values.lamada3 = std::sqrt(eigenvalues(0));
    
    double sum_lamada = feature.values.lamada1 + feature.values.lamada2 + feature.values.lamada3;
    
    // Compute geometric properties
    if (sum_lamada == 0) {
        feature.curvature = 0;
    } else {
        feature.curvature = feature.values.lamada3 / sum_lamada;
    }
    
    feature.linear_2 = (feature.values.lamada1 - feature.values.lamada2) / feature.values.lamada1;
    feature.planar_2 = (feature.values.lamada2 - feature.values.lamada3) / feature.values.lamada1;
    feature.spherical_2 = feature.values.lamada3 / feature.values.lamada1;
}


LidarSLAM::RotatedAxes LidarSLAM::computeRotatedAxes() {
    const Eigen::Vector3f x_axis(1, 0, 0);
    const Eigen::Vector3f y_axis(0, 1, 0);
    const Eigen::Vector3f z_axis(0, 0, 1);
    
    Eigen::Quaternionf rot(T_w_lidar.rot.w(), T_w_lidar.rot.x(),
                          T_w_lidar.rot.y(), T_w_lidar.rot.z());
    rot.normalized();
    
    return RotatedAxes{
        rot * x_axis,
        rot * y_axis,
        rot * z_axis
    };
}

void LidarSLAM::computeTranslationObservability(
        pcaFeature &feature, 
        const RotatedAxes &axes) {
    
    float planar_squared = feature.planar_2 * feature.planar_2;
    
    feature.tx_dot = planar_squared * 
        std::abs(feature.vectors.normalDirection.dot(axes.x));
    feature.ty_dot = planar_squared * 
        std::abs(feature.vectors.normalDirection.dot(axes.y));
    feature.tz_dot = planar_squared * 
        std::abs(feature.vectors.normalDirection.dot(axes.z));
}

void LidarSLAM::analyzeFeatureObservability(pcaFeature &feature) {
    using QualityPair = std::pair<float, Feature_observability>;
    std::vector<QualityPair> rotation_quality = {
        {feature.rx_cross, Feature_observability::rx_cross},
        {feature.neg_rx_cross, Feature_observability::neg_rx_cross},
        {feature.ry_cross, Feature_observability::ry_cross},
        {feature.neg_ry_cross, Feature_observability::neg_ry_cross},
        {feature.rz_cross, Feature_observability::rz_cross},
        {feature.neg_rz_cross, Feature_observability::neg_rz_cross}
    };
    
    std::vector<QualityPair> trans_quality = {
        {feature.tx_dot, Feature_observability::tx_dot},
        {feature.ty_dot, Feature_observability::ty_dot},
        {feature.tz_dot, Feature_observability::tz_dot}
    };
    // Sort quality measures
    std::sort(rotation_quality.begin(), rotation_quality.end(), utils::compare_pair_first);
    std::sort(trans_quality.begin(), trans_quality.end(), utils::compare_pair_first);
    
    // Assign top observability measures
    feature.observability.at(0) = rotation_quality.at(0).second;
    feature.observability.at(1) = rotation_quality.at(1).second;
    feature.observability.at(2) = trans_quality.at(0).second;
    feature.observability.at(3) = trans_quality.at(1).second;
}


void LidarSLAM::computeCrossProducts(pcaFeature &feature, const RotatedAxes &axes) {
    Eigen::Vector3f point(feature.pt.x, feature.pt.y, feature.pt.z);
    Eigen::Vector3f cross = point.cross(feature.vectors.normalDirection);
    
    // Compute cross products with rotated axes
    feature.rx_cross = cross.dot(axes.x);
    feature.neg_rx_cross = -feature.rx_cross;
    feature.ry_cross = cross.dot(axes.y);
    feature.neg_ry_cross = -feature.ry_cross;
    feature.rz_cross = cross.dot(axes.z);
    feature.neg_rz_cross = -feature.rz_cross;
}

void LidarSLAM::setPlaneResults(OptimizationParameter &result, const Eigen::Vector3d &mean, 
                               const Eigen::Vector3d &pInit, const Eigen::Vector3d &plane_normal, 
                               double negative_OA_dot_norm, const pcaFeature &feature, double fitQualityCoeff) {

   result.feature_type = FeatureType::PlaneFeature;
   result.feature = feature;
   result.match_result = MatchingResult::SUCCESS;
   result.Pvalue = mean;
   result.Xvalue = pInit;
   result.NormDir = plane_normal;
   result.negative_OA_dot_norm = negative_OA_dot_norm;
   result.TimeValue =static_cast<double>(1.0);  // TODO:should be the point cloud time
   result.residualCoefficient = fitQualityCoeff;
}   




bool LidarSLAM::initializeAndTransformPoint(const Point &p, 
                                          Eigen::Vector3d &pInit,
                                          Eigen::Vector3d &pFinal) {
    ComputePointInitAndFinalPose(MatchingMode::LOCALIZATION, p, pInit, pFinal);
    return true;
}

bool LidarSLAM::findNearestNeighbors(LocalMap &local_map,
                                    const Eigen::Vector3d &pFinal,
                                    std::vector<Point> &nearest_pts,
                                    std::vector<float> &nearest_dist,
                                    size_t requiredNearest,
                                    size_t min_neighbors,
                                    double square_max_dist,
                                    OptimizationParameter &result) {
    Point pFinal_query;
    pFinal_query.x = pFinal.x();
    pFinal_query.y = pFinal.y();
    pFinal_query.z = pFinal.z();

    bool found = local_map.nearestKSearchSurf(pFinal_query, nearest_pts,
                                            nearest_dist, requiredNearest);

    if (!found || nearest_pts.size() < min_neighbors) {
        result.match_result = MatchingResult::NOT_ENOUGH_NEIGHBORS;
        return false;
    }

    if (nearest_dist.back() > square_max_dist) {
        result.match_result = MatchingResult::NEIGHBORS_TOO_FAR;
        return false;
    }

    return true;
}

bool LidarSLAM::computePCAForFeature(const std::vector<Point> &nearest_pts,
                                  Eigen::Vector3d &mean,
                                  Eigen::Vector3d &eigenvalues,
                                  Eigen::Matrix3d &eigenvectors,
                                  OptimizationParameter &result,
                                  FeatureType feature_type) {

    Eigen::MatrixXd data(nearest_pts.size(), 3);
    for (size_t k = 0; k < nearest_pts.size(); k++) {
        const Point &pt = nearest_pts[k];
        data.row(k) << pt.x, pt.y, pt.z;
    }
    // 2. Compute PCA
    try {
        auto eig = utils::ComputePCA(data, mean);
        eigenvalues = eig.eigenvalues();
        eigenvectors = eig.eigenvectors();
    } catch (const std::exception& e) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return false;
    }
    
    if(feature_type == FeatureType::PlaneFeature){
        if (eigenvalues(0) < 1e-6 || eigenvalues(1) / eigenvalues(2) < OptSet.plane_pca_min_ratio) {
            result.match_result = MatchingResult::BAD_PCA_STRUCTURE;
            return false;
        }
    }else if(feature_type == FeatureType::EdgeFeature){
        if(!eigenvalues.allFinite())
        {
            result.match_result = MatchingResult::INVAVLID_NUMERICAL;
            return false;
        }
        
        if(eigenvalues(2) < LocalizationMinmumLineNeighborRejection * eigenvalues(1)){
            result.match_result = MatchingResult::BAD_PCA_STRUCTURE;
            return false;
        }

    }
    return true;
}

double LidarSLAM::computePlaneQualityMetrics(const std::vector<Point>& nearest_pts,
                                          Eigen::Vector3d &plane_normal,
                                          double &negative_OA_dot_norm,
                                          OptimizationParameter &result) {
    
    if (nearest_pts.size() < 3) {
        result.match_result = MatchingResult::NOT_ENOUGH_NEIGHBORS;
        return 0.0;
    }

     // 1. Set up the system of equations
    Eigen::MatrixXd matA0(nearest_pts.size(), 3);
    Eigen::VectorXd matB0 = -Eigen::VectorXd::Ones(nearest_pts.size());
    // 2. Fill matrix with point coordinates
    for (size_t i = 0; i < nearest_pts.size(); i++) {
        matA0.row(i) << nearest_pts[i].x, nearest_pts[i].y, nearest_pts[i].z;
    }
    
    // 3. Solve for plane normal
    plane_normal = matA0.colPivHouseholderQr().solve(matB0);
    
    // 4. Check if solution is valid
    if (!plane_normal.allFinite()) {
        result.match_result = MatchingResult::INVAVLID_NUMERICAL;
        return 0.0;
    }
    
    // 5. Compute and store plane parameters
    negative_OA_dot_norm = 1.0 / plane_normal.norm();
    plane_normal.normalize();
    

    double meanSquareDist = 0.0;
    const double max_point_distance = localMap.planeRes_ * OptSet.plane_max_point_distance_factor;
    
    // 1. Compute mean square distance to plane
    for (const auto& pt : nearest_pts) {
        double point_to_plane_dist = std::abs(
            plane_normal.x() * pt.x + 
            plane_normal.y() * pt.y +
            plane_normal.z() * pt.z + 
            negative_OA_dot_norm
        );
        
        // 2. Check if point is too far from plane
        if (point_to_plane_dist > max_point_distance) {
            result.match_result = MatchingResult::MSE_TOO_LARGE;
            return 0.0;
        }
        
        meanSquareDist += point_to_plane_dist;
    }
    
    // 3. Compute average distance
    meanSquareDist /= nearest_pts.size();
    result.match_result = MatchingResult::SUCCESS;
    return meanSquareDist;
}


    void LidarSLAM::ResetDistanceParameters() {
        this->OptimizationData.clear();
        for (auto &ele : MatchRejectionHistogramLine) ele = 0;
        for (auto &ele : MatchRejectionHistogramPlane) ele = 0;
        for (auto &ele : PlaneFeatureHistogramObs) ele = 0;
    }

    LidarSLAM::RegistrationError LidarSLAM::EstimateRegistrationError(
            ceres::Problem &problem, const double eigen_thresh) {
        RegistrationError err;

        // Covariance computation options
        ceres::Covariance::Options covOptions;
        covOptions.apply_loss_function = true;
        covOptions.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
        covOptions.null_space_rank = -1;
        covOptions.num_threads = 2;

        ceres::Covariance covarianceSolver(covOptions);
        std::vector<std::pair<const double *, const double *>> covarianceBlocks;
        const double *paramBlock = pose_parameters;
        covarianceBlocks.emplace_back(paramBlock, paramBlock);
        covarianceSolver.Compute(covarianceBlocks, &problem);
        covarianceSolver.GetCovarianceBlockInTangentSpace(paramBlock, paramBlock,
                                                          err.Covariance.data());

        // Estimate max position/orientation errors and directions from covariance
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigPosition(err.Covariance.topLeftCorner<3, 3>());

        err.PositionError = std::sqrt(eigPosition.eigenvalues()(2));
        err.PositionErrorDirection = eigPosition.eigenvectors().col(2);
        err.PosInverseConditionNum = std::sqrt(eigPosition.eigenvalues()(0)) / std::sqrt(eigPosition.eigenvalues()(2));

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigOrientation(err.Covariance.bottomRightCorner<3, 3>());
        err.OrientationError = utils::Rad2Deg(std::sqrt(eigOrientation.eigenvalues()(2)));
        err.OrientationErrorDirection = eigOrientation.eigenvectors().col(2);
        err.OriInverseConditionNum =
                std::sqrt(eigOrientation.eigenvalues()(0)) / std::sqrt(eigOrientation.eigenvalues()(2));

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigPosition2(err.Covariance.inverse());

        return err;
    }
    
   void LidarSLAM::MannualYawCorrection()
   {
     
    Transformd last_current_T = last_T_w_lidar.inverse() * T_w_lidar;
    float translation_norm = last_current_T.pos.norm();

    double roll, pitch, yaw;
    tf2::Quaternion orientation(T_w_lidar.rot.x(), T_w_lidar.rot.y(), T_w_lidar.rot.z(),
                                    T_w_lidar.rot.w());
    tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    
    tf2::Quaternion correct_orientation;

   
    double correct_yaw=yaw+translation_norm*OptSet.yaw_ratio*M_PI/180;
    correct_orientation.setRPY(roll, pitch, correct_yaw);
    
    Eigen::Quaterniond correct_rot;
    correct_rot= Eigen::Quaterniond(correct_orientation.w(), correct_orientation.x(), correct_orientation.y(),
                                correct_orientation.z());
    
    T_w_lidar.rot = correct_rot.normalized();
   }

   void LidarSLAM::EstimateLidarUncertainty() {
        //uncertainty x
        double TotalTransFeature = PlaneFeatureHistogramObs.at(6) +
                                   PlaneFeatureHistogramObs.at(7) +
                                   PlaneFeatureHistogramObs.at(8);
        
        //uncertainty x
        double uncertaintyX = (PlaneFeatureHistogramObs.at(6) / TotalTransFeature) * 3;
        lidarOdomUncer.uncertainty_x = std::min(uncertaintyX, 1.0);

        //uncertainty y
        double uncertaintyY = (PlaneFeatureHistogramObs.at(7) / TotalTransFeature) * 3;
        lidarOdomUncer.uncertainty_y = std::min(uncertaintyY, 1.0);

        //uncertainty Z
        double uncertaintyZ = (PlaneFeatureHistogramObs.at(8) / TotalTransFeature) * 3;
        lidarOdomUncer.uncertainty_z = std::min(uncertaintyZ, 1.0);

        double TotalRotationFeature = PlaneFeatureHistogramObs.at(0) +
                                      PlaneFeatureHistogramObs.at(1) +
                                      PlaneFeatureHistogramObs.at(2) +
                                      PlaneFeatureHistogramObs.at(3) +
                                      PlaneFeatureHistogramObs.at(4) +
                                      PlaneFeatureHistogramObs.at(5);

        //uncertainty roll
        double uncertaintyRoll =
                (PlaneFeatureHistogramObs.at(0) + PlaneFeatureHistogramObs.at(1)) / TotalRotationFeature * 3;
        lidarOdomUncer.uncertainty_roll = std::min(uncertaintyRoll, 1.0);

        //uncertainty pitch
        double uncertaintyPitch =
                (PlaneFeatureHistogramObs.at(2) + PlaneFeatureHistogramObs.at(3)) / TotalRotationFeature * 3;
        lidarOdomUncer.uncertainty_pitch = std::min(uncertaintyPitch, 1.0);

        //uncertainty yaw
        double uncertaintyYaw =
                (PlaneFeatureHistogramObs.at(4) + PlaneFeatureHistogramObs.at(5)) / TotalRotationFeature * 3;
        lidarOdomUncer.uncertainty_yaw = std::min(uncertaintyYaw, 1.0);

              
        if(TotalTransFeature==0 || TotalRotationFeature==0)
        {
          lidarOdomUncer.uncertainty_x=0;
          lidarOdomUncer.uncertainty_y=0;
          lidarOdomUncer.uncertainty_z=0;
          lidarOdomUncer.uncertainty_roll=0;
          lidarOdomUncer.uncertainty_pitch=0;
          lidarOdomUncer.uncertainty_yaw=0;   
        }

        publishUncertainty(lidarOdomUncer.uncertainty_x, lidarOdomUncer.uncertainty_y, lidarOdomUncer.uncertainty_z,
                                     lidarOdomUncer.uncertainty_roll, lidarOdomUncer.uncertainty_pitch, lidarOdomUncer.uncertainty_yaw);

        stats.uncertainty_x=lidarOdomUncer.uncertainty_x;
        stats.uncertainty_y=lidarOdomUncer.uncertainty_y;
        stats.uncertainty_z=lidarOdomUncer.uncertainty_z;
        stats.uncertainty_roll=lidarOdomUncer.uncertainty_roll;
        stats.uncertainty_pitch=lidarOdomUncer.uncertainty_pitch;
        stats.uncertainty_yaw=lidarOdomUncer.uncertainty_yaw;

        updateDegeneracyStateFromPaperReproductionSwitches();
    }

   void LidarSLAM::updateDegeneracyStateFromPaperReproductionSwitches() {
        bool degeneracy_detected = false;

        // Source code original state: OFF. This reproduces the exact public
        // commented gate from LidarSlam.cpp.
        if (paper_repro.enable_degeneracy_state_from_uncertainty_gate) {
            if (lidarOdomUncer.uncertainty_x < 0.2 ||
                lidarOdomUncer.uncertainty_y < 0.1 ||
                lidarOdomUncer.uncertainty_z < 0.2) {
                degeneracy_detected = true;
            }
        }

        // Source code original state: OFF. This reproduces the exact public
        // commented histogram gate from LidarSlam.cpp.
        if (!degeneracy_detected &&
            paper_repro.enable_degeneracy_state_from_histogram_gate) {
            if (PlaneFeatureHistogramObs.at(6) < 20 ||
                PlaneFeatureHistogramObs.at(7) < 10 ||
                PlaneFeatureHistogramObs.at(8) < 10) {
                degeneracy_detected = true;
            }
        }

        isDegenerate = degeneracy_detected;
   }

    void LidarSLAM::publishUncertainty(double uncer_x, double uncer_y, double uncer_z,
        double uncer_roll, double uncer_pitch, double uncer_yaw)
    {

        std_msgs::msg::Float32 uncertainty_x;
        uncertainty_x.data = uncer_x;
        pubUncertaintyX->publish(uncertainty_x);

        std_msgs::msg::Float32 uncertainty_y;
        uncertainty_y.data = uncer_y;
        pubUncertaintyY->publish(uncertainty_y);

        std_msgs::msg::Float32 uncertainty_z;
        uncertainty_z.data = uncer_z;
        pubUncertaintyZ->publish(uncertainty_z);

        std_msgs::msg::Float32 uncertainty_roll;
        uncertainty_roll.data = uncer_roll;
        pubUncertaintyRoll->publish(uncertainty_roll);

        std_msgs::msg::Float32 uncertainty_pitch;
        uncertainty_pitch.data = uncer_pitch;
        pubUncertaintyPitch->publish(uncertainty_pitch);

        std_msgs::msg::Float32 uncertainty_yaw;
        uncertainty_yaw.data = uncer_yaw;
        pubUncertaintyYaw->publish(uncertainty_yaw);

    };

} /* super_odometry */
