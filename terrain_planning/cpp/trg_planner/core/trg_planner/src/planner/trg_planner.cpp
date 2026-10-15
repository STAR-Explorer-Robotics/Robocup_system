/**
 * Copyright 2025, Korea Advanced Institute of Science and Technology
 * Massachusetts Institute of Technology,
 * Daejeon, 34051
 * All Rights Reserved
 * Authors: Dongkyu Lee, et al.
 * See LICENSE for the license information
 */
#include "trg_planner/include/planner/trg_planner.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace {
std::string normalizeModeName(std::string mode) {
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
    if (c == '-') {
      return static_cast<char>('_');
    }
    return static_cast<char>(std::tolower(c));
  });
  return mode;
}

TRG::PathSearchMode parsePathSearchMode(const std::string& raw_mode) {
  const std::string mode = normalizeModeName(raw_mode);
  if (mode == "native" || mode == "trg" || mode == "trg_native") {
    return TRG::PathSearchMode::Native;
  }
  if (mode == "trg_astar" || mode == "trgastar" || mode == "astar") {
    return TRG::PathSearchMode::TRGAStar;
  }
  const std::string msg =
      "Unsupported trg.pathSearchMode: " + raw_mode + " (use native or trg_astar)";
  print_error(msg);
  throw std::invalid_argument(msg);
}

TRG::SamplingMode parseSamplingMode(const std::string& raw_mode, bool deterministic_sampling) {
  const std::string mode = normalizeModeName(raw_mode);
  if (mode == "ring" || mode == "legacy_ring" || mode == "legacy") {
    return deterministic_sampling ? TRG::SamplingMode::DeterministicRing
                                  : TRG::SamplingMode::RandomRing;
  }
  if (mode == "deterministic_ring" || mode == "deterministic" || mode == "uniform_ring") {
    return TRG::SamplingMode::DeterministicRing;
  }
  if (mode == "random_ring" || mode == "random") {
    return TRG::SamplingMode::RandomRing;
  }
  if (mode == "lattice" || mode == "uniform_lattice" || mode == "grid" ||
      mode == "uniform_grid") {
    return TRG::SamplingMode::Lattice;
  }
  const std::string msg =
      "Unsupported trg.samplingMode: " + raw_mode +
      " (use deterministic_ring, random_ring, or lattice)";
  print_error(msg);
  throw std::invalid_argument(msg);
}

float quatToYaw(const Eigen::Vector4f& quat) {
  const float w = quat[0];
  const float x = quat[1];
  const float y = quat[2];
  const float z = quat[3];
  return std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
}
}  // namespace

TRGPlanner::TRGPlanner() {}
TRGPlanner::~TRGPlanner() {
  is_running.store(false);
  if (thd.graph.joinable()) {
    thd.graph.join();
  }
  if (thd.planning.joinable()) {
    thd.planning.join();
  }
}

void TRGPlanner::init() {
  //// Load TRG
  trg_ = std::make_shared<TRG>(param_.isVerbose,
                               param_.expandDist,
                               param_.robotSize,
                               param_.sampleNum,
                               param_.heightThreshold,
                               param_.collisionThreshold,
                               param_.updateCollisionThreshold,
                               param_.safetyFactor,
                               param_.goal_tolerance,
                               param_.randomSeed,
                               param_.deterministicSampling,
                               param_.samplingMode,
                               param_.latticeSpacing,
                               param_.latticeRefinementSpacing,
                               param_.latticeRefinementMargin,
                               param_.latticeConnectRadius,
                               param_.latticeMinNodeSpacing,
                               param_.latticeRefinementEnabled,
                               param_.latticeFallbackRing);

  if (trg_ == nullptr) {
    print_error("Failed to initialize TRG");
    exit(1);
  }
  trg_->setPathSearchConfig(param_.pathSearchMode, param_.trgAStar);

  if (param_.boundaryEnabled) {
    trg_->setAllowedArea(
        param_.boundaryPolygon, param_.boundaryKeepoutMargin, param_.boundaryEnabled);
  }

  //// Load prebuilt map
  if (param_.isPreMap) {
    cs_.preMapPtr.reset(new pcl::PointCloud<PtsDefault>());
    cs_.preMapPtr->clear();
    loadPrebuiltMap();
    if (cs_.preMapPtr == nullptr) {
      print_error("Failed to load prebuilt map");
      exit(1);
    }
    trg_->setGlobalMap(cs_.preMapPtr);
  }

  //// Initialize observation map
  cs_.obsPtr.reset(new pcl::PointCloud<PtsDefault>());

  //// TODO: Load prebuilt graph
  if (param_.isPreGraph) {
    print("Prebuilt graph is loaded");
  } else {
    print("Prebuilt graph is not loaded");
  }

  //// Initialize FSM Threads
  thd.graph    = std::thread(&TRGPlanner::runGraphFSM, this);
  thd.planning = std::thread(&TRGPlanner::runPlanningFSM, this);
}

void TRGPlanner::loadPrebuiltMap() {
  if (param_.preMapPath.empty()) {
    print_error("map.prebuiltMapPath is empty. Pass map_path:=/path/to/prior_map.pcd "
                "or set mapConfigPath to a runtime YAML.");
    exit(1);
  }

  std::string map_type = param_.preMapPath.substr(param_.preMapPath.find_last_of(".") + 1);
  if (map_type != "pcd") {
    print_error("Unsupported map type");
    exit(1);
  }

  std::filesystem::path abs_path(param_.preMapPath);
  if (!abs_path.is_absolute()) {
    abs_path = std::filesystem::path(TRG_DIR) / "../.." / param_.preMapPath;
  }
  pcl::PointCloud<PtsDefault> rawMap   = pcl::PointCloud<PtsDefault>();
  if (pcl::io::loadPCDFile<PtsDefault>(abs_path.string(), rawMap) == -1) {
    print_error("Failed to load prebuilt map: " + abs_path.string());
    exit(1);
  }

  if (param_.isVoxelize) {
    pcl::VoxelGrid<PtsDefault> vg;
    vg.setInputCloud(rawMap.makeShared());
    vg.setLeafSize(param_.VoxelSize, param_.VoxelSize, param_.VoxelSize);
    vg.filter(*cs_.preMapPtr);
  } else {
    *cs_.preMapPtr = rawMap;
  }
  print("Prebuilt map size: " + std::to_string(rawMap.size()) + " -> " +
        std::to_string(cs_.preMapPtr->size()));
  print("Prebuilt map is loaded");
}

void TRGPlanner::loadAllowedArea() {
  if (!param_.boundaryEnabled) {
    return;
  }

  if (param_.boundaryPath.empty()) {
    print_error("boundary.allowedAreaPath is empty while boundary.enabled is true");
    exit(1);
  }

  YAML::Node boundary = YAML::LoadFile(param_.boundaryPath);
  YAML::Node points   = boundary["allowed_area"];
  if (!points) {
    points = boundary["allowedArea"];
  }
  if (!points || !points.IsSequence() || points.size() < 3) {
    print_error("Allowed area file must contain at least 3 points under allowed_area");
    exit(1);
  }

  param_.boundaryPolygon.clear();
  for (const auto& point : points) {
    if (!point["x"] || !point["y"]) {
      print_error("Allowed area points must contain x and y fields");
      exit(1);
    }
    param_.boundaryPolygon.emplace_back(point["x"].as<float>(), point["y"].as<float>());
  }

  if (boundary["keepout_margin"]) {
    param_.boundaryKeepoutMargin = boundary["keepout_margin"].as<float>();
  } else if (boundary["keepoutMargin"]) {
    param_.boundaryKeepoutMargin = boundary["keepoutMargin"].as<float>();
  }

  print("Allowed area loaded: " + std::to_string(param_.boundaryPolygon.size()) + " points from " +
        param_.boundaryPath);
}

void TRGPlanner::setParams(const std::string& config_path) {
  print("Loading config from: " + config_path);

  YAML::Node config = YAML::LoadFile(config_path);
  std::filesystem::path config_dir = std::filesystem::path(config_path).parent_path();

  param_.isVerbose = config["isVerbose"].as<bool>(true);

  param_.graph_rate    = config["timer"]["graphRate"].as<float>(1.0f);
  param_.planning_rate = config["timer"]["planningRate"].as<float>(1.0f);

  param_.isPreMap   = config["map"]["isPrebuiltMap"].as<bool>(false);
  param_.preMapPath = config["map"]["prebuiltMapPath"].as<std::string>("");
  param_.waitPoseBeforeInit = config["map"]["waitPoseBeforeInit"].as<bool>(false);
  param_.isVoxelize = config["map"]["isVoxelize"].as<bool>(false);
  param_.VoxelSize  = config["map"]["voxelSize"].as<float>(0.1f);

  param_.isPreGraph               = config["trg"]["isPrebuiltTRG"].as<bool>(false);
  param_.preGraphPath             = config["trg"]["prebuiltTRGPath"].as<std::string>("");
  param_.isUpdate                 = config["trg"]["isUpdate"].as<bool>(false);
  param_.expandDist               = config["trg"]["expandDist"].as<float>(0.6f);
  param_.robotSize                = config["trg"]["robotSize"].as<float>(0.3f);
  param_.sampleNum                = config["trg"]["sampleNum"].as<int>(20);
  param_.randomSeed               = config["trg"]["randomSeed"].as<int>(-1);
  param_.deterministicSampling    = config["trg"]["deterministicSampling"].as<bool>(false);
  const std::string sampling_mode =
      config["trg"]["samplingMode"].as<std::string>(
          param_.deterministicSampling ? "deterministic_ring" : "random_ring");
  param_.samplingMode = parseSamplingMode(sampling_mode, param_.deterministicSampling);
  if (param_.samplingMode == TRG::SamplingMode::DeterministicRing) {
    param_.deterministicSampling = true;
  } else if (param_.samplingMode == TRG::SamplingMode::RandomRing) {
    param_.deterministicSampling = false;
  }
  param_.latticeSpacing =
      config["trg"]["latticeSpacing"].as<float>(param_.latticeSpacing);
  param_.latticeRefinementSpacing =
      config["trg"]["latticeRefinementSpacing"].as<float>(param_.latticeRefinementSpacing);
  param_.latticeRefinementMargin =
      config["trg"]["latticeRefinementMargin"].as<float>(param_.latticeRefinementMargin);
  param_.latticeConnectRadius =
      config["trg"]["latticeConnectRadius"].as<float>(param_.latticeConnectRadius);
  param_.latticeMinNodeSpacing =
      config["trg"]["latticeMinNodeSpacing"].as<float>(param_.latticeMinNodeSpacing);
  param_.latticeRefinementEnabled =
      config["trg"]["latticeRefinementEnabled"].as<bool>(param_.latticeRefinementEnabled);
  param_.latticeFallbackRing =
      config["trg"]["latticeFallbackRing"].as<bool>(param_.latticeFallbackRing);
  param_.heightThreshold          = config["trg"]["heightThreshold"].as<float>(0.15f);
  param_.collisionThreshold       = config["trg"]["collisionThreshold"].as<float>(0.2f);
  param_.updateCollisionThreshold = config["trg"]["updateCollisionThreshold"].as<float>(0.2f);
  param_.safetyFactor             = config["trg"]["safetyFactor"].as<float>(1.0f);
  param_.goal_tolerance           = config["trg"]["goalTolerance"].as<float>(0.8f);
  const std::string path_search_mode =
      config["trg"]["pathSearchMode"].as<std::string>(
          config["trg"]["plannerMode"].as<std::string>("native"));
  param_.pathSearchMode = parsePathSearchMode(path_search_mode);

  param_.trgAStar.length_scale_m = param_.expandDist;
  param_.trgAStar.climb_scale_m  = param_.heightThreshold;
  param_.trgAStar.risk_weight    = param_.safetyFactor;
  YAML::Node astar_config        = config["trgAStar"];
  if (!astar_config) {
    astar_config = config["trg_astar"];
  }
  if (!astar_config && config["trg"]) {
    astar_config = config["trg"]["trgAStar"];
  }
  if (astar_config) {
    param_.trgAStar.fallback_to_native =
        astar_config["fallbackToNative"].as<bool>(param_.trgAStar.fallback_to_native);
    param_.trgAStar.heading_bins =
        astar_config["headingBins"].as<int>(param_.trgAStar.heading_bins);
    param_.trgAStar.length_weight =
        astar_config["lengthWeight"].as<float>(param_.trgAStar.length_weight);
    param_.trgAStar.risk_weight =
        astar_config["riskWeight"].as<float>(param_.trgAStar.risk_weight);
    param_.trgAStar.climb_weight =
        astar_config["climbWeight"].as<float>(param_.trgAStar.climb_weight);
    param_.trgAStar.slope_weight =
        astar_config["slopeWeight"].as<float>(param_.trgAStar.slope_weight);
    param_.trgAStar.turn_weight =
        astar_config["turnWeight"].as<float>(param_.trgAStar.turn_weight);
    param_.trgAStar.heuristic_weight =
        astar_config["heuristicWeight"].as<float>(param_.trgAStar.heuristic_weight);
    param_.trgAStar.length_scale_m =
        astar_config["lengthScaleM"].as<float>(
            astar_config["lengthScale"].as<float>(param_.trgAStar.length_scale_m));
    param_.trgAStar.climb_scale_m =
        astar_config["climbScaleM"].as<float>(
            astar_config["climbScale"].as<float>(param_.trgAStar.climb_scale_m));
    param_.trgAStar.slope_scale_tan =
        astar_config["slopeScaleTan"].as<float>(param_.trgAStar.slope_scale_tan);
    param_.trgAStar.turn_scale =
        astar_config["turnScale"].as<float>(param_.trgAStar.turn_scale);
    param_.trgAStar.max_edge_climb_m =
        astar_config["maxEdgeClimbM"].as<float>(
            astar_config["maxEdgeClimb"].as<float>(param_.trgAStar.max_edge_climb_m));
    param_.trgAStar.max_edge_slope_tan =
        astar_config["maxEdgeSlopeTan"].as<float>(param_.trgAStar.max_edge_slope_tan);
    if (astar_config["maxEdgeSlopeDeg"]) {
      const float max_slope_deg = astar_config["maxEdgeSlopeDeg"].as<float>();
      param_.trgAStar.max_edge_slope_tan =
          std::tan(max_slope_deg * static_cast<float>(M_PI) / 180.0f);
    }
    param_.trgAStar.footprint_cost_enabled =
        astar_config["footprintCostEnabled"].as<bool>(
            astar_config["rectFootprintCostEnabled"].as<bool>(
                param_.trgAStar.footprint_cost_enabled));
    param_.trgAStar.footprint_reject_invalid =
        astar_config["footprintRejectInvalid"].as<bool>(
            param_.trgAStar.footprint_reject_invalid);
    param_.trgAStar.robot_length_m =
        astar_config["robotLengthM"].as<float>(param_.trgAStar.robot_length_m);
    param_.trgAStar.robot_width_m =
        astar_config["robotWidthM"].as<float>(param_.trgAStar.robot_width_m);
    param_.trgAStar.max_body_height_diff_m =
        astar_config["maxBodyHeightDiffM"].as<float>(
            param_.trgAStar.max_body_height_diff_m);
    param_.trgAStar.max_body_tilt_deg =
        astar_config["maxBodyTiltDeg"].as<float>(param_.trgAStar.max_body_tilt_deg);
    param_.trgAStar.max_interior_penetration_m =
        astar_config["maxInteriorPenetrationM"].as<float>(
            param_.trgAStar.max_interior_penetration_m);
    param_.trgAStar.body_height_weight =
        astar_config["bodyHeightWeight"].as<float>(param_.trgAStar.body_height_weight);
    param_.trgAStar.body_tilt_weight =
        astar_config["bodyTiltWeight"].as<float>(param_.trgAStar.body_tilt_weight);
    param_.trgAStar.body_penetration_weight =
        astar_config["bodyPenetrationWeight"].as<float>(
            param_.trgAStar.body_penetration_weight);
    param_.trgAStar.body_invalid_weight =
        astar_config["bodyInvalidWeight"].as<float>(param_.trgAStar.body_invalid_weight);
    param_.trgAStar.footprint_sample_step_m =
        astar_config["footprintSampleStepM"].as<float>(
            param_.trgAStar.footprint_sample_step_m);
    param_.trgAStar.footprint_edge_band_m =
        astar_config["footprintEdgeBandM"].as<float>(
            param_.trgAStar.footprint_edge_band_m);
  }

  auto fail_config = [](const std::string& msg) {
    print_error(msg);
    throw std::invalid_argument(msg);
  };
  if (param_.graph_rate <= 0.0f || param_.planning_rate <= 0.0f) {
    fail_config("TRG timer rates must be positive");
  }
  if (param_.isVoxelize && param_.VoxelSize <= 0.0f) {
    fail_config("map.voxelSize must be positive when map.isVoxelize is true");
  }
  if (param_.expandDist <= 0.0f || param_.robotSize <= 0.0f || param_.sampleNum <= 0) {
    fail_config("TRG expandDist, robotSize, and sampleNum must be positive");
  }
  if (param_.expandDist <= param_.robotSize) {
    fail_config("Invalid TRG parameters: trg.expandDist must be larger than trg.robotSize "
                "(expandDist=" + std::to_string(param_.expandDist) +
                ", robotSize=" + std::to_string(param_.robotSize) + ")");
  }
  if (param_.samplingMode == TRG::SamplingMode::Lattice &&
      (param_.latticeSpacing <= 0.0f || param_.latticeRefinementSpacing <= 0.0f ||
       param_.latticeRefinementMargin <= 0.0f || param_.latticeConnectRadius <= 0.0f ||
       param_.latticeMinNodeSpacing <= 0.0f)) {
    fail_config("TRG lattice parameters must be positive");
  }
  if (param_.collisionThreshold < 0.0f || param_.collisionThreshold > 1.0f ||
      param_.updateCollisionThreshold < 0.0f || param_.updateCollisionThreshold > 1.0f) {
    fail_config("TRG collision thresholds must be in [0, 1]");
  }
  if (param_.heightThreshold <= 0.0f || param_.safetyFactor < 0.0f ||
      param_.goal_tolerance <= 0.0f) {
    fail_config("TRG heightThreshold and goalTolerance must be positive; safetyFactor must be non-negative");
  }
  if (param_.trgAStar.heading_bins < 4) {
    fail_config("trgAStar.headingBins must be at least 4");
  }
  if (param_.trgAStar.length_weight < 0.0f || param_.trgAStar.risk_weight < 0.0f ||
      param_.trgAStar.climb_weight < 0.0f || param_.trgAStar.slope_weight < 0.0f ||
      param_.trgAStar.turn_weight < 0.0f || param_.trgAStar.heuristic_weight < 0.0f) {
    fail_config("TRG-AStar weights must be non-negative");
  }
  if (param_.trgAStar.length_scale_m <= 0.0f || param_.trgAStar.climb_scale_m <= 0.0f ||
      param_.trgAStar.slope_scale_tan <= 0.0f || param_.trgAStar.turn_scale <= 0.0f) {
    fail_config("TRG-AStar scales must be positive");
  }
  if (param_.trgAStar.max_edge_climb_m < 0.0f ||
      param_.trgAStar.max_edge_slope_tan < 0.0f) {
    fail_config("TRG-AStar max edge limits must be non-negative");
  }
  if (param_.trgAStar.robot_length_m <= 0.0f || param_.trgAStar.robot_width_m <= 0.0f ||
      param_.trgAStar.max_body_height_diff_m <= 0.0f ||
      param_.trgAStar.max_body_tilt_deg <= 0.0f ||
      param_.trgAStar.max_interior_penetration_m <= 0.0f ||
      param_.trgAStar.footprint_sample_step_m < 0.0f ||
      param_.trgAStar.footprint_edge_band_m <= 0.0f) {
    fail_config("TRG-AStar footprint dimensions, limits, sample step, and edge band are invalid");
  }
  if (param_.trgAStar.body_height_weight < 0.0f ||
      param_.trgAStar.body_tilt_weight < 0.0f ||
      param_.trgAStar.body_penetration_weight < 0.0f ||
      param_.trgAStar.body_invalid_weight < 0.0f) {
    fail_config("TRG-AStar footprint cost weights must be non-negative");
  }

  if (config["boundary"]) {
    param_.boundaryEnabled =
        config["boundary"]["enabled"].as<bool>(param_.boundaryEnabled);
    param_.boundaryPath =
        config["boundary"]["allowedAreaPath"].as<std::string>(param_.boundaryPath);
    if (param_.boundaryPath.empty()) {
      param_.boundaryPath =
          config["boundary"]["allowed_area_path"].as<std::string>(param_.boundaryPath);
    }
    param_.boundaryKeepoutMargin =
        config["boundary"]["keepoutMargin"].as<float>(param_.boundaryKeepoutMargin);
    param_.boundaryKeepoutMargin =
        config["boundary"]["keepout_margin"].as<float>(param_.boundaryKeepoutMargin);
  }

  if (param_.boundaryEnabled) {
    std::filesystem::path boundary_path(param_.boundaryPath);
    if (!boundary_path.is_absolute()) {
      boundary_path = config_dir / boundary_path;
    }
    if (!std::filesystem::exists(boundary_path)) {
      print_error("Allowed area file does not exist: " + boundary_path.string());
      exit(1);
    }
    param_.boundaryPath = boundary_path.string();
    loadAllowedArea();
  }
}

void TRGPlanner::runGraphFSM() {
  while (is_running.load()) {
    auto start_loop = tic();
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));

    fsm_.graph.notice();
    switch (fsm_.graph.curr_state_) {
      case graphState::INIT: {
        if (flag_.graphInit) {
          fsm_.graph.transition(graphState::UPDATE);
          break;
        }

        if (param_.isPreMap) {
          if (param_.waitPoseBeforeInit && !flag_.poseIn) {
            print_warning("Waiting for pose before prebuilt-map graph initialization");
            fsm_.graph.transition(graphState::INIT);
            break;
          }
          auto start_init_graph = tic();
          if (!trg_->initGraph(param_.isPreMap, state_.pose3d)) {
            flag_.graphInit = false;
            print_warning("TRG graph initialization failed; waiting for a valid pose/map and retrying");
            fsm_.graph.transition(graphState::INIT);
            break;
          }
          print_warning("Graph initialization time: " + std::to_string(toc(start_init_graph, "s")) +
                        " sec");
          flag_.graphInit = true;
          fsm_.graph.transition(graphState::UPDATE);
          break;
        }

        if (!flag_.poseIn || !flag_.obsIn) {
          print_warning("Pose: " + std::to_string(flag_.poseIn) +
                        ", Obs: " + std::to_string(flag_.obsIn));
          fsm_.graph.transition(graphState::INIT);
          break;
        }
        // 机器人不在允许区域(TRG 生长范围)内时，不初始化图；等待(重新)进入后再建。
        if (!trg_->isWithinAllowedArea(state_.pose2d)) {
          if (flag_.insideArea) {
            print_warning("Robot outside allowed area; waiting to (re-)enter before TRG init");
            flag_.insideArea = false;
          }
          fsm_.graph.transition(graphState::INIT);
          break;
        }
        flag_.insideArea = true;
        mtx.obs.lock();
        trg_->setGlobalMap(cs_.obsPtr);
        mtx.obs.unlock();

        auto start_init_graph = tic();
        if (!trg_->initGraph(param_.isPreMap, state_.pose3d)) {
          flag_.graphInit = false;
          print_warning("TRG graph initialization failed; waiting for a valid pose/map and retrying");
          fsm_.graph.transition(graphState::INIT);
          break;
        }
        print_warning("Graph initialization time: " + std::to_string(toc(start_init_graph, "s")) +
                      " sec");
        flag_.graphInit = true;
        fsm_.graph.transition(graphState::UPDATE);
        break;
      }
      case graphState::UPDATE: {
        if (param_.isUpdate) {
          if (!flag_.poseIn || !flag_.obsIn) {
            print_warning("Pose: " + std::to_string(flag_.poseIn) +
                          ", Obs: " + std::to_string(flag_.obsIn));
            fsm_.graph.transition(graphState::UPDATE);
            break;
          }
          // 机器人离开允许区域 → 清空已生长的 TRG(图 + 累积观测地图)，回到 INIT；
          // 重新进入后由 INIT 以当前位姿为根重新建图、继续增量更新。
          if (!trg_->isWithinAllowedArea(state_.pose2d)) {
            trg_->lockGraph();
            trg_->resetGraph("global");
            trg_->resetGraph("local");
            trg_->resetMap("global");
            trg_->resetMap("local");
            trg_->unlockGraph();
            flag_.graphInit  = false;
            flag_.insideArea = false;
            print_warning("Robot left allowed area; TRG cleared, waiting to re-enter");
            fsm_.graph.transition(graphState::INIT);
            break;
          }
          auto start = tic();
          mtx.obs.lock();  // too slow, and not necessary (then, comment this line)
          trg_->setLocalMap(state_.pose2d, cs_.obsPtr);
          if (!param_.isPreMap) trg_->setGlobalMap(cs_.obsPtr);
          mtx.obs.unlock();
          trg_->updateGraph();
          print_success("Graph update time: " + std::to_string(toc(start, "ms")) + " ms");
        }
        fsm_.graph.transition(graphState::UPDATE);
        break;
      }
      case graphState::LOAD: {
        break;
      }
      case graphState::RESET: {
        break;
      }
      case graphState::SAVE: {
        break;
      }
      default: {
        print_error("Invalid graph state");
        exit(1);
      }
    }
    float loop_time   = toc(start_loop, "ms");
    int   remain_time = 1000 / param_.graph_rate - loop_time;
    if (remain_time > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(remain_time));
    }
    thd.hz["graph"] = std::round(1000 / toc(start_loop, "ms") * 100) / 100;
  }
}

void TRGPlanner::runPlanningFSM() {
  while (is_running.load()) {
    auto start_loop = tic();
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));

    if (flag_.goalIn) {
      flag_.goalIn = false;
      fsm_.planning.transition(planningState::PLANNING);
    }
    fsm_.planning.notice();
    switch (fsm_.planning.curr_state_) {
      case planningState::RESET: {
        flag_.pathFound = false;
        path_.clear();
        fsm_.planning.transition(planningState::RESET);
        break;
      }
      case planningState::PLANNING: {
        path_.clear();
        auto start = tic();
        const float start_yaw = quatToYaw(state_.quat);
        const float goal_yaw  = quatToYaw(goal_state_.quat);
        if (trg_->planSafePath(state_.pose2d,
                               goal_state_.pose,
                               path_.raw,
                               path_.direct_dist,
                               path_.raw_path_length,
                               path_.avg_risk,
                               start_yaw,
                               goal_yaw,
                               param_.isUpdate)) {
          path_.planning_time = toc(start, "ms");
          print_success("Path planning time: " + std::to_string(path_.planning_time) + " ms");
          flag_.pathFound     = true;
          flag_.planningCount = 0;
          trg_->refinePath(path_.raw, path_.smooth);
          fsm_.planning.transition(planningState::ONGOING);
        } else {
          print_error("Failed to find a path, count: " + std::to_string(flag_.planningCount));
          flag_.planningCount++;
          if (flag_.planningCount > 10) {
            fsm_.planning.transition(planningState::RESET);
          } else {
            fsm_.planning.transition(planningState::PLANNING);
          }
        }
        break;
      }
      case planningState::ONGOING: {
        if (trg_->checkReadched(state_.pose2d)) {
          print_success("Goal reached");
          fsm_.planning.transition(planningState::RESET);
          break;
        }
        if (trg_->checkReplan(state_.pose2d, path_.raw)) {
          print_warning("Replanning");
          fsm_.planning.transition(planningState::PLANNING);
          break;
        }
        fsm_.planning.transition(planningState::ONGOING);
        break;
      }
      default: {
        print_error("Invalid planning state");
        exit(1);
      }
    }
    float loop_time   = toc(start_loop, "ms");
    int   remain_time = 1000 / param_.planning_rate - loop_time;
    if (remain_time > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(remain_time));
    }
    thd.hz["planning"] = std::round(1000 / toc(start_loop, "ms") * 100) / 100;
  }
}

void GraphFSM::transition(graphState new_state) {
  prev_state_ = curr_state_;
  curr_state_ = new_state;
  if (prev_state_ != curr_state_) {
    print("[Graph] " + state_map_[prev_state_] + " -> " + state_map_[curr_state_]);
  }
}

void GraphFSM::notice() {
  if (prev_state_ != curr_state_) {
    print("[Graph] " + state_map_[curr_state_]);
  }
}

void PlanningFSM::transition(planningState new_state) {
  prev_state_ = curr_state_;
  curr_state_ = new_state;
  if (prev_state_ != curr_state_) {
    print("[Planning] " + state_map_[prev_state_] + " -> " + state_map_[curr_state_]);
  }
}

void PlanningFSM::notice() {
  if (prev_state_ != curr_state_) {
    print("[Planning] " + state_map_[curr_state_]);
  }
}

std::shared_ptr<TRG> TRGPlanner::getTRG() { return trg_; }

void TRGPlanner::setPose(const Eigen::Vector3f& pose     = Eigen::Vector3f::Zero(),
                         const Eigen::Vector4f& quat     = Eigen::Vector4f(1, 0, 0, 0),
                         const std::string&     frame_id = "map") {
  std::lock_guard<std::mutex> lock(mtx.odom);
  state_.frame_id = frame_id;
  state_.pose3d   = pose;
  state_.pose2d   = pose.head(2);
  state_.quat     = quat;
  Eigen::Quaternionf q(state_.quat[0], state_.quat[1], state_.quat[2], state_.quat[3]);
  state_.T_B2M.block<3, 3>(0, 0) = q.toRotationMatrix();
  state_.T_B2M.block<3, 1>(0, 3) = state_.pose3d;
  flag_.poseIn                   = true;
}

void TRGPlanner::setObs(const Eigen::MatrixXf& obs) {
  if (!flag_.poseIn) {
    print_error("Pose is not initialized");
    return;
  }
  std::lock_guard<std::mutex> lock(mtx.obs);
  cs_.obsPtr->clear();
  for (int i = 0; i < obs.rows(); i++) {
    PtsDefault pt;
    pt.x = obs(i, 0);
    pt.y = obs(i, 1);
    pt.z = obs(i, 2);
    cs_.obsPtr->push_back(pt);
  }
  flag_.obsIn = true;
}

void TRGPlanner::setGoal(const Eigen::Vector3f& pose,
                         const Eigen::Vector4f& quat = Eigen::Vector4f(1, 0, 0, 0)) {
  if (!flag_.graphInit) {
    print_error("Graph is not initialized");
    return;
  }
  Eigen::Vector2f goal2d = pose.head(2);
  if (trg_ != nullptr && !trg_->isWithinAllowedArea(goal2d)) {
    print_error("Goal is outside allowed area");
    return;
  }
  std::lock_guard<std::mutex> lock(mtx.goal);
  goal_state_.pose = pose;
  goal_state_.quat = quat;
  goal_state_.init = true;
  flag_.goalIn     = true;
}

std::vector<Eigen::Vector3f> TRGPlanner::getPlannedPath(const std::string& type) {
  if (!flag_.pathFound) {
    print_error("Path is not found");
    return {};
  }
  std::vector<Eigen::Vector3f> path;
  if (type == "raw") {
    for (auto& pt : path_.raw) {
      path.push_back(pt.head(3));
    }
  } else if (type == "smooth") {
    for (auto& pt : path_.smooth) {
      path.push_back(pt.head(3));
    }
  } else {
    print_error("Invalid path type");
    return {};
  }
  return path;
}

std::vector<float> TRGPlanner::getPathInfo() {
  if (!flag_.pathFound) {
    print_error("Path is not found");
    return {};
  }
  std::vector<float> info;
  info.push_back(path_.direct_dist);
  info.push_back(path_.raw_path_length);
  info.push_back(path_.smooth_path_length);
  info.push_back(path_.planning_time);
  info.push_back(path_.avg_risk);
  return info;
}

Eigen::MatrixXf TRGPlanner::getMapEigen(const std::string& type = "pre") {
  if (type == "pre") {
    if (!param_.isPreMap) {
      print_error("Prebuilt map is not loaded");
      return Eigen::MatrixXf();
    }
    return PointCloudToEigen(cs_.preMapPtr);
  } else if (type == "obs") {
    return PointCloudToEigen(cs_.obsPtr);
  } else {
    print_error("Invalid map type");
    return Eigen::MatrixXf();
  }
}

Eigen::Vector3f TRGPlanner::getGoalPose() {
  if (!goal_state_.init) {
    print_error("Goal is not initialized");
    return Eigen::Vector3f::Zero();
  }
  return goal_state_.pose;
}

Eigen::Vector4f TRGPlanner::getGoalQuat() {
  if (!goal_state_.init) {
    print_error("Goal is not initialized");
    return Eigen::Vector4f::Zero();
  }
  return goal_state_.quat;
}
