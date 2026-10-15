/**
 * Copyright 2025, Korea Advanced Institute of Science and Technology
 * Massachusetts Institute of Technology,
 * Daejeon, 34051
 * All Rights Reserved
 * Authors: Dongkyu Lee, et al.
 * See LICENSE for the license information
 */
#include "trg_planner/include/graph/trg.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <deque>
#include <limits>
#include <queue>
#include <unordered_set>

namespace {
constexpr float kTwoPi = 2.0f * static_cast<float>(M_PI);
constexpr float kEps   = 1e-6f;

float clampRatio(float value, float limit) {
  if (!std::isfinite(value)) {
    return 2.0f;
  }
  return std::min(3.0f, std::max(0.0f, value / std::max(limit, kEps)));
}

float deterministicAngleOffset(int node_id, int random_seed, int sample_num) {
  const std::uint32_t seed = random_seed >= 0 ? static_cast<std::uint32_t>(random_seed) : 0U;
  std::uint32_t value =
      static_cast<std::uint32_t>(node_id) * 1103515245U + (seed + 1U) * 12345U;
  value ^= value >> 16;
  value *= 2246822519U;
  value ^= value >> 13;
  const float unit = static_cast<float>(value & 0x00ffffffU) /
                     static_cast<float>(0x01000000U);
  return unit * kTwoPi / static_cast<float>(std::max(1, sample_num));
}

bool goalNodeCoversRequestedGoal(const char*               planner_name,
                                 const TRG::Node*          goal_node,
                                 bool                      is_goal_known,
                                 const Eigen::Vector2f&    goal2d,
                                 float                     goal_tolerance,
                                 bool                      allow_goal_subgoal) {
  if (goal_node == nullptr) {
    print_error(std::string("Cannot plan ") + planner_name + " path: no TRG node near goal pose");
    return false;
  }

  const float snap_dist = (goal_node->pos_.head(2) - goal2d).norm();
  if (!allow_goal_subgoal && !is_goal_known && snap_dist > goal_tolerance) {
    print_error(std::string("Cannot plan ") + planner_name + " path: nearest TRG node is " +
                std::to_string(snap_dist) + " m from the requested goal, beyond goal tolerance " +
                std::to_string(goal_tolerance) + " m; graph does not cover the goal");
    return false;
  }
  return true;
}
}  // namespace

TRG::TRG(bool  isVerbose,
         float expand_dist,
         float robot_size,
         int   sample_num,
         float height_threshold,
         float collision_threshold,
         float update_collision_threshold,
         float safety_factor,
         float goal_tolerance,
         int   random_seed,
         bool  deterministic_sampling,
         SamplingMode sampling_mode,
         float lattice_spacing,
         float lattice_refinement_spacing,
         float lattice_refinement_margin,
         float lattice_connect_radius,
         float lattice_min_node_spacing,
         bool  lattice_refinement_enabled,
         bool  lattice_fallback_ring)
    : gen_(random_seed >= 0 ? static_cast<std::mt19937::result_type>(random_seed) : rd_()),
      distr_(0.0, 1.0) {
  param_.isVerbose                  = isVerbose;
  param_.expand_dist                = expand_dist;
  param_.robot_size                 = robot_size;
  param_.sample_num                 = sample_num;
  param_.random_seed                = random_seed;
  param_.deterministic_sampling     = deterministic_sampling;
  param_.sampling_mode              = sampling_mode;
  param_.lattice_spacing            = lattice_spacing;
  param_.lattice_refinement_spacing = lattice_refinement_spacing;
  param_.lattice_refinement_margin  = lattice_refinement_margin;
  param_.lattice_connect_radius     = lattice_connect_radius;
  param_.lattice_min_node_spacing   = lattice_min_node_spacing;
  param_.lattice_refinement_enabled = lattice_refinement_enabled;
  param_.lattice_fallback_ring      = lattice_fallback_ring;
  param_.height_threshold           = height_threshold;
  param_.collision_threshold        = collision_threshold;
  param_.update_collision_threshold = update_collision_threshold;
  param_.safety_factor              = safety_factor;
  param_.goal_tolerance             = goal_tolerance;
  this->resetGraph("global");
  this->resetGraph("local");
  this->resetMap("global");
  this->resetMap("local");
}

void TRG::setPathSearchConfig(PathSearchMode mode, const TRGAStarConfig& config) {
  std::lock_guard<std::mutex> lock(mtx.graph);
  param_.path_search_mode = mode;
  param_.trg_astar        = config;
  param_.trg_astar.heading_bins =
      std::max(4, param_.trg_astar.heading_bins);
  if (param_.trg_astar.heading_bins % 2 != 0) {
    param_.trg_astar.heading_bins += 1;
  }
  if (param_.trg_astar.length_scale_m <= EPS) {
    param_.trg_astar.length_scale_m = param_.expand_dist;
  }
  param_.trg_astar.climb_scale_m   = std::max(param_.trg_astar.climb_scale_m, kEps);
  param_.trg_astar.slope_scale_tan = std::max(param_.trg_astar.slope_scale_tan, kEps);
  param_.trg_astar.turn_scale      = std::max(param_.trg_astar.turn_scale, kEps);
  param_.trg_astar.robot_length_m =
      std::max(param_.trg_astar.robot_length_m, param_.robot_size);
  param_.trg_astar.robot_width_m =
      std::max(param_.trg_astar.robot_width_m, param_.robot_size);
  param_.trg_astar.max_body_height_diff_m =
      std::max(param_.trg_astar.max_body_height_diff_m, kEps);
  param_.trg_astar.max_body_tilt_deg =
      std::max(param_.trg_astar.max_body_tilt_deg, kEps);
  param_.trg_astar.max_interior_penetration_m =
      std::max(param_.trg_astar.max_interior_penetration_m, kEps);
  param_.trg_astar.footprint_sample_step_m =
      std::max(param_.trg_astar.footprint_sample_step_m, 0.0f);
  param_.trg_astar.footprint_edge_band_m =
      std::max(param_.trg_astar.footprint_edge_band_m, 0.01f);
  footprint_edge_metric_cache_.clear();

  const std::string mode_name =
      param_.path_search_mode == PathSearchMode::TRGAStar ? "trg_astar" : "native";
  print("TRG path search mode: " + mode_name, param_.isVerbose);
}

bool TRG::initGraph(bool isPreMap, Eigen::Vector3f start3d = Eigen::Vector3f::Zero()) {
  std::lock_guard<std::mutex> lock(mtx.graph);
  print("TRG initGraph", param_.isVerbose);
  trgStruct& graph = *trgMap_["global"];
  this->resetGraph(graph.type);

  if (graph.cloud_map == nullptr || graph.cloud_map->empty()) {
    print_error("Cannot initialize TRG graph: map is empty");
    return false;
  }
  if (!this->isWithinAllowedArea(start3d.head(2))) {
    print_error("Cannot initialize TRG graph: initial pose is outside allowed area");
    return false;
  }

  if (param_.sampling_mode == SamplingMode::Lattice) {
    if (!this->rebuildLatticeGraph(start3d.head(2), false)) {
      print_error("Failed to generate lattice TRG graph near initial pose");
      return false;
    }
    return true;
  }

  graph.root_pos           = start3d.head(2);
  std::vector<Eigen::Vector2f> root_candidates;
  root_candidates.push_back(graph.root_pos);
  const int candidate_num = std::max(8, param_.sample_num);
  for (int i = 0; i < candidate_num; ++i) {
    const float angle = kTwoPi * static_cast<float>(i) / static_cast<float>(candidate_num);
    root_candidates.push_back(graph.root_pos +
                              Eigen::Vector2f(param_.expand_dist * std::cos(angle),
                                              param_.expand_dist * std::sin(angle)));
  }

  for (const auto& candidate : root_candidates) {
    Eigen::Vector2f root_pos = candidate;
    if (this->addNode(graph.node_id, root_pos, NodeState::Valid, graph.type)) {
      break;
    }
  }

  int cnt = 0;
  while (graph.nodes.empty() && cnt < 100) {
    const float angle = distr_(gen_) * kTwoPi;
    const float radius = param_.expand_dist * (0.5f + distr_(gen_));
    Eigen::Vector2f root_pos =
        graph.root_pos + Eigen::Vector2f(radius * std::cos(angle), radius * std::sin(angle));
    this->addNode(graph.node_id, root_pos, NodeState::Valid, graph.type);
    cnt++;
  }

  if (graph.nodes.empty()) {
    print_error("Failed to generate root node near initial pose");
    print_error("Check that the initial pose is inside a valid, sufficiently dense prior-map area");
    return false;
  }

  this->expandGraph(graph.node_id - 1, graph.type);
  this->cleanGraph(false);

  if (graph.nodes.empty()) {
    print_error("TRG graph initialization produced no valid nodes");
    print_error("Check that expandDist is larger than robotSize and that the initial pose is inside the map");
    return false;
  }
  return true;
}

void TRG::loadPrebuiltGraph() {
  //// TODO: load prebuilt graph
  print("TRG loadPrebuiltGraph");
}

void TRG::setGlobalMap(PointCloudPtr& map) {
  print("TRG setGlobalMap", param_.isVerbose);
  trgStruct& global_graph = *trgMap_["global"];
  this->resetMap("global");
  *global_graph.cloud_map = *map;

  for (int i = 0; i < global_graph.cloud_map->size(); ++i) {
    PtsDefault& pt = global_graph.cloud_map->points[i];
    kd_insert2(global_graph.map_tree, pt.x, pt.y, &pt);
  }

  print("Root pos: " + std::to_string(global_graph.root_pos.x()) + ", " +
            std::to_string(global_graph.root_pos.y()),
        param_.isVerbose);
}

void TRG::setLocalMap(Eigen::Vector2f start2d, PointCloudPtr& map) {
  std::lock_guard<std::mutex> lock(mtx.graph);
  trgStruct&                  local_graph = *trgMap_["local"];
  this->resetMap("local");

  local_graph.root_pos   = start2d;
  *local_graph.cloud_map = *map;

  for (int i = 0; i < local_graph.cloud_map->size(); ++i) {
    PtsDefault& pt = local_graph.cloud_map->points[i];
    kd_insert2(local_graph.map_tree, pt.x, pt.y, &pt);
  }

  this->setLocalGraph(false);
}

void TRG::setLocalGraph(bool useMutex = false) {
  if (useMutex) {
    std::lock_guard<std::mutex> lock(mtx.graph);
  }
  trgStruct& global_graph = *trgMap_["global"];
  trgStruct& local_graph  = *trgMap_["local"];
  this->resetGraph("local");
  for (auto& node : global_graph.nodes) {
    if (node.second == nullptr) {
      continue;
    }
    kdres* res = kd_nearest_range2(local_graph.map_tree,
                                   node.second->pos_.x(),
                                   node.second->pos_.y(),
                                   param_.robot_size * 0.5);
    if (res == nullptr) {
      continue;
    }
    if (kd_res_size(res) == 0) {
      kd_res_free(res);
      continue;
    }
    local_graph.nodes[node.first] = node.second;
    kd_insert2(local_graph.node_tree, node.second->pos_.x(), node.second->pos_.y(), node.second);
    kd_res_free(res);
  }
}

bool TRG::addNode(int node_id, Eigen::Vector2f& node_pos, NodeState state, std::string type) {
  trgStruct& graph = *trgMap_[type];
  if (graph.cloud_map == nullptr || graph.cloud_map->empty()) {
    return false;
  }
  if (!this->isWithinAllowedArea(node_pos)) {
    return false;
  }

  /// Check reference position is valid
  if (node_id == 0) {
    if (this->isCollision(node_pos, graph.type, param_.collision_threshold)) {
      return false;
    }
  }

  /// Create new node
  kdres*      res = kd_nearest2(graph.map_tree, node_pos.x(), node_pos.y());
  if (res == nullptr) {
    return false;
  }
  PtsDefault* pt  = reinterpret_cast<PtsDefault*>(kd_res_item_data(res));
  kd_res_free(res);
  if (pt == nullptr) {
    return false;
  }
  Node* node           = new Node(node_id, node_pos, pt->z, state);
  graph.nodes[node_id] = node;
  kd_insert2(graph.node_tree, node_pos.x(), node_pos.y(), node);
  graph.node_id++;
  return true;
}

void TRG::wireEdge(Node* node1, Node* node2, std::string type) {
  if (node1->id_ == node2->id_) {
    return;
  }
  for (auto& edge : node1->edges_) {
    if (edge->dst_id_ == node2->id_) {
      return;
    }
  }
  for (auto& edge : node2->edges_) {
    if (edge->dst_id_ == node1->id_) {
      return;
    }
  }

  float max_slope = atan2(param_.height_threshold, param_.robot_size);
  float slope     = atan2(fabs(node1->pos_.z() - node2->pos_.z()),
                      (node1->pos_.head(2) - node2->pos_.head(2)).norm());
  if (slope > max_slope) {
    return;
  }

  Eigen::Vector2f delta  = node2->pos_.head(2) - node1->pos_.head(2);
  float           dist   = delta.norm();
  if (dist < EPS || dist >= 2.5 * param_.expand_dist) {
    return;
  }
  Eigen::Vector2f dir    = delta / dist;
  Eigen::Vector2f center = node1->pos_.head(2) + 0.5 * dist * dir;

  /// check collision between node1 and node2
  float ds = param_.robot_size * 0.5;
  for (float i = 0; i < dist; i += ds) {
    Eigen::Vector2f pos = node1->pos_.head(2) + i * dir;
    if (!this->isWithinAllowedArea(pos)) {
      return;
    }
    if (this->isCollision(pos, type, param_.collision_threshold)) {
      return;
    }
  }

  /// get Elipse points with focus at node1 and node2
  float c = 0.5 * dist;         // focus distance
  float b = param_.robot_size;  // minor axis
  float a = b;                  // if c < b, ellipse is circle
  if (c >= b) {                 // if c >= b, ellipse is ellipse
    a = sqrt(c * c + b * b);    // major axis
  }
  bool isCircle = (a == b) ? true : false;

  /// get circle points with center at center point and radius of a
  trgStruct&                       graph = *trgMap_[type];
  pcl::PointCloud<PtsDefault>::Ptr ellipse_pts(new pcl::PointCloud<PtsDefault>());
  Eigen::Matrix2f                  R;        // rotation matrix for TF from global to local
  R << dir.x(), -dir.y(), dir.y(), dir.x();  // (dir is x-axis)
  kdres* res = kd_nearest_range2(graph.map_tree, center.x(), center.y(), a);
  if (res == nullptr) {
    return;
  }
  if (kd_res_size(res) == 0) {
    kd_res_free(res);
    return;
  }
  while (!kd_res_end(res)) {
    PtsDefault*     pt = reinterpret_cast<PtsDefault*>(kd_res_item_data(res));
    if (pt == nullptr) {
      kd_res_next(res);
      continue;
    }
    PtsDefault      p;
    Eigen::Vector2f p2d(pt->x - center.x(), pt->y - center.y());
    p2d = R * p2d;
    p.x = p2d.x();
    p.y = p2d.y();
    p.z = pt->z;
    if (isCircle) {
      ellipse_pts->push_back(p);
    } else {
      if ((p.x * p.x) * (b * b) + (p.y * p.y) * (a * a) < a * a * b * b) {
        ellipse_pts->push_back(p);
      }
    }
    kd_res_next(res);
  }
  kd_res_free(res);
  if (ellipse_pts->size() < 3) {
    return;
  }

  /// get normal vector of the ellipse
  int             n = ellipse_pts->size();
  Eigen::MatrixXf A(n, 3);
  for (int i = 0; i < n; i++) {
    A.row(i) << ellipse_pts->points[i].x, ellipse_pts->points[i].y, ellipse_pts->points[i].z;
  }
  Eigen::MatrixXf centered = A.rowwise() - A.colwise().mean();
  Eigen::MatrixXf cov      = (centered.adjoint() * centered) / static_cast<double>(A.rows() - 1);
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
  Eigen::MatrixXf                   eigenvectors = svd.matrixU().normalized();
  Eigen::Vector3f                   normal       = eigenvectors.col(2);

  if (normal.z() < 0) {
    normal = -normal;
  }

  float hor_grad = eigenvectors.col(0).dot(-gravity);
  float ver_grad = eigenvectors.col(1).dot(-gravity);
  if (hor_grad < 0) {
    hor_grad = (-eigenvectors.col(0)).dot(-gravity);
  }
  if (ver_grad < 0) {
    ver_grad = (-eigenvectors.col(1)).dot(-gravity);
  }

  if (hor_grad < 0 || ver_grad < 0) {
    return;
  }

  float ratio  = 0.8;
  float weight = ratio * hor_grad + (1 - ratio) * ver_grad;
  if (weight < 0.1) {
    weight = 0.0;
  }

  Edge* edge_1 = new Edge(node2->id_, weight, dist);
  Edge* edge_2 = new Edge(node1->id_, weight, dist);
  node1->edges_.push_back(edge_1);
  node2->edges_.push_back(edge_2);
  return;
}

void TRG::expandGraph(int ref_id, std::string type) {
  trgStruct& graph    = *trgMap_[type];
  const auto ref_it = graph.nodes.find(ref_id);
  if (ref_it == graph.nodes.end() || ref_it->second == nullptr) {
    print_error("Cannot expand TRG graph: reference node is missing");
    return;
  }
  Node*      ref_node = ref_it->second;

  std::deque<Node*> expand_queue;
  expand_queue.push_back(ref_node);
  while (!expand_queue.empty()) {
    // std::this_thread::sleep_for(std::chrono::milliseconds(5)); // delay for visualization
    Node* node = expand_queue.front();
    expand_queue.pop_front();

    /// Sampling
    std::vector<Eigen::Vector2f> samples;
    int                          max_trial_sample =
        param_.deterministic_sampling ? param_.sample_num : 1000;
    int                          trial_sample     = 0;
    while (samples.size() < param_.sample_num) {
      if (trial_sample >= max_trial_sample) {
        break;
      }
      // 0.75 param_.expand_dist ~ 1.25 param_.expand_dist
      // float expand_dist = 0.75 * param_.expand_dist + distr_(gen_) * (1.25 * param_.expand_dist -
      // 0.75 * param_.expand_dist); // 0.75 ~ 1.25
      float           expand_dist = param_.expand_dist;
      float           angle;
      if (param_.deterministic_sampling) {
        const float angle_step = kTwoPi / static_cast<float>(std::max(1, param_.sample_num));
        angle = deterministicAngleOffset(node->id_, param_.random_seed, param_.sample_num) +
                static_cast<float>(trial_sample) * angle_step;
        trial_sample++;
      } else {
        angle = distr_(gen_) * kTwoPi;
      }
      Eigen::Vector2f sample =
          node->pos_.head(2) + Eigen::Vector2f(expand_dist * cos(angle), expand_dist * sin(angle));
      if (this->isCollision(sample, graph.type, param_.collision_threshold)) {
        if (!param_.deterministic_sampling) {
          trial_sample++;
        }
        continue;
      }
      samples.push_back(sample);
    }

    /// Add new nodes and wire edges
    for (auto& sample : samples) {
      /// 1. Check if the sample is already in the graph
      kdres* res           = kd_nearest2(graph.node_tree, sample.x(), sample.y());
      if (res == nullptr) {
        continue;
      }
      Node*  existing_node = reinterpret_cast<Node*>(kd_res_item_data(res));
      kd_res_free(res);
      if (existing_node == nullptr) {
        continue;
      }
      if (existing_node->state_ == NodeState::Invalid) {
        continue;
      }
      if ((existing_node->pos_.head(2) - sample).norm() < param_.robot_size) {
        this->wireEdge(node, existing_node, graph.type);
        continue;
      }

      /// 2. Add new node
      NodeState new_state = (ref_id == 0) ? NodeState::Valid : NodeState::Frontier;
      if (!this->addNode(graph.node_id, sample, new_state, graph.type)) {
        continue;
      }
      const auto new_node_it = graph.nodes.find(graph.node_id - 1);
      if (new_node_it == graph.nodes.end() || new_node_it->second == nullptr) {
        continue;
      }
      Node* new_node = new_node_it->second;
      this->wireEdge(
          node, new_node, graph.type);  /// 2.1 Wire edge between new node and reference node

      /// 3. Wire edge between new node and existing nodes
      if (param_.expand_dist - param_.robot_size < 0.25 * param_.expand_dist) {
        kdres* res2 = kd_nearest_range2(
            graph.node_tree, new_node->pos_.x(), new_node->pos_.y(), param_.expand_dist);
        if (res2 != nullptr && kd_res_size(res2) > 0) {
          while (!kd_res_end(res2)) {
            Node* existing_node = reinterpret_cast<Node*>(kd_res_item_data(res2));
            if (existing_node == nullptr) {
              kd_res_next(res2);
              continue;
            }
            if (existing_node->state_ == NodeState::Invalid) {
              kd_res_next(res2);
              continue;
            }
            this->wireEdge(new_node, existing_node, graph.type);
            kd_res_next(res2);
          }
        }
        if (res2 != nullptr) {
          kd_res_free(res2);
        }
      }

      /// 4. Add new node to expand queue
      if (new_node->edges_.size() < 1) {
        new_node->state_ = NodeState::Invalid;
        continue;
      }
      expand_queue.push_back(new_node);
    }
  }
}

bool TRG::addSeedNodeNearRoot(const Eigen::Vector2f& root2d, std::string type, int& out_node_id) {
  trgStruct& graph = *trgMap_[type];
  out_node_id      = -1;

  std::vector<Eigen::Vector2f> root_candidates;
  root_candidates.push_back(root2d);
  const int candidate_num = std::max(8, param_.sample_num);
  for (int i = 0; i < candidate_num; ++i) {
    const float angle = kTwoPi * static_cast<float>(i) / static_cast<float>(candidate_num);
    root_candidates.push_back(root2d +
                              Eigen::Vector2f(param_.expand_dist * std::cos(angle),
                                              param_.expand_dist * std::sin(angle)));
  }

  for (const auto& candidate : root_candidates) {
    Eigen::Vector2f seed = candidate;
    if (this->isCollision(seed, graph.type, param_.collision_threshold)) {
      continue;
    }
    if (this->addNode(graph.node_id, seed, NodeState::Valid, graph.type)) {
      out_node_id = graph.node_id - 1;
      return true;
    }
  }

  int cnt = 0;
  while (cnt < 100) {
    const float angle  = distr_(gen_) * kTwoPi;
    const float radius = param_.expand_dist * (0.5f + distr_(gen_));
    Eigen::Vector2f seed =
        root2d + Eigen::Vector2f(radius * std::cos(angle), radius * std::sin(angle));
    if (!this->isCollision(seed, graph.type, param_.collision_threshold) &&
        this->addNode(graph.node_id, seed, NodeState::Valid, graph.type)) {
      out_node_id = graph.node_id - 1;
      return true;
    }
    cnt++;
  }
  return false;
}

bool TRG::tryAddLatticeNode(const Eigen::Vector2f& pos,
                            TRG::NodeState        state,
                            std::string           type,
                            float                 min_spacing) {
  trgStruct& graph = *trgMap_[type];
  Eigen::Vector2f node_pos = pos;
  if (this->isCollision(node_pos, graph.type, param_.collision_threshold)) {
    return false;
  }

  if (graph.node_id > 0) {
    kdres* res = kd_nearest2(graph.node_tree, node_pos.x(), node_pos.y());
    if (res != nullptr) {
      Node* existing_node = reinterpret_cast<Node*>(kd_res_item_data(res));
      kd_res_free(res);
      if (existing_node != nullptr &&
          (existing_node->pos_.head(2) - node_pos).norm() < min_spacing) {
        return false;
      }
    }
  }

  return this->addNode(graph.node_id, node_pos, state, graph.type);
}

bool TRG::isNearCollisionBoundary(const Eigen::Vector2f& pos,
                                  std::string           type,
                                  float                 probe_radius) {
  const float diag = probe_radius * 0.70710678f;
  const std::array<Eigen::Vector2f, 8> probes = {
      Eigen::Vector2f(probe_radius, 0.0f),
      Eigen::Vector2f(-probe_radius, 0.0f),
      Eigen::Vector2f(0.0f, probe_radius),
      Eigen::Vector2f(0.0f, -probe_radius),
      Eigen::Vector2f(diag, diag),
      Eigen::Vector2f(diag, -diag),
      Eigen::Vector2f(-diag, diag),
      Eigen::Vector2f(-diag, -diag),
  };

  for (const auto& delta : probes) {
    Eigen::Vector2f probe = pos + delta;
    if (this->isCollision(probe, type, param_.collision_threshold)) {
      return true;
    }
  }
  return false;
}

void TRG::addLatticeCandidates(std::string type,
                               float       min_x,
                               float       max_x,
                               float       min_y,
                               float       max_y,
                               float       spacing,
                               bool        boundary_only) {
  if (spacing <= EPS) {
    return;
  }

  const float row_step = spacing * 0.8660254f;
  const float start_x  = std::floor(min_x / spacing) * spacing;
  const float start_y  = std::floor(min_y / row_step) * row_step;
  const float end_x    = std::ceil(max_x / spacing) * spacing;
  const float end_y    = std::ceil(max_y / row_step) * row_step;
  const float min_spacing =
      std::max(0.05f, std::min(param_.lattice_min_node_spacing, spacing * 0.85f));

  int row = 0;
  for (float y = start_y; y <= end_y + EPS; y += row_step) {
    const float x_offset = (row % 2 == 0) ? 0.0f : 0.5f * spacing;
    for (float x = start_x + x_offset; x <= end_x + EPS; x += spacing) {
      Eigen::Vector2f pos(x, y);
      if (boundary_only &&
          !this->isNearCollisionBoundary(pos, type, param_.lattice_refinement_margin)) {
        continue;
      }
      this->tryAddLatticeNode(pos, NodeState::Valid, type, min_spacing);
    }
    row++;
  }
}

void TRG::wireLatticeGraph(std::string type) {
  trgStruct& graph = *trgMap_[type];
  for (auto& node_entry : graph.nodes) {
    Node* node = node_entry.second;
    if (node == nullptr || node->state_ == NodeState::Invalid) {
      continue;
    }

    kdres* res = kd_nearest_range2(
        graph.node_tree, node->pos_.x(), node->pos_.y(), param_.lattice_connect_radius);
    if (res == nullptr) {
      continue;
    }
    while (!kd_res_end(res)) {
      Node* other = reinterpret_cast<Node*>(kd_res_item_data(res));
      if (other == nullptr || other->state_ == NodeState::Invalid ||
          other->id_ <= node->id_) {
        kd_res_next(res);
        continue;
      }
      this->wireEdge(node, other, graph.type);
      kd_res_next(res);
    }
    kd_res_free(res);
  }
}

int TRG::pruneToReachableComponent(const Eigen::Vector2f& root2d, std::string type) {
  trgStruct& graph = *trgMap_[type];
  if (graph.nodes.empty()) {
    return 0;
  }

  kdres* res = kd_nearest2(graph.node_tree, root2d.x(), root2d.y());
  if (res == nullptr) {
    return 0;
  }
  Node* start = reinterpret_cast<Node*>(kd_res_item_data(res));
  kd_res_free(res);
  if (start == nullptr || start->state_ == NodeState::Invalid) {
    return 0;
  }

  std::unordered_set<int> reachable;
  std::queue<Node*>       queue;
  reachable.insert(start->id_);
  queue.push(start);

  while (!queue.empty()) {
    Node* node = queue.front();
    queue.pop();
    for (Edge* edge : node->edges_) {
      if (edge == nullptr || reachable.find(edge->dst_id_) != reachable.end()) {
        continue;
      }
      const auto dst_it = graph.nodes.find(edge->dst_id_);
      if (dst_it == graph.nodes.end() || dst_it->second == nullptr ||
          dst_it->second->state_ == NodeState::Invalid) {
        continue;
      }
      reachable.insert(edge->dst_id_);
      queue.push(dst_it->second);
    }
  }

  for (auto& node_entry : graph.nodes) {
    if (node_entry.second == nullptr) {
      continue;
    }
    if (reachable.find(node_entry.second->id_) == reachable.end()) {
      node_entry.second->state_ = NodeState::Invalid;
    }
  }
  return static_cast<int>(reachable.size());
}

void TRG::classifyLatticeFrontiers() {
  trgStruct& local_graph = *trgMap_["local"];
  if (local_graph.cloud_map == nullptr || local_graph.cloud_map->empty()) {
    return;
  }

  trgStruct& graph = *trgMap_["global"];
  for (auto& node_entry : graph.nodes) {
    Node* node = node_entry.second;
    if (node == nullptr || node->state_ == NodeState::Invalid) {
      continue;
    }
    Eigen::Vector2f pos = node->pos_.head(2);
    node->state_        = this->isFrontier(pos) ? NodeState::Frontier : NodeState::Valid;
  }
}

bool TRG::rebuildLatticeGraph(Eigen::Vector2f root2d, bool updateLocal) {
  trgStruct& graph = *trgMap_["global"];
  if (graph.cloud_map == nullptr || graph.cloud_map->empty()) {
    return false;
  }

  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  int   usable_points = 0;
  for (const auto& pt : graph.cloud_map->points) {
    Eigen::Vector2f pos(pt.x, pt.y);
    if (!this->isWithinAllowedArea(pos)) {
      continue;
    }
    min_x = std::min(min_x, pt.x);
    min_y = std::min(min_y, pt.y);
    max_x = std::max(max_x, pt.x);
    max_y = std::max(max_y, pt.y);
    usable_points++;
  }
  if (usable_points == 0 || !std::isfinite(min_x) || !std::isfinite(min_y) ||
      !std::isfinite(max_x) || !std::isfinite(max_y)) {
    return false;
  }

  this->resetGraph(graph.type);
  graph.root_pos = root2d;

  int seed_id = -1;
  this->addSeedNodeNearRoot(root2d, graph.type, seed_id);

  const float padding = std::max(param_.lattice_connect_radius, param_.lattice_spacing);
  this->addLatticeCandidates(graph.type,
                             min_x - padding,
                             max_x + padding,
                             min_y - padding,
                             max_y + padding,
                             param_.lattice_spacing,
                             false);
  if (param_.lattice_refinement_enabled &&
      param_.lattice_refinement_spacing < param_.lattice_spacing) {
    this->addLatticeCandidates(graph.type,
                               min_x - padding,
                               max_x + padding,
                               min_y - padding,
                               max_y + padding,
                               param_.lattice_refinement_spacing,
                               true);
  }

  if (graph.nodes.empty()) {
    return false;
  }

  this->wireLatticeGraph(graph.type);
  this->classifyLatticeFrontiers();
  const int reachable_count = this->pruneToReachableComponent(root2d, graph.type);
  this->cleanGraph(updateLocal);

  if (!graph.nodes.empty()) {
    return true;
  }

  if (!param_.lattice_fallback_ring) {
    return false;
  }

  this->resetGraph(graph.type);
  seed_id = -1;
  if (!this->addSeedNodeNearRoot(root2d, graph.type, seed_id) || seed_id < 0) {
    return false;
  }
  this->expandGraph(seed_id, graph.type);
  this->cleanGraph(updateLocal);
  if (!graph.nodes.empty()) {
    print_warning("Lattice TRG fell back to ring expansion; reachable lattice nodes: " +
                  std::to_string(reachable_count));
    return true;
  }
  return false;
}

void TRG::updateGraph() {
  // print("TRG updateGraph", param_.isVerbose);
  std::lock_guard<std::mutex> lock(mtx.graph);

  trgStruct& global_graph = *trgMap_["global"];
  trgStruct& local_graph  = *trgMap_["local"];

  if (param_.sampling_mode == SamplingMode::Lattice) {
    if (!this->rebuildLatticeGraph(local_graph.root_pos, true)) {
      print_warning("Lattice TRG update produced no valid reachable graph");
    }
    return;
  }

  std::deque<Node*> expand_queue;
  for (auto& node : local_graph.nodes) {
    if (node.second == nullptr) {
      continue;
    }
    Eigen::Vector2f npos2d = node.second->pos_.head(2);
    if ((npos2d - local_graph.root_pos).norm() > 2.0 * param_.expand_dist) {
      if (this->isCollision(npos2d, local_graph.type, param_.update_collision_threshold) ||
          node.second->edges_.size() < 1) {
        node.second->state_ = NodeState::Invalid;
        continue;
      }
    }

    if (this->isFrontier(npos2d) && node.second->state_ == NodeState::Frontier) {
      node.second->state_ = NodeState::Frontier;
      expand_queue.push_back(node.second);
      continue;
    }
    expand_queue.push_back(node.second);
    node.second->state_ = NodeState::Valid;
  }

  while (!expand_queue.empty()) {
    Node* node = expand_queue.front();
    expand_queue.pop_front();
    this->expandGraph(node->id_, global_graph.type);
  }
  this->cleanGraph(true);
}

void TRG::cleanGraph(bool updateLocal = true) {
  trgStruct&                     global_graph = *trgMap_["global"];
  std::unordered_map<int, int>   old2new;
  std::unordered_map<int, Node*> new_nodes;
  int                            new_id = 0;
  for (auto& node : global_graph.nodes) {
    if (node.second == nullptr || node.second->state_ == NodeState::Invalid ||
        node.second->edges_.size() < 1) {
      continue;
    }
    node.second->id_    = new_id;
    new_nodes[new_id]   = node.second;
    old2new[node.first] = new_id;
    new_id++;
  }

  for (auto& node : new_nodes) {
    std::vector<Edge*> new_edges;
    for (auto& edge : node.second->edges_) {
      const auto dst_it = global_graph.nodes.find(edge->dst_id_);
      if (dst_it == global_graph.nodes.end() || dst_it->second == nullptr ||
          dst_it->second->state_ == NodeState::Invalid) {
        continue;
      }
      const auto new_dst_it = old2new.find(edge->dst_id_);
      if (new_dst_it == old2new.end()) {
        continue;
      }
      Edge* new_edge = new Edge(new_dst_it->second, edge->weight_, edge->dist_);
      new_edges.push_back(new_edge);
    }
    node.second->edges_.clear();
    node.second->edges_ = new_edges;
  }

  this->resetGraph(global_graph.type);
  global_graph.nodes   = new_nodes;
  global_graph.node_id = new_id;
  for (auto& node : global_graph.nodes) {
    kd_insert2(global_graph.node_tree, node.second->pos_.x(), node.second->pos_.y(), node.second);
  }

  if (updateLocal) {
    this->setLocalGraph(false);
  }
}

void TRG::setGoal(Eigen::Vector3f& goal) {
  // print("TRG setGoal", param_.isVerbose);
  trgStruct& global_graph = *trgMap_["global"];

  goal_.pose3d = goal;
  goal_.pose2d = goal.head(2);
  goal_.node   = nullptr;
  goal_.isKnown = false;

  if (global_graph.nodes.empty()) {
    return;
  }

  kdres* res = kd_nearest_range2(global_graph.node_tree, goal.x(), goal.y(), param_.robot_size);
  if (res == nullptr || kd_res_size(res) == 0) {
    if (res != nullptr) {
      kd_res_free(res);
    }
    float min_dist = std::numeric_limits<float>::max();
    for (auto& node : global_graph.nodes) {
      if (node.second == nullptr) {
        continue;
      }
      // if (node.second->state_ != NodeState::Frontier) {
      //     continue;
      // }
      float dist = (node.second->pos_.head(2) - goal.head(2)).norm();
      if (dist < min_dist) {
        min_dist   = dist;
        goal_.node = node.second;
      }
    }
    goal_.isKnown = false;
  } else {
    Node* near_node = reinterpret_cast<Node*>(kd_res_item_data(res));
    goal_.node      = near_node;
    kd_res_free(res);
    if (near_node == nullptr) {
      goal_.isKnown = false;
      return;
    }
    goal_.isKnown = true;
  }
}

bool TRG::checkReadched(Eigen::Vector2f& pos2d) {
  // print("TRG checkReached", param_.isVerbose);
  float dist = (goal_.pose2d - pos2d).norm();
  if (dist < param_.goal_tolerance) {
    return true;
  }
  return false;
}

bool TRG::checkReplan(Eigen::Vector2f& pos2d, std::vector<Eigen::Vector3f>& path) {
  // print("TRG checkReplan", param_.isVerbose);
  if (goal_.node == nullptr) {
    return false;
  }

  float dist2subgoal = (goal_.node->pos_.head(2) - pos2d).norm();
  if (!goal_.isKnown && dist2subgoal < param_.goal_tolerance) {
    return true;
  }

  if (!goal_.isKnown && goal_.node->state_ != NodeState::Frontier) {
    return true;
  }

  trgStruct& global_graph = *trgMap_["global"];
  for (auto& pt : path) {
    kdres* res = kd_nearest_range2(global_graph.node_tree, pt.x(), pt.y(), param_.robot_size);
    if (res == nullptr || kd_res_size(res) == 0) {
      if (res != nullptr) {
        kd_res_free(res);
      }
      return true;
    }
    kd_res_free(res);
  }
  return false;
}

bool TRG::planSafePath(Eigen::Vector2f&              start2d,
                       Eigen::Vector3f&              goal_pose,
                       std::vector<Eigen::Vector3f>& out_path,
                       float&                        direct_dist,
                       float&                        path_length,
                       float&                        avg_risk,
                       float                         start_yaw,
                       float                         goal_yaw,
                       bool                          allow_goal_subgoal) {
  // print("TRG planSafePath", param_.isVerbose);
  std::lock_guard<std::mutex> lock(mtx.graph);

  if (param_.path_search_mode == PathSearchMode::TRGAStar) {
    if (planTRGAStarPathLocked(
            start2d,
            goal_pose,
            out_path,
            direct_dist,
            path_length,
            avg_risk,
            start_yaw,
            goal_yaw,
            allow_goal_subgoal)) {
      return true;
    }
    if (!param_.trg_astar.fallback_to_native) {
      return false;
    }
    print_warning("TRG-AStar failed; falling back to native TRG A*", param_.isVerbose);
    out_path.clear();
    direct_dist = 0.0f;
    path_length = 0.0f;
    avg_risk    = 0.0f;
  }

  return planNativeAStarPathLocked(
      start2d, goal_pose, out_path, direct_dist, path_length, avg_risk, allow_goal_subgoal);
}

bool TRG::planNativeAStarPathLocked(Eigen::Vector2f&              start2d,
                                    Eigen::Vector3f&              goal_pose,
                                    std::vector<Eigen::Vector3f>& out_path,
                                    float&                        direct_dist,
                                    float&                        path_length,
                                    float&                        avg_risk,
                                    bool                          allow_goal_subgoal) {
  Eigen::Vector2f goal2d = goal_pose.head(2);
  if (!this->isWithinAllowedArea(start2d)) {
    print_error("Start pose is outside allowed area");
    return false;
  }
  if (!this->isWithinAllowedArea(goal2d)) {
    print_error("Goal pose is outside allowed area");
    return false;
  }
  this->setGoal(goal_pose);

  trgStruct& global_graph = *trgMap_["global"];
  if (global_graph.nodes.empty()) {
    print_error("Cannot plan path: TRG graph is empty");
    return false;
  }

  kdres* res        = kd_nearest2(global_graph.node_tree, start2d.x(), start2d.y());
  if (res == nullptr) {
    print_error("Cannot plan path: TRG node tree is empty");
    return false;
  }
  Node*  start_node = reinterpret_cast<Node*>(kd_res_item_data(res));
  kd_res_free(res);
  if (start_node == nullptr) {
    print_error("Cannot plan path: no TRG node near start pose");
    return false;
  }
  if (!goalNodeCoversRequestedGoal(
          "native TRG",
          goal_.node,
          goal_.isKnown,
          goal2d,
          param_.goal_tolerance,
          allow_goal_subgoal)) {
    return false;
  }

  std::priority_queue<OptimizeNode*,
                      std::vector<OptimizeNode*>,
                      std::function<bool(OptimizeNode*, OptimizeNode*)>>
      open_list([](OptimizeNode* a, OptimizeNode* b) { return a->f_ > b->f_; });
  std::vector<OptimizeNode*> open_check(global_graph.nodes.size(), nullptr);

  direct_dist                = (goal_.node->pos_.head(2) - start_node->pos_.head(2)).norm();
  double        g_cost       = 0.0;
  double        f_cost       = g_cost + direct_dist;
  OptimizeNode* st_opti_node = new OptimizeNode(start_node->id_, f_cost, g_cost);
  st_opti_node->parent_      = nullptr;
  open_list.push(st_opti_node);
  open_check[st_opti_node->id_] = st_opti_node;

  std::vector<OptimizeNode*> close_list(global_graph.nodes.size(), nullptr);

  while (!open_list.empty()) {
    OptimizeNode* opti_node = open_list.top();
    open_list.pop();
    open_check[opti_node->id_] = nullptr;

    if (opti_node->id_ == goal_.node->id_) {
      OptimizeNode* node       = opti_node;
      float         sum_dist   = 0.0;
      float         sum_weight = 0.0;
      float         avg_weight = 0.0;
      while (node != nullptr) {
        const auto node_it = global_graph.nodes.find(node->id_);
        if (node_it == global_graph.nodes.end() || node_it->second == nullptr) {
          return false;
        }
        Node* n = node_it->second;
        for (auto& edge : n->edges_) {
          if (node->parent_ != nullptr && edge->dst_id_ == node->parent_->id_) {
            sum_dist += edge->dist_;
            sum_weight += edge->weight_;
            break;
          }
        }
        out_path.push_back(n->pos_);
        node = node->parent_;
      }
      avg_weight = sum_weight / out_path.size();
      std::reverse(out_path.begin(), out_path.end());
      path_length = sum_dist;
      avg_risk    = avg_weight;
      // print_success("Path found [dist: " + std::to_string(sum_dist) +
      //               ", risk: " + std::to_string(avg_weight) + "]");
      return true;
    }

    const auto curr_node_it = global_graph.nodes.find(opti_node->id_);
    if (curr_node_it == global_graph.nodes.end() || curr_node_it->second == nullptr) {
      continue;
    }
    Node* curr_node            = curr_node_it->second;
    close_list[curr_node->id_] = opti_node;

    for (auto e : curr_node->edges_) {
      const auto dst_node_it = global_graph.nodes.find(e->dst_id_);
      if (dst_node_it == global_graph.nodes.end() || dst_node_it->second == nullptr) {
        continue;
      }
      Node* dst_node = dst_node_it->second;
      if (close_list[dst_node->id_] != nullptr || dst_node->state_ == NodeState::Invalid) {
        continue;
      }

      double next_g_cost = opti_node->g_ + (param_.safety_factor * e->weight_ + 1) * e->dist_;
      double next_f_cost = next_g_cost + (goal_.node->pos_.head(2) - dst_node->pos_.head(2)).norm();

      OptimizeNode* dst_opti_node = new OptimizeNode(dst_node->id_, next_f_cost, next_g_cost);
      dst_opti_node->parent_      = opti_node;

      if (open_check[dst_node->id_] == nullptr) {
        open_list.push(dst_opti_node);
        open_check[dst_node->id_] = dst_opti_node;
      } else if (dst_opti_node->g_ < open_check[dst_node->id_]->g_) {
        open_list.push(dst_opti_node);
        open_check[dst_node->id_] = dst_opti_node;
      }
    }
  }
  return false;
}

bool TRG::planTRGAStarPathLocked(Eigen::Vector2f&              start2d,
                                 Eigen::Vector3f&              goal_pose,
                                 std::vector<Eigen::Vector3f>& out_path,
                                 float&                        direct_dist,
                                 float&                        path_length,
                                 float&                        avg_risk,
                                 float                         start_yaw,
                                 float                         goal_yaw,
                                 bool                          allow_goal_subgoal) {
  Eigen::Vector2f goal2d = goal_pose.head(2);
  if (!this->isWithinAllowedArea(start2d)) {
    print_error("Start pose is outside allowed area");
    return false;
  }
  if (!this->isWithinAllowedArea(goal2d)) {
    print_error("Goal pose is outside allowed area");
    return false;
  }
  this->setGoal(goal_pose);

  trgStruct& global_graph = *trgMap_["global"];
  if (global_graph.nodes.empty()) {
    print_error("Cannot plan path: TRG graph is empty");
    return false;
  }

  kdres* res = kd_nearest2(global_graph.node_tree, start2d.x(), start2d.y());
  if (res == nullptr) {
    print_error("Cannot plan TRG-AStar path: TRG node tree is empty");
    return false;
  }
  Node* start_node = reinterpret_cast<Node*>(kd_res_item_data(res));
  kd_res_free(res);
  if (start_node == nullptr) {
    print_error("Cannot plan TRG-AStar path: no TRG node near start pose");
    return false;
  }
  if (!goalNodeCoversRequestedGoal(
          "TRG-AStar",
          goal_.node,
          goal_.isKnown,
          goal2d,
          param_.goal_tolerance,
          allow_goal_subgoal)) {
    return false;
  }

  const int heading_bins = std::max(4, param_.trg_astar.heading_bins);
  int       max_node_id  = 0;
  for (const auto& node : global_graph.nodes) {
    if (node.second != nullptr) {
      max_node_id = std::max(max_node_id, node.second->id_);
    }
  }
  const std::size_t node_slots = static_cast<std::size_t>(max_node_id + 1);
  const std::size_t state_count = node_slots * static_cast<std::size_t>(heading_bins);
  if (state_count == 0) {
    return false;
  }

  auto state_index = [heading_bins](int node_id, int heading_id) -> std::size_t {
    return static_cast<std::size_t>(node_id) * static_cast<std::size_t>(heading_bins) +
           static_cast<std::size_t>(heading_id);
  };

  struct OpenState {
    float       f;
    float       g;
    int         node_id;
    int         heading_id;
    std::size_t tie;
  };
  auto cmp = [](const OpenState& a, const OpenState& b) {
    if (std::abs(a.f - b.f) > EPS) {
      return a.f > b.f;
    }
    return a.tie > b.tie;
  };

  std::priority_queue<OpenState, std::vector<OpenState>, decltype(cmp)> open_list(cmp);
  std::vector<float> g_cost(state_count, std::numeric_limits<float>::infinity());
  std::vector<int>   parent_node(state_count, -1);
  std::vector<int>   parent_heading(state_count, -1);
  std::vector<char>  closed(state_count, 0);

  const int start_heading = quantizeYawToHeading(start_yaw, heading_bins);
  const int goal_heading  = quantizeYawToHeading(goal_yaw, heading_bins);
  const int start_id      = start_node->id_;
  const int goal_id       = goal_.node->id_;
  const std::size_t start_index = state_index(start_id, start_heading);
  if (start_index >= state_count) {
    return false;
  }

  direct_dist = (goal_.node->pos_.head(2) - start_node->pos_.head(2)).norm();

  std::size_t tie             = 1;
  int         found_node_id   = -1;
  int         found_heading   = -1;
  const float length_scale    = std::max(param_.trg_astar.length_scale_m, kEps);
  const float heuristic_scale = std::max(param_.trg_astar.heuristic_weight, 0.0f) *
                                std::max(param_.trg_astar.length_weight, 0.0f);
  g_cost[start_index] = 0.0f;
  open_list.push(OpenState{
      heuristic_scale * direct_dist / length_scale, 0.0f, start_id, start_heading, 0});

  while (!open_list.empty()) {
    OpenState current = open_list.top();
    open_list.pop();

    if (current.node_id < 0 || current.heading_id < 0 ||
        current.node_id > max_node_id || current.heading_id >= heading_bins) {
      continue;
    }
    const std::size_t curr_index = state_index(current.node_id, current.heading_id);
    if (curr_index >= state_count || closed[curr_index]) {
      continue;
    }
    if (current.g > g_cost[curr_index] + EPS) {
      continue;
    }
    closed[curr_index] = 1;

    if (current.node_id == goal_id) {
      found_node_id = current.node_id;
      found_heading = current.heading_id;
      break;
    }

    const auto curr_node_it = global_graph.nodes.find(current.node_id);
    if (curr_node_it == global_graph.nodes.end() || curr_node_it->second == nullptr) {
      continue;
    }
    const Node* curr_node = curr_node_it->second;

    for (const auto& edge : curr_node->edges_) {
      const auto dst_node_it = global_graph.nodes.find(edge->dst_id_);
      if (dst_node_it == global_graph.nodes.end() || dst_node_it->second == nullptr) {
        continue;
      }
      const Node* dst_node = dst_node_it->second;
      if (dst_node->state_ == NodeState::Invalid) {
        continue;
      }

      const int edge_heading =
          quantizeEdgeHeading(dst_node->pos_.head(2) - curr_node->pos_.head(2), heading_bins);
      const std::size_t dst_index = state_index(dst_node->id_, edge_heading);
      if (dst_index >= state_count || closed[dst_index]) {
        continue;
      }

      const float transition_cost = computeTRGAStarEdgeCost(*curr_node,
                                                            *dst_node,
                                                            *edge,
                                                            current.heading_id,
                                                            edge_heading,
                                                            goal_heading,
                                                            dst_node->id_ == goal_id);
      if (!std::isfinite(transition_cost)) {
        continue;
      }
      const float tentative_g = g_cost[curr_index] + transition_cost;
      if (tentative_g >= g_cost[dst_index]) {
        continue;
      }

      g_cost[dst_index]       = tentative_g;
      parent_node[dst_index]  = current.node_id;
      parent_heading[dst_index] = current.heading_id;

      const float h =
          heuristic_scale * (goal_.node->pos_.head(2) - dst_node->pos_.head(2)).norm() /
          length_scale;
      open_list.push(OpenState{
          tentative_g + h, tentative_g, dst_node->id_, edge_heading, tie++});
    }
  }

  if (found_node_id < 0 || found_heading < 0) {
    print_error("TRG-AStar failed to find a path");
    return false;
  }

  std::vector<int> node_ids;
  int curr_node_id = found_node_id;
  int curr_heading = found_heading;
  while (curr_node_id >= 0 && curr_heading >= 0) {
    const std::size_t curr_index = state_index(curr_node_id, curr_heading);
    if (curr_index >= state_count) {
      return false;
    }
    node_ids.push_back(curr_node_id);
    const int prev_node    = parent_node[curr_index];
    const int prev_heading = parent_heading[curr_index];
    curr_node_id           = prev_node;
    curr_heading           = prev_heading;
  }
  std::reverse(node_ids.begin(), node_ids.end());

  if (node_ids.empty()) {
    return false;
  }

  out_path.clear();
  out_path.reserve(node_ids.size());
  for (const int node_id : node_ids) {
    const auto node_it = global_graph.nodes.find(node_id);
    if (node_it == global_graph.nodes.end() || node_it->second == nullptr) {
      return false;
    }
    out_path.push_back(node_it->second->pos_);
  }

  float sum_dist   = 0.0f;
  float sum_weight = 0.0f;
  int   edge_count = 0;
  for (std::size_t i = 0; i + 1 < node_ids.size(); ++i) {
    const auto src_it = global_graph.nodes.find(node_ids[i]);
    if (src_it == global_graph.nodes.end() || src_it->second == nullptr) {
      return false;
    }
    for (const auto& edge : src_it->second->edges_) {
      if (edge->dst_id_ == node_ids[i + 1]) {
        sum_dist += edge->dist_;
        sum_weight += edge->weight_;
        edge_count++;
        break;
      }
    }
  }

  path_length = sum_dist;
  avg_risk    = edge_count > 0 ? sum_weight / static_cast<float>(edge_count) : 0.0f;
  print_success("TRG-AStar path found [dist: " + std::to_string(path_length) +
                    ", risk: " + std::to_string(avg_risk) + "]",
                param_.isVerbose);
  return true;
}

int TRG::quantizeYawToHeading(float yaw, int heading_bins) const {
  const int   bins      = std::max(1, heading_bins);
  const float bin_width = kTwoPi / static_cast<float>(bins);
  float       wrapped   = std::fmod(yaw, kTwoPi);
  if (wrapped < 0.0f) {
    wrapped += kTwoPi;
  }
  return static_cast<int>(std::floor((wrapped + 0.5f * bin_width) / bin_width)) % bins;
}

int TRG::quantizeEdgeHeading(const Eigen::Vector2f& delta, int heading_bins) const {
  if (delta.norm() <= EPS) {
    return 0;
  }
  return quantizeYawToHeading(std::atan2(delta.y(), delta.x()), heading_bins);
}

float TRG::headingTurnSharpness(int from_heading, int to_heading, int heading_bins) const {
  const int bins = std::max(1, heading_bins);
  int       diff = std::abs(to_heading - from_heading) % bins;
  diff            = std::min(diff, bins - diff);
  const float angle = static_cast<float>(diff) * kTwoPi / static_cast<float>(bins);
  return angle / static_cast<float>(M_PI);
}

float TRG::computeTRGAStarEdgeCost(const Node& node,
                                   const Node& dst_node,
                                   const Edge& edge,
                                   int         prev_heading,
                                   int         edge_heading,
                                   int         goal_heading,
                                   bool        is_goal_transition) const {
  const TRGAStarConfig& cfg  = param_.trg_astar;
  const float           dist = std::max(edge.dist_, kEps);
  const float           dz   = std::abs(dst_node.pos_.z() - node.pos_.z());
  const float           slope_tan = dz / dist;

  if (cfg.max_edge_climb_m > EPS && dz > cfg.max_edge_climb_m) {
    return std::numeric_limits<float>::infinity();
  }
  if (cfg.max_edge_slope_tan > EPS && slope_tan > cfg.max_edge_slope_tan) {
    return std::numeric_limits<float>::infinity();
  }

  const float length_scale = std::max(cfg.length_scale_m, kEps);
  const float climb_scale  = std::max(cfg.climb_scale_m, kEps);
  const float slope_scale  = std::max(cfg.slope_scale_tan, kEps);
  const float turn_scale   = std::max(cfg.turn_scale, kEps);

  const float length_term = dist / length_scale;
  const float risk_term   = std::max(0.0f, edge.weight_);
  const float climb_term  = dz / climb_scale;
  const float slope_term  = slope_tan / slope_scale;
  const float turn_term =
      headingTurnSharpness(prev_heading, edge_heading, cfg.heading_bins) / turn_scale;

  float cost = cfg.length_weight * length_term + cfg.risk_weight * risk_term +
               cfg.climb_weight * climb_term + cfg.slope_weight * slope_term +
               cfg.turn_weight * turn_term;

  if (cfg.footprint_cost_enabled) {
    const FootprintEdgeMetrics footprint_metrics =
        computeFootprintEdgeMetrics(node, dst_node, edge_heading);
    if (cfg.footprint_reject_invalid && footprint_metrics.invalid_ratio > kEps) {
      return std::numeric_limits<float>::infinity();
    }
    cost += cfg.body_height_weight * footprint_metrics.front_back_ratio +
            cfg.body_tilt_weight * footprint_metrics.left_right_tilt_ratio +
            cfg.body_penetration_weight * footprint_metrics.penetration_ratio +
            cfg.body_invalid_weight * footprint_metrics.invalid_ratio;
  }

  if (is_goal_transition) {
    const float goal_turn_term =
        headingTurnSharpness(edge_heading, goal_heading, cfg.heading_bins) / turn_scale;
    cost += cfg.turn_weight * goal_turn_term;
  }
  return std::max(cost, kEps);
}

std::uint64_t TRG::footprintEdgeCacheKey(int src_id, int dst_id, int heading_id) const {
  std::uint64_t key = 1469598103934665603ULL;
  auto mix = [&key](std::uint64_t value) {
    key ^= value;
    key *= 1099511628211ULL;
  };
  mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(src_id)));
  mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(dst_id)));
  mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(std::max(0, heading_id))));
  return key;
}

bool TRG::sampleFootprintPoseMetrics(const Eigen::Vector2f& center,
                                     const Eigen::Vector2f& heading_dir,
                                     FootprintPoseMetrics&  metrics) const {
  const TRGAStarConfig& cfg = param_.trg_astar;
  metrics                  = FootprintPoseMetrics{};

  if (global_trg_.map_tree == nullptr || heading_dir.norm() <= EPS) {
    return false;
  }

  Eigen::Vector2f heading = heading_dir.normalized();
  Eigen::Vector2f side(-heading.y(), heading.x());

  const float front_span = std::max(cfg.robot_length_m, cfg.robot_width_m);
  const float side_span  = std::min(cfg.robot_length_m, cfg.robot_width_m);
  const float half_front = 0.5f * front_span;
  const float half_side  = 0.5f * side_span;
  const float edge_band  = std::max(cfg.footprint_edge_band_m, 0.01f);
  const float radius     = std::hypot(half_front, half_side) + edge_band;

  kdres* res = kd_nearest_range2(global_trg_.map_tree, center.x(), center.y(), radius);
  if (res == nullptr || kd_res_size(res) == 0) {
    if (res != nullptr) {
      kd_res_free(res);
    }
    return false;
  }

  float front_sum = 0.0f;
  float rear_sum  = 0.0f;
  float left_sum  = 0.0f;
  float right_sum = 0.0f;
  float fl_sum    = 0.0f;
  float fr_sum    = 0.0f;
  float rl_sum    = 0.0f;
  float rr_sum    = 0.0f;
  int   front_count = 0;
  int   rear_count  = 0;
  int   left_count  = 0;
  int   right_count = 0;
  int   fl_count    = 0;
  int   fr_count    = 0;
  int   rl_count    = 0;
  int   rr_count    = 0;
  std::vector<float> interior_heights;

  while (!kd_res_end(res)) {
    PtsDefault* pt = reinterpret_cast<PtsDefault*>(kd_res_item_data(res));
    if (pt == nullptr) {
      kd_res_next(res);
      continue;
    }

    const Eigen::Vector2f offset(pt->x - center.x(), pt->y - center.y());
    const float           local_front = offset.dot(heading);
    const float           local_side  = offset.dot(side);
    if (std::abs(local_front) > half_front + edge_band ||
        std::abs(local_side) > half_side + edge_band) {
      kd_res_next(res);
      continue;
    }

    interior_heights.push_back(pt->z);
    const bool is_front = local_front >= half_front - edge_band;
    const bool is_rear  = local_front <= -half_front + edge_band;
    const bool is_left  = local_side >= half_side - edge_band;
    const bool is_right = local_side <= -half_side + edge_band;

    if (is_front) {
      front_sum += pt->z;
      front_count++;
    }
    if (is_rear) {
      rear_sum += pt->z;
      rear_count++;
    }
    if (is_left) {
      left_sum += pt->z;
      left_count++;
    }
    if (is_right) {
      right_sum += pt->z;
      right_count++;
    }
    if (is_front && is_left) {
      fl_sum += pt->z;
      fl_count++;
    }
    if (is_front && is_right) {
      fr_sum += pt->z;
      fr_count++;
    }
    if (is_rear && is_left) {
      rl_sum += pt->z;
      rl_count++;
    }
    if (is_rear && is_right) {
      rr_sum += pt->z;
      rr_count++;
    }
    kd_res_next(res);
  }
  kd_res_free(res);

  const bool has_directional_support = !interior_heights.empty() && front_count > 0 &&
                                       rear_count > 0 && left_count > 0 && right_count > 0 &&
                                       fl_count > 0 && fr_count > 0 && rl_count > 0 &&
                                       rr_count > 0;
  if (!has_directional_support) {
    metrics.front_back_diff_m = 2.0f * cfg.max_body_height_diff_m;
    metrics.left_right_tilt_deg = 2.0f * cfg.max_body_tilt_deg;
    metrics.max_interior_penetration_m = 2.0f * cfg.max_interior_penetration_m;
    metrics.feasible = false;
    return false;
  }

  const float front_mean = front_sum / static_cast<float>(front_count);
  const float rear_mean  = rear_sum / static_cast<float>(rear_count);
  const float left_mean  = left_sum / static_cast<float>(left_count);
  const float right_mean = right_sum / static_cast<float>(right_count);
  const float fl_mean    = fl_sum / static_cast<float>(fl_count);
  const float fr_mean    = fr_sum / static_cast<float>(fr_count);
  const float rl_mean    = rl_sum / static_cast<float>(rl_count);
  const float rr_mean    = rr_sum / static_cast<float>(rr_count);

  const float front_pair_mean = 0.5f * (fl_mean + fr_mean);
  const float rear_pair_mean  = 0.5f * (rl_mean + rr_mean);
  metrics.front_back_diff_m   = std::abs(front_pair_mean - rear_pair_mean);

  const float lr_tan = std::abs(left_mean - right_mean) / std::max(side_span, kEps);
  metrics.left_right_tilt_deg = std::atan(lr_tan) * 180.0f / static_cast<float>(M_PI);

  const float max_corner_height =
      std::max(std::max(fl_mean, fr_mean), std::max(rl_mean, rr_mean));
  metrics.max_interior_penetration_m = 0.0f;
  for (const float z : interior_heights) {
    metrics.max_interior_penetration_m =
        std::max(metrics.max_interior_penetration_m, std::max(0.0f, z - max_corner_height));
  }

  metrics.feasible =
      metrics.front_back_diff_m <= cfg.max_body_height_diff_m + kEps &&
      metrics.left_right_tilt_deg <= cfg.max_body_tilt_deg + kEps &&
      metrics.max_interior_penetration_m <= cfg.max_interior_penetration_m + kEps;
  return true;
}

TRG::FootprintEdgeMetrics TRG::computeFootprintEdgeMetrics(const Node& node,
                                                           const Node& dst_node,
                                                           int         edge_heading) const {
  const std::uint64_t cache_key = footprintEdgeCacheKey(node.id_, dst_node.id_, edge_heading);
  const auto          cache_it  = footprint_edge_metric_cache_.find(cache_key);
  if (cache_it != footprint_edge_metric_cache_.end()) {
    return cache_it->second;
  }

  const TRGAStarConfig& cfg = param_.trg_astar;
  FootprintEdgeMetrics  metrics;
  const Eigen::Vector2f delta = dst_node.pos_.head(2) - node.pos_.head(2);
  const float           dist  = delta.norm();
  if (dist <= EPS) {
    metrics.invalid_ratio = 1.0f;
    metrics.front_back_ratio = 2.0f;
    metrics.left_right_tilt_ratio = 2.0f;
    metrics.penetration_ratio = 2.0f;
    footprint_edge_metric_cache_[cache_key] = metrics;
    return metrics;
  }

  const Eigen::Vector2f heading_dir = delta / dist;
  const float sample_step = cfg.footprint_sample_step_m > EPS
                                ? cfg.footprint_sample_step_m
                                : std::max(0.05f, 0.25f * param_.robot_size);
  const int sample_count = std::max(1, static_cast<int>(std::ceil(dist / sample_step)) + 1);

  int   invalid_count   = 0;
  float front_back_sum  = 0.0f;
  float tilt_sum        = 0.0f;
  float penetration_sum = 0.0f;
  for (int i = 0; i < sample_count; ++i) {
    const float ratio = sample_count == 1 ? 0.0f
                                          : static_cast<float>(i) /
                                                static_cast<float>(sample_count - 1);
    const Eigen::Vector2f center = node.pos_.head(2) + ratio * delta;
    FootprintPoseMetrics  pose_metrics;
    const bool            has_support =
        sampleFootprintPoseMetrics(center, heading_dir, pose_metrics);
    if (!has_support) {
      invalid_count++;
      front_back_sum += 2.0f;
      tilt_sum += 2.0f;
      penetration_sum += 2.0f;
      continue;
    }
    if (!pose_metrics.feasible) {
      invalid_count++;
    }
    front_back_sum += clampRatio(pose_metrics.front_back_diff_m, cfg.max_body_height_diff_m);
    tilt_sum += clampRatio(pose_metrics.left_right_tilt_deg, cfg.max_body_tilt_deg);
    penetration_sum +=
        clampRatio(pose_metrics.max_interior_penetration_m, cfg.max_interior_penetration_m);
  }

  const float denom = static_cast<float>(sample_count);
  metrics.invalid_ratio = static_cast<float>(invalid_count) / denom;
  metrics.front_back_ratio = front_back_sum / denom;
  metrics.left_right_tilt_ratio = tilt_sum / denom;
  metrics.penetration_ratio = penetration_sum / denom;
  footprint_edge_metric_cache_[cache_key] = metrics;
  return metrics;
}

void TRG::refinePath(std::vector<Eigen::Vector3f>& in_path,
                     std::vector<Eigen::Vector3f>& out_path) {
  if (in_path.size() < 2) {
    out_path = in_path;
    return;
  }

  std::deque<Eigen::Vector3f> dense_path;
  int                         point_between = 1;
  for (int i = 0; i < in_path.size() - 1; ++i) {
    Eigen::Vector3f p1   = in_path[i];
    Eigen::Vector3f p2   = in_path[i + 1];
    Eigen::Vector3f dir  = (p2.head(2) - p1.head(2)).normalized();
    float           dist = (p2.head(2) - p1.head(2)).norm();
    dense_path.push_back(p1);
    for (int j = 1; j < point_between; ++j) {
      Eigen::Vector3f p;
      p.head(2) = p1.head(2) + j * dist / point_between * dir;
      p.z()     = p1.z() + j * (p2.z() - p1.z()) / point_between;
      dense_path.push_back(p);
    }
    dense_path.push_back(p2);
  }

  std::deque<Eigen::Vector3f> smooth_path;
  for (int i = 0; i < dense_path.size(); ++i) {
    if (i == dense_path.size() - 1) {
      smooth_path.push_back(dense_path[i]);
      break;
    }
    Eigen::Vector3f sum(0.0, 0.0, 0.0);
    int             cnt = 0;
    for (int j = i - 1; j < i + 2; ++j) {
      if (j < 0 || j >= dense_path.size()) {
        continue;
      }
      sum += dense_path[j];
      cnt++;
    }
    Eigen::Vector3f avg = sum / cnt;
    smooth_path.push_back(avg);
  }
  out_path = std::vector<Eigen::Vector3f>(smooth_path.begin(), smooth_path.end());
}

void TRG::resetGraph(std::string type) {
  trgStruct& graph = *trgMap_[type];
  footprint_edge_metric_cache_.clear();
  graph.nodes.clear();
  kd_clear(graph.node_tree);
  graph.node_id = 0;
}

void TRG::resetMap(std::string type) {
  trgStruct& graph = *trgMap_[type];
  footprint_edge_metric_cache_.clear();
  kd_clear(graph.map_tree);
  graph.cloud_map.reset(new pcl::PointCloud<PtsDefault>());
  graph.cloud_map->clear();
}

void TRG::setAllowedArea(const std::vector<Eigen::Vector2f>& polygon,
                         float                               keepout_margin,
                         bool                                enabled) {
  std::lock_guard<std::mutex> lock(allowed_area_mutex_);
  allowed_area_polygon_        = polygon;
  allowed_area_keepout_margin_ = std::max(0.0f, keepout_margin);
  allowed_area_enabled_        = enabled && allowed_area_polygon_.size() >= 3;

  if (allowed_area_enabled_) {
    print("Allowed area enabled with " + std::to_string(allowed_area_polygon_.size()) +
          " points and keepout margin " + std::to_string(allowed_area_keepout_margin_) + " m");
  } else {
    print("Allowed area disabled", param_.isVerbose);
  }
}

bool TRG::isWithinAllowedArea(const Eigen::Vector2f& pos) const {
  std::lock_guard<std::mutex> lock(allowed_area_mutex_);
  if (!allowed_area_enabled_) {
    return true;
  }
  if (!isInsideAllowedAreaPolygon(pos)) {
    return false;
  }
  if (allowed_area_keepout_margin_ > 0.0f &&
      distanceToAllowedAreaBoundary(pos) < allowed_area_keepout_margin_) {
    return false;
  }
  return true;
}

void TRG::pruneOutsideAllowedArea() {
  std::lock_guard<std::mutex> lock(mtx.graph);
  trgStruct&                  graph = *trgMap_["global"];
  bool                        changed = false;

  for (auto& node : graph.nodes) {
    if (node.second == nullptr || node.second->state_ == NodeState::Invalid) {
      continue;
    }
    Eigen::Vector2f pos2d = node.second->pos_.head(2);
    if (!this->isWithinAllowedArea(pos2d)) {
      node.second->state_ = NodeState::Invalid;
      changed             = true;
    }
  }

  const float step = std::max(0.05f, param_.robot_size * 0.5f);
  for (auto& node : graph.nodes) {
    if (node.second == nullptr || node.second->state_ == NodeState::Invalid) {
      continue;
    }
    std::vector<Edge*> kept_edges;
    for (Edge* edge : node.second->edges_) {
      const auto dst_it = graph.nodes.find(edge->dst_id_);
      if (dst_it == graph.nodes.end() || dst_it->second == nullptr ||
          dst_it->second->state_ == NodeState::Invalid) {
        changed = true;
        continue;
      }

      const Eigen::Vector2f start = node.second->pos_.head(2);
      const Eigen::Vector2f end   = dst_it->second->pos_.head(2);
      const Eigen::Vector2f delta = end - start;
      const float           dist  = delta.norm();
      bool                  edge_inside = true;
      if (dist > EPS) {
        const Eigen::Vector2f dir = delta / dist;
        for (float offset = 0.0f; offset <= dist; offset += step) {
          Eigen::Vector2f sample = start + offset * dir;
          if (!this->isWithinAllowedArea(sample)) {
            edge_inside = false;
            break;
          }
        }
      }
      if (edge_inside && this->isWithinAllowedArea(end)) {
        kept_edges.push_back(edge);
      } else {
        changed = true;
      }
    }
    node.second->edges_ = kept_edges;
  }

  if (changed) {
    this->cleanGraph(true);
  }
}

bool TRG::isInsideAllowedAreaPolygon(const Eigen::Vector2f& pos) const {
  bool inside = false;
  for (std::size_t i = 0, j = allowed_area_polygon_.size() - 1;
       i < allowed_area_polygon_.size();
       j = i++) {
    const Eigen::Vector2f& pi = allowed_area_polygon_[i];
    const Eigen::Vector2f& pj = allowed_area_polygon_[j];
    const bool crosses =
        ((pi.y() > pos.y()) != (pj.y() > pos.y())) &&
        (pos.x() < (pj.x() - pi.x()) * (pos.y() - pi.y()) / (pj.y() - pi.y() + EPS) + pi.x());
    if (crosses) {
      inside = !inside;
    }
  }
  return inside;
}

float TRG::distanceToAllowedAreaBoundary(const Eigen::Vector2f& pos) const {
  if (allowed_area_polygon_.size() < 2) {
    return std::numeric_limits<float>::infinity();
  }

  float min_dist = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < allowed_area_polygon_.size(); ++i) {
    const Eigen::Vector2f& a  = allowed_area_polygon_[i];
    const Eigen::Vector2f& b  = allowed_area_polygon_[(i + 1) % allowed_area_polygon_.size()];
    const Eigen::Vector2f  ab = b - a;
    const float            denom = ab.squaredNorm();
    float                  t = 0.0f;
    if (denom > EPS) {
      t = std::max(0.0f, std::min(1.0f, (pos - a).dot(ab) / denom));
    }
    const Eigen::Vector2f projection = a + t * ab;
    min_dist = std::min(min_dist, (pos - projection).norm());
  }
  return min_dist;
}

bool TRG::isCollision(Eigen::Vector2f& pos, std::string type, float threshold = 0.1) {
  if (!this->isWithinAllowedArea(pos)) {
    return true;
  }

  trgStruct& graph = *trgMap_[type];
  kdres*     res   = kd_nearest_range2(graph.map_tree, pos.x(), pos.y(), param_.robot_size);
  if (res == nullptr || kd_res_size(res) == 0) {
    if (res != nullptr) {
      kd_res_free(res);
    }
    return true;
  }

  std::vector<PtsDefault*> pts;
  float                    z_med = 0.0;
  while (!kd_res_end(res)) {
    PtsDefault* pt = reinterpret_cast<PtsDefault*>(kd_res_item_data(res));
    if (pt != nullptr) {
      pts.push_back(pt);
    }
    kd_res_next(res);
  }
  kd_res_free(res);
  if (pts.empty()) {
    return true;
  }

  std::sort(pts.begin(), pts.end(), [](PtsDefault* a, PtsDefault* b) { return a->z < b->z; });
  z_med = pts[pts.size() / 2]->z;

  int total = pts.size();
  int cnt   = 0;
  for (auto& pt : pts) {
    if (fabs(pt->z - z_med) > param_.height_threshold) {
      cnt++;
    }
  }
  float ratio = static_cast<float>(cnt) / total;
  if (ratio > threshold) {
    return true;
  }
  return false;
}

bool TRG::isFrontier(Eigen::Vector2f& pos) {
  trgStruct&      global_graph = *trgMap_["global"];
  trgStruct&      local_graph  = *trgMap_["local"];
  Eigen::Vector2f dir          = pos - local_graph.root_pos;
  if (dir.norm() < EPS) {
    return false;
  }
  dir.normalize();

  Eigen::Vector2f check = pos + 2 * param_.robot_size * dir;

  kdres* res2 = kd_nearest_range2(global_graph.node_tree, check.x(), check.y(), param_.robot_size);
  if (res2 != nullptr && kd_res_size(res2) > 0) {
    kd_res_free(res2);
    return false;
  }
  if (res2 != nullptr) {
    kd_res_free(res2);
  }

  kdres* res1 =
      kd_nearest_range2(local_graph.map_tree, check.x(), check.y(), 0.5 * param_.robot_size);
  if (res1 == nullptr || kd_res_size(res1) == 0) {
    if (res1 != nullptr) {
      kd_res_free(res1);
    }
    return true;
  }
  kd_res_free(res1);
  return false;
}

std::unordered_map<int, TRG::Node*> TRG::getGraph(std::string type = "global") {
  trgStruct& graph = *trgMap_[type];
  return graph.nodes;
}

std::unordered_map<int, TRG::Node*> TRG::getGraphCopy(std::string type = "global") {
  std::lock_guard<std::mutex>         lock(mtx.graph);
  std::unordered_map<int, TRG::Node*> nodes_copy;
  trgStruct&                          graph = *trgMap_[type];
  for (auto& node : graph.nodes) {
    Eigen::Vector2f pos = node.second->pos_.head(2);
    Node* node_copy = new Node(node.second->id_, pos, node.second->pos_.z(), node.second->state_);
    for (auto& edge : node.second->edges_) {
      Edge* edge_copy = new Edge(edge->dst_id_, edge->weight_, edge->dist_);
      node_copy->edges_.push_back(edge_copy);
    }
    nodes_copy[node.first] = node_copy;
  }
  return nodes_copy;
}

void TRG::lockGraph() { mtx.graph.lock(); }

void TRG::unlockGraph() { mtx.graph.unlock(); }
