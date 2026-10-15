/**
 * Copyright 2025, Korea Advanced Institute of Science and Technology
 * Massachusetts Institute of Technology,
 * Daejeon, 34051
 * All Rights Reserved
 * Authors: Dongkyu Lee, et al.
 * See LICENSE for the license information
 */
#ifndef CPP_TRG_PLANNER_CORE_TRG_PLANNER_INCLUDE_GRAPH_TRG_H_
#define CPP_TRG_PLANNER_CORE_TRG_PLANNER_INCLUDE_GRAPH_TRG_H_

#include <cstdint>

#include "trg_planner/include/kdtree/kdtree.h"
#include "trg_planner/include/utils/common.h"

#define EPS 1e-6
const Eigen::Vector3f gravity(0, 0, -1);

class TRG {
 public:
  struct Edge {
    Edge(int dst_id, float weight, float dist) : dst_id_(dst_id), weight_(weight), dist_(dist) {}
    int   dst_id_;
    float weight_;
    float dist_;
  };

  enum struct NodeState {
    Valid    = 0,
    Invalid  = -1,
    Frontier = 1,
  };

  enum struct PathSearchMode {
    Native   = 0,
    TRGAStar = 1,
  };

  enum struct SamplingMode {
    RandomRing        = 0,
    DeterministicRing = 1,
    Lattice           = 2,
  };

  struct Node {
    Node(int id, Eigen::Vector2f& pos2d, float z, NodeState state)
        : id_(id), pos_(Eigen::Vector3f(pos2d.x(), pos2d.y(), z)), state_(state) {}
    int                id_;
    Eigen::Vector3f    pos_;
    NodeState          state_;
    std::vector<Edge*> edges_;
  };

  struct OptimizeNode {
    OptimizeNode(int i, float f, float g) : id_(i), f_(f), g_(g) {}
    int           id_;
    OptimizeNode* parent_;
    float         f_;
    float         g_;
  };

  struct TRGAStarConfig {
    bool  fallback_to_native = true;
    int   heading_bins       = 8;
    float length_weight      = 1.0f;
    float risk_weight        = 3.0f;
    float climb_weight       = 0.35f;
    float slope_weight       = 0.35f;
    float turn_weight        = 0.2f;
    float heuristic_weight   = 1.0f;
    float length_scale_m     = 0.0f;
    float climb_scale_m      = 0.3f;
    float slope_scale_tan    = 0.7f;
    float turn_scale         = 0.25f;
    float max_edge_climb_m   = 0.0f;
    float max_edge_slope_tan = 0.0f;

    bool  footprint_cost_enabled     = false;
    bool  footprint_reject_invalid   = false;
    float robot_length_m             = 0.70f;
    float robot_width_m              = 0.43f;
    float max_body_height_diff_m     = 0.70f;
    float max_body_tilt_deg          = 35.0f;
    float max_interior_penetration_m = 0.30f;
    float body_height_weight         = 0.35f;
    float body_tilt_weight           = 0.35f;
    float body_penetration_weight    = 0.25f;
    float body_invalid_weight        = 4.0f;
    float footprint_sample_step_m    = 0.05f;
    float footprint_edge_band_m      = 0.06f;
  };

 public:
  TRG(bool  isVerbose,
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
      bool  lattice_fallback_ring);
  virtual ~TRG() = default;

  bool initGraph(bool isPreMap, Eigen::Vector3f start3d);
  void setPathSearchConfig(PathSearchMode mode, const TRGAStarConfig& config);

  void loadPrebuiltGraph();

  void setGlobalMap(PointCloudPtr& map);
  void setLocalMap(Eigen::Vector2f start2d, PointCloudPtr& map);
  void setLocalGraph(bool useMutex);

  bool addNode(int node_id, Eigen::Vector2f& node_pos, NodeState state, std::string type);
  void wireEdge(Node* node1, Node* node2, std::string type);

  void expandGraph(int node_id, std::string type);
  bool rebuildLatticeGraph(Eigen::Vector2f root2d, bool updateLocal);
  void cleanGraph(bool updateLocal);
  void updateGraph();

  void setGoal(Eigen::Vector3f& goal);
  bool checkReadched(Eigen::Vector2f& pos2d);
  bool checkReplan(Eigen::Vector2f& pos2d, std::vector<Eigen::Vector3f>& path);

  bool planSafePath(Eigen::Vector2f&              start2d,
                    Eigen::Vector3f&              goal_pose,
                    std::vector<Eigen::Vector3f>& out_path,
                    float&                        direct_dist,
                    float&                        path_length,
                    float&                        avg_risk,
                    float                         start_yaw = 0.0f,
                    float                         goal_yaw = 0.0f,
                    bool                          allow_goal_subgoal = false);
  void refinePath(std::vector<Eigen::Vector3f>& in_path, std::vector<Eigen::Vector3f>& out_path);

  void setAllowedArea(const std::vector<Eigen::Vector2f>& polygon,
                      float                               keepout_margin,
                      bool                                enabled);
  bool isWithinAllowedArea(const Eigen::Vector2f& pos2d) const;
  void pruneOutsideAllowedArea();

  void resetGraph(std::string type);
  void resetMap(std::string type);

  bool isCollision(Eigen::Vector2f& pos2d, std::string type, float threshold);
  bool isFrontier(Eigen::Vector2f& pos2d);

  std::unordered_map<int, Node*> getGraph(std::string type);
  std::unordered_map<int, Node*> getGraphCopy(std::string type);
  void                           lockGraph();
  void                           unlockGraph();

 protected:
  struct trgStruct {
    explicit trgStruct(std::string type) : type(type), node_id(0) {}
    std::string type;

    std::unordered_map<int, Node*> nodes;
    kdtree*                        node_tree = kd_create(2);
    int                            node_id;
    Eigen::Vector2f                root_pos = Eigen::Vector2f::Zero();

    kdtree*       map_tree  = kd_create(2);
    PointCloudPtr cloud_map = nullptr;
  };
  trgStruct prebuilt_trg_ = trgStruct("prebuilt");
  trgStruct global_trg_   = trgStruct("global");
  trgStruct local_trg_    = trgStruct("local");

  std::unordered_map<std::string, trgStruct*> trgMap_ = {{"prebuilt", &prebuilt_trg_},
                                                         {"global", &global_trg_},
                                                         {"local", &local_trg_}};

  struct goalStruct {
    Eigen::Vector3f pose3d;
    Eigen::Vector2f pose2d;
    Node*           node    = nullptr;
    bool            isKnown = false;
  } goal_;

  std::random_device                    rd_;
  std::mt19937                          gen_;
  std::uniform_real_distribution<float> distr_;

  bool                         allowed_area_enabled_ = false;
  float                        allowed_area_keepout_margin_ = 0.0f;
  std::vector<Eigen::Vector2f> allowed_area_polygon_;
  mutable std::mutex           allowed_area_mutex_;

  bool  isInsideAllowedAreaPolygon(const Eigen::Vector2f& pos2d) const;
  float distanceToAllowedAreaBoundary(const Eigen::Vector2f& pos2d) const;

  struct Param {
    bool  isVerbose                  = true;
    float expand_dist                = 0.5;
    float robot_size                 = 0.5;
    int   sample_num                 = 20;
    int   random_seed                = -1;
    bool  deterministic_sampling     = false;
    SamplingMode sampling_mode       = SamplingMode::RandomRing;
    float lattice_spacing            = 0.4;
    float lattice_refinement_spacing = 0.22;
    float lattice_refinement_margin  = 0.45;
    float lattice_connect_radius     = 0.7;
    float lattice_min_node_spacing   = 0.22;
    bool  lattice_refinement_enabled = true;
    bool  lattice_fallback_ring      = true;
    float height_threshold           = 0.5;
    float collision_threshold        = 0.5;
    float update_collision_threshold = 0.5;
    float safety_factor              = 3.0;
    float goal_tolerance             = 0.2;
    PathSearchMode path_search_mode  = PathSearchMode::Native;
    TRGAStarConfig trg_astar;
  } param_;

  bool planNativeAStarPathLocked(Eigen::Vector2f&              start2d,
                                 Eigen::Vector3f&              goal_pose,
                                 std::vector<Eigen::Vector3f>& out_path,
                                 float&                        direct_dist,
                                 float&                        path_length,
                                 float&                        avg_risk,
                                 bool                          allow_goal_subgoal);
  bool planTRGAStarPathLocked(Eigen::Vector2f&              start2d,
                              Eigen::Vector3f&              goal_pose,
                              std::vector<Eigen::Vector3f>& out_path,
                              float&                        direct_dist,
                              float&                        path_length,
                              float&                        avg_risk,
                              float                         start_yaw,
                              float                         goal_yaw,
                              bool                          allow_goal_subgoal);
  int   quantizeYawToHeading(float yaw, int heading_bins) const;
  int   quantizeEdgeHeading(const Eigen::Vector2f& delta, int heading_bins) const;
  float headingTurnSharpness(int from_heading, int to_heading, int heading_bins) const;
  float computeTRGAStarEdgeCost(const Node& node,
                                const Node& dst_node,
                                const Edge& edge,
                                int         prev_heading,
                                int         edge_heading,
                                int         goal_heading,
                                bool        is_goal_transition) const;
  struct FootprintPoseMetrics {
    bool  feasible                   = false;
    float front_back_diff_m          = 0.0f;
    float left_right_tilt_deg        = 0.0f;
    float max_interior_penetration_m = 0.0f;
  };
  struct FootprintEdgeMetrics {
    float invalid_ratio          = 0.0f;
    float front_back_ratio       = 0.0f;
    float left_right_tilt_ratio  = 0.0f;
    float penetration_ratio      = 0.0f;
  };
  bool sampleFootprintPoseMetrics(const Eigen::Vector2f& center,
                                  const Eigen::Vector2f& heading_dir,
                                  FootprintPoseMetrics&  metrics) const;
  FootprintEdgeMetrics computeFootprintEdgeMetrics(const Node& node,
                                                   const Node& dst_node,
                                                   int         edge_heading) const;
  std::uint64_t footprintEdgeCacheKey(int src_id, int dst_id, int heading_id) const;

  mutable std::unordered_map<std::uint64_t, FootprintEdgeMetrics> footprint_edge_metric_cache_;

  bool addSeedNodeNearRoot(const Eigen::Vector2f& root2d, std::string type, int& out_node_id);
  bool tryAddLatticeNode(const Eigen::Vector2f& pos,
                         TRG::NodeState        state,
                         std::string           type,
                         float                 min_spacing);
  void addLatticeCandidates(std::string type,
                            float       min_x,
                            float       max_x,
                            float       min_y,
                            float       max_y,
                            float       spacing,
                            bool        boundary_only);
  bool isNearCollisionBoundary(const Eigen::Vector2f& pos, std::string type, float probe_radius);
  void wireLatticeGraph(std::string type);
  int  pruneToReachableComponent(const Eigen::Vector2f& root2d, std::string type);
  void classifyLatticeFrontiers();

  struct Mutex {
    std::mutex graph;
  } mtx;
};

#endif  // CPP_TRG_PLANNER_CORE_TRG_PLANNER_INCLUDE_GRAPH_TRG_H_
