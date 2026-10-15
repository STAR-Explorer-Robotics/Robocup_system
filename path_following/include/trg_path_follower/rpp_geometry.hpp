#pragma once

#include <algorithm>
#include <cmath>

namespace trg_path_follower
{

struct Point2D
{
  double x{0.0};
  double y{0.0};
};

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct SegmentProjection
{
  Point2D point;
  Point2D tangent{1.0, 0.0};
  Point2D normal{0.0, 1.0};
  double ratio{0.0};
  double signed_error{0.0};
  double distance{0.0};
};

inline double clamp(const double value, const double lower, const double upper)
{
  return std::max(lower, std::min(value, upper));
}

inline double normalize_angle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

inline double distance2d(const Point2D & a, const Point2D & b)
{
  return std::hypot(a.x - b.x, a.y - b.y);
}

inline Point2D transform_to_body(const Pose2D & robot, const Point2D & point_world)
{
  const double dx = point_world.x - robot.x;
  const double dy = point_world.y - robot.y;
  const double c = std::cos(robot.yaw);
  const double s = std::sin(robot.yaw);
  return Point2D{c * dx + s * dy, -s * dx + c * dy};
}

inline Point2D transform_vector_to_body(const Pose2D & robot, const Point2D & vector_world)
{
  const double c = std::cos(robot.yaw);
  const double s = std::sin(robot.yaw);
  return Point2D{
    c * vector_world.x + s * vector_world.y,
    -s * vector_world.x + c * vector_world.y};
}

inline SegmentProjection project_point_to_segment(
  const Point2D & point,
  const Point2D & start,
  const Point2D & end)
{
  const double dx = end.x - start.x;
  const double dy = end.y - start.y;
  const double length_sq = dx * dx + dy * dy;
  if (length_sq < 1e-12) {
    const double rx = point.x - start.x;
    const double ry = point.y - start.y;
    return SegmentProjection{
      start,
      Point2D{1.0, 0.0},
      Point2D{0.0, 1.0},
      0.0,
      ry,
      std::hypot(rx, ry)};
  }

  const double ratio = clamp(
    ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq,
    0.0,
    1.0);
  const Point2D projection{start.x + ratio * dx, start.y + ratio * dy};
  const double length = std::sqrt(length_sq);
  const Point2D tangent{dx / length, dy / length};
  const Point2D normal{-tangent.y, tangent.x};
  const double rx = point.x - projection.x;
  const double ry = point.y - projection.y;
  const double signed_error = tangent.x * ry - tangent.y * rx;

  return SegmentProjection{
    projection,
    tangent,
    normal,
    ratio,
    signed_error,
    std::hypot(rx, ry)};
}

inline double pure_pursuit_curvature(const Point2D & target_body)
{
  const double l2 = target_body.x * target_body.x + target_body.y * target_body.y;
  if (l2 < 1e-6) {
    return 0.0;
  }
  return 2.0 * target_body.y / l2;
}

inline double limit_rate(
  const double target,
  const double previous,
  const double max_delta_per_second,
  const double dt)
{
  if (max_delta_per_second <= 0.0 || dt <= 0.0) {
    return target;
  }
  const double max_delta = max_delta_per_second * dt;
  return previous + clamp(target - previous, -max_delta, max_delta);
}

}  // namespace trg_path_follower
