#include "trg_path_follower/rpp_geometry.hpp"

#include <cmath>

#include "gtest/gtest.h"

namespace trg_path_follower
{
namespace
{

TEST(RppGeometry, NormalizesAngles)
{
  EXPECT_NEAR(normalize_angle(3.0 * M_PI), M_PI, 1e-9);
  EXPECT_NEAR(normalize_angle(-3.0 * M_PI), -M_PI, 1e-9);
  EXPECT_NEAR(normalize_angle(0.25), 0.25, 1e-9);
}

TEST(RppGeometry, TransformsWorldPointToBodyFrame)
{
  const Pose2D robot{1.0, 2.0, M_PI_2};
  const Point2D world{1.0, 3.0};
  const Point2D body = transform_to_body(robot, world);

  EXPECT_NEAR(body.x, 1.0, 1e-9);
  EXPECT_NEAR(body.y, 0.0, 1e-9);
}

TEST(RppGeometry, TransformsWorldVectorToBodyFrame)
{
  const Pose2D robot{1.0, 2.0, M_PI_2};
  const Point2D world_vector{0.0, 1.0};
  const Point2D body = transform_vector_to_body(robot, world_vector);

  EXPECT_NEAR(body.x, 1.0, 1e-9);
  EXPECT_NEAR(body.y, 0.0, 1e-9);
}

TEST(RppGeometry, ProjectsPointToSegmentWithSignedError)
{
  const SegmentProjection projection = project_point_to_segment(
    Point2D{0.5, 0.2},
    Point2D{0.0, 0.0},
    Point2D{1.0, 0.0});

  EXPECT_NEAR(projection.point.x, 0.5, 1e-9);
  EXPECT_NEAR(projection.point.y, 0.0, 1e-9);
  EXPECT_NEAR(projection.ratio, 0.5, 1e-9);
  EXPECT_NEAR(projection.tangent.x, 1.0, 1e-9);
  EXPECT_NEAR(projection.tangent.y, 0.0, 1e-9);
  EXPECT_NEAR(projection.normal.x, 0.0, 1e-9);
  EXPECT_NEAR(projection.normal.y, 1.0, 1e-9);
  EXPECT_NEAR(projection.signed_error, 0.2, 1e-9);
  EXPECT_NEAR(projection.distance, 0.2, 1e-9);
}

TEST(RppGeometry, ComputesPurePursuitCurvature)
{
  EXPECT_NEAR(pure_pursuit_curvature(Point2D{1.0, 0.0}), 0.0, 1e-9);
  EXPECT_NEAR(pure_pursuit_curvature(Point2D{1.0, 1.0}), 1.0, 1e-9);
  EXPECT_NEAR(pure_pursuit_curvature(Point2D{1.0, -1.0}), -1.0, 1e-9);
}

TEST(RppGeometry, LimitsRate)
{
  EXPECT_NEAR(limit_rate(1.0, 0.0, 2.0, 0.1), 0.2, 1e-9);
  EXPECT_NEAR(limit_rate(-1.0, 0.0, 2.0, 0.1), -0.2, 1e-9);
  EXPECT_NEAR(limit_rate(0.1, 0.0, 2.0, 0.1), 0.1, 1e-9);
}

}  // namespace
}  // namespace trg_path_follower
