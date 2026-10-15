// Copyright 2026 bichy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include "unitree_adapter/command_limiter.hpp"

namespace unitree_adapter
{
namespace
{

TEST(CommandLimiter, ClampAndDeadband)
{
  CommandLimits limits;
  limits.max_vx = 0.5;
  limits.max_vy = 0.4;
  limits.max_vyaw = 0.8;
  limits.deadband_v = 0.05;
  limits.deadband_w = 0.10;

  const auto limited = applyClampAndDeadband({1.0, -0.03, 0.04}, limits);

  EXPECT_DOUBLE_EQ(limited.vx, 0.5);
  EXPECT_DOUBLE_EQ(limited.vy, 0.0);
  EXPECT_DOUBLE_EQ(limited.vyaw, 0.0);
}

TEST(CommandLimiter, AccelerationLimit)
{
  CommandLimits limits;
  limits.max_accel_x = 1.0;
  limits.max_accel_y = 2.0;
  limits.max_accel_yaw = 3.0;

  const auto limited = applyAccelerationLimit({1.0, -1.0, 1.0}, {0.0, 0.0, 0.0}, limits, 0.1);

  EXPECT_NEAR(limited.vx, 0.1, 1e-9);
  EXPECT_NEAR(limited.vy, -0.2, 1e-9);
  EXPECT_NEAR(limited.vyaw, 0.3, 1e-9);
}

TEST(CommandLimiter, NegativeDtDoesNotRateLimit)
{
  CommandLimits limits;
  const auto limited = applyAccelerationLimit({0.3, 0.2, -0.1}, {0.0, 0.0, 0.0}, limits, -1.0);

  EXPECT_DOUBLE_EQ(limited.vx, 0.3);
  EXPECT_DOUBLE_EQ(limited.vy, 0.2);
  EXPECT_DOUBLE_EQ(limited.vyaw, -0.1);
}

TEST(CommandLimiter, ZeroCommand)
{
  EXPECT_TRUE(isZeroCommand({0.0, 0.0, 0.0}));
  EXPECT_FALSE(isZeroCommand({0.0, 0.01, 0.0}));
}

}  // namespace
}  // namespace unitree_adapter
