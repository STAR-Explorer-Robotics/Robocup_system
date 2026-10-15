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

#pragma once

#include <cmath>
#include <string>

namespace unitree_adapter
{

struct VelocityCommand
{
  double vx{0.0};
  double vy{0.0};
  double vyaw{0.0};
};

struct CommandLimits
{
  double max_vx{0.45};
  double max_vy{0.35};
  double max_vyaw{0.70};
  double max_accel_x{0.80};
  double max_accel_y{0.80};
  double max_accel_yaw{1.20};
  double deadband_v{0.02};
  double deadband_w{0.03};
};

double clamp(double value, double min_value, double max_value);
double applyDeadband(double value, double deadband);
double limitRate(double target, double previous, double max_delta);

VelocityCommand applyClampAndDeadband(
  const VelocityCommand & command,
  const CommandLimits & limits);
VelocityCommand applyAccelerationLimit(
  const VelocityCommand & target,
  const VelocityCommand & previous,
  const CommandLimits & limits,
  double dt_sec);

bool isZeroCommand(const VelocityCommand & command);

std::string commandToString(const VelocityCommand & command);

}  // namespace unitree_adapter
