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

#include "unitree_adapter/command_limiter.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace unitree_adapter
{

double clamp(double value, double min_value, double max_value)
{
  return std::max(min_value, std::min(max_value, value));
}

double applyDeadband(double value, double deadband)
{
  return std::abs(value) < std::abs(deadband) ? 0.0 : value;
}

double limitRate(double target, double previous, double max_delta)
{
  if (max_delta <= 0.0) {
    return target;
  }
  return clamp(target, previous - max_delta, previous + max_delta);
}

VelocityCommand applyClampAndDeadband(const VelocityCommand & command, const CommandLimits & limits)
{
  VelocityCommand limited;
  limited.vx = applyDeadband(clamp(command.vx, -limits.max_vx, limits.max_vx), limits.deadband_v);
  limited.vy = applyDeadband(clamp(command.vy, -limits.max_vy, limits.max_vy), limits.deadband_v);
  limited.vyaw = applyDeadband(
    clamp(command.vyaw, -limits.max_vyaw, limits.max_vyaw),
    limits.deadband_w);
  return limited;
}

VelocityCommand applyAccelerationLimit(
  const VelocityCommand & target,
  const VelocityCommand & previous,
  const CommandLimits & limits,
  double dt_sec)
{
  if (dt_sec <= 0.0) {
    return target;
  }

  VelocityCommand limited;
  limited.vx = limitRate(target.vx, previous.vx, limits.max_accel_x * dt_sec);
  limited.vy = limitRate(target.vy, previous.vy, limits.max_accel_y * dt_sec);
  limited.vyaw = limitRate(target.vyaw, previous.vyaw, limits.max_accel_yaw * dt_sec);
  return limited;
}

bool isZeroCommand(const VelocityCommand & command)
{
  return command.vx == 0.0 && command.vy == 0.0 && command.vyaw == 0.0;
}

std::string commandToString(const VelocityCommand & command)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(3)
      << "vx=" << command.vx
      << ", vy=" << command.vy
      << ", vyaw=" << command.vyaw;
  return out.str();
}

}  // namespace unitree_adapter
