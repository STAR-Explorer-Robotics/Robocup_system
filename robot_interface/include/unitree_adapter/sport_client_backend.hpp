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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "unitree_adapter/command_limiter.hpp"

namespace unitree_adapter
{

struct BackendConfig
{
  std::string backend_type{"mock"};
  std::string robot_model{"a2"};
  std::string network_interface;
  int domain_id{0};
  double sdk_timeout_sec{2.0};
  bool balance_stand_on_connect{false};
  bool recovery_stand_on_connect{false};
};

class SportClientBackend
{
public:
  virtual ~SportClientBackend() = default;

  virtual bool init(const BackendConfig & config, std::string * error) = 0;
  virtual int32_t move(const VelocityCommand & command) = 0;
  virtual int32_t stopMove() = 0;
  virtual int32_t damp() = 0;
  virtual int32_t balanceStand() = 0;
  virtual int32_t recoveryStand() = 0;
  virtual int32_t bodyHeight(double height_m) = 0;
  virtual std::string name() const = 0;
};

std::unique_ptr<SportClientBackend> createSportClientBackend(const std::string & backend_type);

}  // namespace unitree_adapter
