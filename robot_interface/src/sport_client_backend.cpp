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

#include "unitree_adapter/sport_client_backend.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#if ROBOCUP_UNITREE_USE_SDK2
#include <unitree/robot/a2/sport/sport_client.hpp>
#include <unitree/robot/b2/sport/sport_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#endif

namespace unitree_adapter
{
namespace
{

std::string toLower(std::string value)
{
  std::transform(
    value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
  return value;
}

template<typename T, typename = void>
struct HasBodyHeight : std::false_type {};

template<typename T>
struct HasBodyHeight<
  T,
  std::void_t<decltype(std::declval<T &>().BodyHeight(std::declval<float>()))>>
  : std::true_type {};

class MockSportClientBackend : public SportClientBackend
{
public:
  bool init(const BackendConfig & config, std::string * error) override
  {
    (void)error;
    config_ = config;
    initialized_ = true;
    return true;
  }

  int32_t move(const VelocityCommand & command) override
  {
    last_command_ = command;
    ++move_count_;
    return initialized_ ? 0 : -1;
  }

  int32_t stopMove() override
  {
    last_command_ = {};
    ++stop_count_;
    return initialized_ ? 0 : -1;
  }

  int32_t damp() override
  {
    ++damp_count_;
    return initialized_ ? 0 : -1;
  }

  int32_t balanceStand() override
  {
    ++balance_count_;
    return initialized_ ? 0 : -1;
  }

  int32_t recoveryStand() override
  {
    ++recovery_count_;
    return initialized_ ? 0 : -1;
  }

  int32_t bodyHeight(double height_m) override
  {
    last_body_height_m_ = height_m;
    ++body_height_count_;
    return initialized_ ? 0 : -1;
  }

  std::string name() const override
  {
    return "mock";
  }

private:
  BackendConfig config_;
  VelocityCommand last_command_;
  bool initialized_{false};
  int64_t move_count_{0};
  int64_t stop_count_{0};
  int64_t damp_count_{0};
  int64_t balance_count_{0};
  int64_t recovery_count_{0};
  double last_body_height_m_{0.0};
  int64_t body_height_count_{0};
};

#if ROBOCUP_UNITREE_USE_SDK2
template<typename ClientT>
class UnitreeSdk2TypedBackend : public SportClientBackend
{
public:
  bool init(const BackendConfig & config, std::string * error) override
  {
    if (config.network_interface.empty()) {
      if (error != nullptr) {
        *error = "backend sdk2 requires backend.network_interface";
      }
      return false;
    }

    try {
      unitree::robot::ChannelFactory::Instance()->Init(
        static_cast<uint32_t>(config.domain_id),
        config.network_interface);
      client_ = std::make_unique<ClientT>();
      client_->SetTimeout(static_cast<float>(config.sdk_timeout_sec));
      client_->Init();

      if (config.recovery_stand_on_connect) {
        const int32_t ret = client_->RecoveryStand();
        if (ret != 0) {
          if (error != nullptr) {
            *error = "RecoveryStand failed with code " + std::to_string(ret);
          }
          return false;
        }
      }

      if (config.balance_stand_on_connect) {
        const int32_t ret = client_->BalanceStand();
        if (ret != 0) {
          if (error != nullptr) {
            *error = "BalanceStand failed with code " + std::to_string(ret);
          }
          return false;
        }
      }
    } catch (const std::exception & ex) {
      if (error != nullptr) {
        *error = ex.what();
      }
      return false;
    }

    return true;
  }

  int32_t move(const VelocityCommand & command) override
  {
    return client_->Move(
      static_cast<float>(command.vx),
      static_cast<float>(command.vy),
      static_cast<float>(command.vyaw));
  }

  int32_t stopMove() override
  {
    return client_->StopMove();
  }

  int32_t damp() override
  {
    return client_->Damp();
  }

  int32_t balanceStand() override
  {
    return client_->BalanceStand();
  }

  int32_t recoveryStand() override
  {
    return client_->RecoveryStand();
  }

  int32_t bodyHeight(double height_m) override
  {
    if constexpr (HasBodyHeight<ClientT>::value) {
      return client_->BodyHeight(static_cast<float>(height_m));
    } else {
      (void)height_m;
      return -2;
    }
  }

  std::string name() const override
  {
    return "sdk2";
  }

private:
  std::unique_ptr<ClientT> client_;
};

class UnitreeSdk2DispatchBackend : public SportClientBackend
{
public:
  bool init(const BackendConfig & config, std::string * error) override
  {
    const auto model = toLower(config.robot_model);
    if (model == "a2" || model == "g2-w" || model == "g2w") {
      typed_ = std::make_unique<UnitreeSdk2TypedBackend<unitree::robot::a2::SportClient>>();
    } else if (model == "b2" || model == "b2w") {
      typed_ = std::make_unique<UnitreeSdk2TypedBackend<unitree::robot::b2::SportClient>>();
    } else if (model == "go2" || model == "go2w") {
      typed_ = std::make_unique<UnitreeSdk2TypedBackend<unitree::robot::go2::SportClient>>();
    } else {
      if (error != nullptr) {
        *error = "unsupported robot_model for sdk2 backend: " + config.robot_model;
      }
      return false;
    }
    model_ = model;
    return typed_->init(config, error);
  }

  int32_t move(const VelocityCommand & command) override {return typed_->move(command);}
  int32_t stopMove() override {return typed_->stopMove();}
  int32_t damp() override {return typed_->damp();}
  int32_t balanceStand() override {return typed_->balanceStand();}
  int32_t recoveryStand() override {return typed_->recoveryStand();}
  int32_t bodyHeight(double height_m) override {return typed_->bodyHeight(height_m);}

  std::string name() const override
  {
    return "sdk2:" + model_;
  }

private:
  std::string model_;
  std::unique_ptr<SportClientBackend> typed_;
};

#else

class MissingSdk2Backend : public SportClientBackend
{
public:
  bool init(const BackendConfig & config, std::string * error) override
  {
    (void)config;
    if (error != nullptr) {
      *error =
        "backend sdk2 requested, but package was built without "
        "-DROBOCUP_UNITREE_USE_SDK2=ON";
    }
    return false;
  }

  int32_t move(const VelocityCommand & command) override {(void)command; return -1;}
  int32_t stopMove() override {return -1;}
  int32_t damp() override {return -1;}
  int32_t balanceStand() override {return -1;}
  int32_t recoveryStand() override {return -1;}
  int32_t bodyHeight(double height_m) override {(void)height_m; return -1;}
  std::string name() const override {return "sdk2-unavailable";}
};

#endif

}  // namespace

std::unique_ptr<SportClientBackend> createSportClientBackend(const std::string & backend_type)
{
  const auto backend = toLower(backend_type);
  if (backend == "mock" || backend == "dry_run" || backend == "dry-run") {
    return std::make_unique<MockSportClientBackend>();
  }

  if (backend == "sdk2" || backend == "unitree_sdk2") {
#if ROBOCUP_UNITREE_USE_SDK2
    return std::make_unique<UnitreeSdk2DispatchBackend>();
#else
    return std::make_unique<MissingSdk2Backend>();
#endif
  }

  return nullptr;
}

}  // namespace unitree_adapter
