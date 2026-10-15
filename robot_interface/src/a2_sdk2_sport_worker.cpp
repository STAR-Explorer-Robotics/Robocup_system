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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <unitree/robot/a2/sport/sport_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

namespace
{

std::string toUpper(std::string value)
{
  std::transform(
    value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::toupper(c));
    });
  return value;
}

std::string toLower(std::string value)
{
  std::transform(
    value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
  return value;
}

bool isA2Model(const std::string & model)
{
  const auto normalized = toLower(model);
  return normalized == "a2" || normalized == "g2-w" || normalized == "g2w";
}

float parseFloatArg(const char * value, float fallback)
{
  if (value == nullptr) {
    return fallback;
  }
  char * end = nullptr;
  const float parsed = std::strtof(value, &end);
  return end == value ? fallback : parsed;
}

void printRet(const std::string & command, int32_t ret)
{
  std::cout << "RET " << ret << " " << command << std::endl;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::cerr << "usage: a2_sdk2_sport_worker <network_interface> "
              << "[sdk_timeout_sec] [robot_model]"
              << std::endl;
    return 2;
  }

  const std::string network_interface = argv[1];
  const float sdk_timeout_sec = argc >= 3 ? parseFloatArg(argv[2], 2.0F) : 2.0F;
  const std::string robot_model = argc >= 4 ? argv[3] : "a2";
  if (!isA2Model(robot_model)) {
    std::cerr << "a2_sdk2_sport_worker only supports A2-compatible sport "
              << "clients, requested robot_model="
              << robot_model << std::endl;
    return 2;
  }

  try {
    unitree::robot::ChannelFactory::Instance()->Init(
      static_cast<uint32_t>(0),
      network_interface);

    unitree::robot::a2::SportClient client;
    client.SetTimeout(sdk_timeout_sec);
    client.Init();
    std::cout << "READY a2_sdk2_sport_worker iface=" << network_interface
              << " timeout=" << sdk_timeout_sec << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
      std::istringstream input(line);
      std::string command;
      input >> command;
      command = toUpper(command);
      if (command.empty()) {
        continue;
      }

      if (command == "MOVE") {
        float vx = 0.0F;
        float vy = 0.0F;
        float vyaw = 0.0F;
        if (!(input >> vx >> vy >> vyaw)) {
          std::cerr << "ERR malformed MOVE command: " << line << std::endl;
          continue;
        }
        printRet(command, client.Move(vx, vy, vyaw));
      } else if (command == "STOP") {
        printRet(command, client.StopMove());
      } else if (command == "DAMP") {
        printRet(command, client.Damp());
      } else if (command == "BALANCE" || command == "BALANCE_STAND") {
        printRet(command, client.BalanceStand());
      } else if (command == "RECOVERY" || command == "RECOVERY_STAND") {
        printRet(command, client.RecoveryStand());
      } else if (command == "BODY_HEIGHT") {
        float height = 0.0F;
        if (!(input >> height)) {
          std::cerr << "ERR malformed BODY_HEIGHT command: " << line << std::endl;
          continue;
        }
        printRet(command, client.BodyHeight(height));
      } else if (command == "QUIT") {
        printRet("STOP", client.StopMove());
        std::cout << "BYE" << std::endl;
        return 0;
      } else {
        std::cerr << "ERR unknown command: " << line << std::endl;
      }
    }

    client.StopMove();
  } catch (const std::exception & ex) {
    std::cerr << "a2_sdk2_sport_worker exception: " << ex.what() << std::endl;
    return 1;
  }

  return 0;
}
