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

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "unitree_adapter/command_limiter.hpp"
#include "unitree_adapter/srv/set_body_height.hpp"

namespace unitree_adapter
{
namespace
{

using namespace std::chrono_literals;

enum class AdapterState : int32_t
{
  DISABLED = 0,
  READY = 1,
  ACTIVE = 2,
  CMD_TIMEOUT = 3,
  STOPPED = 4,
  SDK_ERROR = 5,
  DAMPING = 6,
  INVALID_CMD = 7,
};

const char * stateName(AdapterState state)
{
  switch (state) {
    case AdapterState::DISABLED:
      return "DISABLED";
    case AdapterState::READY:
      return "READY";
    case AdapterState::ACTIVE:
      return "ACTIVE";
    case AdapterState::CMD_TIMEOUT:
      return "CMD_TIMEOUT";
    case AdapterState::STOPPED:
      return "STOPPED";
    case AdapterState::SDK_ERROR:
      return "SDK_ERROR";
    case AdapterState::DAMPING:
      return "DAMPING";
    case AdapterState::INVALID_CMD:
      return "INVALID_CMD";
  }
  return "UNKNOWN";
}

bool isFinite(const geometry_msgs::msg::Twist & msg)
{
  return std::isfinite(msg.linear.x) &&
         std::isfinite(msg.linear.y) &&
         std::isfinite(msg.angular.z);
}

VelocityCommand fromTwist(const geometry_msgs::msg::Twist & msg)
{
  VelocityCommand command;
  command.vx = msg.linear.x;
  command.vy = msg.linear.y;
  command.vyaw = msg.angular.z;
  return command;
}

geometry_msgs::msg::Twist toTwist(const VelocityCommand & command)
{
  geometry_msgs::msg::Twist msg;
  msg.linear.x = command.vx;
  msg.linear.y = command.vy;
  msg.angular.z = command.vyaw;
  return msg;
}

std::string executableDirectory()
{
  char buffer[4096];
  const ssize_t length = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (length <= 0) {
    return ".";
  }
  buffer[length] = '\0';
  std::string path(buffer);
  const auto slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return ".";
  }
  return path.substr(0, slash);
}

std::string formatMoveCommand(const VelocityCommand & command)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(6)
      << "MOVE " << command.vx << " " << command.vy << " " << command.vyaw;
  return out.str();
}

std::string formatBodyHeightCommand(double height_m)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << "BODY_HEIGHT " << height_m;
  return out.str();
}

std::string errnoString()
{
  return std::strerror(errno);
}

}  // namespace

class A2CmdVelBridgeNode : public rclcpp::Node
{
public:
  explicit A2CmdVelBridgeNode(const rclcpp::NodeOptions & options)
  : Node("a2_cmd_vel_bridge", options)
  {
    ::signal(SIGPIPE, SIG_IGN);
    readParameters();
    setupRosInterfaces();
    if (!startWorker()) {
      setState(AdapterState::SDK_ERROR, worker_error_);
    } else {
      setState(enabled_ ? AdapterState::READY : AdapterState::DISABLED, "worker_started");
    }

    const auto period = std::chrono::duration<double>(1.0 / control_rate_hz_);
    control_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&A2CmdVelBridgeNode::controlTimer, this));
  }

  ~A2CmdVelBridgeNode() override
  {
    stopWorker();
  }

private:
  void readParameters()
  {
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    last_cmd_topic_ = declare_parameter<std::string>(
      "last_cmd_topic",
      "/unitree_adapter/last_cmd");
    state_topic_ = declare_parameter<std::string>("state_topic", "/unitree_adapter/state");
    status_topic_ = declare_parameter<std::string>("status_topic", "/unitree_adapter/status");
    set_enabled_service_ = declare_parameter<std::string>(
      "set_enabled_service",
      "/unitree_adapter/set_enabled");
    stop_service_ = declare_parameter<std::string>("stop_service", "/unitree_adapter/stop");
    damp_service_ = declare_parameter<std::string>("damp_service", "/unitree_adapter/damp");
    balance_stand_service_ = declare_parameter<std::string>(
      "balance_stand_service",
      "/unitree_adapter/balance_stand");
    recovery_stand_service_ = declare_parameter<std::string>(
      "recovery_stand_service",
      "/unitree_adapter/recovery_stand");
    body_height_service_ = declare_parameter<std::string>(
      "body_height_service",
      "/unitree_adapter/body_height");

    network_interface_ = declare_parameter<std::string>("network_interface", "");
    robot_model_ = declare_parameter<std::string>("robot_model", "a2");
    worker_path_ = declare_parameter<std::string>("worker_path", "");
    const char * sdk_prefix = std::getenv("UNITREE_SDK_PREFIX");
    const std::string default_worker_ld_library_path =
      sdk_prefix == nullptr ? "" : std::string(sdk_prefix) + "/lib";
    worker_ld_library_path_ = declare_parameter<std::string>(
      "worker_ld_library_path",
      default_worker_ld_library_path);
    sdk_timeout_sec_ = declare_parameter<double>("sdk_timeout_sec", 2.0);

    control_rate_hz_ = declare_parameter<double>("control_rate_hz", 50.0);
    cmd_timeout_sec_ = declare_parameter<double>("cmd_timeout_sec", 0.25);
    enabled_ = declare_parameter<bool>("enabled_on_start", true);
    publish_status_rate_hz_ = declare_parameter<double>("publish_status_rate_hz", 5.0);
    yaw_rate_sign_ = declare_parameter<double>("yaw_sign", 1.0);
    zero_command_is_stop_ = declare_parameter<bool>("zero_command_is_stop", true);
    zero_stop_repeat_period_sec_ = declare_parameter<double>(
      "zero_stop_repeat_period_sec",
      0.25);
    max_consecutive_write_errors_ = declare_parameter<int>(
      "max_consecutive_write_errors",
      3);

    body_height_service_enabled_ = declare_parameter<bool>("body_height_enabled", false);
    body_height_min_m_ = declare_parameter<double>("body_height_min_m", 0.30);
    body_height_max_m_ = declare_parameter<double>("body_height_max_m", 0.50);
    body_height_stop_before_call_ = declare_parameter<bool>(
      "body_height_stop_before_call",
      true);
    body_height_clear_velocity_on_request_ = declare_parameter<bool>(
      "body_height_clear_velocity_on_request",
      true);

    limits_.max_vx = declare_parameter<double>("max_vx", 0.45);
    limits_.max_vy = declare_parameter<double>("max_vy", 0.35);
    limits_.max_vyaw = declare_parameter<double>("max_vyaw", 0.70);
    limits_.max_accel_x = declare_parameter<double>("max_accel_x", 0.80);
    limits_.max_accel_y = declare_parameter<double>("max_accel_y", 0.80);
    limits_.max_accel_yaw = declare_parameter<double>("max_accel_yaw", 1.20);
    limits_.deadband_v = declare_parameter<double>("deadband_v", 0.02);
    limits_.deadband_w = declare_parameter<double>("deadband_w", 0.03);

    control_rate_hz_ = std::max(1.0, control_rate_hz_);
    cmd_timeout_sec_ = std::max(0.01, cmd_timeout_sec_);
    publish_status_rate_hz_ = std::max(0.1, publish_status_rate_hz_);
    sdk_timeout_sec_ = std::max(0.05, sdk_timeout_sec_);
    zero_stop_repeat_period_sec_ = std::max(0.05, zero_stop_repeat_period_sec_);
    max_consecutive_write_errors_ = std::max(1, max_consecutive_write_errors_);
    if (std::abs(yaw_rate_sign_) < 1e-6) {
      RCLCPP_WARN(get_logger(), "yaw_sign is near zero; using 1.0");
      yaw_rate_sign_ = 1.0;
    }
    if (body_height_min_m_ > body_height_max_m_) {
      std::swap(body_height_min_m_, body_height_max_m_);
    }
  }

  void setupRosInterfaces()
  {
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_,
      rclcpp::SystemDefaultsQoS(),
      std::bind(&A2CmdVelBridgeNode::cmdCallback, this, std::placeholders::_1));
    last_cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(
      last_cmd_topic_,
      rclcpp::SystemDefaultsQoS());
    state_pub_ = create_publisher<std_msgs::msg::Int32>(
      state_topic_,
      rclcpp::SystemDefaultsQoS());
    status_pub_ = create_publisher<std_msgs::msg::String>(
      status_topic_,
      rclcpp::SystemDefaultsQoS());

    set_enabled_srv_ = create_service<std_srvs::srv::SetBool>(
      set_enabled_service_,
      std::bind(
        &A2CmdVelBridgeNode::handleSetEnabled,
        this,
        std::placeholders::_1,
        std::placeholders::_2));
    stop_srv_ = create_service<std_srvs::srv::Trigger>(
      stop_service_,
      std::bind(
        &A2CmdVelBridgeNode::handleStop,
        this,
        std::placeholders::_1,
        std::placeholders::_2));
    damp_srv_ = create_service<std_srvs::srv::Trigger>(
      damp_service_,
      std::bind(
        &A2CmdVelBridgeNode::handleDamp,
        this,
        std::placeholders::_1,
        std::placeholders::_2));
    balance_stand_srv_ = create_service<std_srvs::srv::Trigger>(
      balance_stand_service_,
      std::bind(
        &A2CmdVelBridgeNode::handleBalanceStand,
        this,
        std::placeholders::_1,
        std::placeholders::_2));
    recovery_stand_srv_ = create_service<std_srvs::srv::Trigger>(
      recovery_stand_service_,
      std::bind(
        &A2CmdVelBridgeNode::handleRecoveryStand,
        this,
        std::placeholders::_1,
        std::placeholders::_2));
    body_height_srv_ = create_service<unitree_adapter::srv::SetBodyHeight>(
      body_height_service_,
      std::bind(
        &A2CmdVelBridgeNode::handleBodyHeight,
        this,
        std::placeholders::_1,
        std::placeholders::_2));
  }

  bool startWorker()
  {
    if (network_interface_.empty()) {
      worker_error_ = "network_interface is empty";
      return false;
    }

    resolved_worker_path_ = worker_path_.empty() ?
      executableDirectory() + "/a2_sdk2_sport_worker" :
      worker_path_;

    int pipe_fds[2] = {-1, -1};
    if (::pipe(pipe_fds) != 0) {
      worker_error_ = "pipe failed: " + errnoString();
      return false;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
      worker_error_ = "fork failed: " + errnoString();
      ::close(pipe_fds[0]);
      ::close(pipe_fds[1]);
      return false;
    }

    if (pid == 0) {
      ::close(pipe_fds[1]);
      if (::dup2(pipe_fds[0], STDIN_FILENO) < 0) {
        std::perror("dup2");
        _exit(127);
      }
      ::close(pipe_fds[0]);
      ::unsetenv("CYCLONEDDS_URI");
      ::unsetenv("RMW_IMPLEMENTATION");
      if (worker_ld_library_path_.empty()) {
        ::unsetenv("LD_LIBRARY_PATH");
      } else {
        ::setenv("LD_LIBRARY_PATH", worker_ld_library_path_.c_str(), 1);
      }
      const std::string timeout_arg = std::to_string(sdk_timeout_sec_);
      ::execl(
        resolved_worker_path_.c_str(),
        resolved_worker_path_.c_str(),
        network_interface_.c_str(),
        timeout_arg.c_str(),
        robot_model_.c_str(),
        static_cast<char *>(nullptr));
      std::perror("execl a2_sdk2_sport_worker");
      _exit(127);
    }

    ::close(pipe_fds[0]);
    const int flags = ::fcntl(pipe_fds[1], F_GETFL, 0);
    if (flags >= 0) {
      ::fcntl(pipe_fds[1], F_SETFL, flags | O_NONBLOCK);
    }
    const int fd_flags = ::fcntl(pipe_fds[1], F_GETFD, 0);
    if (fd_flags >= 0) {
      ::fcntl(pipe_fds[1], F_SETFD, fd_flags | FD_CLOEXEC);
    }

    worker_pid_ = pid;
    worker_stdin_fd_ = pipe_fds[1];
    worker_ready_ = true;
    RCLCPP_INFO(
      get_logger(),
      "started %s for %s on %s, pid=%d",
      resolved_worker_path_.c_str(),
      robot_model_.c_str(),
      network_interface_.c_str(),
      static_cast<int>(worker_pid_));
    return true;
  }

  void stopWorker()
  {
    pid_t pid = -1;
    {
      std::lock_guard<std::mutex> lock(worker_mutex_);
      if (worker_stdin_fd_ >= 0) {
        writeRawLocked("STOP\n");
        writeRawLocked("QUIT\n");
        ::close(worker_stdin_fd_);
        worker_stdin_fd_ = -1;
      }
      pid = worker_pid_;
      worker_pid_ = -1;
      worker_ready_ = false;
    }

    if (pid > 0) {
      for (int i = 0; i < 20; ++i) {
        int status = 0;
        const pid_t ret = ::waitpid(pid, &status, WNOHANG);
        if (ret == pid) {
          return;
        }
        std::this_thread::sleep_for(50ms);
      }
      ::kill(pid, SIGTERM);
      for (int i = 0; i < 10; ++i) {
        int status = 0;
        const pid_t ret = ::waitpid(pid, &status, WNOHANG);
        if (ret == pid) {
          return;
        }
        std::this_thread::sleep_for(50ms);
      }
      ::kill(pid, SIGKILL);
      int status = 0;
      ::waitpid(pid, &status, 0);
    }
  }

  void cmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    latest_cmd_time_ = now();
    if (!isFinite(*msg)) {
      latest_cmd_valid_ = false;
      has_latest_cmd_ = true;
      return;
    }
    latest_cmd_ = fromTwist(*msg);
    latest_cmd_valid_ = true;
    has_latest_cmd_ = true;
  }

  void controlTimer()
  {
    const auto current_time = now();
    checkWorkerAlive();
    if (!worker_ready_) {
      setState(AdapterState::SDK_ERROR, worker_error_.empty() ? "worker_not_ready" : worker_error_);
      publishStatusIfNeeded(current_time);
      return;
    }

    if (!enabled_) {
      sendStopIfNeeded(current_time, AdapterState::DISABLED, "disabled");
      publishStatusIfNeeded(current_time);
      return;
    }

    VelocityCommand target;
    std::optional<rclcpp::Time> target_time;
    bool target_valid = false;
    bool has_target = false;
    {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      target = latest_cmd_;
      target_time = latest_cmd_time_;
      target_valid = latest_cmd_valid_;
      has_target = has_latest_cmd_;
    }

    if (!has_target || !target_time.has_value()) {
      sendStopIfNeeded(current_time, AdapterState::READY, "waiting_for_cmd_vel");
      publishStatusIfNeeded(current_time);
      return;
    }

    if (!target_valid) {
      sendStopIfNeeded(current_time, AdapterState::INVALID_CMD, "invalid_cmd_vel");
      publishStatusIfNeeded(current_time);
      return;
    }

    const double cmd_age = (current_time - *target_time).seconds();
    if (cmd_age > cmd_timeout_sec_) {
      std::ostringstream reason;
      reason << "cmd_timeout age=" << std::fixed << std::setprecision(3) << cmd_age;
      sendStopIfNeeded(current_time, AdapterState::CMD_TIMEOUT, reason.str());
      publishStatusIfNeeded(current_time);
      return;
    }

    target.vyaw *= yaw_rate_sign_;
    VelocityCommand limited = applyClampAndDeadband(target, limits_);
    const double dt_sec = last_sent_cmd_time_.has_value() ?
      std::max(0.0, (current_time - *last_sent_cmd_time_).seconds()) :
      1.0 / control_rate_hz_;
    limited = applyAccelerationLimit(limited, last_limited_cmd_, limits_, dt_sec);

    if (zero_command_is_stop_ && isZeroCommand(limited)) {
      sendStopIfNeeded(current_time, AdapterState::STOPPED, "zero_cmd_vel");
      publishStatusIfNeeded(current_time);
      return;
    }

    if (sendWorkerCommand(formatMoveCommand(limited)) == 0) {
      consecutive_write_errors_ = 0;
      stop_sent_ = false;
      last_limited_cmd_ = limited;
      last_sent_cmd_ = limited;
      last_sent_cmd_time_ = current_time;
      setState(AdapterState::ACTIVE, "move");
      publishLastCmd(limited);
    } else {
      handleWriteError("move");
    }
    publishStatusIfNeeded(current_time);
  }

  void handleSetEnabled(
    const std_srvs::srv::SetBool::Request::SharedPtr request,
    std_srvs::srv::SetBool::Response::SharedPtr response)
  {
    checkWorkerAlive();
    if (request->data && !worker_ready_) {
      response->success = false;
      response->message = worker_error_.empty() ? "worker not ready" : worker_error_;
      return;
    }

    enabled_ = request->data;
    manual_stop_latched_ = false;
    if (!enabled_) {
      sendStopNow("disabled_by_service");
      setState(AdapterState::DISABLED, "disabled_by_service");
    } else {
      stop_sent_ = false;
      setState(AdapterState::READY, "enabled_by_service");
    }
    response->success = true;
    response->message = enabled_ ? "enabled" : "disabled";
  }

  void handleStop(
    const std_srvs::srv::Trigger::Request::SharedPtr request,
    std_srvs::srv::Trigger::Response::SharedPtr response)
  {
    (void)request;
    enabled_ = false;
    manual_stop_latched_ = true;
    const bool ok = sendStopNow("manual_stop");
    setState(
      ok ? AdapterState::STOPPED : AdapterState::SDK_ERROR,
      ok ? "manual_stop" : "stop_failed");
    response->success = ok;
    response->message = ok ? "stopped and disabled" : "failed to send STOP";
  }

  void handleDamp(
    const std_srvs::srv::Trigger::Request::SharedPtr request,
    std_srvs::srv::Trigger::Response::SharedPtr response)
  {
    (void)request;
    enabled_ = false;
    manual_stop_latched_ = true;
    const bool ok = sendSimpleWorkerCommand("DAMP", AdapterState::DAMPING, "damp");
    response->success = ok;
    response->message = ok ? "damp sent" : "failed to send DAMP";
  }

  void handleBalanceStand(
    const std_srvs::srv::Trigger::Request::SharedPtr request,
    std_srvs::srv::Trigger::Response::SharedPtr response)
  {
    (void)request;
    const bool ok = sendSimpleWorkerCommand("BALANCE", AdapterState::READY, "balance_stand");
    response->success = ok;
    response->message = ok ? "balance stand sent" : "failed to send BALANCE";
  }

  void handleRecoveryStand(
    const std_srvs::srv::Trigger::Request::SharedPtr request,
    std_srvs::srv::Trigger::Response::SharedPtr response)
  {
    (void)request;
    const bool ok = sendSimpleWorkerCommand("RECOVERY", AdapterState::READY, "recovery_stand");
    response->success = ok;
    response->message = ok ? "recovery stand sent" : "failed to send RECOVERY";
  }

  void handleBodyHeight(
    const unitree_adapter::srv::SetBodyHeight::Request::SharedPtr request,
    unitree_adapter::srv::SetBodyHeight::Response::SharedPtr response)
  {
    if (!body_height_service_enabled_) {
      response->success = false;
      response->message = "body height service disabled";
      response->accepted_height_m = last_body_height_command_m_;
      return;
    }

    const double requested_height = request->height_m;
    if (!std::isfinite(requested_height)) {
      response->success = false;
      response->message = "height is not finite";
      response->accepted_height_m = last_body_height_command_m_;
      return;
    }
    if (requested_height < body_height_min_m_ || requested_height > body_height_max_m_) {
      std::ostringstream out;
      out << "height " << requested_height << " outside ["
          << body_height_min_m_ << ", " << body_height_max_m_ << "]";
      response->success = false;
      response->message = out.str();
      response->accepted_height_m = last_body_height_command_m_;
      return;
    }

    if (body_height_clear_velocity_on_request_) {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      latest_cmd_ = {};
      latest_cmd_valid_ = true;
      has_latest_cmd_ = true;
      latest_cmd_time_ = now();
    }
    if (body_height_stop_before_call_ && !sendStopNow("body_height_stop")) {
      response->success = false;
      response->message = "failed to stop before body height call";
      response->accepted_height_m = last_body_height_command_m_;
      return;
    }

    const bool ok = sendWorkerCommand(formatBodyHeightCommand(requested_height)) == 0;
    if (ok) {
      last_body_height_command_m_ = requested_height;
      response->success = true;
      response->message = "body height command sent";
      response->accepted_height_m = requested_height;
    } else {
      response->success = false;
      response->message = "failed to send body height command";
      response->accepted_height_m = last_body_height_command_m_;
      handleWriteError("body_height");
    }
  }

  bool sendSimpleWorkerCommand(
    const std::string & command,
    AdapterState success_state,
    const std::string & reason)
  {
    if (sendWorkerCommand(command) != 0) {
      handleWriteError(reason);
      return false;
    }
    consecutive_write_errors_ = 0;
    setState(success_state, reason);
    return true;
  }

  bool sendStopNow(const std::string & reason)
  {
    const auto current_time = now();
    if (sendWorkerCommand("STOP") != 0) {
      handleWriteError(reason);
      return false;
    }
    consecutive_write_errors_ = 0;
    stop_sent_ = true;
    last_stop_time_ = current_time;
    last_limited_cmd_ = {};
    last_sent_cmd_ = {};
    last_sent_cmd_time_ = current_time;
    publishLastCmd({});
    return true;
  }

  void sendStopIfNeeded(
    const rclcpp::Time & current_time,
    AdapterState stop_state,
    const std::string & reason)
  {
    const bool repeat_due =
      !last_stop_time_.has_value() ||
      (current_time - *last_stop_time_).seconds() >= zero_stop_repeat_period_sec_;
    if (!stop_sent_ || repeat_due) {
      if (!sendStopNow(reason)) {
        return;
      }
    }
    setState(stop_state, reason);
  }

  int sendWorkerCommand(const std::string & command)
  {
    const std::string payload = command + "\n";
    std::lock_guard<std::mutex> lock(worker_mutex_);
    if (!checkWorkerAliveLocked()) {
      return -EPIPE;
    }
    return writeRawLocked(payload);
  }

  int writeRawLocked(const std::string & payload)
  {
    if (worker_stdin_fd_ < 0) {
      errno = EPIPE;
      worker_error_ = "worker stdin is closed";
      return -EPIPE;
    }

    size_t offset = 0;
    while (offset < payload.size()) {
      const ssize_t written =
        ::write(worker_stdin_fd_, payload.data() + offset, payload.size() - offset);
      if (written > 0) {
        offset += static_cast<size_t>(written);
        continue;
      }
      if (written < 0 && errno == EINTR) {
        continue;
      }
      if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        worker_error_ = "worker command pipe is full";
        return -EAGAIN;
      }
      worker_error_ = "worker write failed: " + errnoString();
      worker_ready_ = false;
      return -EPIPE;
    }
    return 0;
  }

  void checkWorkerAlive()
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    checkWorkerAliveLocked();
  }

  bool checkWorkerAliveLocked()
  {
    if (worker_pid_ <= 0) {
      worker_ready_ = false;
      return false;
    }
    int status = 0;
    const pid_t ret = ::waitpid(worker_pid_, &status, WNOHANG);
    if (ret == 0) {
      return true;
    }
    if (ret == worker_pid_) {
      std::ostringstream out;
      if (WIFEXITED(status)) {
        out << "worker exited with code " << WEXITSTATUS(status);
      } else if (WIFSIGNALED(status)) {
        out << "worker killed by signal " << WTERMSIG(status);
      } else {
        out << "worker exited";
      }
      worker_error_ = out.str();
      if (worker_stdin_fd_ >= 0) {
        ::close(worker_stdin_fd_);
        worker_stdin_fd_ = -1;
      }
      worker_pid_ = -1;
      worker_ready_ = false;
      return false;
    }
    if (errno != ECHILD) {
      worker_error_ = "waitpid failed: " + errnoString();
    }
    return worker_ready_;
  }

  void handleWriteError(const std::string & command)
  {
    ++consecutive_write_errors_;
    std::ostringstream reason;
    reason << command << "_write_error";
    if (!worker_error_.empty()) {
      reason << ": " << worker_error_;
    }
    setState(AdapterState::SDK_ERROR, reason.str());
    if (consecutive_write_errors_ >= max_consecutive_write_errors_) {
      enabled_ = false;
      RCLCPP_ERROR(
        get_logger(),
        "disabling bridge after %d consecutive worker write errors: %s",
        consecutive_write_errors_,
        worker_error_.c_str());
    }
  }

  void setState(AdapterState state, const std::string & reason)
  {
    if (state_ != state || last_reason_ != reason) {
      publish_state_now_ = true;
    }
    state_ = state;
    last_reason_ = reason;
  }

  void publishLastCmd(const VelocityCommand & command)
  {
    if (last_cmd_pub_) {
      last_cmd_pub_->publish(toTwist(command));
    }
  }

  void publishStatusIfNeeded(const rclcpp::Time & current_time)
  {
    if (publish_state_now_) {
      std_msgs::msg::Int32 state_msg;
      state_msg.data = static_cast<int32_t>(state_);
      state_pub_->publish(state_msg);
      publish_state_now_ = false;
    }

    const bool status_due =
      !last_status_publish_time_.has_value() ||
      (current_time - *last_status_publish_time_).seconds() >=
      (1.0 / publish_status_rate_hz_);
    if (!status_due) {
      return;
    }
    last_status_publish_time_ = current_time;

    const double cmd_age = latest_cmd_time_.has_value() ?
      (current_time - *latest_cmd_time_).seconds() :
      -1.0;
    std::ostringstream out;
    out << "{"
        << "\"state\":\"" << stateName(state_) << "\","
        << "\"enabled\":" << (enabled_ ? "true" : "false") << ","
        << "\"manual_stop_latched\":" << (manual_stop_latched_ ? "true" : "false") << ","
        << "\"worker_ready\":" << (worker_ready_ ? "true" : "false") << ","
        << "\"worker_pid\":" << static_cast<int>(worker_pid_) << ","
        << "\"worker_path\":\"" << resolved_worker_path_ << "\","
        << "\"network_interface\":\"" << network_interface_ << "\","
        << "\"robot_model\":\"" << robot_model_ << "\","
        << "\"cmd_age_sec\":" << std::fixed << std::setprecision(3) << cmd_age << ","
        << "\"last_cmd\":\"" << commandToString(last_sent_cmd_) << "\","
        << "\"write_errors\":" << consecutive_write_errors_ << ","
        << "\"reason\":\"" << last_reason_ << "\""
        << "}";
    std_msgs::msg::String status_msg;
    status_msg.data = out.str();
    status_pub_->publish(status_msg);
  }

  std::string cmd_vel_topic_;
  std::string last_cmd_topic_;
  std::string state_topic_;
  std::string status_topic_;
  std::string set_enabled_service_;
  std::string stop_service_;
  std::string damp_service_;
  std::string balance_stand_service_;
  std::string recovery_stand_service_;
  std::string body_height_service_;

  std::string network_interface_;
  std::string robot_model_;
  std::string worker_path_;
  std::string worker_ld_library_path_;
  std::string resolved_worker_path_;
  double sdk_timeout_sec_{2.0};
  double control_rate_hz_{50.0};
  double cmd_timeout_sec_{0.25};
  double publish_status_rate_hz_{5.0};
  double yaw_rate_sign_{1.0};
  bool zero_command_is_stop_{true};
  double zero_stop_repeat_period_sec_{0.25};
  int max_consecutive_write_errors_{3};
  CommandLimits limits_;

  bool body_height_service_enabled_{false};
  double body_height_min_m_{0.30};
  double body_height_max_m_{0.50};
  bool body_height_stop_before_call_{true};
  bool body_height_clear_velocity_on_request_{true};
  double last_body_height_command_m_{0.0};

  std::mutex worker_mutex_;
  pid_t worker_pid_{-1};
  int worker_stdin_fd_{-1};
  bool worker_ready_{false};
  std::string worker_error_;

  std::mutex cmd_mutex_;
  bool has_latest_cmd_{false};
  bool latest_cmd_valid_{true};
  VelocityCommand latest_cmd_;
  std::optional<rclcpp::Time> latest_cmd_time_;

  bool enabled_{false};
  bool manual_stop_latched_{false};
  bool stop_sent_{false};
  AdapterState state_{AdapterState::DISABLED};
  bool publish_state_now_{true};
  std::string last_reason_{"startup"};
  int consecutive_write_errors_{0};
  VelocityCommand last_limited_cmd_;
  VelocityCommand last_sent_cmd_;
  std::optional<rclcpp::Time> last_sent_cmd_time_;
  std::optional<rclcpp::Time> last_stop_time_;
  std::optional<rclcpp::Time> last_status_publish_time_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr last_cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_enabled_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr damp_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr balance_stand_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr recovery_stand_srv_;
  rclcpp::Service<unitree_adapter::srv::SetBodyHeight>::SharedPtr body_height_srv_;
  rclcpp::TimerBase::SharedPtr control_timer_;
};

}  // namespace unitree_adapter

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<unitree_adapter::A2CmdVelBridgeNode>(rclcpp::NodeOptions{});
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
