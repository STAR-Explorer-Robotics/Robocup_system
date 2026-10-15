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
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "unitree_adapter/command_limiter.hpp"
#include "unitree_adapter/sport_client_backend.hpp"
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

bool sdkReturnOk(int32_t ret)
{
  return ret == 0;
}

std::string sdkReturnDescription(int32_t ret)
{
  switch (ret) {
    case 0:
      return "Success";
    case -2:
      return "Unsupported by selected backend";
    case 3102:
      return "Send request error";
    case 3103:
      return "API is not registered";
    case 3104:
      return "Call API timeout";
    case 3105:
      return "Response API mismatch";
    case 3106:
      return "Response data error";
    case 3107:
      return "Lease is invalid";
    case 3205:
      return "Request denied by lease";
    default:
      return "SDK error";
  }
}

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

}  // namespace

class UnitreeAdapterNode : public rclcpp::Node
{
public:
  explicit UnitreeAdapterNode(const rclcpp::NodeOptions & options)
  : Node("unitree_adapter_node", options)
  {
    readParameters();
    setupRosInterfaces();
    initBackend();

    const auto period = std::chrono::duration<double>(1.0 / control_rate_hz_);
    control_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&UnitreeAdapterNode::controlTimer, this),
      control_callback_group_);
  }

private:
  void readParameters()
  {
    topics_input_cmd_ = declare_parameter<std::string>("topics.input_cmd", "/cmd_vel");
    topics_last_cmd_ =
      declare_parameter<std::string>("topics.last_cmd", "/unitree_adapter/last_cmd");
    topics_state_ = declare_parameter<std::string>("topics.state", "/unitree_adapter/state");
    topics_status_ = declare_parameter<std::string>("topics.status", "/unitree_adapter/status");

    services_set_enabled_ = declare_parameter<std::string>(
      "services.set_enabled", "/unitree_adapter/set_enabled");
    services_stop_ = declare_parameter<std::string>("services.stop", "/unitree_adapter/stop");
    services_damp_ = declare_parameter<std::string>("services.damp", "/unitree_adapter/damp");
    services_balance_stand_ = declare_parameter<std::string>(
      "services.balance_stand", "/unitree_adapter/balance_stand");
    services_recovery_stand_ = declare_parameter<std::string>(
      "services.recovery_stand", "/unitree_adapter/recovery_stand");
    services_body_height_ = declare_parameter<std::string>(
      "services.body_height", "/unitree_adapter/body_height");

    backend_config_.backend_type = declare_parameter<std::string>("backend.type", "mock");
    backend_config_.robot_model = declare_parameter<std::string>("backend.robot_model", "a2");
    backend_config_.network_interface = declare_parameter<std::string>(
      "backend.network_interface", "");
    backend_config_.domain_id = declare_parameter<int>("backend.domain_id", 0);
    backend_config_.sdk_timeout_sec = declare_parameter<double>("backend.sdk_timeout_sec", 2.0);
    backend_config_.balance_stand_on_connect = declare_parameter<bool>(
      "backend.balance_stand_on_connect", false);
    backend_config_.recovery_stand_on_connect = declare_parameter<bool>(
      "backend.recovery_stand_on_connect", false);

    control_rate_hz_ = declare_parameter<double>("control.rate_hz", 50.0);
    cmd_timeout_sec_ = declare_parameter<double>("control.cmd_timeout_sec", 0.25);
    enabled_ = declare_parameter<bool>("control.enabled_on_start", true);
    publish_status_rate_hz_ = declare_parameter<double>("control.publish_status_rate_hz", 5.0);
    yaw_rate_sign_ = declare_parameter<double>("control.yaw_rate_sign", 1.0);
    zero_command_is_stop_ = declare_parameter<bool>("control.zero_command_is_stop", true);
    zero_stop_repeat_period_sec_ = declare_parameter<double>(
      "control.zero_stop_repeat_period_sec", 0.25);
    max_consecutive_sdk_errors_ = declare_parameter<int>("control.max_consecutive_sdk_errors", 3);

    body_height_service_enabled_ = declare_parameter<bool>("body_height.enabled", false);
    body_height_min_m_ = declare_parameter<double>("body_height.min_m", 0.30);
    body_height_max_m_ = declare_parameter<double>("body_height.max_m", 0.50);
    body_height_stop_before_call_ = declare_parameter<bool>("body_height.stop_before_call", true);
    body_height_clear_velocity_on_request_ = declare_parameter<bool>(
      "body_height.clear_velocity_on_request", true);

    limits_.max_vx = declare_parameter<double>("limits.max_vx", 0.45);
    limits_.max_vy = declare_parameter<double>("limits.max_vy", 0.35);
    limits_.max_vyaw = declare_parameter<double>("limits.max_vyaw", 0.70);
    limits_.max_accel_x = declare_parameter<double>("limits.max_accel_x", 0.80);
    limits_.max_accel_y = declare_parameter<double>("limits.max_accel_y", 0.80);
    limits_.max_accel_yaw = declare_parameter<double>("limits.max_accel_yaw", 1.20);
    limits_.deadband_v = declare_parameter<double>("limits.deadband_v", 0.02);
    limits_.deadband_w = declare_parameter<double>("limits.deadband_w", 0.03);

    control_rate_hz_ = std::max(1.0, control_rate_hz_);
    cmd_timeout_sec_ = std::max(0.01, cmd_timeout_sec_);
    publish_status_rate_hz_ = std::max(0.1, publish_status_rate_hz_);
    if (std::abs(yaw_rate_sign_) < 1e-6) {
      RCLCPP_WARN(get_logger(), "control.yaw_rate_sign is near zero; using 1.0");
      yaw_rate_sign_ = 1.0;
    }
    zero_stop_repeat_period_sec_ = std::max(0.05, zero_stop_repeat_period_sec_);
    max_consecutive_sdk_errors_ = std::max(1, max_consecutive_sdk_errors_);
    if (body_height_min_m_ > body_height_max_m_) {
      RCLCPP_WARN(get_logger(), "body_height.min_m > max_m; swapping limits");
      std::swap(body_height_min_m_, body_height_max_m_);
    }
  }

  void setupRosInterfaces()
  {
    cmd_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    control_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    service_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions cmd_sub_options;
    cmd_sub_options.callback_group = cmd_callback_group_;

    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      topics_input_cmd_,
      rclcpp::QoS(10),
      std::bind(&UnitreeAdapterNode::cmdCallback, this, std::placeholders::_1),
      cmd_sub_options);

    last_cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(topics_last_cmd_, 10);
    state_pub_ = create_publisher<std_msgs::msg::Int32>(topics_state_, 10);
    status_pub_ = create_publisher<std_msgs::msg::String>(topics_status_, 10);

    set_enabled_srv_ = create_service<std_srvs::srv::SetBool>(
      services_set_enabled_,
      std::bind(
        &UnitreeAdapterNode::handleSetEnabled,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      service_callback_group_);

    stop_srv_ = create_service<std_srvs::srv::Trigger>(
      services_stop_,
      std::bind(
        &UnitreeAdapterNode::handleStop,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      service_callback_group_);

    damp_srv_ = create_service<std_srvs::srv::Trigger>(
      services_damp_,
      std::bind(
        &UnitreeAdapterNode::handleDamp,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      service_callback_group_);

    balance_stand_srv_ = create_service<std_srvs::srv::Trigger>(
      services_balance_stand_,
      std::bind(
        &UnitreeAdapterNode::handleBalanceStand,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      service_callback_group_);

    recovery_stand_srv_ = create_service<std_srvs::srv::Trigger>(
      services_recovery_stand_,
      std::bind(
        &UnitreeAdapterNode::handleRecoveryStand,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      service_callback_group_);

    body_height_srv_ = create_service<unitree_adapter::srv::SetBodyHeight>(
      services_body_height_,
      std::bind(
        &UnitreeAdapterNode::handleBodyHeight,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      service_callback_group_);
  }

  void initBackend()
  {
    backend_ = createSportClientBackend(backend_config_.backend_type);
    if (!backend_) {
      backend_error_ = "unknown backend type: " + backend_config_.backend_type;
      backend_ready_ = false;
      transitionTo(AdapterState::SDK_ERROR, backend_error_);
      RCLCPP_ERROR(get_logger(), "%s", backend_error_.c_str());
      return;
    }

    std::string error;
    backend_ready_ = backend_->init(backend_config_, &error);
    if (!backend_ready_) {
      backend_error_ = error.empty() ? "backend initialization failed" : error;
      enabled_ = false;
      transitionTo(AdapterState::SDK_ERROR, backend_error_);
      RCLCPP_ERROR(get_logger(), "%s", backend_error_.c_str());
      return;
    }

    transitionTo(enabled_ ? AdapterState::READY : AdapterState::DISABLED, "backend ready");
    RCLCPP_INFO(
      get_logger(),
      "Unitree adapter backend=%s input=%s control_rate=%.1fHz timeout=%.2fs",
      backend_->name().c_str(),
      topics_input_cmd_.c_str(),
      control_rate_hz_,
      cmd_timeout_sec_);
  }

  void cmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    latest_cmd_time_ = now();
    latest_cmd_valid_ = isFinite(*msg);
    if (latest_cmd_valid_) {
      latest_cmd_ = fromTwist(*msg);
      latest_cmd_.vyaw *= yaw_rate_sign_;
    } else {
      latest_cmd_ = {};
    }
    has_latest_cmd_ = true;
  }

  void handleSetEnabled(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response)
  {
    if (!backend_ready_) {
      response->success = false;
      response->message = backend_error_.empty() ? "backend is not ready" : backend_error_;
      return;
    }

    if (request->data) {
      if (enabled_) {
        response->success = true;
        response->message = "already enabled";
        return;
      }
      enabled_ = true;
      has_latest_cmd_ = false;
      latest_cmd_valid_ = true;
      manual_stop_latched_ = false;
      stop_sent_ = true;
      last_limited_cmd_ = {};
      last_sent_cmd_ = {};
      last_sent_cmd_time_.reset();
      transitionTo(AdapterState::READY, "enabled; waiting for fresh /cmd_vel");
      response->success = true;
      response->message = "enabled";
    } else {
      enabled_ = false;
      manual_stop_latched_ = false;
      stopSent("disabled by service", false);
      transitionTo(AdapterState::DISABLED, "disabled by service");
      response->success = true;
      response->message = "disabled";
    }
  }

  void handleStop(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    (void)request;
    enabled_ = false;
    manual_stop_latched_ = true;
    const bool ok = stopSent("manual stop service", true);
    transitionTo(AdapterState::STOPPED, "manual stop service");
    response->success = ok;
    response->message = ok ? "stopped and disabled" : "StopMove failed";
  }

  void handleDamp(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    (void)request;
    enabled_ = false;
    manual_stop_latched_ = true;
    const int32_t ret =
      backend_ready_ ? callBackend("Damp", [this]() {return backend_->damp();}) : -1;
    transitionTo(AdapterState::DAMPING, "damp service");
    response->success = sdkReturnOk(ret);
    response->message = response->success ? "damping requested" : "Damp failed";
  }

  void handleBalanceStand(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    (void)request;
    const int32_t ret = backend_ready_ ?
      callBackend("BalanceStand", [this]() {return backend_->balanceStand();}) : -1;
    response->success = sdkReturnOk(ret);
    response->message = response->success ? "balance stand requested" : "BalanceStand failed";
  }

  void handleRecoveryStand(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    (void)request;
    const int32_t ret = backend_ready_ ?
      callBackend("RecoveryStand", [this]() {return backend_->recoveryStand();}) : -1;
    response->success = sdkReturnOk(ret);
    response->message = response->success ? "recovery stand requested" : "RecoveryStand failed";
  }

  void handleBodyHeight(
    const std::shared_ptr<unitree_adapter::srv::SetBodyHeight::Request> request,
    std::shared_ptr<unitree_adapter::srv::SetBodyHeight::Response> response)
  {
    response->accepted_height_m = 0.0;
    if (!body_height_service_enabled_) {
      response->success = false;
      response->message = "body height service disabled";
      return;
    }
    if (!backend_ready_) {
      response->success = false;
      response->message = backend_error_.empty() ? "backend is not ready" : backend_error_;
      return;
    }
    const double height_m = request->height_m;
    if (!std::isfinite(height_m)) {
      response->success = false;
      response->message = "height_m is not finite";
      return;
    }
    if (height_m < body_height_min_m_ || height_m > body_height_max_m_) {
      response->success = false;
      std::ostringstream message;
      message << "height_m " << height_m << " outside safe range ["
              << body_height_min_m_ << ", " << body_height_max_m_ << "]";
      response->message = message.str();
      return;
    }

    if (body_height_clear_velocity_on_request_) {
      has_latest_cmd_ = false;
      latest_cmd_ = {};
      last_limited_cmd_ = {};
      last_sent_cmd_ = {};
      last_sent_cmd_time_.reset();
    }
    if (body_height_stop_before_call_) {
      stopSent("body height service pre-stop", true);
    }

    const int32_t ret = callBackend(
      "BodyHeight",
      [this, height_m]() {return backend_->bodyHeight(height_m);});
    const bool ok = sdkReturnOk(ret);
    response->success = ok;
    response->accepted_height_m = ok ? height_m : 0.0;
    response->message = ok ? "body height requested" : "BodyHeight failed";
    if (ok) {
      last_body_height_command_m_ = height_m;
      transitionTo(enabled_ ? AdapterState::READY : AdapterState::DISABLED, "body height service");
    }
  }

  void controlTimer()
  {
    const rclcpp::Time stamp = now();

    if (!backend_ready_) {
      publishStatusIfNeeded(stamp, true);
      return;
    }

    if (!enabled_) {
      if (!manual_stop_latched_) {
        const bool stopped = stopSent("disabled", false);
        if (stopped || state_ != AdapterState::SDK_ERROR) {
          transitionTo(AdapterState::DISABLED, "disabled");
        }
      }
      publishStatusIfNeeded(stamp, false);
      return;
    }

    if (!has_latest_cmd_) {
      const bool stopped = stopSent("waiting for first /cmd_vel", false);
      if (stopped || state_ != AdapterState::SDK_ERROR) {
        transitionTo(AdapterState::READY, "waiting for first /cmd_vel");
      }
      publishStatusIfNeeded(stamp, false);
      return;
    }

    if (!latest_cmd_valid_) {
      const bool stopped = stopSent("invalid /cmd_vel", false);
      if (stopped || state_ != AdapterState::SDK_ERROR) {
        transitionTo(AdapterState::INVALID_CMD, "invalid /cmd_vel");
      }
      publishStatusIfNeeded(stamp, false);
      return;
    }

    const double cmd_age_sec = (stamp - *latest_cmd_time_).seconds();
    if (cmd_age_sec > cmd_timeout_sec_) {
      const bool stopped = stopSent("stale /cmd_vel", false);
      if (stopped || state_ != AdapterState::SDK_ERROR) {
        transitionTo(AdapterState::CMD_TIMEOUT, "stale /cmd_vel");
      }
      publishStatusIfNeeded(stamp, false);
      return;
    }

    const auto target = applyClampAndDeadband(latest_cmd_, limits_);
    const double dt_sec = last_sent_cmd_time_ ? (stamp - *last_sent_cmd_time_).seconds() :
      (1.0 / control_rate_hz_);
    const auto limited = applyAccelerationLimit(target, last_limited_cmd_, limits_, dt_sec);
    last_limited_cmd_ = limited;
    last_sent_cmd_ = applyClampAndDeadband(limited, limits_);
    last_sent_cmd_time_ = stamp;

    if (zero_command_is_stop_ && isZeroCommand(last_sent_cmd_)) {
      maybeRepeatStopForZeroCommand(stamp);
      transitionTo(AdapterState::READY, "fresh zero command");
    } else {
      const int32_t ret = callBackend("Move", [this]() {return backend_->move(last_sent_cmd_);});
      if (sdkReturnOk(ret)) {
        stop_sent_ = false;
        transitionTo(
          AdapterState::ACTIVE,
          "streaming Move(" + commandToString(last_sent_cmd_) + ")");
      }
    }

    last_cmd_pub_->publish(toTwist(last_sent_cmd_));
    publishStatusIfNeeded(stamp, false);
  }

  template<typename Fn>
  int32_t callBackend(const std::string & operation, Fn && fn)
  {
    std::lock_guard<std::mutex> lock(backend_call_mutex_);
    int32_t ret = -1;
    try {
      ret = fn();
    } catch (const std::exception & ex) {
      backend_error_ = operation + " threw: " + ex.what();
      ret = -1;
    }

    last_sdk_return_code_ = ret;
    if (!sdkReturnOk(ret)) {
      ++consecutive_sdk_errors_;
      backend_error_ = operation + " failed with code " + std::to_string(ret) +
        " (" + sdkReturnDescription(ret) + ")";
      RCLCPP_WARN(
        get_logger(),
        "%s failed with code %d (%s, consecutive=%d)",
        operation.c_str(),
        ret,
        sdkReturnDescription(ret).c_str(),
        consecutive_sdk_errors_);
      if (consecutive_sdk_errors_ >= max_consecutive_sdk_errors_) {
        enabled_ = false;
        manual_stop_latched_ = false;
        transitionTo(AdapterState::SDK_ERROR, operation + " repeated failures");
      }
    } else {
      consecutive_sdk_errors_ = 0;
      backend_error_.clear();
    }
    return ret;
  }

  bool stopSent(const std::string & reason, bool force)
  {
    if (!backend_ready_) {
      return false;
    }
    if (stop_sent_ && !force) {
      last_reason_ = reason;
      return true;
    }
    const int32_t ret = callBackend("StopMove", [this]() {return backend_->stopMove();});
    stop_sent_ = sdkReturnOk(ret);
    last_stop_time_ = now();
    last_limited_cmd_ = {};
    last_sent_cmd_ = {};
    last_sent_cmd_time_ = last_stop_time_;
    last_cmd_pub_->publish(toTwist({}));
    last_reason_ = reason;
    return sdkReturnOk(ret);
  }

  void maybeRepeatStopForZeroCommand(const rclcpp::Time & stamp)
  {
    const bool should_repeat =
      !last_stop_time_ ||
      (stamp - *last_stop_time_).seconds() >= zero_stop_repeat_period_sec_;
    if (should_repeat) {
      stopSent("fresh zero command", false);
    }
  }

  void transitionTo(AdapterState state, const std::string & reason)
  {
    if (state != state_) {
      RCLCPP_INFO(
        get_logger(),
        "state %s -> %s: %s",
        stateName(state_),
        stateName(state),
        reason.c_str());
      state_ = state;
      publish_state_now_ = true;
    }
    last_reason_ = reason;
  }

  void publishStatusIfNeeded(const rclcpp::Time & stamp, bool force)
  {
    const bool time_due =
      !last_status_publish_time_ ||
      (stamp - *last_status_publish_time_).seconds() >= (1.0 / publish_status_rate_hz_);
    if (!force && !publish_state_now_ && !time_due) {
      return;
    }

    std_msgs::msg::Int32 state_msg;
    state_msg.data = static_cast<int32_t>(state_);
    state_pub_->publish(state_msg);

    std_msgs::msg::String status_msg;
    std::ostringstream json;
    json << "{"
         << "\"state\":\"" << stateName(state_) << "\","
         << "\"enabled\":" << (enabled_ ? "true" : "false") << ","
         << "\"backend\":\"" << (backend_ ? backend_->name() : "none") << "\","
         << "\"last_return_code\":" << last_sdk_return_code_ << ","
         << "\"last_return_text\":\"" << sdkReturnDescription(last_sdk_return_code_) << "\","
         << "\"consecutive_errors\":" << consecutive_sdk_errors_ << ","
         << "\"last_body_height_command_m\":" << last_body_height_command_m_ << ","
         << "\"last_reason\":\"" << last_reason_ << "\""
         << "}";
    status_msg.data = json.str();
    status_pub_->publish(status_msg);

    last_status_publish_time_ = stamp;
    publish_state_now_ = false;
  }

  std::string topics_input_cmd_;
  std::string topics_last_cmd_;
  std::string topics_state_;
  std::string topics_status_;

  std::string services_set_enabled_;
  std::string services_stop_;
  std::string services_damp_;
  std::string services_balance_stand_;
  std::string services_recovery_stand_;
  std::string services_body_height_;

  BackendConfig backend_config_;
  CommandLimits limits_;
  double control_rate_hz_{50.0};
  double cmd_timeout_sec_{0.25};
  double publish_status_rate_hz_{5.0};
  bool zero_command_is_stop_{true};
  double zero_stop_repeat_period_sec_{0.25};
  int max_consecutive_sdk_errors_{3};
  bool body_height_service_enabled_{false};
  double body_height_min_m_{0.30};
  double body_height_max_m_{0.50};
  bool body_height_stop_before_call_{true};
  bool body_height_clear_velocity_on_request_{true};
  double last_body_height_command_m_{0.0};

  std::unique_ptr<SportClientBackend> backend_;
  std::mutex backend_call_mutex_;
  bool backend_ready_{false};
  std::string backend_error_;

  bool enabled_{false};
  bool manual_stop_latched_{false};
  bool stop_sent_{true};
  AdapterState state_{AdapterState::DISABLED};
  bool publish_state_now_{true};
  std::string last_reason_{"startup"};
  int32_t last_sdk_return_code_{0};
  int consecutive_sdk_errors_{0};

  bool has_latest_cmd_{false};
  bool latest_cmd_valid_{true};
  double yaw_rate_sign_{1.0};
  VelocityCommand latest_cmd_;
  std::optional<rclcpp::Time> latest_cmd_time_;
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
  rclcpp::CallbackGroup::SharedPtr cmd_callback_group_;
  rclcpp::CallbackGroup::SharedPtr control_callback_group_;
  rclcpp::CallbackGroup::SharedPtr service_callback_group_;
};

}  // namespace unitree_adapter

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<unitree_adapter::UnitreeAdapterNode>(rclcpp::NodeOptions{});
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
