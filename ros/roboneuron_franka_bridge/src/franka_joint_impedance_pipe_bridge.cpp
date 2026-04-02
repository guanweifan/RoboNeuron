// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <franka/active_control.h>
#include <franka/exception.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

namespace {

constexpr std::array<double, 7> kJointImpedance{{3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0,
                                                  2000.0}};
constexpr std::array<double, 6> kCartesianImpedance{{3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0}};

void set_default_behavior(franka::Robot& robot) {
  robot.setCollisionBehavior(
      {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
      {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
      {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
      {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
      {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
      {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
      {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
      {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}});
  robot.setJointImpedance(kJointImpedance);
  robot.setCartesianImpedance(kCartesianImpedance);
}

class PipeBridge {
 public:
  PipeBridge(std::string robot_ip, double state_rate_hz)
      : robot_ip_(std::move(robot_ip)),
        state_publish_period_sec_(1.0 / state_rate_hz) {
    if (robot_ip_.empty()) {
      throw std::invalid_argument("robot_ip must not be empty");
    }
    if (state_rate_hz <= 0.0) {
      throw std::invalid_argument("state_rate_hz must be positive");
    }
  }

  int run() {
    std::thread input_thread([this]() { input_loop(); });
    std::thread state_thread([this]() { state_output_loop(); });
    control_loop();
    if (input_thread.joinable()) {
      input_thread.join();
    }
    if (state_thread.joinable()) {
      state_thread.join();
    }
    return 0;
  }

 private:
  void input_loop() {
    std::string line;
    while (running_.load(std::memory_order_acquire) && std::getline(std::cin, line)) {
      if (line.empty()) {
        continue;
      }
      if (line == "QUIT") {
        running_.store(false, std::memory_order_release);
        break;
      }
      if (line.rfind("SET ", 0) != 0) {
        std::cerr << "ERR unsupported command" << std::endl;
        continue;
      }

      std::istringstream stream(line.substr(4));
      std::array<double, 7> target{};
      for (size_t index = 0; index < target.size(); ++index) {
        if (!(stream >> target[index])) {
          std::cerr << "ERR invalid SET payload" << std::endl;
          goto next_line;
        }
      }

      {
        std::lock_guard<std::mutex> lock(target_mutex_);
        target_q_ = target;
        has_target_ = true;
      }
      if (!first_command_logged_) {
        std::cerr << "bridge:received_first_target" << std::endl;
        first_command_logged_ = true;
      }
    next_line:
      continue;
    }
    running_.store(false, std::memory_order_release);
  }

  void control_loop() {
    try {
      franka::Robot robot(robot_ip_);
      robot.automaticErrorRecovery();
      set_default_behavior(robot);
      std::cerr << "bridge:robot_connected" << std::endl;

      while (running_.load(std::memory_order_acquire)) {
        try {
          run_control_session(robot);
        } catch (const franka::ControlException& ex) {
          std::cerr << "bridge:control_exception " << ex.what() << std::endl;
          if (!running_.load(std::memory_order_acquire)) {
            break;
          }
          try {
            robot.automaticErrorRecovery();
            std::cerr << "bridge:recovered" << std::endl;
          } catch (const franka::Exception& recovery_ex) {
            std::cerr << "bridge:recovery_failed " << recovery_ex.what() << std::endl;
            running_.store(false, std::memory_order_release);
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
      }
    } catch (const franka::Exception& ex) {
      std::cerr << "bridge:start_failed " << ex.what() << std::endl;
      running_.store(false, std::memory_order_release);
    }
  }

  void run_control_session(franka::Robot& robot) {
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      previous_velocity_command_.fill(0.0);
      previous_velocity_acceleration_.fill(0.0);
    }
    robot.control([this](const franka::RobotState& state,
                         franka::Duration period) -> franka::JointVelocities {
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_q_ = state.q;
        current_dq_ = state.dq;
        state_ready_ = true;
      }

      std::array<double, 7> desired_target{};
      {
        std::lock_guard<std::mutex> lock(target_mutex_);
        if (!reference_initialized_) {
          target_q_ = state.q;
          reference_q_ = state.q;
          reference_initialized_ = true;
        }
        desired_target = has_target_ ? target_q_ : reference_q_;
        update_reference_towards_locked(desired_target, period.toSec());
      }

      std::array<double, 7> raw_velocity{};
      for (size_t index = 0; index < raw_velocity.size(); ++index) {
        raw_velocity[index] =
            kVelocityServoGains_[index] * (reference_q_[index] - state.q_d[index]);
      }

      auto upper_velocity_limits = franka::computeUpperLimitsJointVelocity(state.q);
      auto lower_velocity_limits = franka::computeLowerLimitsJointVelocity(state.q);
      auto limited_velocity = franka::limitRate(
          upper_velocity_limits,
          lower_velocity_limits,
          kVelocityAccelerationLimits_,
          franka::kMaxJointJerk,
          raw_velocity,
          previous_velocity_command_,
          previous_velocity_acceleration_);

      const double dt = std::max(period.toSec(), 1e-3);
      for (size_t index = 0; index < limited_velocity.size(); ++index) {
        previous_velocity_acceleration_[index] =
            (limited_velocity[index] - previous_velocity_command_[index]) / dt;
        previous_velocity_command_[index] = limited_velocity[index];
      }

      franka::JointVelocities velocities(limited_velocity);
      if (!running_.load(std::memory_order_acquire)) {
        return franka::MotionFinished(velocities);
      }
      return velocities;
    }, franka::ControllerMode::kJointImpedance, true);
  }

  void state_output_loop() {
    while (running_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::duration<double>(state_publish_period_sec_));
      publish_state_snapshot();
    }
  }

  void publish_state_snapshot() {
    std::array<double, 7> q{};
    std::array<double, 7> dq{};
    bool ready = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      q = current_q_;
      dq = current_dq_;
      ready = state_ready_;
    }
    if (!ready) {
      return;
    }

    std::ostringstream line;
    line << std::setprecision(17) << "STATE";
    for (double value : q) {
      line << ' ' << value;
    }
    for (double value : dq) {
      line << ' ' << value;
    }
    std::cout << line.str() << std::endl;
  }

  void update_reference_towards_locked(const std::array<double, 7>& desired_target,
                                       double period_sec) {
    constexpr double kMinStepSec = 1e-3;
    const double dt = std::max(period_sec, kMinStepSec);
    for (size_t index = 0; index < reference_q_.size(); ++index) {
      const double max_step = kReferenceVelocityLimits_[index] * dt;
      const double delta = desired_target[index] - reference_q_[index];
      const double clipped_delta = std::clamp(delta, -max_step, max_step);
      reference_q_[index] += clipped_delta;
    }
  }

  std::string robot_ip_;
  double state_publish_period_sec_;
  std::atomic_bool running_{true};

  const std::array<double, 7> kReferenceVelocityLimits_{{0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7}};
  const std::array<double, 7> kVelocityServoGains_{{6.0, 6.0, 6.0, 5.0, 5.0, 4.0, 4.0}};
  const std::array<double, 7> kVelocityAccelerationLimits_{{1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5}};

  std::mutex target_mutex_;
  std::array<double, 7> target_q_{{0, 0, 0, 0, 0, 0, 0}};
  std::array<double, 7> reference_q_{{0, 0, 0, 0, 0, 0, 0}};
  std::array<double, 7> previous_velocity_command_{{0, 0, 0, 0, 0, 0, 0}};
  std::array<double, 7> previous_velocity_acceleration_{{0, 0, 0, 0, 0, 0, 0}};
  bool has_target_{false};
  bool reference_initialized_{false};
  bool first_command_logged_{false};

  std::mutex state_mutex_;
  std::array<double, 7> current_q_{{0, 0, 0, 0, 0, 0, 0}};
  std::array<double, 7> current_dq_{{0, 0, 0, 0, 0, 0, 0}};
  bool state_ready_{false};
};

}  // namespace

int main(int argc, char** argv) {
  std::string robot_ip;
  double state_rate_hz = 100.0;

  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--robot-ip" && index + 1 < argc) {
      robot_ip = argv[++index];
      continue;
    }
    if (arg == "--state-rate-hz" && index + 1 < argc) {
      state_rate_hz = std::stod(argv[++index]);
      continue;
    }
    std::cerr << "Unknown argument: " << arg << std::endl;
    return 2;
  }

  try {
    PipeBridge bridge(robot_ip, state_rate_hz);
    return bridge.run();
  } catch (const std::exception& ex) {
    std::cerr << "bridge:fatal " << ex.what() << std::endl;
    return 1;
  }
}
