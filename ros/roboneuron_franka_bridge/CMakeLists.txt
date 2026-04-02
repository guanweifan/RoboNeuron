cmake_minimum_required(VERSION 3.8)
project(roboneuron_franka_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_path(
  ROBONEURON_LIBFRANKA_INCLUDE_DIR
  NAMES franka/robot.h
  HINTS
    "$ENV{HOME}/franka_ros2_ws/install/libfranka/include"
    /usr/local/include
    /usr/include
)

find_library(
  ROBONEURON_LIBFRANKA_LIBRARY
  NAMES franka libfranka
  HINTS
    "$ENV{HOME}/franka_ros2_ws/install/libfranka/lib"
    /usr/local/lib
    /usr/lib
    /usr/lib/x86_64-linux-gnu
)

if(NOT ROBONEURON_LIBFRANKA_INCLUDE_DIR)
  message(FATAL_ERROR "Could not locate libfranka headers (expected franka/robot.h).")
endif()

if(NOT ROBONEURON_LIBFRANKA_LIBRARY)
  message(FATAL_ERROR "Could not locate the libfranka shared library.")
endif()

add_executable(franka_joint_impedance_pipe_bridge
  src/franka_joint_impedance_pipe_bridge.cpp
)

target_compile_features(franka_joint_impedance_pipe_bridge PUBLIC cxx_std_17)
target_include_directories(
  franka_joint_impedance_pipe_bridge
  PRIVATE
  "${ROBONEURON_LIBFRANKA_INCLUDE_DIR}"
)
target_link_libraries(
  franka_joint_impedance_pipe_bridge
  "${ROBONEURON_LIBFRANKA_LIBRARY}"
)

install(TARGETS
  franka_joint_impedance_pipe_bridge
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
