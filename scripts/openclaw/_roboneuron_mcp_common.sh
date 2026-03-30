#!/usr/bin/env bash
set -euo pipefail

roboneuron_project_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/../.." && pwd
}

source_ros_runtime() {
  set +u
  if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
  elif [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
  else
    echo "No supported ROS 2 setup.bash found under /opt/ros." >&2
    return 1
  fi
  set -u
}

source_ros_humble() {
  source_ros_runtime
}

source_roboneuron_ros_workspace() {
  local project_root="${1:?project root required}"
  set +u
  source "${project_root}/ros/install/setup.bash"
  set -u
}

run_roboneuron_mcp() {
  local entrypoint="${1:?entrypoint required}"
  local project_root="${2:?project root required}"
  shift 2

  export PYTHONUNBUFFERED=1
  exec uv --directory "${project_root}" run "$@" "${entrypoint}"
}
