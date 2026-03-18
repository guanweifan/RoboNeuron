#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_roboneuron_mcp_common.sh"

source_ros_humble
source_roboneuron_ros_workspace "${PROJECT_ROOT}"
run_roboneuron_mcp "roboneuron-mcp-control" "${PROJECT_ROOT}"
