# RoboNeuron Project Status and Usage (2026-02-28)

## 1. Document Goal

This document describes the current repository state, including:

- Core architecture and module responsibilities
- Main directories and file purposes
- Functional scope and boundaries
- Usage flows (development, runtime, testing)
- Version-control and maintenance guidance

> Note: The repository has completed the structural refactor. Core implementation lives in `src/roboneuron_core/`.

---

## 2. Current Architecture Overview

The core package now uses a pragmatic, function-oriented layout under `src/roboneuron_core/`:

- `cli`: command entrypoints and generation utilities
- `servers`: MCP service implementations
- `adapters`: camera/robot/VLA integrations and registries
- `types`: pure configuration and data model definitions
- `utils`: reusable helpers (config, logging, process, message parsing)

Design principles:

- Keep module responsibilities obvious and focused
- Avoid layering for its own sake
- Keep extension points explicit via adapter registries

---

## 3. Repository Structure and Purpose

### 3.1 Key Root Files

- `README.md`
  - Public-facing overview (project intro, quick start, MCP setup, extension points)
- `pyproject.toml`
  - Python project config (dependencies, script entrypoints, ruff/mypy/pytest)
- `uv.lock`
  - Locked dependency graph
- `.gitignore`
  - Ignore rules (cache, ROS build artifacts, large local model files)

### 3.2 `src/roboneuron_core/` (Core Implementation)

- `src/roboneuron_core/cli/`
  - `mcp_entrypoints.py`: script dispatcher for all MCP command entrypoints
  - `mcp_tool_generator.py`: template-based MCP tool generator

- `src/roboneuron_core/servers/`
  - `perception_server.py`: camera stream publishing MCP server
  - `vla_server.py`: VLA inference MCP server
  - `control_server.py`: IK control MCP server
  - `simulation_server.py`: simulation-driven MCP server
  - `generated/twist_server.py`: generated Twist MCP server
  - `generated/eef_delta_server.py`: generated EEF delta MCP server

- `src/roboneuron_core/adapters/`
  - `camera/`: camera adapters and registry
  - `robot/`: robot/simulation adapters and registry
  - `vla/`: VLA model adapters and registry

- `src/roboneuron_core/types/`
  - Type-only config and data models

- `src/roboneuron_core/utils/`
  - `msg_parser.py`: ROS message schema parsing
  - `process.py`: multi-process lifecycle helpers
  - `config.py`: config loading helpers
  - `logging.py`: logging initialization helpers

### 3.3 Assets and Config Directories

- `configs/`
  - `vla_models.json`: model-name to model-path mapping
  - `vla_accel_presets.json`: inference acceleration presets
- `templates/`
  - MCP tool generation templates
- `assets/`
  - README/media assets
- `urdf/`
  - Robot URDFs (for example `panda.urdf`, `fr3.urdf`)

### 3.4 ROS and Third-Party Code

- `ros/roboneuron_interfaces/`
  - First-party ROS custom message package (`EEFDeltaCommand.msg`)
- `third_party/vla_src/`
  - Upstream/external VLA source code

### 3.5 Tests

- `tests/`
  - Flat test suite with marker-based grouping

---

## 4. Current Functional Coverage

The project currently exposes 6 MCP service entry commands:

1. `roboneuron-mcp-perception`
2. `roboneuron-mcp-vla`
3. `roboneuron-mcp-control`
4. `roboneuron-mcp-simulation`
5. `roboneuron-mcp-twist`
6. `roboneuron-mcp-eef-delta`

These are registered in `pyproject.toml` under `[project.scripts]` and dispatched by `roboneuron_core.cli.mcp_entrypoints`.

---

## 5. How To Use

### 5.1 Install Dependencies

```bash
uv sync
```

### 5.2 Prepare ROS Environment

```bash
source /opt/ros/humble/setup.bash
```

Build and source the custom interface workspace before using services that publish or consume `EEFDeltaCommand` (`vla`, `control`, `simulation`, `eef_delta`):

```bash
cd /home/guanweifan/RoboNeuron/ros
rosdep install --from-paths roboneuron_interfaces -i -y
colcon build --packages-select roboneuron_interfaces --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

Then source the generated workspace:

```bash
source /home/guanweifan/RoboNeuron/ros/install/setup.bash
```

### 5.3 Start MCP Services (Examples)

```bash
uv run roboneuron-mcp-perception
uv run roboneuron-mcp-vla
uv run roboneuron-mcp-control
uv run roboneuron-mcp-simulation
uv run roboneuron-mcp-twist
uv run roboneuron-mcp-eef-delta
```

### 5.4 Quality Checks

```bash
uv run ruff check .
uv run mypy
uv run python -m unittest discover -s tests -p "test_*.py"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests -q
```

---

## 6. Version-Control Guidance (Aligned with `.gitignore`)

Include in Git:

- Source: `src/roboneuron_core/`, `tests/`, `configs/`, `templates/`, `urdf/`
- Docs: `README.md`, `docs/`
- Assets: `assets/`
- Project config: `pyproject.toml`, `uv.lock`, `.gitignore`
- ROS custom interface source: `ros/roboneuron_interfaces/`

Do not include in Git:

- Python cache: `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`
- Virtual envs: `.venv/`, `venv/`
- Build artifacts: `build/`, `dist/`, `*.egg-info/`
- ROS build artifacts: `ros/build/`, `ros/install/`, `ros/log/`
- Logs/temp files: `*.log`, `tmp/`
- Local model weights and large files: `*.pt`, `*.pth`, `*.safetensors`, `*.onnx`
- Local IDE config: `.vscode/`, `.idea/`

---

## 7. Current Status Summary

- Main architecture migrated to `src/roboneuron_core`
- Unified command entrypoints: `roboneuron-mcp-*`
- Placeholder directories have been removed
- `roboneuron_interfaces` is managed as first-party code, `vla_src` as third-party code
- Quality gates (ruff/mypy/pytest) are runnable
