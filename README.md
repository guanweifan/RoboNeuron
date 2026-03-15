# RoboNeuron: A Modular Framework Linking Foundation Models and ROS for Embodied AI

[![arXiv](https://img.shields.io/badge/arXiv-2512.10394-b31b1b.svg)](https://arxiv.org/abs/2512.10394)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROS](https://img.shields.io/badge/ROS-Humble-blue.svg)](https://docs.ros.org/en/humble/)

<p align="center">
  <img src="./assets/logo.png" width="500"
       style="border: 1px solid #ccc; border-radius: 8px;">
</p>

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation & Configuration](#installation--configuration)
- [Demo Showcase](#demo-showcase)
- [Extending RoboNeuron](#extending-roboneuron)
- [Citation](#citation)

## Project Overview

**RoboNeuron** is a modular **middleware for embodied AI systems** that bridges high-level intelligence (e.g., LLMs, VLA models, and skill policies) with **ROS 2-based robotic execution**. It is designed for researchers and developers who want to integrate perception, inference, and control modules without tightly coupling model logic to robot-specific runtime details.

<p align="center">
  <img src="./assets/stack.png" width="60%"
       style="border: 1px solid #ccc; border-radius: 8px;">
</p>

<p align="center"><em>RoboNeuron as the middleware layer between embodied intelligence and ROS 2 execution.</em></p>

At its core, RoboNeuron separates **agent-facing orchestration** from **ROS 2 data transport**, enabling two common execution patterns under one unified framework:

- **Direct tool-to-ROS execution** for low-latency robot commands
- **Closed-loop perception → VLA inference → control pipelines** for embodied tasks

<p align="center">
  <img src="./assets/framework.png" width="70%"
       style="border: 1px solid #ccc; border-radius: 8px;">
</p>

<p align="center"><em>Unified architecture for capability exposure, orchestration, and ROS-backed execution.</em></p>

### Highlights

- **Modular by design**: perception, VLA, control, and robot backends can be developed and replaced independently
- **ROS-native execution**: integrates with existing ROS 2 topics, messages, controllers, and simulators
- **VLA-ready infrastructure**: supports heterogeneous VLA backends through adapter-oriented integration
- **Tool-based capability exposure**: maps robot functions into structured callable interfaces for agentic workflows
- **Deployment-friendly**: supports both simulation and real-world robotic systems under the same abstraction

In short, RoboNeuron is a **reusable infrastructure layer** for building, testing, and deploying embodied AI pipelines on top of ROS 2.

## Directory Structure

```text
roboneuron/
├── README.md                     # Project documentation
├── pyproject.toml
├── uv.lock
├── src/
│   └── roboneuron_core/          # Core implementation
│       ├── cli/                  # Command/entry modules
│       ├── servers/              # MCP server implementations
│       │   └── generated/        # Generated MCP server modules
│       ├── adapters/             # Camera/robot/VLA adapters
│       ├── types/                # Typed config/data models
│       └── utils/                # Reusable utilities
├── tests/                        # Flat test suite (unit/integration via pytest markers)
├── docs/                         # Project documentation
├── configs/                      # Configuration files
│   ├── vla_models.json           # VLA model paths and configurations
│   └── vla_accel_presets.json    # Inference acceleration presets
├── templates/                    # Templates for generating new MCP tools
├── assets/                       # README assets
├── urdf/                         # Robot description files
│   ├── panda.urdf                # Franka Panda robot URDF
│   └── fr3.urdf                  # Franka Research 3 robot URDF
├── ros/
│   └── roboneuron_interfaces/    # First-party ROS interface package
└── third_party/
    └── vla_src/                  # Vendored VLA source trees
```

## Installation & Configuration

### Prerequisites

- **Python 3.10+**
- **ROS 2 Humble** (recommended)
- **UV** Python package manager
- **CLine** (VS Code Extension): Required for acting as the MCP Client to orchestrate the AI models and tools.

### Step 1: Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Set Up ROS 2 Environment

```bash
# Install ROS 2 Humble (Ubuntu 22.04)
# Follow official instructions: https://docs.ros.org/en/humble/Installation.html

# Source ROS 2 environment
source /opt/ros/humble/setup.bash
```

### Step 3: Clone and Install RoboNeuron

```bash
# Clone the repository
git clone https://github.com/guanweifan/RoboNeuron.git
cd roboneuron

# Install dependencies
uv sync
```

### Step 4: Build the ROS Interface Workspace

The unified `EEFDeltaCommand` message is defined in `ros/roboneuron_interfaces`. Build it locally before starting MCP services that publish or consume task-space actions.

```bash
cd ros
rosdep install --from-paths roboneuron_interfaces -i -y
colcon build --packages-select roboneuron_interfaces --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
cd ..
```

### Step 5: Configure MCP Servers for Cline

To enable the Cline extension to communicate with RoboNeuron, add the following configuration to your MCP settings file.
Note: Please replace /home/user/roboneuron with the absolute path to your cloned repository.

```json
{
  "mcpServers": {
    "roboneuron-perception": {
    "autoApprove": [],
    "disabled": false,
    "timeout": 60,
    "type": "stdio",
    "command": "bash",
    "args": [
      "-c",
      "source /opt/ros/humble/setup.bash && uv --directory /home/user/roboneuron run roboneuron-mcp-perception"
    ],
    "cwd": "/home/user/roboneuron"
    },
    "roboneuron-vla": {
    "autoApprove": [],
    "disabled": false,
    "timeout": 60,
    "type": "stdio",
    "command": "bash",
    "args": [
      "-c",
      "source /opt/ros/humble/setup.bash && source /home/user/roboneuron/ros/install/setup.bash && uv --directory /home/user/roboneuron run roboneuron-mcp-vla"
    ],
    "cwd": "/home/user/roboneuron"
    },
    "roboneuron-control": {
    "autoApprove": [],
    "disabled": false,
    "timeout": 60,
    "type": "stdio",
    "command": "bash",
    "args": [
      "-c",
      "source /opt/ros/humble/setup.bash && source /home/user/roboneuron/ros/install/setup.bash && uv --directory /home/user/roboneuron run roboneuron-mcp-control"
    ],
    "cwd": "/home/user/roboneuron"
    },
    "roboneuron-simulation": {
      "autoApprove": [],
      "disabled": false,
      "timeout": 60,
      "type": "stdio",
      "command": "bash",
      "args": [
        "-c",
        "source /opt/ros/humble/setup.bash && source /home/user/roboneuron/ros/install/setup.bash && uv --directory /home/user/roboneuron run roboneuron-mcp-simulation"
      ],
      "cwd": "/home/user/roboneuron"
    },
    "roboneuron-twist": {
    "autoApprove": [],
    "disabled": false,
    "timeout": 60,
    "type": "stdio",
    "command": "bash",
    "args": [
      "-c",
      "source /opt/ros/humble/setup.bash && uv --directory /home/user/roboneuron run roboneuron-mcp-twist"
    ],
    "cwd": "/home/user/roboneuron"
    },
    "roboneuron-eef-delta": {
    "autoApprove": [],
    "disabled": false,
    "timeout": 60,
    "type": "stdio",
    "command": "bash",
    "args": [
      "-c",
      "source /opt/ros/humble/setup.bash && source /home/user/roboneuron/ros/install/setup.bash && uv --directory /home/user/roboneuron run roboneuron-mcp-eef-delta"
    ],
    "cwd": "/home/user/roboneuron"
    }
  }
}
```

### Step 6: Set Up the Dedicated OpenVLA Runtime

OpenVLA now runs in its own Python environment instead of the main RoboNeuron environment. This keeps the primary `uv sync` environment minimal and isolates model-specific dependencies such as `transformers`, `flash-attn`, and vendored `prismatic`.

Create the runtime with:

```bash
bash scripts/setup_openvla_runtime.sh
```

By default this creates `.venvs/openvla` and installs:

- `torch==2.2.0`, `torchvision==0.17.0`, `torchaudio==2.2.0` from the CUDA 11.8 PyTorch index
- the minimal OpenVLA inference dependencies
- the local `flash_attn` wheel if `flash_attn-*.whl` exists at the repository root

### Step 7: Configure VLA Models

Edit `configs/vla_models.json` to specify model checkpoints and runtime configuration. String values are still supported for simple models, but OpenVLA should use the richer object form:

```json
{
  "openvla": {
    "path": "checkpoints/openvla/openvla-7b",
    "runtime_python": ".venvs/openvla/bin/python",
    "attn_implementation": "flash_attention_2",
    "default_unnorm_key": "bridge_orig",
    "runtime_startup_timeout_sec": 900
  },
  "openvla-oft": "/path/to/your/openvla-oft/model",
  "pi0": "/path/to/your/pi0/model"
}
```

When `model_path` is omitted from `start_vla_inference`, RoboNeuron resolves both the checkpoint path and the runtime-specific kwargs from this config file.

### Optional Components Added in Current Mainline

- `src/roboneuron_core/adapters/robot/calvin_adapter.py`: CALVIN adapter (optional runtime dependency).
- `src/roboneuron_core/adapters/vla/dummy_vla.py`: lightweight pipeline test model.
- `src/roboneuron_core/adapters/vla/pi0.py`: OpenPI (`pi0`) wrapper (optional runtime dependency).

### CLI Entrypoints

Canonical service entrypoints are:

- `uv run roboneuron-mcp-perception`
- `source ros/install/setup.bash && uv run roboneuron-mcp-vla`
- `source ros/install/setup.bash && uv run roboneuron-mcp-control`
- `source ros/install/setup.bash && uv run roboneuron-mcp-simulation`
- `uv run roboneuron-mcp-twist`
- `source ros/install/setup.bash && uv run roboneuron-mcp-eef-delta`

## Demo Showcase

### Case I: Unified Control of Heterogeneous Vehicles

`Instruction: "Make the car move forward at a speed of 0.5m/s"`

<div align="center">
    <img src="assets/isaac_vechcle_demo.gif" width="650" style="max-width: 100%; border: 1px solid #ccc; border-radius: 8px;">
</div>


---

### Case II: Kinematic-Aware Manipulation in Simulation

`Instruction: "Move the robotic arm gripper forward at a speed of 0.1m/s"`

<div align="center">
    <img src="assets/isaac_franka_demo.gif" width="650" style="max-width: 100%; border: 1px solid #ccc; border-radius: 8px;">
</div>

---

### Case III: Real-World VLA-Driven Object Grasping

`Instruction: "Using RealSense camera and OpenVLA model, to pick up the blue bowl."`

<div align="center">
    <img src="assets/franka_vla_demo.gif" width="650" style="max-width: 100%; border: 1px solid #ccc; border-radius: 8px;">
</div>

---

## Extending RoboNeuron

### Adding New Camera Wrappers

1. Create a new file in `src/roboneuron_core/adapters/camera/`
2. Inherit from `CameraWrapper` base class
3. Implement required methods:
   ```python
   class NewCameraWrapper(CameraWrapper):
       def open(self):
           # Initialize camera connection
           pass
       
       def read(self):
           # Capture and return image
           pass
       
       def close(self):
           # Clean up resources
           pass
   ```
4. Register in `src/roboneuron_core/adapters/camera/__init__.py`

### Integrating New VLA Models

1. Create wrapper in `src/roboneuron_core/adapters/vla/`
2. Inherit from `ModelWrapper` base class
3. Implement model loading and inference:
   ```python
   class NewVLAWrapper(ModelWrapper):
       def load(self, model_path, **kwargs):
           # Load model weights and configuration
           pass
       
       def predict_action(self, image, instruction):
           # Generate action from image and instruction
           pass
   ```
4. Add model configuration to `configs/vla_models.json`

### Supporting New Robot Platforms

1. Create adapter in `src/roboneuron_core/adapters/robot/`
2. Implement platform-specific communication
3. Provide URDF or kinematic description
4. Test with simulation before hardware deployment

### Registering a Custom ROS 2 Message

If you create a new ROS message file under your project directory, for example: `ros/roboneuron_interfaces/msg/Test.msg`, you must rebuild and source the ROS 2 workspace so the new message type becomes available to ROS and your MCP tools.

```bash
# Navigate to your workspace
cd ros

# Install dependencies
rosdep install --from-paths roboneuron_interfaces -i -y

# Build the workspace
colcon build --symlink-install

# Source the workspace
source install/setup.bash
```


### Creating Custom MCP Tools

Use the template system to generate new MCP tools:

```bash
# Generate new MCP tool template
uv run python -m roboneuron_core.cli.mcp_tool_generator your_topic your_msg
```


## Citation

If you use this project, please cite the paper:
```
@misc{guan2025roboneuronmodularframeworklinking,
      title={RoboNeuron: A Modular Framework Linking Foundation Models and ROS for Embodied AI}, 
      author={Weifan Guan and Huasen Xi and Chenxiao Zhang and Aosheng Li and Qinghao Hu and Jian Cheng},
      year={2025},
      eprint={2512.10394},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.10394}, 
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---
