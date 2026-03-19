# RoboNeuron Structure and Roadmap

## 1. Project Definition

RoboNeuron should be understood as a **ROS2-native embodied AI middleware layer**, not as a training framework, not as a hardware driver stack, and not as a collection of unrelated adapters.

Its job is to connect:

- perception inputs
- VLA inference
- action/control interpretation
- ROS2 command publication
- MCP-based capability exposure

The project is valuable when it makes that chain stable, inspectable, and easy to integrate into real robots, simulation stacks, and agent systems.

## 2. Core Boundaries

RoboNeuron should own:

- perception/VLA/control runtime composition
- canonical ROS message boundaries
- action protocol handling
- MCP-facing tool exposure
- lightweight configuration and deployment glue

RoboNeuron should not own:

- low-level robot drivers such as `libfranka`
- heavyweight external model source trees beyond vendored runtime dependencies
- benchmark-specific environment code inside the core control path
- local reference projects as part of the main architecture

## 3. Core Runtime Modules

The project should remain centered on a **small set of strong runtime modules**.

### `perception_server`

Responsibilities:

- start and stop camera ROS nodes
- manage one or more camera streams
- publish image topics for downstream consumers

This module should stay focused on sensor bring-up and ROS publishing.

### `vla_server`

Responsibilities:

- subscribe to required observation topics
- assemble model-facing observations
- run VLA inference
- publish action outputs such as `EEFDeltaCommand` or `RawActionChunk`

This module should remain the single center for model-facing observation and inference orchestration.

### `control_server`

Responsibilities:

- subscribe to action and state feedback topics
- interpret incoming action protocols
- resolve them into robot command messages
- publish ROS2 control outputs

This module should be the control execution host, not a thin wrapper around a single IK library.

### MCP / CLI Layer

Responsibilities:

- expose the runtime modules as callable MCP services
- generate tool servers from ROS message definitions where useful
- provide stable entrypoints for OpenClaw and other orchestration layers

This layer should expose capabilities, not absorb core control logic.

## 4. Supporting Layers

### `adapters/`

`adapters/` should stay narrow.

It is the right place for:

- camera wrappers
- VLA wrappers

It is **not** the right place for per-robot control logic. Robot differences affect control semantics, state feedback, and command output, so they belong to the control subsystem rather than a generic adapter layer.

### `runtime/`

`runtime/` should host process boundaries and client/worker protocols for heavyweight model stacks such as OpenVLA and OpenVLA-OFT.

### `utils/`

`utils/` is acceptable for small shared helpers, but it should not become a generic dumping ground. Today, `control_runtime.py` is already more than a generic utility file; it is effectively an internal control kernel. That is acceptable for now, but it should be treated as **internal control support code**, not as random utility code.

### `ros/roboneuron_interfaces`

This is the canonical ROS boundary of the project and is one of the strongest parts of the current structure.

## 5. How Robot Adaptation Should Work

Robot adaptation should not be modeled as “one adapter per robot”.

The preferred model is:

- one generic control runtime
- robot-specific configuration profiles
- small internal control special cases only when necessary

Most ROS2 + URDF robots should be integrated through:

- URDF path
- state feedback topic
- command topic
- command message type
- end-effector link and base link
- gripper semantics

Only robots that truly need special resolution logic should require extra code.

## 6. How Benchmark and Environment Integration Should Work

Simulation and benchmark ecosystems such as LIBERO or CALVIN should not be mixed into the core control runtime.

They should be treated as **integration surfaces**:

- benchmark observation adapters
- action/step/reset bridges
- environment startup helpers

This keeps the core of RoboNeuron small while still making environment integration explicit and first-class.

## 7. OpenClaw as a First-Class Direction

OpenClaw should be treated as one of RoboNeuron’s strategic integrations, not as an afterthought.

That means:

- repo-scoped MCP launcher config belongs with stable project config
- OpenClaw-specific skills and references should stay grouped under `openclaw/`
- the relationship between RoboNeuron and OpenClaw should be documented, not implied through scattered scripts

## 8. Current Structural Assessment

### What is already working well

- the runtime spine is visibly centered on `perception`, `vla`, and `control`
- ROS message contracts are explicit and first-party
- heavy VLA runtime code is isolated from the main environment
- OpenClaw now has a clearer path to becoming a first-class integration

### What still needs tightening

- naming and docstrings still need to stay aligned with the current architecture
- `utils/` is carrying increasingly domain-specific control code
- top-level documentation must keep pace with the actual structure
- integration-specific assets should remain grouped and intentional

## 9. Practical Next Steps

### Near term

- keep consolidating the project around the three runtime servers
- continue using ROS2 as the execution boundary
- treat OpenClaw as a formal integration target
- avoid reintroducing per-robot adapter abstractions

### Mid term

- add robot profile configuration for real and simulated robots
- define a clean integration path for benchmark environments such as LIBERO and CALVIN
- tighten the distinction between core runtime code and integration glue

### Longer term

- turn RoboNeuron into a stable execution substrate for embodied agents
- support both direct tool-driven control and closed-loop VLA pipelines
- keep the core small while allowing ecosystem integrations to grow around it

## 10. Final Position

RoboNeuron’s value is not that it owns every layer of robotics.

Its value is that it provides a **coherent middle layer**:

- between perception and action
- between VLA models and ROS2 execution
- between robot capabilities and MCP/agent orchestration

If the project keeps its core small, its boundaries explicit, and its integrations intentional, it has real long-term value.
