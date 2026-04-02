# RoboNeuron Product Roadmap

## Current Position

- Active phase: **Phase 2 — Manipulation Runtime**
- Status: **In Progress**
- Canonical lane: `perception -> VLA -> RawActionChunk -> edge control -> execution -> StateSnapshot`
- Current release channel: **pre-release**

RoboNeuron is evolving through three product layers:

1. **Bridge**: AI-to-ROS capability exposure
2. **Runtime**: stable embodied execution
3. **Platform**: workflow, evaluation, and data systems built on top of the runtime

The current priority is to finish the Runtime layer before expanding the Platform layer.

## Phase 1 — Bridge Kit

- Status: **Foundational shape completed, still maintained**
- Product shape: **AI-to-ROS bridge kit**
- Primary user: Developers who already have a ROS 2 environment and want AI to call robot capabilities
- Core capabilities:
  - AI-facing ROS 2 tools and resources
  - Basic quickstart
  - Host configuration and launch surfaces
  - At least two runnable bridge demos
- Representative demos:
  - `Twist` mobile robot control
  - Typed arm control
- Scope boundary:
  - Solves capability bridging only
  - Does not promise stable embodied execution
  - Does not include deployment operations
  - Does not include multi-step task orchestration
- Phase complete when:
  - A new developer can run at least two bridge demos from the docs without reading the source

## Phase 2 — Manipulation Runtime

- Status: **Current phase**
- Product shape: **A single-lane manipulation execution runtime**
- Primary user: Researchers and engineers who want to connect Agent or VLA outputs to a real robot execution path
- Core capabilities:
  - One blessed manipulation lane
  - Stable action contracts
  - Stable state feedback
  - Local resolving on edge
  - Repeatable single-task execution
  - Runtime surfaces that are genuinely useful for debugging
- Representative demo:
  - One blessed pick-and-place task or an equivalent manipulation task
- Scope boundary:
  - Supports one blessed robot / model / runtime combination
  - Does not optimize for multi-robot support yet
  - Does not optimize for deployment and operations yet
  - Does not include workflow, evaluation, or data capabilities
- Phase complete when:
  - The blessed manipulation lane is stable
  - The blessed demo is repeatable
  - Runs are observable, explainable, and debuggable
- Release channel on completion:
  - **internal alpha**

## Phase 3 — Deployable Runtime

- Status: **Next phase**
- Product shape: **An execution system that can be installed, started, observed, and recovered**
- Primary user: Engineers who want to deploy RoboNeuron into a lab setup or onto real devices
- Core capabilities:
  - Install, setup, and bring-up
  - Start, stop, status, and restart
  - Local and hybrid deployment modes
  - Logs, health, and traces
  - Failure diagnosis and recovery
- Representative demos:
  - Bring-up from a clean environment
  - Both single-node and split core/edge deployment paths run successfully
- Scope boundary:
  - Still limited to a small set of blessed combinations
  - Not yet a multi-robot or multi-model platform
- Phase complete when:
  - A user can install and start the system without reading the source
  - Failures can be observed, diagnosed, and recovered
- Release channel on completion:
  - **public alpha**

## Phase 4 — Adapter Platform

- Status: **Later phase**
- Product shape: **A reusable execution substrate across robots and models**
- Primary user: Teams that want to reuse the same execution base across different robots and embodied models
- Core capabilities:
  - A second robot integration
  - A second model path
  - Adapter contracts
  - A supported configuration matrix
- Representative demo:
  - The same task runs on two robot / model combinations with minimal core changes
- Scope boundary:
  - Focuses on adapter reuse, not workflow, evaluation, or data systems
- Phase complete when:
  - Adding a new robot or model is primarily an adapter task rather than a core rewrite
- Release channel on completion:
  - **beta / 1.0 candidate**

## Phase 5 — Workflow Layer

- Status: **Later phase**
- Product shape: **A task orchestration layer built on top of the runtime**
- Primary user: Teams moving from single commands to reusable multi-step tasks
- Core capabilities:
  - Workflow schemas
  - Skill and job composition
  - Retry, approval, and scheduling
- Representative demo:
  - A full task covering perception, decision, grasp, place, verification, and retry
- Scope boundary:
  - Built on top of the Runtime layer; it does not replace it
- Phase complete when:
  - Multi-step tasks become first-class objects instead of ad hoc prompt-and-tool chains

## Phase 6 — Evaluation Layer

- Status: **Later phase**
- Product shape: **A run evaluation layer**
- Primary user: Teams iterating on prompts, models, and runtime behavior
- Core capabilities:
  - Replay
  - Run reports
  - Metrics and experiment tracking
  - Benchmark adapters
- Representative demo:
  - The same task can be compared systematically across models or runtime configurations
- Scope boundary:
  - Built on top of a stable Runtime layer
  - Does not own long-term data assets
- Phase complete when:
  - Core changes can be captured by evaluation results

## Phase 7 — Data Layer

- Status: **Later phase**
- Product shape: **A data feedback layer for embodied execution**
- Primary user: Teams that want to turn runtime outputs into training, analysis, and optimization assets
- Core capabilities:
  - Session recording
  - Dataset export
  - Lineage and provenance
  - Failure mining
  - Training and deployment connectors
- Representative demo:
  - A run can be recorded, exported, and fed into a training or analysis loop
- Scope boundary:
  - Built on top of stable Runtime and Evaluation layers
- Phase complete when:
  - Runtime outputs can enter a usable data loop
  - Data assets remain traceable to runtime conditions and model versions

## Release Ladder

- Current codebase: **pre-release**
- Phase 2 complete: **internal alpha**
- Phase 3 complete: **public alpha**
- Phase 4 complete: **beta / 1.0 candidate**
- RoboNeuron 1.0 means:
  - **A deployable, reusable, and extensible execution platform**
