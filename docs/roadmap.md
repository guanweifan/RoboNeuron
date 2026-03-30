# RoboNeuron Roadmap

## 1. Project Positioning

### 1.1 Mission Statement
RoboNeuron is the "Execution Hub" for Embodied AI. It bridges the gap between high-level Agents/VLAs and low-level hardware (ROS2, simulation, and real robots), ensuring that complex model outputs are translated into stable, deterministic physical actions.

### 1.2 Core Pillars
- **Execution Abstraction**: A unified layer connecting Agents, VLAs, perception, and control logic.
- **Unified Runtime**: A standardized environment for orchestrating models, robot backends, and deployment modes.
- **Modular Evolution**: Starting with a rock-solid Execution Substrate, then expanding into evaluation, workflows, and ecosystem connectors.

### 1.3 Current Focus
- **Hardening the Core**: Prioritize a stable, well-defined execution kernel over feature bloat.
- **Closing the Loop**: Validate the entire architecture through a single, end-to-end "Canonical Path."
- **Validation through Scaling**: Verify abstractions by adding new hardware and models only after the core is stable.

## 2. Development Milestones

### Phase A — Foundation & Kernel Definition
**Goal**
Define the system boundaries and core primitives to prevent architectural drift.

**Deliverable**
A formal Kernel Specification and a Minimal API to serve as the development baseline.

**Key Focus**
- Standardize core primitives: State, Action, Session, and Health.
- Establish clear boundaries between interfaces, core, edge, and backends.
- Ship a functional schema, focusing on "good enough" interfaces to unblock development.

### Phase B — End-to-End MVP (The Canonical Path)
**Goal**
Close the loop on a real hardware/simulation setup to prove the execution logic is sound.

**Deliverable**
A Minimal Viable Product (MVP) capable of executing requests from a model down to the hardware.

**Key Focus**
- Implement the primary lane: remote core + local edge + Franka + OpenVLA/OFT.
- Validate the pipeline: RawActionChunk -> State Alignment -> Local Resolving -> Execution.
- Basic operational tooling: Start/Stop, telemetry, and minimal logging.

### Phase C — Productionization & Deployment
**Goal**
Transition from "research code" to a deployable, observable, and maintainable system.

**Deliverable**
A production-ready Execution Substrate that is easy to install and monitor.

**Key Focus**
- Robustness features: system profiles, health watchdogs, and structured logging.
- Deployment flexibility: support for both "single-node" and "hybrid (cloud core + edge)" setups.
- Standardized packaging: Dockerization, installation scripts, and comprehensive bring-up docs.

### Phase D — Scalability & Abstraction Validation
**Goal**
Test the modularity of the system by introducing heterogeneous hardware and models.

**Deliverable**
A general-purpose runtime proven to handle diverse robot/model configurations.

**Key Focus**
- Integrate a second robot backend, such as SO-101, to test interface universality.
- Support a second model/runtime path, such as ACT or OpenPI.
- Metric of success: can we add new hardware and models without touching the core logic?

### Phase E — Ecosystem & Capability Expansion
**Goal**
Scale vertically by adding orchestration, evaluation, and data pipeline capabilities.

**Deliverable**
A platform that integrates execution with benchmarking and automated workflows.

**Key Focus**
- Advanced telemetry: replay, trace reports, and automated evaluation.
- Benchmark adapters: native support for LIBERO and CALVIN.
- Data and training connectors: streamline the data collection-to-deployment loop.

## 3. Task Checklist

### Phase A — Hardening the Core
- [x] Define `StateSnapshot` and `ActionContract` primitives.
- [x] Architect `ExecutionSession` lifecycle management.
- [ ] Enforce module boundaries: interfaces, core, edge, backends.
- [x] Draft schema: lifecycle, event, trace, profile.
- [ ] Finalize the Kernel Spec as the source of truth.

### Phase B — Closing the Loop
- [ ] Implement the core -> edge -> backend communication link.
- [ ] Develop the Franka backend driver.
- [ ] Build the local resolver and control runtime.
- [ ] Connect OpenVLA/OFT to output `RawActionChunk`.
- [ ] Deploy a thin entry via OpenClaw.
- [ ] Release the Quickstart Guide.

### Phase C — Engineering Excellence
- [ ] Formalize profiles: simulation, robot, backend, deployment.
- [ ] Implement system-wide watchdog and health monitoring.
- [ ] Build a structured event-logging system.
- [ ] Optimize Docker images and deployment playbooks.
- [ ] Support hybrid cloud/edge deployment architectures.

### Phase D — Generalization
- [ ] Integrate SO-101, or another second robot backend.
- [ ] Support ACT / OpenPI inference paths.
- [ ] Validate core stability during hardware and model migration.
- [ ] Refactor non-generic code identified during scaling.

### Phase E — Platform Features
- [ ] Implement the workflow/task plane.
- [ ] Add an evaluation plane with benchmarking adapters.
- [ ] Develop data connectors for training and collection feedback loops.
- [ ] Integrate vLLM / SGLang optimization hooks.

## 4. Current Phase Expansion

### 4.1 Active Phase
**Phase A — Foundation & Kernel Definition**

### 4.2 Current Baseline
- The project already has a visible execution spine around perception, VLA, and control.
- OpenVLA / OpenVLA-OFT runtime isolation is already in place.
- `RawActionChunk`-based control and local resolution already exist in prototype form.
- OpenClaw currently serves as a thin entry layer rather than a kernel dependency.

### 4.3 Phase A Expanded Task List

**A. Runtime Primitives**
- [x] Define `StateSnapshot`  as the canonical runtime-facing state object.
- [x] Define `ActionContract` , with `RawActionChunk` as the first validated contract.
- [x] Define `ExecutionSession` , including lifecycle ownership and minimal session states.
- [x] Define `HealthStatus`  for kernel-level runtime health reporting.

**B. Ownership Boundaries**
- [ ] Freeze the ownership boundaries of `roboneuron_interfaces`, `roboneuron_core`, `roboneuron_edge`, and `roboneuron_backends`.
- [ ] Clarify what belongs to RoboNeuron versus vendor stacks such as `franka_ros2`.
- [ ] Clarify which concepts are kernel objects and which are deferred capability-plane objects.

**C. Schema Baseline**
- [x] Publish the kernel schema baseline covering lifecycle, event, trace, and profile drafts.
- [ ] Align the runtime schema with the canonical path instead of abstract future workflows.
- [ ] Keep the schema minimal and sufficient to unblock Phase B.

**D. Canonical Path Definition**
- [ ] Formalize the canonical path: remote core + local edge + Franka + OpenVLA/OFT + `RawActionChunk`.
- [ ] Define the ownership of local state alignment and local resolving on edge.
- [ ] Define the minimum operational surface required for Phase B validation.

**E. Kernel Spec**
- [ ] Publish the first Kernel Spec as the source of truth for the next development cycle.
- [ ] Use the Kernel Spec to gate Phase B implementation choices and prevent architectural drift.

### 4.4 Exit Criteria for Phase A
- [ ] `StateSnapshot`, `ActionContract`, `ExecutionSession`, and `HealthStatus` are frozen.
- [ ] Module ownership across `interfaces / core / edge / backends` is explicit and accepted.
- [ ] The canonical path is formally defined and accepted as the Phase B target.
- [ ] The Kernel Spec is published and becomes the baseline for subsequent implementation.

### 4.5 Next Step
- Publish the Kernel Spec and schema baseline as the Phase A source of truth, then use it to freeze edge ownership for local state alignment and local resolving before Phase B validation.
