# RoboNeuron Development Roadmap

## Document Role

- [product-roadmap.md](./product-roadmap.md) defines the product phases and product shapes.
- This document defines the engineering work required to reach those product phases.
- The current phase is expanded in detail.
- Later phases keep a lighter structure: objective, main workstreams, and exit criteria.

## Current Phase Snapshot

- Active phase: **Phase 2 — Manipulation Runtime**
- Engineering objective: **Converge RoboNeuron into an explainable, repeatable, and observable manipulation execution runtime**
- Canonical lane: `perception -> VLA -> RawActionChunk -> edge control -> execution -> StateSnapshot`
- Explicitly out of scope for the current phase:
  - A second robot
  - A second model path
  - Workflow
  - Evaluation
  - Data systems
  - Generalized abstractions built for hypothetical future needs

## Current Baseline

The codebase already has several foundations in place:

- A visible execution spine across `perception`, `vla`, and `control`
- A first kernel baseline around runtime contracts, session, health, and profile
- A first `core / edge / backends` split
- Edge-side control runtime and local resolving
- Local and distributed smoke coverage for the main runtime lane

The main gaps are now about **semantic closure**, **observability**, and **repeatable runtime behavior**, not about adding more unrelated features.

## Phase 2 — Detailed Workstreams

### P0 — Semantic Closure

- [ ] Freeze `MotionIntent` as the single semantic core inside the manipulation runtime
- [ ] Freeze `RawActionChunk` as the model-facing transport contract
- [ ] Freeze `EEFDeltaCommand` as the debug / manual transport contract
- [ ] Make every manipulation control path pass through `MotionIntent`

### P1 — Canonical Lane Closure

- [ ] Prove the real distributed lane: `core -> /raw_action_chunk -> edge -> robot`
- [ ] Prove `OpenVLA / OFT -> RawActionChunk -> edge -> execution`
- [ ] Settle on one blessed manipulation demo
- [ ] Settle on one blessed robot / model / runtime combination

### P2 — Observability

- [ ] Introduce structured logging
- [ ] Make execution trace directly useful for debugging
- [ ] Make session state visible during runs
- [ ] Make health state actionable during runs
- [ ] Capture the critical evidence of a run: input, action, state, and outcome

### P3 — Repeatability

- [ ] Make the blessed demo repeatable
- [ ] Separate model failures from runtime failures and robot / vendor failures
- [ ] Establish a minimal troubleshooting path

### P4 — Delivery Surface

- [ ] Write the minimal quickstart for the blessed demo
- [ ] Stabilize the current thin host / launch surface
- [ ] Clarify what belongs to RoboNeuron and what belongs to the vendor stack

## Phase 2 — Current Debt

### Must Be Solved in This Phase

- [ ] Action semantics are still split across multiple representations
- [ ] Transport and semantic core are not yet formally separated
- [ ] The real distributed lane still lacks final hard evidence
- [ ] Logging, trace, session, and health are not yet strong enough
- [ ] The demo story is not yet fully converged into one blessed path

### Intentionally Deferred

- [ ] Second robot backend
- [ ] Second model path
- [ ] Workflow, evaluation, and data systems
- [ ] Forward-looking platform abstractions that are not required to finish Phase 2

## Phase 2 — Exit Criteria

Phase 2 is complete only when all of the following are true:

- [ ] The `MotionIntent`-centered execution semantics are frozen
- [ ] The roles of `RawActionChunk` and `EEFDeltaCommand` are frozen
- [ ] The canonical lane has been proven on a real distributed path
- [ ] The blessed manipulation demo is repeatable, observable, and diagnosable
- [ ] Session, trace, health, and logging have become practical debugging tools
- [ ] The minimal quickstart for the blessed demo is usable
- [ ] Runtime and vendor boundaries are explicitly accepted

## Phase 3 — Deployable Runtime

### Objective

Turn the current runtime into an execution system that can be installed, started, observed, and recovered.

### Main Workstreams

- [ ] Install, setup, and bring-up
- [ ] Start, stop, status, and restart
- [ ] Profiles for local and hybrid deployment
- [ ] Watchdog and health handling
- [ ] Structured logs and traces
- [ ] Docker and packaging

### Exit Criteria

- [ ] A user can install and start the system without reading the source
- [ ] Both single-node and split deployment paths run successfully
- [ ] Failures can be observed, diagnosed, and recovered

## Phase 4 — Adapter Platform

### Objective

Make new robots and new models primarily an adapter problem instead of a core rewrite.

### Main Workstreams

- [ ] Integrate a second robot
- [ ] Integrate a second model path
- [ ] Define adapter contracts
- [ ] Add conformance tests
- [ ] Publish a supported configuration matrix

### Exit Criteria

- [ ] The same task works across two robot / model combinations
- [ ] Adding a robot or model mainly changes adapters rather than the core runtime

## Phase 5 — Workflow Layer

### Objective

Build task orchestration on top of the Runtime layer.

### Main Workstreams

- [ ] Workflow schema
- [ ] Skill / job executor
- [ ] Retry, approval, and scheduling

### Exit Criteria

- [ ] Multi-step tasks become first-class objects
- [ ] The same task can be re-run, retried, and managed systematically

## Phase 6 — Evaluation Layer

### Objective

Build a proper run evaluation surface.

### Main Workstreams

- [ ] Replay engine
- [ ] Metrics, reports, and run registry
- [ ] Benchmark adapters
- [ ] Regression gating

### Exit Criteria

- [ ] Core changes can be captured by evaluation results
- [ ] The same task can be compared across models and runtime configurations

## Phase 7 — Data Layer

### Objective

Build a runtime-to-data feedback loop.

### Main Workstreams

- [ ] Session recording
- [ ] Dataset export
- [ ] Lineage and provenance
- [ ] Failure mining
- [ ] Training and deployment connectors

### Exit Criteria

- [ ] A run can enter a usable data loop
- [ ] Runtime outputs remain traceable to runtime conditions and model versions

## Execution Order

1. Finish **Phase 2 P0 + P1** before adding more feature surface.
2. Then finish **Phase 2 P2 + P3** so the runtime becomes genuinely debuggable.
3. Finish **Phase 2 P4** to form a stable delivery surface.
4. **Do not enter Phase 3 before Phase 2 actually exits.**
