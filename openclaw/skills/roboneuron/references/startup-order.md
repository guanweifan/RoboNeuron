RoboNeuron MCP startup order for OpenClaw:

1. Perception path
- Start `roboneuron-perception` before any workflow that needs RGB input.

2. Cartesian control path
- Start `roboneuron-control` before using `roboneuron-eef-delta` in closed-loop robot workflows.

3. VLA path
- Start `roboneuron-perception`.
- Start `roboneuron-control`.
- Start `roboneuron-vla`.
- Stop in reverse order when tearing down if the workflow is complete.

4. Base motion path
- Use `roboneuron-twist` independently for `/cmd_vel` publishing tasks.
