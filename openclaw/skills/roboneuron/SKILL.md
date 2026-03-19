---
name: roboneuron
description: Control RoboNeuron through its project-scoped MCP servers via mcporter.
metadata: {"openclaw":{"emoji":"🤖","os":["linux"],"requires":{"bins":["mcporter"]}}}
---

Use this skill when the task involves controlling RoboNeuron through its MCP servers from OpenClaw.

Scope:
- Robot perception/control/VLA bring-up through RoboNeuron MCP.
- Inspecting available RoboNeuron tools with `mcporter`.
- Starting or stopping the RoboNeuron MCP daemon/runtime paths used by OpenClaw.

Required command form:
- Always invoke `mcporter` with `--config /home/guanweifan/RoboNeuron/configs/openclaw/mcporter.json`.
- Never rely on editor-imported MCP configs for RoboNeuron.

Startup order:
1. For camera-only workflows, use `roboneuron-perception`.
2. For motion/control workflows, start `roboneuron-control` before sending end-effector commands.
3. For instruction-conditioned policy execution, start `roboneuron-perception`, then `roboneuron-control`, then `roboneuron-vla`.
4. Use `roboneuron-twist` only for `/cmd_vel` style base motion publishing.
5. Use `roboneuron-eef-delta` for direct end-effector delta commands.

Useful commands:
- `mcporter --config /home/guanweifan/RoboNeuron/configs/openclaw/mcporter.json list`
- `mcporter --config /home/guanweifan/RoboNeuron/configs/openclaw/mcporter.json list roboneuron-control --schema`
- `mcporter daemon status`
- `mcporter daemon restart --log`

Rules:
- Treat RoboNeuron MCP servers as stateful and persistent. Do not replace them with ad-hoc `--stdio` calls unless explicitly debugging.
- Reuse configured server names (`roboneuron-perception`, `roboneuron-vla`, `roboneuron-control`, `roboneuron-twist`, `roboneuron-eef-delta`).
- If a tool call fails after a crash or stale connection, restart the daemon with `mcporter daemon restart`.

References:
- `references/startup-order.md`
- `references/troubleshooting.md`
