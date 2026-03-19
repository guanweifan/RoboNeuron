RoboNeuron + OpenClaw troubleshooting:

- If `mcporter list` or `mcporter call` fails before tool discovery, verify:
  - `mcporter --config /home/guanweifan/RoboNeuron/configs/openclaw/mcporter.json list`
  - `mcporter daemon status`
  - `/home/guanweifan/RoboNeuron/ros/install/setup.bash` exists

- If a RoboNeuron server becomes unresponsive:
  - `mcporter daemon restart --log`
  - Retry the same `mcporter --config ... list <server> --schema` command first

- If ROS message imports fail:
  - Rebuild and source the RoboNeuron ROS workspace under `ros/install`

- If VLA startup is slow:
  - Expect the first load to take much longer than simple publishers
  - Verify the model/runtime environment separately before blaming OpenClaw
