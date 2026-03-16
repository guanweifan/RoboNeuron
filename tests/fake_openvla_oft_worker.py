from __future__ import annotations

import json
import sys

from roboneuron_core.runtime.openvla_oft_protocol import decode_observation_from_transport


def _emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    _emit(
        {
            "event": "ready",
            "device": "cpu",
            "dtype": "float32",
            "norm_keys": ["vr_banana"],
            "robot_platform": "bridge",
            "num_images_in_input": 1,
            "use_proprio": True,
            "use_film": True,
            "use_l1_regression": True,
            "use_diffusion": False,
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request = json.loads(line)
        request_id = request["id"]
        method = request["method"]
        params = request.get("params", {})

        if method == "predict_action":
            observation = decode_observation_from_transport(params.get("observation", {}))
            images = observation.get("images")
            if images is None:
                images = [observation[key] for key in observation if "image" in key]

            state = observation.get("state") or observation.get("proprio") or []
            action = [[float(len(images)), float(len(state)), float(bool(params.get("unnorm_key")))]]
            _emit({"id": request_id, "ok": True, "result": {"action": action}})
        elif method == "shutdown":
            _emit({"id": request_id, "ok": True, "result": {"status": "bye"}})
            break
        else:
            _emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error": {"type": "ValueError", "message": f"Unsupported method: {method}"},
                }
            )


if __name__ == "__main__":
    main()
