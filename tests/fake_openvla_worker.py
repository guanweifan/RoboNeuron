from __future__ import annotations

import json
import sys


def _emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    _emit({"event": "ready", "device": "cpu", "dtype": "float32", "norm_keys": ["bridge_orig"]})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request = json.loads(line)
        request_id = request["id"]
        method = request["method"]
        params = request.get("params", {})

        if method == "predict_action":
            instruction = params.get("instruction", "")
            unnorm_key = params.get("unnorm_key")
            action = [[float(len(instruction)), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if unnorm_key else 0.0]]
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
