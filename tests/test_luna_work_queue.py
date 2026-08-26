#!/usr/bin/env python3

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "luna_work_queue", ROOT / "tools" / "luna_work_queue.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        state_path = Path(temporary) / "queue.json"
        initial = {
            "ready": ["newer", "older"], "waiting": [], "completed": [],
            "running": [], "blocked": [], "wait_reasons": {},
            "state_since": {"newer": 20, "older": 10}, "turn_counts": {},
        }
        state_path.write_text(json.dumps(initial) + "\\n")
        state, repaired = MODULE.load_state(state_path, recover=True)
        assert repaired
        assert MODULE.claim(state, None, "luna-1", 30) == "older"
        assert state["running"][0]["agent"] == "luna-1"
        assert state["running"][0]["lease_deadline"] == 930
        assert state["turn_counts"]["older"] == 1
        MODULE.heartbeat(state, "older", "luna-1", 35, 60)
        assert state["running"][0]["lease_deadline"] == 95
        MODULE.transition(state, "older", "luna-1", "waiting", "child C1", 40)
        assert state["waiting"] == ["older"]
        assert state["wait_reasons"]["older"]["event"] == "child C1"
        MODULE.write_state(state_path, state)
        decoded = json.loads(state_path.read_text())
        assert decoded["waiting"] == ["older"]
        assert not state_path.read_text().endswith("\\n")
        decoded["waiting"] = []
        decoded["ready"] = ["older"]
        decoded["state_since"]["older"] = 50
        state_path.write_text(json.dumps(decoded) + "\n")
        command = [
            sys.executable, str(ROOT / "tools" / "luna_work_queue.py"),
            "--state", str(state_path), "claim", "--oldest", "--agent",
            "luna-cli",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        claimed = json.loads(result.stdout)
        assert claimed["logical"] == "older"
        assert claimed["agent"] == "luna-cli"
    print("luna work queue tests: PASS")


if __name__ == "__main__":
    main()
