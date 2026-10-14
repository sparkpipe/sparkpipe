#!/usr/bin/env python3
"""Hub work that grows stays off the hub's root filesystem: the release kit's work root and every API engine log live on the workspace volume."""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane  # noqa: E402

WORKSPACE = "/srv/workspace/"
failures = []

lib = (ROOT / "scripts" / "rollover_lib.sh").read_text()
work = re.search(r"^HUB_WORK=\$\{HUB_WORK:-([^}]+)\}$", lib, re.M)
if work is None or not work.group(1).startswith(WORKSPACE):
    failures.append(f"scripts/rollover_lib.sh: HUB_WORK defaults to {work.group(1) if work else 'nothing'}, not a path on {WORKSPACE}")


def log_target(command):
    targets = re.findall(r">>\s*(\S+)\s+2>&1", command)
    return targets[0] if len(targets) == 1 else None


for name, model in stage_lane.MODELS.items():
    entry = stage_lane.swap_entry(name, model, 0)
    target = log_target(entry["api_start"])
    if target is None or not target.startswith(WORKSPACE):
        failures.append(f"tools/stage_lane.py {name}: api_start logs to {target!r}, not a file on {WORKSPACE}")

for model in json.loads((ROOT / "deployment" / "hub" / "model_swap.json").read_text())["models"]:
    target = log_target(model["api_start"])
    if target is None or not target.startswith(WORKSPACE):
        failures.append(f"deployment/hub/model_swap.json {model['id']}: api_start logs to {target!r}, not a file on {WORKSPACE}")

for line in failures:
    print("FAIL " + line)
if failures:
    raise SystemExit(1)
print(f"PASS test_hub_paths_on_workspace: the release kit's hub work root and {len(stage_lane.MODELS)} generated and every deployed API engine log are on the workspace volume")
