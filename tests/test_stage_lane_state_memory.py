#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane  # noqa: E402


def render(name, directory, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/stage_lane.py"), "--model", name, "--lane", "9",
         "--runtime-root", "/home/{host}/lane9", "--weightd-socket", "/run/weightd.sock",
         "--output-dir", str(directory), *extra],
        capture_output=True, text=True)


def main():
    failures = []
    with tempfile.TemporaryDirectory() as temporary:
        for name, model in sorted(stage_lane.MODELS.items()):
            out = pathlib.Path(temporary) / name
            result = render(name, out)
            if result.returncode != 0:
                failures.append(f"{name}: render failed: {result.stderr.strip()[-200:]}")
                continue
            adapters = [json.loads(path.read_text()) for path in sorted(out.glob("adapter.*.json"))]
            floor = (2 * adapters[0]["resident_capacity"] + 2) * model["recurrent_bytes"]
            for adapter in adapters:
                if model["recurrent_bytes"] and adapter.get("state_memory_bytes", 0) < floor:
                    failures.append(f"{name}: state_memory_bytes {adapter.get('state_memory_bytes')} below the binding's {floor}")
                if not model["recurrent_bytes"] and "state_memory_bytes" in adapter:
                    failures.append(f"{name}: a model without recurrent state names state_memory_bytes")
            if model["recurrent_bytes"]:
                short = render(name, pathlib.Path(temporary) / f"{name}-short", "--state-memory-bytes", str(floor - 1))
                if short.returncode == 0 or "recurrent checkpoints" not in short.stderr:
                    failures.append(f"{name}: state memory below two checkpoints per sequence plus two was accepted")
    for line in failures:
        print("FAILED: " + line)
    if failures:
        return 1
    print(f"PASS {len(stage_lane.MODELS)} stage lanes give recurrent-state records the memory the KV binding needs and no key to stateless models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
