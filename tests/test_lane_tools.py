#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPARE = ROOT / "tools" / "release_compare.py"
LANE_CONFIG = ROOT / "tools" / "lane_config.py"


def compare(directory, old, new):
    paths = []
    for name, document in (("old.json", old), ("new.json", new)):
        path = directory / name
        if document is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(json.dumps(document))
        paths.append(str(path))
    return subprocess.run([sys.executable, str(COMPARE), *paths], capture_output=True, text=True, timeout=60).returncode


def render(record, directory, environment):
    subprocess.run([sys.executable, str(LANE_CONFIG), "render", record, str(directory)], env=environment, check=True, capture_output=True, timeout=300)
    return {path.relative_to(directory).as_posix(): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}


def main():
    failures = []
    capture = {"capital": {"tokens": [1, 2, 3], "logprobs": [-0.1, -0.2, -0.3]}, "code": {"tokens": [4, 5], "logprobs": [-0.5, -0.6]}}
    drift = json.loads(json.dumps(capture))
    drift["code"]["logprobs"][1] = -0.61
    diverged = json.loads(json.dumps(capture))
    diverged["code"]["tokens"][1] = 9
    missing = {"capital": capture["capital"]}
    with tempfile.TemporaryDirectory() as scratch:
        directory = pathlib.Path(scratch)
        for old, new, wanted, why in ((capture, capture, 0, "identical captures"), (capture, drift, 0, "equal tokens with logprob drift"),
                                      (capture, diverged, 1, "a token that differs"), (capture, missing, 1, "a prompt the new capture lacks"),
                                      ({}, {}, 1, "an empty capture"), (capture, None, 1, "a missing new capture")):
            code = compare(directory, old, new)
            if (code == 0) != (wanted == 0):
                failures.append(f"release_compare on {why} exits {code}; the accuracy gate may be skipped only on proof that every prompt's tokens match")

        records = json.loads((ROOT / "deployment" / "lanes.json").read_text())
        record = next(name for name, lane in sorted(records["lanes"].items()) if lane["launcher"] == "stage_lane")
        clean = {name: os.environ[name] for name in ("PATH", "HOME") if name in os.environ}
        polluted = dict(os.environ, STAGE_EXTRA_ENV="SPARK_REQUIRED_CHECK=0", STAGE_MEMORY_MAX="1G", STAGE_LANE="0",
                        STAGE_WEIGHTD_SOCKET="/tmp/other.sock", STAGE_EXPERT_POOL_BYTES="1")
        first, second, changed = directory / "clean", directory / "polluted", directory / "changed"
        for path in (first, second, changed):
            path.mkdir()
        baseline = render(record, first, clean)
        if render(record, second, polluted) != baseline:
            failures.append(f"{record}: lane_config render must be a pure function of the record; an operator's STAGE_* environment changed the rendered files")
        edited = json.loads(json.dumps(records))
        edited["lanes"][record]["memory_max"] = "1G" if edited["lanes"][record]["memory_max"] != "1G" else "2G"
        candidate = directory / "lanes-candidate.json"
        candidate.write_text(json.dumps(edited))
        if render(record, changed, dict(clean, SPARK_LANES=str(candidate))) == baseline:
            failures.append(f"{record}: a record with another memory_max must render differently (the comparison above must see changes)")
        if any(b"-E SPARK_REQUIRED_CHECK" in data for data in baseline.values()):
            failures.append(f"{record}: a rendered lane.sh carries extra residentd environment")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS lane tools: release_compare passes only when every prompt's tokens match (not on a diverged token, a missing prompt, an empty or a missing capture); "
          "a lane record renders the same files under an operator's STAGE_* environment, and a changed record renders differently")


if __name__ == "__main__":
    main()
