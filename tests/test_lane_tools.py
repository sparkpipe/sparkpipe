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
        direct = subprocess.run([sys.executable, str(ROOT / "tools" / "stage_lane.py"), "--record", record, "--output-dir", str(directory / "direct")],
                                cwd=ROOT, env=dict(clean, SPARK_LANES=str(candidate)), capture_output=True, text=True)
        if direct.returncode != 0 or not (directory / "direct" / "deployment.json").is_file():
            failures.append(f"{record}: tools/stage_lane.py must render a candidate lanes file outside the repository (SPARK_LANES), as stage_lane.sh setup does: {direct.stderr.strip()[-200:]}")
        if any(b"-E SPARK_REQUIRED_CHECK" in data for data in baseline.values()):
            failures.append(f"{record}: a rendered lane.sh carries extra residentd environment")
    sys.path.insert(0, str(ROOT / "tools"))
    import lane_config
    for record_id, lane in sorted(json.loads((ROOT / "deployment" / "lanes.json").read_text())["lanes"].items()):
        if "draftd" not in lane:
            continue
        environment = lane_config.draftd_environment(record_id)[1]
        if f"{environment['DRAFTD_TAP_HOST']}:{environment['DRAFTD_TAP_PORT']}" != lane["taps"]["stream"] or "tap_host" in lane["draftd"] or "tap_port" in lane["draftd"]:
            failures.append(f"{record_id}: the draft server must listen where the rank's taps.stream points, from that one field")
    original = lane_config.record
    try:
        for broken in ({}, {"stream": "10.10.250.2"}, {"stream": "10.10.250.2:port"}):
            lane_config.record = lambda record_id, broken=broken: dict(original("kimi-k3"), taps=broken)
            try:
                lane_config.draftd_environment("kimi-k3")
                failures.append(f"a draft server record with taps {broken} must be refused")
            except SystemExit:
                pass
    finally:
        lane_config.record = original
    swap = {entry["id"]: entry for entry in json.loads((ROOT / "deployment" / "hub" / "model_swap.json").read_text())["models"]}
    for record_id, lane in sorted(records["lanes"].items()):
        if lane["launcher"] != "k3_lane":
            continue
        root = f"$HOME/k3-lanes/lane{lane['lane']}/root"
        start, warm = swap[record_id]["node_start"], swap[record_id]["node_warm"]
        wanted = (root, f"--unit=sp-k3-rd{lane['lane']} ", f"MemoryMax={lane['memory_max']} ", f"SPARK_WEIGHTD_LANE={lane['lane']} ",
                  f"SPARK_WEIGHTD_SOCKET={lane['weightd_socket']} ", f"SPARK_WEIGHTD_EXPERT_POOL_BYTES={lane['expert_pool_bytes']} ")
        missing = [piece for piece in wanted if piece not in start]
        if lane["runtime_root"] != f"/home/{{host}}/k3-lanes/lane{lane['lane']}/root" or missing or \
                f"SPARK_WEIGHTD_EXPERT_POOL_BYTES={lane['expert_pool_bytes']} " not in warm or f" {lane['weightd_socket']} " not in warm:
            failures.append(f"{record_id}: the swap entry starts and warms the lane with values the record does not hold: {missing or 'runtime_root or warm'}")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS lane tools: release_compare passes only when every prompt's tokens match (not on a diverged token, a missing prompt, an empty or a missing capture); "
          "a lane record renders the same files under an operator's STAGE_* environment, and a changed record renders differently; a draft server listens where taps.stream points, and a record without one is refused")


if __name__ == "__main__":
    main()
