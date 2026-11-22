#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import spec_soak  # noqa: E402


def line(rounds, kept, unavailable, wait_ms):
    return (f"batch_speculation frames={rounds} drafted={rounds * 4} kept_drafts={kept} tree_frames=0 buckets r4:n=1,rows=4.0,kept=2.0,ms=1.0 "
            f"model rounds={rounds} frames={rounds} drafted={rounds * 4} kept_drafts={kept} extended=0 extension_drafts=0 extension_kept=0 "
            f"declined=0 unavailable={unavailable} backoffs=0 failed=0 abandoned=0 wait_ms={wait_ms}\n")


def arm(directory, log, start, end, before=None, after=None):
    directory.mkdir()
    (directory / "arm.json").write_text(json.dumps({"log": str(log), "start_offset": start, "end_offset": end, "requests": [{}],
                                                    "units_before": before or {}, "units_after": after or {}}))


def main():
    tasks = spec_soak.workload()
    tiers = sorted({task["context_chars"] for task in tasks})
    assert len(tiers) == 3 and tiers[0] == 0 and tiers[1] >= 160_000 and tiers[2] >= 400_000, tiers
    assert sum(1 for task in tasks if task["context_chars"] == 0) >= 48 and len(tasks) > 64, len(tasks)
    assert tasks[0]["sampling"] == {"temperature": 0} and tasks[1]["sampling"]["seed"] == 1001, tasks[1]["sampling"]
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        log = work / "api.log"
        text = line(100, 250, 0, 10.0)
        python_start = len(text)
        text += "noise\n" + line(1100, 2750, 2, 10.0)
        python_end = engine_start = len(text)
        text += line(1600, 4000, 3, 9.0) + line(400, 1010, 0, 8.0) + line(1400, 3530, 1, 8.0)
        engine_end = len(text)
        log.write_text(text)
        total, segments = spec_soak.summed(log, python_start, python_end)
        assert total["rounds"] == 1000 and total["kept_drafts"] == 2500 and total["unavailable"] == 2 and segments == 1, total
        total, segments = spec_soak.summed(log, engine_start, engine_end)
        assert segments == 2 and total["rounds"] == 500 + 1400 and total["kept_drafts"] == 1250 + 3530 and total["unavailable"] == 1 + 1, total
        arm(work / "python", log, python_start, python_end)
        arm(work / "engine", log, engine_start, engine_end, {"sparkpipe-draftd-engine": 0}, {"sparkpipe-draftd-engine": 0})
        script = str(ROOT / "tools" / "spec_soak.py")
        passed = subprocess.run([sys.executable, script, "compare", str(work / "python"), str(work / "engine")], capture_output=True, text=True)
        assert passed.returncode == 0 and "PASS engine against python" in passed.stdout, passed.stdout + passed.stderr
        (work / "restarted").mkdir()
        (work / "restarted" / "arm.json").write_text(json.dumps(dict(json.loads((work / "engine" / "arm.json").read_text()),
                                                                     units_after={"sparkpipe-draftd-engine": 1})))
        failed = subprocess.run([sys.executable, script, "compare", str(work / "python"), str(work / "restarted")], capture_output=True, text=True)
        assert failed.returncode == 1 and "FAIL" in failed.stdout and "sparkpipe-draftd-engine" in failed.stdout, failed.stdout
    print("PASS spec_soak: the workload has three context tiers and more than 64 tasks with fixed sampling, the counters sum over an API "
          "restart, and compare passes matching arms and fails an arm whose unit restarted")


if __name__ == "__main__":
    main()
