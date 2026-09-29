#!/usr/bin/env python3
"""Route traces become a sorted .wset, rank disagreement fails, a cap keeps every layer."""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/glm5_next_wset_from_trace.py"
STEPS = [
    (41, {3: "5,17", 4: "9"}),
    (42, {3: "5", 4: "9,200"}),
    (43, {3: "17", 4: "9"}),
]


def trace(rank, steps):
    return "".join(f"G5N-ROUTE rank={rank} rows=1 pos={pos} layer={layer} n={len(experts.split(','))} e={experts}\n"
                   for pos, layers in steps for layer, experts in sorted(layers.items()))


def run(directory, *arguments):
    return subprocess.run([sys.executable, str(TOOL), *map(str, arguments)], cwd=directory, capture_output=True, text=True)


def pairs(path):
    raw = path.read_bytes()
    return [struct.unpack_from("<II", raw, offset) for offset in range(0, len(raw), 8)]


def main():
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        for rank in (0, 1):
            (directory / f"run.stage00.rank{rank:02d}.trace").write_text(trace(rank, STEPS))
        traces = sorted(directory.glob("run.*.trace"))
        result = run(directory, *traces, "--output", directory / "full.wset")
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert pairs(directory / "full.wset") == [(3, 5), (3, 17), (4, 9), (4, 200)]
        assert report["union_keys"] == 4 and report["union_per_layer"] == {"3": 2, "4": 2}
        assert report["steps"] == 3 and report["all_hit_steps"] == 3 and report["ranks"] == [0, 1]
        again = run(directory, *traces, "--output", directory / "again.wset", "--expect-route-sha256", report["route_sha256"])
        assert again.returncode == 0 and (directory / "again.wset").read_bytes() == (directory / "full.wset").read_bytes()
        assert run(directory, *traces, "--output", directory / "x.wset", "--expect-route-sha256", "0" * 64).returncode != 0
        capped = run(directory, *traces, "--output", directory / "cap.wset", "--cap-keys", 2)
        assert capped.returncode == 0, capped.stderr
        assert pairs(directory / "cap.wset") == [(3, 5), (4, 9)]
        assert json.loads(capped.stdout)["all_hit_steps"] == 0
        assert run(directory, *traces, "--output", directory / "x.wset", "--cap-keys", 1).returncode != 0
        (directory / "run.stage00.rank01.trace").write_text(trace(1, STEPS[:2] + [(43, {3: "18", 4: "9"})]))
        diverged = run(directory, *traces, "--output", directory / "x.wset")
        assert diverged.returncode != 0 and "ranks route differently" in diverged.stderr
        (directory / "run.stage00.rank01.trace").write_text("G5N-ROUTE rank=1 rows=1 pos=41 layer=3 n=2 e=17,5\n")
        assert run(directory, *traces, "--output", directory / "x.wset").returncode != 0
        (directory / "run.stage00.rank01.trace").write_text("G5N-ROUTE rank=1 rows=1 pos=41 layer=3 n=1 e=288\n")
        assert run(directory, *traces, "--output", directory / "x.wset").returncode != 0
    print("PASS route traces build a deterministic .wset; rank drift, bad records and short caps fail")


if __name__ == "__main__":
    main()
