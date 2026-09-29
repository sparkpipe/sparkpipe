#!/usr/bin/env python3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MESH_TAIL = "gates=1 gate_us=64/64 gate_ms=0 self=0 starts=1 start_us=64/64 start_ms=0 start_self=0 worst_us=40 worst_tag=23:1 worst_closer=2 peers=1:32/32/0/0/0,2:64/64/1/1/37,3:32/32/0/0/0"
MESH_SELF = "gates=1 gate_us=1/1 gate_ms=0 self=1 starts=1 start_us=1/1 start_ms=0 start_self=1 worst_us=0 worst_tag=0:0 worst_closer=0 peers=1:128/128/0/0/0,2:128/128/0/0/0,3:128/128/0/0/0"
WAVE = (ROOT / "tests" / "fixtures" / "glm5_next_wave_timing.txt").read_text().strip()
WAVE_BEFORE_STEPS = WAVE.replace("rank=3", "rank=5").replace(" steps=6", "").replace(" linear_walk_ms=45", "")
WAVE_BEFORE_LINEAR = WAVE_BEFORE_STEPS.replace("rank=5", "rank=4").replace(" linear=1", "").replace(" linear_run_ms=60", "")


def run(tool, *logs):
    return subprocess.run([sys.executable, str(ROOT / "tools" / tool), *logs], capture_output=True, text=True, check=True).stdout


def check(condition, message, output):
    if not condition:
        print("FAIL %s\n%s" % (message, output))
        sys.exit(1)


def main():
    mock = (ROOT / "tests" / "test_weightd_mesh_mock.c").read_text()
    check(MESH_TAIL in mock and MESH_SELF in mock, "the weightd lines match what test_weightd_mesh_mock.c asserts", "")
    with tempfile.TemporaryDirectory() as directory:
        mesh0, mesh1, wave, older, linear = Path(directory) / "w0.log", Path(directory) / "w1.log", Path(directory) / "r3.log", Path(directory) / "r4.log", Path(directory) / "r5.log"
        prefix = "WD-MESH-TIMING posts=1 post_us=1/1 ship_us=1/1 credits=1 credit_us=1/1 "
        mesh0.write_text(prefix + MESH_TAIL + "\n")
        mesh1.write_text(prefix + MESH_SELF + "\n")
        wave.write_text(WAVE + "\n")
        older.write_text(WAVE_BEFORE_LINEAR + "\n")
        linear.write_text(WAVE_BEFORE_STEPS + "\n")
        output = run("mesh_timing_report.py", f"0={mesh0}", f"1={mesh1}")
        check("share of all gates each rank closed, as the last sender or by reaching its own gate last: 0:0.0%, 1:50.0%, 2:50.0%, 3:0.0%" in output,
              "the rank that closed its own gate and the last sender share the gates", output)
        check("share of chain-start gates each rank closed: 0:0.0%, 1:50.0%, 2:50.0%, 3:0.0%" in output, "start gates are attributed separately", output)
        check("(ms, all receivers): 1:0.0, 2:0.0, 3:0.0" in output and ["40", "0", "23:1", "2"] in [line.split() for line in output.splitlines()],
              "excess wait and the worst gate are reported", output)
        output = run("wave_timeline_report.py", f"3={wave}", f"4={older}", f"5={linear}")
        lines = [line.split() for line in output.splitlines()]
        check("3 1 3 8.0 2.5 33.3 66.7 33.3 on 0.7 1/1/0/0/0 | 3531.0 1.0 0.0 84.0 70.0 0.3 | 20.0 1.0 2.0 3.0 44.0 | 75.0 0.0 60.0 45.0 1.5 1 250 | 4096 131072".split() in lines,
              "the per-wave budget splits a wave into contiguous intervals, decode steps, graph, eager and linear runs, the linear walk, and busy reasons", output)
        check("4 1 3 8.0 1.0 33.3 66.7 0.0 on 0.7 1/1/0/0/0 | 3531.0 1.0 0.0 84.0 70.0 0.3 | 20.0 1.0 2.0 3.0 44.0 | 75.0 60.0 0.0 0.0 1.5 1 250 | 4096 131072".split() in lines,
              "a line written before linear chains existed counts every non-graph wave as eager", output)
        check("5 1 3 8.0 1.0 33.3 66.7 33.3 on 0.7 1/1/0/0/0 | 3531.0 1.0 0.0 84.0 70.0 0.3 | 20.0 1.0 2.0 3.0 44.0 | 75.0 0.0 60.0 0.0 1.5 1 250 | 4096 131072".split() in lines,
              "a line written before decode chains existed counts one step per wave and no walk", output)
        check(["311", "3", "7", "11/12", "0/100/300/251000/60000/500"] in lines, "the slowest wave carries its request, epochs and six parts", output)
    print("PASS timing reports: tools parse the exact WD-MESH-TIMING and G5N-WAVE-TIMING lines the C tests assert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
