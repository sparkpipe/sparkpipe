#!/usr/bin/env python3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MESH_TAIL = "gates=1 gate_us=64/64 gate_ms=0 self=0 starts=1 start_us=64/64 start_ms=0 start_self=0 worst_us=40 worst_tag=23:1 worst_closer=2 peers=1:32/32/0/0/0,2:64/64/1/1/37,3:32/32/0/0/0"
MESH_SELF = "gates=1 gate_us=1/1 gate_ms=0 self=1 starts=1 start_us=1/1 start_ms=0 start_self=1 worst_us=0 worst_tag=0:0 worst_closer=0 peers=1:128/128/0/0/0,2:128/128/0/0/0,3:128/128/0/0/0"
WAVE = "G5N-WAVE-TIMING rank=3 waves=3 rows=24 prefill=1 graph=2 eager=1 graph_path=1 retries=2 busy=1/1/0/0/0 captures=1 capture_ms=250 idle_us=8388608/8388608 wait_us=128/4096 key_us=512/512 setup_us=1024/262144 run_us=65536/131072 post_us=512/512 idle_ms=10593 wait_ms=3 key_ms=0 setup_ms=252 run_ms=210 post_ms=1 graph_run_ms=150 eager_run_ms=60 decode_wait_ms=3 source_wait_ms=3 peer_wait_ms=60 copy_ms=6 combine_ms=9 worst_ms=311 worst_request=7 worst_epochs=11/12 worst_us=0/100/300/251000/60000/500"


def run(tool, *logs):
    return subprocess.run([sys.executable, str(ROOT / "tools" / tool), *logs], capture_output=True, text=True, check=True).stdout


def check(condition, message, output):
    if not condition:
        print("FAIL %s\n%s" % (message, output))
        sys.exit(1)


def main():
    mock = (ROOT / "tests" / "test_weightd_mesh_mock.c").read_text()
    harness = (ROOT / "tests" / "test_glm5_next_stage_context.py").read_text()
    check(MESH_TAIL in mock and MESH_SELF in mock, "the weightd lines match what test_weightd_mesh_mock.c asserts", "")
    check(WAVE in harness, "the wave line matches what test_glm5_next_stage_context.py asserts", "")
    with tempfile.TemporaryDirectory() as directory:
        mesh0, mesh1, wave = Path(directory) / "w0.log", Path(directory) / "w1.log", Path(directory) / "r3.log"
        prefix = "WD-MESH-TIMING posts=1 post_us=1/1 ship_us=1/1 credits=1 credit_us=1/1 "
        mesh0.write_text(prefix + MESH_TAIL + "\n")
        mesh1.write_text(prefix + MESH_SELF + "\n")
        wave.write_text(WAVE + "\n")
        output = run("mesh_timing_report.py", f"0={mesh0}", f"1={mesh1}")
        check("share of all gates each rank closed, as the last sender or by reaching its own gate last: 0:0.0%, 1:50.0%, 2:50.0%, 3:0.0%" in output,
              "the rank that closed its own gate and the last sender share the gates", output)
        check("share of chain-start gates each rank closed: 0:0.0%, 1:50.0%, 2:50.0%, 3:0.0%" in output, "start gates are attributed separately", output)
        check("(ms, all receivers): 1:0.0, 2:0.0, 3:0.0" in output and ["40", "0", "23:1", "2"] in [line.split() for line in output.splitlines()],
              "excess wait and the worst gate are reported", output)
        output = run("wave_timeline_report.py", f"3={wave}")
        lines = [line.split() for line in output.splitlines()]
        check("3 1 3 8.0 33.3 66.7 on 0.7 1/1/0/0/0 | 3531.0 1.0 0.0 84.0 70.0 0.3 | 20.0 1.0 2.0 3.0 44.0 | 75.0 60.0 1.5 1 250 | 4096 131072".split() in lines,
              "the per-wave budget splits a wave into contiguous intervals, graph and eager runs, and busy reasons", output)
        check(["311", "3", "7", "11/12", "0/100/300/251000/60000/500"] in lines, "the slowest wave carries its request, epochs and six parts", output)
    print("PASS timing reports: tools parse the exact WD-MESH-TIMING and G5N-WAVE-TIMING lines the C tests assert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
