#!/usr/bin/env python3
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c"
COLLECTIVE = ROOT / "ring/transport/tp_device_collective.c"


def printed(path, tag):
    match = re.search(r'"(' + re.escape(tag) + r' [^"]*)\\n"', path.read_text())
    if not match:
        print("FAIL %s prints no %s line" % (path.name, tag))
        sys.exit(1)
    return re.sub(r"%ll([ud])", r"%\1", match.group(1)).replace("%u", "%d")


def check(condition, message, output):
    if not condition:
        print("FAIL %s\n%s" % (message, output))
        sys.exit(1)


def rank_log(rank, formats, old):
    chain, collective, replay, key, adopt, capture = formats
    lines = [key % (5, 50, 1000) if rank == 0 else adopt % (rank, 5, 4)]
    lines += [capture % ((4,) + (0,) * (capture.count("%d") - 2) + (2047,))] if old else []
    lines += [replay % (1, 10000000, 0), replay % (1, 10000000, 0)]
    lines += [replay % (3, 99000000, 0)] if rank < 2 and not old else []
    stages = (0.0,) * 8
    if old:
        lines += [re.sub(r" rows=%d epoch=%d", "", chain) % ((1, "graph", 2, 0, 22.0, 0.0, 0.0, 0) + stages)]
    else:
        lines += [chain % ((1, "graph", 2, 1, 5, 0, 22.0, 0.0, 0.0, 0) + stages)]
    lines += [collective % (1, 0.4, 4.0 + 2.0 * rank, 0.1, 0.1)]
    if rank < 2 and not old:
        lines += [chain % ((3, "graph", 1, 1, 6, 0, 30.0, 0.0, 0.0, 0) + stages), collective % (3, 0.0, 1.0, 0.0, 0.0)]
    if not old:
        lines += [chain % ((0, "graph", 1, 1, 7, -3, 30.0, 0.0, 0.0, 0) + stages), collective % (0, 0.0, 1.0, 0.0, 0.0)]
        lines += [chain % ((2, "linear", 1, 8, 8, 0, 60.0, 30.0, 5.0, 91) + stages), collective % (2, 0.5, 10.0 + 2.0 * rank, 0.5, 0.5)]
    return "\n".join(lines) + "\n"


def run(directory, formats, old):
    logs = []
    for rank in range(3):
        path = Path(directory) / ("%s%d.log" % ("old" if old else "new", rank))
        path.write_text(rank_log(rank, formats, old))
        logs.append("%d=%s" % (rank, path))
    return subprocess.run([sys.executable, str(ROOT / "tools/tp_chain_budget.py"), *logs], capture_output=True, text=True, check=True).stdout


def main():
    formats = (printed(MODULE, "CHAIN-TIME"), printed(MODULE, "COLLECTIVE-GPU-TIME"), printed(MODULE, "GRAPH-REPLAY-TIME"),
               printed(COLLECTIVE, "CKEY-WRITE"), printed(COLLECTIVE, "CKEY-ADOPT"), printed(MODULE, "GRAPH-CAPTURE-OK"))
    with tempfile.TemporaryDirectory() as directory:
        output = run(directory, formats, False)
        rows = [line.split() for line in output.splitlines()]
        check("graph 1 1 2.00 11.00 10.00 6.70 3.00 2.00 1.00 0.30 1.00 0.00 | 0:1".split() in rows,
              "a graph chain splits into compute, rank-mean peer wait, the last rank's floor, skew, collective copies and host time between replays", output)
        check("linear 8 1 1.00 60.00 60.00 46.50 12.00 10.00 2.00 1.50 0.00 30.00 | 0:1".split() in rows,
              "a linear chain's run is its chain time and its host enqueue is reported as walk", output)
        check(sum(1 for row in rows if row[:1] in (["graph"], ["linear"])) == 2,
              "a chain that one rank never ran and a chain that failed are left out", output)
        output = run(directory, formats, True)
        rows = [line.split() for line in output.splitlines()]
        check("graph 4 1 2.00 11.00 10.00 6.70 3.00 2.00 1.00 0.30 1.00 0.00 | 0:1".split() in rows,
              "a log without rows and epoch on CHAIN-TIME aligns by the chain key epoch and takes rows from the capture before it", output)
    print("PASS tp chain budget: ranks align by epoch; each step splits into compute, collective latency floor, rank skew and host time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
