import glob
import re
import statistics
import sys
from collections import defaultdict
from state_align import lateness

TIMING = re.compile(r"STAGE-STEP-TIMING rank=(\d+) steps=1 .*?submit_us=(\d+) graph_us=(\d+) source_wait_us=(\d+) peer_wait_us=(\d+) copy_us=(\d+) combine_us=(\d+)")
PARTS = ["submit", "graph", "source_wait", "peer_wait", "copy", "combine"]


def main(run, threshold=300.0):
    directory = run.lower()
    late, steps = lateness(directory)
    timing = defaultdict(list)
    for path in glob.glob(f"{directory}/rank-*.log"):
        for line in open(path, errors="replace"):
            m = TIMING.search(line)
            if m:
                timing[int(m.group(1))].append([int(v) / 1000.0 for v in m.groups()[1:]])
    rows_late, rows_calm = defaultdict(list), defaultdict(list)
    for rank in range(16):
        for step in range(3, min(steps, len(timing[rank])) - 1):
            parts = timing[rank][step]
            others = [timing[r][step] for r in range(16) if r != rank and step < len(timing[r])]
            median_other = [statistics.median(o[i] for o in others) for i in range(len(PARTS))]
            delta = [parts[i] - median_other[i] for i in range(len(PARTS))]
            (rows_late if late.get((rank, step), 0) > threshold else rows_calm)["all"].append(delta)
    print(f"{run}: the late rank minus the median of its peers in the same step, ms (median over late rank-steps; calm rank-steps for contrast)")
    print("set    count  " + "  ".join(f"{p:>11s}" for p in PARTS))
    for name, rows in (("late", rows_late["all"]), ("calm", rows_calm["all"])):
        if rows:
            print(f"{name:5s} {len(rows):6d}  " + "  ".join(f"{statistics.median(r[i] for r in rows):11.1f}" for i in range(len(PARTS))))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 300.0)
