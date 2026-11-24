import datetime
import glob
import json
import re
import statistics
import sys
from collections import defaultdict

ARRIVAL = re.compile(r"STAGE-STEP-ARRIVAL rank=(\d+) steps=1 first_us=(\d+) peer_us=([\d,]+)")
WAVE = re.compile(r"batch_step kind=prefill request=(\d+) rows=(\d+) .*?service_ms=([\d.]+) .*?context=(\d+)")
HOSTS = ["spark0", "spark1", "spark2", "spark3", "spark4", "spark5", "spark6", "spark7", "spark8", "spark9", "sparka", "sparkb", "sparkc", "sparkd",
         "sparke", "sparkf"]


def lateness(directory):
    arrivals = defaultdict(list)
    for path in glob.glob(f"{directory}/rank-*.log"):
        for line in open(path, errors="replace"):
            match = ARRIVAL.search(line)
            if match:
                arrivals[int(match.group(1))].append([int(value) for value in match.group(3).split(",")])
    ranks = sorted(arrivals)
    steps = min(len(arrivals[rank]) for rank in ranks)
    late = {}
    for step in range(steps):
        seen = {peer: statistics.median([arrivals[rank][step][peer] for rank in ranks if rank != peer]) / 1000.0 for peer in ranks}
        base = statistics.median(seen.values())
        for peer in ranks:
            late[(peer, step)] = seen[peer] - base
    return late, steps


def main(run, directory, files_path):
    late, steps = lateness(directory)
    waves = [(int(m.group(1)), float(m.group(3)), int(m.group(4))) for m in map(WAVE.search, open(f"api-{run}.log", errors="replace")) if m]
    offset = steps - len(waves)
    probe = [json.loads(line) for line in open(f"probe-{run}.jsonl") if line.strip()]
    row = [r for r in probe if r.get("tokens") == 262144][-1]
    end = datetime.datetime.strptime(row["utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    start = end - row["wall_s"]
    request = waves[-1][0]
    long_steps = [offset + index for index, wave in enumerate(waves) if wave[0] == request]
    total = sum(waves[step - offset][1] for step in long_steps)
    scale = row["wall_s"] * 1000.0 / total
    bounds, clock = [], start
    for step in long_steps:
        begin = clock
        clock += waves[step - offset][1] * scale / 1000.0
        bounds.append((step, begin, clock))
    writes = defaultdict(list)
    for line in open(files_path):
        host, stamp, size = line.split()
        if int(size) > 1000000 and start <= float(stamp) <= end:
            for step, begin, finish in bounds:
                if begin <= float(stamp) < finish:
                    writes[HOSTS.index(host)].append(step)
                    break
    print(f"{run}: 262,144-token request {request}, steps {long_steps[0]}..{long_steps[-1]}, scale {scale:.4f}; "
          f"lateness of each rank in the wave its node wrote a recurrent state (>1 MB file) and the wave after")
    print("rank  states  late>300_same  late>300_next  late>100_same_or_next  median_ms_at_state  median_ms_other  waves>300_elsewhere")
    for rank in range(16):
        state_steps = sorted(set(writes[rank]))
        near = set(state_steps) | {step + 1 for step in state_steps}
        same = sum(1 for step in state_steps if late.get((rank, step), 0.0) > 300)
        after = sum(1 for step in state_steps if late.get((rank, step + 1), 0.0) > 300)
        either = sum(1 for step in state_steps if max(late.get((rank, step), 0.0), late.get((rank, step + 1), 0.0)) > 100)
        at = [max(late.get((rank, step), 0.0), late.get((rank, step + 1), 0.0)) for step in state_steps]
        other = [late[(rank, step)] for step in long_steps if step not in near and (rank, step) in late]
        elsewhere = sum(1 for step in long_steps if step not in near and late.get((rank, step), 0.0) > 300)
        print(f"{rank:4d} {len(state_steps):7d} {same:14d} {after:14d} {either:22d} {statistics.median(at) if at else 0.0:19.0f} "
              f"{statistics.median(other) if other else 0.0:16.0f} {elsewhere:20d}")



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
