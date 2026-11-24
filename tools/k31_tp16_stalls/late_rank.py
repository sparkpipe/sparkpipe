import re
import statistics
import sys
from collections import Counter, defaultdict

ARRIVAL = re.compile(r"STAGE-STEP-ARRIVAL rank=(\d+) steps=(\d+) first_us=(\d+) peer_us=([\d,]+)")
TIMING = re.compile(r"STAGE-STEP-TIMING rank=(\d+) steps=(\d+) .*?submit_us=(\d+) .*?peer_wait_us=(\d+) copy_us=(\d+) combine_us=(\d+)")
WAVE = re.compile(r"batch_step kind=prefill request=(\d+) rows=(\d+) .*?service_ms=([\d.]+) .*?context=(\d+)")


def read_ranks(paths):
    arrivals, timings = defaultdict(list), defaultdict(list)
    for path in paths:
        for line in open(path, errors="replace"):
            match = ARRIVAL.search(line)
            if match and match.group(2) == "1":
                arrivals[int(match.group(1))].append([int(value) for value in match.group(4).split(",")])
            match = TIMING.search(line)
            if match and match.group(2) == "1":
                timings[int(match.group(1))].append((int(match.group(3)), int(match.group(4)), int(match.group(5)), int(match.group(6))))
    return arrivals, timings


def read_waves(path):
    return [(int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))) for m in map(WAVE.search, open(path, errors="replace")) if m]


def main(api_log, rank_logs):
    arrivals, timings = read_ranks(rank_logs)
    waves = read_waves(api_log)
    ranks = sorted(arrivals)
    steps = min(len(arrivals[rank]) for rank in ranks)
    print(f"{len(ranks)} ranks, {steps} steps with one arrival line each, {len(waves)} prefill waves in the API log")
    offset = steps - len(waves) if steps >= len(waves) else 0
    late_counter, margin_by_rank = Counter(), defaultdict(list)
    print("step\tcontext\tservice_ms\tlate_rank\tlate_ms\tnext_ms\tleast_wait_rank\tleast_wait_ms\tmedian_wait_ms")
    for step in range(steps):
        by_peer = {}
        for peer in ranks:
            seen = [arrivals[rank][step][peer] for rank in ranks if rank != peer and peer < len(arrivals[rank][step])]
            by_peer[peer] = statistics.median(seen) / 1000.0 if seen else 0.0
        order = sorted(by_peer, key=lambda peer: -by_peer[peer])
        waits = {rank: timings[rank][step][1] / 1000.0 for rank in ranks if step < len(timings[rank])}
        least = min(waits, key=waits.get) if waits else -1
        wave = waves[step - offset] if 0 <= step - offset < len(waves) else None
        context, service = (wave[3], wave[2]) if wave else (-1, 0.0)
        late, margin = order[0], by_peer[order[0]] - by_peer[order[1]]
        if margin > 200.0:
            late_counter[late] += 1
            margin_by_rank[late].append(margin)
        print(f"{step}\t{context}\t{service:.0f}\t{late}\t{by_peer[late]:.0f}\t{by_peer[order[1]]:.0f}\t{least}\t{waits.get(least, 0.0):.0f}\t"
              f"{statistics.median(waits.values()) if waits else 0.0:.0f}")
    print("\nsteps where one rank was later than the next by more than 200 ms, by rank:")
    for rank, count in late_counter.most_common():
        print(f"rank {rank}: {count} steps, median margin {statistics.median(margin_by_rank[rank]):.0f} ms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
