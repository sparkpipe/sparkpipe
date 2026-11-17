import glob
import re
import statistics
import sys
from collections import defaultdict

ARRIVAL = re.compile(r"STAGE-STEP-ARRIVAL rank=(\d+) steps=1 first_us=(\d+) peer_us=([\d,]+)")
VENDOR = {0: "Gigabyte", 1: "MSI", 2: "Gigabyte", 3: "Gigabyte", 4: "Gigabyte", 5: "Gigabyte", 6: "Gigabyte", 7: "Lenovo", 8: "Lenovo",
          9: "Lenovo", 10: "Gigabyte", 11: "Gigabyte", 12: "Dell", 13: "Gigabyte", 14: "Gigabyte", 15: "Gigabyte"}


def main(directory, first_step=2):
    arrivals = defaultdict(list)
    for path in glob.glob(f"{directory}/rank-*.log"):
        for line in open(path, errors="replace"):
            match = ARRIVAL.search(line)
            if match:
                arrivals[int(match.group(1))].append([int(value) for value in match.group(3).split(",")])
    ranks = sorted(arrivals)
    steps = min(len(arrivals[rank]) for rank in ranks)
    late = {}
    for step in range(first_step, steps - 1):
        seen = {peer: statistics.median([arrivals[rank][step][peer] for rank in ranks if rank != peer]) / 1000.0 for peer in ranks}
        base = statistics.median(seen.values())
        for peer in ranks:
            late[(peer, step)] = seen[peer] - base
    span = range(first_step, steps - 1)
    print(f"{len(ranks)} ranks, waves {first_step}..{steps - 2}; '#' = the rank's data reached its peers more than 300 ms after the median rank's, '+' 100-300 ms")
    for peer in ranks:
        print(f"{peer:2d} {VENDOR[peer]:9s} " + "".join("#" if late[(peer, s)] > 300 else ("+" if late[(peer, s)] > 100 else ".") for s in span))
    counts = [sum(1 for peer in ranks if late[(peer, s)] > 300) for s in span]
    print("ranks late    " + "".join(str(min(count, 9)) for count in counts))
    periodic = [s for s in span if counts[s - first_step] > 0]
    print("waves with a late rank:", len(periodic), "of", len(span), "; their wave numbers mod 8:",
          dict(sorted(defaultdict(int, {k: sum(1 for s in periodic if s % 8 == k) for k in range(8)}).items())))
    print("\nrank vendor    waves_over_300ms  waves_over_100ms  mean_ms")
    for peer in ranks:
        values = [late[(peer, s)] for s in span]
        print(f"{peer:4d} {VENDOR[peer]:9s} {sum(1 for v in values if v > 300):16d} {sum(1 for v in values if v > 100):17d} {statistics.mean(values):8.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
