import datetime
import json
import sys
from collections import defaultdict
from state_align import lateness, WAVE, HOSTS


def bounds_for(run, steps):
    waves = [(int(m.group(1)), float(m.group(3)), int(m.group(4))) for m in map(WAVE.search, open(f"api-{run}.log", errors="replace")) if m]
    offset = steps - len(waves)
    probe = [json.loads(line) for line in open(f"probe-{run}.jsonl") if line.strip()]
    row = [r for r in probe if r.get("tokens") == 262144][-1]
    end = datetime.datetime.strptime(row["utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    start = end - row["wall_s"]
    request = waves[-1][0]
    long_steps = [offset + i for i, w in enumerate(waves) if w[0] == request]
    scale = row["wall_s"] * 1000.0 / sum(waves[s - offset][1] for s in long_steps)
    out, clock = [], start
    for s in long_steps:
        begin = clock
        clock += waves[s - offset][1] * scale / 1000.0
        out.append((s, begin, clock))
    return out


def main(runs, events_path, threshold=300.0):
    events = defaultdict(list)
    for line in open(events_path):
        host, stamp, kind = line.split()
        events[(host, kind)].append(float(stamp))
    for run in runs:
        late, steps = lateness(run.lower())
        bounds = bounds_for(run, steps)
        long_steps = [b[0] for b in bounds]
        print(f"{run}: per node, events in the prompt window and how often the rank of that node was more than {threshold:.0f} ms late "
              f"in the wave of the event or the next, against its rate over all waves")
        print("node    kind      events  hit  hit_rate  base_rate(2-wave)")
        for rank, host in enumerate(HOSTS):
            base = [max(late.get((rank, s), 0), late.get((rank, s + 1), 0)) > threshold for s in long_steps]
            base_rate = sum(base) / len(base)
            for kind in ("uplink", "osdguard", "cephcli", "sysstat"):
                stamps = [t for t in events.get((host, kind), []) if bounds[0][1] <= t < bounds[-1][2]]
                if not stamps:
                    continue
                hits = 0
                for t in stamps:
                    s = next(b[0] for b in bounds if b[1] <= t < b[2])
                    hits += max(late.get((rank, s), 0), late.get((rank, s + 1), 0)) > threshold
                print(f"{host:7s} {kind:9s} {len(stamps):6d} {hits:4d} {hits / len(stamps):9.2f} {base_rate:9.2f}")
        print()


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 300.0)
