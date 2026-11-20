import datetime
import json
import sys
from state_align import lateness, WAVE


def main(run, directory, threshold=300.0):
    late, steps = lateness(directory)
    waves = [(int(m.group(1)), float(m.group(3)), int(m.group(4))) for m in map(WAVE.search, open(f"api-{run}.log", errors="replace")) if m]
    offset = steps - len(waves)
    probe = [json.loads(line) for line in open(f"probe-{run}.jsonl") if line.strip()]
    row = [r for r in probe if r.get("tokens") == 262144][-1]
    end = datetime.datetime.strptime(row["utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    start = end - row["wall_s"]
    request = waves[-1][0]
    long_steps = [offset + i for i, w in enumerate(waves) if w[0] == request]
    scale = row["wall_s"] * 1000.0 / sum(waves[s - offset][1] for s in long_steps)
    clock = start
    print(f"{run}: waves with a rank more than {threshold:.0f} ms late; wall-clock window of the wave (UTC), seconds into the minute, late ranks")
    phases = []
    for s in long_steps:
        begin = clock
        clock += waves[s - offset][1] * scale / 1000.0
        ranks = [r for r in range(16) if late.get((r, s), 0.0) > threshold]
        if ranks:
            b = datetime.datetime.fromtimestamp(begin, datetime.timezone.utc)
            e = datetime.datetime.fromtimestamp(clock, datetime.timezone.utc)
            phases.append((b.second + b.microsecond / 1e6, e.second + e.microsecond / 1e6))
            print(f"  step {s:3d} {b:%H:%M:%S.%f}"[:-3] + f" - {e:%H:%M:%S.%f}"[:-3] + f"  sec {b.second:2d}-{e.second:2d}  ranks {ranks} "
                  + " ".join(f"{r}:{late[(r, s)]:.0f}" for r in ranks))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 300.0)
