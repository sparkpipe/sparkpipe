import re
import statistics
import sys

HOSTS = ["spark0", "spark1", "spark2", "spark3", "spark4", "spark5", "spark6", "spark7", "spark8", "spark9", "sparka", "sparkb", "sparkc", "sparkd",
         "sparke", "sparkf"]
RTT = re.compile(r"time=([\d.]+) ms")
SEQ = re.compile(r"icmp_seq=(\d+)")


def window(path, start, end, tail_bytes=60000000):
    with open(path, errors="replace") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - tail_bytes))
        handle.readline()
        for line in handle:
            if start <= line[:19] <= end:
                yield line


def gpu(root, host, start, end):
    rows = []
    for line in window(f"{root}/{host}/gpu.log", start, end):
        parts = [part.strip() for part in line.split(" ", 1)[1].split(",")]
        try:
            rows.append((float(parts[1]), float(parts[4].split()[0]), [int(part.split()[0]) for part in parts[7:11]]))
        except (IndexError, ValueError):
            continue
    throttled = [(rows[-1][2][i] - rows[0][2][i]) / 1e6 for i in range(4)]
    clocks = sorted(row[1] for row in rows)
    return throttled, clocks[0], statistics.median(clocks), max(row[0] for row in rows)


def system(root, host, start, end):
    memory, load = [], []
    for line in window(f"{root}/{host}/sys.log", start, end):
        parts = line.split()
        try:
            memory.append(int(parts[parts.index("memavail_kb:") + 1]) / 1048576.0)
            load.append(float(parts[parts.index("load:") + 1]))
        except (ValueError, IndexError):
            continue
    return min(memory), statistics.median(memory), max(load)


def ping(root, host, start, end):
    rtts, sequences = [], []
    for line in window(f"{root}/{host}/ping.log", start, end, 20000000):
        match = RTT.search(line)
        if match:
            rtts.append(float(match.group(1)))
        match = SEQ.search(line)
        if match:
            sequences.append(int(match.group(1)))
    lost = max(sequences) - min(sequences) + 1 - len(sequences) if sequences else -1
    return lost, max(rtts)


def dmesg(root, host, start, end):
    return sum(1 for _ in window(f"{root}/{host}/dmesg.log", start, end, 20000000))


def main(root, start, end):
    print(f"fleet telemetry from {start}Z to {end}Z")
    print("node    power_cap_s sw_therm_s hw_therm_s brake_s  sm_min sm_med temp_max  memavail_min_GiB memavail_med_GiB load_max  ping_lost ping_max_ms  dmesg")
    for host in HOSTS:
        throttled, clock_min, clock_med, temp = gpu(root, host, start, end)
        memory_min, memory_med, load_max = system(root, host, start, end)
        lost, rtt_max = ping(root, host, start, end)
        print(f"{host:7s} {throttled[0]:11.1f} {throttled[1]:10.1f} {throttled[2]:10.1f} {throttled[3]:7.1f}  {clock_min:6.0f} {clock_med:6.0f} {temp:8.0f}"
              f"  {memory_min:16.2f} {memory_med:16.2f} {load_max:8.2f}  {lost:9d} {rtt_max:11.1f}  {dmesg(root, host, start, end):5d}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
