#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
import threading
import time

GIB = 1 << 30
EARLYOOM_FRACTION = 0.02
REQUIRED_ABOVE_LINE = GIB
FLEET_HOSTS = ("spark0", "spark1", "spark2", "spark3", "spark4", "spark5", "spark6", "spark7",
               "spark8", "spark9", "sparka", "sparkb", "sparkc", "sparkd", "sparke", "sparkf")
SAMPLER = """
import sys, time
source = sys.argv[3] if len(sys.argv) > 3 else '/proc/meminfo'
def read():
    values = {}
    with open(source) as handle:
        for line in handle:
            name, rest = line.split(':', 1)
            values[name] = int(rest.split()[0]) * 1024
    return values['MemTotal'], values['MemAvailable']
interval = float(sys.argv[1])
end = time.time() + float(sys.argv[2])
total = read()[0]
while time.time() < end:
    began, low, last = time.time(), None, None
    while time.time() - began < 1.0:
        last = read()[1]
        low = last if low is None else min(low, last)
        time.sleep(interval)
    print('%.3f %d %d %d' % (time.time(), total, low, last), flush=True)
"""


def parse_sample(line):
    fields = line.split()
    if len(fields) != 4:
        return None
    try:
        return {"t": float(fields[0]), "total": int(fields[1]), "low": int(fields[2]), "last": int(fields[3])}
    except ValueError:
        return None


def node_summary(samples):
    if not samples:
        return {"samples": 0}
    total = samples[0]["total"]
    line = int(total * EARLYOOM_FRACTION)
    lowest = min(sample["low"] for sample in samples)
    levels = sorted(sample["last"] for sample in samples)
    steady = levels[len(levels) // 2]
    return {"samples": len(samples), "total_bytes": total, "earlyoom_line_bytes": line, "lowest_available_bytes": lowest,
            "steady_available_bytes": steady, "dip_below_steady_bytes": max(0, steady - lowest), "lowest_above_line_bytes": lowest - line,
            "passes": lowest - line >= REQUIRED_ABOVE_LINE}


def verdict(nodes, hosts):
    missing = [host for host in hosts if nodes.get(host, {}).get("samples", 0) == 0]
    failing = [host for host in hosts if host not in missing and not nodes[host]["passes"]]
    measured = [nodes[host] for host in hosts if host not in missing]
    return {"hosts": len(hosts), "missing": missing, "failing": failing, "passes": not missing and not failing,
            "lowest_above_line_bytes": min((node["lowest_above_line_bytes"] for node in measured), default=None),
            "largest_dip_bytes": max((node["dip_below_steady_bytes"] for node in measured), default=None),
            "required_above_line_bytes": REQUIRED_ABOVE_LINE}


class MemoryWatch:
    def __init__(self, hosts, ssh=("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"), interval=0.1, seconds=86400):
        self.hosts = list(hosts)
        self.ssh = list(ssh)
        self.interval = interval
        self.seconds = seconds
        self.samples = {host: [] for host in self.hosts}
        self.errors = {}
        self.processes = {}
        self.readers = []

    def read(self, host, process):
        for raw in process.stdout:
            sample = parse_sample(raw.decode(errors="replace"))
            if sample is not None:
                self.samples[host].append(sample)
        error = process.stderr.read().decode(errors="replace").strip()
        if error:
            self.errors[host] = error[-400:]

    def start(self):
        for host in self.hosts:
            command = self.ssh + [host, "python3 - " + shlex.quote(str(self.interval)) + " " + shlex.quote(str(self.seconds))]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.stdin.write(SAMPLER.encode())
            process.stdin.close()
            self.processes[host] = process
            reader = threading.Thread(target=self.read, args=(host, process), daemon=True)
            reader.start()
            self.readers.append(reader)
        return self

    def stop(self):
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self.processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for reader in self.readers:
            reader.join(timeout=10)
        return self.summary()

    def summary(self):
        nodes = {host: node_summary(self.samples[host]) for host in self.hosts}
        return {"nodes": nodes, "verdict": verdict(nodes, self.hosts), "errors": dict(self.errors)}


def main():
    parser = argparse.ArgumentParser(description="Sample MemAvailable on fleet nodes at 10 Hz and judge the 1 GiB margin above earlyoom's line")
    parser.add_argument("--hosts", default=",".join(FLEET_HOSTS))
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    watch = MemoryWatch(arguments.hosts.split(","), interval=arguments.interval, seconds=arguments.seconds).start()
    try:
        time.sleep(arguments.seconds + 5)
    except KeyboardInterrupt:
        pass
    summary = watch.stop()
    with open(arguments.output, "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary["verdict"]))
    if not summary["verdict"]["passes"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
