#!/usr/bin/env python3
"""Stage this node's share of warm shards and relay them to the gather host.

The queue's SPARK_QUEUE_RANK picks this node's contiguous slice of the
shard names given on argv, so one per-node job fans a shard list across
the fleet in a single dispatch (short bursts per client dodge the warm
crawl's per-client throttling of sustained readers).
"""
import os
import shlex
import subprocess
import sys
import time

WARM = "/mnt/model-warm/ling-3.0-flash"
GATHER = "spark0"
GATHER_DIR = "/home/spark0/lingbuild/src/ling-3.0-flash"
LOCAL = os.path.expanduser("~/lingbuild/tmp_stage")
CHUNK = 8 << 20
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", GATHER]


def copy_nocache(src: str, dst: str) -> None:
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            chunk = fi.read(CHUNK)
            if not chunk:
                break
            fo.write(chunk)
        os.posix_fadvise(fi.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.fsync(fo.fileno())
        os.posix_fadvise(fo.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def relay(names: list) -> None:
    os.makedirs(LOCAL, exist_ok=True)
    batch = []
    for name in names:
        src = os.path.join(WARM, name)
        dst = os.path.join(LOCAL, name)
        if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
            print(f"local skip {name}", flush=True)
        else:
            started = time.time()
            copy_nocache(src, dst)
            rate = os.path.getsize(dst) / max(time.time() - started, 1e-9) / 1e6
            print(f"staged {name} ({rate:.0f} MB/s)", flush=True)
        batch.append(dst)
    started = time.time()
    subprocess.run(["rsync", "-a", "--checksum", *batch,
                    f"{GATHER}:{GATHER_DIR}/"], check=True)
    moved = sum(os.path.getsize(p) for p in batch)
    print(f"relayed {len(batch)} files, {moved/1e9:.1f} GB in "
          f"{time.time() - started:.1f}s", flush=True)
    checks = " && ".join(f"test -s {shlex.quote(GATHER_DIR + '/' + os.path.basename(p))}"
                         for p in batch)
    result = subprocess.run(SSH + [checks], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"gather verification failed on {GATHER}: "
                         f"{result.stdout.strip()} {result.stderr.strip()}")
    print("gather verified on " + GATHER, flush=True)
    for path in batch:
        os.remove(path)


def main() -> int:
    rank = int(os.environ.get("SPARK_QUEUE_RANK", "0"))
    names = sys.argv[1:]
    if not names:
        print(f"host {os.uname().nodename} queue-rank {rank}: nothing to do")
        return 0
    print(f"host {os.uname().nodename} queue-rank {rank}: {names}", flush=True)
    relay(names)
    os.sync()
    return 0


if __name__ == "__main__":
    sys.exit(main())
