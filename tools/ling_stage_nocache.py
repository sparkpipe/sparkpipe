#!/usr/bin/env python3
"""Stage warm checkpoint shards to local disk without flooding page cache."""
import os
import sys
import time

CHUNK = 8 << 20


def copy_nocache(src: str, dst: str) -> float:
    started = time.time()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            chunk = fi.read(CHUNK)
            if not chunk:
                break
            fo.write(chunk)
        os.posix_fadvise(fi.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.fsync(fo.fileno())
        os.posix_fadvise(fo.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    elapsed = time.time() - started
    if os.path.getsize(src) != os.path.getsize(dst):
        raise SystemExit(f"size mismatch: {src} {os.path.getsize(src)} vs {dst} {os.path.getsize(dst)}")
    return elapsed


def main() -> int:
    src_root, dst_root = sys.argv[1], sys.argv[2]
    for name in sys.argv[3:]:
        src = os.path.join(src_root, name)
        dst = os.path.join(dst_root, name)
        if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
            print(f"skip {name}: staged size matches", flush=True)
            continue
        elapsed = copy_nocache(src, dst)
        rate = os.path.getsize(dst) / max(elapsed, 1e-9) / 1e6
        print(f"staged {name} in {elapsed:.1f}s ({rate:.0f} MB/s)", flush=True)
    os.sync()
    return 0


if __name__ == "__main__":
    sys.exit(main())
