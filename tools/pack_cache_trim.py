#!/usr/bin/env python3
import argparse
import os
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description="Drop the page cache of the given pack files only (never node-global), once or every N seconds.")
    parser.add_argument("--every", type=float, default=0.0)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    missing = [p for p in args.paths if not os.path.isfile(p)]
    if args.every < 0 or missing:
        print("pack_cache_trim: --every must be >= 0 and every path a file; not a file: " + " ".join(missing), file=sys.stderr)
        return 2
    while True:
        done = 0
        for path in args.paths:
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            done += 1
        if args.every == 0:
            print(f"pack_cache_trim: dropped the page cache of {done} of {len(args.paths)} files")
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
