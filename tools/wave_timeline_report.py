#!/usr/bin/env python3
import argparse
import re

LINE = re.compile(r"G5N-WAVE-TIMING rank=(?P<rank>\d+) waves=(?P<waves>\d+) rows=(?P<rows>\d+) retries=(?P<retries>\d+) "
                  r"idle_us=\d+/\d+ pre_us=\d+/(?P<pre99>\d+) key_us=\d+/\d+ gpu_us=\d+/(?P<gpu99>\d+) post_us=\d+/\d+ "
                  r"idle_ms=(?P<idle>\d+) pre_ms=(?P<pre>\d+) key_ms=(?P<key>\d+) gpu_ms=(?P<gpu>\d+) post_ms=(?P<post>\d+) "
                  r"source_wait_ms=(?P<source>\d+) peer_wait_ms=(?P<peer>\d+) copy_ms=(?P<copy>\d+) combine_ms=(?P<combine>\d+) "
                  r"worst_ms=(?P<worst>\d+) worst_request=(?P<request>\d+) worst_epochs=(?P<main>\d+)/(?P<hc>\d+) "
                  r"worst_us=(?P<w_idle>\d+)/(?P<w_pre>\d+)/(?P<w_key>\d+)/(?P<w_gpu>\d+)/(?P<w_post>\d+)")
PARTS = ("idle", "pre", "key", "gpu", "post", "peer", "source", "copy", "combine")


def windows(path):
    with open(path, errors="replace") as log:
        return [match for match in map(LINE.search, log) if match]


def main():
    parser = argparse.ArgumentParser(description="Combine every rank's G5N-WAVE-TIMING lines into a per-wave time budget.")
    parser.add_argument("logs", nargs="+", metavar="RANK=PATH", help="the residentd stderr log of each rank")
    arguments = parser.parse_args()
    worst = []
    print("per-wave means in ms; compute = gpu - peer - source - copy - combine")
    print("rank windows  waves rows/wave retries |   idle    pre    key    gpu   post |   peer source   copy combine compute | pre99_us gpu99_us")
    for argument in arguments.logs:
        rank, path = argument.split("=", 1)
        rows = windows(path)
        if not rows:
            print(f"{int(rank):>4} no G5N-WAVE-TIMING lines")
            continue
        waves = sum(int(row.group("waves")) for row in rows)
        mean = {part: sum(int(row.group(part)) for row in rows) / waves for part in PARTS}
        compute = mean["gpu"] - mean["peer"] - mean["source"] - mean["copy"] - mean["combine"]
        print(f"{int(rank):>4} {len(rows):>7} {waves:>6} {sum(int(row.group('rows')) for row in rows) / waves:>9.1f} {sum(int(row.group('retries')) for row in rows):>7} | "
              + " ".join(f"{mean[part]:6.1f}" for part in PARTS[:5]) + " | "
              + " ".join(f"{mean[part]:6.1f}" for part in PARTS[5:]) + f" {compute:7.1f} | "
              + f"{max(int(row.group('pre99')) for row in rows):>8} {max(int(row.group('gpu99')) for row in rows):>8}")
        worst.extend((int(row.group("worst")), int(rank), row) for row in rows)
    print("\nslowest waves (ms, rank, request, main/hc epoch, idle/pre/key/gpu/post us)")
    for waited, rank, row in sorted(worst, key=lambda item: item[0], reverse=True)[:12]:
        print(f"{waited:>6} {rank:>4} {row.group('request'):>12} {row.group('main'):>10}/{row.group('hc'):<10} "
              f"{row.group('w_idle')}/{row.group('w_pre')}/{row.group('w_key')}/{row.group('w_gpu')}/{row.group('w_post')}")


if __name__ == "__main__":
    main()
