#!/usr/bin/env python3
import argparse
import re

INTERVALS = ("idle", "wait", "key", "setup", "run", "post")
COLLECTIVE = ("peer", "source", "copy", "combine")
BUSY = ("chain", "stream", "slot", "lanes", "other")
PATHS = {"0": "off", "1": "on", "2": "degraded"}
LINE = re.compile(r"G5N-WAVE-TIMING rank=(?P<rank>\d+) waves=(?P<waves>\d+) rows=(?P<rows>\d+) prefill=(?P<prefill>\d+) graph=(?P<graph>\d+) "
                  r"eager=(?P<eager>\d+) graph_path=(?P<path>\d+) retries=(?P<retries>\d+) "
                  r"busy=(?P<chain>\d+)/(?P<stream>\d+)/(?P<slot>\d+)/(?P<lanes>\d+)/(?P<other>\d+) captures=(?P<captures>\d+) capture_ms=(?P<capture>\d+) "
                  r"idle_us=\d+/\d+ wait_us=\d+/(?P<wait99>\d+) key_us=\d+/\d+ setup_us=\d+/\d+ run_us=\d+/(?P<run99>\d+) post_us=\d+/\d+ "
                  r"idle_ms=(?P<idle>\d+) wait_ms=(?P<wait>\d+) key_ms=(?P<key>\d+) setup_ms=(?P<setup>\d+) run_ms=(?P<run>\d+) post_ms=(?P<post>\d+) "
                  r"graph_run_ms=(?P<graph_run>\d+) eager_run_ms=(?P<eager_run>\d+) decode_wait_ms=(?P<decode_wait>\d+) "
                  r"source_wait_ms=(?P<source>\d+) peer_wait_ms=(?P<peer>\d+) copy_ms=(?P<copy>\d+) combine_ms=(?P<combine>\d+) "
                  r"worst_ms=(?P<worst>\d+) worst_request=(?P<request>\d+) worst_epochs=(?P<main>\d+)/(?P<hc>\d+) "
                  r"worst_us=(?P<parts>\d+/\d+/\d+/\d+/\d+/\d+)")


def windows(path):
    with open(path, errors="replace") as log:
        return [match for match in map(LINE.search, log) if match]


def total(rows, field):
    return sum(int(row.group(field)) for row in rows)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def rank_line(rank, rows):
    waves, graph, prefill = total(rows, "waves"), total(rows, "graph"), total(rows, "prefill")
    mean = {part: ratio(total(rows, part), waves) for part in INTERVALS + COLLECTIVE}
    other = mean["run"] - sum(mean[part] for part in COLLECTIVE)
    path = "+".join(sorted({PATHS.get(row.group("path"), row.group("path")) for row in rows}))
    busy = "/".join(str(total(rows, reason)) for reason in BUSY)
    return (f"{rank:>4} {len(rows):>7} {waves:>6} {ratio(total(rows, 'rows'), waves):>9.1f} {100.0 * ratio(prefill, waves):>8.1f} {100.0 * ratio(graph, waves):>6.1f} "
            f"{path:>8} {ratio(total(rows, 'retries'), waves):>12.1f} {busy:>24} | " + " ".join(f"{mean[part]:6.1f}" for part in INTERVALS) + " | "
            + " ".join(f"{mean[part]:6.1f}" for part in COLLECTIVE) + f" {other:6.1f} | {ratio(total(rows, 'graph_run'), graph):9.1f} "
            f"{ratio(total(rows, 'eager_run'), waves - graph):9.1f} {ratio(total(rows, 'decode_wait'), waves - prefill):11.1f} {total(rows, 'captures'):>8} "
            f"{total(rows, 'capture'):>10} | {max(int(row.group('wait99')) for row in rows):>9} {max(int(row.group('run99')) for row in rows):>9}")


def main():
    parser = argparse.ArgumentParser(description="Combine every rank's G5N-WAVE-TIMING lines into a per-wave time budget.")
    parser.add_argument("logs", nargs="+", metavar="RANK=PATH", help="the residentd stderr log of each rank")
    arguments = parser.parse_args()
    worst = []
    print("per-wave means in ms; wait overlaps the run of the chain the wave queued behind; other = run - peer - source - copy - combine")
    print("busy = BUSY retries by what blocked the claim: chain/stream/slot/lanes/other; path = graph path off, on, or degraded after a GRAPH-FAILED")
    print(f"{'rank':>4} {'windows':>7} {'waves':>6} {'rows/wave':>9} {'prefill%':>8} {'graph%':>6} {'path':>8} {'retries/wave':>12} {'busy':>24} | "
          + " ".join(f"{part:>6}" for part in INTERVALS) + " | " + " ".join(f"{part:>6}" for part in COLLECTIVE + ("other",))
          + f" | {'graph_run':>9} {'eager_run':>9} {'decode_wait':>11} {'captures':>8} {'capture_ms':>10} | {'wait99_us':>9} {'run99_us':>9}")
    for argument in arguments.logs:
        rank, path = argument.split("=", 1)
        rows = windows(path)
        if not rows:
            print(f"{int(rank):>4} no G5N-WAVE-TIMING lines")
            continue
        print(rank_line(int(rank), rows))
        worst.extend((int(row.group("worst")), int(rank), row) for row in rows)
    print("\nslowest waves (ms, rank, request, main/hc epoch, idle/wait/key/setup/run/post us)")
    for waited, rank, row in sorted(worst, key=lambda item: item[0], reverse=True)[:12]:
        print(f"{waited:>6} {rank:>4} {row.group('request'):>12} {row.group('main'):>10}/{row.group('hc'):<10} {row.group('parts')}")


if __name__ == "__main__":
    main()
