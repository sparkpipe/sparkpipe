#!/usr/bin/env python3
import argparse
import re
import statistics

LINE = re.compile(r"WD-MESH-TIMING posts=(\d+) post_us=(\d+)/(\d+) ship_us=(\d+)/(\d+) credits=(\d+) credit_us=(\d+)/(\d+) gates=(\d+) gate_us=(\d+)/(\d+) self=(\d+) peers=(\S*)")


def windows(path):
    with open(path, errors="replace") as log:
        return [match for match in map(LINE.search, log) if match]


def median(rows, group):
    return int(statistics.median(int(row.group(group)) for row in rows))


def peers(row):
    for item in filter(None, row.group(13).split(",")):
        sender, values = item.split(":")
        p50, p99, last = values.split("/")
        yield int(sender), int(p50), int(p99), int(last)


def main():
    parser = argparse.ArgumentParser(description="Combine every rank's WD-MESH-TIMING lines into one latency table.")
    parser.add_argument("logs", nargs="+", metavar="RANK=PATH", help="the weightd stderr log of each rank")
    arguments = parser.parse_args()
    lag, last, ranks = {}, {}, []
    print("rank windows post_us p50/p99 ship_us p50/p99 credit_us p50/p99 gate_us p50/p99 (medians over 10 s windows)")
    for argument in arguments.logs:
        rank, path = argument.split("=", 1)
        rows = windows(path)
        ranks.append(int(rank))
        if not rows:
            print(f"{rank:>4} no WD-MESH-TIMING lines")
            continue
        print(f"{rank:>4} {len(rows):>7} {median(rows, 2):>7}/{median(rows, 3):<7} {median(rows, 4):>7}/{median(rows, 5):<7} {median(rows, 7):>9}/{median(rows, 8):<7} {median(rows, 10):>7}/{median(rows, 11):<7}")
        for row in rows:
            last[int(rank)] = last.get(int(rank), 0) + int(row.group(12))
            for sender, p50, _, count in peers(row):
                lag.setdefault((int(rank), sender), []).append(p50)
                last[sender] = last.get(sender, 0) + count
    senders = sorted({sender for _, sender in lag})
    print("\narrival lag p50 in us after the receiver's own publish (rows receive, columns send)")
    print("recv " + " ".join(f"{sender:>6}" for sender in senders))
    for rank in ranks:
        print(f"{rank:>4} " + " ".join(f"{int(statistics.median(lag[(rank, sender)])):>6}" if (rank, sender) in lag else "     -" for sender in senders))
    total = sum(last.values())
    if total:
        print("\nshare of all gates each rank closed, as the last sender or by reaching its own gate last: " + ", ".join(f"{rank}:{100.0 * last.get(rank, 0) / total:.1f}%" for rank in sorted(set(senders) | set(last))))


if __name__ == "__main__":
    main()
