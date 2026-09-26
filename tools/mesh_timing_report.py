#!/usr/bin/env python3
import argparse
import re
import statistics

LINE = re.compile(r"WD-MESH-TIMING posts=(?P<posts>\d+) post_us=(?P<post50>\d+)/(?P<post99>\d+) ship_us=(?P<ship50>\d+)/(?P<ship99>\d+) "
                  r"credits=(?P<credits>\d+) credit_us=(?P<credit50>\d+)/(?P<credit99>\d+) gates=(?P<gates>\d+) gate_us=(?P<gate50>\d+)/(?P<gate99>\d+) "
                  r"gate_ms=(?P<gate_ms>\d+) self=(?P<self>\d+) starts=(?P<starts>\d+) start_us=(?P<start50>\d+)/(?P<start99>\d+) start_ms=(?P<start_ms>\d+) "
                  r"start_self=(?P<start_self>\d+) worst_us=(?P<worst>\d+) worst_tag=(?P<epoch>\d+):(?P<round>\d+) worst_closer=(?P<closer>\d+) peers=(?P<peers>\S*)")


def windows(path):
    with open(path, errors="replace") as log:
        return [match for match in map(LINE.search, log) if match]


def median(rows, field):
    return int(statistics.median(int(row.group(field)) for row in rows))


def mean(rows, field):
    return sum(int(row.group(field)) for row in rows) / len(rows)


def peers(row):
    for item in filter(None, row.group("peers").split(",")):
        sender, values = item.split(":")
        p50, p99, last, start_last, excess = values.split("/")
        yield int(sender), int(p50), int(last), int(start_last), int(excess)


def add(table, key, value):
    table[key] = table.get(key, 0) + value


def shares(title, closed):
    total = sum(closed.values())
    if total:
        print(f"\n{title}: " + ", ".join(f"{rank}:{100.0 * closed[rank] / total:.1f}%" for rank in sorted(closed)))


def main():
    parser = argparse.ArgumentParser(description="Combine every rank's WD-MESH-TIMING lines into one latency table.")
    parser.add_argument("logs", nargs="+", metavar="RANK=PATH", help="the weightd stderr log of each rank")
    arguments = parser.parse_args()
    lag, closed, started, excess, worst, ranks = {}, {}, {}, {}, [], []
    print("rank windows post_us p50/p99 ship_us p50/p99 credit_us p50/p99 gate_us p50/p99 start_us p50/p99 | gate_ms start_ms gates (means per 10 s window)")
    for argument in arguments.logs:
        rank, path = argument.split("=", 1)
        rank = int(rank)
        rows = windows(path)
        ranks.append(rank)
        if not rows:
            print(f"{rank:>4} no WD-MESH-TIMING lines")
            continue
        print(f"{rank:>4} {len(rows):>7} {median(rows, 'post50'):>7}/{median(rows, 'post99'):<7} {median(rows, 'ship50'):>7}/{median(rows, 'ship99'):<7} "
              f"{median(rows, 'credit50'):>9}/{median(rows, 'credit99'):<7} {median(rows, 'gate50'):>7}/{median(rows, 'gate99'):<7} "
              f"{median(rows, 'start50'):>8}/{median(rows, 'start99'):<7} | {mean(rows, 'gate_ms'):7.0f} {mean(rows, 'start_ms'):8.0f} {mean(rows, 'gates'):6.0f}")
        for row in rows:
            add(closed, rank, int(row.group("self")))
            add(started, rank, int(row.group("start_self")))
            worst.append((int(row.group("worst")), rank, row.group("epoch") + ":" + row.group("round"), int(row.group("closer"))))
            for sender, p50, last, start_last, waited in peers(row):
                lag.setdefault((rank, sender), []).append(p50)
                add(closed, sender, last)
                add(started, sender, start_last)
                add(excess, sender, waited)
    senders = sorted({sender for _, sender in lag})
    print("\narrival lag p50 in us after the receiver's own publish (rows receive, columns send)")
    print("recv " + " ".join(f"{sender:>6}" for sender in senders))
    for rank in ranks:
        print(f"{rank:>4} " + " ".join(f"{int(statistics.median(lag[(rank, sender)])):>6}" if (rank, sender) in lag else "     -" for sender in senders))
    shares("share of all gates each rank closed, as the last sender or by reaching its own gate last", closed)
    shares("share of chain-start gates each rank closed", started)
    if excess:
        print("\nwait charged to each rank as the last sender (ms, all receivers): " + ", ".join(f"{rank}:{excess[rank] / 1000.0:.1f}" for rank in sorted(excess)))
    print("\nworst gates (us, receiver, epoch:round, closer)")
    for waited, rank, tag, closer in sorted(worst, reverse=True)[:10]:
        print(f"{waited:>9} {rank:>4} {tag:>16} {closer:>4}")


if __name__ == "__main__":
    main()
