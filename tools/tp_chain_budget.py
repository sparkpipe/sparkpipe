#!/usr/bin/env python3
import argparse
import collections
import re
import statistics

CHAIN = re.compile(r"CHAIN-TIME slot=(?P<slot>\d+) path=(?P<path>\w*) steps=(?P<steps>\d+) (?:rows=(?P<rows>\d+) epoch=(?P<epoch>\d+) )?status=(?P<status>-?\d+) "
                   r"total_ms=(?P<total>[\d.]+) walk_ms=(?P<walk>[\d.]+)")
REPLAY = re.compile(r"GRAPH-REPLAY-TIME slot=(?P<slot>\d+) wall_ns=(?P<wall>\d+)")
COLLECTIVE = re.compile(r"COLLECTIVE-GPU-TIME slot=(?P<slot>\d+) source_wait_ms=(?P<source>[\d.]+) peer_wait_ms=(?P<peer>[\d.]+) copy_ms=(?P<copy>[\d.]+) combine_ms=(?P<combine>[\d.]+)")
KEY = re.compile(r"CKEY-(?:CELL-)?(?:WRITE|ADOPT) rank=\d+ epoch=(?P<epoch>\d+)")
CAPTURE = re.compile(r"GRAPH-CAPTURE-OK rows=(?P<rows>\d+)")


def chain_record(match, replays, epoch, rows):
    return {"path": match.group("path") or "none", "steps": int(match.group("steps")), "status": int(match.group("status")),
            "rows": int(match.group("rows")) if match.group("rows") else rows,
            "epoch": int(match.group("epoch")) if match.group("epoch") else epoch,
            "total": float(match.group("total")), "walk": float(match.group("walk")), "replays": replays}


def parse(path):
    chains, replays, first_epoch, last = {}, collections.defaultdict(list), {}, {}
    epoch, capture = None, None
    with open(path, errors="replace") as log:
        for line in log:
            match = KEY.search(line)
            if match:
                epoch = int(match.group("epoch"))
                continue
            match = CAPTURE.search(line)
            if match:
                capture = int(match.group("rows"))
                continue
            match = REPLAY.search(line)
            if match:
                slot = match.group("slot")
                first_epoch.setdefault(slot, epoch)
                replays[slot].append(int(match.group("wall")) / 1e6)
                continue
            match = CHAIN.search(line)
            if match:
                slot = match.group("slot")
                record = chain_record(match, replays.pop(slot, []), first_epoch.pop(slot, epoch), capture if match.group("path") == "graph" else None)
                capture = None
                if record["epoch"] is not None and record["status"] == 0:
                    chains[record["epoch"]] = record
                    last[slot] = record
                continue
            match = COLLECTIVE.search(line)
            if match and match.group("slot") in last:
                record = last.pop(match.group("slot"))
                record.update({field: float(match.group(field)) for field in ("source", "peer", "copy", "combine")})
    return chains


def group_name(record):
    return (record["path"], record["rows"] if record["rows"] is not None else 0)


def aligned(ranks):
    epochs = set.intersection(*[set(chains) for chains in ranks.values()])
    groups = collections.defaultdict(list)
    for epoch in sorted(epochs):
        records = {rank: chains[epoch] for rank, chains in ranks.items()}
        if all("peer" in record for record in records.values()) and len({(record["path"], record["steps"]) for record in records.values()}) == 1:
            groups[group_name(next(iter(records.values())))].append(records)
    return groups


def chain_budget(records):
    steps = next(iter(records.values()))["steps"]
    peers = {rank: record["peer"] / steps for rank, record in records.items()}
    mean = lambda field: statistics.mean(field(record) for record in records.values()) / steps
    budget = {"steps": steps, "chain": mean(lambda record: record["total"]), "walk": mean(lambda record: record["walk"]),
              "peer": statistics.mean(peers.values()), "floor": min(peers.values()), "last": min(peers, key=peers.get),
              "other": mean(lambda record: record["source"] + record["copy"] + record["combine"])}
    budget["skew"] = budget["peer"] - budget["floor"]
    budget["run"] = mean(lambda record: sum(record["replays"])) if all(record["replays"] for record in records.values()) else budget["chain"]
    budget["compute"] = budget["run"] - budget["peer"] - budget["other"]
    budget["host"] = budget["chain"] - budget["run"]
    return budget


FIELDS = ("steps", "chain", "run", "compute", "peer", "floor", "skew", "other", "host", "walk")


def group_line(name, budgets):
    medians = {field: statistics.median(budget[field] for budget in budgets) for field in FIELDS}
    last = collections.Counter(budget["last"] for budget in budgets).most_common(3)
    rows = str(name[1]) if name[1] else "?"
    return ("%-8s %4s %6d " % (name[0], rows, len(budgets)) + " ".join("%7.2f" % medians[field] for field in FIELDS)
            + " | " + ",".join("%s:%d" % item for item in last))


def main():
    parser = argparse.ArgumentParser(description="Align every rank's chains by collective epoch and split each decode step into compute, collective latency, rank skew and host time.")
    parser.add_argument("logs", nargs="+", metavar="RANK=PATH", help="the residentd stderr log of each rank; use all ranks of the deployment")
    arguments = parser.parse_args()
    ranks = {}
    for item in arguments.logs:
        rank, separator, path = item.partition("=")
        if not separator:
            parser.error("expected RANK=PATH, got %s" % item)
        ranks[rank] = parse(path)
    print("per-step medians in ms over chains that every rank ran; rows ? = a log without rows= on CHAIN-TIME (only chains right after a capture are known)")
    print("run = graph replay wall (or chain time for linear chains); compute = run - peer - other; peer = rank-mean device peer wait;")
    print("floor = the least-waiting rank's peer wait (latency the last-arriving rank still pays); skew = peer - floor; other = source wait + copy + combine;")
    print("host = chain - run (between graph replays); walk = host enqueue time of a linear chain; last = ranks most often arriving last")
    print("%-8s %4s %6s " % ("path", "rows", "chains") + " ".join("%7s" % field for field in FIELDS) + " | last")
    for name, group in sorted(aligned(ranks).items()):
        print(group_line(name, [chain_budget(records) for records in group]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
