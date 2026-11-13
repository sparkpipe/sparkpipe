#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / "spec_offline" / "models.json"
PAIR = re.compile(r"(\w+)=(\S+)")
DECODE_KINDS = ("single", "chain", "verify")
CONTENDED_GAP = 0.5


def pairs(line: str) -> dict:
    return dict(PAIR.findall(line))


def number(values: dict, name: str) -> float:
    try:
        return float(values[name])
    except (KeyError, ValueError):
        raise ValueError(f"no number for {name}") from None


def read_api(lines) -> dict:
    frames: dict = {}
    for line in lines:
        if not line.startswith("batch_step "):
            continue
        values = pairs(line)
        try:
            frame = {name: number(values, name) for name in ("rows", "steps", "kept", "service_ms", "wall_ms", "gap_ms", "draft_wait_ms", "context")}
        except ValueError:
            continue
        frames.setdefault(values.get("kind", "?"), []).append(frame)
    return frames


def read_rank(lines) -> dict:
    result: dict = {"windows": [], "chains": [], "waves": [], "arrivals": [], "rank": None, "bytes": None}
    for line in lines:
        values = pairs(line)
        try:
            if line.startswith("STAGE-STEP-TIMING "):
                result["windows"].append({name: number(values, name) for name in ("steps", "submit_us", "source_wait_us", "peer_wait_us", "copy_us", "combine_us")})
                if result["rank"] is None and values.get("rank", "").isdigit():
                    result["rank"] = int(values["rank"])
            elif line.startswith("STAGE-STEP-ARRIVAL "):
                result["arrivals"].append({"steps": number(values, "steps"), "first_us": number(values, "first_us"),
                                           "peers": [float(value) for value in values["peer_us"].split(",")]})
            elif line.startswith("WD-SPINE-PRIVATE ") and result["bytes"] is None:
                result["bytes"] = number(values, "bytes")
            elif line.startswith("STAGE-CHAIN-HOST "):
                result["chains"].append({name: number(values, name) for name in ("chains", "mean_steps", "admit_ms", "submit_ms", "capture_ms", "sync_ms", "finish_ms")})
            elif line.startswith("STAGE-WAVE-HOST "):
                result["waves"].append({name: number(values, name) for name in ("prefill_waves", "admit_ms", "submit_ms", "capture_ms", "sync_ms", "finish_ms")})
        except ValueError:
            continue
    return result


def weighted(rows: list, weight: str, name: str) -> float:
    total = sum(row[weight] for row in rows)
    return sum(row[name] * row[weight] for row in rows) / total if total else 0.0


def rank_step(ranks: list, rounds: float | None) -> dict | None:
    windows = [window for rank in ranks for window in rank["windows"] if window["steps"] > 0]
    if not windows:
        return None
    step = {name: weighted(windows, "steps", name) / 1000.0 for name in ("submit_us", "source_wait_us", "peer_wait_us", "copy_us", "combine_us")}
    return {"steps": int(sum(window["steps"] for window in windows)), "windows": len(windows), "step_ms": step["submit_us"],
            "peer_wait_ms": step["peer_wait_us"], "source_wait_ms": step["source_wait_us"], "copy_ms": step["copy_us"], "combine_ms": step["combine_us"],
            "collective_ms": step["peer_wait_us"] + step["source_wait_us"] + step["copy_us"] + step["combine_us"],
            "rounds_per_step": rounds, "peer_wait_us_per_round": 1000.0 * step["peer_wait_us"] / rounds if rounds else None}


def rank_table(ranks: list) -> list:
    table = []
    for index, rank in enumerate(ranks):
        windows = [window for window in rank["windows"] if window["steps"] > 0]
        if not windows:
            continue
        step = {name: weighted(windows, "steps", name) / 1000.0 for name in ("submit_us", "source_wait_us", "peer_wait_us", "copy_us", "combine_us")}
        row = {"rank": rank["rank"] if rank["rank"] is not None else index, "gb": rank["bytes"] / 1e9 if rank["bytes"] else None,
               "steps": int(sum(window["steps"] for window in windows)), "step_ms": step["submit_us"], "peer_wait_ms": step["peer_wait_us"],
               "own_ms": step["submit_us"] - step["peer_wait_us"] - step["source_wait_us"] - step["copy_us"] - step["combine_us"]}
        arrivals = [arrival for arrival in rank["arrivals"] if arrival["steps"] > 0 and len(arrival["peers"]) > 1]
        if arrivals:
            count = min(len(arrival["peers"]) for arrival in arrivals)
            total = sum(arrival["steps"] for arrival in arrivals)
            peers = [sum(arrival["peers"][peer] * arrival["steps"] for arrival in arrivals) / total for peer in range(count)]
            others = sorted(value for peer, value in enumerate(peers) if peer != row["rank"] and value > 0.0)
            if others:
                row.update({"first_arrival_us": sum(arrival["first_us"] * arrival["steps"] for arrival in arrivals) / total,
                            "median_arrival_us": others[len(others) // 2], "last_arrival_us": others[-1]})
        table.append(row)
    return sorted(table, key=lambda row: row["rank"])


def rank_host(ranks: list, key: str, weight: str) -> dict | None:
    rows = [row for rank in ranks for row in rank[key] if row[weight] > 0]
    if not rows:
        return None
    host = {name: weighted(rows, weight, name) for name in ("admit_ms", "submit_ms", "capture_ms", "sync_ms", "finish_ms")}
    host["frames"] = int(sum(row[weight] for row in rows))
    host["outside_submit_ms"] = host["admit_ms"] + host["capture_ms"] + host["sync_ms"] + host["finish_ms"]
    return host


def frame_kind(kind: str, frames: list) -> dict:
    count = len(frames)
    tokens = sum(frame["kept"] for frame in frames)
    wall = sum(frame["wall_ms"] for frame in frames)
    gap = sum(frame["gap_ms"] for frame in frames)
    service = sum(frame["service_ms"] for frame in frames)
    entry = {"frames": count, "tokens": int(tokens), "rows_per_frame": sum(frame["rows"] for frame in frames) / count, "tokens_per_frame": tokens / count,
             "service_ms": service / count, "host_ms": (wall - service) / count, "gap_ms": gap / count,
             "draft_wait_ms": sum(frame["draft_wait_ms"] for frame in frames) / count, "context": sum(frame["context"] for frame in frames) / count,
             "tokens_per_second": 1000.0 * tokens / (wall + gap) if wall + gap > 0 else None}
    if kind not in DECODE_KINDS:
        return entry
    quiet = [frame for frame in frames if frame["gap_ms"] <= CONTENDED_GAP * frame["service_ms"]]
    entry["contended_frames"] = count - len(quiet)
    passes = sum(1.0 if kind == "verify" else frame["steps"] for frame in quiet)
    if passes > 0:
        quiet_service = sum(frame["service_ms"] for frame in quiet)
        entry["per_pass"] = {"passes": passes, "tokens": sum(frame["kept"] for frame in quiet) / passes, "service_ms": quiet_service / passes,
                             "host_ms": (sum(frame["wall_ms"] for frame in quiet) - quiet_service) / passes, "gap_ms": sum(frame["gap_ms"] for frame in quiet) / passes}
    return entry


def summarize(api: dict, ranks: list, node_gb: float | None, read_gb_s: float, rounds: float | None = None) -> dict:
    summary: dict = {"kinds": {kind: frame_kind(kind, frames) for kind, frames in sorted(api.items())}, "rank_step": rank_step(ranks, rounds),
                     "rank_chain_host": rank_host(ranks, "chains", "chains"), "rank_wave_host": rank_host(ranks, "waves", "prefill_waves"),
                     "ranks": len(ranks), "read_gb_s": read_gb_s, "node_gb_per_token": node_gb, "rank_table": rank_table(ranks)}
    floor = 1000.0 * node_gb / read_gb_s if node_gb else None
    summary["floor_ms"] = floor
    summary["floor_tokens_per_second"] = 1000.0 / floor if floor else None
    step = summary["rank_step"]
    share = step["collective_ms"] / step["step_ms"] if step is not None and step["step_ms"] > 0 else None
    if summary["rank_table"]:
        slowest = max(summary["rank_table"], key=lambda row: row["own_ms"])
        summary["slowest_rank"] = slowest["rank"]
        if slowest["step_ms"] > 0:
            share = (slowest["step_ms"] - slowest["own_ms"]) / slowest["step_ms"]
    summary["collective_share"] = share
    for kind in DECODE_KINDS:
        entry = summary["kinds"].get(kind)
        if entry is None or "per_pass" not in entry:
            continue
        per_pass = entry["per_pass"]
        total = per_pass["service_ms"] + per_pass["host_ms"] + per_pass["gap_ms"]
        parts = {}
        rest = per_pass["service_ms"]
        if floor:
            parts["floor_ms"] = floor
            rest -= floor
        if share is not None:
            parts["collective_ms"] = share * per_pass["service_ms"]
            rest -= parts["collective_ms"]
        parts["rank_other_ms" if parts else "service_ms"] = rest
        parts["host_ms"] = per_pass["host_ms"]
        parts["gap_ms"] = per_pass["gap_ms"]
        entry["decomposition"] = {"total_ms": total, **parts}
        entry["ms_per_token"] = total / per_pass["tokens"] if per_pass["tokens"] > 0 else None
        if floor and total > 0:
            entry["of_roofline"] = floor * per_pass["tokens"] / total
    return summary


def render(summary: dict, label: str) -> list:
    lines = [f"roofline meter: {label}"]
    lines.append(f"{'kind':8s} {'frames':>7s} {'tokens':>8s} {'tok/s':>8s} {'rows':>7s} {'tokens/frame':>12s} {'service ms':>11s} {'host ms':>8s} {'gap ms':>7s} {'draft ms':>9s} {'context':>8s}")
    for kind, entry in summary["kinds"].items():
        rate = entry["tokens_per_second"]
        lines.append(f"{kind:8s} {entry['frames']:7d} {entry['tokens']:8d} {rate if rate is not None else 0.0:8.1f} {entry['rows_per_frame']:7.1f} {entry['tokens_per_frame']:12.2f} "
                     f"{entry['service_ms']:11.2f} {entry['host_ms']:8.2f} {entry['gap_ms']:7.2f} {entry['draft_wait_ms']:9.2f} {entry['context']:8.0f}")
    step = summary["rank_step"]
    if step is not None:
        rounds = (f" over {step['rounds_per_step']:.0f} rounds, {step['peer_wait_us_per_round']:.1f} us a round" if step["rounds_per_step"] else "")
        lines.append(f"rank step ({summary['ranks']} rank logs, {step['steps']} steps of every kind): {step['step_ms']:.2f} ms, {100.0 * step['collective_ms'] / step['step_ms']:.0f}% in collectives: "
                     f"{step['peer_wait_ms']:.2f} peer wait{rounds} + {step['source_wait_ms']:.2f} source wait + {step['copy_ms']:.2f} copy + {step['combine_ms']:.2f} combine")
    table = summary["rank_table"]
    if len(table) > 1:
        slow = max(table, key=lambda row: row["own_ms"])
        fast = min(table, key=lambda row: row["own_ms"])
        sizes = [row["gb"] for row in table if row["gb"]]
        weights = f"; weights {min(sizes):.3f} to {max(sizes):.3f} GB a rank" if sizes else ""
        lines.append(f"ranks: own work {fast['own_ms']:.2f} ms a step on rank {fast['rank']} to {slow['own_ms']:.2f} on rank {slow['rank']}, "
                     f"which the others wait for and whose {100.0 * summary['collective_share']:.0f}% in collectives the split below uses{weights}")
        lines.append(f"{'rank':>5s} {'GB':>7s} {'own ms':>8s} {'peer wait ms':>13s} {'step ms':>8s}")
        for row in table:
            size = f"{row['gb']:7.3f}" if row["gb"] else f"{'':7s}"
            lines.append(f"{row['rank']:5d} {size} {row['own_ms']:8.2f} {row['peer_wait_ms']:13.2f} {row['step_ms']:8.2f}")
    landed = [row for row in table if "first_arrival_us" in row]
    if landed:
        rounds = step["rounds_per_step"] if step is not None and step["rounds_per_step"] else None
        scale = rounds if rounds else 1.0
        span = lambda name, unit=1.0: f"{min(row[name] for row in landed) * unit / scale:.1f} to {max(row[name] for row in landed) * unit / scale:.1f}"
        lines.append(f"peers land after a {'round' if rounds else 'step'}'s wait starts, in us, fastest to slowest rank log: first {span('first_arrival_us')}, "
                     f"median {span('median_arrival_us')}, last on average {span('last_arrival_us')}, all {span('peer_wait_ms', 1000.0)}")
    for name, title in (("rank_chain_host", "rank host per chain"), ("rank_wave_host", "rank host per prefill wave")):
        host = summary[name]
        if host is not None:
            lines.append(f"{title} ({host['frames']} frames): submit {host['submit_ms']:.2f} ms; outside it {host['outside_submit_ms']:.2f} ms "
                         f"(admit {host['admit_ms']:.2f}, capture {host['capture_ms']:.2f}, sync {host['sync_ms']:.2f}, finish {host['finish_ms']:.2f})")
    if summary["floor_ms"]:
        lines.append(f"memory floor: {summary['node_gb_per_token']:.3f} GB a node a pass / {summary['read_gb_s']:.0f} GB/s = {summary['floor_ms']:.2f} ms a pass, "
                     f"{summary['floor_tokens_per_second']:.1f} tok/s at one token a pass")
    names = {"floor_ms": "memory floor", "collective_ms": "collectives", "rank_other_ms": "rank over both", "service_ms": "service",
             "host_ms": "host and wire", "gap_ms": "gap between frames"}
    for kind in DECODE_KINDS:
        entry = summary["kinds"].get(kind)
        if entry is None or "decomposition" not in entry:
            continue
        parts = entry["decomposition"]
        total = parts["total_ms"]
        share = ", ".join(f"less {-value:.2f} ({100.0 * -value / total:.0f}%) of the floor read during the waits" if name == "rank_other_ms" and value < 0.0
                          else f"{names[name]} {value:.2f} ({100.0 * value / total:.0f}%)" for name, value in parts.items() if name != "total_ms")
        roofline = f"; {100.0 * entry['of_roofline']:.0f}% of the one-token roofline" if "of_roofline" in entry else ""
        contended = f"; {entry['contended_frames']} frames that waited behind other work left out" if entry["contended_frames"] else ""
        lines.append(f"{kind}: {total:.2f} ms a pass for {entry['per_pass']['tokens']:.2f} tokens, {entry['ms_per_token']:.2f} ms a token{roofline}: {share}{contended}")
    return lines


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roofline_meter", description="Where a lane's frame time goes, and how far its decode is from the memory roofline, "
                                     "from the API log's batch_step lines and the ranks' STAGE-STEP-TIMING, STAGE-CHAIN-HOST and STAGE-WAVE-HOST lines")
    parser.add_argument("--api-log", required=True)
    parser.add_argument("--rank-log", action="append", default=[])
    parser.add_argument("--node-gb-per-token", type=float, help="bytes one node reads for one decode step, in GB; without it no roofline is stated")
    parser.add_argument("--read-gb-s", type=float, help="a node's measured streaming read rate; default: the registry's")
    parser.add_argument("--rounds-per-step", type=float, help="collective rounds one decode step issues, from the model's schedule; gives the peer wait a round")
    parser.add_argument("--label", default="lane")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.node_gb_per_token is not None and args.node_gb_per_token <= 0.0:
        parser.error("--node-gb-per-token must be positive")
    read_gb_s = args.read_gb_s if args.read_gb_s is not None else json.loads(REGISTRY.read_text())["peak_read_gb_s"]
    if read_gb_s <= 0.0:
        parser.error("--read-gb-s must be positive")
    if args.rounds_per_step is not None and args.rounds_per_step <= 0.0:
        parser.error("--rounds-per-step must be positive")
    with open(args.api_log, errors="replace") as handle:
        api = read_api(handle)
    if not api:
        print(f"roofline_meter: {args.api_log} has no batch_step line", file=sys.stderr)
        return 1
    ranks = []
    for path in args.rank_log:
        with open(path, errors="replace") as handle:
            ranks.append(read_rank(handle))
    summary = summarize(api, ranks, args.node_gb_per_token, read_gb_s, args.rounds_per_step)
    print(json.dumps(summary, indent=1) if args.json else "\n".join(render(summary, args.label)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
