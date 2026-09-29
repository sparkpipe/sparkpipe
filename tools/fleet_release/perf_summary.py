#!/usr/bin/env python3
import argparse
import json
import os
import sys

METRIC_FIELDS = {"file", "key", "scale", "prev", "lo", "hi", "tol", "lower_is_better", "gate", "note", "batch"}
ROOFLINE_FIELDS = ("bytes_per_token", "read_peak", "rounds", "round_floor_s", "topology_bytes", "link", "b1_ms_metrics")


class ConfigError(Exception):
    pass


def load_expect(path):
    with open(path) as f:
        doc = json.load(f)
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ConfigError("expectations need a metrics object")
    for name, m in metrics.items():
        unknown = set(m) - METRIC_FIELDS
        if unknown:
            raise ConfigError(f"{name}: unknown fields {sorted(unknown)}")
        for key in ("file", "key", "prev", "lo", "hi", "tol", "lower_is_better", "gate"):
            if key not in m:
                raise ConfigError(f"{name}: missing {key}")
        if not m["lo"] <= m["hi"]:
            raise ConfigError(f"{name}: lo > hi")
    roof = doc.get("roofline")
    if roof is not None:
        missing = [k for k in ROOFLINE_FIELDS if k not in roof]
        if missing:
            raise ConfigError(f"roofline: missing {' '.join(missing)}")
    return doc


def read_metric(results, m):
    try:
        with open(os.path.join(results, m["file"])) as f:
            cur = json.load(f)
    except (OSError, ValueError):
        return None
    for k in m["key"]:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if not isinstance(cur, (int, float)):
        return None
    return cur * m.get("scale", 1)


def peer_ms_from_budget(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) > 12 and fields[1] == "1" and fields[0] != "path":
                try:
                    return float(fields[7])
                except ValueError:
                    return None
    return None


def verdict(v, m):
    lower = m["lower_is_better"]
    if m["lo"] <= v <= m["hi"]:
        word = "IN-RANGE"
    elif (v < m["lo"]) != lower:
        word = "SLOWER" if lower else "BELOW"
    else:
        word = "BETTER"
    regress = v > m["prev"] * m["tol"] if lower else v < m["prev"] * m["tol"]
    if regress and m["gate"]:
        return "REGRESSION", True
    return word, False


def roofline_line(roof, ms, peer_ms):
    step = ms / 1e3
    mem = roof["bytes_per_token"] / step / roof["read_peak"] * 100
    ceiling = roof["read_peak"] / roof["bytes_per_token"]
    if peer_ms:
        bw = roof["topology_bytes"] / roof["link"] / (peer_ms / 1e3) * 100
        lat = roof["rounds"] * roof["round_floor_s"] / (peer_ms / 1e3) * 100
        transport = f"transport bw {bw:.0f}% + latency {lat:.0f}%"
    else:
        transport = "transport: peer time unavailable (no CHAIN-TIME budget)"
    return (f"roofline @B=1 (end-to-end fleet per token): memory {mem:.0f}% (ceiling {ceiling:.0f} tok/s) | compute {roof.get('compute', '~1-2%')} | {transport} "
            f"(inputs: {roof['bytes_per_token'] / 1e9:.2f} GB/token/node, {roof['read_peak'] / 1e9:.0f} GB/s read peak, {ms:.2f} ms/token, "
            f"{roof['rounds']} rounds x {roof['round_floor_s'] * 1e6:.0f} us floor, ~{roof['topology_bytes'] / 1e6:.0f} MB topology bytes vs {roof['link'] / 1e9:.0f} GB/s, peer {peer_ms} ms)")


def summarize(doc, results, out=print):
    metrics = doc["metrics"]
    got = {name: read_metric(results, m) for name, m in metrics.items()}
    bad = 0
    out(f"{'metric':12s} {'measured':>9s} {'prev':>7s} {'expected':>13s}  verdict  note")
    for name, m in metrics.items():
        v = got[name]
        if v is None:
            out(f"{name:12s} {'-':>9s} {m['prev']:7.2f} {m['lo']:6.2f}-{m['hi']:<6.2f}  MISSING  {m.get('note', '')}")
            bad += 1
            continue
        word, failed = verdict(v, m)
        bad += failed
        out(f"{name:12s} {v:9.2f} {m['prev']:7.2f} {m['lo']:6.2f}-{m['hi']:<6.2f}  {word:8s} {m.get('note', '')}")
    roof = doc.get("roofline")
    if roof:
        peer = peer_ms_from_budget(os.path.join(results, roof.get("budget_file", "budget.txt")))
        for name in roof["b1_ms_metrics"]:
            if got.get(name):
                out(f"{name}: " + roofline_line(roof, got[name], peer))
    for name, m in metrics.items():
        if "batch" in m and got.get(name):
            b = m["batch"]
            out(f"{name}: roofline @B={b}: aggregate {got[name]:.1f} tok/s; memory ceiling at B={b} needs the touched-expert bytes per step (ROOFLINE_REPORTING.md); effective {b / got[name] * 1e3:.1f} ms per fleet step of {b} tokens (prefill included)")
    out(f"PERF-SUMMARY regressions/missing={bad}")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare a release's perf results with its expectations and print roofline lines.")
    ap.add_argument("--expect", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args(argv)
    try:
        doc = load_expect(args.expect)
    except (OSError, ValueError, ConfigError) as e:
        print(f"perf_summary.py: {e}", file=sys.stderr)
        return 2
    return summarize(doc, args.results)


if __name__ == "__main__":
    sys.exit(main())
