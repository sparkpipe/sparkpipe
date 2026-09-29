import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["src/spark_speculation_ngram_draft.c", "src/spark_speculation_lookup_draft.c"]
MAX_TOKENS = 32


class Request(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("descriptor_bytes", ctypes.c_uint32),
                ("requested_token_count", ctypes.c_uint32), ("active_sequence_index", ctypes.c_uint32),
                ("priority", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("request_id", ctypes.c_uint64),
                ("sequence_id", ctypes.c_uint64), ("sequence_position", ctypes.c_uint64),
                ("tap_generation", ctypes.c_uint64)]


class Result(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("descriptor_bytes", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("token_count", ctypes.c_uint32),
                ("confidence_milli", ctypes.c_uint32 * MAX_TOKENS), ("token_ids", ctypes.c_uint32 * MAX_TOKENS)]


class Ngram(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in ("lane_count", "lane_capacity", "min_match", "max_match", "scan_limit",
                                               "bucket_mask")] + \
               [("tokens", ctypes.c_void_p), ("lengths", ctypes.c_void_p), ("sequence_ids", ctypes.c_void_p),
                ("chain", ctypes.c_void_p), ("heads", ctypes.c_void_p)]


class Lookup(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in ("lane_count", "lane_capacity", "min_match", "max_match")] + \
               [("tokens", ctypes.c_void_p), ("lengths", ctypes.c_void_p), ("sequence_ids", ctypes.c_void_p)]


def build(cache_dir):
    digest = hashlib.sha256()
    for source in SOURCES:
        with open(os.path.join(ROOT, source), "rb") as fh:
            digest.update(fh.read())
    os.makedirs(cache_dir, exist_ok=True)
    library = os.path.join(cache_dir, f"draftd_ngram_{digest.hexdigest()[:16]}.so")
    if not os.path.exists(library):
        subprocess.run(["cc", "-O2", "-std=gnu11", "-shared", "-fPIC", "-I" + os.path.join(ROOT, "include"), "-o",
                        library + ".partial"] + [os.path.join(ROOT, s) for s in SOURCES], check=True)
        os.replace(library + ".partial", library)
    return ctypes.CDLL(library)


def reference_next(history, suffix, min_match, max_match, scan_limit):
    for order in range(min(len(suffix), max_match), min_match - 1, -1):
        context = suffix[len(suffix) - order:]
        seen = []
        for end in range(len(history) - 2, order - 2, -1):
            if history[end + 1 - order:end + 1] == context:
                seen.append(history[end + 1])
                if len(seen) == scan_limit:
                    break
        if seen:
            counts = {}
            for token in seen:
                counts[token] = counts.get(token, 0) + 1
            best = max(counts.values())
            return next(t for t in seen if counts[t] == best), order
    return None, 0


def reference_draft(history, depth, min_match, max_match, scan_limit):
    take = min(len(history), max_match)
    suffix = list(history[len(history) - take:])
    drafts = []
    for _ in range(depth):
        token, _ = reference_next(history, suffix[len(suffix) - take:], min_match, max_match, scan_limit)
        if token is None:
            break
        drafts.append(token)
        suffix.append(token)
    return drafts


def load_streams(paths):
    streams = []
    for path in paths:
        data = json.load(open(path))
        for index, row in enumerate(data["results"]):
            streams.append({"id": f"{os.path.basename(path)}:{index}", "class": row.get("class", "unknown"),
                            "tokens": [int(t) for t in row["token_ids"]]})
    return streams


def accepted(drafts, truth):
    count = 0
    for draft, want in zip(drafts, truth):
        if draft != want:
            break
        count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description="draftd n-gram and lookup drafters on recorded greedy streams")
    parser.add_argument("--streams", nargs="+", required=True)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--min-match", type=int, default=2)
    parser.add_argument("--max-match", type=int, default=8)
    parser.add_argument("--scan-limit", type=int, default=64)
    parser.add_argument("--lookup-min-match", type=int, default=3)
    parser.add_argument("--cache", default=os.path.expanduser("~/.cache/draftd"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    lib = build(args.cache)
    ngram, lookup = Ngram(), Lookup()
    capacity = 1 << 16
    if lib.SparkSpeculationNgramDraftInitialize(ctypes.byref(ngram), 1, capacity, args.min_match, args.max_match,
                                                args.scan_limit):
        raise SystemExit("NGRAM-FAIL initialize")
    if lib.SparkSpeculationLookupDraftInitialize(ctypes.byref(lookup), 1, capacity, args.lookup_min_match,
                                                 args.max_match):
        raise SystemExit("NGRAM-FAIL lookup initialize")
    streams = load_streams(args.streams)
    report = {"config": vars(args), "classes": {}, "positions": 0, "parity_mismatches": [], "call_us": []}
    request, result = Request(), Result()
    for sequence, stream in enumerate(streams, start=1):
        tokens = stream["tokens"]
        cls = report["classes"].setdefault(stream["class"], {
            "positions": 0, "ngram": {"rounds_with_draft": 0, "accepted": 0, "per_position": [0] * args.depth},
            "lookup": {"rounds_with_draft": 0, "accepted": 0, "per_position": [0] * args.depth}})
        per_anchor = {"ngram": [], "lookup": []}
        for anchor in range(len(tokens) - 1):
            one = (ctypes.c_uint32 * 1)(tokens[anchor])
            lib.SparkSpeculationNgramDraftObserve(ctypes.byref(ngram), 0, sequence, anchor, one, 1)
            lib.SparkSpeculationLookupDraftObserve(ctypes.byref(lookup), 0, sequence, anchor, one, 1)
            request.requested_token_count = args.depth
            request.active_sequence_index = 0
            request.sequence_id = sequence
            request.sequence_position = anchor
            truth = tokens[anchor + 1:anchor + 1 + args.depth]
            for name, fn, state in (("ngram", lib.SparkSpeculationNgramDraftTokens, ngram),
                                    ("lookup", lib.SparkSpeculationLookupDraftTokens, lookup)):
                started = time.perf_counter()
                fn(ctypes.byref(state), ctypes.byref(request), ctypes.byref(result))
                if name == "ngram":
                    report["call_us"].append((time.perf_counter() - started) * 1e6)
                drafts = list(result.token_ids[:result.token_count])
                if name == "ngram":
                    want = reference_draft(tokens[:anchor + 1], args.depth, args.min_match, args.max_match,
                                           args.scan_limit)
                    if drafts != want:
                        report["parity_mismatches"].append({"stream": stream["id"], "anchor": anchor,
                                                            "c": drafts, "reference": want})
                per_anchor[name].append(drafts)
                row = cls[name]
                if drafts:
                    row["rounds_with_draft"] += 1
                hit = accepted(drafts, truth)
                row["accepted"] += hit
                for i in range(hit):
                    row["per_position"][i] += 1
            cls["positions"] += 1
            report["positions"] += 1
        for name in ("ngram", "lookup"):
            row = cls[name]
            anchor = 0
            while anchor < len(tokens) - 1:
                hit = accepted(per_anchor[name][anchor], tokens[anchor + 1:anchor + 1 + args.depth])
                row["rounds"] = row.get("rounds", 0) + 1
                row["committed"] = row.get("committed", 0) + hit + 1
                anchor += hit + 1
    for cls in report["classes"].values():
        for name in ("ngram", "lookup"):
            row = cls[name]
            row["tokens_per_round_chain"] = row["committed"] / row["rounds"]
            row["p_cumulative"] = [round(v / cls["positions"], 4) for v in row["per_position"]]
    calls = sorted(report.pop("call_us"))
    report["ngram_call_us_p50"] = calls[len(calls) // 2]
    report["ngram_call_us_p99"] = calls[int(len(calls) * 0.99)]
    parity = len(report["parity_mismatches"]) == 0
    report["verdict"] = "PASS" if parity and report["positions"] >= 1000 else "FAIL"
    json.dump(report, open(args.output, "w"), indent=1)
    for name, cls in sorted(report["classes"].items()):
        print(f"NGRAM-EVAL class={name} positions={cls['positions']} "
              f"ngram_tau={cls['ngram']['tokens_per_round_chain']:.3f} lookup_tau={cls['lookup']['tokens_per_round_chain']:.3f} "
              f"ngram_p={cls['ngram']['p_cumulative'][:4]} lookup_p={cls['lookup']['p_cumulative'][:4]}")
    print(f"NGRAM-EVAL {report['verdict']} positions={report['positions']} parity_mismatches={len(report['parity_mismatches'])} "
          f"call_us_p50={report['ngram_call_us_p50']:.1f} p99={report['ngram_call_us_p99']:.1f}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
