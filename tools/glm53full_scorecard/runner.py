#!/usr/bin/env python3
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roofline


def build_batch(session, rows):
    cases = session["cases"]
    return {"schema_version": 1, "connect_timeout_ms": 20000, "request_capacity": len(cases),
            "max_context_tokens": session.get("max_context_tokens", 2048),
            "max_prefill_rows_per_submission": rows, "maximum_messages_per_rank_per_progress": 8,
            "maximum_new_submissions_per_progress": rows, "stop_token_ids": [],
            "requests": [{"request_id": i + 1, "sequence_id": i + 1, "priority": 0, "output_token_budget": c["budget"],
                          "prompt_token_ids": c["prompt"]} for i, c in enumerate(cases)]}


def fits(session, capacity):
    concurrent = 1 if session.get("sequential") else len(session["cases"])
    longest = max(len(c["prompt"]) + c["budget"] for c in session["cases"])
    if session.get("kind") in ("decode", "prefill") and concurrent > capacity["sequences"]:
        return f"needs {concurrent} concurrent sequences, arm has {capacity['sequences']}"
    if longest > capacity["positions"]:
        return f"needs {longest} positions, arm has {capacity['positions']}"
    return None


def read_capacity(root):
    values = {}
    for line in Path(root, "RELEASE").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return {"sequences": int(values["sequences"]), "rows": int(values["rows"]), "positions": int(values["positions"])}


def summarize_case(c, ids, ts, accepted, t0):
    gaps = [(b - a) / 1e6 for a, b in zip(ts, ts[1:])]
    return {"label": c["label"], "class": c.get("class"), "prompt_len": len(c["prompt"]), "budget": c["budget"],
            "tokens": len(ids), "token_ids": ids,
            "ttft_s": (ts[0] - accepted) / 1e9 if ts else None,
            "decode_ms_per_token": (ts[-1] - ts[0]) / 1e6 / (len(ts) - 1) if len(ts) > 1 else None,
            "inter_token_median_ms": statistics.median(gaps) if gaps else None,
            "inter_token_p95_ms": sorted(gaps)[min(len(gaps) - 1, int(len(gaps) * 0.95))] if gaps else None,
            "first_s": (ts[0] - t0) / 1e9 if ts else None, "last_s": (ts[-1] - t0) / 1e9 if ts else None}


def run_session(session_path, out_dir, fw, root, tag):
    session = json.loads(Path(session_path).read_text())
    cases = session["cases"]
    capacity = read_capacity(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = f"{tag}-{session['label']}"
    reason = fits(session, capacity)
    if reason:
        record = {"label": label, "session": session["label"], "tag": tag, "rc": None, "skipped": reason,
                  "capacity": capacity, "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(out / "results.jsonl", "a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        return 4
    batch_path = out / f"{label}.batch.json"
    batch_path.write_text(json.dumps(build_batch(session, capacity["rows"])))
    env = dict(os.environ, SPARK_GLM52_SERVING_FLAT_RANKS="16", LD_LIBRARY_PATH=f"{root}/lib")
    if session.get("sequential"):
        env["SPARK_MODEL_BATCH_SEQUENTIAL"] = "1"
    events_path, stderr_path = out / f"{label}.ndjson", out / f"{label}.stderr"
    t0 = time.monotonic_ns()
    accepted, tokens, times = {}, {}, {}
    with open(events_path, "w") as events, open(stderr_path, "w") as stderr_file:
        proc = subprocess.Popen([f"{fw}/sparkpipe_model_batch", "--deployment", f"{root}/model_resident.json", "--runtime-root", root,
                                 "--batch", str(batch_path)], stdout=subprocess.PIPE, stderr=stderr_file, text=True, bufsize=1, env=env)
        for line in proc.stdout:
            now = time.monotonic_ns()
            events.write(line)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            rid = event.get("request_id")
            if event.get("event") == "accepted":
                accepted[rid] = now
            elif event.get("event") == "token":
                tokens.setdefault(rid, []).append(event["token_id"])
                times.setdefault(rid, []).append(now)
        rc = proc.wait()
    t1 = time.monotonic_ns()
    per, failures = [], []
    for i, c in enumerate(cases):
        rid = i + 1
        ids = tokens.get(rid, [])
        record = summarize_case(c, ids, times.get(rid, []), accepted.get(rid, t0), t0)
        record["request_id"] = rid
        if c.get("require_full") and len(ids) != c["budget"]:
            record["check"] = "INVALID-SHORT"
            failures.append(f"{c['label']}: {len(ids)} of {c['budget']} tokens; a fixed-length case must not stop early")
        if c.get("expect") is not None:
            exact = ids[:len(c["expect"])] == c["expect"]
            record["check"] = "EXACT" if exact else "MISMATCH"
            if not exact:
                failures.append(f"{c['label']}: got {ids[:len(c['expect'])]} expected {c['expect']}")
        per.append(record)
    result = {"label": label, "session": session["label"], "kind": session.get("kind", session["label"]), "tag": tag, "rc": rc,
              "sequential": bool(session.get("sequential")), "capacity": capacity, "wall_s": (t1 - t0) / 1e9,
              "total_tokens": sum(len(v) for v in tokens.values()), "per_request": per, "failures": failures,
              "stderr_tail": stderr_path.read_text()[-1500:], "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    result["aggregate_tok_s"] = result["total_tokens"] / result["wall_s"]
    if len(cases) > 1 and not session.get("sequential") and all(r["first_s"] for r in per):
        first_last, last = max(r["first_s"] for r in per), max(r["last_s"] for r in per)
        after = sum(1 for r in per for t in times[r["request_id"]] if (t - t0) / 1e9 > first_last)
        result["tok_s_after_last_first_token"] = after / (last - first_last) if last > first_last else None
    if rc != 0:
        failures.append(f"sparkpipe_model_batch exit {rc}")
    with open(out / "results.jsonl", "a") as handle:
        handle.write(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_request", "stderr_tail")}))
    for r in per:
        print(json.dumps({k: v for k, v in r.items() if k != "token_ids"} | {"head": r["token_ids"][:8]}))
    if rc != 0 or any(r.get("check") == "MISMATCH" for r in per):
        return 1
    return 3 if failures else 0


def latest_records(results_path):
    latest = {}
    for line in Path(results_path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            latest[(record["tag"], record["session"])] = record
    return latest


def compare(results_path, reference_tag, candidate_tag, sessions):
    pairs = [tuple(s.split("=", 1)) if "=" in s else (s, s) for s in sessions]
    latest = {k: v for k, v in latest_records(results_path).items() if v.get("rc") == 0}
    failures = 0
    for session, reference_session in pairs:
        reference, candidate = latest.get((reference_tag, reference_session)), latest.get((candidate_tag, session))
        if reference is None or candidate is None:
            print(f"{session}: missing {reference_tag + ' ' + reference_session if reference is None else candidate_tag + ' ' + session}")
            failures += 1
            continue
        by_label = {r["label"]: r for r in reference["per_request"]}
        for right in candidate["per_request"]:
            left = by_label.get(right["label"])
            if left is None:
                print(f"{session} {right['label']}: missing in {reference_tag} {reference_session}")
                failures += 1
                continue
            a, b = left["token_ids"], right["token_ids"]
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
            print(f"{session} {left['label']}: {'EQUAL' if first is None else f'DIFFER at token {first}'} ({len(a)} vs {len(b)} tokens)")
            failures += first is not None
    return 0 if failures == 0 else 1


def mean_context(cases):
    return int(statistics.mean(c["prompt_len"] + max(1, c["tokens"]) / 2 for c in cases))


def decode_metrics(record, params):
    per = [c for c in record["per_request"] if c.get("inter_token_median_ms")]
    if not per:
        return []
    out = []
    if record["sequential"]:
        groups = {}
        for c in per:
            groups.setdefault(c.get("class") or f"o{c['budget']}", []).append(c)
        for name, cases in sorted(groups.items()):
            step = statistics.median(c["inter_token_median_ms"] for c in cases)
            text, ev = roofline.decode_line(1, step, mean_context(cases), p=params)
            out.append({"metric": f"B1 decode {name}", "rows": 1, "tok_s": 1e3 / step, "step_ms": step,
                        "cases_ms": [round(c["inter_token_median_ms"], 2) for c in cases], "roofline": text, "binding": ev["binding"],
                        "percent": ev["percent"], "ceiling_overlapped_tok_s": ev["ceiling_overlapped_tok_s"]})
        return out
    rows = len(record["per_request"])
    step = statistics.median(c["inter_token_median_ms"] for c in per)
    text, ev = roofline.decode_line(rows, step, mean_context(per), p=params)
    return [{"metric": f"B{rows} decode", "rows": rows, "tok_s": rows * 1e3 / step, "step_ms": step,
             "steady_tok_s_after_last_first_token": record.get("tok_s_after_last_first_token"),
             "wall_tok_s": record.get("aggregate_tok_s"), "roofline": text, "binding": ev["binding"], "percent": ev["percent"],
             "ceiling_overlapped_tok_s": ev["ceiling_overlapped_tok_s"]}]


def prefill_metrics(record, params):
    per = [c for c in record["per_request"] if c.get("ttft_s")]
    if not per:
        return []
    out = []
    if record["sequential"]:
        groups = {}
        for c in per:
            groups.setdefault(c["prompt_len"], []).append(c["ttft_s"])
        for n, values in sorted(groups.items()):
            ttft = statistics.median(values)
            text, ev = roofline.prefill_line(n, ttft * 1e3, p=params)
            out.append({"metric": f"TTFT {n} prompt tokens", "prompt_tokens": n, "ttft_s": ttft, "prefill_tok_s": n / ttft,
                        "roofline": text, "binding": ev["binding"], "percent": ev["percent"]})
        return out
    total = sum(c["prompt_len"] for c in per)
    span = max(c["first_s"] for c in per)
    text, ev = roofline.prefill_line(total, span * 1e3, p=params)
    return [{"metric": f"prefill under load: {len(per)} concurrent prompts, {total} tokens", "prompt_tokens": total,
             "ttft_s": span, "prefill_tok_s": total / span, "ttft_median_s": statistics.median(c["ttft_s"] for c in per),
             "roofline": text, "binding": ev["binding"], "percent": ev["percent"]}]


def gate_metrics(record):
    checks = [c.get("check") for c in record["per_request"] if c.get("check")]
    return {"exact": sum(1 for c in checks if c == "EXACT"), "mismatch": sum(1 for c in checks if c == "MISMATCH"),
            "invalid_short": sum(1 for c in checks if c == "INVALID-SHORT"), "cases": len(record["per_request"])}


def summary(results_path, out_dir, title, params=None):
    params = params or {}
    latest = latest_records(results_path)
    rows, skipped, failed, gates = [], [], [], {}
    for (tag, name), record in sorted(latest.items()):
        if record.get("skipped"):
            skipped.append(f"{tag} {name}: {record['skipped']}")
            continue
        if record["rc"] != 0:
            failed.append(f"{tag} {name}: exit {record['rc']} {'; '.join(record.get('failures', []))[:300]}")
        kind = record.get("kind", name)
        gates[f"{tag} {name}"] = gate_metrics(record)
        if kind in ("t1", "compsec17", "seq6"):
            continue
        metrics = prefill_metrics(record, params) if kind in ("ttft", "prefill") else decode_metrics(record, params)
        for m in metrics:
            rows.append({"tag": tag, "session": name, **m})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    data = {"title": title, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "metrics": rows,
            "gates": gates, "skipped": skipped, "failed": failed,
            "roofline_inputs": {**roofline.DEFAULTS, **params, "sources": roofline.SOURCES}}
    Path(out_dir, "summary.json").write_text(json.dumps(data, indent=1))
    lines = [f"# {title}", "", f"Generated {data['generated_utc']} from `{results_path}`.", "",
             "Speculative and non-speculative arms are separate tags; B1 is split per content class when the session labels one.", "",
             "| arm | session | metric | result | binding |", "|---|---|---|---|---|"]
    for r in rows:
        if "tok_s" in r:
            result = f"{r['tok_s']:.1f} tok/s ({r['step_ms']:.2f} ms/step)"
            if r.get("steady_tok_s_after_last_first_token"):
                result += f", steady {r['steady_tok_s_after_last_first_token']:.1f} tok/s"
        else:
            result = f"TTFT {r['ttft_s']:.2f} s, {r['prefill_tok_s']:.0f} prompt tok/s"
        lines.append(f"| {r['tag']} | {r['session']} | {r['metric']} | {result} | {r['binding']} |")
    lines += ["", "## Roofline lines", ""]
    lines += [f"- {r['tag']} {r['metric']}: {r['roofline']}" for r in rows]
    lines += ["", "## Gates", ""]
    lines += [f"- {k}: {v['exact']} exact, {v['mismatch']} mismatch, {v['invalid_short']} short of {v['cases']} cases" for k, v in gates.items()]
    if failed:
        lines += ["", "## Failed", ""] + [f"- {f}" for f in failed]
    if skipped:
        lines += ["", "## Skipped (arm capacity)", ""] + [f"- {s}" for s in skipped]
    lines += ["", "## Roofline inputs", ""]
    lines += [f"- {k}: {v}" for k, v in {**roofline.DEFAULTS, **params}.items()]
    lines += [f"- source {k}: {v}" for k, v in roofline.SOURCES.items()]
    Path(out_dir, "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if not failed else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: runner.py run SESSION OUT_DIR FW ROOT TAG | compare RESULTS REF_TAG CAND_TAG SESSIONS | summary RESULTS OUT_DIR TITLE")
    if sys.argv[1] == "run":
        sys.exit(run_session(*sys.argv[2:7]))
    if sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(",")))
    if sys.argv[1] == "summary":
        sys.exit(summary(sys.argv[2], sys.argv[3], sys.argv[4]))
    sys.exit("usage: runner.py run SESSION OUT_DIR FW ROOT TAG | compare RESULTS REF_TAG CAND_TAG SESSIONS | summary RESULTS OUT_DIR TITLE")
