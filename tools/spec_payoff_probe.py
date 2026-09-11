"""Speculation payoff probe: measures tokens/s with and without speculation on a live fleet api.

Usage: python3 tools/spec_payoff_probe.py --api http://spark0:8433 --corpus corpus.json \
    --batch 8 --requests 24 --max-tokens 256 [--label mtp-on]

Corpus format: {"cases": [{"id": ..., "ids": [token ints]}]} (the ds4_eval fixture shape).
Each request: greedy completion of one corpus prompt, prompt truncated to --prompt-tokens.
Concurrency is the batch width: --batch lanes in flight, --requests total. Reports per-leg
wall time, aggregate tokens/s, per-lane tokens/s, and completion-token histogram data.
Stdlib only.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request


def post_completion(api, ids, max_tokens):
    body = json.dumps({
        "model": "default",
        "prompt_token_ids": ids,
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        api + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as resp:
        payload = json.loads(resp.read())
    wall = time.monotonic() - t0
    if "choices" in payload:
        usage = payload.get("usage", {})
        return wall, int(usage.get("completion_tokens", 0)), \
            payload["choices"][0].get("finish_reason", "?")
    return wall, len(payload.get("tokens", [])), \
        ("stop" if payload.get("status") == 0 else f"status={payload.get('status')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--label", default="leg")
    args = ap.parse_args()

    cases = json.load(open(args.corpus))["cases"]
    prompts = [c["ids"][:args.prompt_tokens] for c in cases]
    prompts = [p for p in prompts if len(p) >= 32]
    if not prompts:
        raise SystemExit("corpus has no usable prompts")

    results = []
    lock = threading.Lock()
    next_index = [0]

    def worker():
        while True:
            with lock:
                i = next_index[0]
                next_index[0] += 1
            if i >= args.requests:
                return
            ids = prompts[i % len(prompts)]
            try:
                wall, completion_tokens, finish = post_completion(
                    args.api, ids, args.max_tokens)
                with lock:
                    results.append((wall, completion_tokens, finish))
            except Exception as exc:
                with lock:
                    results.append((-1.0, 0, f"error: {exc}"))

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(args.batch)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_total = time.monotonic() - t0

    errors = [r for r in results if r[0] < 0]
    ok = [r for r in results if r[0] >= 0]
    total_completion = sum(r[1] for r in ok)
    latencies = [r[0] for r in ok]
    out = {
        "label": args.label,
        "batch": args.batch,
        "requests": len(results),
        "errors": len(errors),
        "wall_s": round(wall_total, 3),
        "completion_tokens": total_completion,
        "aggregate_toks": round(total_completion / wall_total, 2) if wall_total else 0,
        "lane_toks_p50": round(statistics.median(
            r[1] / r[0] for r in ok if r[0] > 0), 2) if ok else 0,
        "latency_p50_s": round(statistics.median(latencies), 3) if ok else 0,
        "latency_max_s": round(max(latencies), 3) if ok else 0,
        "finish_reasons": {f: sum(1 for r in ok if r[2] == f) for f in {r[2] for r in ok}},
    }
    print(json.dumps(out))
    for r in errors[:3]:
        print("ERROR", r[2])


if __name__ == "__main__":
    main()
