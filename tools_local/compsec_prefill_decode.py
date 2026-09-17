#!/usr/bin/env python3
"""compsec_prefill_decode - prefill/decode wall-clock split over the COMPSEC
fixture set against the local serving endpoint (127.0.0.1:8433, fixed).

Per case: one max_tokens=1 request (prefill + one step) and one max_tokens=33
request; decode rate is (33-1)/(t33 - t1), prefill is t1. Run it on the node
the API listens on so the measurement never crosses another network.

usage: compsec_prefill_decode.py --fixture quality-fixtures.json \
    [--decode-tokens 32]
"""
import argparse
import http.client
import json
import statistics
import time

HOST = "127.0.0.1"
PORT = 8433


def call(ids, n):
    body = json.dumps({
        "prompt_token_ids": ids,
        "max_tokens": n,
        "temperature": 0,
    })
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    t0 = time.monotonic()
    conn.request("POST", "/v1/completions", body=body,
                 headers={"Content-Type": "application/json"})
    resp = json.loads(conn.getresponse().read())
    elapsed = time.monotonic() - t0
    conn.close()
    return elapsed, resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--decode-tokens", type=int, default=32)
    args = ap.parse_args()
    doc = json.load(open(args.fixture))
    cases = doc if isinstance(doc, list) else doc.get("cases", doc.get("fixtures", []))
    cases = [c for c in cases if str(c.get("id", "")).startswith("compsec")]
    rows = []
    for c in cases:
        ids = c["prompt_token_ids"]
        t1, p1 = call(ids, 1)
        tN, pN = call(ids, args.decode_tokens + 1)
        got = len(pN.get("tokens", []))
        decode_s = max(tN - t1, 1e-9)
        steps = max(got - 1, 1)
        rows.append((c["id"], len(ids), t1 * 1000, decode_s, got, steps / decode_s))
        print(f"{c['id']} prompt={len(ids):4d} prefill={t1*1000:7.0f}ms "
              f"decode={steps/decode_s:6.2f} tok/s ({got} tok in {decode_s:.2f}s)",
              flush=True)
    prefills = [r[2] for r in rows]
    rates = [r[5] for r in rows]
    print(f"\nmedian prefill {statistics.median(prefills):.0f} ms | "
          f"median decode {statistics.median(rates):.2f} tok/s | "
          f"min {min(rates):.2f} max {max(rates):.2f} (n={len(rows)})")


if __name__ == "__main__":
    main()
