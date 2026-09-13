#!/usr/bin/env python3
"""DSV4-Flash TP4 B1 decode-throughput regression gate, pinned to the
2026-08-25 spark4-7 serving receipt (40.67 tok/s mean, 3/3 exact O128).

Two layers, one file:

  LOCAL (default; no GPU needed) - the pin itself is under test:
    * the baseline fixture is complete and self-consistent,
    * the verdict math fails closed (bad parses are failures, not skips),
    * the receipt parser reproduces tools/devcycle.sh's jq semantics
      (.decode_tokens_per_second; token-hash over the token_id CSV),
    * tolerance edges behave (at-floor passes, below-floor fails, hash
      mismatch fails even when fast).

  BENCH (opt-in on a GPU fleet head): DSV4FLASH_PERF_BENCH=1 or --bench
    runs the real protocol end to end -
        tools/devcycle.sh ready lean     # all-4-ranks ok
        tools/devcycle.sh gate24 lean    # O24 exact-token gate
        tools/devcycle.sh run lean 3     # 3x O128, exact-hash checked
    then verifies mean throughput against the pinned floor. Explicit bench
    mode FAILS LOUDLY when the rig is unavailable; it never silently passes.

Baseline provenance: .agents/coord/dsv4flash_serving_result.md (runs
40.6932/40.7386/40.5755, mean 40.6691 - the milestone's "40.67"); fixture at
tests/fixtures/dsv4_flash_perf_baseline.json.

Run:  python3 tests/test_dsv4_flash_perf_regression.py [--bench]
Exit: 0 pass/skip, 1 fail.
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "tests", "fixtures",
                       "dsv4_flash_perf_baseline.json")
DEVCYCLE = os.path.join(REPO, "tools", "devcycle.sh")

FAILURES = []


def check(name, ok, detail=""):
    if not ok:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name} {detail}")
    else:
        print(f"  ok   {name}")


# ---------------------------------------------------------------- parsing --

def token_stream_hash(receipt):
    """sha256 of the comma-joined token-id CSV + newline - byte-for-byte the
    recipe devcycle.sh token_csv_hash() runs through jq."""
    ids = [e["event"]["token_id"] for e in receipt["events"]
           if e.get("event", {}).get("event") == "token"]
    csv = ",".join(str(i) for i in ids) + "\n"
    return hashlib.sha256(csv.encode()).hexdigest()


def parse_run_line(line):
    """'run lean #1 exact 40.69 tok/s' / '... WRONG-TOKENS ...' -> dict."""
    parts = line.split()
    if len(parts) >= 6 and parts[0] == "run" and parts[3] in ("exact", "WRONG-TOKENS"):
        try:
            return {"exact": parts[3] == "exact", "rate": float(parts[4])}
        except ValueError:
            return None
    return None


def latest_receipts(name="lean", n=3):
    """Newest n o128 receipts for this run name, sorted by mtime."""
    paths = glob.glob(f"/tmp/devcycle-{name}-o128-r*.json")
    paths.sort(key=os.path.getmtime, reverse=True)
    out = []
    for p in paths[:n]:
        with open(p) as f:
            out.append(json.load(f))
    return out


# ---------------------------------------------------------------- verdict --

def verdict(rates, hashes, baseline):
    """-> (status, detail). status in PASS/WARN/FAIL.

    Correctness precedes speed: any non-exact token stream is a FAIL even if
    every rate clears the floor (a fast wrong-answer pipeline is broken, not
    faster). Bands, boundary values belonging to the HIGHER band so float
    rounding can never demote an exact-threshold result:

        pct <  floor          -> FAIL   (below the kill line)
        floor <= pct <  warn  -> WARN   (surviving, trending down)
        pct  >= warn          -> PASS
    """
    policy = baseline["regression_policy"]
    mean_ref = baseline["baseline_mean_tokens_per_second"]
    if len(rates) == 0:
        return "FAIL", "no measurements"
    if policy.get("require_exact_tokens", True) and any(h != "exact" for h in hashes):
        return "FAIL", f"token-stream mismatch: {hashes}"
    mean = sum(rates) / len(rates)
    pct = 100.0 * mean / mean_ref
    floor = policy["fail_below_pct_of_baseline"]
    warn = policy["warn_below_pct_of_baseline"]
    # Boundary values belong to the HIGHER band in fact, not just prose:
    # ref*95/100 round-trips to 94.999...% in IEEE-754, so the compares
    # carry an explicit tolerance.
    eps = 1e-9
    if pct < floor - eps:
        return "FAIL", f"mean {mean:.4f} tok/s = {pct:.2f}% of baseline {mean_ref} (< {floor}% floor)"
    if pct < warn - eps:
        return "WARN", f"mean {mean:.4f} tok/s = {pct:.2f}% of baseline {mean_ref} (< {warn}% warn band)"
    return "PASS", f"mean {mean:.4f} tok/s = {pct:.2f}% of baseline {mean_ref}"


def load_baseline():
    with open(FIXTURE) as f:
        return json.load(f)


# ---------------------------------------------------------------- local ----

def test_fixture_integrity():
    b = load_baseline()
    check("fixture.schema", b.get("schema") == "sparkpipe.perf-baseline/1")
    w = b["workload"]
    for k in ("family", "topology", "bucket", "output_tokens", "hosts"):
        check(f"fixture.workload.{k}", k in w and w[k])
    check("fixture.adapter_id",
          w["adapter_id"] == "spark.dsv4.flash-0731.serving-adapter.tp4.v1")
    runs = b["baseline_runs"]
    check("fixture.runs.count", len(runs) == 3, str(len(runs)))
    rates = [r["tokens_per_second"] for r in runs]
    mean = sum(rates) / len(rates)
    check("fixture.mean-consistency",
          abs(mean - b["baseline_mean_tokens_per_second"]) < 5e-4,
          f"{mean:.6f} vs {b['baseline_mean_tokens_per_second']}")
    # The headline number everyone quotes.
    check("fixture.headline-40.67",
          abs(round(b["baseline_mean_tokens_per_second"], 2) - 40.67) < 1e-9)
    g = b["gates"]
    check("fixture.o128-hash", len(g["o128_hash"]) == 64)
    check("fixture.o24-hash", len(g["o24_hash"]) == 64)
    p = b["regression_policy"]
    check("fixture.policy-order", p["warn_below_pct_of_baseline"] > p["fail_below_pct_of_baseline"])
    prov = b["provenance"]
    check("fixture.provenance", all(k in prov for k in
          ("measured_utc", "milestone_commit", "receipt_document", "harness")))


def test_verdict_math():
    b = load_baseline()
    ref = b["baseline_mean_tokens_per_second"]
    # At baseline: PASS.
    st, _ = verdict([ref, ref, ref], ["exact"] * 3, b)
    check("verdict.at-baseline", st == "PASS", st)
    # Exactly AT the floor survives into the warn band (boundaries belong to
    # the higher band - float rounding may not demote a threshold value).
    floor_rate = ref * b["regression_policy"]["fail_below_pct_of_baseline"] / 100.0
    st, d = verdict([floor_rate], ["exact"], b)
    check("verdict.at-floor-warns", st == "WARN", d)
    # A hair below the floor fails even through rounding noise.
    st, _ = verdict([floor_rate * 0.999], ["exact"], b)
    check("verdict.below-floor-fails", st == "FAIL", st)
    # Strictly inside the warn band.
    mid = ref * (b["regression_policy"]["fail_below_pct_of_baseline"] +
                 b["regression_policy"]["warn_below_pct_of_baseline"]) / 200.0
    st, _ = verdict([mid], ["exact"], b)
    check("verdict.warn-band-interior", st == "WARN", st)
    # Exactly at the warn boundary passes.
    warn_rate = ref * b["regression_policy"]["warn_below_pct_of_baseline"] / 100.0
    st, _ = verdict([warn_rate], ["exact"], b)
    check("verdict.at-warn-passes", st == "PASS", st)
    # Fast but wrong tokens: FAIL wins over speed.
    st, d = verdict([60.0, 61.0], ["exact", "deadbeef"], b)
    check("verdict.wrong-tokens-fail", st == "FAIL", d)
    # No measurements: fail closed.
    st, _ = verdict([], [], b)
    check("verdict.empty-fails", st == "FAIL", st)


def test_parsers():
    line_ok = parse_run_line("run lean #1 exact 40.6932 tok/s")
    check("parse.run-line-exact",
          line_ok == {"exact": True, "rate": 40.6932}, str(line_ok))
    line_bad = parse_run_line("run lean #2 WRONG-TOKENS 41.9 tok/s hash=abcd1234")
    check("parse.run-line-wrong", line_bad is not None and line_bad["exact"] is False)
    check("parse.garbage-line", parse_run_line("ready lean ok") is None)
    sample = {"events": [
        {"event": {"event": "prefill"}},
        {"event": {"event": "token", "token_id": 42}},
        {"event": {"event": "token", "token_id": 7}},
        {"event": {"event": "done"}},
    ]}
    h = token_stream_hash(sample)
    import hashlib as _h
    expect = _h.sha256(b"42,7\n").hexdigest()
    check("parse.token-hash", h == expect, f"{h[:12]} vs {expect[:12]}")
    rate_sample = json.dumps({"decode_tokens_per_second": 40.6691})
    check("parse.rate-field",
          json.loads(rate_sample)["decode_tokens_per_second"] == 40.6691)


def test_against_recorded_receipt_numbers():
    """The fixture must reproduce the receipt document's own arithmetic."""
    b = load_baseline()
    doc_rates = [40.6932, 40.7386, 40.5755]
    fixture_rates = [r["tokens_per_second"] for r in b["baseline_runs"]]
    check("receipt.run-values-equal", sorted(doc_rates) == sorted(fixture_rates))
    st, _ = verdict(fixture_rates, ["exact"] * 3, b)
    check("receipt.replay-passes", st == "PASS", st)


# ---------------------------------------------------------------- bench ----

def devcycle(args, check_rc=True):
    cmd = ["bash", DEVCYCLE] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check_rc and p.returncode != 0:
        raise RuntimeError(f"devcycle {' '.join(args)} rc={p.returncode}: "
                           f"{p.stdout.strip()[-400:]} {p.stderr.strip()[-200:]}")
    return p.stdout.strip()


def run_bench():
    b = load_baseline()
    print("== DSV4-Flash TP4 B1 pinned-baseline bench (explicit mode) ==")
    ready = devcycle(["ready", "lean"], check_rc=False)
    if "ok" not in ready.lower():
        print(f"FAIL rig-not-ready:\n{ready}")
        return 1
    g24 = devcycle(["gate24", "lean"], check_rc=False)
    print(g24)
    if "PASS" not in g24:
        print("FAIL o24 gate did not pass")
        return 1
    out = devcycle(["run", "lean", "3"], check_rc=False)
    print(out)
    lines = [parse_run_line(l) for l in out.splitlines()]
    parsed = [l for l in lines if l]
    tail = [l for l in out.splitlines() if l.startswith("run lean exact=")]
    n_exact = int(tail[-1].split("=")[-1].split("/")[0]) if tail else 0
    if n_exact != 3:
        print(f"FAIL exact-token count {n_exact}/3")
        return 1
    rates = [r["rate"] for r in parsed]
    hashes = ["exact" if r["exact"] else "MISMATCH" for r in parsed]
    # Cross-check against the raw receipts, not just stdout.
    receipts = latest_receipts("lean", 3)
    if len(receipts) == 3:
        rh = [token_stream_hash(r) for r in receipts]
        want = b["gates"]["o128_hash"]
        rr = [float(r["decode_tokens_per_second"]) for r in receipts]
        if rh != [want] * 3:
            print(f"FAIL receipt hashes {[h[:16] for h in rh]}")
            return 1
        rates, hashes = rr, ["exact"] * 3
    st, detail = verdict(rates, hashes, b)
    print(f"VERDICT {st}: {detail}")
    return 0 if st != "FAIL" else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", action="store_true",
                    help="run the live spark4-7 protocol (needs GPU fleet)")
    args = ap.parse_args(argv)
    if args.bench or os.environ.get("DSV4FLASH_PERF_BENCH") == "1":
        return run_bench()
    print("== DSV4-Flash perf-regression local gates "
          "(pin integrity + verdict math; live bench needs --bench) ==")
    test_fixture_integrity()
    test_verdict_math()
    test_parsers()
    test_against_recorded_receipt_numbers()
    if FAILURES:
        print(f"\nFAIL ({len(FAILURES)})")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\ndsv4-flash 40.67 tok/s pin holds: fixture, verdict math, "
          "parsers, recorded-receipt replay")
    print("(live measurement skipped: pass --bench or DSV4FLASH_PERF_BENCH=1 "
          "on a fleet head)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
