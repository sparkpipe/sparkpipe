"""
accept_live.py — read speculative-decode acceptance off the LIVE server under whatever
traffic is really hitting it. SENDS NO REQUESTS. Safe to run mid-production, any time.

★ WHY THIS EXISTS (2026-08-04)
  accept.py (v1) hardcoded `temperature: 0` in its own request body. Every acceptance
  number it ever produced was therefore measured at temp 0 — the condition under which
  acceptance looks BEST — while production served clients that send no temperature at
  all and get vLLM's default of 1.0. The instrument was structurally incapable of
  observing the bug it was supposed to catch, and a full day of tuning went past it.
  Measured 2026-08-04 under real Hermes traffic: 48.1% acceptance, 3.41 tok/step,
  against 79.6% / 5.78 reported by another operator at forced temp 0.

  So this file exists to measure PRODUCTION, not a fixture. It cannot lie about
  sampling parameters because it does not choose them — the real clients do.

  Acceptance is the right health number because it is a RATIO, not a rate: it stays
  comparable across different prompt lengths, concurrency levels and days, where a
  tok/s figure does not. Step time is fixed by memory bandwidth; tokens-per-step is
  the only lever on top of it.

USAGE
    export DSFV4_HOST=<your-server-ip>             # or pass it as argv[2]
    python3 accept_live.py [seconds] [host]        # default 60s
    python3 accept_live.py 300                     # longer window = tighter estimate

EXIT 0 always healthy-or-not; exit 3 if NO drafts occurred (no traffic to measure).
"""
import json, os, sys, time, urllib.request

def scrape(host):
    m = urllib.request.urlopen(f"http://{host}:8000/metrics", timeout=30).read().decode()
    tot, per_pos = {}, {}
    for line in m.split("\n"):
        if not line.startswith("vllm:spec_decode") or line.startswith("#"):
            continue
        name = line.split("{", 1)[0]
        try:
            val = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = line.split('position="', 1)[1].split('"', 1)[0]
            per_pos[int(pos)] = per_pos.get(int(pos), 0.0) + val
        elif name.endswith("_total"):
            tot[name] = tot.get(name, 0.0) + val
    return tot, per_pos


def running(host):
    m = urllib.request.urlopen(f"http://{host}:8000/metrics", timeout=30).read().decode()
    for line in m.split("\n"):
        if line.startswith("vllm:num_requests_running{"):
            return float(line.rsplit(" ", 1)[-1])
    return float("nan")


if __name__ == "__main__":
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    host = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DSFV4_HOST", "127.0.0.1")
    K = "vllm:spec_decode_num_"

    a, ap = scrape(host)
    r0 = running(host)
    print(f"\n  === LIVE ACCEPTANCE — {host}, {secs}s window, ZERO requests sent ===")
    print(f"  sampling... (concurrency at start: {r0:.0f})", flush=True)
    time.sleep(secs)
    b, bp = scrape(host)
    r1 = running(host)

    d = lambda k: b.get(K + k + "_total", 0.0) - a.get(K + k + "_total", 0.0)
    drafts, dtok, atok = d("drafts"), d("draft_tokens"), d("accepted_tokens")

    if drafts <= 0 or dtok <= 0:
        print("\n  ** NO DRAFTS IN WINDOW — the server was idle, or spec decode is off. **")
        print("     Nothing to measure. Re-run while traffic is flowing.\n")
        sys.exit(3)

    acc = atok / dtok * 100
    tps = 1 + atok / drafts                       # tokens emitted per decode step
    k_eff = dtok / drafts                         # effective draft length actually used
    print(f"\n  concurrency        {r0:.0f} -> {r1:.0f}")
    print(f"  drafts             {drafts:>12,.0f}")
    print(f"  draft tokens       {dtok:>12,.0f}   (effective k = {k_eff:.2f})")
    print(f"  accepted tokens    {atok:>12,.0f}")
    print(f"  ACCEPTANCE         {acc:>11.1f}%")
    print(f"  TOKENS / STEP      {tps:>12.2f}   <- this multiplies your output speed")

    # per-position ladder: the shape tells you whether a SHORTER k would lose anything
    pos = sorted(set(ap) | set(bp))
    if pos:
        print("\n  per-position acceptance (of drafts that reached that position):")
        prev = drafts
        for p in pos:
            got = bp.get(p, 0.0) - ap.get(p, 0.0)
            if prev <= 0:
                break
            print(f"    pos {p}   {got/prev*100:>5.1f}%   ({got:,.0f} of {prev:,.0f})")
            prev = got

    print("\n  reference points, same hardware + model + MTP:")
    print("    79.6% / 5.78 tok/step   another operator, forced temperature 0, k=6")
    print("    48.1% / 3.41 tok/step   THIS fleet, 2026-08-04, real Hermes traffic")
    if acc < 60:
        print("\n  ⚠ Below 60%. The known cause on this fleet is SAMPLING, not the model:")
        print("    the server runs `--generation-config vllm` with no")
        print("    `--override-generation-config`, so a client that omits `temperature`")
        print("    gets vLLM's default of 1.0, and MTP rejection-sampling craters.")
        print("    Confirm by comparing with: python3 accept.py --sweep 0,0.3,0.7,1.0")
    print()
