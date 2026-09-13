"""
accept.py v2 — reproducible DSpark acceptance measurement on a CONTROLLED fixture.

★★ WHAT WAS WRONG WITH v1, AND WHY IT MATTERED (fixed 2026-08-04)
   v1 line 21 was:   post({... "temperature":0})
   Temperature was HARDCODED to 0 — the single condition under which MTP acceptance
   looks best. Production meanwhile served clients that send NO temperature and so get
   vLLM's default of 1.0. The instrument could therefore only ever confirm the healthy
   case; it was structurally incapable of observing the defect it existed to catch.
   A full day of DeepSeek tuning ran past a live acceptance of 48.1% because every
   number came from this file at temp 0. Another operator found it with a wire capture.

   Two changes follow from that:
     1. THE DEFAULT NO LONGER SENDS A TEMPERATURE AT ALL. Omitting it is what real
        clients do, so the default now reproduces production instead of an assumption.
        Pass --temperature X only when you deliberately want to pin it.
     2. --sweep exists. One command now quantifies the temperature->acceptance curve,
        which is the measurement that would have caught this on day one.

   General rule this is an instance of: a benchmark that hardcodes a parameter can
   never discover that the parameter is wrong. Measure what production does, or state
   loudly that you are not.

USAGE
    python3 accept.py                          # DEFAULT: send no temperature (= prod)
    python3 accept.py --temperature 0          # pin it (the old v1 behaviour)
    python3 accept.py --sweep 0,0.3,0.7,1.0    # the curve. run this one.
    export DSFV4_HOST=<your-server-ip>   # or --host <ip>
    python3 accept.py --reps 3

NOTE: this sends its own requests, so it adds load and its counter deltas include
      only its own traffic ONLY IF the server is otherwise idle. To measure real
      production traffic with zero added load, use accept_live.py instead.
"""
import argparse, json, os, time, urllib.request

K = "vllm:spec_decode_num_"

PROMPTS = [("Write a Python function that merges two sorted lists. Explain each step.", 256),
           ("Summarize the tradeoffs between tensor parallelism and pipeline parallelism.", 256),
           ("List 10 shell commands for inspecting a Linux system and what each shows.", 256),
           ("Explain how prefix caching works in an LLM inference server.", 256)]


def requests_done(host):
    """Global completed-request counter — used to detect FOREIGN traffic in our window."""
    m = urllib.request.urlopen(f"http://{host}:8000/metrics", timeout=30).read().decode()
    n = 0.0
    for line in m.split("\n"):
        if line.startswith("vllm:request_success_total{"):
            n += float(line.rsplit(" ", 1)[-1])
    return n


def scrape(host):
    m = urllib.request.urlopen(f"http://{host}:8000/metrics", timeout=30).read().decode()
    tot, per_pos = {}, {}
    for line in m.split("\n"):
        if not line.startswith("vllm:spec_decode") or line.startswith("#"):
            continue
        name = line.split("{", 1)[0]
        try: val = float(line.rsplit(" ", 1)[-1])
        except ValueError: continue
        if name == K + "accepted_tokens_per_pos_total":
            p = int(line.split('position="', 1)[1].split('"', 1)[0])
            per_pos[p] = per_pos.get(p, 0.0) + val
        elif name.endswith("_total"):
            tot[name] = tot.get(name, 0.0) + val
    return tot, per_pos


def post(host, model, body):
    r = urllib.request.Request(f"http://{host}:8000/v1/completions",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Connection": "close"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


def measure(host, model, temperature, reps):
    """temperature=None means OMIT the field entirely — what a real client does."""
    a, ap = scrape(host)
    r_before = requests_done(host)
    t0 = time.perf_counter(); toks = 0
    for _ in range(reps):
        for p, mx in PROMPTS:
            body = {"model": model, "prompt": p, "max_tokens": mx}
            if temperature is not None:
                body["temperature"] = temperature
            toks += post(host, model, body)["usage"]["completion_tokens"]
    wall = time.perf_counter() - t0
    b, bp = scrape(host)
    mine = reps * len(PROMPTS)
    observed = requests_done(host) - r_before
    foreign = max(0.0, observed - mine)
    d = lambda k: b.get(K + k + "_total", 0.0) - a.get(K + k + "_total", 0.0)
    drafts, dtok, atok = d("drafts"), d("draft_tokens"), d("accepted_tokens")
    return {"contaminated": foreign > 0.05 * mine, "foreign": foreign, "mine": mine,
            "temperature": "omitted" if temperature is None else temperature,
            "wall": wall, "gen": toks, "tok_s": toks / wall if wall else 0,
            "drafts": drafts, "draft_tokens": dtok, "accepted": atok,
            "acceptance": (atok / dtok * 100) if dtok else float("nan"),
            "tok_per_step": (1 + atok / drafts) if drafts else float("nan"),
            "k_eff": (dtok / drafts) if drafts else float("nan"),
            "per_pos": {p: bp.get(p, 0.0) - ap.get(p, 0.0) for p in set(ap) | set(bp)}}


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--host", default=os.environ.get("DSFV4_HOST", "127.0.0.1"))
    ap_.add_argument("--model", default=os.environ.get("DSFV4_MODEL", "deepseek-v4-flash-0731"))
    ap_.add_argument("--reps", type=int, default=3)
    ap_.add_argument("--temperature", default=None,
                     help="pin a temperature. DEFAULT is to omit the field, as real clients do.")
    ap_.add_argument("--sweep", default=None,
                     help="comma list, e.g. 0,0.3,0.7,1.0 — quantifies temperature -> acceptance")
    a = ap_.parse_args()

    if a.sweep:
        temps = [None if t.strip().lower() in ("none", "omit") else float(t)
                 for t in a.sweep.split(",")] + [None]
        print(f"\n  === ACCEPTANCE vs TEMPERATURE — {a.host} ===")
        print(f"  {len(PROMPTS)} prompts x {a.reps} reps per point. 'omitted' = what real clients send.\n")
        print(f"  {'temperature':<14}{'accept':>9}{'tok/step':>11}{'k_eff':>8}{'tok/s':>9}{'foreign':>9}")
        rows = []
        for t in temps:
            r = measure(a.host, a.model, t, a.reps); rows.append(r)
            flag = f"{r['foreign']:>8.0f}!" if r["contaminated"] else f"{r['foreign']:>9.0f}"
            print(f"  {str(r['temperature']):<14}{r['acceptance']:>8.1f}%{r['tok_per_step']:>11.2f}"
                  f"{r['k_eff']:>8.2f}{r['tok_s']:>9.1f}{flag}", flush=True)
        bad = [r for r in rows if r["contaminated"]]
        if bad:
            tot = sum(r["foreign"] for r in rows); allreq = tot + sum(r["mine"] for r in rows)
            print(f"\n  ❌ SWEEP INVALID — {len(bad)}/{len(rows)} points contaminated by FOREIGN traffic.")
            print(f"     {tot:,.0f} of {allreq:,.0f} requests in these windows ({tot/allreq*100:.0f}%) were not mine.")
            print("     These counters are GLOBAL: another client's requests land in every delta.")
            print("     No conclusion about temperature can be drawn. Re-run against an IDLE server,")
            print("     or use accept_live.py, which measures production ON PURPOSE and adds no load.")
            print("     (2026-08-04: a sweep run against live Hermes traffic came back NON-MONOTONIC —")
            print("      temp 0.7 scored above temp 0.0 — which is what contamination looks like.)")
            raise SystemExit(4)
        pinned0 = next((r for r in rows if r["temperature"] == 0.0), None)
        omitted = next((r for r in rows if r["temperature"] == "omitted"), None)
        if pinned0 and omitted and pinned0["acceptance"] == pinned0["acceptance"]:
            gap = pinned0["acceptance"] - omitted["acceptance"]
            print(f"\n  temp-0 minus omitted: {gap:+.1f} pts acceptance, "
                  f"{pinned0['tok_per_step'] - omitted['tok_per_step']:+.2f} tok/step")
            if gap > 5:
                print("  ⚠ Omitting temperature costs real throughput on this server.")
                print("    Fix server-side: --override-generation-config '{\"temperature\":0.0}'")
                print("    AND client-side: extra_body {\"temperature\":0} in each provider block.")
                print("    Server-side alone does NOTHING if the client sends temperature explicitly.")
            else:
                print("  No meaningful gap — the sampling default is not costing you here.")
    else:
        t = None if a.temperature is None else float(a.temperature)
        r = measure(a.host, a.model, t, a.reps)
        print(f"\n  === ACCEPTANCE — {a.host}, temperature={r['temperature']} ===")
        if t is None:
            print("  (field omitted, reproducing a real client. --temperature X to pin.)")
        print(f"  requests        : {a.reps * len(PROMPTS)}   generated {r['gen']} tok "
              f"in {r['wall']:.1f}s ({r['tok_s']:.1f} tok/s)")
        print(f"  drafts          : {r['drafts']:,.0f}")
        print(f"  draft tokens    : {r['draft_tokens']:,.0f}   (effective k = {r['k_eff']:.2f})")
        print(f"  accepted tokens : {r['accepted']:,.0f}")
        print(f"  ACCEPTANCE RATE : {r['acceptance']:.2f}%")
        print(f"  TOKENS / STEP   : {r['tok_per_step']:.3f}")
        if r["per_pos"]:
            prev = r["drafts"]
            print("  per-position    :")
            for p in sorted(r["per_pos"]):
                got = r["per_pos"][p]
                if prev <= 0: break
                print(f"      pos {p}  {got/prev*100:>5.1f}%")
                prev = got
    print()
