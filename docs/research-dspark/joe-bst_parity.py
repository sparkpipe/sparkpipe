"""
bst_parity.py — like-for-like against TECHMD's published TP=4 decode benchmark.

★ WHY: our acceptance measures ~31% on four generic 256-token prompts; the published
figure is 79.6% on ONE specific 2048-token code-gen prompt. Prompt shape drives draft
acceptance more than any config knob, so those numbers were never comparable. This runs
THEIR prompt, THEIR protocol, and reports THEIR metric — the only way to tell whether we
are actually behind or just measuring different text.

★ SOURCE: techmd, TP=4 (NOT r0b0tlab — techmd uses r0b0tlab's PROMPT verbatim, but the run
  is his own TP=4 config). Same topology as ours, so this is a true apples-to-apples.
  His result: K=6, 104.17 tok/s MEAN (median 104.31, sd 3.51, min 98.84, max 110.23, n=10)
              5.777 tok/step, 55.46 ms/step, 79.6% acceptance over 3,513 steps
              per-position 96.2 / 90.0 / 83.5 / 76.3 / 69.5 / 62.3

PROTOCOL — his, matched exactly (bench-k-ladder.sh):
  prompt        : the BST prompt below, unchanged. DO NOT EDIT IT — changing the prompt
                  changes acceptance and voids comparability. That is the entire point.
  max_tokens    : 2048
  temperature   : 0
  concurrency   : 1
  warmup        : 2 FULL-LENGTH 2048-token generations (absorbs first-request JIT),
                  then 1+5 further DISCARDED repeats (settles FlashInfer autotune +
                  prefix cache). He measured warmup depth alone swinging results by
                  2.7 tok/s — more than most tuning changes. Do not shorten this.
  measured      : 10 repeats, report mean/median/sd/min/max
  metric        : delta generation_tokens_total / delta request_decode_time_seconds_sum
                  (server-side, decode-only, EXCLUDES prefill)

THEIR RESULT: 5.777 tok/step · 55.46 ms/step · 79.6% acceptance over 3,513 steps
              per-position 96.2 / 90.0 / 83.5 / 76.3 / 69.5 / 62.3   (k=6)

Aborts if foreign traffic lands in the window — these are global counters.
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = ("Write a complete, idiomatic Python implementation of a binary search tree with "
          "insert, delete, search, traversal, height, docstrings, and tests. Code only.")
MAX_TOKENS = 2048
WARMUP_FULL, WARMUP_DISCARD, REPEATS = 2, 6, 10
K = "vllm:spec_decode_num_"


def scrape(host):
    m = urllib.request.urlopen(f"http://{host}:8000/metrics", timeout=30).read().decode()
    tot, pos = {}, {}
    for line in m.split("\n"):
        if line.startswith("#") or not line.startswith("vllm:"):
            continue
        name = line.split("{", 1)[0]
        try: val = float(line.rsplit(" ", 1)[-1])
        except ValueError: continue
        if name == K + "accepted_tokens_per_pos_total":
            p = int(line.split('position="', 1)[1].split('"', 1)[0]); pos[p] = pos.get(p, 0.0) + val
        elif name in (K + "drafts_total", K + "draft_tokens_total", K + "accepted_tokens_total",
                      "vllm:generation_tokens_total", "vllm:request_decode_time_seconds_sum",
                      "vllm:request_success_total"):
            tot[name] = tot.get(name, 0.0) + val
    return tot, pos


def ask(host, model):
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS, "temperature": 0}
    r = urllib.request.Request(f"http://{host}:8000/v1/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Connection": "close"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


if __name__ == "__main__":
    # Endpoint comes from DSFV4_HOST so this file carries no site-specific address.
    # Set it once:  export DSFV4_HOST=<your-server-ip>
    #
    host = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DSFV4_HOST", "127.0.0.1")
    model = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DSFV4_MODEL", "deepseek-v4-flash-0731")
    # ★ CONCURRENCY MATTERS AND WAS THE BLIND SPOT. Decode batch = seqs x (k+1); a
    # cudagraph capture that is too small silently drops to EAGER above a certain
    # concurrency. At c=1 the batch is (k+1) tokens and always fits, so a capture
    # regression is INVISIBLE. Always measure at production concurrency too.
    CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    LABEL = sys.argv[4] if len(sys.argv) > 4 else ""
    print(f"\n  === BST PARITY vs techmd TP=4 — {host} ===")
    print(f"  {LABEL}")
    print(f"  his prompt + his protocol: max_tokens={MAX_TOKENS}, temp 0, concurrency {CONC},"
          f" {WARMUP_FULL} full warmups + {WARMUP_DISCARD} discarded + {REPEATS} measured\n")

    def burst(n):
        if n == 1: return [ask(host, model)]
        with ThreadPoolExecutor(max_workers=n) as ex:
            return list(ex.map(lambda _: ask(host, model), range(n)))
    for i in range(WARMUP_FULL):
        burst(CONC); print(f"  full-length warmup {i+1}/{WARMUP_FULL} (JIT)", flush=True)
    for i in range(WARMUP_DISCARD):
        burst(CONC); print(f"  discarded repeat {i+1}/{WARMUP_DISCARD}", flush=True)

    a, ap = scrape(host)
    t0 = time.perf_counter()
    finish, per_run = [], []
    for i in range(REPEATS):
        s0 = scrape(host)[0]
        ds = burst(CONC); d = ds[0]
        s1 = scrape(host)[0]
        dg = s1.get("vllm:generation_tokens_total",0)-s0.get("vllm:generation_tokens_total",0)
        dd = s1.get("vllm:request_decode_time_seconds_sum",0)-s0.get("vllm:request_decode_time_seconds_sum",0)
        rate = dg/dd if dd else 0
        per_run.append(rate)
        finish.append(d["choices"][0].get("finish_reason"))
        print(f"  rep {i+1:>2}/{REPEATS}  {d['usage']['completion_tokens']} tok"
              f"  {rate:6.2f} tok/s  finish={finish[-1]}", flush=True)
    wall = time.perf_counter() - t0
    b, bp = scrape(host)

    g = lambda k: b.get(k, 0.0) - a.get(k, 0.0)
    foreign = g("vllm:request_success_total") - REPEATS * CONC
    steps, dtok, atok = g(K + "drafts_total"), g(K + "draft_tokens_total"), g(K + "accepted_tokens_total")
    gen, dec = g("vllm:generation_tokens_total"), g("vllm:request_decode_time_seconds_sum")

    if foreign > 0:
        print(f"\n  ❌ INVALID — {foreign:.0f} foreign request(s) landed in the window."
              f"\n     These are GLOBAL counters. Re-run against an idle server.\n")
        sys.exit(4)
    if steps <= 0:
        print("\n  ❌ no decode steps recorded\n"); sys.exit(3)

    import statistics as st
    print(f"\n  per-run decode tok/s: mean {st.mean(per_run):.2f}  median {st.median(per_run):.2f}"
          f"  sd {st.pstdev(per_run):.2f}  min {min(per_run):.2f}  max {max(per_run):.2f}  n={len(per_run)}")
    print(f"  techmd published   : mean 104.17  median 104.31  sd 3.51  min 98.84  max 110.23  n=10")
    print(f"\n  {'':<22}{'OURS':>12}{'techmd TP=4':>14}")
    print(f"  {'-'*48}")
    print(f"  {'acceptance':<22}{atok/dtok*100:>11.1f}%{79.6:>13.1f}%")
    print(f"  {'tok/step':<22}{1+atok/steps:>12.3f}{5.777:>14.3f}")
    print(f"  {'ms/step':<22}{dec/steps*1000:>12.2f}{55.46:>14.2f}")
    print(f"  {'decode tok/s':<22}{gen/dec:>12.2f}{104.17:>14.2f}")
    print(f"  {'effective k':<22}{dtok/steps:>12.2f}{6:>14}")
    print(f"  {'steps measured':<22}{steps:>12,.0f}{3513:>14,}")

    print("\n  per-position acceptance          ours       techmd")
    theirs = [96.2, 90.0, 83.5, 76.3, 69.5, 62.3]
    prev = steps
    for p in sorted(set(ap) | set(bp)):
        got = bp.get(p, 0.0) - ap.get(p, 0.0)
        if prev <= 0: break
        t = f"{theirs[p]:>9.1f}%" if p < len(theirs) else f"{'—':>10}"
        print(f"    pos {p}                    {got/prev*100:>8.1f}%{t}")
        prev = got
    print(f"  MACHINE {LABEL}\tc={CONC}\tacc={atok/dtok*100:.1f}\ttokstep={1+atok/steps:.3f}"
          f"\tmsstep={dec/steps*1000:.2f}\ttoks={gen/dec:.2f}\tkeff={dtok/steps:.2f}")
    print(f"\n  wall {wall:.1f}s · finish_reasons {set(finish)}"
          f" · foreign requests {foreign:.0f}\n")
