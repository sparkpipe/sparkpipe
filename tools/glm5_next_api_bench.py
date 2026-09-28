#!/usr/bin/env python3
import argparse, json, os, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path.home()
LOG = HOME / "g53-api-channel" / "api.log"
TOK = HOME / "g53-api-channel" / "runtime" / "tokenizer" / "tokenizer.json"
FIX = HOME / "api-build-09fdad6-r2" / "qualification" / "ds4_eval" / "quality-fixtures-glm5.3-flash.json"
EP = "http://127.0.0.1:8433/v1/completions"
PRE = "[gMASK]<sop><|user|>\n"
POST_OFF = "<|assistant|>\n<think></think>\n"
POST_ON = "<|assistant|>\n<think>"
ESSAY = ("Write a long, detailed essay of at least 1500 words on the history of the Roman Empire, "
         "from the founding of the city through the fall of Constantinople. Use many paragraphs "
         "and cover politics, economy, military, culture and religion.")
FILLER = ("The river valley supported farming communities for thousands of years, and each season "
          "brought floods that renewed the soil. Traders moved grain, timber, salt and cloth between "
          "the towns along the banks, and the records they kept describe prices, weather and disputes. ")


def post(body, timeout=900):
    req = urllib.request.Request(EP, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        p = json.loads(r.read())
    return time.monotonic() - t0, p


def log_offset():
    return LOG.stat().st_size


def measurements_since(off):
    with open(LOG, "rb") as f:
        f.seek(off)
        data = f.read().decode("utf-8", "replace")
    out = []
    for line in data.splitlines():
        if line.startswith('{"event":"request_measurements"'):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    errs = [l for l in data.splitlines() if "ERRSITE" in l]
    return out, errs


def timing(m):
    toks = m.get("tokens") or []
    ts = [t[1] for t in toks]
    r = {"request_id": m["request_id"], "prompt_tokens": m["prompt_tokens"],
         "cached_prompt_tokens": m["cached_prompt_tokens"], "output_tokens": len(ts),
         "finish_reason": m.get("finish_reason"), "status": m.get("status")}
    if ts:
        r["ttft_ms"] = round((ts[0] - m["accepted_ns"]) / 1e6, 2)
    if len(ts) > 1:
        d = [(b - a) / 1e6 for a, b in zip(ts, ts[1:])]
        r["decode_ms_per_token_mean"] = round((ts[-1] - ts[0]) / 1e6 / (len(ts) - 1), 3)
        r["decode_ms_per_token_median"] = round(statistics.median(d), 3)
        r["decode_tok_s"] = round((len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9), 2)
        r["e2e_engine_ms"] = round((ts[-1] - m["accepted_ns"]) / 1e6, 2)
    return r


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK))


def exact_prompt(tk, n, nonce):
    head = tk.encode(PRE + f"Reference {nonce}. Summarize the following notes in two sentences.\n",
                     add_special_tokens=False).ids
    tail = tk.encode(POST_OFF, add_special_tokens=False).ids
    fill = tk.encode(FILLER * 40, add_special_tokens=False).ids
    need = n - len(head) - len(tail)
    assert 0 < need <= len(fill)
    return head + fill[:need] + tail


def b1(args):
    res = {}
    tk = tokenizer()
    for mt in (128, 512):
        runs = []
        for rep in range(3):
            ids = tk.encode(PRE + f"[{mt}-{rep}-{time.time_ns()}] " + ESSAY + POST_OFF,
                            add_special_tokens=False).ids
            off = log_offset()
            wall, p = post({"prompt_token_ids": ids, "max_tokens": mt, "temperature": 0})
            time.sleep(0.3)
            ms, errs = measurements_since(off)
            t = timing(ms[-1]) if ms else {}
            t["wall_s"] = round(wall, 3)
            t["completion_tokens"] = p["usage"]["completion_tokens"]
            t["wall_tok_s"] = round(p["usage"]["completion_tokens"] / wall, 2)
            t["errsite"] = errs
            runs.append(t)
            print(json.dumps({"b1": mt, "rep": rep, **t}), flush=True)
        res[str(mt)] = {"runs": runs,
                        "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in runs),
                        "median_ms_per_token": statistics.median(r["decode_ms_per_token_mean"] for r in runs),
                        "median_wall_tok_s": statistics.median(r["wall_tok_s"] for r in runs),
                        "median_wall_s": statistics.median(r["wall_s"] for r in runs)}
    return res


def ttft(args):
    tk = tokenizer()
    res = {}
    for n in (372, 742):
        runs = []
        for rep in range(3):
            ids = exact_prompt(tk, n, f"{n}-{rep}-{time.time_ns()}")
            off = log_offset()
            wall, p = post({"prompt_token_ids": ids, "max_tokens": 1, "temperature": 0})
            time.sleep(0.3)
            ms, errs = measurements_since(off)
            t = timing(ms[-1]) if ms else {}
            t["wall_s"] = round(wall, 3)
            t["errsite"] = errs
            runs.append(t)
            print(json.dumps({"ttft": n, "rep": rep, **t}), flush=True)
        res[str(n)] = {"runs": runs,
                       "median_ttft_ms": statistics.median(r["ttft_ms"] for r in runs),
                       "median_wall_s": statistics.median(r["wall_s"] for r in runs)}
    return res


def conc(args):
    tk = tokenizer()
    fx = json.loads(FIX.read_text())
    cases = sorted([c for c in fx["cases"] if c["id"] in [f"compsec-{i:03d}" for i in range(76, 93)]],
                   key=lambda c: c["id"])
    from tokenizers import Tokenizer
    qs = [tk.decode(c["prompt_token_ids"], skip_special_tokens=False) for c in cases]
    res = {}
    for n in args.streams:
        runs = []
        for rep in range(-1 if args.warm else 0, 3):
            tag = "" if args.warm else f"[{n}-{rep}-{time.time_ns()}]\n"
            bodies = [{"prompt": PRE + tag + qs[i % len(qs)] + POST_ON,
                       "max_tokens": args.conc_tokens, "temperature": 0} for i in range(n)]
            off = log_offset()
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=n) as pool:
                outs = list(pool.map(post, bodies))
            wall = time.monotonic() - t0
            time.sleep(0.5)
            ms, errs = measurements_since(off)
            toks = sum(p["usage"]["completion_tokens"] for _, p in outs)
            per = [timing(m) for m in ms]
            r = {"streams": n, "wall_s": round(wall, 3), "completion_tokens": toks,
                 "aggregate_tok_s": round(toks / wall, 2),
                 "finish_reasons": [p["choices"][0].get("finish_reason") for _, p in outs],
                 "statuses": [p.get("status") for _, p in outs],
                 "median_stream_decode_tok_s": statistics.median(x["decode_tok_s"] for x in per if "decode_tok_s" in x) if per else None,
                 "median_ttft_ms": statistics.median(x["ttft_ms"] for x in per if "ttft_ms" in x) if per else None,
                 "measurements": len(ms), "errsite": errs}
            r["cached_prompt_tokens"] = sum(m["cached_prompt_tokens"] for m in ms)
            r["prompt_tokens"] = sum(m["prompt_tokens"] for m in ms)
            r["warmup"] = rep < 0
            if rep >= 0:
                runs.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "finish_reasons"}), flush=True)
        res[str(n)] = {"runs": runs, "median_aggregate_tok_s": statistics.median(r["aggregate_tok_s"] for r in runs)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["b1", "ttft", "conc"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--conc-tokens", type=int, default=128)
    ap.add_argument("--warm", action="store_true")
    a = ap.parse_args()
    r = {"b1": b1, "ttft": ttft, "conc": conc}[a.mode](a)
    r["generated_at"] = time.strftime("%FT%TZ", time.gmtime())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
