import json, sys, time, statistics, http.client, threading
HOST, PORT = "127.0.0.1", int(sys.argv[1]); modes = sys.argv[2].split(","); reps = int(sys.argv[3]); out = sys.argv[4]; label = sys.argv[5] if len(sys.argv) > 5 else ""
def chat(q): return "<bos><|turn>user\n" + q + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
ESSAY = chat("Write a long, detailed essay (at least 1200 words) on the history of the printing press, from Gutenberg to the digital age. Use many paragraphs.")
PASSAGE = ("The river town had grown slowly for three centuries, first as a ferry crossing, then as a market for grain and timber, and finally as a small industrial center with a textile mill, a brickworks and a railway depot. "
  "Each generation left a layer of buildings: stone warehouses near the water, brick terraces on the slope, and later a ring of timber houses built quickly for mill workers. "
  "When the mill closed, the population fell by a third within a decade, shops emptied, and the railway reduced service to two trains a day. "
  "A group of residents then proposed converting the mill into workshops and apartments, restoring the riverside path, and running a weekly market in the old depot yard. "
  "The council hesitated because the building needed a new roof, the foundations had shifted, and the flood risk had increased since the upstream dam was lowered. ") * 3
TTFT_PROMPT = chat(PASSAGE + "Summarize the passage in one sentence.")
NONCE = [0]
def run(prompt, max_tokens):
    NONCE[0] += 1
    if prompt.startswith("<bos><|turn>user\n"):
        prompt = prompt.replace("<bos><|turn>user\n", "<bos><|turn>user\n[run %d.%d] " % (int(time.time()), NONCE[0]), 1)
    c = http.client.HTTPConnection(HOST, PORT, timeout=900)
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True})
    t0 = time.monotonic(); c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse(); times = []; usage = None; buf = b""
    while True:
        line = r.readline()
        if not line: break
        line = line.strip()
        if not line.startswith(b"data: "): continue
        payload = line[6:]
        if payload == b"[DONE]": break
        d = json.loads(payload)
        if d.get("tokens"): times.extend([time.monotonic()] * len(d["tokens"]))
        if d.get("usage"): usage = d["usage"]
    return {"t0": t0, "times": times, "usage": usage}
def summarize(r):
    t = r["times"]; n = len(t)
    return {"prompt_tokens": r["usage"]["prompt_tokens"] if r["usage"] else None, "tokens": n,
            "ttft_s": round(t[0] - r["t0"], 4) if n else None,
            "decode_tok_s": round((n - 1) / (t[-1] - t[0]), 3) if n > 1 else None,
            "e2e_s": round(t[-1] - r["t0"], 3) if n else None}
results = []
def emit(rec):
    rec["label"] = label; rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(rec), flush=True); results.append(rec)
run(chat("Say hi."), 4)
for m in modes:
    if m in ("o128", "o512"):
        n = int(m[1:]); rs = [summarize(run(ESSAY, n)) for _ in range(reps)]
        emit({"mode": f"B1 decode {n}", "method": "no-spec greedy", "content": "essay (natural language)", "runs": rs,
              "median_decode_tok_s": statistics.median(x["decode_tok_s"] for x in rs)})
    elif m == "ttft":
        rs = [summarize(run(TTFT_PROMPT, 8)) for _ in range(reps)]
        emit({"mode": "TTFT", "method": "no-spec", "content": "long passage + summary", "runs": rs,
              "median_ttft_s": statistics.median(x["ttft_s"] for x in rs)})
    elif m.startswith("streams"):
        k = int(m[7:]); n = 256; box = [None] * k
        def worker(i): box[i] = run(ESSAY, n)
        th = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
        start = time.monotonic(); [x.start() for x in th]; [x.join() for x in th]
        total = sum(len(b["times"]) for b in box); first = max(b["times"][0] for b in box); last = min(b["times"][-1] for b in box)
        steady = sum(sum(1 for t in b["times"] if first <= t <= last) for b in box) / (last - first) if last > first else None
        emit({"mode": f"{k}-stream aggregate", "method": "no-spec greedy", "content": "essay x%d" % k, "max_tokens": n,
              "total_tokens": total, "wall_s": round(max(b["times"][-1] for b in box) - start, 3),
              "aggregate_tok_s": round(total / (max(b["times"][-1] for b in box) - start), 3),
              "steady_all_active_tok_s": round(steady, 3) if steady else None,
              "per_stream": [summarize(b) for b in box]})
json.dump(results, open(out, "a")); open(out, "a").write("\n")
