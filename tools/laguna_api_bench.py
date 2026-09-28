#!/usr/bin/env python3
import argparse
import http.client
import importlib.util
import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("laguna_compsec17", ROOT / "tools/laguna_compsec17.py")
chat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chat)

REFERENCE = ROOT / "model-families/laguna/reference_tokens.json"
ESSAY = "Write a long, detailed essay (at least 1200 words) on the history of the printing press, from Gutenberg to the digital age. Use many paragraphs."
PASSAGE = ("The river town had grown slowly for three centuries, first as a ferry crossing, then as a market for grain and timber, and finally as a small industrial center with a textile mill, a brickworks and a railway depot. "
           "Each generation left a layer of buildings: stone warehouses near the water, brick terraces on the slope, and later a ring of timber houses built quickly for mill workers. "
           "When the mill closed, the population fell by a third within a decade, shops emptied, and the railway reduced service to two trains a day. "
           "A group of residents then proposed converting the mill into workshops and apartments, restoring the riverside path, and running a weekly market in the old depot yard. "
           "The council hesitated because the building needed a new roof, the foundations had shifted, and the flood risk had increased since the upstream dam was lowered. ") * 3


def accepts(result, generated):
    strict = result["strict_steps"]
    expected = result["generated_token_ids"]
    for step in range(min(strict, len(expected))):
        if step >= len(generated) or generated[step] != expected[step]:
            return False, step
    if strict < len(expected):
        if strict >= len(generated):
            return False, strict
        if generated[strict] not in result["steps"][strict]["fp32_top2_tokens"]:
            return False, strict
    return True, None


class Client:
    def __init__(self, host, port):
        self.host, self.port, self.nonce = host, port, 0
        self.lock = threading.Lock()

    def unique(self, question):
        with self.lock:
            self.nonce += 1
            return "[run %d.%d] %s" % (int(time.time()), self.nonce, question)

    def stream(self, body):
        body = dict(body, stream=True, temperature=0)
        connection = http.client.HTTPConnection(self.host, self.port, timeout=900)
        started = time.monotonic()
        connection.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("HTTP %d: %s" % (response.status, response.read()[:300]))
        times, tokens, usage = [], [], None
        while True:
            line = response.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(json.dumps(event["error"]))
            if event.get("tokens"):
                now = time.monotonic()
                tokens.extend(event["tokens"])
                times.extend([now] * len(event["tokens"]))
            if event.get("usage"):
                usage = event["usage"]
        connection.close()
        return {"started": started, "times": times, "tokens": tokens, "usage": usage}


def summary(run):
    times, count = run["times"], len(run["times"])
    return {"prompt_tokens": (run["usage"] or {}).get("prompt_tokens"), "tokens": count,
            "ttft_s": round(times[0] - run["started"], 4) if count else None,
            "decode_tok_s": round((count - 1) / (times[-1] - times[0]), 3) if count > 1 and times[-1] > times[0] else None,
            "e2e_s": round(times[-1] - run["started"], 3) if count else None}


def reference_mode(client):
    document = json.loads(REFERENCE.read_text())
    records = []
    for result in document["results"]:
        run = client.stream({"prompt_token_ids": result["prompt_token_ids"], "max_tokens": len(result["generated_token_ids"])})
        ok, step = accepts(result, run["tokens"])
        records.append({"name": result["name"], "strict_steps": result["strict_steps"], "accepted": ok,
                        "first_mismatch_step": step, "generated": run["tokens"],
                        "reference": result["generated_token_ids"]})
    return {"mode": "reference", "method": "no-spec greedy", "cases": records,
            "passed": sum(r["accepted"] for r in records), "total": len(records)}


def main():
    parser = argparse.ArgumentParser(description="Laguna S 2.1 API checks and no-spec benchmarks")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--modes", required=True, help="comma list of reference,o128,o512,ttft,streams8")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    client = Client(args.host, args.port)
    essay = lambda: chat.build_prompt(client.unique(ESSAY), "off")
    records = []
    failed = False

    def emit(record):
        record["label"] = args.label
        record["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(json.dumps(record), flush=True)
        records.append(record)

    client.stream({"prompt": chat.build_prompt("Say hi.", "off"), "max_tokens": 4})
    for mode in args.modes.split(","):
        if mode == "reference":
            record = reference_mode(client)
            failed = failed or record["passed"] != record["total"]
            emit(record)
        elif mode in ("o128", "o512"):
            runs = [summary(client.stream({"prompt": essay(), "max_tokens": int(mode[1:])})) for _ in range(args.reps)]
            emit({"mode": "B1 decode %s" % mode[1:], "method": "no-spec greedy", "runs": runs,
                  "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in runs)})
        elif mode == "ttft":
            prompt = lambda: chat.build_prompt(client.unique(PASSAGE + "Summarize the passage in one sentence."), "off")
            runs = [summary(client.stream({"prompt": prompt(), "max_tokens": 8})) for _ in range(args.reps)]
            emit({"mode": "TTFT", "method": "no-spec", "runs": runs,
                  "median_ttft_s": statistics.median(r["ttft_s"] for r in runs)})
        elif mode.startswith("streams"):
            count = int(mode[len("streams"):])
            results = [None] * count

            def worker(index):
                results[index] = client.stream({"prompt": essay(), "max_tokens": 256})

            threads = [threading.Thread(target=worker, args=(index,)) for index in range(count)]
            began = time.monotonic()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            ended = max(r["times"][-1] for r in results if r["times"])
            total = sum(len(r["tokens"]) for r in results)
            emit({"mode": "%d-stream aggregate" % count, "method": "no-spec greedy", "max_tokens": 256,
                  "tokens": total, "wall_s": round(ended - began, 3),
                  "aggregate_tok_s": round(total / (ended - began), 3),
                  "per_stream": [summary(r) for r in results]})
        else:
            parser.error("unknown mode %s" % mode)
    with open(args.out, "a") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
