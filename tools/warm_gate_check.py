#!/usr/bin/env python3
import argparse
import http.client
import importlib.util
import json
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path

COUNT_PROMPT = "Count upward from 1, one number per line, and do not stop."
CACHE_MODULE = Path(__file__).resolve().parent / "serving" / "chat_cache.py"


def seed_check(config, model, warm):
    spec = importlib.util.spec_from_file_location("sparkpipe_chat_cache", CACHE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = next((item for item in json.loads(config.read_text())["models"] if item["id"] == model), None)
    if entry is None or entry.get("cold_prefill") is None or not warm.get("prompt_tokens") or not warm.get("prefill_ms") or warm.get("cached_tokens"):
        return {"estimate_ms": None, "measured_ms": warm.get("prefill_ms"), "error": None, "within_10_percent": None}
    estimate = module.PrefillSeed(model, entry["cold_prefill"]).rate(warm["prompt_tokens"]) * warm["prompt_tokens"]
    error = estimate / warm["prefill_ms"] - 1
    return {"estimate_ms": round(estimate), "measured_ms": warm["prefill_ms"], "error": round(error, 4), "within_10_percent": abs(error) <= 0.1}


class Client:
    def __init__(self, base, key, headers):
        self.url = urllib.parse.urlsplit(base.rstrip("/"))
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = "Bearer " + key
        self.headers.update(headers)

    def connect(self):
        kind = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
        return kind(self.url.netloc, timeout=7200)

    def post(self, path, body):
        connection = self.connect()
        try:
            connection.request("POST", self.url.path + path, json.dumps(body), self.headers)
            response = connection.getresponse()
            text = response.read().decode()
            return response.status, json.loads(text) if text else {}
        finally:
            connection.close()

    def stream(self, path, body):
        connection = self.connect()
        started, arrivals, record = time.monotonic(), [], None
        try:
            connection.request("POST", self.url.path + path, json.dumps(dict(body, stream=True)), self.headers)
            response = connection.getresponse()
            if response.status != 200:
                return {"status": response.status, "error": response.read().decode()[:400]}
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                event = json.loads(line[5:])
                if any((choice.get("delta") or {}).get("content") or (choice.get("delta") or {}).get("reasoning_content") for choice in event.get("choices") or []):
                    arrivals.append(time.monotonic())
                record = event.get("sparkpipe") or record
        finally:
            connection.close()
        return {"status": 200, "ttft_s": arrivals[0] - started if arrivals else None, "arrivals": arrivals, "record": record}


def gaps(arrivals, until=None):
    return [later - earlier for earlier, later in zip(arrivals, arrivals[1:]) if until is None or later <= until]


def long_body(client, model, corpus, tokens):
    text = corpus.read_text()
    status, counted = client.post("/v1/tokenize", {"model": model, "text": text})
    if status != 200 or not counted.get("count"):
        raise SystemExit(f"FAIL /v1/tokenize answered {status}: {json.dumps(counted)[:300]}")
    repeats = max(1, -(-tokens // counted["count"]))
    return {"model": model, "messages": [{"role": "user", "content": (text + "\n") * repeats + "\nSummarize the text above in one line."}]}


def gate(client, model, corpus, tokens, decode_tokens, lead_seconds, config=None):
    body = long_body(client, model, corpus, tokens)
    counter = {"model": model, "messages": [{"role": "user", "content": COUNT_PROMPT}], "max_tokens": decode_tokens, "temperature": 0}
    idle = client.stream("/v1/chat/completions", counter)
    holder = {}
    warm_thread = threading.Thread(target=lambda: holder.update(warm=client.post("/v1/cache/warm", dict(body, format="openai_chat")), warm_done=time.monotonic()))
    stream_holder = {}
    stream_thread = threading.Thread(target=lambda: stream_holder.update(during=client.stream("/v1/chat/completions", counter)))
    warm_started = time.monotonic()
    warm_thread.start()
    time.sleep(lead_seconds)
    stream_thread.start()
    warm_thread.join()
    stream_thread.join()
    during = stream_holder["during"]
    warm_status, warm = holder["warm"]
    real_status, real = client.post("/v1/chat/completions", dict(body, max_tokens=1))
    record = real.get("sparkpipe") or {}
    idle_gaps = gaps(idle.get("arrivals") or [])
    step = statistics.median(idle_gaps) if idle_gaps else None
    arrivals = during.get("arrivals") or []
    window = gaps(arrivals, holder["warm_done"])
    worst = max(window) if window else None
    report = {"model": model, "target_tokens": tokens,
              "warm": {"status": warm_status, "prompt_tokens": warm.get("prompt_tokens"), "committed_tokens": warm.get("committed_tokens"),
                       "cached_tokens": warm.get("cached_tokens"), "prefill_ms": warm.get("prefill_ms"), "seconds": round(holder["warm_done"] - warm_started, 3)},
              "real": {"status": real_status, "prompt_tokens": record.get("prompt_tokens"), "cached_tokens": record.get("cached_tokens"),
                       "ttft_ms": record.get("ttft_ms"), "cold_prefill_ms_estimate": record.get("cold_prefill_ms_estimate")},
              "interactive": {"idle_step_s": step, "idle_ttft_s": idle.get("ttft_s"), "during_ttft_s": during.get("ttft_s"), "during_worst_gap_s": worst,
                              "during_gaps": len(window), "overlapped_warm": bool(arrivals and arrivals[0] < holder["warm_done"] < arrivals[-1]),
                              "delay_steps": round((worst - step) / step, 2) if worst is not None and step else None}}
    report["matched"] = (warm_status == 200 and real_status == 200 and isinstance(warm.get("committed_tokens"), int) and
                         isinstance(record.get("cached_tokens"), int) and record["cached_tokens"] >= warm["committed_tokens"])
    report["seed"] = seed_check(config, model, report["warm"]) if config is not None else None
    report["yielded"] = report["interactive"]["delay_steps"] is not None and report["interactive"]["delay_steps"] <= 1.0 and report["interactive"]["overlapped_warm"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gate B: warm a long prompt, check the next real request matches it, and measure what the warm-up "
                                     "costs an interactive stream that decodes through it")
    parser.add_argument("--base", required=True, help="the chat layer or gateway, for example http://127.0.0.1:8433")
    parser.add_argument("--key-file", help="a file holding the API key, for a base behind the public door")
    parser.add_argument("--header", action="append", default=[], help="an extra request header, Name: value (repeatable)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True, help="a text file repeated until the prompt reaches --tokens")
    parser.add_argument("--tokens", type=int, default=150000)
    parser.add_argument("--decode-tokens", type=int, default=8192, help="the interactive stream's length; it must outlast the warm-up")
    parser.add_argument("--lead-seconds", type=float, default=1.0, help="how long after the warm-up starts the interactive stream starts")
    parser.add_argument("--hub-config", type=Path, help="the hub's chat_frontend.json: also compare the model's cold_prefill seed with the warm-up's "
                        "measured cold prefill (C4)")
    parser.add_argument("--out", type=Path, help="write the report here as JSON")
    arguments = parser.parse_args(argv)
    headers = dict(item.split(":", 1) for item in arguments.header)
    client = Client(arguments.base, Path(arguments.key_file).read_text().strip() if arguments.key_file else None,
                    {name.strip(): value.strip() for name, value in headers.items()})
    report = gate(client, arguments.model, arguments.corpus, arguments.tokens, arguments.decode_tokens, arguments.lead_seconds, arguments.hub_config)
    text = json.dumps(report, indent=1)
    if arguments.out:
        arguments.out.write_text(text + "\n")
    print(text)
    ok = report["matched"] and report["yielded"]
    print(f"{'PASS' if ok else 'FAIL'} Gate B: warmed {report['warm']['committed_tokens']} tokens, the next request cached {report['real']['cached_tokens']}; "
          f"an interactive stream through the warm-up was delayed {report['interactive']['delay_steps']} decode steps at worst")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
