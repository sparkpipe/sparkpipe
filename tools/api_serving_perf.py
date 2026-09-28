#!/usr/bin/env python3
import argparse
import http.client
import json
import statistics
import threading
import time
import urllib.parse

STORY = "Write a long, detailed story about a lighthouse keeper who finds a map in a bottle. Describe every day of the journey."
PARAGRAPH = ("The committee reviewed the quarterly logistics report, which covered warehouse throughput, "
             "late shipments, staffing changes, fuel costs and the new routing software. ")


def stream_completion(endpoint, body):
    url = urllib.parse.urlparse(endpoint)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=900)
    payload = dict(body)
    payload["stream"] = True
    started = time.monotonic()
    connection.request("POST", "/v1/completions", json.dumps(payload), {"Content-Type": "application/json"})
    response = connection.getresponse()
    token_times = []
    usage = None
    error = None
    finish = None
    buffer = b""
    while True:
        chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
        if not chunk:
            break
        buffer += chunk
        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            line = event.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            document = json.loads(data)
            if "error" in document:
                error = document["error"]
                continue
            now = time.monotonic()
            for _ in document.get("tokens", []):
                token_times.append(now - started)
            choice = document.get("choices", [{}])[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            if "usage" in document:
                usage = document["usage"]
    connection.close()
    if response.status != 200 and error is None:
        error = {"http_status": response.status}
    return {"started": started, "token_times": token_times, "usage": usage, "error": error, "finish_reason": finish}


def summarize(result):
    times = result["token_times"]
    record = {"error": result["error"], "finish_reason": result["finish_reason"], "usage": result["usage"],
              "completion_tokens": len(times)}
    if times:
        record["ttft_ms"] = round(times[0] * 1000.0, 1)
    if len(times) > 1:
        record["decode_tok_s"] = round((len(times) - 1) / (times[-1] - times[0]), 2)
        record["decode_ms_per_token"] = round((times[-1] - times[0]) * 1000.0 / (len(times) - 1), 2)
    return record


def run_b1(endpoint, max_tokens, repeats):
    records = []
    for _ in range(repeats):
        records.append(summarize(stream_completion(endpoint, {"prompt": STORY, "max_tokens": max_tokens, "temperature": 0})))
    rates = [r["decode_tok_s"] for r in records if "decode_tok_s" in r]
    return {"case": "b1_o%d" % max_tokens, "prompt": "story", "repeats": records,
            "median_decode_tok_s": statistics.median(rates) if rates else None}


def run_ttft(endpoint, prompt_words, repeats):
    text = ""
    while len(text.split()) < prompt_words:
        text += PARAGRAPH
    records = []
    for _ in range(repeats):
        records.append(summarize(stream_completion(endpoint, {"prompt": text, "max_tokens": 1, "temperature": 0})))
    ttfts = [r["ttft_ms"] for r in records if "ttft_ms" in r]
    prompt_tokens = records[0]["usage"]["prompt_tokens"] if records and records[0]["usage"] else None
    return {"case": "ttft", "prompt_tokens": prompt_tokens, "repeats": records,
            "median_ttft_ms": statistics.median(ttfts) if ttfts else None}


def run_streams(endpoint, streams, max_tokens):
    results = [None] * streams
    prompts = ["%s Chapter %d." % (STORY, index + 1) for index in range(streams)]

    def worker(index):
        results[index] = stream_completion(endpoint, {"prompt": prompts[index], "max_tokens": max_tokens, "temperature": 0})

    started = time.monotonic()
    threads = [threading.Thread(target=worker, args=(index,)) for index in range(streams)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - started
    total = sum(len(r["token_times"]) for r in results)
    first = [r["started"] + r["token_times"][0] for r in results if r["token_times"]]
    last = [r["started"] + r["token_times"][-1] for r in results if r["token_times"]]
    steady = (total - len(first)) / (max(last) - max(first)) if first and max(last) > max(first) else None
    return {"case": "streams%d_o%d" % (streams, max_tokens), "wall_s": round(wall, 3), "total_tokens": total,
            "aggregate_tok_s": round(total / wall, 2),
            "aggregate_after_last_first_token_tok_s": round(steady, 2) if steady else None,
            "errors": [r["error"] for r in results if r["error"]],
            "per_stream": [summarize(r) for r in results]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    with open(arguments.output, "a") as handle:
        for case in arguments.cases.split(","):
            if case.startswith("o"):
                record = run_b1(arguments.endpoint, int(case[1:]), arguments.repeats)
            elif case.startswith("ttft"):
                record = run_ttft(arguments.endpoint, int(case[4:]), arguments.repeats)
            elif case.startswith("streams"):
                count, tokens = case[7:].split("x")
                record = run_streams(arguments.endpoint, int(count), int(tokens))
            else:
                raise SystemExit("unknown case " + case)
            record["label"] = arguments.label
            record["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps({key: value for key, value in record.items() if key not in ("repeats", "per_stream")}), flush=True)


if __name__ == "__main__":
    main()
