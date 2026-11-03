#!/usr/bin/env python3
import argparse
import http.client
import json
import statistics
import threading
import time
import urllib.parse

HEADER = "Reference document for the long-context batch sweep.\n\n"


def post(endpoint, body, timeout):
    url = urllib.parse.urlparse(endpoint)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    payload = dict(body, stream=True)
    started = time.monotonic()
    connection.request("POST", "/v1/completions", json.dumps(payload), {"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        detail = response.read().decode(errors="replace")
        connection.close()
        return {"started": started, "token_times": [], "usage": None, "finish_reason": None,
                "error": {"http_status": response.status, "body": detail[:600]}}
    token_times, usage, error, finish, buffer = [], None, None, None, b""
    while True:
        chunk = response.read1(65536)
        if not chunk:
            break
        buffer += chunk
        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            line = event.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            document = json.loads(line[5:].strip())
            if "error" in document:
                error = document["error"]
                continue
            now = time.monotonic() - started
            token_times.extend(now for _ in document.get("tokens", []))
            choice = (document.get("choices") or [{}])[0]
            finish = choice.get("finish_reason") or finish
            usage = document.get("usage") or usage
    connection.close()
    return {"started": started, "token_times": token_times, "usage": usage, "finish_reason": finish, "error": error}


def document_text(path, characters):
    source = open(path, encoding="utf-8", errors="replace").read()
    body = (source * (characters // len(source) + 2))[:characters]
    return HEADER + body[:body.rfind(". ") + 1]


def cached(result):
    return ((result["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


def warm(arguments):
    probe = post(arguments.endpoint, {"prompt": document_text(arguments.document, 40000), "max_tokens": 1, "temperature": 0}, 3600)
    if probe["error"] or not probe["usage"]:
        raise SystemExit("probe failed: %s" % json.dumps(probe["error"]))
    characters_per_token = 40000 / probe["usage"]["prompt_tokens"]
    target = arguments.context - arguments.max_tokens - arguments.reserve
    characters = int(target * characters_per_token)
    for _ in range(12):
        text = document_text(arguments.document, characters)
        result = post(arguments.endpoint, {"prompt": text, "max_tokens": 1, "temperature": 0}, arguments.warm_timeout)
        if result["error"] and "context" in json.dumps(result["error"]):
            characters = int(characters * 0.97)
            continue
        if result["error"]:
            raise SystemExit("warm request failed: %s" % json.dumps(result["error"]))
        seconds = result["token_times"][0] if result["token_times"] else None
        return text, {"case": "warm", "prompt_tokens": result["usage"]["prompt_tokens"], "cached_tokens": cached(result),
                      "seconds": round(seconds, 1) if seconds else None, "characters": len(text)}
    raise SystemExit("no document size fits the context")


def question(index, count):
    return ("\n\nQuestion %d of %d: write a long, detailed essay of at least one thousand words on what part %d of the "
            "reference document above describes, in your own words, quoting nothing.\n\nEssay:" % (index + 1, count, index + 1))


def arm(arguments, text, count, speculation):
    results = [None] * count

    def worker(index):
        body = {"prompt": text + question(index, count), "max_tokens": arguments.max_tokens, "temperature": 0}
        if not speculation:
            body["speculation"] = False
        results[index] = post(arguments.endpoint, body, arguments.arm_timeout)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(count)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - started
    streams = []
    for result in results:
        times = result["token_times"]
        stream = {"tokens": len(times), "error": result["error"], "finish_reason": result["finish_reason"],
                  "prompt_tokens": (result["usage"] or {}).get("prompt_tokens"), "cached_tokens": cached(result)}
        if times:
            stream["ttft_s"] = round(times[0], 3)
        if len(times) > 1:
            stream["decode_tok_s"] = round((len(times) - 1) / (times[-1] - times[0]), 2)
        streams.append(stream)
    record = {"case": "b%d" % count, "speculation": speculation, "streams": count, "wall_s": round(wall, 2),
              "errors": [s["error"] for s in streams if s["error"]], "per_stream": streams}
    decoding = [r for r in results if len(r["token_times"]) > 1]
    if len(decoding) == count:
        begin = max(r["started"] + r["token_times"][0] for r in decoding)
        end = min(r["started"] + r["token_times"][-1] for r in decoding)
        if end > begin:
            inside = sum(1 for r in decoding for t in r["token_times"] if begin < r["started"] + t <= end)
            record["together_s"] = round(end - begin, 2)
            record["aggregate_tok_s"] = round(inside / (end - begin), 2)
        rates = [s["decode_tok_s"] for s in streams]
        record["stream_tok_s_median"] = statistics.median(rates)
        record["stream_tok_s_min"] = min(rates)
        record["ttft_s_max"] = max(s["ttft_s"] for s in streams)
        record["cached_tokens_min"] = min((s["cached_tokens"] or 0) for s in streams)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--document", required=True)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reserve", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warm-timeout", type=int, default=14400)
    parser.add_argument("--arm-timeout", type=int, default=3600)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    plan = [(1, True)] + [(int(count), False) for count in arguments.batches.split(",")]
    with open(arguments.output, "a") as handle:
        def emit(record):
            record.update(label=arguments.label, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps({key: value for key, value in record.items() if key != "per_stream"}), flush=True)
        text, record = warm(arguments)
        emit(record)
        for repeat in range(arguments.repeats):
            for count, speculation in plan:
                record = arm(arguments, text, count, speculation)
                record["repeat"] = repeat
                emit(record)


if __name__ == "__main__":
    main()
