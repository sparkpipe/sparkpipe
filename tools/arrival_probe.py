#!/usr/bin/env python3
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

port, tag, rounds, tokens = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
STREAM = "Once upon a time, in a village by the sea,"


def post(body):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.monotonic()
    reply = json.load(urllib.request.urlopen(request, timeout=600))
    return reply, time.monotonic() - started


def stream():
    reply, seconds = post({"prompt": STREAM, "max_tokens": tokens, "temperature": 0})
    return len(reply.get("tokens") or []), seconds


def arrival(label):
    nonce = f"{tag}-{label}-{random.random():.9f}"
    _, seconds = post({"prompt": f"Note {nonce}: the capital of France is", "max_tokens": 1, "temperature": 0, "speculation": False})
    return seconds


def summary(name, values):
    if not values:
        return f"{name} n=0"
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return f"{name} n={len(ordered)} median_ms={statistics.median(ordered) * 1e3:.1f} mean_ms={statistics.fmean(ordered) * 1e3:.1f} p90_ms={pick(0.9) * 1e3:.1f} max_ms={ordered[-1] * 1e3:.1f}"


random.seed(1234)
learned = stream()
drafted = stream()
alone = [arrival("alone") for _ in range(30)]
during, streams = [], []
for index in range(rounds):
    done = threading.Event()
    box = {}

    def run():
        box["stream"] = stream()
        done.set()

    worker = threading.Thread(target=run)
    worker.start()
    time.sleep(0.5)
    while not done.is_set():
        seconds = arrival(f"during{index}")
        if not done.is_set():
            during.append(seconds)
        time.sleep(random.uniform(0.0, 0.04))
    worker.join()
    streams.append(box["stream"])
print(f"probe {tag} stream_learn tokens={learned[0]} seconds={learned[1]:.3f} stream_drafted tokens={drafted[0]} seconds={drafted[1]:.3f} tok_s={drafted[0] / drafted[1]:.1f}")
print(f"probe {tag} " + summary("arrival_alone", alone))
print(f"probe {tag} " + summary("arrival_during_stream", during))
print(f"probe {tag} stream_with_arrivals " + " ".join(f"{count}tok/{seconds:.2f}s={count / seconds:.1f}tok_s" for count, seconds in streams))
