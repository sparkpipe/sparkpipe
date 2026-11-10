#!/usr/bin/env python3
"""Check a served model against a reference greedy run of the same pack.

For each prompt of a gate prompt file (prompt_token_ids) it compares, through
the model API's /v1/completions:
- greedy tokens with the reference's, and at the first difference the
  logprobs of the reference's token and the served token there, so a near tie
  shows as a small gap;
- a continuation from a cached prefix (the prompt plus the first half of the
  reference's tokens) with the second half, and how many prompt tokens came
  from the cache;
- seeded sampling twice with one seed and once with another;
- the prompts sent at once with the same prompts sent one at a time.

  serving_greedy_check.py --prompts GATE_PROMPTS.json --reference GREEDY.json --url http://127.0.0.1:8471 --out REPORT.json
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def ask(url, ids, max_tokens, **extra):
    body = dict({"prompt_token_ids": ids, "max_tokens": max_tokens, "temperature": 0, "stream": False}, **extra)
    request = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=3600) as response:
        reply = json.load(response)
    return reply, round(time.monotonic() - started, 3)


def first_difference(left, right):
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), None)


def greedy(url, prompt, want):
    reply, seconds = ask(url, prompt["prompt_token_ids"], len(want))
    record = {"name": prompt["name"], "match": reply["tokens"] == want, "tokens": reply["tokens"], "seconds": seconds, "usage": reply["usage"]}
    index = first_difference(reply["tokens"], want)
    if index is not None:
        probe, _ = ask(url, prompt["prompt_token_ids"] + want[:index], 1, logprobs=5)
        top = {token: logprob for token, logprob in probe["token_logprobs"][0][1:]}
        record.update(first_difference=index, served=reply["tokens"][index], reference=want[index],
                      served_logprob=top.get(reply["tokens"][index]), reference_logprob=top.get(want[index]))
    return record


def resume(url, prompt, want):
    half = len(want) // 2
    reply, seconds = ask(url, prompt["prompt_token_ids"] + want[:half], len(want) - half)
    return {"name": prompt["name"], "match": reply["tokens"] == want[half:], "prompt_tokens": reply["usage"]["prompt_tokens"],
            "cached_tokens": reply["usage"]["prompt_tokens_details"]["cached_tokens"], "seconds": seconds}


def sampling(url, prompt, count):
    rules = {"temperature": 0.8, "top_p": 0.95, "top_k": 50}
    draws = [ask(url, prompt["prompt_token_ids"], count, seed=seed, **rules)[0]["tokens"] for seed in (7, 7, 8)]
    return {"name": prompt["name"], "rules": rules, "same_seed_identical": draws[0] == draws[1], "other_seed_differs": draws[2] != draws[0],
            "seed_7": draws[0], "seed_8": draws[2]}


def concurrent(url, prompts, reference):
    requests = [(prompt["prompt_token_ids"], len(reference[prompt["name"]])) for prompt in prompts]
    with ThreadPoolExecutor(len(requests)) as pool:
        started = time.monotonic()
        together = list(pool.map(lambda request: ask(url, *request), requests))
        wall = round(time.monotonic() - started, 3)
    alone = [ask(url, *request) for request in requests]
    return {"wall_seconds": wall, "prompts": [{"name": prompt["name"], "match_alone": a[0]["tokens"] == b[0]["tokens"], "seconds": a[1]}
                                              for prompt, a, b in zip(prompts, together, alone)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    prompts = json.load(open(args.prompts))["prompts"]
    reference = {prompt["name"]: prompt["generated"] for prompt in json.load(open(args.reference))["prompts"]}
    report = {"prompts": args.prompts, "reference": args.reference, "url": args.url}
    report["greedy"] = [greedy(args.url, prompt, reference[prompt["name"]]) for prompt in prompts]
    report["resume"] = [resume(args.url, prompt, reference[prompt["name"]]) for prompt in prompts]
    report["sampling"] = sampling(args.url, prompts[0], len(reference[prompts[0]["name"]]))
    report["concurrent"] = concurrent(args.url, prompts, reference)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    for kind in ("greedy", "resume"):
        for record in report[kind]:
            print(kind, json.dumps({key: value for key, value in record.items() if key != "tokens"}))
    print("sampling", json.dumps({k: v for k, v in report["sampling"].items() if not k.startswith("seed_")}))
    print("concurrent", json.dumps(report["concurrent"]))


if __name__ == "__main__":
    main()
