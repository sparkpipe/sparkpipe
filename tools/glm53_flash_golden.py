#!/usr/bin/env python3
"""Capture or replay the GLM-5.3 Flash serving golden through a model API.

The same prompt set runs against the legacy driver (golden) and the common
runner port (replay), so the two archives compare token for token:

  B1        every prompt alone, greedy, top-4 logprobs
  B4        four short prompts submitted together
  reuse     a second question on the 8K document (prefix restore), then the
            first 8K prompt again (whole-prompt restore)

Prompts are rendered with the GLM chat template from the publisher tokenizer
and repository documents (fixed truncation), so they are reproducible.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

SHORT = [
    ("capital", "What is the capital of France? Answer in one sentence."),
    ("arith", "Compute 37 * 43 and explain the steps briefly."),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("story", "Write the opening paragraph of a short story about a lighthouse keeper who finds a message in a bottle."),
]
BATCH = [
    ("batch_haiku", "Write a haiku about autumn rain."),
    ("batch_sort", "Explain how merge sort works in three sentences."),
    ("batch_planet", "Name the planets of the solar system in order from the sun."),
    ("batch_sql", "Write an SQL query that returns the ten most recent orders from a table named orders."),
]
DOCS = [
    ("doc2k_kernel", "KERNEL_PLAYBOOK.md", 1900, "Summarize the main points of the document above in five bullet points."),
    ("doc2k_contracts", "INTERFACE_CONTRACTS.md", 2010, "What are the three most important rules in this document? Explain each briefly."),
    ("doc8k_weightd", "WEIGHTD_DESIGN.md", 8000, "What problem does this design solve, and what are its main components?"),
    ("doc8k_dma", "CX7_DMA_INVESTIGATION_2026-10.md", 8000, "Summarize the investigation: what was observed, what was ruled out, and what remains open?"),
]
REUSE_QUESTION = "List every daemon, socket or file this design names, with one line on its role."


def render(tokenizer, text: str, document: list[int] | None = None) -> list[int]:
    prefix = tokenizer.encode("[gMASK]<sop><|user|>\n", add_special_tokens=False).ids
    body = tokenizer.encode(text, add_special_tokens=False).ids
    tail = tokenizer.encode("<|assistant|>\n<think></think>\n", add_special_tokens=False).ids
    if document is None:
        return prefix + body + tail
    newline = tokenizer.encode("\n\n", add_special_tokens=False).ids
    return prefix + document + newline + body + tail


def prompts(tokenizer_path: str, docs_dir: str) -> list[dict]:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    out = []
    for name, text in SHORT + BATCH:
        out.append({"name": name, "group": "batch" if name.startswith("batch_") else "short", "prompt_token_ids": render(tokenizer, text)})
    for name, path, count, question in DOCS:
        with open(os.path.join(docs_dir, path), encoding="utf-8") as handle:
            ids = tokenizer.encode(handle.read(), add_special_tokens=False).ids
        if len(ids) < count:
            raise SystemExit(f"{path} has {len(ids)} tokens, fewer than {count}")
        out.append({"name": name, "group": "doc", "document": path, "document_tokens": count,
                    "prompt_token_ids": render(tokenizer, question, ids[:count])})
        if name == "doc8k_weightd":
            out.append({"name": "reuse_weightd_q2", "group": "reuse", "document": path, "document_tokens": count,
                        "prompt_token_ids": render(tokenizer, REUSE_QUESTION, ids[:count])})
    return out


def request(port: int, ids: list[int], tokens: int, logprobs: int) -> dict:
    fields = {"prompt_token_ids": ids, "max_tokens": tokens, "temperature": 0}
    if logprobs:
        fields["logprobs"] = logprobs
    message = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(fields).encode(),
                                     {"Content-Type": "application/json"})
    try:
        reply = json.load(urllib.request.urlopen(message, timeout=1800))
    except urllib.error.HTTPError as error:
        return {"error": f"HTTP {error.code}: {error.read().decode(errors='replace')[:300]}", "logprobs_requested": logprobs}
    return {"tokens": reply["tokens"], "logprobs": reply.get("token_logprobs"), "text": reply.get("text", ""),
            "finish_reason": reply.get("finish_reason"), "logprobs_requested": logprobs}


LOGPROBS = 4


def complete(port: int, ids: list[int], tokens: int) -> dict:
    return request(port, ids, tokens, LOGPROBS)


def health(port: int) -> str:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30).read().decode(errors="replace")
    except Exception as error:
        return f"unavailable: {error}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="/mnt/cold-raid6/models/glm-5.3-flash-bf16-official/tokenizer.json")
    parser.add_argument("--docs", default=os.path.expanduser("~/api-build-72eaa5fa3/docs"))
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--label", required=True)
    parser.add_argument("--logprobs", type=int, default=4, help="top logprobs per token; 0 asks for tokens only")
    parser.add_argument("--probe-logprobs", action="store_true", help="after the tokens-only run, send one top-4 logprob request and record its reply")
    args = parser.parse_args()
    global LOGPROBS
    LOGPROBS = args.logprobs
    items = prompts(args.tokenizer, args.docs)
    by_name = {item["name"]: item for item in items}
    result = {"label": args.label, "port": args.port, "logprobs": args.logprobs, "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "health": health(args.port), "max_tokens": args.tokens, "prompts": items, "b1": {}, "b4": {}, "reuse": {}}
    for item in items:
        if item["group"] == "reuse":
            continue
        result["b1"][item["name"]] = complete(args.port, item["prompt_token_ids"], args.tokens)
        entry = result["b1"][item["name"]]
        print("b1", item["name"], len(item["prompt_token_ids"]), entry.get("error") or repr(entry["text"][:100]), flush=True)
    names = [name for name, _ in BATCH]
    with concurrent.futures.ThreadPoolExecutor(len(names)) as pool:
        futures = {name: pool.submit(complete, args.port, by_name[name]["prompt_token_ids"], args.tokens) for name in names}
        for name in names:
            result["b4"][name] = futures[name].result()
    for name in names:
        print("b4", name, "tokens_equal_b1", result["b4"][name].get("tokens") == result["b1"][name].get("tokens"), flush=True)
    result["reuse"]["reuse_weightd_q2"] = complete(args.port, by_name["reuse_weightd_q2"]["prompt_token_ids"], args.tokens)
    result["reuse"]["doc8k_weightd_again"] = complete(args.port, by_name["doc8k_weightd"]["prompt_token_ids"], args.tokens)
    print("reuse q2", repr(result["reuse"]["reuse_weightd_q2"].get("text", "")[:100]), flush=True)
    print("reuse again tokens_equal_b1", result["reuse"]["doc8k_weightd_again"].get("tokens") == result["b1"]["doc8k_weightd"].get("tokens"), flush=True)
    if args.probe_logprobs:
        result["logprob_probe"] = request(args.port, by_name["capital"]["prompt_token_ids"], 8, 4)
        print("logprob probe", result["logprob_probe"].get("error") or result["logprob_probe"].get("tokens"), flush=True)
    result["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(args.output, "w") as handle:
        json.dump(result, handle)
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
