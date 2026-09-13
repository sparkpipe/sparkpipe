#!/usr/bin/env python3
"""Generate DSV4 Flash TP4 benchmark-matrix batch files.

Cells: ctx in {512, 2048} x B in {1, 4, 16} x kind in {prefill, decode}.
Prompts are exact-length tilings of the canonical O128 prompt block so token
ids stay in-distribution. Prefill cells use output_token_budget=2 (the stream
harness requires >=2 token events); decode cells use the canonical 128.

usage: gen_matrix_batches.py OUTDIR
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CANONICAL = os.path.join(HERE, "..", "batches", "o128_batch.json")

CTXS = (512, 2048)
BS = (1, 4, 16)
SEQ_BASE = 761000

def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    src = json.load(open(CANONICAL))
    block = src["requests"][0]["prompt_token_ids"]  # 128 ids
    manifest = []
    cell = 0
    for ctx in CTXS:
        assert ctx % len(block) == 0
        prompt = (block * (ctx // len(block)))[:ctx]
        for b in BS:
            for kind, budget in (("prefill", 2), ("decode", 128)):
                doc = {
                    "schema_version": 1,
                    "connect_timeout_ms": 30000,
                    "request_capacity": b,
                    "max_context_tokens": 4096,
                    "max_prefill_rows_per_submission": b,
                    "maximum_messages_per_rank_per_progress": 8 * b,
                    "maximum_new_submissions_per_progress": 1,
                    "stop_token_ids": [],
                    "requests": [],
                }
                for i in range(b):
                    rid = SEQ_BASE + cell * 10 + i
                    doc["requests"].append({
                        "request_id": rid,
                        "sequence_id": rid,
                        "priority": 0,
                        "output_token_budget": budget,
                        "prompt_token_ids": list(prompt),
                    })
                name = f"{kind}_ctx{ctx}_b{b}.json"
                path = os.path.join(outdir, name)
                with open(path, "w") as f:
                    json.dump(doc, f)
                manifest.append({"cell": f"{kind}_ctx{ctx}_b{b}", "file": path,
                                 "prompt_tokens_total": ctx * b})
                cell += 1
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"generated {len(manifest)} batches -> {outdir}")

if __name__ == "__main__":
    main()
