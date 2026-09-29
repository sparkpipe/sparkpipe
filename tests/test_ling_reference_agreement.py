#!/usr/bin/env python3
import copy
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))

from t1_reference_common import read_fixture

RECEIPT = REPOSITORY / "qualification/ling_reference/ling_hf_reference.json"
SERVING_RECEIPT = REPOSITORY / "qualification/ling_reference/ling_hf_reference_serving_limits.json"
SWIGLU_LIMITED_LAYERS = list(range(34, 42))
PROMPTS = REPOSITORY / "qualification/ling_reference/prompts.json"
MODELING = REPOSITORY / "model_contracts/references/modeling_ling_bailing_moe_v3.py"
T1_DIRECTORY = REPOSITORY / "qualification/t1_reference/ling"
NEW_TOKENS = 16


def fixture_stream(name):
    _, arrays = read_fixture(T1_DIRECTORY / f"{name}.t1r")
    prompt = arrays["prompt_token_ids"].tolist()
    tops = sorted(key for key in arrays if key.endswith("_head_top1_token"))
    return prompt, [int(arrays[key][0]) for key in tops]


def receipt_problems(receipt, prompts, modeling_sha256, t1_manifest, t1_streams):
    problems = []
    if receipt.get("modeling_sha256") != modeling_sha256:
        problems.append("receipt modeling sha256 differs from the committed publisher modeling")
    if receipt.get("config_sha256") != t1_manifest["checkpoint"]["config_sha256"] or receipt.get("index_sha256") != t1_manifest["checkpoint"]["index_sha256"]:
        problems.append("receipt checkpoint differs from the T1 fixture checkpoint")
    if receipt.get("new_tokens") != NEW_TOKENS:
        problems.append(f"receipt new_tokens {receipt.get('new_tokens')} != {NEW_TOKENS}")
    results = {result["name"]: result for result in receipt.get("results", [])}
    if sorted(results) != sorted(prompt["name"] for prompt in prompts):
        problems.append("receipt prompt set differs from qualification/ling_reference/prompts.json")
    for prompt in prompts:
        result = results.get(prompt["name"])
        if result is None:
            continue
        if result["text"] != prompt["text"]:
            problems.append(f"{prompt['name']}: prompt text differs")
        if "prompt_token_ids" in prompt and result["prompt_token_ids"] != prompt["prompt_token_ids"]:
            problems.append(f"{prompt['name']}: prompt token ids differ from the pinned ids")
        tokens = result["tokens"]
        if len(tokens) != NEW_TOKENS or len(result["steps"]) != NEW_TOKENS:
            problems.append(f"{prompt['name']}: {len(tokens)} tokens, expected {NEW_TOKENS}")
            continue
        for index, (token, step) in enumerate(zip(tokens, result["steps"])):
            scores = [score for _, score in step["top"]]
            if step["token"] != token or step["top"][0][0] != token or scores != sorted(scores, reverse=True):
                problems.append(f"{prompt['name']}: step {index} is not the greedy argmax of its logits")
                break
        if 156895 in tokens:
            problems.append(f"{prompt['name']}: greedy stream reaches end of text inside {NEW_TOKENS} tokens")
    for name, (prompt_ids, stream) in t1_streams.items():
        result = results.get(name)
        if result is None:
            problems.append(f"{name}: T1 fixture prompt missing from the receipt")
            continue
        if result["prompt_token_ids"] != prompt_ids:
            problems.append(f"{name}: T1 fixture prompt ids differ")
        if result["tokens"][:len(stream)] != stream:
            problems.append(f"{name}: T1 fixture stream {stream} differs from the publisher-code stream {result['tokens'][:len(stream)]}")
    return problems


def main():
    receipt = json.loads(RECEIPT.read_text())
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    modeling_sha256 = hashlib.sha256(MODELING.read_bytes()).hexdigest()
    t1_manifest = json.loads((T1_DIRECTORY / "MANIFEST.json").read_text())
    t1_streams = {name: fixture_stream(name) for name in ("capital_of_france", "count_up")}
    failures = receipt_problems(receipt, prompts, modeling_sha256, t1_manifest, t1_streams)
    if receipt.get("swiglu_limits", "modeling") != "modeling":
        failures.append("the primary receipt must follow the publisher modeling (no SwiGLU clamp)")
    serving = json.loads(SERVING_RECEIPT.read_text())
    failures += [f"serving-limits receipt: {problem}" for problem in receipt_problems(serving, prompts, modeling_sha256, t1_manifest, t1_streams)]
    if serving.get("swiglu_limits") != "serving" or serving.get("swiglu_limited_layers") != SWIGLU_LIMITED_LAYERS:
        failures.append(f"serving-limits receipt clamps layers {serving.get('swiglu_limited_layers')}, expected {SWIGLU_LIMITED_LAYERS}")
    controls = {
        "token flip": lambda r: r["results"][0]["tokens"].__setitem__(2, r["results"][0]["tokens"][2] + 1),
        "short stream": lambda r: r["results"][1]["tokens"].pop(),
        "modeling drift": lambda r: r.update(modeling_sha256="0" * 64),
        "checkpoint drift": lambda r: r.update(config_sha256="1" * 64),
        "non-argmax step": lambda r: r["results"][2]["steps"][5]["top"].reverse(),
        "prompt drift": lambda r: r["results"][1]["prompt_token_ids"].__setitem__(0, 1),
    }
    for label, edit in controls.items():
        corrupted = copy.deepcopy(receipt)
        edit(corrupted)
        if not receipt_problems(corrupted, prompts, modeling_sha256, t1_manifest, t1_streams):
            failures.append(f"negative control not convicted: {label}")
    if failures:
        for failure in failures:
            print("FAIL", failure)
        return 1
    matched = sum(len(stream) for _, stream in t1_streams.values())
    print(f"PASS ling publisher-code reference: {len(prompts)} prompts x {NEW_TOKENS} greedy tokens, with and without the serving SwiGLU clamp; T1 fixture streams agree on {matched} tokens; {len(controls)} negative controls convicted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
