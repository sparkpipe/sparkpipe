import argparse
import json
import sys

GENERATOR = "tools/laguna_reference_torch.py"
HEADER_KEYS = ("torch", "device", "config_sha256", "index_sha256", "prompts_sha256", "generated_utc")


class FixtureError(ValueError):
    pass


def margin(step):
    logits = step["fp32_top2_logits"]
    return float(logits[0]) - float(logits[1])


def strict_steps(steps, tie_margin):
    for index, step in enumerate(steps):
        if margin(step) < tie_margin:
            return index
    return len(steps)


def fixture_result(result, decode, tie_margin):
    generated = [int(t) for t in result["generated_token_ids"]]
    steps = result["steps"]
    if [int(s["token"]) for s in steps] != generated:
        raise FixtureError(f"{result['name']}: step tokens disagree with generated_token_ids")
    for step in steps:
        if len(step["fp32_top2_tokens"]) != 2 or len(step["fp32_top2_logits"]) != 2:
            raise FixtureError(f"{result['name']}: every step needs the f32 top-2 tokens and logits")
        if int(step["token"]) not in [int(t) for t in step["fp32_top2_tokens"]]:
            raise FixtureError(f"{result['name']}: step at position {step['position']} emitted a token outside the f32 top-2")
    return {
        "name": result["name"],
        "prompt_token_ids": [int(t) for t in result["prompt_token_ids"]],
        "generated_token_ids": generated,
        "generated_text": decode(generated),
        "strict_steps": strict_steps(steps, tie_margin),
        "steps": [{"position": int(s["position"]), "token": int(s["token"]),
                   "fp32_top2_tokens": [int(t) for t in s["fp32_top2_tokens"]],
                   "fp32_top2_logits": [round(float(v), 4) for v in s["fp32_top2_logits"]]} for s in steps],
    }


def build_fixture(raw, checkpoint_label, decode, tie_margin):
    if raw.get("generator") != GENERATOR:
        raise FixtureError(f"input was not written by {GENERATOR}")
    if not checkpoint_label.strip():
        raise FixtureError("checkpoint label must name the checkpoint and revision")
    if not tie_margin > 0:
        raise FixtureError("tie margin must be positive")
    document = {"generator": GENERATOR}
    for key in HEADER_KEYS:
        if key not in raw:
            raise FixtureError(f"input lacks {key}")
    document["torch"] = raw["torch"]
    document["device"] = raw["device"]
    document["checkpoint"] = checkpoint_label
    for key in HEADER_KEYS[2:]:
        document[key] = raw[key]
    document["tie_margin"] = tie_margin
    document["results"] = [fixture_result(r, decode, tie_margin) for r in raw["results"]]
    return document


def encode(document):
    return json.dumps(document, indent=1) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--tie-margin", type=float, required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(arguments.tokenizer)
    raw = json.load(open(arguments.raw))
    document = build_fixture(raw, arguments.checkpoint_label, tokenizer.decode, arguments.tie_margin)
    with open(arguments.output, "w") as handle:
        handle.write(encode(document))
    print(json.dumps({r["name"]: r["strict_steps"] for r in document["results"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
