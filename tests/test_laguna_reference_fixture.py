import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import laguna_reference_fixture as fixture

FAMILY = ROOT / "model-families" / "laguna"


def step(position, token, top, logits):
    return {"position": position, "token": token, "bf16_logit": logits[0], "fp32_top2_tokens": top,
            "fp32_top2_logits": logits, "seconds": 1.0, "routes": [{"layer": 1, "experts": [0], "weights": [1.0]}]}


def raw_document():
    return {
        "generator": "tools/laguna_reference_torch.py", "torch": "t", "device": "d", "checkpoint": "/abs/path",
        "config_sha256": "c", "index_sha256": "i", "prompts_sha256": "p", "generated_utc": "u",
        "results": [
            {"name": "clear", "prompt_token_ids": [2, 5], "generated_token_ids": [7, 8, 9],
             "steps": [step(1, 7, [7, 3], [10.0, 9.0]), step(2, 8, [8, 4], [5.123456, 1.0]), step(3, 9, [9, 1], [2.0, 1.5])]},
            {"name": "tied", "prompt_token_ids": [2, 6], "generated_token_ids": [4, 3, 1],
             "steps": [step(1, 4, [4, 3], [10.0, 9.0]), step(2, 3, [5, 3], [8.00001, 8.0]), step(3, 1, [1, 2], [9.0, 1.0])]},
        ],
    }


def decode(ids):
    return "|".join(str(i) for i in ids)


def check_build(failures):
    document = fixture.build_fixture(raw_document(), "label rev", decode, 0.1)
    if document["checkpoint"] != "label rev" or document["tie_margin"] != 0.1:
        failures.append(f"header not carried: {document}")
    if list(document) != ["generator", "torch", "device", "checkpoint", "config_sha256", "index_sha256",
                          "prompts_sha256", "generated_utc", "tie_margin", "results"]:
        failures.append(f"header key order changed: {list(document)}")
    clear, tied = document["results"]
    if clear["strict_steps"] != 3 or tied["strict_steps"] != 1:
        failures.append(f"strict steps wrong: {clear['strict_steps']} {tied['strict_steps']}")
    if clear["generated_text"] != "7|8|9":
        failures.append(f"text not decoded: {clear['generated_text']}")
    if clear["steps"][1]["fp32_top2_logits"] != [5.1235, 1.0]:
        failures.append(f"logits not rounded to 4 places: {clear['steps'][1]}")
    if set(clear["steps"][0]) != {"position", "token", "fp32_top2_tokens", "fp32_top2_logits"}:
        failures.append(f"step keys: {sorted(clear['steps'][0])}")
    if fixture.build_fixture(raw_document(), "label rev", decode, 0.6)["results"][0]["strict_steps"] != 2:
        failures.append("a margin below the tie margin must end the strict prefix")


def refused(mutate, text):
    raw = raw_document()
    label = mutate(raw)
    try:
        fixture.build_fixture(raw, label or "label rev", decode, 0.1)
    except fixture.FixtureError as error:
        return text in str(error)
    return False


def check_refusals(failures):
    def wrong_generator(raw):
        raw["generator"] = "other"

    def token_mismatch(raw):
        raw["results"][0]["generated_token_ids"][1] = 99

    def outside_top2(raw):
        raw["results"][0]["steps"][0]["fp32_top2_tokens"] = [3, 1]

    def missing_header(raw):
        del raw["prompts_sha256"]

    def empty_label(raw):
        return " "

    cases = [(wrong_generator, "written by"), (token_mismatch, "disagree"), (outside_top2, "outside the f32 top-2"),
             (missing_header, "prompts_sha256"), (empty_label, "label")]
    for mutate, text in cases:
        if not refused(mutate, text):
            failures.append(f"{mutate.__name__} must be refused")
    try:
        fixture.build_fixture(raw_document(), "label rev", decode, 0.0)
        failures.append("a zero tie margin must be refused")
    except fixture.FixtureError:
        pass


def check_committed_fixture(failures):
    prompts_path = FAMILY / "reference_prompts.json"
    document = json.loads((FAMILY / "reference_tokens.json").read_text())
    prompts = {p["name"]: p for p in json.loads(prompts_path.read_text())["prompts"]}
    if document["prompts_sha256"] != hashlib.sha256(prompts_path.read_bytes()).hexdigest():
        failures.append("reference_tokens.json was generated from a different reference_prompts.json")
    if [r["name"] for r in document["results"]] != list(prompts):
        failures.append("committed results do not cover the reference prompts in order")
    margin = document["tie_margin"]
    for result in document["results"]:
        name = result["name"]
        if result["prompt_token_ids"] != prompts[name]["prompt_token_ids"]:
            failures.append(f"{name}: prompt ids differ from reference_prompts.json")
        if len(result["generated_token_ids"]) != prompts[name]["new_tokens"]:
            failures.append(f"{name}: expected {prompts[name]['new_tokens']} tokens")
        if [s["token"] for s in result["steps"]] != result["generated_token_ids"]:
            failures.append(f"{name}: step tokens differ from generated ids")
        if result["strict_steps"] != fixture.strict_steps(result["steps"], margin):
            failures.append(f"{name}: strict_steps does not follow tie_margin {margin}")
    expected = {"capital_of_france": 13, "python_is_prime": 16, "count_up": 13}
    got = {r["name"]: r["strict_steps"] for r in document["results"]}
    if got != expected:
        failures.append(f"strict prefixes changed: {got}")
    regenerated = fixture.encode(document)
    if regenerated != (FAMILY / "reference_tokens.json").read_text():
        failures.append("reference_tokens.json is not in the tool's output encoding")


def main():
    failures = []
    check_build(failures)
    check_refusals(failures)
    check_committed_fixture(failures)
    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print("PASS laguna reference fixture: build, strict prefix, refusals, committed fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
