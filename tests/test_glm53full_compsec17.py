#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("glm53full_compsec17", ROOT / "tools/glm53full_compsec17.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

CONTRACT = ROOT / "model_contracts/glm53_full_authoritative.json"
FIXTURE = ROOT / "qualification/ds4_eval/quality-fixtures-glm5.3-flash.json"
TOKENIZER = ROOT / "qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"
GMASK, SOP, SYSTEM, USER, ASSISTANT, THINK, UNTHINK = 154822, 154824, 154826, 154827, 154828, 154841, 154842


def prompt_problems(thinking):
    failures = []
    path = ROOT / f"qualification/glm53full/compsec17_prompts_{thinking}.json"
    prompts = json.loads(path.read_text())
    contract = CONTRACT.read_text()
    fixture = {c["id"]: c for c in json.loads(FIXTURE.read_text())["cases"]}
    decode = tool.compsec.load_decoder(TOKENIZER)
    if prompts["format"] != tool.FORMAT or prompts["thinking"] != thinking:
        failures.append(f"{path.name}: format or thinking")
    if f'"sha256": "{prompts["template_sha256"]}"' not in contract or f'"sha256": "{prompts["tokenizer_sha256"]}"' not in contract:
        failures.append(f"{path.name}: template or tokenizer is not the contract's")
    if prompts["tokenizer_sha256"] != hashlib.sha256(TOKENIZER.read_bytes()).hexdigest():
        failures.append(f"{path.name}: tokenizer differs from the fixture tokenizer")
    if prompts["fixture_sha256"] != hashlib.sha256(FIXTURE.read_bytes()).hexdigest():
        failures.append(f"{path.name}: fixture changed since rendering")
    if [c["id"] for c in prompts["cases"]] != tool.compsec.COMPSEC_IDS:
        failures.append(f"{path.name}: case ids")
    tail = [ASSISTANT, THINK, UNTHINK] if thinking == "off" else [ASSISTANT, THINK]
    for case in prompts["cases"]:
        ids = case["prompt_token_ids"]
        user = ids.index(USER) if USER in ids else -1
        if ids[:3] != [GMASK, SOP, SYSTEM] or user < 0 or decode(ids[3:user]) != "Reasoning Effort: Max":
            failures.append(f"{case['id']}: header is not [gMASK]<sop><|system|>Reasoning Effort: Max")
        if ids[user + 1:len(ids) - len(tail)] != fixture[case["id"]]["prompt_token_ids"]:
            failures.append(f"{case['id']}: user turn is not the fixture question")
        if ids[-len(tail):] != tail or case["answer"] != fixture[case["id"]]["answer"]:
            failures.append(f"{case['id']}: generation prompt or answer")
    return failures


def batch_problems():
    failures = []
    prompts = json.loads((ROOT / "qualification/glm53full/compsec17_prompts_off.json").read_text())
    files = tool.batch_files(prompts, 512, 16, 2048)
    requests = [r for f in files for r in f["requests"]]
    if [len(f["requests"]) for f in files] != [16, 1] or [r["request_id"] for r in requests] != list(range(1, 18)):
        failures.append("17 cases at request_capacity 16 are not two files of 16 and 1 with request ids 1..17")
    if any(f["request_capacity"] != 16 or f["stop_token_ids"] != [] for f in files):
        failures.append("batch file capacity or stop list")
    if [r["prompt_token_ids"] for r in requests] != [c["prompt_token_ids"] for c in prompts["cases"]]:
        failures.append("batch prompts differ from the rendered prompts")
    subset = tool.batch_files(prompts, 512, 16, 2048, ["compsec-077", "compsec-082", "compsec-087"])
    if [r["request_id"] for r in subset[0]["requests"]] != [1, 2, 3]:
        failures.append("subset request ids")
    try:
        tool.batch_files(prompts, 2048, 16, 2048)
        failures.append("prompt + budget over max_context_tokens was accepted")
    except SystemExit:
        pass
    try:
        tool.batch_files(prompts, 512, 16, 2048, ["compsec-999"])
        failures.append("an unknown case id was accepted")
    except SystemExit:
        pass
    return failures


def grade_problems():
    failures = []
    prompts = {"thinking": "off", "template_sha256": "t", "eos_token_ids": tool.EOS_TOKEN_IDS,
               "cases": [{"id": "compsec-077", "answer": "18-20"}, {"id": "compsec-082", "answer": "9-10"}, {"id": "compsec-087", "answer": "8,20-22"}]}
    pieces = {1: "reasoning\n", 2: "Answer: 18", 3: "Answer: 6", 4: "Answer: 14,18"}
    decode = lambda ids: "".join(pieces[i] for i in ids)
    summary = tool.grade_cases(prompts, {1: [1, 2, 154827], 2: [1, 3, 154820], 3: [1] * 8}, decode, 8)
    results = {r["id"]: r for r in summary["results"]}
    if summary["completed"] != 3 or summary["passed"] != 1 or not results["compsec-077"]["passed"] or results["compsec-082"]["passed"]:
        failures.append(f"grades: {summary['passed']}/{summary['completed']}")
    if not results["compsec-087"]["hit_budget"] or results["compsec-087"]["stopped_on_eos"] or not results["compsec-077"]["stopped_on_eos"]:
        failures.append("budget and eos flags")
    if results["compsec-077"]["text"] != "reasoning\nAnswer: 18":
        failures.append("eos tokens were not removed from the graded text")
    with tempfile.TemporaryDirectory() as directory:
        events = Path(directory) / "out.ndjson"
        events.write_text('{"schema_version":1,"event":"ready"}\n' + "\n".join(json.dumps({"event": "token", "request_id": 2, "token_id": t}) for t in (5, 6)) + "\nnot json\n")
        if tool.collect_tokens([events]) != {2: [5, 6]}:
            failures.append("collect_tokens")
        events.write_text(json.dumps({"event": "error", "request_id": 1}) + "\n")
        try:
            tool.collect_tokens([events])
            failures.append("an error event was graded")
        except SystemExit:
            pass
    return failures


def main():
    failures = prompt_problems("off") + prompt_problems("on") + batch_problems() + grade_problems()
    for failure in failures:
        print(failure)
    print(f"glm53full compsec17: {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
