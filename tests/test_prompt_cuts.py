#!/usr/bin/env python3
"""A structural cut is the last cache block at or before a block's own token count less the margin, so it never names a token the tokenizer may change when more text follows; the margin tool counts, for each paragraph end of a text, how many trailing tokens change under a tokenizer, and fails when the margin does not stay above the worst count."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("prompt_cuts", ROOT / "tools" / "serving" / "prompt_cuts.py")
prompt_cuts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prompt_cuts)
FIXTURE = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
fixture = importlib.util.module_from_spec(FIXTURE)
FIXTURE.loader.exec_module(fixture)

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


block, margin = prompt_cuts.BLOCK_TOKENS, prompt_cuts.MARGIN_TOKENS
check([prompt_cuts.structural_cut(count) for count in (0, margin, block + margin - 1, block + margin, 2 * block + margin + 5, 1000)] ==
      [0, 0, 0, block, 2 * block, (1000 - margin) // block * block], "a cut is the last cache block at or before the count less the margin, and 0 when there is none")
check(all(prompt_cuts.structural_cut(count) % block == 0 and 0 <= count - margin - prompt_cuts.structural_cut(count) < block for count in range(block + margin, 4000)),
      "every cut leaves the margin and under one block before the block's end")
check(prompt_cuts.changed_tail([1, 2, 3, 4], [1, 2, 3, 4, 9]) == 0 and prompt_cuts.changed_tail([1, 2, 3, 4], [1, 2, 7, 8, 9]) == 2 and prompt_cuts.changed_tail([], [1]) == 0,
      "the changed tail is how many trailing tokens of the text alone are not kept when more text follows")
with tempfile.TemporaryDirectory() as directory:
    model = fixture.model_directory(Path(directory) / "model")
    text = Path(directory) / "text.md"
    text.write_text("\n\n".join(" ".join(fixture.CORPUS[index % len(fixture.CORPUS)] for _ in range(3)) + "." for index in range(12)))
    encode = prompt_cuts.load_encoder(model)
    report = prompt_cuts.measure(encode, [text.read_text()], 400, 4000)
    check(report["samples"] == 11 * len(prompt_cuts.JOINS) and report["worst"] < margin and sum(report["changed_tail_tokens"].values()) == report["samples"],
          f"the measure takes every paragraph end with every join and stays under the margin on the fixture tokenizer: {report}")
    stdout = sys.stdout
    try:
        sys.stdout = open(Path(directory) / "out.json", "w")
        status = prompt_cuts.main(["margin", "--tokenizer", str(model), "--text", str(text)])
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    printed = json.loads((Path(directory) / "out.json").read_text())
    check(status == 0 and printed["margin_holds"] is True and printed["margin_tokens"] == margin and list(printed["tokenizers"]) == [str(model)],
          f"the margin command reports each tokenizer and whether the margin holds: {status} {printed}")
    wide = prompt_cuts.MARGIN_TOKENS
    prompt_cuts.MARGIN_TOKENS = 0
    try:
        sys.stdout = open(Path(directory) / "out.json", "w")
        status = prompt_cuts.main(["margin", "--tokenizer", str(model), "--text", str(text)])
    finally:
        sys.stdout.close()
        sys.stdout = stdout
        prompt_cuts.MARGIN_TOKENS = wide
    check(status == 1, "a margin that is not above the worst count fails the command")

lanes = json.loads((ROOT / "deployment" / "lanes.json").read_text())["lanes"]
table = json.loads((ROOT / "tools" / "serving" / "prompt_cut_margins.json").read_text())
named = [lane for record in table["records"] for lane in record["lanes"]]
check(table["margin_tokens"] == margin and named and len(named) == len(set(named)) and all(lane in lanes for lane in named),
      f"the margin records name served lanes, each once, under the margin the cut rule uses: {named}")
check(all(record["samples"] > 0 and 0 <= record["worst"] < margin and len({lanes[lane]["tokenizer_sha256"] for lane in record["lanes"]}) == 1 for record in table["records"]),
      "a margin record holds a measurement under the margin, for lanes that serve one tokenizer")
measured = prompt_cuts.margin_record("qwen3.8-27b-fp8")
check(measured is not None and measured["worst"] < margin and prompt_cuts.margin_record("no-such-lane") is None and
      sorted(lane for lane in lanes if prompt_cuts.margin_record(lane) is None) == sorted(lane for lane in lanes if lane not in named),
      "a lane with a record answers with it and a lane without one answers nothing")
with tempfile.TemporaryDirectory() as directory:
    other = Path(directory) / "margins.json"
    other.write_text(json.dumps(dict(table, margin_tokens=margin + 1)))
    stale = prompt_cuts.margin_record("qwen3.8-27b-fp8", other)
    other.write_text(json.dumps(dict(table, records=[dict(table["records"][0], worst=margin)])))
    check(stale is None and prompt_cuts.margin_record("qwen3.8-27b-fp8", other) is None and prompt_cuts.margin_record("qwen3.8-27b-fp8", Path(directory) / "absent.json") is None,
          "records measured for another margin, a record whose worst count reaches the margin, and a missing table all answer nothing")

for failure in failures:
    print("FAIL test_prompt_cuts: " + failure)
if failures:
    sys.exit(1)
print("PASS test_prompt_cuts: a structural cut leaves the margin before a block's end, and the margin command measures what the margin has to cover")
