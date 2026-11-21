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

for failure in failures:
    print("FAIL test_prompt_cuts: " + failure)
if failures:
    sys.exit(1)
print("PASS test_prompt_cuts: a structural cut leaves the margin before a block's end, and the margin command measures what the margin has to cover")
