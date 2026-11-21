#!/usr/bin/env python3
"""Where a prompt's structure ends a prefill wave (docs/EXACTNESS_GATE.md).

A request resumes only at a wave end it names, so a prefix that several prompts share is reused only when each of them names where it ends. The end of a block of
text (a system block, the tools, a content block of the first message) is not a token boundary: the tokenizer may merge its last characters with what follows. The
structural cut for a block is therefore taken from the block's own text: count the tokens of the prompt's text through the block's end, tokenized by itself, take
MARGIN_TOKENS off, and round down to a cache block. Two prompts that share the text through the block get the same cut, and the tokens before it are the same in
both unless the tokenizer changed more than the margin's worth of trailing tokens at the join; then the two prompts share less and only miss.

This module has no dependencies: the chat layer and gate GX both take the cut from structural_cut.

  prompt_cuts.py margin --tokenizer DIR_OR_JSON... --text FILE...

measures the margin's evidence: for every paragraph end in the given texts, how many trailing tokens of the text before it change when the rest of the text
follows, under each tokenizer. MARGIN_TOKENS has to stay above the largest count seen.
"""
import argparse
import json
import sys
from pathlib import Path

BLOCK_TOKENS = 64
MARGIN_TOKENS = 8
JOINS = ("\n\n", "\n", " ", "")


def structural_cut(prefix_tokens, block=BLOCK_TOKENS, margin=MARGIN_TOKENS):
    return max(0, prefix_tokens - margin) // block * block


def changed_tail(alone, joined):
    same = 0
    for left, right in zip(alone, joined):
        if left != right:
            break
        same += 1
    return len(alone) - same


def paragraph_ends(text, limit):
    ends, at = [], text.find("\n\n")
    while at > 0 and len(ends) < limit:
        if text[at - 1] not in "\n" and at + 2 < len(text):
            ends.append(at)
        at = text.find("\n\n", at + 2)
    return ends


def measure(encode, texts, limit, window):
    worst, counts, samples = 0, {}, 0
    for text in texts:
        for end in paragraph_ends(text, limit):
            start = max(0, end - window)
            prefix, rest = text[start:end], text[end:].lstrip("\n")[:window]
            alone = encode(prefix)
            for join in JOINS:
                changed = changed_tail(alone, encode(prefix + join + rest))
                counts[changed] = counts.get(changed, 0) + 1
                worst = max(worst, changed)
                samples += 1
    return {"samples": samples, "worst": worst, "changed_tail_tokens": {str(key): counts[key] for key in sorted(counts)}}


def load_encoder(path):
    from tokenizers import Tokenizer
    source = Path(path)
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json" if source.is_dir() else source))
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    margin = commands.add_parser("margin")
    margin.add_argument("--tokenizer", action="append", required=True, help="a model directory or its tokenizer.json")
    margin.add_argument("--text", action="append", required=True, help="a text file whose paragraph ends stand for block ends")
    margin.add_argument("--limit", type=int, default=400, help="paragraph ends taken from each text")
    margin.add_argument("--window", type=int, default=4000, help="characters of text taken before and after each end")
    arguments = parser.parse_args(argv)
    texts = [Path(name).read_text(errors="replace") for name in arguments.text]
    report = {"margin_tokens": MARGIN_TOKENS, "block_tokens": BLOCK_TOKENS, "texts": len(texts), "tokenizers": {}}
    for name in arguments.tokenizer:
        report["tokenizers"][name] = measure(load_encoder(name), texts, arguments.limit, arguments.window)
    report["worst"] = max((entry["worst"] for entry in report["tokenizers"].values()), default=0)
    report["margin_holds"] = report["worst"] < MARGIN_TOKENS
    print(json.dumps(report, indent=2))
    return 0 if report["margin_holds"] else 1


if __name__ == "__main__":
    sys.exit(main())
