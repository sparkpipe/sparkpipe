#!/usr/bin/env python3
"""Tokenize the canonical Q3F-T1 prompts with the family tokenizer.json.

Pure-python byte-level BPE, faithful to the Qwen split regex for ASCII
inputs (the canonical prompt set is ASCII-only; the script fails closed on
non-ASCII rather than approximating a unicode class).
"""
import argparse
import json
import re
import sys

SPLIT_RE = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]{1,3}| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

BYTES_TO_UNICODE = {}
for _i in range(256):
    _b = chr(_i)
    if 33 <= _i <= 126 or 161 <= _i <= 172 or 174 <= _i <= 255:
        BYTES_TO_UNICODE[_b] = _b
    else:
        BYTES_TO_UNICODE[_b] = chr(0x100 + _i)
UNICODE_TO_BYTES = {v: k for k, v in BYTES_TO_UNICODE.items()}


def byte_encode(text):
    out = []
    for raw in text.encode("utf-8"):
        out.append(BYTES_TO_UNICODE[chr(raw)])
    return "".join(out)


def apply_bpe(word, ranks):
    parts = list(word)
    while len(parts) > 1:
        best = None
        best_rank = None
        for i in range(len(parts) - 1):
            pair = (parts[i], parts[i + 1])
            rank = ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best = i
        if best is None:
            break
        parts[best:best + 2] = [parts[best] + parts[best + 1]]
    return parts


def encode(text, model):
    vocab = model["vocab"]
    ranks = model["ranks"]
    ids = []
    for match in SPLIT_RE.finditer(text):
        word = byte_encode(match.group(0))
        for token in apply_bpe(word, ranks):
            if token not in vocab:
                raise ValueError(f"token {token!r} absent from vocab")
            ids.append(vocab[token])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", required=True, help="prompts json with 'text' fields")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    tokenizer = json.load(open(arguments.tokenizer))
    if tokenizer.get("model", {}).get("type") != "BPE":
        raise ValueError("only the BPE model is supported")
    pretokenizers = tokenizer.get("pre_tokenizer", {})
    if pretokenizers and "ByteLevel" not in json.dumps(pretokenizers):
        raise ValueError(f"unsupported pre_tokenizer {json.dumps(pretokenizers)[:120]}")
    model = {"vocab": tokenizer["model"]["vocab"], "ranks": {}}
    for rank, pair in enumerate(tokenizer["model"]["merges"]):
        parts = pair.split(" ") if isinstance(pair, str) else list(pair)
        model["ranks"][(parts[0], parts[1])] = rank
    document = json.load(open(arguments.prompts))
    for prompt in document["prompts"]:
        text = prompt["text"]
        if not text.isascii():
            raise ValueError(f"non-ASCII prompt {text!r} fails closed")
        ids = encode(text, model)
        prompt["prompt_token_ids"] = ids
        check = decode_ids(ids, model)
        if check != text:
            raise ValueError(f"round-trip mismatch: {check!r} != {text!r}")
        print(json.dumps({"text": text, "ids": ids}), flush=True)
    with open(arguments.output, "w") as out:
        json.dump(document, out, indent=1)
        out.write("\n")
    return 0


def decode_ids(ids, model):
    vocab = model["vocab"]
    inverse = {v: k for k, v in vocab.items()}
    raw = "".join(UNICODE_TO_BYTES[c] for token in ids for c in inverse[token])
    return raw.decode("utf-8", "replace")


if __name__ == "__main__":
    raise SystemExit(main())
