#!/usr/bin/env python3
"""Derive the special tokens a publisher tokenizer puts in front of a raw text prompt.

Hugging Face encodes a text prompt with add_special_tokens on (vLLM does so for
/v1/completions), so the tokenizer's post-processor wraps the text. The ids it
places before the text are the deployment's text_prompt_prefix_token_ids.

Rules, in Hugging Face order:
  tokenizer_config.json add_bos_token false: no prefix.
  tokenizer_config.json add_bos_token true, or absent for a tokenizer_class
    whose transformers implementation defaults it to true (BOS_DEFAULT_CLASSES):
    the bos_token id, since transformers rebuilds the post-processor as
    "bos $A" and ignores the template in tokenizer.json.
  otherwise tokenizer.json post_processor: the leading special tokens of a
    TemplateProcessing "single" template, the cls token of BertProcessing or
    RobertaProcessing, the composition of a Sequence, nothing for ByteLevel or
    an absent post-processor.
  a tiktoken directory without tokenizer.json (TikTokenTokenizer): no prefix,
    its encode adds no special tokens.
Anything else is refused, so a new tokenizer shape fails loudly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BOS_DEFAULT_CLASSES = frozenset({
    "CodeLlamaTokenizer", "CodeLlamaTokenizerFast",
    "GemmaTokenizer", "GemmaTokenizerFast",
    "LlamaTokenizer", "LlamaTokenizerFast",
})


class TokenizerPrefixError(ValueError):
    pass


def token_content(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def token_id(tokenizer: dict, content: str) -> int:
    for added in tokenizer.get("added_tokens") or []:
        if added.get("content") == content:
            return int(added["id"])
    vocab = (tokenizer.get("model") or {}).get("vocab")
    if isinstance(vocab, dict) and content in vocab:
        return int(vocab[content])
    raise TokenizerPrefixError(f"token {content!r} is not in the tokenizer")


def processor_prefix(processor) -> list[int]:
    if processor is None:
        return []
    kind = processor.get("type")
    if kind == "Sequence":
        prefix: list[int] = []
        for inner in processor.get("processors") or []:
            prefix = processor_prefix(inner) + prefix
        return prefix
    if kind == "TemplateProcessing":
        prefix = []
        for piece in processor.get("single") or []:
            if "Sequence" in piece:
                return prefix
            name = piece["SpecialToken"]["id"]
            prefix += [int(value) for value in processor["special_tokens"][name]["ids"]]
        raise TokenizerPrefixError("TemplateProcessing single template has no $A sequence")
    if kind in ("BertProcessing", "RobertaProcessing"):
        return [int(processor["cls"][1])]
    if kind == "ByteLevel":
        return []
    raise TokenizerPrefixError(f"post_processor type {kind!r} is not understood")


def text_prompt_prefix_token_ids(tokenizer: dict, config: dict | None) -> list[int]:
    add_bos = (config or {}).get("add_bos_token")
    if add_bos is None and (config or {}).get("tokenizer_class") in BOS_DEFAULT_CLASSES:
        add_bos = True
    if add_bos is False:
        return []
    if add_bos is True:
        bos = token_content((config or {}).get("bos_token"))
        if bos is None:
            raise TokenizerPrefixError("add_bos_token is true but bos_token is unset")
        return [token_id(tokenizer, bos)]
    if add_bos is not None:
        raise TokenizerPrefixError(f"add_bos_token {add_bos!r} is not a boolean")
    return processor_prefix(tokenizer.get("post_processor"))


def directory_prefix(directory: Path) -> list[int]:
    config_path = directory / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else None
    tokenizer_path = directory / "tokenizer.json"
    if tokenizer_path.is_file():
        return text_prompt_prefix_token_ids(json.loads(tokenizer_path.read_text()), config)
    if (directory / "tiktoken.model").is_file() and (config or {}).get("tokenizer_class") == "TikTokenTokenizer":
        if (config or {}).get("add_bos_token") is True:
            raise TokenizerPrefixError("TikTokenTokenizer with add_bos_token true is not understood")
        return []
    raise TokenizerPrefixError(f"{directory} holds neither tokenizer.json nor a TikTokenTokenizer tiktoken.model")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory", type=Path, help="directory holding tokenizer.json and tokenizer_config.json")
    args = parser.parse_args()
    try:
        print(json.dumps(directory_prefix(args.directory)))
    except (TokenizerPrefixError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        print(f"tokenizer_text_prefix: {args.directory}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
