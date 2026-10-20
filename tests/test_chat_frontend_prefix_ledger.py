#!/usr/bin/env python3
"""The chat layer's prefix ledger reports, for each request, the longest block-aligned token prefix an earlier engine request to the same model already carried (seen_tokens) and the length of that earlier sequence (seen_of), so a log reader can tell a client that changed its history (seen_of far above seen) from an engine that lost KV it had (cached_tokens below seen) and from a new conversation (seen near zero). Memory is bounded: the oldest blocks leave first."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
PrefixLedger = chat_frontend.PrefixLedger

failures = []


def check(condition, what):
    if not condition:
        failures.append(what)


history = list(range(1000, 1200))
ledger = PrefixLedger(block_tokens=64)
check(ledger.seen(history) == (0, 0), "a prompt nothing carried before has seen 0")
ledger.record(history)
check(ledger.seen(history + [7, 8, 9]) == (192, 200), "a continuation sees every whole block of the earlier sequence and its length")
check(ledger.seen(history[:70] + [5] * 130) == (64, 200), "a history changed at token 70 sees only the block before the change")
check(ledger.seen([5] + history[1:]) == (0, 0), "a history changed at its first token sees nothing")
check(ledger.seen(history[:63]) == (0, 0), "a prompt shorter than a block sees nothing")
longer = history + list(range(5000, 5300))
ledger.record(longer)
check(ledger.seen(history[:128]) == (128, 500), "seen_of is the longest earlier sequence through the matched prefix")

small = PrefixLedger(block_tokens=4, capacity_blocks=3)
small.record(list(range(40)))
check(len(small.blocks) == 3, "the ledger keeps at most its capacity of blocks")
check(small.seen(list(range(40))) == (0, 0), "the oldest blocks leave first, so the start of an evicted sequence is not seen")


def fake_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(WordLevel({"t%d" % index: index for index in range(32)}, unk_token="t0"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


class FakeModel(chat_frontend.Model):
    def __init__(self):
        self.id = "fake"
        self.context_tokens = 4096
        self.max_output_tokens = 1024
        self.sampling_defaults = {}
        self.vocabulary = chat_frontend.TokenizersVocabulary.from_tokenizer(fake_tokenizer())
        self.reasoning_effort = {}
        self.template_kwargs = set()
        self.reasoning = {"start": "<think>", "end": "</think>"}
        self.channels = None
        self.tool_calls = {"format": "key_value", "start": "<tool_call>", "end": "</tool_call>", "key_start": "<arg_key>", "key_end": "</arg_key>",
                           "value_start": "<arg_value>", "value_end": "</arg_value>"}
        self.content_markers = []
        self.end_markers = []
        self.stop_token_ids = [2]

    def render(self, messages, tools, kwargs):
        return "t1 t3 t4", self.vocabulary.encode("t1 t3 t4", True)


frontend = chat_frontend.Frontend.__new__(chat_frontend.Frontend)
model = FakeModel()
body = {"messages": [{"role": "user", "content": "hi"}]}
first = frontend.plan(model, body, None)
check(first["trace"]["seen_tokens"] == 0 and first["trace"]["seen_of"] == 0, "the first request's trace reports nothing seen")
frontend.prefixes["fake"] = PrefixLedger(block_tokens=1)
frontend.prefix_ledger(model).record(first["engine_body"]["prompt_token_ids"] + [9, 9])
second = frontend.plan(model, body, None)
check(second["trace"]["seen_tokens"] == 3 and second["trace"]["seen_of"] == 5, "a repeated prompt's trace reports the earlier request's prefix and length")

for failure in failures:
    print("FAIL test_chat_frontend_prefix_ledger: " + failure)
if failures:
    sys.exit(1)
print("ok test_chat_frontend_prefix_ledger")
