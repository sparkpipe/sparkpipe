#!/usr/bin/env python3
"""The chat layer's prefix ledger reports, for each request, the longest block-aligned token prefix an earlier engine request to the same model already carried (seen_tokens) and the length of that earlier sequence (seen_of), so a log reader can tell a client that changed its history (seen_of far above seen) from an engine that lost KV it had (cached_tokens below seen) and from a new conversation (seen near zero). Memory is bounded: the oldest blocks leave first."""
import asyncio
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

    def structure(self, messages, marks, tools, kwargs, pack=None, tools_kept=False):
        return None


frontend = chat_frontend.Frontend.__new__(chat_frontend.Frontend)
model = FakeModel()
body = {"messages": [{"role": "user", "content": "hi"}]}
first = asyncio.run(frontend.plan(model, body, None))
check(first["trace"]["seen_tokens"] == 0 and first["trace"]["seen_of"] == 0, "the first request's trace reports nothing seen")
frontend.prefixes["fake"] = PrefixLedger(block_tokens=1)
frontend.prefix_ledger(model).record(first["engine_body"]["prompt_token_ids"] + [9, 9])
second = asyncio.run(frontend.plan(model, body, None))
check(second["trace"]["seen_tokens"] == 3 and second["trace"]["seen_of"] == 5, "a repeated prompt's trace reports the earlier request's prefix and length")
check(all(name not in plan["engine_body"] for plan in (first, second) for name in ("cache_points", "cuts", "keep")),
      "a first prompt and a prompt an earlier request carried whole name no cut: what the ledger has seen never reaches the engine")
check(not hasattr(chat_frontend, "cache_points") and chat_frontend.structure_cuts(None, 9) == [] and
      chat_frontend.structure_cuts({"cuts": [12, 4, 4, 0, 9, 7]}, 9) == [4, 7] and chat_frontend.structure_cuts({"cuts": [7]}, 9, [3, 9]) == [3, 7] and
      chat_frontend.structure_cuts(None, 9, [3]) == [3],
      "cuts come from the prompt's opening and the turns it replays: rising, distinct, inside the prompt")
limit = chat_frontend.ENGINE_MAX_CUTS
long_body = {"prompt_token_ids": [0] * (3 * limit + 100)}
named, dropped = chat_frontend.name_cuts(model, long_body, {"cuts": list(range(2, 2 * limit + 40, 2))}, [1, 2 * limit + 51], [1, 2 * limit + 51])
check(named == long_body["cuts"] == [1] + list(range(2, 2 * limit, 2)) and len(named) == limit and dropped == 21 and long_body["keep"] == [1] and
      chat_frontend.CUTS_CAPPED[model.id] == 1,
      f"past the engine's limit the oldest cuts are kept and the newest are not named, so the first waves stay the same from turn to turn; the event is counted: {len(named)} {dropped}")


class ForkModel(FakeModel):
    def render(self, messages, tools, kwargs):
        text = "t1 t3 t5" if messages[-1]["content"] == "ho" else "t1 t3 t4"
        return text, self.vocabulary.encode(text, True)


fork = ForkModel()
frontend.prefixes["fake"] = PrefixLedger(block_tokens=1)
frontend.prefix_ledger(fork).record([1, 3, 4, 9, 9])
diverged = asyncio.run(frontend.plan(fork, {"messages": [{"role": "user", "content": "ho"}]}, None))
check(diverged["trace"]["seen_tokens"] == 2 and all(name not in diverged["engine_body"] for name in ("cache_points", "cuts", "keep")),
      f"a prompt that shares a prefix with an earlier one and then diverges reports what it has seen in its trace only, so earlier traffic never moves a cut: {diverged['engine_body'].keys()}")

captured = []


async def capture(model_, body_):
    captured.append(body_)
    yield {"tokens": [], "usage": {}}


frontend.gated_events = capture
background = asyncio.run(frontend.memory_complete(fork, [1, 3, 6, 7], 4))
check(captured and all(name not in captured[-1] for name in ("cache_points", "cuts", "keep")),
      f"a background completion that shares a prefix names no cut either: {sorted(captured[-1]) if captured else None}")
host = chat_frontend.MemoryHost(frontend)
frontend.models = {"fake": fork}
request = {"model": {"id": "fake"}, "ids": [1, 3, 6, 7, 8], "max_tokens": 4, "reasoning_open": False, "kind": "background", "trace": None}
asyncio.run(host.complete(dict(request, cuts=[2, 4], keep=[2])))
check(captured[-1].get("cuts") == [2, 4] and captured[-1].get("keep") == [2],
      f"a host completion that names where its shared view ends passes the cuts and the kept cut to the engine: {captured[-1].get('cuts')} {captured[-1].get('keep')}")
asyncio.run(host.complete(request))
check(all(name not in captured[-1] for name in ("cuts", "keep")), "the fields are optional: a host completion without them names none")
for wrong in ({"cuts": [4, 2]}, {"cuts": [0]}, {"cuts": [5]}, {"keep": [2, 2]}, {"cuts": "2"}, {"cuts": [True]}, {"keep": [1.5]}):
    try:
        asyncio.run(host.complete(dict(request, **wrong)))
        refused = None
    except chat_frontend.RequestError as error:
        refused = (error.status, error.code)
    check(refused == (400, "invalid_request"), f"cuts and keep must be rising token counts inside the prompt, as on the engine API: {wrong} gave {refused}")

for failure in failures:
    print("FAIL test_chat_frontend_prefix_ledger: " + failure)
if failures:
    sys.exit(1)
print("ok test_chat_frontend_prefix_ledger")
