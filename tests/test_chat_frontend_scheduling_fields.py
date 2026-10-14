#!/usr/bin/env python3
"""A chat request's deadline_ms and priority reach the engine request, which cancels at the deadline and schedules by priority; malformed values are a 400 instead of being dropped."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
RequestError = chat_frontend.RequestError


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
        self.max_output_tokens = 4096
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


def engine_fields(extra):
    frontend = chat_frontend.Frontend.__new__(chat_frontend.Frontend)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    try:
        engine_body, _, _ = frontend.prepare(FakeModel(), body)
    except RequestError as error:
        return error.status
    return {key: engine_body[key] for key in ("deadline_ms", "priority") if key in engine_body}


def main():
    failures = []
    cases = [
        ("no scheduling fields", {}, {}),
        ("a deadline", {"deadline_ms": 30000}, {"deadline_ms": 30000}),
        ("a priority", {"priority": 3}, {"priority": 3}),
        ("both", {"deadline_ms": 1, "priority": 0}, {"deadline_ms": 1, "priority": 0}),
        ("a zero deadline", {"deadline_ms": 0}, 400),
        ("a fractional deadline", {"deadline_ms": 1.5}, 400),
        ("a negative priority", {"priority": -1}, 400),
        ("a string deadline", {"deadline_ms": "100"}, 400),
    ]
    for label, extra, expected in cases:
        got = engine_fields(extra)
        if got != expected:
            failures.append(f"{label}: got {got!r}, expected {expected!r}")
    for line in failures:
        print("FAIL " + line)
    if failures:
        return 1
    print("PASS test_chat_frontend_scheduling_fields: deadline_ms and priority reach the engine request, malformed values are a 400")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
