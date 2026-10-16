#!/usr/bin/env python3
"""A chat request's reply budget never exceeds the model's output limit: with no max_tokens the engine gets the smaller of that limit and the room left in the context window, and an explicit max_tokens above the limit is a 400, so a long-context model never forwards a budget its engine refuses."""
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
    def __init__(self, context_tokens, max_output_tokens):
        self.id = "fake"
        self.context_tokens = context_tokens
        self.max_output_tokens = max_output_tokens
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


def budget(model, extra):
    frontend = chat_frontend.Frontend.__new__(chat_frontend.Frontend)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    try:
        engine_body, _, prompt_tokens = frontend.prepare(model, body)
    except RequestError as error:
        return f"{error.status}"
    return engine_body["max_tokens"]


def main():
    failures = []
    long_window = FakeModel(262144, 131072)
    short_window = FakeModel(4096, 131072)
    prompt = 3
    cases = [
        ("no max_tokens on a window longer than the output limit", long_window, {}, 131072),
        ("no max_tokens on a window shorter than the output limit", short_window, {}, 4096 - prompt),
        ("an explicit max_tokens under the limit", long_window, {"max_tokens": 2048}, 2048),
        ("an explicit max_completion_tokens under the limit", long_window, {"max_completion_tokens": 4096}, 4096),
        ("an explicit max_tokens at the limit", long_window, {"max_tokens": 131072}, 131072),
        ("an explicit max_tokens above the limit", long_window, {"max_tokens": 200000}, "400"),
    ]
    for label, model, extra, expected in cases:
        got = budget(model, extra)
        if got != expected:
            failures.append(f"{label}: got {got!r}, expected {expected!r}")
    spec = {"id": "fake", "model_dir": ".", "context_tokens": 1024, "engine": "http://127.0.0.1:1"}
    try:
        chat_frontend.Model(spec, ROOT)
        failures.append("a model config without max_output_tokens loaded")
    except KeyError as error:
        if "max_output_tokens" not in str(error):
            failures.append(f"a model config without max_output_tokens failed on another field: {error}")
    except Exception as error:
        failures.append(f"a model config without max_output_tokens got past the field check ({type(error).__name__})")
    for line in failures:
        print("FAIL " + line)
    if failures:
        return 1
    print("PASS test_chat_frontend_output_cap: the default reply budget is the smaller of the output limit and the room left; an explicit budget above the limit is a 400")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
