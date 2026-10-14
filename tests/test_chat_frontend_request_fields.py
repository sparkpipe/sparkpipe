#!/usr/bin/env python3
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
        self.reasoning_effort = {"high": {"thinking": True}}
        self.template_kwargs = {"thinking"}
        self.reasoning = {"start": "<think>", "end": "</think>"}
        self.channels = None
        self.tool_calls = {"format": "key_value", "start": "<tool_call>", "end": "</tool_call>", "key_start": "<arg_key>", "key_end": "</arg_key>",
                           "value_start": "<arg_value>", "value_end": "</arg_value>"}
        self.content_markers = []
        self.end_markers = []
        self.stop_token_ids = [2]

    def render(self, messages, tools, kwargs):
        return "t1 t3 t4", self.vocabulary.encode("t1 t3 t4", True)


TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}


def outcome(frontend, model, extra):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    try:
        frontend.prepare(model, body)
    except RequestError as error:
        return error.status
    except Exception as error:
        return f"{type(error).__name__}"
    return 200


def main():
    frontend = chat_frontend.Frontend.__new__(chat_frontend.Frontend)
    model = FakeModel()
    failures = []
    accepted = [{"tools": [TOOL]}, {"tools": [TOOL], "tool_choice": "auto"}, {"tools": [TOOL], "tool_choice": "none"}, {"stop": "t5"},
                {"stop": ["t5", "t6"]}, {"max_tokens": 16}, {"max_completion_tokens": 16}, {"chat_template_kwargs": {"thinking": False}},
                {"reasoning_effort": "high"}, {"stream": True}]
    refused = [{"tools": "lookup"}, {"tools": [{"type": "function"}]}, {"tools": [{"type": "retrieval"}]}, {"tool_choice": "required"},
               {"tool_choice": {"type": "function", "function": {"name": "lookup"}}}, {"max_tokens": "16"}, {"max_tokens": 1.5},
               {"max_completion_tokens": True}, {"max_tokens": 0}, {"stop": 5}, {"stop": ["t5", 6]}, {"stop": [""]}, {"chat_template_kwargs": ["thinking"]},
               {"reasoning_effort": ["high"]}, {"reasoning_effort": "extreme"}, {"stream": "yes"}]
    for extra in accepted:
        if outcome(frontend, model, extra) != 200:
            failures.append(f"{extra} is accepted (got {outcome(frontend, model, extra)})")
    for extra in refused:
        if outcome(frontend, model, extra) != 400:
            failures.append(f"{extra} is a 400 (got {outcome(frontend, model, extra)})")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print(f"PASS chat frontend request fields: {len(accepted)} well-typed requests are prepared; {len(refused)} malformed fields and unsupported tool_choice values are 400s, never a crash")


if __name__ == "__main__":
    main()
