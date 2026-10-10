#!/usr/bin/env python3
import asyncio
import importlib.util
import sys
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
        self.vocabulary = chat_frontend.TokenizersVocabulary.__new__(chat_frontend.TokenizersVocabulary)
        self.vocabulary.tokenizer = fake_tokenizer()
        self.reasoning_effort = {}
        self.template_kwargs = set()
        self.reasoning = {"start": "<think>", "end": "</think>"}
        self.tool_calls = {"format": "key_value", "start": "<tool_call>", "end": "</tool_call>", "key_start": "<arg_key>", "key_end": "</arg_key>",
                           "value_start": "<arg_value>", "value_end": "</arg_value>"}
        self.content_markers = []
        self.end_markers = []
        self.stop_token_ids = [2]

    def render(self, messages, tools, kwargs):
        return "t1 t3 t4", self.vocabulary.encode("t1 t3 t4", True)


def refused(body, status=400):
    try:
        chat_frontend.sampling_fields(body)
    except RequestError as error:
        return error.status == status
    return False


def main():
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    fields = chat_frontend.sampling_fields
    check(fields({}) == {}, "a request without sampling fields decodes greedily and forwards nothing")
    check(fields({"temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 3}) == {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 3},
          "temperature, top_p, top_k and seed are forwarded")
    check(fields({"temperature": 0, "top_p": 0.5, "top_k": 3, "seed": 9}) == {}, "greedy requests drop truncation and seed, which cannot change the argmax")
    check(fields({"temperature": 1, "top_k": -1}) == {"temperature": 1}, "top_k -1 means off")
    check(fields({"temperature": 1, "top_k": 0}) == {"temperature": 1}, "top_k 0 means off")
    check(fields({"logprobs": True, "top_logprobs": 5}) == {"logprobs": 5}, "top_logprobs becomes the engine's alternative count")
    check(fields({"logprobs": True}) == {"logprobs": 0}, "logprobs alone returns the emitted token's logprob")
    check(fields({"logprobs": False}) == {}, "logprobs false forwards nothing")
    for body, name in (({"top_logprobs": 2}, "top_logprobs without logprobs"), ({"logprobs": True, "top_logprobs": 21}, "more than twenty alternatives"),
                       ({"logprobs": "yes"}, "a non-boolean logprobs"), ({"temperature": 1, "top_p": 0}, "top_p zero"),
                       ({"temperature": 1, "top_p": 1.5}, "top_p above one"), ({"temperature": 3}, "temperature above two"),
                       ({"temperature": 0.00001}, "temperature below the minimum"), ({"temperature": 1, "top_k": 1.5}, "a fractional top_k"),
                       ({"temperature": 1, "seed": -1}, "a negative seed"), ({"temperature": True}, "a boolean temperature")):
        check(refused(body), name + " is a 400")

    frontend = chat_frontend.Frontend.__new__(chat_frontend.Frontend)
    model = FakeModel()
    messages = [{"role": "user", "content": "hi"}]
    engine_body, _, prompt_tokens = frontend.prepare(model, {"messages": messages, "temperature": 0.5, "top_p": 0.8, "logprobs": True, "top_logprobs": 2})
    check(engine_body["temperature"] == 0.5 and engine_body["top_p"] == 0.8 and engine_body["logprobs"] == 2 and prompt_tokens == 3 and engine_body["prompt_token_ids"] == [1, 3, 4],
          "prepare forwards the sampling fields to the engine")
    for field, value in (("frequency_penalty", 0.5), ("presence_penalty", -1), ("repetition_penalty", 1.1), ("min_p", 0.05), ("n", 2)):
        try:
            frontend.prepare(model, {"messages": messages, field: value})
            failures.append(f"{field}={value} was served although the engine cannot honour it")
        except RequestError as error:
            check(error.status == 400 and error.code == "unsupported_parameter", f"{field}={value} answers unsupported_parameter")
    engine_body, _, _ = frontend.prepare(model, {"messages": messages, "frequency_penalty": 0, "repetition_penalty": 1})
    check("temperature" not in engine_body, "neutral penalties are accepted and change nothing")

    chunk = {"tokens": [5, 2], "token_logprobs": [[[5, -0.25], [9, -1.5], [7, None]], [[2, -0.5], [9, -2.0], [7, -3.0]]]}
    entries = frontend.logprob_entries(model, chunk, 3)
    check(len(entries) == 1 and entries[0]["token"] == "t5" and entries[0]["logprob"] == -0.25 and entries[0]["bytes"] == list(b"t5"),
          "the emitted token maps to an OpenAI logprob entry and the stop token is left out")
    check([alt["token"] for alt in entries[0]["top_logprobs"]] == ["t9", "t7"] and entries[0]["top_logprobs"][1]["logprob"] == -9999.0,
          "alternatives keep their order and an underflowed logprob is -9999")
    for bad, name in (({"tokens": [5]}, "missing token_logprobs"), ({"tokens": [5], "token_logprobs": [[[6, -0.1], [9, -1.0], [7, -2.0]]]}, "a first entry that is not the token"),
                      ({"tokens": [5], "token_logprobs": [[[5, -0.1]]]}, "a short row")):
        try:
            frontend.logprob_entries(model, bad, 3)
            failures.append(name + " was accepted")
        except RequestError as error:
            check(error.status == 502, name + " is an engine error")

    async def events(model_, body):
        yield {"tokens": [5], "token_logprobs": [[[5, -0.25], [9, -1.5]]], "choices": [{"finish_reason": None}]}
        yield {"tokens": [6, 2], "token_logprobs": [[[6, -0.75], [9, -1.25]], [[2, -0.1], [9, -2.0]]], "choices": [{"finish_reason": "stop"}],
               "usage": {"completion_tokens": 3}}

    async def collect():
        frontend.engine_events = events
        parser = chat_frontend.OutputParser(model, None, False, [])
        return [item async for item in frontend.generate(model, {"logprobs": 1}, parser)]

    produced = asyncio.run(collect())
    kinds = [kind for kind, _ in produced]
    logprob_tokens = [entry["token"] for kind, value in produced if kind == "logprobs" for entry in value]
    check(kinds[0] == "logprobs" and logprob_tokens == ["t5", "t6"] and kinds[-1] == "finish", "generate yields each chunk's logprobs before its text")
    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print("PASS chat frontend sampling: fields validated and forwarded (greedy drops truncation), unsupported penalties refused, "
          "engine logprobs mapped to OpenAI entries in order with the stop token left out, malformed engine logprobs are 502")
    return 0


if __name__ == "__main__":
    sys.exit(main())
