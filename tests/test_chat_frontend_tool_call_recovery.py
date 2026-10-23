#!/usr/bin/env python3
"""A tool call that does not parse is never shown to the client as text: the chat layer
regenerates it from where its envelope starts (prompt plus the tokens before it, the rest of
the token budget, a bumped seed) up to TOOL_CALL_RETRIES times, then fails with an explicit
tool_call_invalid error. Greedy and logprob requests are never regenerated. A call cut off by
the token budget ends the reply as "length" without the partial call. A call the model ended with a stop
token before its end marker is accepted as it is when its body parses and names every required parameter
of a declared tool, greedy or not. Every fault is counted."""

import asyncio
import importlib.util
import json
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
RequestError = chat_frontend.RequestError

O, C, S = "<|open|>", "<|close|>", "<|sep|>"
PIECES = ["Checking. ", O + "call", ' tool="read"' + S, O + 'argument key="path" type="string"' + S, "a.txt", C + "argument" + S,
          C + "call" + S, " broken", "<eos>", O + "message", C + "call"]
GOOD = [1, 2, 3, 4, 5, 6]
BAD = [1, 7, 6]
PROMPT = [900, 901, 902]


class Vocabulary:
    def decoder(self):
        return lambda token: PIECES[token]


class TaggedModel:
    id = "fake"
    reasoning = {"start": O + "think" + S, "end": C + "think" + S}
    tool_calls = {"format": "tagged", "start": O + "call", "end": C + "call" + S, "attribute_end": S,
                  "argument_start": O + "argument", "argument_end": C + "argument" + S,
                  "object_start": O + "json", "object_end": C + "json" + S}
    content_markers = []
    end_markers = []
    stop_token_ids = {8}
    vocabulary = Vocabulary()

    def __init__(self, engine):
        self.engine = engine

    def logprob(self, pair):
        return {"token": str(pair[0]), "logprob": pair[1]}


class Engine:
    def __init__(self):
        self.replies = []
        self.requests = []

    async def completions(self, request):
        body = await request.json()
        self.requests.append(body)
        tokens, finish = self.replies.pop(0)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for index, token in enumerate(tokens):
            last = index + 1 == len(tokens)
            event = {"tokens": [token], "choices": [{"finish_reason": finish if last else None}]}
            if body.get("logprobs") is not None:
                event["token_logprobs"] = [[[token, -0.25]]]
            if last:
                event["usage"] = {"prompt_tokens": len(body["prompt_token_ids"]), "completion_tokens": len(tokens),
                                  "prompt_tokens_details": {"cached_tokens": len(body["prompt_token_ids"]) - 1}}
            await response.write(("data: " + json.dumps(event) + "\n\n").encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response


async def collect(frontend, model, body):
    parser = chat_frontend.OutputParser(model, [{"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}], False, [])
    events = []
    try:
        async for event in frontend.generate(model, body, parser):
            events.append(event)
    except RequestError as error:
        return events, error
    return events, None


async def run(failures):
    def check(condition, name):
        if not condition:
            failures.append(name)

    engine = Engine()
    app = web.Application()
    app.router.add_post("/v1/completions", engine.completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    frontend = object.__new__(chat_frontend.Frontend)
    frontend.session = aiohttp.ClientSession()
    frontend.memory = None
    frontend.prefixes = None
    frontend.tool_call_faults = {}
    model = TaggedModel(f"http://127.0.0.1:{port}")
    model.gate = None
    sampled = {"prompt_token_ids": PROMPT, "max_tokens": 40, "temperature": 0.7, "seed": 11, "stream": True}
    try:
        engine.replies = [([0] + BAD, "stop"), (GOOD + [8], "stop")]
        events, error = await collect(frontend, model, dict(sampled))
        calls = [value for kind, value in events if kind == "tool_call"]
        content = "".join(value for kind, value in events if kind == "content")
        finish = [value for kind, value in events if kind == "finish"]
        check(error is None and len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt"},
              "a call that does not parse is regenerated and the good call reaches the client")
        check(content == "Checking. " and "broken" not in content, "the client sees the content before the call once and never the bad call")
        check(len(engine.requests) == 2 and engine.requests[1]["prompt_token_ids"] == PROMPT + [0],
              "the regeneration continues from the prompt plus the tokens before the call")
        check(engine.requests[1]["max_tokens"] == 39 and engine.requests[1]["seed"] == 12, "the regeneration gets the rest of the budget and a new seed")
        check(finish and finish[0]["finish_reason"] == "tool_calls" and finish[0]["token_ids"] == [0] + GOOD + [8],
              "the reply's tokens are the kept prefix plus the regenerated call")
        check(finish and finish[0]["usage"]["prompt_tokens"] == len(PROMPT) and finish[0]["usage"]["completion_tokens"] == 1 + len(GOOD) + 1 and
              finish[0]["usage"]["prompt_tokens_details"]["cached_tokens"] <= len(PROMPT),
              "usage counts the original prompt and every reply token")
        check(frontend.tool_call_faults.get("regenerated") == 1, "the regeneration is counted")

        engine.requests.clear()
        engine.replies = [(BAD, "stop"), (BAD, "stop"), (BAD, "stop")]
        events, error = await collect(frontend, model, dict(sampled))
        check(isinstance(error, RequestError) and error.status == 502 and error.code == "tool_call_invalid",
              "a call that keeps failing ends the reply with an explicit tool_call_invalid error")
        check(len(engine.requests) == chat_frontend.TOOL_CALL_RETRIES + 1 and frontend.tool_call_faults.get("refused") == 1,
              "the call is regenerated TOOL_CALL_RETRIES times, then refused")
        check(not any(kind == "content" and "broken" in value for kind, value in events), "a refused call never reaches the client as text")

        engine.requests.clear()
        engine.replies = [(BAD, "stop")]
        events, error = await collect(frontend, model, {"prompt_token_ids": PROMPT, "max_tokens": 40, "stream": True})
        check(isinstance(error, RequestError) and error.code == "tool_call_invalid" and len(engine.requests) == 1,
              "a greedy request is not regenerated, since it would produce the same call")
        engine.requests.clear()
        engine.replies = [(BAD, "stop")]
        events, error = await collect(frontend, model, dict(sampled, logprobs=0))
        check(isinstance(error, RequestError) and error.code == "tool_call_invalid" and len(engine.requests) == 1,
              "a logprob request is not regenerated, since its logprobs were already streamed")

        engine.requests.clear()
        engine.replies = [([0, 1, 2, 3, 4], "length")]
        events, error = await collect(frontend, model, dict(sampled, max_tokens=5))
        finish = [value for kind, value in events if kind == "finish"]
        content = "".join(value for kind, value in events if kind == "content")
        check(error is None and finish and finish[0]["finish_reason"] == "length" and content == "Checking. " and len(engine.requests) == 1,
              "a call cut off by the token budget ends the reply as length, without the partial call")
        check(frontend.tool_call_faults.get("cut_by_length") == 1, "the cut call is counted")

        engine.requests.clear()
        engine.replies = [([0, 1, 2, 8], "stop"), (GOOD, "stop")]
        events, error = await collect(frontend, model, dict(sampled))
        calls = [value for kind, value in events if kind == "tool_call"]
        check(error is None and len(calls) == 1 and len(engine.requests) == 2,
              "a call left open by the end of the sequence without its required arguments is regenerated")

        for name, tail in (("no end marker", []), ("half an end marker", [10])):
            engine.requests.clear()
            before = frontend.tool_call_faults.get("repaired", 0)
            engine.replies = [([0, 1, 2, 3, 4, 5] + tail + [8], "stop")]
            events, error = await collect(frontend, model, {"prompt_token_ids": PROMPT, "max_tokens": 40, "stream": True})
            calls = [value for kind, value in events if kind == "tool_call"]
            finish = [value for kind, value in events if kind == "finish"]
            check(error is None and len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt"} and len(engine.requests) == 1,
                  f"a greedy call ended by a stop token with {name} is accepted as it is, without regenerating")
            check(finish and finish[0]["finish_reason"] == "tool_calls" and frontend.tool_call_faults.get("repaired", 0) == before + 1,
                  f"the call with {name} ends the reply as tool_calls and is counted as repaired")

        engine.requests.clear()
        cut = frontend.tool_call_faults.get("cut_by_length", 0)
        engine.replies = [([0, 1, 2, 3, 4, 5], "length")]
        events, error = await collect(frontend, model, dict(sampled, max_tokens=6))
        finish = [value for kind, value in events if kind == "finish"]
        check(error is None and not [value for kind, value in events if kind == "tool_call"] and finish and finish[0]["finish_reason"] == "length" and
              frontend.tool_call_faults.get("cut_by_length", 0) == cut + 1,
              "a parseable call cut by the token budget is still cut, since the model may not have finished it")
    finally:
        await frontend.session.close()
        await runner.cleanup()


def main():
    failures = []
    asyncio.run(run(failures))
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS chat frontend tool call recovery: bad calls are regenerated from their envelope start, then refused with an explicit error; "
          "greedy and logprob requests are never regenerated; a call cut by the budget ends as length; a complete call missing only its end marker at a stop is accepted; every fault is counted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
