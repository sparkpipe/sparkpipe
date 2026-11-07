#!/usr/bin/env python3
import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
sys.modules["chat_frontend"] = chat_frontend
SPEC.loader.exec_module(chat_frontend)
RequestError = chat_frontend.RequestError
Frontend = next(value for value in vars(chat_frontend).values() if isinstance(value, type) and "engine_stream" in vars(value))


REFUSED = b'data: {"error":{"message":"this prompt\'s uncached tokens need the streamed fill","type":"invalid_request_error","code":"long_fill"}}\n\n'
TOKENS = b'data: {"tokens":[1]}\n\n'
DONE = b"data: [DONE]\n\n"


class Reply:
    def __init__(self, status, record=None, lines=()):
        self.status, self.record, self.lines = status, record, lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *arguments):
        return False

    async def json(self, content_type=None):
        return self.record

    @property
    def content(self):
        async def lines():
            for line in self.lines:
                yield line
        return lines()


class Session:
    def __init__(self, fill_status, record, refusals):
        self.fill_status, self.record, self.refusals, self.fills, self.engine = fill_status, record, refusals, [], []

    def post(self, url, headers=None, json=None, timeout=None):
        if url.endswith("/fill"):
            self.fills.append((url, json))
            return Reply(self.fill_status, self.record)
        self.engine.append(dict(json))
        self.refusals -= 1
        return Reply(200, lines=[REFUSED, DONE] if self.refusals >= 0 and json.get("long_fill_check") else [TOKENS, DONE])


def harness(record, refusals=1):
    server = Frontend.__new__(Frontend)
    server.session = Session(200 if record.get("ok") else 502, record, refusals)
    server.control = {"fill_url": "http://controller/fill"}
    server.controller_headers = {}
    server.memory = None
    return server


async def run(server):
    model = types.SimpleNamespace(id="kimi-k3", gate=None, engine="http://engine")
    return [chunk async for chunk in server.engine_stream(model, {"prompt_token_ids": [1, 2, 3], "long_fill_check": True}, False)]


def main():
    server = harness({"ok": True})
    chunks = asyncio.run(run(server))
    assert chunks == [{"tokens": [1]}], chunks
    assert server.session.fills == [("http://controller/fill", {"model": "kimi-k3", "prompt_token_ids": [1, 2, 3]})], server.session.fills
    assert [body.get("long_fill_check") for body in server.session.engine] == [True, True], server.session.engine
    server = harness({"ok": True}, refusals=2)
    try:
        asyncio.run(run(server))
    except RequestError as error:
        assert error.code == "long_fill_failed" and "cannot resume" in str(error.message), error.message
    else:
        raise AssertionError("a fill the engine cannot resume from must fail the request, not run the prompt as waves")
    assert len(server.session.fills) == 1 and len(server.session.engine) == 2, (server.session.fills, server.session.engine)
    server = harness({"ok": False, "error": "streamed fill answered 500"})
    try:
        asyncio.run(run(server))
    except RequestError as error:
        assert error.code == "long_fill_failed" and "streamed fill answered 500" in str(error.message), error.message
    else:
        raise AssertionError("a failed fill must fail the request, not fall back to waves")
    assert len(server.session.engine) == 1, server.session.engine
    config = json.loads((ROOT / "deployment" / "hub" / "chat_frontend.json").read_text())
    assert [model["id"] for model in config["models"] if model.get("long_fill")] == ["kimi-k3"] and config["control"]["fill_url"].endswith("/fill"), config["control"]
    print("PASS test_chat_frontend_long_fill: a prompt the engine refuses inside its 200 stream runs the fill and retries once; a failed fill or a fill the engine cannot resume from fails the request")
    return 0


if __name__ == "__main__":
    sys.exit(main())
