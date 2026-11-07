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


class Fill:
    def __init__(self, status, record):
        self.status, self.record, self.bodies = status, record, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.bodies.append((url, json))
        outer = self

        class Reply:
            status = outer.status

            async def __aenter__(self):
                return self

            async def __aexit__(self, *arguments):
                return False

            async def json(self, content_type=None):
                return outer.record

        return Reply()


def harness(record):
    server = Frontend.__new__(Frontend)
    server.session = Fill(200 if record.get("ok") else 502, record)
    server.control = {"fill_url": "http://controller/fill"}
    server.controller_headers = {}
    server.memory = None
    bodies = []

    async def engine_events(model, engine_body):
        bodies.append(dict(engine_body))
        if engine_body.get("long_fill_check"):
            raise RequestError(409, "long_fill", "needs the fill")
        yield {"tokens": [1]}

    server.engine_events = engine_events
    return server, bodies


async def run(server):
    model = types.SimpleNamespace(id="kimi-k3", gate=None)
    return [chunk async for chunk in server.engine_stream(model, {"prompt_token_ids": [1, 2, 3], "long_fill_check": True}, False)]


def main():
    server, bodies = harness({"ok": True})
    chunks = asyncio.run(run(server))
    assert chunks == [{"tokens": [1]}], chunks
    assert server.session.bodies == [("http://controller/fill", {"model": "kimi-k3", "prompt_token_ids": [1, 2, 3]})], server.session.bodies
    assert [body.get("long_fill_check") for body in bodies] == [True, None], bodies
    server, bodies = harness({"ok": False, "error": "streamed fill answered 500"})
    try:
        asyncio.run(run(server))
    except RequestError as error:
        assert error.code == "long_fill_failed" and "streamed fill answered 500" in str(error.message), error.message
    else:
        raise AssertionError("a failed fill must fail the request, not fall back to waves")
    assert len(bodies) == 1, bodies
    config = json.loads((ROOT / "deployment" / "hub" / "chat_frontend.json").read_text())
    assert [model["id"] for model in config["models"] if model.get("long_fill")] == ["kimi-k3"] and config["control"]["fill_url"].endswith("/fill"), config["control"]
    print("PASS test_chat_frontend_long_fill: a refused prompt runs the fill and retries once; a failed fill fails the request")
    return 0


if __name__ == "__main__":
    sys.exit(main())
