#!/usr/bin/env python3
import asyncio
import contextlib
import importlib.util
import io
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
RequestError = chat_frontend.RequestError
EVENT = b'data: {"tokens":[5],"text":"x"}\n\n'


class Engine:
    def __init__(self):
        self.mode = "complete"

    async def completions(self, request):
        if self.mode == "string_error":
            return web.json_response({"error": "worker 3 at 10.0.0.7 crashed"}, status=500)
        if self.mode == "client_error":
            return web.json_response({"error": {"code": "invalid_request", "message": "max_tokens must be positive"}}, status=400)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if self.mode == "garbage":
            await response.write(b"data: {not json\n\n")
        else:
            await response.write(EVENT)
        if self.mode != "truncated":
            await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response


class Model:
    def __init__(self, base):
        self.id = "fake"
        self.engine = base


async def collect(frontend, model):
    try:
        return [event async for event in frontend.engine_events(model, {"prompt_token_ids": [1], "max_tokens": 1, "stream": True})]
    except RequestError as error:
        return error


async def run(failures):
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
    model = Model(f"http://127.0.0.1:{port}")
    try:
        result = await collect(frontend, model)
        if not (isinstance(result, list) and result == [{"tokens": [5], "text": "x"}]):
            failures.append("a stream that ends with [DONE] yields its events")
        engine.mode = "truncated"
        result = await collect(frontend, model)
        if not (isinstance(result, RequestError) and result.status == 502 and result.code == "engine_error"):
            failures.append("a stream that closes before [DONE] is an engine error, not a complete reply")
        engine.mode = "garbage"
        result = await collect(frontend, model)
        if not (isinstance(result, RequestError) and result.status == 502 and result.code == "engine_error"):
            failures.append("a stream event that is not JSON is an engine error, not an uncaught exception")
        engine.mode = "string_error"
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            result = await collect(frontend, model)
        if not (isinstance(result, RequestError) and result.status == 500 and result.code == "engine_error" and "10.0.0.7" not in result.message
                and "10.0.0.7" in log.getvalue()):
            failures.append(f"an engine 500 whose error is a string gives a fixed message, and the detail goes only to the log: {result!r}")
        engine.mode = "client_error"
        with contextlib.redirect_stderr(io.StringIO()):
            result = await collect(frontend, model)
        if not (isinstance(result, RequestError) and result.status == 400 and result.code == "invalid_request" and result.message == "max_tokens must be positive"):
            failures.append(f"an engine 400 keeps its code and message, which describe the caller's own request: {result!r}")
        down = Model("http://127.0.0.1:9")
        with contextlib.redirect_stderr(io.StringIO()):
            result = await collect(frontend, down)
        if not (isinstance(result, RequestError) and result.status == 503 and result.code == "engine_unavailable" and "127.0.0.1" not in result.message):
            failures.append(f"an unreachable engine answers 503 without its address: {result!r}")
    finally:
        await frontend.session.close()
        await runner.cleanup()


def main():
    failures = []
    asyncio.run(run(failures))
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS chat frontend engine stream: [DONE] completes; a stream cut short or a non-JSON event becomes a 502 engine error; engine failures reach the caller as fixed messages without addresses or engine text, which go to the log")


if __name__ == "__main__":
    main()
