#!/usr/bin/env python3
import asyncio
import importlib.util
import json
import time
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
RequestError = chat_frontend.RequestError


class FakeEngine:
    def __init__(self):
        self.ready_at = None
        self.failed = False
        self.activations = 0
        self.completions = 0

    def ready(self):
        return self.ready_at is not None and time.monotonic() >= self.ready_at

    async def health(self, request):
        return web.json_response({"status": "ok"} if self.ready() else {"status": "starting"}, status=200 if self.ready() else 503)

    async def completions_handler(self, request):
        if self.failed:
            return web.json_response({"error": {"message": "latched", "code": "engine_failed"}}, status=503)
        if not self.ready():
            return web.json_response({"status": "starting", "phase": "connect"}, status=503)
        self.completions += 1
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(("data: " + json.dumps({"tokens": [5], "choices": [{"finish_reason": "stop"}]}) + "\n\n").encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    async def activate(self, request):
        self.activations += 1
        if self.ready_at is None:
            self.ready_at = time.monotonic() + 0.6
        return web.json_response({"ok": True})


class FakeModel:
    def __init__(self, engine_url, warm):
        self.id = "fake"
        self.engine = engine_url
        self.gate = chat_frontend.WarmGate(self.id, engine_url, warm)


async def collect(frontend, model):
    chunks = []
    async for chunk in frontend.engine_stream(model, {"prompt_token_ids": [1], "max_tokens": 1, "stream": True}):
        chunks.append(chunk)
    return chunks


async def outcome(frontend, model):
    try:
        return await collect(frontend, model)
    except RequestError as error:
        return error


async def run(failures):
    def check(condition, name):
        if not condition:
            failures.append(name)

    chat_frontend.WarmGate.POLL_SECONDS = 0.05
    engine = FakeEngine()
    app = web.Application()
    app.router.add_get("/health", engine.health)
    app.router.add_post("/v1/completions", engine.completions_handler)
    app.router.add_post("/activate", engine.activate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    frontend = object.__new__(chat_frontend.Frontend)
    frontend.session = aiohttp.ClientSession()
    try:
        model = FakeModel(base, {"depth": 4, "wait_seconds": 10, "activation_url": base + "/activate"})
        results = await asyncio.gather(*(outcome(frontend, model) for _ in range(3)))
        check(all(isinstance(result, list) and result and result[0]["tokens"] == [5] for result in results),
              "requests that arrive during bringup complete once the engine is ready")
        check(engine.activations == 1, "the cold engine is activated exactly once for concurrent waiters")
        check(model.gate.released == 3 and model.gate.waiting == 0 and engine.completions == 3, "every waiter is released and served")
        result = await outcome(frontend, model)
        check(isinstance(result, list) and engine.activations == 1, "a ready engine is served without waiting or activating")

        engine.ready_at = time.monotonic() + 3600
        small = FakeModel(base, {"depth": 2, "wait_seconds": 0.5, "activation_url": None})
        results = await asyncio.gather(*(outcome(frontend, small) for _ in range(3)))
        full = [result for result in results if isinstance(result, RequestError) and result.code == "warm_queue_full"]
        expired = [result for result in results if isinstance(result, RequestError) and result.code == "warm_queue_expired"]
        check(len(full) == 1 and full[0].status == 503, "a request past the queue depth is refused with 503")
        check(len(expired) == 2 and all(result.status == 503 for result in expired), "requests that wait past wait_seconds are refused with 503")
        check(small.gate.overflowed == 1 and small.gate.expired == 2 and small.gate.waiting == 0, "the gate counts overflow and expiry")
        check(engine.activations == 1, "a gate without an activation URL does not activate")

        engine.ready_at = time.monotonic() - 1
        engine.failed = True
        started = time.monotonic()
        result = await outcome(frontend, small)
        check(isinstance(result, RequestError) and result.code == "engine_failed" and time.monotonic() - started < 0.4,
              "a latched engine failure is returned at once, not queued")
        engine.failed = False

        for bad in (None, {"depth": 0, "wait_seconds": 1, "activation_url": None}, {"depth": 1, "wait_seconds": 1}, {"depth": 1, "wait_seconds": 1, "activation_url": None, "extra": 1}):
            try:
                chat_frontend.WarmGate("bad", base, bad)
                check(False, f"warm_queue {bad!r} is refused")
            except SystemExit:
                pass
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
    print("PASS chat frontend warm queue: bringup waiters drain after readiness with one activation; overflow and expiry answer 503; latched failures are not queued")


if __name__ == "__main__":
    main()
