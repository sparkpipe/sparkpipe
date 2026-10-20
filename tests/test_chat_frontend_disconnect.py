#!/usr/bin/env python3
import asyncio
import contextlib
import importlib.util
import io
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
QUEUED = 150


class Engine:
    def __init__(self):
        self.open = 0
        self.peak = 0
        self.closed = 0

    async def completions(self, request):
        self.open += 1
        self.peak = max(self.peak, self.open)
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.closed += 1
            raise
        finally:
            self.open -= 1
        return web.Response(status=500)


class Vocabulary:
    def decoder(self):
        return lambda token: ""


class Model:
    def __init__(self, engine):
        self.id = "queued"
        self.engine = engine
        self.vocabulary = Vocabulary()
        self.stop_token_ids = []


async def serve(app, **options):
    runner = web.AppRunner(app, **options)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


async def until(condition, seconds):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while not condition() and loop.time() < deadline:
        await asyncio.sleep(0.02)
    return condition()


async def run(tmp, failures):
    engine = Engine()
    upstream = web.Application()
    upstream.router.add_post("/v1/completions", engine.completions)
    engine_runner, engine_base = await serve(upstream, handler_cancellation=True)
    frontend = chat_frontend.Frontend({"models": []}, tmp)
    model = Model(engine_base)
    frontend.model_for = lambda body: model
    frontend.plan = lambda model, body, headers: {"engine_body": {"prompt_token_ids": [1], "max_tokens": 4, "stream": True},
                                                  "parser": None, "prompt_tokens": 1, "session": None, "body": body}
    front_runner, front = await serve(chat_frontend.build(frontend), **chat_frontend.SERVER_OPTIONS)
    captured = io.StringIO()
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=None)) as client:
            async def ask(stream):
                async with client.post(front + "/v1/chat/completions", json={"model": "queued", "stream": stream, "messages": []}) as response:
                    await response.read()

            with contextlib.redirect_stderr(captured):
                waiting = [asyncio.create_task(ask(True)) for _ in range(QUEUED)]
                if not await until(lambda: engine.open == QUEUED, 20):
                    failures.append(f"{QUEUED} queued streaming requests should all wait at the engine at once; it saw {engine.open} (peak {engine.peak})")
                for task in waiting:
                    task.cancel()
                await asyncio.gather(*waiting, return_exceptions=True)
                if not await until(lambda: engine.open == 0 and engine.closed == QUEUED, 10):
                    failures.append(f"every client that disconnects while queued should close its engine request; {engine.open} stay open, {engine.closed} closed")
                single = asyncio.create_task(ask(False))
                if not await until(lambda: engine.open == 1, 10):
                    failures.append("a queued non-streaming request should reach the engine")
                single.cancel()
                await asyncio.gather(single, return_exceptions=True)
                if not await until(lambda: engine.open == 0 and engine.closed == QUEUED + 1, 10):
                    failures.append("a non-streaming client that disconnects while queued should close its engine request")
                await asyncio.sleep(0.2)
        gone = captured.getvalue().count('"finish_reason": "client_gone"')
        if gone != QUEUED + 1:
            failures.append(f"each disconnected request should log one client_gone line; logged {gone}")
    finally:
        await front_runner.cleanup()
        await engine_runner.cleanup()


def main():
    failures = []
    soft = chat_frontend.raise_open_files()
    if soft < 4 * QUEUED + 64:
        failures.append(f"the chat layer should raise its open-files limit well past {4 * QUEUED} descriptors; it got {soft}")
    probe = subprocess.run([sys.executable, "-c", (
        "import importlib.util, resource, sys\n"
        f"spec = importlib.util.spec_from_file_location('chat_frontend', {str(ROOT / 'tools' / 'serving' / 'chat_frontend.py')!r})\n"
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))\n"
        "raised = module.raise_open_files()\n"
        "print(raised, hard, resource.RLIM_INFINITY)\n")], capture_output=True, text=True)
    raised, hard, infinity = (int(field) for field in probe.stdout.split())
    if not (raised == hard or (hard == infinity and raised > 256)):
        failures.append(f"a 256-descriptor soft limit should rise to the hard limit {hard}; it became {raised}")
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp), failures))
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print(f"PASS {QUEUED} queued requests all wait at the engine with no connection cap, a client that disconnects while queued (streaming or not) closes its engine request and logs client_gone, and the chat layer raises its open-files limit to the hard limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
