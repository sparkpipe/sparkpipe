#!/usr/bin/env python3
"""The chat layer reaches memory only through a generic connector to a separate program on localhost: while the program is down the connector logs that memory is unavailable and fails calls at once until its retry time, then says hello and reaches it; the program reaches the chat layer's host services with the same bearer token, and the host refuses a missing token, an unknown method and a body that is not a JSON object, and answers a host failure with its own status; a program error comes back with its status and code and leaves the connector ready, while a timeout marks memory unavailable; when the program gives up on a slow host call, the host stops that work; notices reach the program in the order they were sent; a session handle's calls reach the program with the handle; after the program restarts, the connector says hello again and retries; a program that speaks another protocol or refuses the token counts as unavailable; with the program down, the chat layer serves a chat request raw and logs it."""
import asyncio
import contextlib
import importlib.util
import io
import json
import socket
import tempfile
import time
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("memory_connector", ROOT / "tools" / "serving" / "memory_connector.py")
mc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mc)
TOKEN = "k" * 48


class Failure(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


mc.RequestError = Failure


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeProgram:
    def __init__(self, port, protocol=mc.PROTOCOL, token=TOKEN):
        self.port, self.protocol, self.token = port, protocol, token
        self.greeted = False
        self.notices = []

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/hello", self.hello)
        app.router.add_post("/v1/{method}", self.call)
        self.runner = web.AppRunner(app, handler_cancellation=True)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()
        self.client = aiohttp.ClientSession()
        return self

    async def stop(self):
        await self.client.close()
        await self.runner.cleanup()

    async def hello(self, request):
        if request.headers.get("Authorization") != "Bearer " + self.token:
            return web.json_response({"error": {"code": "unauthorized", "message": "the token is wrong"}}, status=401)
        self.host_url = (await request.json())["host_url"]
        self.greeted = True
        return web.json_response({"protocol": self.protocol, "methods": ["prepare"]})

    async def host(self, method, body, timeout=10.0):
        async with self.client.post(f"{self.host_url}/v1/{method}", json=body, headers={"Authorization": "Bearer " + TOKEN},
                                    timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            return await response.json()

    async def call(self, request):
        method = request.match_info["method"]
        if not self.greeted:
            return web.json_response({"error": {"code": "hello_first", "message": "say hello"}}, status=409)
        body = await request.json()
        if method == "prepare":
            return web.json_response({"tokens": (await self.host("count", {"model": body["model"], "text": "a b c"}))["tokens"]})
        if method == "fail":
            return web.json_response({"error": {"code": "engine_unavailable", "message": "the engine is switching", "status": 503}}, status=503)
        if method == "sleep":
            await asyncio.sleep(2)
        if method == "abandon":
            with contextlib.suppress(asyncio.TimeoutError):
                await self.host("complete", {"model": body["model"], "ids": [1], "max_tokens": 4}, 0.3)
        if method in ("foreground", "session.release"):
            self.notices.append((method, body))
        answers = {"session.claims": {"claimed": body.get("call", {}).get("name") == "expand"}, "session.can_continue": {"can": True},
                   "session.continue_with": {"messages": [{"role": "tool", "content": body.get("content")}]}}
        return web.json_response(answers.get(method, {"method": method, "body": body}))


class FakeHost:
    def __init__(self):
        self.cancelled = asyncio.Event()

    def count(self, request):
        return {"tokens": len(request["text"].split())}

    def render_ids(self, request):
        raise Failure(503, "engine_starting", "the model is starting")

    async def complete(self, request):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return {"text": "late"}


async def refused(call):
    try:
        await call
    except Failure as error:
        return error
    raise AssertionError("the call was not refused")


async def run(directory, check):
    (directory / "memory.token").write_text(TOKEN)
    port, host_port = free_port(), free_port()
    host = FakeHost()
    connector = mc.Connector({"url": f"http://127.0.0.1:{port}", "host_listen": f"127.0.0.1:{host_port}", "token_file": "memory.token"}, directory, host,
                             lambda path: Path(path).read_text().strip())
    logged = io.StringIO()
    with contextlib.redirect_stderr(logged):
        await connector.start()
        started = time.monotonic()
        error = await refused(connector.call("prepare", {"model": {"id": "m"}}))
        check(not connector.ready and '"event": "memory_unavailable"' in logged.getvalue(), f"a program that is down is logged as unavailable: {logged.getvalue()[:300]}")
        check(error.code == mc.UNAVAILABLE and time.monotonic() - started < 0.1, "inside the retry window a call fails at once, without trying the program")

        program = await FakeProgram(port).start()
        connector.retry_at = 0.0
        answer = await connector.call("prepare", {"model": {"id": "m"}})
        check(connector.ready and answer == {"tokens": 3}, f"after the retry window the connector says hello, and the program reaches the host's count: {answer}")

        base, authorized = f"http://127.0.0.1:{host_port}/v1/", {"Authorization": "Bearer " + TOKEN}
        async with aiohttp.ClientSession() as raw:
            statuses = []
            for method, data, headers in (("count", b'{"text": "a"}', {}), ("shutdown", b"{}", authorized), ("count", b"[1]", authorized),
                                          ("render_ids", b"{}", authorized)):
                async with raw.post(base + method, data=data, headers=headers) as response:
                    statuses.append((response.status, (await response.json())["error"]["code"]))
        check(statuses == [(401, "unauthorized"), (404, "unsupported"), (400, "invalid_request"), (503, "engine_starting")],
              f"the host refuses a missing token, an unknown method and a non-object body, and answers its own failure with its status: {statuses}")

        error = await refused(connector.call("fail", {}))
        check(error.status == 503 and error.code == "engine_unavailable" and connector.ready, f"a program error keeps its status and code: {vars(error)}")
        error = await refused(connector.call("sleep", {}, 0.2))
        check(error.code == mc.UNAVAILABLE and not connector.ready, "a call that times out marks memory unavailable")
        connector.retry_at = 0.0

        await connector.call("abandon", {"model": {"id": "m"}})
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(host.cancelled.wait(), 5)
        check(host.cancelled.is_set(), "a host call the program gives up on stops the host's work")

        for requests in (1, 2, 3, 2, 1, 0):
            connector.notify("foreground", {"model": {"id": "m"}, "requests": requests})
        session = mc.Session(connector, "s7")
        claimed = (await session.claims({"name": "expand"}), await session.claims({"name": "read"}))
        messages = await session.continue_with("found", "", [{"name": "expand"}]) if await session.can_continue() else None
        finished = await session.finished([], "done", "", [], "stop", (1, 2), (3,))
        session.release()
        for _ in range(100):
            if len(program.notices) == 7:
                break
            await asyncio.sleep(0.02)
        check([body.get("requests", body.get("session")) for _, body in program.notices] == [1, 2, 3, 2, 1, 0, "s7"],
              f"notices reach the program in the order they were sent: {program.notices}")
        check(claimed == (True, False) and messages == [{"role": "tool", "content": "found"}] and session.rounds == 1 and finished is None,
              f"a session's calls reach the program with its handle: {claimed} {messages}")

        await program.stop()
        program = await FakeProgram(port).start()
        answer = await connector.call("prepare", {"model": {"id": "m"}})
        check(answer == {"tokens": 3} and program.greeted, "after the program restarts, the connector says hello again and retries the call")

        await program.stop()
        program = await FakeProgram(port, protocol=mc.PROTOCOL + 1).start()
        error = await refused(connector.call("prepare", {"model": {"id": "m"}}))
        check(error.code == mc.UNAVAILABLE and not connector.ready, "a program that speaks another protocol is unavailable")
        await program.stop()
        program = await FakeProgram(port, token="x" * 48).start()
        connector.retry_at = 0.0
        error = await refused(connector.call("prepare", {"model": {"id": "m"}}))
        check(error.code == mc.UNAVAILABLE and not connector.ready, "a program that refuses the token is unavailable")
        await program.stop()
        await connector.stop()


async def fallback(directory, check):
    spec = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    token = directory / "memory.token"
    token.write_text(TOKEN)
    token.chmod(0o600)
    settings = {"url": f"http://127.0.0.1:{free_port()}", "host_listen": f"127.0.0.1:{free_port()}", "token_file": "memory.token"}
    logged = io.StringIO()
    with contextlib.redirect_stderr(logged):
        async with fixture.Harness(directory, extra={"memory": settings}) as harness:
            body = {"model": "alpha", "messages": [{"role": "user", "content": "hi"}]}
            harness.engine.reply("Thinking.</think>Served raw.")
            status, _, text = await harness.post("/v1/chat/completions", body, **{"X-OpenWebUI-Chat-Id": "chat-1", "X-OpenWebUI-User-Message-Id": "message-1"})
            served = harness.engine.requests[-1]["prompt_token_ids"]
            raw = harness.frontend.prepare(harness.frontend.models["alpha"], body)[0]["prompt_token_ids"]
    answer = json.loads(text)["choices"][0]["message"]["content"] if status == 200 else None
    check(status == 200 and answer == "Served raw." and list(served) == list(raw),
          f"with the memory program down a chat request is served on exactly its raw prompt: {status} {answer!r} {len(served)} {len(raw)}")
    check('"event": "memory_unavailable"' in logged.getvalue(), "and the chat layer logs that memory is unavailable")


def main():
    failures = []
    with tempfile.TemporaryDirectory(prefix="memory-connector-") as directory:
        asyncio.run(run(Path(directory), lambda condition, message: failures.append(message) if not condition else None))
    with tempfile.TemporaryDirectory(prefix="memory-fallback-") as directory:
        asyncio.run(fallback(Path(directory), lambda condition, message: failures.append(message) if not condition else None))
    for failure in failures:
        print("FAIL test_memory_connector:", failure)
    if failures:
        raise SystemExit(1)
    print("PASS test_memory_connector: unavailable memory is logged and fails fast until the retry time, hello reaches a program that comes up, host services need the token and "
          "keep their own status, program errors keep theirs, a timeout marks memory unavailable, an abandoned host call stops, notices keep their order, session calls "
          "carry their handle, a restarted program is greeted again, another protocol or token counts as unavailable, and the chat layer serves raw while memory is down")


if __name__ == "__main__":
    main()
