import asyncio
import contextlib
import hmac
import json
import sys
import time

import aiohttp
from aiohttp import web

PROTOCOL = 1
UNAVAILABLE = "memory_unavailable"
HOST_METHODS = ("model_ready", "render_ids", "continuation_ids", "cut_ids", "count", "complete")
CALL_SECONDS = 30.0
HELLO_SECONDS = 10.0
RETRY_SECONDS = 5.0
RequestError = None


def log(event, **fields):
    print(json.dumps(dict(event=event, **fields)), file=sys.stderr, flush=True)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def error_of(exc):
    if isinstance(getattr(exc, "code", None), str) and isinstance(getattr(exc, "status", None), int):
        return {"code": exc.code, "message": str(getattr(exc, "message", exc)), "status": exc.status}
    return {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}


def refusal(status, code, message):
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


class Connector:
    def __init__(self, settings, base, host, read_token):
        self.url = settings["url"].rstrip("/")
        address, port = settings["host_listen"].rsplit(":", 1)
        self.listen = (address, int(port))
        self.token = read_token(base / settings["token_file"])
        self.host = host
        self.timeout = float(settings.get("timeout_seconds", CALL_SECONDS))
        self.ready = False
        self.retry_at = 0.0
        self.session = None
        self.runner = None
        self.notices = None
        self.sender = None

    async def start(self):
        self.session = aiohttp.ClientSession()
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_post("/v1/{method}", self.serve)
        self.runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await self.runner.setup()
        await web.TCPSite(self.runner, *self.listen).start()
        self.notices = asyncio.Queue()
        self.sender = asyncio.create_task(self.send())
        await self.connect()

    async def stop(self):
        if self.sender is not None:
            self.sender.cancel()
        if self.runner is not None:
            await self.runner.cleanup()
        if self.session is not None:
            await self.session.close()

    def headers(self):
        return {"Authorization": "Bearer " + self.token, "Content-Type": "application/json"}

    async def post(self, method, request, timeout):
        try:
            async with self.session.post(f"{self.url}/v1/{method}", data=encode(request).encode("utf-8"), headers=self.headers(),
                                         timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                status, data = response.status, await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            raise RequestError(503, UNAVAILABLE, f"the memory program did not answer {method}: {type(error).__name__}: {error}")
        try:
            answer = json.loads(data)
        except ValueError:
            answer = None
        if status == 200 and isinstance(answer, dict):
            return answer
        error = answer.get("error") if isinstance(answer, dict) and isinstance(answer.get("error"), dict) else {}
        if status == 401 or not error:
            raise RequestError(503, UNAVAILABLE, f"the memory program answered {method} with {status}: {error.get('message', '')}")
        raise RequestError(error.get("status") or status, error.get("code") or "internal_error", error.get("message") or f"{method} failed")

    async def connect(self):
        try:
            answer = await self.post("hello", {"protocol": PROTOCOL, "host_url": "http://%s:%d" % self.listen, "host_methods": list(HOST_METHODS)}, HELLO_SECONDS)
            if answer.get("protocol") != PROTOCOL:
                raise RequestError(503, UNAVAILABLE, f"the memory program speaks protocol {answer.get('protocol')}, the chat layer {PROTOCOL}")
        except RequestError as error:
            self.down(f"{error.code}: {error.message}")
            return False
        if not self.ready:
            log("memory_available", url=self.url, name=answer.get("name"), version=answer.get("version"))
        self.ready = True
        return True

    def down(self, reason):
        if self.ready or time.monotonic() >= self.retry_at:
            log("memory_unavailable", url=self.url, reason=reason)
        self.ready = False
        self.retry_at = time.monotonic() + RETRY_SECONDS

    async def attempt(self, method, request, timeout):
        try:
            return await self.post(method, request, timeout)
        except RequestError as error:
            if error.code == UNAVAILABLE:
                self.down(error.message)
            raise

    async def call(self, method, request=None, timeout=None):
        if not self.ready and (time.monotonic() < self.retry_at or not await self.connect()):
            raise RequestError(503, UNAVAILABLE, "the memory program is unavailable")
        try:
            return await self.attempt(method, request or {}, timeout or self.timeout)
        except RequestError as error:
            if error.code != "hello_first":
                raise
        if not await self.connect():
            raise RequestError(503, UNAVAILABLE, "the memory program is unavailable")
        return await self.attempt(method, request or {}, timeout or self.timeout)

    def notify(self, method, request=None):
        if self.notices is not None:
            self.notices.put_nowait((method, request or {}))

    async def send(self):
        while True:
            method, request = await self.notices.get()
            with contextlib.suppress(RequestError):
                await self.call(method, request)

    async def serve(self, request):
        if not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer " + self.token):
            return refusal(401, "unauthorized", "the chat layer's memory host needs the memory token")
        method = request.match_info["method"]
        handler = getattr(self.host, method, None) if method in HOST_METHODS else None
        if handler is None:
            return refusal(404, "unsupported", f"the chat layer has no host method {method}")
        try:
            value = json.loads(await request.read())
        except ValueError:
            value = None
        if not isinstance(value, dict):
            return refusal(400, "invalid_request", f"{method} takes a JSON object")
        try:
            result = handler(value)
            if asyncio.iscoroutine(result):
                result = await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = error_of(exc)
            return web.json_response({"error": error}, status=error.get("status", 500))
        return web.json_response(result, dumps=encode)


class Session:
    def __init__(self, connector, handle):
        self.connector = connector
        self.handle = handle
        self.rounds = 0

    async def claims(self, call):
        return (await self.connector.call("session.claims", {"session": self.handle, "call": call}))["claimed"]

    async def can_continue(self):
        return (await self.connector.call("session.can_continue", {"session": self.handle}))["can"]

    async def continue_with(self, content, reasoning, calls):
        self.rounds += 1
        return (await self.connector.call("session.continue_with", {"session": self.handle, "content": content, "reasoning": reasoning, "calls": calls}))["messages"]

    async def finished(self, plan_messages, content, reasoning, visible_calls, engine_finish, prompt_ids, token_ids):
        await self.connector.call("session.finished", {"session": self.handle, "plan_messages": plan_messages, "content": content, "reasoning": reasoning,
                                                       "visible_calls": visible_calls, "engine_finish": engine_finish, "prompt_ids": list(prompt_ids),
                                                       "token_ids": list(token_ids)})

    def release(self):
        self.connector.notify("session.release", {"session": self.handle})
