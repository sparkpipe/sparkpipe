#!/usr/bin/env python3
import asyncio
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chat_frontend = load("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
api_keys = load("api_keys", ROOT / "tools" / "serving" / "api_keys.py")
DOOR = "X-SparkPipe-Public"
CONTROLLER_TOKEN = "k" * 40


class Controller:
    def __init__(self):
        self.state = {"fleet": "F1", "user": "ACONE", "holder": None, "wiped": 100, "held_until": 0, "confirmed": True, "handover": None}
        self.admitted = []
        self.refusal = None

    async def view(self, request):
        return web.json_response(self.state)

    async def admit(self, request):
        if request.headers.get("Authorization") != "Bearer " + CONTROLLER_TOKEN:
            return web.json_response({"error": {"code": "unauthorized", "message": "no"}}, status=401)
        self.admitted.append((await request.json())["account"])
        if self.refusal is not None:
            status, code, retry = self.refusal
            return web.json_response({"error": {"code": code, "message": f"refused with {code}"}}, status=status, headers={"Retry-After": retry})
        return web.json_response(self.state)


class Engine:
    def __init__(self):
        self.open = 0
        self.closed = 0

    async def completions(self, request):
        self.open += 1
        try:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(("data: " + json.dumps({"tokens": [1], "choices": [{}]}) + "\n\n").encode())
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.closed += 1
            raise
        finally:
            self.open -= 1
        return response


class Parser:
    def __init__(self):
        self.fed = 0
        self.stopped = False
        self.ended = False
        self.calls = 0

    def feed(self, text):
        self.fed += 1
        return [("content", "partial")] if self.fed == 1 else []

    def finish(self):
        return []


class Model:
    def __init__(self, engine):
        self.id = "m"
        self.engine = engine
        self.vocabulary = SimpleNamespace(decoder=lambda: (lambda token: ""))
        self.stop_token_ids = []
        self.gate = SimpleNamespace(wait_seconds=1)


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


def refused(config, base):
    try:
        chat_frontend.Frontend(config, base)
    except SystemExit as error:
        return str(error)
    return None


async def run(tmp, failures):
    def check(condition, message):
        if not condition:
            failures.append(message)

    keys = tmp / "api_keys.json"
    made = subprocess.run([sys.executable, str(ROOT / "tools" / "serving" / "api_keys.py"), "--file", str(keys), "create", "--gateway", "--note", "market"],
                          capture_output=True, text=True, check=True)
    gateway_id, gateway_key = made.stdout.split()
    plain_id, plain_key = api_keys.create(keys, 0, "direct")
    listed = subprocess.run([sys.executable, str(ROOT / "tools" / "serving" / "api_keys.py"), "--file", str(keys), "list"], capture_output=True, text=True, check=True).stdout
    check(any(line.startswith(gateway_id) and " gateway " in line for line in listed.splitlines()) and
          not any(line.startswith(plain_id) and "gateway" in line for line in listed.splitlines()), f"list marks only the gateway key: {listed!r}")
    check(json.loads(keys.read_text())["keys"][0]["gateway"] is True and gateway_key not in keys.read_text(), "create --gateway marks the key in the keys file, which keeps only its hash")

    controller, engine = Controller(), Engine()
    upstream = web.Application()
    upstream.router.add_get("/working-state", controller.view)
    upstream.router.add_post("/working-state/admit", controller.admit)
    upstream.router.add_post("/v1/completions", engine.completions)
    upstream_runner, upstream_base = await serve(upstream, handler_cancellation=True)
    token_file = tmp / "controller-token"
    token_file.write_text(CONTROLLER_TOKEN + "\n")
    token_file.chmod(0o600)
    control = {"status_url": upstream_base + "/status", "working_state_url": upstream_base + "/working-state"}
    chat_frontend.Frontend.WORKING_STATE_POLL_SECONDS = 0.05
    frontend = chat_frontend.Frontend({"models": [], "controller_token_file": token_file.name, "control": control,
                                       "public": {"header": DOOR, "keys_file": keys.name, "models": []}}, tmp)
    model = Model(upstream_base)
    frontend.model_for = lambda body: model
    frontend.public_models = frozenset({"m"})
    frontend.plan = lambda model, body, headers: {"trace": {}, "engine_body": {"prompt_token_ids": [1, 2], "max_tokens": 8, "stream": True},
                                                  "parser": Parser(), "prompt_tokens": 2, "session": None, "body": body}
    seen = []

    async def answer(request, model, body, plan, started):
        seen.append(plan["account"])
        return web.json_response({"ok": True})
    frontend.answer = answer
    front_runner, front = await serve(chat_frontend.build(frontend), **chat_frontend.SERVER_OPTIONS)
    gateway = {DOOR: "1", "Authorization": "Bearer " + gateway_key}
    plain = {DOOR: "1", "Authorization": "Bearer " + plain_key}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as client:
            async def ask(headers, stream=False):
                async with client.post(front + "/v1/chat/completions", headers=headers, json={"model": "m", "stream": stream, "messages": []}) as response:
                    text = await response.text()
                    return response.status, text, response.headers.get("Retry-After")

            def code(text):
                try:
                    return (json.loads(text).get("error") or {}).get("code")
                except json.JSONDecodeError:
                    return None

            status, text, _ = await ask(dict(gateway, **{"X-SparkPipe-Account": "ACONE", "X-SparkPipe-Fleet": "F1"}))
            check(status == 200 and controller.admitted == ["ACONE"] and seen == ["ACONE"], f"a gateway key's request runs as the account in its header: {status} {controller.admitted} {seen}")
            for headers, expected, why in (({"X-SparkPipe-Fleet": "F1"}, "account_required", "a gateway request without the account header"),
                                           ({"X-SparkPipe-Account": "ACONE"}, "fleet_required", "a gateway request without the fleet header"),
                                           ({"X-SparkPipe-Account": "ACONE", "X-SparkPipe-Fleet": "F2"}, "wrong_fleet", "a gateway request for another fleet"),
                                           ({"X-SparkPipe-Account": "AC ONE", "X-SparkPipe-Fleet": "F1"}, "invalid_account", "an account header that is not an id"),
                                           ({"X-SparkPipe-Account": "sparkpipe", "X-SparkPipe-Fleet": "F1"}, "invalid_account", "a gateway request posing as SparkPipe's own use")):
                status, text, _ = await ask(dict(gateway, **headers))
                check(status == 400 and code(text) == expected and controller.admitted == ["ACONE"], f"{why} gets 400 {expected} and is never admitted: {status} {text}")
            await ask(dict(plain, **{"X-SparkPipe-Account": "ACONE", "X-SparkPipe-Fleet": "F1"}))
            await ask({"X-SparkPipe-Account": "ACONE"})
            check(controller.admitted[1:] == ["sparkpipe", "sparkpipe"] and seen[1:] == ["sparkpipe", "sparkpipe"],
                  f"another public key and an internal request run as sparkpipe whatever headers they send: {controller.admitted}")
            controller.refusal = (503, "fleet_held", "1234")
            status, text, retry = await ask(dict(gateway, **{"X-SparkPipe-Account": "ACOTHER", "X-SparkPipe-Fleet": "F1"}))
            check(status == 503 and code(text) == "fleet_held" and retry == "1234" and len(seen) == 3,
                  f"a refused admission reaches the client with the controller's status, code and Retry-After, and nothing runs: {status} {text} {retry}")
            controller.refusal = (503, "holder_schedule_unknown", "30")
            status, text, retry = await ask({})
            check(status == 503 and code(text) == "holder_schedule_unknown" and retry == "30", f"an unknown holder schedule refuses SparkPipe's own requests too: {status} {text}")
            controller.refusal = None

            release = asyncio.Event()
            held = []

            async def held_answer(request, model, body, plan, started):
                held.append(plan["account"])
                await release.wait()
                return web.json_response({"ok": True})
            frontend.answer = held_answer
            holder = dict(gateway, **{"X-SparkPipe-Account": "ACONE", "X-SparkPipe-Fleet": "F1"})
            holder_requests = [asyncio.ensure_future(ask(holder)) for _ in range(chat_frontend.PUBLIC_KEY_CONCURRENCY + 2)]
            check(await until(lambda: len(held) == chat_frontend.PUBLIC_KEY_CONCURRENCY + 2, 10),
                  f"the gateway key is not held to the per-key limit, so a fleet's holder runs more than {chat_frontend.PUBLIC_KEY_CONCURRENCY} requests at once: {len(held)}")
            plain_requests = [asyncio.ensure_future(ask(plain)) for _ in range(chat_frontend.PUBLIC_KEY_CONCURRENCY + 1)]
            check(await until(lambda: sum(1 for task in plain_requests if task.done()) == 1, 10), "one request over another key's limit is answered at once")
            over = [task.result() for task in plain_requests if task.done()]
            check(over and over[0][0] == 429 and code(over[0][1]) == "rate_limited" and len(held) == 2 * chat_frontend.PUBLIC_KEY_CONCURRENCY + 2,
                  f"another public key still runs at most {chat_frontend.PUBLIC_KEY_CONCURRENCY} requests at once: {over}")
            release.set()
            finished = await asyncio.gather(*holder_requests, *plain_requests)
            check(sorted(result[0] for result in finished) == [200] * (2 * chat_frontend.PUBLIC_KEY_CONCURRENCY + 2) + [429], f"every held request then finishes: {finished}")
            check(frontend.public_inflight == {}, f"no request is left counted against a key: {frontend.public_inflight}")

            del frontend.answer
            frontend.prefixes = {"m": chat_frontend.PrefixLedger()}
            frontend.public_prefixes = {(gateway_id, "m"): chat_frontend.PrefixLedger()}
            old = dict(gateway, **{"X-SparkPipe-Account": "ACONE", "X-SparkPipe-Fleet": "F1"})
            new = dict(gateway, **{"X-SparkPipe-Account": "ACTWO", "X-SparkPipe-Fleet": "F1"})
            captured = io.StringIO()
            with contextlib.redirect_stderr(captured):
                streamed = []

                async def stream(headers):
                    async with client.post(front + "/v1/chat/completions", headers=headers, json={"model": "m", "stream": True, "messages": []}) as response:
                        async for line in response.content:
                            if line.strip():
                                streamed.append(line.decode().strip())
                        return response.status

                old_stream = asyncio.ensure_future(stream(old))
                old_whole = asyncio.ensure_future(ask(old))
                new_stream = asyncio.ensure_future(ask(new, True))
                check(await until(lambda: engine.open == 3 and any("partial" in line for line in streamed), 10), f"three requests reach the engine: {engine.open}")
                controller.state = dict(controller.state, handover={"from": "ACONE", "to": "ACTWO", "phase": "draining"})
                status = await asyncio.wait_for(old_stream, 10)
                whole_status, whole_text, _ = await asyncio.wait_for(old_whole, 10)
                errors = [json.loads(line[5:]) for line in streamed if line.startswith("data: {\"error\"")]
                check(status == 200 and errors and errors[0]["error"]["code"] == "fleet_handover" and streamed[-1] == "data: [DONE]",
                      f"the old account's stream ends with a fleet_handover error and [DONE]: {streamed}")
                check(whole_status == 503 and code(whole_text) == "fleet_handover", f"the old account's waiting request gets 503 fleet_handover: {whole_status} {whole_text}")
                check(await until(lambda: engine.closed == 2 and engine.open == 1, 5), f"both of the old account's engine requests are closed and the new account's stays: open {engine.open} closed {engine.closed}")
                check(not new_stream.done(), "the new account's request is not cancelled")
                check(frontend.prefixes == {} and frontend.public_prefixes == {}, "the handover clears the prefix ledgers")
                check(frontend.running and set(frontend.running.values()) == {"ACTWO"}, f"only the new account's request is still running: {frontend.running}")
                new_stream.cancel()
                await asyncio.gather(new_stream, return_exceptions=True)
                check(await until(lambda: engine.open == 0, 5), "the new account's request closes when its client leaves")
            check(captured.getvalue().count('"finish_reason": "handover"') == 2, f"each cancelled request logs a handover finish: {captured.getvalue()[-600:]}")
            controller.state = dict(controller.state, user="ACTWO", wiped=200, handover=None)
            await asyncio.sleep(0.2)
            frontend.public_prefixes = {(gateway_id, "m"): chat_frontend.PrefixLedger()}
            await asyncio.sleep(0.2)
            check(frontend.public_prefixes != {}, "with no change of user the ledgers are left alone")
    finally:
        await front_runner.cleanup()
        await upstream_runner.cleanup()

    stub = chat_frontend.Model
    chat_frontend.Model = lambda spec, base: SimpleNamespace(id=spec["id"], gate=SimpleNamespace(activation_url=None))
    try:
        check("needs control" in (refused({"models": [{"id": "m"}]}, tmp) or ""), "a front end that serves models without the fleet controller is refused")
        check(refused({"models": [], "controller_token_file": token_file.name, "control": {"status_url": control["status_url"]}}, tmp),
              "a control block without the working-state controller is refused")
    finally:
        chat_frontend.Model = stub


def main():
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(run(Path(directory), failures))
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS test_chat_frontend_working_state: a gateway key's request runs as the account in X-SparkPipe-Account and must name this fleet in "
          "X-SparkPipe-Fleet, every other request runs as sparkpipe; every request is admitted by the controller first and a refusal keeps its "
          "status, code and Retry-After; at a handover the old account's requests are cancelled (streams end with fleet_handover and [DONE]), "
          "their engine requests close, the new account's keep running and the prefix ledgers are cleared; the controller is required to serve models")


if __name__ == "__main__":
    main()
