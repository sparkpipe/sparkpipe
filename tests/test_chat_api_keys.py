#!/usr/bin/env python3
"""The public door of the chat layer: a request the door marks needs a valid key and sees only the public models and paths; internal requests are unchanged."""
import asyncio
import importlib.util
import json
import tempfile
import time
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
HEADER = "X-SparkPipe-Public"


class FakeModel:
    def __init__(self, name):
        self.id = name
        self.context_tokens = 4096
        self.reasoning_effort = {}
        self.gate = SimpleNamespace(state=lambda: {"waiting": 1 if name == "inside" else 0})


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAIL test_chat_api_keys: {message}")


def refused(config, base):
    try:
        chat_frontend.Frontend(config, base)
    except SystemExit as error:
        return str(error)
    return None


async def run(tmp):
    keys = tmp / "api_keys.json"
    good_id, good = api_keys.create(keys, 0, "owner")
    gone_id, gone = api_keys.create(keys, 0, "")
    api_keys.revoke(keys, gone_id)
    old_id, old = api_keys.create(keys, 1, "")
    document = json.loads(keys.read_text())
    for entry in document["keys"]:
        if entry["id"] == old_id:
            entry["expires"] = int(time.time()) - 1
    keys.write_text(json.dumps(document))
    stored = keys.read_text()
    check(good not in stored and gone not in stored and good_id in stored, "the keys file keeps ids and hashes, never a key")
    check(oct(keys.stat().st_mode & 0o777) in ("0o600", "0o644"), "the keys file is written privately")

    second_id, second = api_keys.create(keys, 0, "second")
    controller_status = {"active": "inside", "state": "failed", "target": "open", "wanted": "inside", "phase": None,
                         "error": "memory probe failed on 1 node(s); first spark7 exit 1: secret shell output",
                         "memory": {"spark7": 12}, "models": [{"id": "open", "active": False}, {"id": "inside", "active": True}],
                         "draining": {"model": "inside", "live_requests": 2, "waited_s": 3.0, "limit_s": 600, "quiet_s": 0.0, "quiet_needed_s": 15},
                         "history": [{"from": "inside", "to": "open", "finished": "2026-10-27T10:00:00Z", "seconds": 40.0, "forced": False, "ok": False,
                                      "error": "start failed on 2 node(s); first spark3 exit 9: secret"}]}
    looked_up = []

    async def status(request):
        return web.json_response(controller_status)

    async def auths(request):
        looked_up.append(request.headers.get("Authorization", ""))
        return web.json_response({"role": "admin"}) if request.headers.get("Authorization") == "Bearer admin-login" else web.json_response({"role": "user"})
    upstream = web.Application()
    upstream.router.add_get("/status", status)
    upstream.router.add_get("/api/v1/auths/", auths)
    upstream_runner = web.AppRunner(upstream)
    await upstream_runner.setup()
    upstream_site = web.TCPSite(upstream_runner, "127.0.0.1", 0)
    await upstream_site.start()
    upstream_base = f"http://127.0.0.1:{upstream_site._server.sockets[0].getsockname()[1]}"
    token_file = tmp / "controller-token"
    token_file.write_text("k" * 40 + "\n")
    token_file.chmod(0o600)
    frontend = chat_frontend.Frontend({"models": [], "public": {"header": HEADER, "keys_file": keys.name, "models": []},
                                       "controller_token_file": token_file.name, "control": {"status_url": upstream_base + "/status", "force_url": upstream_base + "/force", "admin_url": upstream_base + "/api/v1/auths/"}}, tmp)
    frontend.models = {"open": FakeModel("open"), "inside": FakeModel("inside")}
    frontend.public_models = frozenset({"open"})
    app = chat_frontend.build(frontend)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    public = {HEADER: "1"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base + "/v1/models") as response:
                names = [entry["id"] for entry in (await response.json())["data"]]
                check(response.status == 200 and "inside" in names and "open" in names, "an internal request sees every model, with no key")
            async with session.get(base + "/v1/models", headers=public) as response:
                check(response.status == 401 and (await response.json())["error"]["code"] == "invalid_api_key", "a public request without a key is refused")
            for key, why in ((gone, "a revoked key"), (old, "an expired key"), ("sk-sp-made-up", "an unknown key"), (good[len("sk-sp-"):], "a key without its prefix")):
                async with session.get(base + "/v1/models", headers=dict(public, Authorization="Bearer " + key)) as response:
                    check(response.status == 401, f"{why} is refused")
            with_key = dict(public, Authorization="Bearer " + good)
            async with session.get(base + "/v1/models", headers=with_key) as response:
                names = [entry["id"] for entry in (await response.json())["data"]]
                check(response.status == 200 and names == ["open", "open-bulk"], f"a public key sees only the public models: {names}")
            async with session.post(base + "/v1/chat/completions", headers=with_key, json={"model": "inside", "messages": [{"role": "user", "content": "hi"}]}) as response:
                body = await response.json()
                check(response.status == 404 and body["error"]["message"].endswith("served: ['open']"), "a public key cannot reach an internal model, and the served list it gets names only the public ones")
            planned = []

            def plan(model, body, headers):
                planned.append(headers)
                raise chat_frontend.RequestError(400, "planned", "stop after planning")
            frontend.plan = plan
            identified = {"X-OpenWebUI-User-Id": "u1", "X-OpenWebUI-Chat-Id": "c1"}
            for headers in (dict(with_key, **identified), identified):
                async with session.post(base + "/v1/chat/completions", headers=headers, json={"model": "open", "messages": [{"role": "user", "content": "hi"}]}) as response:
                    check(response.status == 400, f"the planning stub answers: {response.status}")
            check(planned[0] is None and planned[1] is not None and planned[1].get("X-OpenWebUI-User-Id") == "u1", "a keyed request is planned with no caller identity, so it opens no chat memory and runs no chat command; an internal one keeps its headers")
            limit = frontend.public_body_limit()
            check(limit == 16 * 4096 + (1 << 20), f"the public body limit is 16 bytes per token of the largest public context plus 1 MiB: {limit}")
            oversize = json.dumps({"model": "open", "messages": [{"role": "user", "content": "x" * (limit + 1)}]})
            async with session.post(base + "/v1/chat/completions", headers=dict(with_key, **{"Content-Type": "application/json"}), data=oversize) as response:
                check(response.status == 413 and (await response.json())["error"]["code"] == "request_too_large", f"an oversize public body is refused before parsing: {response.status}")
            check(len(planned) == 2, "an oversize public body never reaches planning")
            async with session.post(base + "/v1/chat/completions", headers={"Content-Type": "application/json"}, data=oversize) as response:
                check(response.status == 400 and len(planned) == 3, f"the same body from inside reaches planning: {response.status}")

            async def chunks():
                yield b'{"model": "open", "messages": []}'
            async with session.post(base + "/v1/chat/completions", headers=dict(with_key, **{"Content-Type": "application/json"}), data=chunks()) as response:
                check(response.status == 411 and len(planned) == 3, f"a public body without a length is refused: {response.status}")

            release = asyncio.Event()
            answered = []

            def plan_ok(model, body, headers):
                return {"trace": {}, "engine_body": {"prompt_token_ids": list(range(128))}, "prompt_tokens": 128}

            async def answer_wait(request, model, body, plan, started):
                answered.append(plan.get("public_key", {}).get("id"))
                await release.wait()
                return web.json_response({"ok": True})
            frontend.plan, frontend.answer = plan_ok, answer_wait
            chat_body = {"model": "open", "messages": [{"role": "user", "content": "hi"}]}

            async def send(headers):
                async with session.post(base + "/v1/chat/completions", headers=headers, json=chat_body) as response:
                    return response.status, (await response.json()).get("error", {}).get("code")
            running = [asyncio.ensure_future(send(with_key)) for _ in range(chat_frontend.PUBLIC_KEY_CONCURRENCY)]
            for _ in range(100):
                if len(answered) == chat_frontend.PUBLIC_KEY_CONCURRENCY:
                    break
                await asyncio.sleep(0.01)
            check(answered == [good_id] * chat_frontend.PUBLIC_KEY_CONCURRENCY, f"a key runs up to its concurrency limit: {answered}")
            check(await send(with_key) == (429, "rate_limited"), "one more request from the same key is refused while its limit is in use")
            other = asyncio.ensure_future(send(dict(public, Authorization="Bearer " + second)))
            inside = asyncio.ensure_future(send({}))
            for _ in range(100):
                if len(answered) == chat_frontend.PUBLIC_KEY_CONCURRENCY + 2:
                    break
                await asyncio.sleep(0.01)
            check(sorted(answered[chat_frontend.PUBLIC_KEY_CONCURRENCY:], key=str) == sorted([second_id, None], key=str), f"another key and an internal request are not held by the first key's limit: {answered}")
            release.set()
            results = await asyncio.gather(*running, other, inside)
            check(all(status == 200 for status, _ in results) and frontend.public_inflight == {}, f"every admitted request completes and the counts return to zero: {results} {frontend.public_inflight}")
            check(await send(with_key) == (200, None), "the key is admitted again once its requests finish")

            class FailingCommands:
                async def run(self, model, identity, messages, name, argument, memory_off):
                    raise chat_frontend.RequestError(503, "engine_unavailable", "the engine is switching")
            frontend.commands = FailingCommands()
            frontend.plan = lambda model, body, headers: {"trace": {}, "command": ("compact", "50%"), "identity": None, "messages": [], "memory_off": True}
            async with session.post(base + "/v1/chat/completions", json={"model": "open", "messages": [{"role": "user", "content": "/compact 50%"}]}) as response:
                text = await response.text()
                failed = json.loads(text) if response.content_type == "application/json" else {"error": {"code": text[:60]}}
                check(response.status == 503 and response.content_type == "application/json" and failed["error"]["code"] == "engine_unavailable",
                      f"a command that hits an engine error answers with the OpenAI error body, not a text 500: {response.status} {failed}")
            frontend.commands, frontend.plan = None, plan_ok

            prompt = list(range(256))
            engine_usage = {"prompt_tokens": 256, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 192}}
            entry, other_entry = {"id": good_id}, {"id": second_id}
            model = frontend.models["open"]

            def keyed(key):
                return {"public_key": key, "engine_body": {"prompt_token_ids": prompt}, "public_cached": frontend.public_prefix_ledger(key, model).seen(prompt)[0]}
            check(frontend.usage_block(engine_usage, 256, {})["prompt_tokens_details"]["cached_tokens"] == 192, "an internal request reports the engine's cached tokens")
            check(frontend.usage_block(engine_usage, 256, keyed(entry))["prompt_tokens_details"]["cached_tokens"] == 0,
                  "a key reports no cached tokens for a prefix only someone else sent")
            frontend.record_public_prefix(model, keyed(entry), {"token_ids": [7, 8, 9, 10]})
            check(frontend.usage_block(engine_usage, 256, keyed(entry))["prompt_tokens_details"]["cached_tokens"] == 192,
                  "a key that sent the prompt itself reports the engine's cached tokens")
            check(frontend.usage_block(engine_usage, 256, keyed(other_entry))["prompt_tokens_details"]["cached_tokens"] == 0,
                  "another key still reports none for the first key's prompt")

            async with session.get(base + "/v1/sparkpipe/status", headers=with_key) as response:
                view = await response.json()
            check(response.status == 200 and "error" not in view and "memory" not in view and "secret" not in json.dumps(view),
                  f"a public key's status carries no controller error, memory or shell text: {view}")
            check(view["active"] is None and view["wanted"] is None and view["target"] == "open" and [m["id"] for m in view["models"]] == ["open"] and view["draining"]["model"] is None,
                  f"a public key's status names only public models: {view}")
            check(view["history"] == [{"finished": "2026-10-27T10:00:00Z", "seconds": 40.0, "forced": False, "ok": False, "from": None, "to": "open"}] and view["draining"]["live_requests"] == 2 and view["waiting"] == {},
                  f"a public key's status keeps the history and drain timing without their detail: {view}")
            check(looked_up == [], "a public key is never sent to SparkPipe Chat to check a login")
            async with session.get(base + "/v1/sparkpipe/status") as response:
                anonymous = await response.json()
            check(anonymous == view, f"a status request without a login gets the same view as a public key: {anonymous}")
            async with session.get(base + "/v1/sparkpipe/status", headers={"Authorization": "Bearer user-login"}) as response:
                check("error" not in await response.json(), "a regular user's login gets the view without detail")
            async with session.get(base + "/v1/sparkpipe/status", headers={"Authorization": "Bearer admin-login"}) as response:
                full = await response.json()
            check(full["error"] == controller_status["error"] and full["memory"] == {"spark7": 12} and full["history"][0]["error"].endswith("secret") and full["waiting"] == {"inside": 1},
                  f"the admin's login gets the controller's full status: {full}")
            async with session.get(base + "/health", headers=with_key) as response:
                check(response.status == 404, "the engine health detail is not a public path")
            async with session.get(base + "/health/liveliness", headers=public) as response:
                check(response.status == 200, "liveliness answers the public door without a key")
            async with session.get(base + "/v1/models", headers={HEADER: "0", "Authorization": "Bearer " + good}) as response:
                names = [entry["id"] for entry in (await response.json())["data"]]
                check(names == ["open", "open-bulk"], "any value of the door's header marks the request public")
            api_keys.revoke(keys, good_id)
            async with session.get(base + "/v1/models", headers=with_key) as response:
                check(response.status == 401, "a key revoked while the chat layer runs is refused from the next request")
    finally:
        await runner.cleanup()
        await upstream_runner.cleanup()

    check(refused({"models": [], "public": {"header": HEADER}}, tmp), "a public block without a keys file is refused")
    check(refused({"models": [], "public": {"header": HEADER, "keys_file": keys.name, "models": ["nope"]}}, tmp), "public models must be served models")
    check(refused({"models": [], "public": {"header": "", "keys_file": keys.name}}, tmp), "the door's header must be named")


with tempfile.TemporaryDirectory() as directory:
    asyncio.run(run(Path(directory)))
print("PASS test_chat_api_keys: the public door needs a valid, unrevoked, unexpired key; it shows and serves only the public models and paths, never naming an internal model; a keyed request is planned without caller identity, so no memory or command state; an oversize or unsized public body is refused before parsing; a key runs at most four requests at once; a key reports cached tokens only for prefixes it sent; public and anonymous status carry no controller error, memory or shell text and public status names only public models, while the admin login sees it all; liveliness is open; a revoke applies from the next request; the keys file holds hashes only; internal requests are unchanged")
