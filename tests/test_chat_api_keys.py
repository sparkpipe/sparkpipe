#!/usr/bin/env python3
"""The public door of the chat layer: a request the door marks needs a valid key and sees only the public models and paths; internal requests are unchanged."""
import asyncio
import importlib.util
import json
import tempfile
import time
from pathlib import Path

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

    frontend = chat_frontend.Frontend({"models": [], "public": {"header": HEADER, "keys_file": keys.name, "models": []}}, tmp)
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

    check(refused({"models": [], "public": {"header": HEADER}}, tmp), "a public block without a keys file is refused")
    check(refused({"models": [], "public": {"header": HEADER, "keys_file": keys.name, "models": ["nope"]}}, tmp), "public models must be served models")
    check(refused({"models": [], "public": {"header": "", "keys_file": keys.name}}, tmp), "the door's header must be named")


with tempfile.TemporaryDirectory() as directory:
    asyncio.run(run(Path(directory)))
print("PASS test_chat_api_keys: the public door needs a valid, unrevoked, unexpired key; it shows and serves only the public models and paths, never naming an internal model; a keyed request is planned without caller identity, so no memory or command state; liveliness is open; a revoke applies from the next request; the keys file holds hashes only; internal requests are unchanged")
