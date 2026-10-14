#!/usr/bin/env python3
"""The swap controller answers only callers that present its token: without it, with a wrong one or with any other scheme, /status, /activate and /force answer 401 and the fleet is not touched; with it a switch runs. The controller refuses to start without --token-file, and refuses a token file that group or others can read or that holds a short token."""
import asyncio
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile

import aiohttp
from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("model_swap", ROOT / "tools" / "serving" / "model_swap.py")
model_swap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_swap)

GIB = 2 ** 30
TOKEN = "c" * 40
FAKE_SSH = """import sys
words = sys.argv[2].split()
if words[0] == "PROBE":
    print(200 * 2 ** 30)
elif words[0] == "READY":
    print("READY")
"""


def model(name, port, state):
    return {"id": name, "kv_label": name + "_stage", "node_memory_bytes": 10 * GIB, "node_reclaim": "RECLAIM " + name,
            "node_start": f"START {name}", "node_ready": "READY {run_id}", "node_stop": f"STOP {name}",
            "api_start": f"touch {state}/api-started-{name}", "api_stop": f"rm -f {state}/api-started-{name}",
            "health": f"http://127.0.0.1:{port}/{name}"}


def token_file(directory, name, text, mode):
    path = pathlib.Path(directory, name)
    path.write_text(text)
    path.chmod(mode)
    return path


def refused(path):
    try:
        model_swap.read_token(path)
    except SystemExit as error:
        return str(error)
    return None


async def scenario(failures):
    def check(condition, name):
        if not condition:
            failures.append(name)

    with tempfile.TemporaryDirectory() as state:
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        pathlib.Path(state, "api-started-old").touch()

        async def health(request):
            up = pathlib.Path(state, f"api-started-{request.match_info['name']}").exists()
            return web.json_response({"connected_ranks": 2 if up else 0, "live_requests": 0})

        engines = web.Application()
        engines.router.add_get("/{name}", health)
        engine_runner = web.AppRunner(engines)
        await engine_runner.setup()
        engine_site = web.TCPSite(engine_runner, "127.0.0.1", 0)
        await engine_site.start()
        port = engine_site._server.sockets[0].getsockname()[1]
        swapper = model_swap.Swapper({
            "nodes": ["n0", "n1"], "ssh": [sys.executable, os.path.join(state, "fake_ssh.py")], "drop_kv": "DROP {label}",
            "memory_probe": "PROBE", "memory_margin_bytes": GIB,
            "ready_seconds": 5, "api_seconds": 5, "drain_seconds": 5, "node_seconds": 10,
            "models": [model("old", port, state), model("target", port, state)],
        })
        token = model_swap.read_token(token_file(state, "token", TOKEN + "\n", 0o600))
        check(token == TOKEN, "a private token file is read with its newline stripped")
        runner = web.AppRunner(model_swap.build(swapper, token))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        good = {"Authorization": "Bearer " + TOKEN}
        bad = [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer " + TOKEN + "x"}, {"Authorization": "Bearer " + TOKEN[:-1]},
               {"Authorization": "Basic " + TOKEN}, {"Authorization": TOKEN}]
        try:
            async with aiohttp.ClientSession() as session:
                for headers in bad:
                    for method, path, body in (("GET", "/status", None), ("POST", "/activate", {"model": "target"}),
                                               ("POST", "/force", {"model": "target"})):
                        async with session.request(method, base + path, json=body, headers=headers) as response:
                            check(response.status == 401, f"{method} {path} with {headers or 'no Authorization'} answers 401, not {response.status}")
                check(swapper.wanted is None and swapper.task is None and swapper.history == [],
                      "refused requests never start a switch")
                check(not pathlib.Path(state, "api-started-target").exists() and pathlib.Path(state, "api-started-old").exists(),
                      "refused requests never stop or start an engine")
                async with session.get(base + "/status", headers=good) as response:
                    body = await response.json()
                    check(response.status == 200 and body["active"] == "old", f"the token reads the status: {response.status} {body}")
                async with session.post(base + "/activate", json={"model": "target"}, headers=good) as response:
                    check(response.status == 202, f"the token starts a switch: {response.status}")
                for _ in range(100):
                    if swapper.task is not None and swapper.task.done():
                        break
                    await asyncio.sleep(0.05)
                check(swapper.active == "target" and swapper.history and swapper.history[-1]["ok"], f"the switch completes: {swapper.history}")
        finally:
            await runner.cleanup()
            await engine_runner.cleanup()

        for name, text, mode in (("shared", TOKEN, 0o640), ("world", TOKEN, 0o604), ("short", "c" * 31, 0o600), ("blank", "\n", 0o600)):
            check(refused(token_file(state, name, text, mode)) is not None, f"a {name} token file is refused")
        message = refused(token_file(state, "group", TOKEN, 0o660)) or ""
        check("660" in message and "owner only" in message, f"the refusal names the mode: {message!r}")

        config = pathlib.Path(state, "config.json")
        config.write_text("{}")
        started = subprocess.run([sys.executable, str(ROOT / "tools" / "serving" / "model_swap.py"), "--config", str(config)],
                                 capture_output=True, text=True, timeout=60)
        check(started.returncode != 0 and "--token-file" in started.stderr, f"the controller does not start without --token-file: {started.returncode} {started.stderr[-200:]}")


def main():
    failures = []
    asyncio.run(scenario(failures))
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS test_model_swap_token: every route needs the controller token; refused calls never touch the fleet; no token file, a shared one or a short token stops the controller")


if __name__ == "__main__":
    main()
