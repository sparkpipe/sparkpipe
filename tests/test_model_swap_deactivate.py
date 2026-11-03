#!/usr/bin/env python3
"""POST /deactivate stops every configured model's engines and API, releases every model's KV pool and frees every model's cold weightd arenas, then leaves the controller idle with no active model. It is refused while a switch runs, and an /activate sent during it starts after it."""
import asyncio
import os
import pathlib
import sys
import tempfile
import time

import aiohttp
from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "serving"))
import model_swap  # noqa: E402

GIB = 2 ** 30
FAKE_SSH = """import os, sys, time
words = sys.argv[2].split()
with open(os.path.join(os.path.dirname(sys.argv[0]), "ssh.log"), "a") as log:
    log.write(sys.argv[1] + " " + sys.argv[2] + "\\n")
if words[0] == "PROBE":
    print(200 * 2 ** 30)
elif words[0] == "READY":
    print("READY")
elif words[0] == "STOP":
    time.sleep(0.3)
"""


def model(name, port, state):
    return {"id": name, "kv_label": name + "_stage", "node_memory_bytes": 10 * GIB, "node_reclaim": "RECLAIM " + name,
            "node_start": f"START {name}", "node_ready": "READY {run_id}", "node_stop": f"STOP {name}",
            "api_start": f"touch {state}/api-started-{name}", "api_stop": f"rm -f {state}/api-started-{name}",
            "health": f"http://127.0.0.1:{port}/{name}"}


async def wait_idle(swapper, limit):
    begun = time.monotonic()
    while (swapper.state == "swapping" or (swapper.task is not None and not swapper.task.done())) and time.monotonic() - begun < limit:
        await asyncio.sleep(0.05)


async def scenario():
    with tempfile.TemporaryDirectory() as state:
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        pathlib.Path(state, "api-started-old").touch()
        log = pathlib.Path(state, "ssh.log")

        async def health(request):
            name = request.match_info["name"]
            up = pathlib.Path(state, f"api-started-{name}").exists()
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
        control_runner = web.AppRunner(model_swap.build(swapper, "secret"))
        await control_runner.setup()
        control_site = web.TCPSite(control_runner, "127.0.0.1", 0)
        await control_site.start()
        base = f"http://127.0.0.1:{control_site._server.sockets[0].getsockname()[1]}"
        headers = {"Authorization": "Bearer secret"}
        results = {}
        async with aiohttp.ClientSession(headers=headers) as session:
            results["active_before"] = swapper.active
            async with session.post(base + "/deactivate") as response:
                results["deactivate"] = response.status
                results["deactivate_state"] = (await response.json())["state"]
            async with session.post(base + "/deactivate") as response:
                results["deactivate_busy"] = response.status
            await wait_idle(swapper, 20)
            results["after"] = (swapper.state, swapper.active, swapper.error)
            results["record"] = swapper.history[-1] if swapper.history else None
            results["old_api_up"] = pathlib.Path(state, "api-started-old").exists()
            results["log"] = log.read_text().splitlines()
            log.write_text("")
            async with session.post(base + "/deactivate") as response:
                results["again"] = response.status
            await asyncio.sleep(0.05)
            async with session.post(base + "/activate", json={"model": "target"}) as response:
                results["activate_during"] = response.status
            await wait_idle(swapper, 20)
            results["final"] = (swapper.state, swapper.active, swapper.error)
            results["final_history"] = [(entry["from"], entry["to"], entry["ok"]) for entry in swapper.history]
        await control_runner.cleanup()
        await engine_runner.cleanup()
        return results


def main():
    r = asyncio.run(scenario())
    assert r["active_before"] == "old", r
    assert r["deactivate"] == 202 and r["deactivate_state"] == "swapping", r
    assert r["deactivate_busy"] == 409, r
    assert r["after"] == ("idle", None, None), r
    assert r["record"] and r["record"]["from"] == "old" and r["record"]["to"] is None and r["record"]["ok"], r["record"]
    assert not r["old_api_up"], "the active model's API still runs after /deactivate"
    for node in ("n0", "n1"):
        for command in ("STOP old", "STOP target", "DROP old_stage", "DROP target_stage", "RECLAIM old", "RECLAIM target"):
            assert f"{node} {command}" in r["log"], (node, command, r["log"])
    assert max(r["log"].index(f"n0 STOP {name}") for name in ("old", "target")) < min(r["log"].index(f"n0 DROP {name}_stage") for name in ("old", "target")), r["log"]
    assert max(r["log"].index(f"n0 DROP {name}_stage") for name in ("old", "target")) < min(r["log"].index(f"n0 RECLAIM {name}") for name in ("old", "target")), r["log"]
    assert r["again"] == 202 and r["activate_during"] == 202, r
    assert r["final"] == ("idle", "target", None), r
    assert r["final_history"][-2:] == [(None, None, True), (None, "target", True)], r["final_history"]
    print("PASS test_model_swap_deactivate: /deactivate stopped every model, released every KV pool and freed every model's cold arenas, left no active model, refused a second call while running, and an /activate sent during it ran after it")


main()
