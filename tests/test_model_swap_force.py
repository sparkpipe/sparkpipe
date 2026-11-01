#!/usr/bin/env python3
"""The admin's Switch now ends a model switch's wait at once: while the old engine still has a live reply, POST /force makes the controller stop it within a poll instead of waiting out the drain, and records the swap as forced. With no switch waiting /force is refused, and /force with a model starts a forced switch to it."""
import asyncio
import os
import pathlib
import sys
import tempfile
import time

from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "serving"))
import model_swap  # noqa: E402
import working_state_fixture  # noqa: E402

GIB = 2 ** 30
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


async def scenario():
    with tempfile.TemporaryDirectory() as state:
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        pathlib.Path(state, "api-started-old").touch()

        async def health(request):
            name = request.match_info["name"]
            up = pathlib.Path(state, f"api-started-{name}").exists()
            return web.json_response({"connected_ranks": 2 if up else 0, "live_requests": 1 if up and name == "old" else 0})

        engines = web.Application()
        engines.router.add_get("/{name}", health)
        engine_runner = web.AppRunner(engines)
        await engine_runner.setup()
        engine_site = web.TCPSite(engine_runner, "127.0.0.1", 0)
        await engine_site.start()
        port = engine_site._server.sockets[0].getsockname()[1]
        working_state_fixture.write_token(state)
        swapper = model_swap.Swapper({
            "working_state": working_state_fixture.block(state),
            "nodes": ["n0", "n1"], "ssh": [sys.executable, os.path.join(state, "fake_ssh.py")], "drop_kv": "DROP {label}",
            "memory_probe": "PROBE", "memory_margin_bytes": GIB,
            "ready_seconds": 5, "api_seconds": 5, "drain_seconds": 60, "drain_quiet_seconds": 15, "node_seconds": 10,
            "models": [model("old", port, state), model("target", port, state), model("third", port, state)],
        })
        control = web.Application()
        control.on_startup.append(swapper.startup)
        control.on_cleanup.append(swapper.cleanup)
        control.router.add_get("/status", swapper.get_status)
        control.router.add_post("/activate", swapper.activate)
        control.router.add_post("/force", swapper.force)
        control_runner = web.AppRunner(control)
        await control_runner.setup()
        control_site = web.TCPSite(control_runner, "127.0.0.1", 0)
        await control_site.start()
        working_state_fixture.unheld(swapper)
        base = f"http://127.0.0.1:{control_site._server.sockets[0].getsockname()[1]}"
        results = {}
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(base + "/force") as response:
                results["idle_force"] = response.status
            async with session.post(base + "/activate", json={"model": "target"}) as response:
                results["activate"] = response.status
            draining = None
            for _ in range(50):
                async with session.get(base + "/status") as response:
                    draining = (await response.json())["draining"]
                if draining:
                    break
                await asyncio.sleep(0.1)
            results["draining"] = draining
            await asyncio.sleep(1.5)
            results["still_old"] = pathlib.Path(state, "api-started-old").exists()
            forced_at = time.monotonic()
            async with session.post(base + "/force") as response:
                results["force"] = response.status
            while swapper.state == "swapping" or swapper.active != "target":
                if time.monotonic() - forced_at > 15:
                    break
                await asyncio.sleep(0.1)
            results["switch_seconds_after_force"] = time.monotonic() - forced_at
            results["active_after_force"] = swapper.active
            results["record"] = swapper.history[-1] if swapper.history else None
            results["flag_after"] = swapper.force_requested
            async with session.post(base + "/force", json={"model": "nope"}) as response:
                results["unknown_model"] = response.status
            async with session.post(base + "/force", json={"model": "third"}) as response:
                results["force_model"] = response.status
            begun = time.monotonic()
            while swapper.active != "third" and time.monotonic() - begun < 15:
                await asyncio.sleep(0.1)
            results["third_seconds"] = time.monotonic() - begun
            results["third_record"] = swapper.history[-1] if swapper.history else None
        await control_runner.cleanup()
        await engine_runner.cleanup()
        return results


def main():
    r = asyncio.run(scenario())
    assert r["idle_force"] == 409, r
    assert r["activate"] == 202, r
    assert r["draining"] and r["draining"]["model"] == "old" and r["draining"]["live_requests"] == 1, r
    assert r["still_old"], "the old engine stopped before Switch now although its reply was still live"
    assert r["force"] == 202, r
    assert r["active_after_force"] == "target" and r["switch_seconds_after_force"] < 10, r
    assert r["record"]["ok"] and r["record"]["forced"] is True and r["record"]["to"] == "target", r["record"]
    assert r["flag_after"] is False, r
    assert r["unknown_model"] == 404, r
    assert r["force_model"] == 202 and r["third_seconds"] < 10, r
    assert r["third_record"] and r["third_record"]["to"] == "third" and r["third_record"]["forced"] is True, r["third_record"]
    print("PASS test_model_swap_force: Switch now cut a live drain within seconds and recorded it; refused with nothing waiting; forced a direct switch to a named model")


main()
