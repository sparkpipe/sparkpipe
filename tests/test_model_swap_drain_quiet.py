#!/usr/bin/env python3
"""A model switch waits for the old model's replies to finish: the controller stops the old engine only after it has had no live request for the whole quiet window, so the short idle gap between the requests of one tool-calling reply does not let a switch cut that reply."""
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

GIB = 2 ** 30
FAKE_SSH = """import sys
words = sys.argv[2].split()
if words[0] == "PROBE":
    print(200 * 2 ** 30)
elif words[0] == "READY":
    print("READY")
"""
LIVE = [1, 1, 0, 1, 1, 0, 1]


def model(name, port, state):
    return {"id": name, "kv_label": name + "_stage", "node_memory_bytes": 10 * GIB, "node_reclaim": "RECLAIM " + name,
            "node_start": f"START {name}", "node_ready": "READY {run_id}", "node_stop": f"STOP {name}",
            "api_start": "true", "api_stop": f"touch {state}/api-stopped-{name}", "health": f"http://127.0.0.1:{port}/{name}"}


async def scenario():
    with tempfile.TemporaryDirectory() as state:
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        polls = []
        target_up = {"ready": False}

        async def health(request):
            name = request.match_info["name"]
            if name == "target":
                return web.json_response({"connected_ranks": 2 if target_up["ready"] else 0, "live_requests": 0})
            live = LIVE[len(polls)] if len(polls) < len(LIVE) else 0
            polls.append((time.monotonic(), live, pathlib.Path(state, "api-stopped-old").exists()))
            return web.json_response({"connected_ranks": 2, "live_requests": live})

        app = web.Application()
        app.router.add_get("/{name}", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        swapper = model_swap.Swapper({
            "nodes": ["n0", "n1"], "ssh": [sys.executable, os.path.join(state, "fake_ssh.py")], "drop_kv": "DROP {label}",
            "memory_probe": "PROBE", "memory_margin_bytes": GIB,
            "ready_seconds": 5, "api_seconds": 5, "drain_seconds": 30, "drain_quiet_seconds": 2.5, "node_seconds": 10,
            "models": [model("old", port, state), model("target", port, state)],
        })
        await swapper.startup(None)
        assert swapper.active == "old", swapper.active
        original_start = swapper.start

        async def start(spec):
            target_up["ready"] = True
            await original_start(spec)

        swapper.start = start
        began = time.monotonic()
        await swapper.swap("target")
        stopped = pathlib.Path(state, "api-stopped-old").exists()
        result = (swapper.state, swapper.error, polls, began, stopped)
        await swapper.cleanup(None)
        await runner.cleanup()
        return result


def main():
    state, error, polls, began, stopped = asyncio.run(scenario())
    assert state == "idle" and error is None, (state, error)
    assert stopped, "the old engine was never stopped"
    assert len(polls) >= len(LIVE) + 3, f"the controller stopped the old engine after only {len(polls)} health polls: {polls}"
    last_busy = max(index for index, (_, live, _) in enumerate(polls) if live)
    quiet_polls = polls[last_busy + 1:]
    assert quiet_polls[-1][0] - quiet_polls[0][0] >= 2.0, f"the old engine was quiet for only {quiet_polls[-1][0] - quiet_polls[0][0]:.1f} s before the stop: {polls}"
    assert not any(stopped_then for _, _, stopped_then in polls), "the old engine was stopped while the drain was still polling it"
    print("PASS test_model_swap_drain_quiet: the switch waited through a one-poll idle gap and stopped the old engine only after a full quiet window")


main()
