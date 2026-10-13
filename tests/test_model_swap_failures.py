#!/usr/bin/env python3
import asyncio
import os
import pathlib
import sys
import tempfile

from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "serving"))
import model_swap  # noqa: E402

GIB = 2 ** 30
FAKE_SSH = """import os, sys
host, command = sys.argv[1], sys.argv[2]
words = command.split()
if words[0] == "PROBE":
    print(200 * 2 ** 30)
elif words[0] == "READY":
    print("READY")
"""


def model(name, port, state):
    return {"id": name, "kv_label": name + "_stage", "node_memory_bytes": 10 * GIB, "node_reclaim": "RECLAIM " + name,
            "node_start": f"START {name}", "node_ready": "READY {run_id}", "node_stop": f"STOP {name}",
            "api_start": "true", "api_stop": f"touch {state}/api-stopped-{name}", "health": f"http://127.0.0.1:{port}/{name}"}


async def scenario(break_start):
    with tempfile.TemporaryDirectory() as state:
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        nodes = ["n0", "n1"]
        polls = {"target": 0}
        observed = []

        async def health(request):
            name = request.match_info["name"]
            if name != "target":
                return web.json_response({"connected_ranks": 0, "live_requests": 0})
            live = max(0, 3 - polls["target"])
            polls["target"] += 1
            observed.append((pathlib.Path(state, "api-stopped-target").exists(), live))
            return web.json_response({"connected_ranks": 1, "live_requests": live})

        app = web.Application()
        app.router.add_get("/{name}", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        swapper = model_swap.Swapper({
            "nodes": nodes, "ssh": [sys.executable, os.path.join(state, "fake_ssh.py")], "drop_kv": "DROP {label}",
            "memory_probe": "PROBE", "memory_margin_bytes": GIB,
            "ready_seconds": 5, "api_seconds": 1, "drain_seconds": 10, "node_seconds": 10,
            "models": [model("target", port, state), model("other", port, state)],
        })
        await swapper.startup(None)
        if break_start:
            async def broken(model):
                raise RuntimeError("engine start exploded")
            swapper.start = broken
        escaped = None
        try:
            await swapper.swap("target")
        except Exception as error:
            escaped = error
        result = (swapper.state, swapper.error, list(swapper.history), observed, escaped)
        await swapper.cleanup(None)
        await runner.cleanup()
        return result


def main():
    failures = []
    state, error, history, observed, escaped = asyncio.run(scenario(False))
    stopped_at = next((index for index, (stopped, live) in enumerate(observed) if stopped), None)
    if stopped_at is None or stopped_at == 0 or observed[stopped_at - 1][1] != 0:
        failures.append(f"a degraded target with live requests is drained before its API is stopped (health polls {observed})")
    state, error, history, observed, escaped = asyncio.run(scenario(True))
    if escaped is not None or state != "failed" or error is None or "RuntimeError" not in error or not history or history[-1]["ok"] is not False:
        failures.append(f"an unexpected exception during a swap leaves it failed with the error recorded (escaped {escaped!r}, state {state!r}, error {error!r})")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS model swap failures: a degraded target drains before its API stops, and an unexpected exception fails the swap with a history record")


if __name__ == "__main__":
    main()
