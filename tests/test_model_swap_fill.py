#!/usr/bin/env python3
"""POST /fill stops the decode lane, streams the prompt to the last checkpoint on the fill lane, and reloads the decode lane."""
import asyncio
import json
import os
import pathlib
import sys
import tempfile

import aiohttp
from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "serving"))
import model_swap  # noqa: E402

GIB = 2 ** 30
FAKE_SSH = """import os, re, sys
for command in re.findall(r"[(] (.+?) [)]", sys.argv[2]) or [sys.argv[2]]:
    words = command.split()
    with open(os.path.join(os.path.dirname(sys.argv[0]), "ssh.log"), "a") as log:
        log.write(sys.argv[1] + " " + command + "\\n")
    if words[0] == "PROBE":
        print(200 * 2 ** 30)
    elif words[0] == "READY":
        print("READY")
"""


async def scenario(fail_fill=False):
    with tempfile.TemporaryDirectory() as state:
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        order = pathlib.Path(state, "api")
        up = pathlib.Path(state, "up")
        up.touch()
        received = []
        live = [1, 1]

        async def health(request):
            return web.json_response({"connected_ranks": 2 if up.exists() else 0, "live_requests": live.pop(0) if live else 0, "idle_ms": 0})

        async def completions(request):
            body = await request.json()
            received.append(body)
            if fail_fill:
                return web.Response(status=500, text="streamed fill failed")
            return web.json_response({"usage": {}})

        engines = web.Application(client_max_size=2 ** 26)
        engines.router.add_get("/k3/health", health)
        engines.router.add_post("/k3/v1/completions", completions)
        runner = web.AppRunner(engines)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        entry = {"id": "k3", "kv_label": "k3_stage", "node_memory_bytes": 10 * GIB, "node_reclaim": "RECLAIM k3", "node_start": "START decode", "node_ready": "READY {run_id}",
                 "node_stop": "STOP k3", "node_warm": "WARM k3", "api_start": f"echo decode >> {order}; touch {up}", "api_stop": f"rm -f {up}",
                 "hub_start": f"echo hub >> {order}", "hub_stop": f"echo hubstop >> {order}", "health": f"http://127.0.0.1:{port}/k3/health",
                 "fill": {"checkpoint_tokens": 16, "node_start": "START fill", "api_start": f"echo fill >> {order}; touch {up}"}}
        swapper = model_swap.Swapper({
            "nodes": ["n0", "n1"], "ssh": [sys.executable, os.path.join(state, "fake_ssh.py")], "drop_kv": "DROP {label}", "memory_probe": "PROBE", "memory_margin_bytes": GIB,
            "ready_seconds": 5, "api_seconds": 5, "drain_seconds": 20, "drain_quiet_seconds": 30, "node_seconds": 10, "models": [entry],
        })
        control = web.AppRunner(model_swap.build(swapper, "secret"))
        await control.setup()
        control_site = web.TCPSite(control, "127.0.0.1", 0)
        await control_site.start()
        base = f"http://127.0.0.1:{control_site._server.sockets[0].getsockname()[1]}"
        async with aiohttp.ClientSession(headers={"Authorization": "Bearer secret"}) as session:
            async with session.post(base + "/fill", json={"model": "k3", "prompt_token_ids": [1, 2]}) as response:
                short = response.status
            async with session.post(base + "/fill", json={"model": "other", "prompt_token_ids": [1] * 40}) as response:
                unknown = response.status
            async with session.post(base + "/fill", json={"model": "k3", "prompt_token_ids": list(range(58))}) as response:
                status, record = response.status, await response.json()
        log = [line.split(" ", 1)[1] for line in pathlib.Path(state, "ssh.log").read_text().splitlines() if line.startswith("n0 ")]
        api = order.read_text().split() if order.exists() else []
        assert not live and record["start_s"] < 10, (live, record)
        result = (short, unknown, status, record, received, log, api, swapper.state, swapper.active)
        await control.cleanup()
        await runner.cleanup()
        return result


def main():
    short, unknown, status, record, received, log, api, state, active = asyncio.run(scenario())
    assert short == 400 and unknown == 400, (short, unknown)
    assert status == 200 and record["ok"] and record["tokens"] == 48, record
    assert len(received) == 1 and received[0]["prompt_token_ids"] == list(range(48)) and received[0]["max_tokens"] == 1, received
    walk = [line for line in log if line.split()[0] in ("START", "STOP", "RECLAIM", "WARM")]
    assert walk == ["STOP k3", "RECLAIM k3", "START fill", "STOP k3", "RECLAIM k3", "START decode", "WARM k3"], walk
    assert api == ["hubstop", "fill", "hub", "decode"], api
    assert state == "idle" and active == "k3", (state, active)
    short, unknown, status, record, received, log, api, state, active = asyncio.run(scenario(fail_fill=True))
    assert status == 502 and not record["ok"] and "streamed fill answered 500" in record["error"], record
    assert state == "failed", state
    print("PASS test_model_swap_fill: the prompt fills to its last checkpoint on the fill lane, then the decode lane reloads; the stop waits for live replies but not the quiet window of a model switch; a failed fill fails loudly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
