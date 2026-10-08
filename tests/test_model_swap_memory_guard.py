#!/usr/bin/env python3
import asyncio
import json
import os
import pathlib
import sys
import tempfile

from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "serving"))
import model_swap  # noqa: E402

GIB = 2 ** 30
FAKE_SSH = """import json, os, sys
host, command = sys.argv[1], sys.argv[2]
state = os.environ["SWAP_TEST_STATE"]
with open(os.path.join(state, "log"), "a") as handle:
    handle.write(json.dumps([host, command]) + "\\n")
if command == "PROBE":
    memory = json.load(open(os.path.join(state, "memory.json")))
    started = os.path.exists(os.path.join(state, "started-" + host))
    reclaimed = os.path.exists(os.path.join(state, "reclaimed-" + host))
    print(memory[host]["after" if started else "reclaimed" if reclaimed and "reclaimed" in memory[host] else "before"])
elif command.startswith("RECLAIM"):
    open(os.path.join(state, "reclaimed-" + host), "w").close()
elif command.startswith("START"):
    open(os.path.join(state, "started-" + host), "w").close()
elif command.startswith("READY"):
    print("READY")
"""


class RequestStub:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


def config(state, port):
    return {
        "nodes": ["n0", "n1", "n2"],
        "ssh": [sys.executable, os.path.join(state, "fake_ssh.py")],
        "drop_kv": "DROP {label}",
        "memory_probe": "PROBE",
        "memory_margin_bytes": 6 * GIB,
        "ready_seconds": 10, "api_seconds": 10, "drain_seconds": 1, "node_seconds": 10,
        "models": [{
            "id": "small", "kv_label": "small_stage", "node_memory_bytes": 30 * GIB, "node_reclaim": "RECLAIM small",
            "node_start": "START small", "node_ready": "READY {run_id}", "node_stop": "STOP small",
            "api_start": "true", "api_stop": "true", "health": f"http://127.0.0.1:{port}/small",
        }, {
            "id": "big", "kv_label": "big_stage", "node_memory_bytes": 100 * GIB, "node_reclaim": "RECLAIM big",
            "node_start": "START {rank}", "node_ready": "READY {run_id}", "node_stop": "STOP",
            "api_start": "true", "api_stop": "true", "health": f"http://127.0.0.1:{port}/health",
        }],
    }


async def scenario(memory, expect_ok):
    with tempfile.TemporaryDirectory() as state:
        os.environ["SWAP_TEST_STATE"] = state
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        pathlib.Path(state, "memory.json").write_text(json.dumps(memory))
        live = {"up": False}

        async def health(request):
            return web.json_response({"connected_ranks": 3 if live["up"] else 0})

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_get("/small", lambda request: web.json_response({"connected_ranks": 0}))
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        swapper = model_swap.Swapper(config(state, port))
        await swapper.startup(None)

        async def connect_after_start():
            while not any(pathlib.Path(state).glob("started-*")):
                await asyncio.sleep(0.05)
            live["up"] = True

        waiter = asyncio.ensure_future(connect_after_start())
        await swapper.swap("big")
        waiter.cancel()
        if swapper.active == "big":
            for started_marker in pathlib.Path(state).glob("started-*"):
                started_marker.unlink()
            live["up"] = False
            waiter = asyncio.ensure_future(connect_after_start())
            response = await swapper.activate(RequestStub({"model": "big"}))
            await swapper.task
            waiter.cancel()
            swapper.restarted = response.status == 202 and swapper.active == "big" and len(list(pathlib.Path(state).glob("started-*"))) == 3
        log = [json.loads(line) for line in pathlib.Path(state, "log").read_text().splitlines()]
        await swapper.cleanup(None)
        await runner.cleanup()
        started = sorted({host for host, command in log if command.startswith("START")})
        reclaims = sorted({command for host, command in log if command.startswith("RECLAIM")})
        return swapper, started, reclaims


def main():
    checks = 0

    def check(name, condition):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError("FAILED: " + name)
        print("ok   " + name)

    roomy = {"n0": {"before": 115 * GIB, "after": 13 * GIB}, "n1": {"before": 112 * GIB, "after": 9 * GIB}, "n2": {"before": 114 * GIB, "after": 12 * GIB}}
    swapper, started, reclaims = asyncio.run(scenario(roomy, True))
    check("a model whose projection plus margin fits every node starts", swapper.state == "idle" and swapper.active == "big" and sorted(started) == ["n0", "n1", "n2"])
    check("cold arenas stay warm when every node already has room", reclaims == [])
    check("requesting the active model after its engines died starts it again", getattr(swapper, "restarted", False))
    record = swapper.history[-1]
    check("the swap records the memory each node actually used", record["memory"]["used_gib_max"] == 103.0 and record["memory"]["used_gib_min"] == 102.0)
    check("the swap records the tightest node after the load", record["memory"]["tightest"] == "n1" and record["memory"]["available_gib_min"] == 9.0)

    reclaimable = dict(roomy, n1={"before": 80 * GIB, "reclaimed": 112 * GIB, "after": 9 * GIB})
    swapper, started, reclaims = asyncio.run(scenario(reclaimable, True))
    check("a short node first releases the other models' cold arenas, then the model starts",
          swapper.state == "idle" and len(started) == 3 and reclaims == ["RECLAIM small"])
    short = dict(roomy, n1={"before": 80 * GIB, "reclaimed": 105 * GIB, "after": 1 * GIB})
    swapper, started, reclaims = asyncio.run(scenario(short, False))
    check("a node still below projection plus margin after the release refuses the start on every node",
          swapper.state == "failed" and started == [] and swapper.active is None and reclaims == ["RECLAIM small"])
    check("the refusal names the short node and its available memory", "n1 105.0" in swapper.error and "n0" not in swapper.error.split("GiB available:")[-1])

    exact = dict(roomy, n2={"before": 106 * GIB, "after": 4 * GIB})
    swapper, started, _ = asyncio.run(scenario(exact, True))
    check("projection plus margin exactly available is admitted", swapper.state == "idle" and len(started) == 3)

    garbled = dict(roomy, n0={"before": "unknown", "after": "unknown"})
    swapper, started, _ = asyncio.run(scenario(garbled, False))
    check("a probe that prints no byte count fails the swap before any start", swapper.state == "failed" and started == [] and "n0" in swapper.error)

    try:
        model_swap.Swapper({key: value for key, value in config("/tmp", 1).items() if key != "memory_margin_bytes"})
        check("a config without a memory margin is refused", False)
    except SystemExit:
        check("a config without a memory margin is refused", True)
    broken = config("/tmp", 1)
    del broken["models"][1]["node_memory_bytes"]
    try:
        model_swap.Swapper(broken)
        check("a model without a memory projection is refused", False)
    except SystemExit:
        check("a model without a memory projection is refused", True)
    broken = config("/tmp", 1)
    del broken["models"][0]["node_reclaim"]
    try:
        model_swap.Swapper(broken)
        check("a model without a cold arena release command is refused", False)
    except SystemExit:
        check("a model without a cold arena release command is refused", True)
    print(f"PASS {checks} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
