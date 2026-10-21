#!/usr/bin/env python3
import asyncio
import glob
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
FAKE_SSH = """import glob, json, os, sys
host, command = sys.argv[1], sys.argv[2]
state = os.environ["SWAP_TEST_STATE"]
with open(os.path.join(state, "log"), "a") as handle:
    handle.write(json.dumps([host, command]) + "\\n")
words = command.split()
if words[0] == "PROBE":
    print((100 - 30 * len(glob.glob(os.path.join(state, "up-*-" + host)))) * 2 ** 30)
elif words[0] == "START":
    open(os.path.join(state, "holder-" + host + "-" + words[2] + "-" + words[1]), "w").close()
    open(os.path.join(state, "up-" + words[1] + "-" + host), "w").close()
elif words[0] == "STOP":
    for path in glob.glob(os.path.join(state, "holder-" + host + "-*-" + words[1])) + glob.glob(os.path.join(state, "up-" + words[1] + "-" + host)):
        os.remove(path)
elif words[0] == "DROP":
    holders = glob.glob(os.path.join(state, "holder-" + host + "-" + words[1] + "-*"))
    if holders:
        print("pool " + words[1] + " still attached by " + os.path.basename(holders[0]).split("-")[-1])
        sys.exit(1)
elif words[0] == "READY":
    print("READY")
"""
NODES = ["n0", "n1", "n2"]


def model(name, port, gib=10):
    return {"id": name, "kv_label": name + "_stage", "node_memory_bytes": gib * GIB, "node_reclaim": "RECLAIM " + name,
            "node_start": f"START {name} {name}_stage", "node_ready": "READY {run_id}", "node_stop": f"STOP {name}",
            "api_start": "true", "api_stop": "true", "health": f"http://127.0.0.1:{port}/{name}"}


class Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


async def scenario(up, targets, sets, extra=(), after=None):
    with tempfile.TemporaryDirectory() as state:
        os.environ["SWAP_TEST_STATE"] = state
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        for name in up:
            for host in NODES:
                pathlib.Path(state, f"holder-{host}-{name}_stage-{name}").touch()
                pathlib.Path(state, f"up-{name}-{host}").touch()

        async def health(request):
            name = request.match_info["name"]
            return web.json_response({"connected_ranks": len(glob.glob(os.path.join(state, f"up-{name}-*"))), "live_requests": 0})

        app = web.Application()
        app.router.add_get("/{name}", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        models = [model(name, port) for name in ("a", "b", "c", "k")] + [model(name, port, gib) for name, gib in extra]
        swapper = model_swap.Swapper({
            "nodes": NODES, "ssh": [sys.executable, os.path.join(state, "fake_ssh.py")], "drop_kv": "DROP {label}",
            "memory_probe": "PROBE", "memory_margin_bytes": GIB,
            "ready_seconds": 10, "api_seconds": 10, "drain_seconds": 1, "node_seconds": 10, "release_seconds": 1,
            "resident_sets": sets, "models": models,
        })
        await swapper.startup(None)
        outcomes = []
        for target in targets:
            pathlib.Path(state, "log").write_text("")
            await swapper.swap(target)
            log = [json.loads(line) for line in pathlib.Path(state, "log").read_text().splitlines()]
            running = sorted({os.path.basename(path).split("-")[1] for path in glob.glob(os.path.join(state, "up-*"))})
            outcomes.append({"active": swapper.active, "resident": list(swapper.resident), "state": swapper.state, "error": swapper.error,
                             "running": running, "stopped": sorted({command.split()[1] for host, command in log if command.startswith("STOP")}),
                             "dropped": sorted({command.split()[1] for host, command in log if command.startswith("DROP")}),
                             "kept": swapper.history[-1]["kept"]})
        if after is not None:
            outcomes[-1]["after"] = await after(swapper)
        await swapper.cleanup(None)
        await runner.cleanup()
        return outcomes


def main():
    checks = 0

    def check(name, condition):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError("FAILED: " + name)
        print("ok   " + name)

    steps = asyncio.run(scenario(["a"], ["b", "c", "a", "k", "a"], [["a", "b", "c"]]))
    check("starting a set member keeps the resident member running",
          steps[0]["state"] == "idle" and steps[0]["active"] == "b" and steps[0]["running"] == ["a", "b"] and steps[0]["resident"] == ["a", "b"])
    check("a kept member is never stopped and its KV pool never released",
          "a" not in steps[0]["stopped"] and "a_stage" not in steps[0]["dropped"] and steps[0]["kept"] == ["a"])
    check("a third member joins the first two", steps[1]["running"] == ["a", "b", "c"] and steps[1]["resident"] == ["a", "b", "c"])
    check("activating a resident member only moves the active model",
          steps[2]["active"] == "a" and steps[2]["running"] == ["a", "b", "c"] and steps[2]["stopped"] == [] and steps[2]["dropped"] == [])
    check("a model outside the set stops every member and releases their pools",
          steps[3]["running"] == ["k"] and steps[3]["resident"] == ["k"] and {"a", "b", "c"} <= set(steps[3]["stopped"])
          and {"a_stage", "b_stage", "c_stage"} <= set(steps[3]["dropped"]))
    check("returning to a member stops the outsider", steps[4]["running"] == ["a"] and steps[4]["resident"] == ["a"])

    step, = asyncio.run(scenario(["a"], ["b"], []))
    check("without resident sets a swap still stops the running model", step["running"] == ["b"] and step["resident"] == ["b"] and "a" in step["stopped"])

    async def switch_to_b(swapper):
        response = await swapper.activate(Request({"model": "b"}))
        return {"http": response.status, "active": swapper.active, "state": swapper.state, "error": swapper.error,
                "instant": swapper.history[-1].get("instant")}

    step, = asyncio.run(scenario(["a", "b"], ["big"], [["a", "b", "big"]], extra=[("big", 60)], after=switch_to_b))
    check("a member that does not fit beside the residents is refused and the residents keep running",
          step["state"] == "failed" and "beside resident a, b" in step["error"] and step["running"] == ["a", "b"] and step["resident"] == ["a", "b"])
    check("a refused member leaves a resident active", step["active"] == "a")
    check("activating a resident after a refused load answers at once and clears the failure",
          step["after"] == {"http": 200, "active": "b", "state": "idle", "error": None, "instant": True})

    try:
        model_swap.Swapper({"nodes": NODES, "ssh": ["true"], "drop_kv": "DROP {label}", "memory_probe": "PROBE", "memory_margin_bytes": GIB,
                            "resident_sets": [["a", "b"]], "models": [dict(model("a", 1), kv_label="same"), dict(model("b", 1), kv_label="same")]})
        refused = False
    except SystemExit:
        refused = True
    check("a resident set whose members share a KV pool label is refused at startup", refused)
    print(f"PASS {checks} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
