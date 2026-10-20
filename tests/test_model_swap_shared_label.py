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
FAKE_SSH = """import glob, json, os, sys
host, command = sys.argv[1], sys.argv[2]
state = os.environ["SWAP_TEST_STATE"]
with open(os.path.join(state, "log"), "a") as handle:
    handle.write(json.dumps([host, command]) + "\\n")
words = command.split()
if words[0] == "PROBE":
    print(200 * 2 ** 30)
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


def model(name, label, port):
    return {"id": name, "kv_label": label, "node_memory_bytes": 10 * GIB, "node_reclaim": "RECLAIM " + name,
            "node_start": f"START {name} {label}", "node_ready": "READY {run_id}", "node_stop": f"STOP {name}",
            "api_start": "true", "api_stop": "true", "health": f"http://127.0.0.1:{port}/{name}"}


async def scenario(active, targets):
    with tempfile.TemporaryDirectory() as state:
        os.environ["SWAP_TEST_STATE"] = state
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        nodes = ["n0", "n1", "n2"]
        labels = {"flash": "shared_stage", "fin": "shared_stage", "other": "other_stage"}
        for host in nodes:
            pathlib.Path(state, f"holder-{host}-{labels[active]}-{active}").touch()
            pathlib.Path(state, f"up-{active}-{host}").touch()

        async def health(request):
            name = request.match_info["name"]
            return web.json_response({"connected_ranks": len(list(pathlib.Path(state).glob(f"up-{name}-*")))})

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
            "ready_seconds": 10, "api_seconds": 10, "drain_seconds": 1, "node_seconds": 10,
            "models": [model("flash", labels["flash"], port), model("fin", labels["fin"], port), model("other", labels["other"], port)],
        })
        await swapper.startup(None)
        outcomes = []
        for target in targets:
            pathlib.Path(state, "log").write_text("")
            await swapper.swap(target)
            log = [json.loads(line) for line in pathlib.Path(state, "log").read_text().splitlines()]
            outcomes.append((swapper.active, swapper.state, swapper.error, log))
        await swapper.cleanup(None)
        await runner.cleanup()
        return outcomes


def drops(log):
    return sorted(command for host, command in log if command.startswith("DROP"))


def main():
    checks = 0

    def check(name, condition):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError("FAILED: " + name)
        print("ok   " + name)

    (active, state, error, log), = asyncio.run(scenario("fin", ["other"]))
    check("leaving a model whose KV pool label an idle model listed before it shares succeeds", state == "idle" and active == "other" and error is None)
    check("a label shared by two stopped models is released once per node", drops(log) == ["DROP shared_stage"] * 3)
    last_stop = max(index for index, (host, command) in enumerate(log) if command.startswith("STOP"))
    first_drop = min(index for index, (host, command) in enumerate(log) if command.startswith("DROP"))
    check("every engine stops before any KV pool is released", last_stop < first_drop)

    outcomes = asyncio.run(scenario("other", ["fin", "flash", "other"]))
    check("swaps between two models sharing a label and a third model all succeed",
          [(active, state) for active, state, error, log in outcomes] == [("fin", "idle"), ("flash", "idle"), ("other", "idle")])
    check("switching to a model releases every other label, including one the target shares",
          drops(outcomes[0][3]) == ["DROP other_stage"] * 3 + ["DROP shared_stage"] * 3
          and drops(outcomes[1][3]) == ["DROP other_stage"] * 3 + ["DROP shared_stage"] * 3)
    print(f"PASS {checks} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
