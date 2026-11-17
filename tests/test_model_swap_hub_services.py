#!/usr/bin/env python3
import asyncio
import json
import os
import pathlib
import shlex
import sys
import tempfile

from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "serving"))
sys.path.insert(0, str(ROOT / "tools"))
import lane_config  # noqa: E402
import model_swap  # noqa: E402

GIB = 2 ** 30
FAKE_SSH = """import os, sys
host, command = sys.argv[1], sys.argv[2]
words = command.split()
if words[0] == "PROBE":
    print(200 * 2 ** 30)
elif words[0] == "READY":
    print("READY")
elif words[0] in ("START", "STOP"):
    with open(os.environ["SWAP_ORDER"], "a") as order:
        order.write(command + "\\n")
"""


def model(name, port, state, hub_start=None):
    spec = {"id": name, "kv_label": name + "_stage", "node_memory_bytes": 10 * GIB, "node_reclaim": "RECLAIM " + name,
            "node_start": f"START {name}", "node_ready": "READY {run_id}", "node_stop": f"STOP {name}",
            "api_start": "true", "api_stop": "true", "health": f"http://127.0.0.1:{port}/{name}",
            "hub_stop": f"echo HUBSTOP {name} >> {state}/order"}
    spec["hub_start"] = hub_start or f"echo HUBSTART {name} >> {state}/order"
    return spec


async def scenario(hub_start):
    with tempfile.TemporaryDirectory() as state:
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        os.environ["SWAP_ORDER"] = os.path.join(state, "order")
        started = []

        async def health(request):
            name = request.match_info["name"]
            order = pathlib.Path(state, "order")
            connected = name == "target" and order.exists() and "START target" in order.read_text().replace("HUBSTART target", "")
            started.append(connected)
            return web.json_response({"connected_ranks": 1 if connected else 0, "live_requests": 0})

        app = web.Application()
        app.router.add_get("/{name}", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        swapper = model_swap.Swapper({
            "nodes": ["n0"], "ssh": [sys.executable, os.path.join(state, "fake_ssh.py")], "drop_kv": "DROP {label}",
            "memory_probe": "PROBE", "memory_margin_bytes": GIB,
            "ready_seconds": 5, "api_seconds": 2, "drain_seconds": 1, "node_seconds": 10,
            "models": [model("target", port, state, hub_start), model("other", port, state)],
        })
        await swapper.startup(None)
        await swapper.swap("target")
        await runner.cleanup()
        await swapper.cleanup(None)
        path = pathlib.Path(state, "order")
        return swapper.state, swapper.error, path.read_text().split("\n") if path.exists() else []


def lane_records_match_the_swap():
    swap = {spec["id"]: spec for spec in json.loads((ROOT / "deployment/hub/model_swap.json").read_text())["models"]}
    lanes = json.loads((ROOT / "deployment/lanes.json").read_text())
    lanes = lanes.get("lanes", lanes)
    records = lanes if isinstance(lanes, dict) else {record["id"]: record for record in lanes}
    checked = 0
    for record_id, record in records.items():
        if not record.get("draftd"):
            continue
        spec = swap[record["draft_server"]["target_model"]]
        source, environment = lane_config.draftd_environment(record_id)
        for key, value in environment.items():
            assert f"{key}={shlex.quote(value)}" in spec["hub_start"], f"{record_id}: {key} in hub_start differs from the lane record"
        assert f"cd {source} " in spec["hub_start"] and f"cd {source} " in spec["hub_stop"], f"{record_id}: the swap runs another draft server source"
        assert f"DRAFTD_UNIT={environment['DRAFTD_UNIT']} " in spec["hub_stop"], f"{record_id}: hub_stop names another unit"
        checked += 1
    assert checked >= 2, "the records with a draft server were not found"


def production_drafters_record_nothing():
    for spec in json.loads((ROOT / "deployment/hub/model_swap.json").read_text())["models"]:
        for key, command in spec.items():
            if isinstance(command, str):
                assert "DRAFTD_RECORD_DIR" not in command and "--record-dir" not in command, f"{spec['id']}: {key} records users' tap streams; record only hold corpora, outside the production swap"


def main():
    status, error, order = asyncio.run(scenario(None))
    assert status == "idle", error
    assert order.index("HUBSTOP other") < order.index("HUBSTART target"), order
    assert "HUBSTART target" in order and "START target" in order, order
    status, error, order = asyncio.run(scenario("echo refused; exit 7"))
    assert status == "failed" and "target hub service start exit 7: refused" in error, error
    lane_records_match_the_swap()
    production_drafters_record_nothing()
    print("PASS test_model_swap_hub_services: a model's drafter stops with the model, starts with its engines, a start that fails fails the swap, and no production drafter records users' tap streams")


if __name__ == "__main__":
    main()
