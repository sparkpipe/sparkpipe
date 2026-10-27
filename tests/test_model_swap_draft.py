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
FAKE_SSH = """import os, sys, time
host, command = sys.argv[1], sys.argv[2]
words = command.split()
if words[0] == "PROBE":
    print(200 * 2 ** 30)
elif words[0] == "READY":
    print("READY")
elif words[0] == "START":
    time.sleep(0.3)
"""


def model(name, port, state, draft_start):
    spec = {"id": name, "kv_label": name + "_stage", "node_memory_bytes": 10 * GIB, "node_reclaim": "RECLAIM " + name,
            "node_start": f"START {name}", "node_ready": "READY {run_id}", "node_stop": f"STOP {name}",
            "api_start": f"rm -f {state}/api-stopped-{name}; if test -f {state}/draft-up-{name}; then touch {state}/api-started-{name}; else touch {state}/api-started-early-{name}; fi",
            "api_stop": f"touch {state}/api-stopped-{name}", "health": f"http://127.0.0.1:{port}/{name}"}
    if draft_start:
        spec["draft_start"] = draft_start.format(state=state, name=name)
        spec["draft_stop"] = f"rm -f {state}/draft-up-{name} && touch {state}/draft-stopped-{name}"
    return spec


async def scenario(draft_start):
    with tempfile.TemporaryDirectory() as state:
        pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH)
        pathlib.Path(state, "draft-up-other").touch()

        async def health(request):
            name = request.match_info["name"]
            started = pathlib.Path(state, f"api-started-{name}").exists() or pathlib.Path(state, f"api-started-early-{name}").exists()
            up = started and not pathlib.Path(state, f"api-stopped-{name}").exists()
            return web.json_response({"connected_ranks": 2 if name == "target" and up else 0, "live_requests": 0})

        app = web.Application()
        app.router.add_get("/{name}", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        swapper = model_swap.Swapper({
            "nodes": ["n0", "n1"], "ssh": [sys.executable, os.path.join(state, "fake_ssh.py")], "drop_kv": "DROP {label}",
            "memory_probe": "PROBE", "memory_margin_bytes": GIB, "ready_seconds": 5, "api_seconds": 2, "drain_seconds": 1,
            "node_seconds": 10, "draft_seconds": 10,
            "models": [model("target", port, state, draft_start), model("other", port, state, "true")],
        })
        await swapper.startup(None)
        await swapper.swap("target")
        files = {path.name for path in pathlib.Path(state).iterdir()}
        result = (swapper.state, swapper.error, swapper.status()["draft"], files)
        await swapper.cleanup(None)
        await runner.cleanup()
        return result


def main():
    failures = []
    state, error, draft, files = asyncio.run(scenario("sleep 0.5 && touch {state}/draft-up-{name}"))
    if state != "idle" or error is not None:
        failures.append(f"a swap whose draft server starts serves (state {state!r}, error {error!r})")
    if "draft-stopped-other" not in files or "draft-up-other" in files:
        failures.append(f"the other model's draft server is stopped before the target starts (files {sorted(files)})")
    if "api-started-target" not in files or "api-started-early-target" in files or not draft or draft.get("ok") is not True or draft.get("model") != "target":
        failures.append(f"the target's API starts only after its draft server is up, and the status records it (draft {draft}, files {sorted(files)})")
    state, error, draft, files = asyncio.run(scenario("echo no gpu memory >&2; exit 3"))
    if state != "idle" or error is not None or "api-started-early-target" not in files:
        failures.append(f"a draft server that fails to start leaves the swap serving without model drafts (state {state!r}, error {error!r})")
    if not draft or draft.get("ok") is not False or "no gpu memory" not in draft.get("detail", ""):
        failures.append(f"a draft server that fails to start is reported in the status with its output (draft {draft})")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS model swap draft server: a model's draft server starts with its engines and is up before its API, other models' draft servers stop first, and a failed start serves without model drafts but shows in the status")


if __name__ == "__main__":
    main()
