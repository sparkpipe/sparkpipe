#!/usr/bin/env python3
"""The fleet gateway answers only callers with its token, reads only the configured logs, and runs a command only on a known host, with an approver named, after the shipped refusal rules: banned commands never reach a host, boot-path rules apply to sparks but not the hub, every run is audited, secrets in output are masked, long output is cut and a command past its timeout is stopped."""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("fleet_gateway", ROOT / "tools" / "serving" / "fleet_gateway.py")
fleet_gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fleet_gateway)
SHIPPED = json.loads((ROOT / "deployment" / "hub" / "fleet_gateway.json").read_text())
TOKEN = "t" * 40
FAKE_SSH = """import os, subprocess, sys
host, command = sys.argv[1], sys.argv[2]
with open(os.environ["FAKE_SSH_LOG"], "a") as log:
    log.write(host + "\\t" + command + "\\n")
env = dict(os.environ, HOME=os.path.join(os.environ["FAKE_HOMES"], host))
sys.exit(subprocess.call(["bash", "-c", command], env=env))
"""


async def serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


async def run(tmp):
    tmp = Path(tmp)
    (tmp / "token").write_text(TOKEN + "\n")
    (tmp / "fake_ssh.py").write_text(FAKE_SSH)
    (tmp / "chat.log").write_text("first\nsecond LITELLM_MASTER_KEY=abcdef123456\nthird sk-abcdefghijklmnopqrstuvwx\n")
    (tmp / "logs").mkdir()
    (tmp / "logs" / "k3-kv-api.log").write_text("api one\napi two\n")
    engine_root = tmp / "homes" / "spark1" / "k3-lanes" / "lane8" / "root"
    engine_root.mkdir(parents=True)
    (engine_root / "residentd.log").write_text("rank ready\ndecode ok\n")
    (tmp / "homes" / "spark2").mkdir(parents=True)
    (tmp / "swap.json").write_text(json.dumps({"models": [
        {"id": "kimi-k3", "node_start": "cd $HOME/k3-lanes/lane8/root && mkdir -p logs"},
        {"id": "glm-5.3", "node_start": "$HOME/stage-lanes/glm53_full/lane6/root/lane.sh start {run_id}"}]}))
    os.environ["FAKE_SSH_LOG"] = str(tmp / "ssh.log")
    os.environ["FAKE_HOMES"] = str(tmp / "homes")
    (tmp / "ssh.log").write_text("")

    async def swap_status(request):
        return web.json_response({"active": "kimi-k3", "state": "idle", "history": [{"from": "glm-5.3", "to": "kimi-k3", "ok": True}] * 7})

    control = web.Application()
    control.router.add_get("/status", swap_status)
    control_runner, control_url = await serve(control)
    config = {
        "token_file": str(tmp / "token"), "audit_log": str(tmp / "audit.jsonl"), "hub": "hub0",
        "nodes": ["spark1", "spark2"], "ssh": [sys.executable, str(tmp / "fake_ssh.py")],
        "status_url": control_url + "/status", "swap_config": str(tmp / "swap.json"),
        "logs_dir": str(tmp / "logs"), "logs": {"chat": str(tmp / "chat.log")},
        "max_output_chars": 400, "default_timeout_seconds": 30, "max_timeout_seconds": 60, "max_log_lines": 500,
        "refuse": SHIPPED["refuse"], "refuse_on_nodes": SHIPPED["refuse_on_nodes"],
    }
    gateway_runner, base = await serve(fleet_gateway.build(config))
    auth = {"Authorization": "Bearer " + TOKEN}
    try:
        async with aiohttp.ClientSession() as session:
            async def call(method, path, **kwargs):
                async with session.request(method, base + path, **kwargs) as response:
                    return response.status, await response.json(content_type=None)

            assert (await call("GET", "/v1/status"))[0] == 401
            assert (await call("GET", "/v1/status", headers={"Authorization": "Bearer wrong"}))[0] == 401

            status, body = await call("GET", "/v1/status", headers=auth)
            assert status == 200 and body["active"] == "kimi-k3" and len(body["recent_switches"]) == 5, body

            status, body = await call("GET", "/v1/logs", headers=auth, params={"name": "chat", "lines": "2"})
            assert status == 200 and body["text"].splitlines()[0].startswith("second"), body
            assert "abcdef123456" not in body["text"] and "sk-abcdefghij" not in body["text"] and "LITELLM_MASTER_KEY=[redacted]" in body["text"], body

            status, body = await call("GET", "/v1/logs", headers=auth, params={"name": "api:k3"})
            assert status == 200 and body["text"] == "api one\napi two", body
            for name in ("api:../../etc/passwd", "api:k3/../../x", "secrets", "../chat"):
                assert (await call("GET", "/v1/logs", headers=auth, params={"name": name}))[0] == 404, name

            status, body = await call("GET", "/v1/logs", headers=auth, params={"name": "engine", "node": "spark1", "lines": "1"})
            assert status == 200 and body["text"].strip() == "decode ok", body
            assert (await call("GET", "/v1/logs", headers=auth, params={"name": "engine", "node": "hub0"}))[0] == 400

            status, body = await call("POST", "/v1/run", headers=auth, json={"host": "elsewhere", "command": "uptime", "approved_by": "admin@x"})
            assert status == 400, body
            status, body = await call("POST", "/v1/run", headers=auth, json={"host": "spark1", "command": "uptime"})
            assert status == 400, body

            ssh_before = (tmp / "ssh.log").read_text()
            for host, command in (("spark1", "sync"), ("spark1", "pkill -f sf17_arm"), ("hub0", "cat ~/.env"),
                                  ("spark2", "sudo poweroff"), ("spark1", "crontab -l"), ("spark1", "rm -rf ~")):
                status, body = await call("POST", "/v1/run", headers=auth, json={"host": host, "command": command, "approved_by": "admin@x"})
                assert status == 403 and body["error"].startswith("refused:"), (host, command, body)
            assert (tmp / "ssh.log").read_text() == ssh_before, "a refused command reached a host"

            status, body = await call("POST", "/v1/run", headers=auth, json={"host": "hub0", "command": "crontab -l >/dev/null 2>&1; echo hub-ok", "approved_by": "admin@x"})
            assert status == 200 and "hub-ok" in body["output"], body

            status, body = await call("POST", "/v1/run", headers=auth, json={"host": "spark2", "command": "echo hello; echo GITHUB_TOKEN=abc123xyz", "approved_by": "admin@x"})
            assert status == 200 and body["exit"] == 0 and "hello" in body["output"] and "abc123xyz" not in body["output"], body
            assert (tmp / "ssh.log").read_text().endswith("spark2\techo hello; echo GITHUB_TOKEN=abc123xyz\n")

            status, body = await call("POST", "/v1/run", headers=auth, json={"host": "spark1", "command": "python3 -c \"print('x' * 5000)\"", "approved_by": "admin@x"})
            assert status == 200 and len(body["output"]) < 500 and "characters cut" in body["output"], len(body["output"])

            started = time.monotonic()
            status, body = await call("POST", "/v1/run", headers=auth, json={"host": "hub0", "command": "sleep 20", "timeout": 1, "approved_by": "admin@x"})
            assert status == 200 and body["timed_out"] is True and time.monotonic() - started < 10, body

            status, body = await call("GET", "/v1/nodes", headers=auth)
            assert status == 200 and set(body) == {"hub0", "spark1", "spark2"}, body

        audit = [json.loads(line) for line in (tmp / "audit.jsonl").read_text().splitlines()]
        refused = [entry for entry in audit if entry["event"] == "refused"]
        runs = [entry for entry in audit if entry["event"] == "run"]
        assert len(refused) == 6 and all(entry["approved_by"] == "admin@x" for entry in refused + runs), audit
        assert any(entry["host"] == "spark2" and entry["command"].startswith("echo hello") for entry in runs), runs
        assert sum(1 for entry in audit if entry["event"] == "done") == len(runs), audit
    finally:
        await gateway_runner.cleanup()
        await control_runner.cleanup()


with tempfile.TemporaryDirectory() as directory:
    asyncio.run(run(directory))
print("PASS test_fleet_gateway: token required, configured logs only, banned commands never reach a host, boot rules on sparks only, audited runs, masked secrets, cut output, timeouts stop commands")
