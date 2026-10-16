#!/usr/bin/env python3
"""The fleet gateway answers only callers with its token, reads only the configured logs, and runs a command only on a known host after the shipped refusal rules: as spec only with an approver named, in the sandbox only inside a chat's live grant (refused before the grant, after a revoke or expiry, for unsaved chat ids, and beyond the grant cap), with the shipped sandbox properties and this host's home and group filled in; banned commands never reach a host, boot-path rules apply to sparks but not the hub, every run is audited in a private log, secrets in output are masked, long output is cut, and a command past its timeout is stopped together with the processes it started, on the hub and on a spark."""
import asyncio
import grp
import importlib.util
import json
import os
import stat
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
CONTROLLER_TOKEN = "k" * 40
FAKE_SSH = "\n".join([
    "import os, subprocess, sys",
    "host, command = sys.argv[-2], sys.argv[-1]",
    "open(os.environ['FAKE_SSH_LOG'], 'a').write('\\t'.join(['ssh', host, command]) + '\\n')",
    "env = dict(os.environ, HOME=os.path.join(os.environ['FAKE_HOMES'], host))",
    "sys.exit(subprocess.call(['bash', '-c', command], env=env))",
])
FAKE_RUNNER = "\n".join([
    "import os, subprocess, sys",
    "args = sys.argv[1:]",
    "cut = args.index('--')",
    "open(os.environ['FAKE_SSH_LOG'], 'a').write('\\t'.join(['sandbox', os.environ.get('HOME', ''), '|'.join(args[:cut]), args[-1]]) + '\\n')",
    "sys.exit(subprocess.call(args[cut + 1:]))",
])


async def serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


async def run(tmp):
    tmp = Path(tmp)
    (tmp / "token").write_text(TOKEN + "\n")
    (tmp / "token").chmod(0o600)
    (tmp / "controller-token").write_text(CONTROLLER_TOKEN + "\n")
    (tmp / "controller-token").chmod(0o600)
    (tmp / "wrong-controller-token").write_text("w" * 40)
    (tmp / "wrong-controller-token").chmod(0o600)
    (tmp / "shared-token").write_text(TOKEN)
    (tmp / "shared-token").chmod(0o644)
    (tmp / "fake_ssh.py").write_text(FAKE_SSH)
    (tmp / "fake_runner.py").write_text(FAKE_RUNNER)
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

    presented = []

    async def swap_status(request):
        presented.append(request.headers.get("Authorization", ""))
        if presented[-1] != "Bearer " + CONTROLLER_TOKEN:
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response({"active": "kimi-k3", "state": "idle", "history": [{"from": "glm-5.3", "to": "kimi-k3", "ok": True}] * 7})

    control = web.Application()
    control.router.add_get("/status", swap_status)
    control_runner, control_url = await serve(control)
    config = {
        "token_file": str(tmp / "token"), "controller_token_file": str(tmp / "controller-token"), "audit_log": str(tmp / "audit.jsonl"), "hub": "hub0",
        "nodes": ["spark1", "spark2"], "ssh": [sys.executable, str(tmp / "fake_ssh.py")],
        "status_url": control_url + "/status", "swap_config": str(tmp / "swap.json"),
        "logs_dir": str(tmp / "logs"), "logs": {"chat": str(tmp / "chat.log")},
        "max_output_chars": 400, "default_timeout_seconds": 30, "max_timeout_seconds": 60, "max_log_lines": 500,
        "refuse": SHIPPED["refuse"], "refuse_on_nodes": SHIPPED["refuse_on_nodes"],
        "sandbox_runner": [sys.executable, str(tmp / "fake_runner.py")], "sandbox_properties": SHIPPED["sandbox_properties"],
        "grant_hours": 0.0015, "max_grants": 3,
    }
    for broken in ({"token_file": str(tmp / "shared-token")}, {"controller_token_file": str(tmp / "shared-token")}, {"controller_token_file": None}):
        trial = {key: value for key, value in dict(config, **broken).items() if value is not None}
        try:
            fleet_gateway.build(trial)
            raise AssertionError(f"the gateway started with {broken}")
        except SystemExit:
            pass
    gateway_runner, base = await serve(fleet_gateway.build(config))
    wrong_runner, wrong_base = await serve(fleet_gateway.build(dict(config, controller_token_file=str(tmp / "wrong-controller-token"))))
    auth = {"Authorization": "Bearer " + TOKEN}
    log = tmp / "ssh.log"
    chat = "0c9a2f1e-5b7d-4e2a-9f10-3a6b8c1d2e4f"
    try:
        async with aiohttp.ClientSession() as session:
            async def call(method, path, **kwargs):
                async with session.request(method, base + path, **kwargs) as response:
                    return response.status, await response.json(content_type=None)

            assert (await call("GET", "/v1/status"))[0] == 401
            assert (await call("GET", "/v1/status", headers={"Authorization": "Bearer wrong"}))[0] == 401
            status, body = await call("GET", "/v1/status", headers=auth)
            assert status == 200 and body["active"] == "kimi-k3" and len(body["recent_switches"]) == 5, body
            assert presented == ["Bearer " + CONTROLLER_TOKEN], presented
            async with session.get(wrong_base + "/v1/status", headers=auth) as response:
                assert response.status == 503 and "401" in (await response.json())["error"], response.status
            async with session.get(wrong_base + "/v1/logs", headers=auth, params={"name": "engine", "node": "spark1"}) as response:
                assert response.status == 503, response.status
            assert TOKEN not in "".join(presented), presented

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

            assert (await call("POST", "/v1/run", headers=auth, json={"host": "elsewhere", "command": "uptime", "approved_by": "admin@x"}))[0] == 400
            assert (await call("POST", "/v1/run", headers=auth, json={"host": "spark1", "command": "uptime"}))[0] == 400
            before = log.read_text()
            for host, command in (("spark1", "sync"), ("spark1", "pkill -f sf17_arm"), ("hub0", "cat ~/.env"), ("hub0", "cat /srv/workspace/sparkpipe/swap-controller/token"),
                                  ("spark2", "sudo poweroff"), ("spark1", "crontab -l"), ("spark1", "rm -rf ~")):
                status, body = await call("POST", "/v1/run", headers=auth, json={"host": host, "command": command, "approved_by": "admin@x"})
                assert status == 403 and body["error"].startswith("refused:"), (host, command, body)
            assert log.read_text() == before, "a refused command reached a host"

            status, body = await call("POST", "/v1/run", headers=auth, json={"host": "hub0", "command": "crontab -l >/dev/null 2>&1; echo hub-ok", "approved_by": "admin@x"})
            assert status == 200 and "hub-ok" in body["output"] and body["as"] == "spec", body
            status, body = await call("POST", "/v1/run", headers=auth, json={"host": "spark2", "command": "echo hello; echo GITHUB_TOKEN=abc123xyz", "approved_by": "admin@x"})
            assert status == 200 and body["exit"] == 0 and "hello" in body["output"] and "abc123xyz" not in body["output"], body
            assert log.read_text().splitlines()[-1].startswith("ssh\tspark2\texec timeout -k 5 30 bash -lc "), log.read_text()[-200:]
            status, body = await call("POST", "/v1/run", headers=auth, json={"host": "spark1", "command": "python3 -c \"print('x' * 5000)\"", "approved_by": "admin@x"})
            assert status == 200 and len(body["output"]) < 500 and "characters cut" in body["output"], len(body["output"])

            for host in ("hub0", "spark2"):
                marker = tmp / f"late-{host}"
                started = time.monotonic()
                status, body = await call("POST", "/v1/run", headers=auth, json={"host": host, "command": f"( sleep 3; touch {marker} ) & echo started; sleep 30", "timeout": 1, "approved_by": "admin@x"})
                assert status == 200 and body["timed_out"] is True and time.monotonic() - started < 9, (host, body, time.monotonic() - started)
                await asyncio.sleep(4)
                assert not marker.exists(), f"a background process outlived the timeout on {host}"

            status, body = await call("GET", "/v1/nodes", headers=auth)
            assert status == 200 and set(body) == {"hub0", "spark1", "spark2"}, body

            sandbox = {"host": "spark1", "command": "echo sandbox-ok", "as": "sandbox", "chat_id": chat, "timeout": 20}
            assert (await call("POST", "/v1/run", headers=auth, json=sandbox))[0] == 403
            for bad in ("local:abc12345", "short", "../x/y/z/w"):
                assert (await call("POST", "/v1/grant", headers=auth, json={"chat_id": bad, "approved_by": "admin@x"}))[0] == 400, bad
            assert (await call("POST", "/v1/grant", headers=auth, json={"chat_id": chat}))[0] == 400
            status, body = await call("POST", "/v1/grant", headers=auth, json={"chat_id": chat, "approved_by": "admin@x"})
            assert status == 200 and body["expires_utc"], body
            status, body = await call("GET", "/v1/grant", headers=auth, params={"chat_id": chat})
            assert body["granted"] is True and body["ready"] is True, body
            assert (await call("GET", "/v1/grant", headers=auth, params={"chat_id": "another-chat-id"}))[1]["granted"] is False

            status, body = await call("POST", "/v1/run", headers=auth, json=sandbox)
            assert status == 200 and body["as"] == "sandbox" and "sandbox-ok" in body["output"], body
            kind, home, props, command = log.read_text().splitlines()[-1].split("\t")
            spark_home = str(tmp / "homes" / "spark1")
            assert kind == "sandbox" and home == spark_home and command == "echo sandbox-ok", (kind, home, command)
            props = props.split("|")
            for required in ("DynamicUser=yes", "IPAddressDeny=any", "RestrictAddressFamilies=AF_UNIX AF_NETLINK", "NoNewPrivileges=yes",
                             "ProtectSystem=strict", "ProtectHome=read-only", "PrivateDevices=yes", "RuntimeMaxSec=25"):
                assert required in props, (required, props)
            assert f"SupplementaryGroups={grp.getgrgid(os.getgid()).gr_name}" in props, props
            assert any(p.startswith(f"InaccessiblePaths=-{spark_home}/.ssh ") for p in props), props
            status, body = await call("POST", "/v1/run", headers=auth, json=dict(sandbox, host="hub0", command="echo hub-sandbox"))
            assert status == 200 and "hub-sandbox" in body["output"], body
            kind, _, props, command = log.read_text().splitlines()[-1].split("\t")
            assert kind == "sandbox" and command == "echo hub-sandbox" and f"InaccessiblePaths=-{Path.home()}/.ssh" in props, (kind, props)

            before = log.read_text()
            assert (await call("POST", "/v1/run", headers=auth, json=dict(sandbox, chat_id="another-chat-id")))[0] == 403
            assert (await call("POST", "/v1/run", headers=auth, json=dict(sandbox, command="sync")))[0] == 403
            assert (await call("POST", "/v1/run", headers=auth, json=dict(sandbox, **{"as": "root"})))[0] == 400
            for extra in ("chat-extra-0001", "chat-extra-0002"):
                assert (await call("POST", "/v1/grant", headers=auth, json={"chat_id": extra, "approved_by": "admin@x"}))[0] == 200
            assert (await call("POST", "/v1/grant", headers=auth, json={"chat_id": "chat-extra-0003", "approved_by": "admin@x"}))[0] == 429
            assert (await call("DELETE", "/v1/grant", headers=auth, params={"chat_id": chat}))[1]["revoked"] is True
            assert (await call("POST", "/v1/run", headers=auth, json=sandbox))[0] == 403
            assert (await call("POST", "/v1/grant", headers=auth, json={"chat_id": chat, "approved_by": "admin@x"}))[0] == 200
            await asyncio.sleep(6)
            assert (await call("POST", "/v1/run", headers=auth, json=sandbox))[0] == 403, "an expired grant still ran a command"
            assert log.read_text() == before, "a refused sandbox command reached a host"

        assert stat.S_IMODE((tmp / "audit.jsonl").stat().st_mode) == 0o600, oct((tmp / "audit.jsonl").stat().st_mode)
        audit = [json.loads(line) for line in (tmp / "audit.jsonl").read_text().splitlines()]
        refused = [entry for entry in audit if entry["event"] == "refused"]
        runs = [entry for entry in audit if entry["event"] == "run"]
        assert len(refused) == 8 and all(entry["approved_by"] == "admin@x" for entry in refused + runs), audit
        assert any(entry["as"] == "sandbox" and entry["chat_id"] == chat for entry in runs), runs
        assert any(entry["event"] == "grant" for entry in audit) and any(entry["event"] == "revoke" for entry in audit), audit
        assert sum(1 for entry in audit if entry["event"] == "done") == len(runs), audit
    finally:
        await gateway_runner.cleanup()
        await wrong_runner.cleanup()
        await control_runner.cleanup()
    hidden = next(prop for prop in SHIPPED["sandbox_properties"] if prop.startswith("InaccessiblePaths=")).split("=", 1)[1].split()
    for secret in (SHIPPED["token_file"], SHIPPED["controller_token_file"]):
        assert "-" + str(Path(secret).parent) in hidden, (secret, hidden)


with tempfile.TemporaryDirectory() as directory:
    asyncio.run(run(directory))
print("PASS test_fleet_gateway: token required, private token files only, the swap controller read with its own token (a refusal is a 503), the sandbox cannot see either token, configured logs only, sandbox only within a live chat grant with the shipped properties, banned commands never reach a host, boot rules on sparks only, private audit log, masked secrets, cut output, timeouts stop the whole command")
