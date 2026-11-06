#!/usr/bin/env python3
"""Switch now is admin-only: the chat layer forwards a forced switch to the swap controller only after SparkPipe Chat confirms the caller's login is the admin, refuses a missing login with 401 and a non-admin login with 403 without touching the controller, and its status reports how many requests wait for each model. The chat layer presents the controller's token on every controller call and never passes it to SparkPipe Chat or a caller's login to the controller; there is no route that switches models without the admin; and it will not start with a control block but no private token file, or on an address other than loopback."""
import asyncio
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)


async def serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


CONTROLLER_TOKEN = "k" * 40


def refused(config, base):
    try:
        chat_frontend.Frontend(config, base)
    except SystemExit as error:
        return str(error)
    return None


async def run(tmp):
    forced = []
    checked = []
    presented = []
    token_file = tmp / "controller-token"
    token_file.write_text(CONTROLLER_TOKEN + "\n")
    token_file.chmod(0o600)

    async def auth(request):
        token = request.headers.get("Authorization", "")
        checked.append(token)
        if token == "Bearer admin-token":
            return web.json_response({"id": "a", "role": "admin"})
        if token == "Bearer user-token":
            return web.json_response({"id": "u", "role": "user"})
        return web.json_response({"detail": "Not authenticated"}, status=401)

    def controller_refuses(request):
        presented.append((request.path, request.headers.get("Authorization", "")))
        return request.headers.get("Authorization") != "Bearer " + CONTROLLER_TOKEN

    async def force(request):
        if controller_refuses(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        forced.append(await request.json())
        return web.json_response({"state": "swapping", "force_requested": True}, status=202)

    async def status(request):
        if controller_refuses(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response({"active": "m", "state": "swapping", "draining": {"model": "m", "live_requests": 1}})

    upstream = web.Application()
    upstream.router.add_get("/api/v1/auths/", auth)
    upstream.router.add_post("/force", force)
    upstream.router.add_get("/status", status)
    upstream_runner, base = await serve(upstream)

    control = {"status_url": base + "/status", "working_state_url": base + "/working-state", "force_url": base + "/force", "admin_url": base + "/api/v1/auths/"}
    frontend = chat_frontend.Frontend({"models": [], "controller_token_file": token_file.name, "control": control}, tmp)
    front_runner, front = await serve(chat_frontend.build(frontend))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(front + "/v1/sparkpipe/switch-now") as response:
                assert response.status == 401, response.status
            async with session.post(front + "/v1/sparkpipe/switch-now", headers={"Authorization": "Bearer user-token"}) as response:
                assert response.status == 403, response.status
            async with session.post(front + "/v1/sparkpipe/switch-now", headers={"Authorization": "Bearer stolen"}) as response:
                assert response.status == 403, response.status
            assert forced == [], forced
            async with session.post(front + "/v1/sparkpipe/switch-now", headers={"Authorization": "Bearer admin-token"}) as response:
                assert response.status == 202 and (await response.json())["force_requested"] is True, response.status
            assert forced == [{"account": "sparkpipe"}], forced
            async with session.post(front + "/v1/sparkpipe/switch-now", headers={"Authorization": "Bearer admin-token"}, json={"model": "not-served"}) as response:
                assert response.status == 404, response.status
            assert forced == [{"account": "sparkpipe"}], forced
            assert checked == ["Bearer user-token", "Bearer stolen", "Bearer admin-token", "Bearer admin-token"], checked
            async with session.get(front + "/v1/sparkpipe/status") as response:
                body = await response.json()
                assert body["draining"]["live_requests"] == 1 and body["waiting"] == {}, body
            assert presented and all(header == "Bearer " + CONTROLLER_TOKEN for _, header in presented), presented
            assert CONTROLLER_TOKEN not in "".join(checked), checked
            for headers in ({}, {"Authorization": "Bearer admin-token"}):
                async with session.post(front + "/v1/sparkpipe/activate", headers=headers, json={"model": "m"}) as response:
                    assert response.status in (404, 405), response.status
            assert forced == [{"account": "sparkpipe"}], forced
    finally:
        await front_runner.cleanup()
        await upstream_runner.cleanup()

    assert refused({"models": [], "control": control}, tmp), "a control block without controller_token_file is refused"
    assert refused({"models": [], "controller_token_file": token_file.name, "control": dict(control, activate_url=base + "/activate")}, tmp), "a control activate_url is refused"
    assert refused({"models": [], "controller_token_file": token_file.name}, tmp), "a token file with nothing to control is refused"
    without = {key: value for key, value in control.items() if key != "working_state_url"}
    assert "working_state_url" in (refused({"models": [], "controller_token_file": token_file.name, "control": without}, tmp) or ""), "a control block without the working-state controller is refused"
    shared = tmp / "shared-token"
    shared.write_text(CONTROLLER_TOKEN)
    shared.chmod(0o644)
    assert "owner only" in (refused({"models": [], "controller_token_file": shared.name, "control": control}, tmp) or ""), "a token file others can read is refused"
    short = tmp / "short-token"
    short.write_text("k" * 31)
    short.chmod(0o600)
    assert refused({"models": [], "controller_token_file": short.name, "control": control}, tmp), "a short controller token is refused"

    script = ROOT / "tools" / "serving" / "chat_frontend.py"
    for host in ("0.0.0.0", "192.168.50.4", "100.123.97.61", "::", "localhost"):
        started = subprocess.run([sys.executable, str(script), "--config", str(tmp / "absent.json"), "--host", host, "--port", "1"],
                                 capture_output=True, text=True, timeout=120)
        assert started.returncode != 0 and "not a loopback address" in started.stderr, (host, started.stderr[-300:])
    for host in ("127.0.0.1", "::1"):
        started = subprocess.run([sys.executable, str(script), "--config", str(tmp / "absent.json"), "--host", host, "--port", "1"],
                                 capture_output=True, text=True, timeout=120)
        assert started.returncode != 0 and "not a loopback address" not in started.stderr and "absent.json" in started.stderr, (host, started.stderr[-300:])


with tempfile.TemporaryDirectory() as directory:
    asyncio.run(run(Path(directory)))
print("PASS test_chat_frontend_switch_now: forced switch only for the admin login; 401 and 403 never reach the controller; every controller call carries the controller token and only there; no unauthenticated activate route; status lists waiting requests; a missing or shared token file and a non-loopback address stop the chat layer")
