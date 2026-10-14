#!/usr/bin/env python3
"""Switch now is admin-only: the chat layer forwards a forced switch to the swap controller only after SparkPipe Chat confirms the caller's login is the admin, refuses a missing login with 401 and a non-admin login with 403 without touching the controller, and its status reports how many requests wait for each model."""
import asyncio
import importlib.util
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


async def run():
    forced = []
    checked = []

    async def auth(request):
        token = request.headers.get("Authorization", "")
        checked.append(token)
        if token == "Bearer admin-token":
            return web.json_response({"id": "a", "role": "admin"})
        if token == "Bearer user-token":
            return web.json_response({"id": "u", "role": "user"})
        return web.json_response({"detail": "Not authenticated"}, status=401)

    async def force(request):
        forced.append(await request.json())
        return web.json_response({"state": "swapping", "force_requested": True}, status=202)

    async def status(request):
        return web.json_response({"active": "m", "state": "swapping", "draining": {"model": "m", "live_requests": 1}})

    upstream = web.Application()
    upstream.router.add_get("/api/v1/auths/", auth)
    upstream.router.add_post("/force", force)
    upstream.router.add_get("/status", status)
    upstream_runner, base = await serve(upstream)

    frontend = chat_frontend.Frontend({"models": [], "control": {
        "status_url": base + "/status", "activate_url": base + "/activate",
        "force_url": base + "/force", "admin_url": base + "/api/v1/auths/"}}, ROOT)
    app = web.Application()
    app.on_startup.append(frontend.start)
    app.on_cleanup.append(frontend.stop)
    app.router.add_get("/v1/sparkpipe/status", frontend.control_status)
    app.router.add_post("/v1/sparkpipe/switch-now", frontend.control_switch_now)
    front_runner, front = await serve(app)
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
            assert forced == [{}], forced
            async with session.post(front + "/v1/sparkpipe/switch-now", headers={"Authorization": "Bearer admin-token"}, json={"model": "not-served"}) as response:
                assert response.status == 404, response.status
            assert forced == [{}], forced
            assert checked == ["Bearer user-token", "Bearer stolen", "Bearer admin-token", "Bearer admin-token"], checked
            async with session.get(front + "/v1/sparkpipe/status") as response:
                body = await response.json()
                assert body["draining"]["live_requests"] == 1 and body["waiting"] == {}, body
    finally:
        await front_runner.cleanup()
        await upstream_runner.cleanup()


asyncio.run(run())
print("PASS test_chat_frontend_switch_now: forced switch only for the admin login; 401 and 403 never reach the controller; status lists waiting requests")
