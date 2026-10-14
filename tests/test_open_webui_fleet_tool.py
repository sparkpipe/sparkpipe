#!/usr/bin/env python3
"""SparkPipe Chat's Fleet tool works only for the admin and runs a command only after the admin approves it: a non-admin, a chat that cannot show the approval dialog, a rejected or failed approval never reach the gateway; an approved command is sent once with the admin named as approver; gateway errors come back as text."""
import asyncio
import importlib.util
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("fleet_tool", ROOT / "deployment" / "hub" / "open_webui_tools" / "fleet.py")
fleet_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fleet_tool)

ADMIN = {"id": "a1", "email": "owner@example.test", "role": "admin"}
USER = {"id": "u1", "email": "friend@example.test", "role": "user"}


async def run():
    seen = []

    async def handler(request):
        body = await request.json() if request.can_read_body else None
        seen.append((request.method, request.path, dict(request.query), body, request.headers.get("Authorization")))
        if request.path == "/v1/run" and body and body.get("command") == "fail":
            return web.json_response({"error": "refused: test"}, status=403)
        return web.json_response({"ok": True, "path": request.path})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    tool = fleet_tool.Tools()
    tool.valves.GATEWAY_URL = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        assert await tool.fleet_status(__user__=ADMIN) == "The Fleet tool has no gateway token configured."
        tool.valves.GATEWAY_TOKEN = "gateway-token"

        for call in (tool.fleet_status(__user__=USER), tool.fleet_nodes(__user__=USER), tool.fleet_logs("chat", __user__=USER),
                     tool.fleet_run("spark1", "uptime", __user__=USER, __event_call__=lambda event: asyncio.sleep(0, True)),
                     tool.fleet_status(__user__=None)):
            assert await call == fleet_tool.ADMIN_ONLY
        assert seen == [], seen

        reply = await tool.fleet_status(__user__=ADMIN)
        assert '"path": "/v1/status"' in reply and seen[-1][4] == "Bearer gateway-token", (reply, seen)
        await tool.fleet_logs("engine", lines=20, node="spark3", __user__=ADMIN)
        assert seen[-1][1] == "/v1/logs" and seen[-1][2] == {"name": "engine", "lines": "20", "node": "spark3"}, seen[-1]
        count = len(seen)

        assert "cannot show the approval dialog" in await tool.fleet_run("spark1", "uptime", __user__=ADMIN)
        events = []

        async def reject(event):
            events.append(event)
            return False

        async def broken(event):
            return {"error": "disconnected"}

        assert "did not approve" in await tool.fleet_run("spark1", "uptime", __user__=ADMIN, __event_call__=reject)
        assert "did not approve" in await tool.fleet_run("spark1", "uptime", __user__=ADMIN, __event_call__=broken)
        assert len(seen) == count, seen[count:]
        assert events[0]["type"] == "confirmation" and "uptime" in events[0]["data"]["message"] and "spark1" in events[0]["data"]["title"], events

        async def approve(event):
            return True

        reply = await tool.fleet_run("spark1", "uptime", timeout=30, __user__=ADMIN, __event_call__=approve)
        assert len(seen) == count + 1 and seen[-1][1] == "/v1/run", seen
        assert seen[-1][3] == {"host": "spark1", "command": "uptime", "timeout": 30, "approved_by": "owner@example.test"}, seen[-1]
        assert await tool.fleet_run("spark1", "fail", __user__=ADMIN, __event_call__=approve) == 'Fleet gateway error 403: {"error": "refused: test"}'
    finally:
        await runner.cleanup()


asyncio.run(run())
print("PASS test_open_webui_fleet_tool: admin only, nothing runs without the admin's approval, approved commands name the approver, gateway errors reported")
