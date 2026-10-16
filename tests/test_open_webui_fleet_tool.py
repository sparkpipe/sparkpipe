#!/usr/bin/env python3
"""SparkPipe Chat's Fleet tool works only for the admin. A command runs in the sandbox only after the admin grants access for that chat, asked once: a rejected or failed grant runs nothing, later commands in the granted chat run without asking, a chat without an id or a gateway without a sandbox runs nothing. An elevated command runs as spec only after the admin approves that exact command. Gateway answers come back as plain text (command output, logs and node status unescaped), and gateway errors as text."""
import asyncio
import importlib.util
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("fleet_tool", ROOT / "tools" / "serving" / "open_webui" / "tools" / "fleet.py")
fleet_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fleet_tool)

ADMIN = {"id": "a1", "email": "owner@example.test", "role": "admin"}
USER = {"id": "u1", "email": "friend@example.test", "role": "user"}


async def run():
    seen = []
    state = {"granted": set(), "ready": True}

    async def handler(request):
        body = await request.json() if request.can_read_body else None
        seen.append((request.method, request.path, dict(request.query), body, request.headers.get("Authorization")))
        if request.path == "/v1/grant" and request.method == "GET":
            chat = request.query.get("chat_id")
            return web.json_response({"chat_id": chat, "granted": chat in state["granted"], "ready": state["ready"], "user": "sparkagent"})
        if request.path == "/v1/grant" and request.method == "POST":
            state["granted"].add(body["chat_id"])
            return web.json_response({"chat_id": body["chat_id"], "user": "sparkagent", "expires_utc": "later"})
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
    events = []

    async def approve(event):
        events.append(event)
        return True

    async def reject(event):
        events.append(event)
        return False

    async def broken(event):
        events.append(event)
        return {"error": "disconnected"}

    async def never(event):
        raise AssertionError(f"asked for approval again: {event}")

    try:
        assert await tool.fleet_status(__user__=ADMIN) == "The Fleet tool has no gateway token configured."
        tool.valves.GATEWAY_TOKEN = "gateway-token"

        for call in (tool.fleet_status(__user__=USER), tool.fleet_nodes(__user__=USER), tool.fleet_logs("chat", __user__=USER),
                     tool.fleet_run("spark1", "uptime", __user__=USER, __event_call__=approve, __chat_id__="c1"),
                     tool.fleet_status(__user__=None)):
            assert await call == fleet_tool.ADMIN_ONLY
        assert seen == [] and events == [], (seen, events)

        reply = await tool.fleet_status(__user__=ADMIN)
        assert reply == "ok: True\npath: /v1/status" and seen[-1][4] == "Bearer gateway-token", (reply, seen)
        await tool.fleet_logs("engine", lines=20, node="spark3", __user__=ADMIN)
        assert seen[-1][1] == "/v1/logs" and seen[-1][2] == {"name": "engine", "lines": "20", "node": "spark3"}, seen[-1]

        assert "cannot show the approval dialog" in await tool.fleet_run("spark1", "uptime", __user__=ADMIN, __chat_id__="c1")
        assert "no id" in await tool.fleet_run("spark1", "uptime", __user__=ADMIN, __event_call__=never)
        assert "no id" in await tool.fleet_run("spark1", "uptime", __user__=ADMIN, __event_call__=never, __chat_id__="local:abc")

        for decline in (reject, broken):
            reply = await tool.fleet_run("spark1", "uptime", __user__=ADMIN, __event_call__=decline, __chat_id__="c1")
            assert "did not grant" in reply, reply
        assert not any(path == "/v1/run" or (path == "/v1/grant" and method == "POST") for method, path, *_ in seen), seen
        assert "for this chat" in events[0]["data"]["title"] and "uptime" in events[0]["data"]["message"] and "sandbox" in events[0]["data"]["message"], events[0]

        events.clear()
        await tool.fleet_run("spark1", "uptime", timeout=30, __user__=ADMIN, __event_call__=approve, __chat_id__="c1")
        assert len(events) == 1, events
        grant_posts = [body for method, path, _, body, _ in seen if path == "/v1/grant" and method == "POST"]
        assert grant_posts == [{"chat_id": "c1", "approved_by": "owner@example.test"}], grant_posts
        assert seen[-1][1] == "/v1/run" and seen[-1][3] == {"host": "spark1", "command": "uptime", "timeout": 30, "as": "sandbox", "chat_id": "c1"}, seen[-1]

        await tool.fleet_run("spark2", "df -h", __user__=ADMIN, __event_call__=never, __chat_id__="c1")
        assert seen[-1][3] == {"host": "spark2", "command": "df -h", "timeout": 60, "as": "sandbox", "chat_id": "c1"}, seen[-1]

        count = len(seen)
        assert "did not approve" in await tool.fleet_run("rtx5090", "systemctl --user restart sparkpipe-chat", elevated=True, __user__=ADMIN, __event_call__=reject, __chat_id__="c1")
        assert all(path != "/v1/run" for _, path, *_ in seen[count:]), seen[count:]
        events.clear()
        await tool.fleet_run("rtx5090", "systemctl --user restart sparkpipe-chat", elevated=True, __user__=ADMIN, __event_call__=approve, __chat_id__="c1")
        assert "spec" in events[0]["data"]["title"] and "restart sparkpipe-chat" in events[0]["data"]["message"], events
        assert seen[-1][3] == {"host": "rtx5090", "command": "systemctl --user restart sparkpipe-chat", "timeout": 60, "as": "spec", "approved_by": "owner@example.test"}, seen[-1]

        state["ready"] = False
        count = len(seen)
        assert "no command sandbox" in await tool.fleet_run("spark1", "uptime", __user__=ADMIN, __event_call__=never, __chat_id__="c2")
        assert all(path != "/v1/run" for _, path, *_ in seen[count:]), seen[count:]
        state["ready"] = True
        assert await tool.fleet_run("spark1", "fail", __user__=ADMIN, __event_call__=never, __chat_id__="c1") == 'Fleet gateway error 403: {"error": "refused: test"}'
        ran = fleet_tool.plain_text({"host": "spark2", "as": "sandbox", "exit": 0, "timed_out": False, "seconds": 0.4, "output": "line one\nline \u00e9 two\n"})
        assert ran == "spark2 (sandbox), exit 0 after 0.4 s\nline one\nline \u00e9 two\n", ran
        assert fleet_tool.plain_text({"host": "spark2", "as": "spec", "exit": None, "timed_out": True, "seconds": 60, "output": ""}).startswith("spark2 (spec), timed out after 60 s")
        assert fleet_tool.plain_text({"name": "chat", "text": "a\nb"}) == "chat\na\nb"
        nodes = fleet_tool.plain_text({"spark0": {"exit": 0, "status": "up 3 days\nload 0.1"}, "spark1": {"exit": 255, "status": "unreachable"}})
        assert nodes == "== spark0 (exit 0)\nup 3 days\nload 0.1\n\n== spark1 (exit 255)\nunreachable", nodes
        assert fleet_tool.plain_text({"active": "kimi-k3", "history": [{"to": "kimi-k3", "s": 12}]}) == 'active: kimi-k3\nhistory:\n- {"to": "kimi-k3", "s": 12}'
    finally:
        await runner.cleanup()


asyncio.run(run())
print("PASS test_open_webui_fleet_tool: admin only, one grant per chat for the sandbox, nothing runs before it or without a chat id, elevated commands approved one by one, plain-text answers, gateway errors reported")
