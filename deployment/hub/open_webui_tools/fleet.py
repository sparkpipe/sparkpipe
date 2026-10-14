"""
title: Fleet
author: SparkPipe
version: 1.0.0
description: Admin only. Reads the SparkPipe fleet's state and logs, and runs commands on the hub or a spark as a restricted user after the admin grants access for the chat.
"""

import json as jsonlib

import aiohttp
from pydantic import BaseModel, Field

ADMIN_ONLY = "The Fleet tool is for the SparkPipe Chat admin only."


class Tools:
    class Valves(BaseModel):
        GATEWAY_URL: str = Field(default="http://172.17.0.1:8441", description="The hub's fleet gateway")
        GATEWAY_TOKEN: str = Field(default="", description="The gateway's token")
        TIMEOUT_SECONDS: int = Field(default=660, description="Longest wait for one gateway call")

    def __init__(self):
        self.valves = self.Valves()

    async def fleet_status(self, __user__: dict = None) -> str:
        """
        Show which model the SparkPipe fleet has loaded, any model switch in progress (and what it is waiting for), and the recent switches.
        """
        if not self._admin(__user__):
            return ADMIN_ONLY
        return await self._call("GET", "/v1/status")

    async def fleet_nodes(self, __user__: dict = None) -> str:
        """
        Show each machine's uptime, load, free memory, root disk use, GPU clock, utilization, temperature, memory and power, and its busiest processes, for the hub (rtx5090) and every spark (spark0 to sparkf).
        """
        if not self._admin(__user__):
            return ADMIN_ONLY
        return await self._call("GET", "/v1/nodes")

    async def fleet_logs(self, name: str, lines: int = 100, node: str = "", __user__: dict = None) -> str:
        """
        Read the last lines of a SparkPipe log.
        :param name: chat (chat layer), swap (model swap controller), litellm-chat, litellm, litellm-external, fleet-audit (commands run through this tool), api:<name> (an engine API log on the hub, for example api:k3 or api:glm53_full), or engine (the loaded model's engine log on the spark given in node)
        :param lines: how many lines, at most 500
        :param node: for name engine, the spark to read it from, for example spark3
        """
        if not self._admin(__user__):
            return ADMIN_ONLY
        return await self._call("GET", "/v1/logs", params={"name": name, "lines": str(lines), "node": node})

    async def fleet_run(self, host: str, command: str, timeout: int = 60, elevated: bool = False, __user__: dict = None, __event_call__=None, __chat_id__: str = None) -> str:
        """
        Run a shell command on the hub (rtx5090) or one spark (spark0 to sparkf). It runs as the restricted user sparkagent, which has no sudo and cannot read secrets or other users' files; the admin grants that once per chat, and later commands in the chat run without asking. Set elevated to true only when the command needs the spec user's rights, for example restarting a SparkPipe service; the admin then approves that exact command. Prefer fleet_status, fleet_nodes and fleet_logs when they answer the question.
        :param host: rtx5090, or a spark from spark0 to sparkf
        :param command: the shell command, run with bash on the hub or through ssh on a spark
        :param timeout: seconds before the command is stopped, at most 600
        :param elevated: run as spec instead of sparkagent; the admin approves each such command
        """
        if not self._admin(__user__):
            return ADMIN_ONLY
        if __event_call__ is None:
            return "Fleet commands need the admin's approval, and this chat cannot show the approval dialog, so nothing was run."
        approver = __user__.get("email") or __user__.get("id")
        if elevated:
            approved = await __event_call__(
                {
                    "type": "confirmation",
                    "data": {
                        "title": f"Run as spec on {host}?",
                        "message": f"Kimi wants to run this with the spec user's full rights on {host} (stopped after {timeout} s):\n\n```\n{command}\n```",
                    },
                }
            )
            if approved is not True:
                return "The admin did not approve this elevated command, so it was not run."
            return await self._call("POST", "/v1/run", json={"host": host, "command": command, "timeout": timeout, "as": "spec", "approved_by": approver})
        if not __chat_id__ or str(__chat_id__).startswith("local:"):
            return "Fleet access is granted per saved chat, and this chat has no id; run it with elevated true to approve this one command."
        status = await self._call_json("GET", "/v1/grant", params={"chat_id": __chat_id__})
        if status is None or not status.get("ready"):
            return "The restricted fleet user is not set up yet (deployment/hub/fleet_agent_setup.sh); use elevated true to approve commands one by one."
        if not status.get("granted"):
            approved = await __event_call__(
                {
                    "type": "confirmation",
                    "data": {
                        "title": "Give Kimi fleet access for this chat?",
                        "message": (
                            f"For the next 8 hours, Kimi can run commands in this chat as {status.get('user')} on rtx5090 and spark0 to sparkf "
                            "without asking again. That user has no sudo and cannot read secrets or other users' files; every command is audited. "
                            f"Its first command, on {host}:\n\n```\n{command}\n```"
                        ),
                    },
                }
            )
            if approved is not True:
                return "The admin did not grant fleet access for this chat, so nothing was run."
            granted = await self._call_json("POST", "/v1/grant", json={"chat_id": __chat_id__, "approved_by": approver})
            if granted is None or "user" not in granted:
                return "The fleet gateway refused the grant, so nothing was run."
        return await self._call("POST", "/v1/run", json={"host": host, "command": command, "timeout": timeout, "as": "agent", "chat_id": __chat_id__})

    async def _call_json(self, method, path, params=None, json=None):
        text = await self._call(method, path, params=params, json=json)
        try:
            return jsonlib.loads(text)
        except ValueError:
            return None

    def _admin(self, user):
        return isinstance(user, dict) and user.get("role") == "admin"

    async def _call(self, method, path, params=None, json=None):
        if not self.valves.GATEWAY_TOKEN:
            return "The Fleet tool has no gateway token configured."
        headers = {"Authorization": f"Bearer {self.valves.GATEWAY_TOKEN}"}
        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, self.valves.GATEWAY_URL.rstrip("/") + path, params=params, json=json, headers=headers) as response:
                    text = await response.text()
                    if response.status != 200:
                        return f"Fleet gateway error {response.status}: {text[:2000]}"
                    return text
        except (aiohttp.ClientError, TimeoutError) as error:
            return f"Fleet gateway unreachable: {error or type(error).__name__}"
