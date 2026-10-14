"""
title: Fleet
author: SparkPipe
version: 1.0.0
description: Admin only. Reads the SparkPipe fleet's state and logs, and runs commands on the hub or a spark after the admin approves each one.
"""

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

    async def fleet_run(self, host: str, command: str, timeout: int = 60, __user__: dict = None, __event_call__=None) -> str:
        """
        Run a shell command on the hub (rtx5090) or one spark (spark0 to sparkf) as the spec user. The admin sees the exact command and approves or rejects it before it runs. Prefer fleet_status, fleet_nodes and fleet_logs when they answer the question, and prefer read-only commands.
        :param host: rtx5090, or a spark from spark0 to sparkf
        :param command: the shell command, run with bash on the hub or through ssh on a spark
        :param timeout: seconds before the command is stopped, at most 600
        """
        if not self._admin(__user__):
            return ADMIN_ONLY
        if __event_call__ is None:
            return "This command needs the admin's approval, and this chat cannot show the approval dialog, so it was not run."
        approved = await __event_call__(
            {
                "type": "confirmation",
                "data": {
                    "title": f"Run on {host}?",
                    "message": f"Kimi wants to run this as spec on {host} (stopped after {timeout} s):\n\n```\n{command}\n```",
                },
            }
        )
        if approved is not True:
            return "The admin did not approve this command, so it was not run."
        return await self._call(
            "POST", "/v1/run",
            json={"host": host, "command": command, "timeout": timeout, "approved_by": __user__.get("email") or __user__.get("id")},
        )

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
