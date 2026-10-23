"""
title: Full context
author: SparkPipe
version: 1.0.0
description: Turns SparkPipe chat memory off for this chat, so the model reads the whole transcript every turn. Memory keeps long chats fast; turn this on only when a reply needs the raw earlier turns.
"""

MEMORY_OFF = "\ue000SPARKPIPE-MEMORY-OFF\ue001"


class Filter:
    def __init__(self):
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS41Ij48cGF0aCBkPSJNNCA2aDE2TTQgMTJoMTZNNCAxOGgxMCIvPjwvc3ZnPg=="

    async def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        body["messages"] = [{"role": "system", "content": MEMORY_OFF}] + list(body.get("messages") or [])
        return body
