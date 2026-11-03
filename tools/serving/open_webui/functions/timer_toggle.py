"""
title: Timer
author: SparkPipe
version: 1.0.0
description: While ticked, this chat's timer prompt runs again on its schedule after each reply. Set it with /timer every 10m <prompt>; the model can change it with its set_chat_timer tool. Sending a message restarts the countdown.
"""


class Filter:
    def __init__(self):
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS41Ij48Y2lyY2xlIGN4PSIxMiIgY3k9IjEzIiByPSI4Ii8+PHBhdGggZD0iTTEyIDl2NGwzIDJNOSAyaDYiLz48L3N2Zz4="

    async def inlet(self, body: dict) -> dict:
        return body
