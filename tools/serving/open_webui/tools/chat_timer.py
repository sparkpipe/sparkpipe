"""
title: Chat timer
author: SparkPipe
version: 1.0.0
description: Lets the model set or stop this chat's timer prompt. The prompt runs on its schedule while the user's Timer box is ticked.
"""

import json
import os
import re
from pathlib import Path

EVERY = re.compile(r"^\s*(\d+)\s*(s|m|h|d)\s*$", re.I)
SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
MINIMUM_SECONDS = 60


def store_path():
    return Path(os.environ.get("DATA_DIR", "/app/backend/data")) / "sparkpipe" / "chat_timers.json"


def load():
    try:
        return json.loads(store_path().read_text())
    except (OSError, ValueError):
        return {}


def save(timers):
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(timers, ensure_ascii=False, indent=1))
    os.replace(temporary, path)


class Tools:
    async def set_chat_timer(self, prompt: str, every: str, __chat_id__: str | None = None) -> str:
        """
        Set the prompt this chat's timer sends and how often. It runs while the user's Timer box is ticked; the next run uses what you set.
        :param prompt: The message the timer sends into this chat, written as the user would write it
        :param every: The interval: a number and s, m, h or d, such as 10m or 2h (at least one minute)
        :return: What the timer now sends and how often
        """
        found = EVERY.match(every or "")
        if not __chat_id__ or not (prompt or "").strip() or not found:
            return "Not set: give a non-empty prompt and an interval such as 10m or 2h."
        seconds = max(int(found.group(1)) * SECONDS[found.group(2).lower()], MINIMUM_SECONDS)
        timers = load()
        timers[__chat_id__] = {"prompt": prompt.strip(), "every": f"{seconds}s", "by": "model"}
        save(timers)
        return f"Timer set: every {seconds} s it sends: {prompt.strip()}. It runs while the user's Timer box is ticked."

    async def stop_chat_timer(self, __chat_id__: str | None = None) -> str:
        """
        Stop this chat's timer.
        :return: Confirmation
        """
        timers = load()
        timers.pop(__chat_id__ or "", None)
        save(timers)
        return "Timer stopped for this chat."
