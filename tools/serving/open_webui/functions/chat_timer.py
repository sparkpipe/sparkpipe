"""
title: Chat timer
author: SparkPipe
version: 1.0.0
description: Keeps each chat's timer prompt (set with /timer every 10m <prompt>, or by the model's set_chat_timer tool) and, while the Timer box is ticked, schedules it again after every reply.
"""

import json
import os
import re
from pathlib import Path

TOGGLE = "sparkpipe_timer"
COMMAND = re.compile(r"^\s*/timer(?:\s+(.*))?\s*$", re.S | re.I)
EVERY = re.compile(r"^\s*every\s+(\d+)\s*(s|m|h|d)\s+(.+)$", re.S | re.I)
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


def interval(count, unit):
    seconds = int(count) * SECONDS[unit.lower()]
    return f"{max(seconds, MINIMUM_SECONDS)}s"


def apply_command(chat, text, by):
    found = COMMAND.match(text or "")
    if not found or not chat:
        return None
    argument = (found.group(1) or "").strip()
    timers = load()
    if argument.lower() in ("off", "stop"):
        timers.pop(chat, None)
    else:
        every = EVERY.match(argument)
        if not every:
            return None
        timers[chat] = {"prompt": every.group(3).strip(), "every": interval(every.group(1), every.group(2)), "by": by}
    save(timers)
    return timers.get(chat)


def text_of(content):
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


class Filter:
    async def inlet(self, body: dict, __metadata__: dict | None = None) -> dict:
        messages = body.get("messages") or []
        chat = (__metadata__ or {}).get("chat_id") or ""
        if messages and messages[-1].get("role") == "user":
            apply_command(chat, text_of(messages[-1].get("content")), "user")
        return body

    async def outlet(self, body: dict, __user__: dict | None = None, __metadata__: dict | None = None, __request__=None) -> dict:
        chat = body.get("chat_id") or ""
        entry = load().get(chat)
        if not entry or TOGGLE not in (body.get("filter_ids") or []) or __request__ is None or not __user__:
            return body
        from open_webui.utils.timers import create_timer

        await create_timer(prompt=entry["prompt"], at=entry["every"], cancel_on=["chat.user_message"], request=__request__,
                           user_data=__user__, metadata=__metadata__ or {}, parent_chat_id=chat, parent_message_id=body.get("id"))
        return body
