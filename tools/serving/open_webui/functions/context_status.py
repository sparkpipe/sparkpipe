"""
title: Context status
author: SparkPipe
version: 1.0.0
description: After each reply, shows how much of the model's context the chat's prompt used, whether SparkPipe memory is on, the context target and any compaction. Type /context for details, /compact 50% to keep the most relevant half of a chat with memory off.
"""

import base64
import hashlib
import hmac
import json
import os
import time

import aiohttp

COMMAND = "/context json"


def b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def user_token(user_id, secret):
    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64(json.dumps({"sub": user_id, "iss": "open-webui", "exp": int(time.time()) + 300}).encode())
    return f"{head}.{body}.{b64(hmac.new(secret.encode(), f'{head}.{body}'.encode(), hashlib.sha256).digest())}"


def short(tokens):
    if tokens >= 1048576 and tokens % 1048576 == 0:
        return f"{tokens // 1048576}M"
    if tokens >= 10240:
        return f"{tokens / 1024:.0f}K"
    if tokens >= 1024:
        return f"{tokens / 1024:.1f}K"
    return str(tokens)


def describe(state, usage):
    limit = state["context_tokens"]
    used = (usage or {}).get("prompt_tokens") or state["history_tokens"]
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    parts = [f"Context {short(used)} of {short(limit)} ({100 * used / limit:.1f}%)"]
    if cached:
        parts.append(f"{short(cached)} cached")
    if state["memory"] == "on":
        parts.append("memory on" + (f", target {short(state['target'])}" if state.get("target") else ""))
    else:
        parts.append("memory off")
    if state.get("compaction"):
        parts.append(f"compacted to {state['compaction']['kept']} of {state['compaction']['of']} exchanges")
    return " · ".join(parts)


def connection(request):
    config = request.app.state.config
    urls, keys = list(config.OPENAI_API_BASE_URLS or []), list(config.OPENAI_API_KEYS or [])
    return (urls[0].rstrip("/"), keys[0] if keys else "") if urls else (None, None)


class Filter:
    async def outlet(self, body: dict, __user__: dict | None = None, __request__=None, __event_emitter__=None) -> dict:
        messages = [m for m in body.get("messages") or [] if m.get("role") in ("system", "user", "assistant")]
        chat = body.get("chat_id") or ""
        secret = os.environ.get("FORWARD_USER_INFO_HEADER_JWT_SECRET", "")
        if __event_emitter__ is None or __request__ is None or not __user__ or not messages or not chat or chat.startswith("local:") or not secret:
            return body
        base, key = connection(__request__)
        if base is None:
            return body
        request = {"model": body.get("model"), "stream": False,
                   "messages": [{"role": m["role"], "content": m.get("content") or ""} for m in messages] + [{"role": "user", "content": COMMAND}]}
        headers = {"Authorization": f"Bearer {key}", "X-OpenWebUI-User-Jwt": user_token(__user__["id"], secret), "X-OpenWebUI-Chat-Id": chat}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.post(base + "/chat/completions", json=request, headers=headers) as response:
                    reply = await response.json(content_type=None)
            state = json.loads(reply["choices"][0]["message"]["content"])
        except (aiohttp.ClientError, TimeoutError, KeyError, IndexError, TypeError, ValueError):
            return body
        usage = next((m.get("usage") for m in reversed(messages) if m.get("role") == "assistant"), None)
        await __event_emitter__({"type": "status", "data": {"description": describe(state, usage), "done": True}})
        return body
