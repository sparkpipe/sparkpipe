#!/usr/bin/env python3
"""The chat layer's per-request trace attributes a request to its chat, user message and tool round with hashes and numbers only: the chat and user message ids are hashed, the round counts the assistant tool-call messages of the current exchange, the newest round's tool results are listed by tool name (taken from the call when the message has none) and size, a trailing tool-image message does not end the round, and no message content or raw id ever appears in it."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAIL test_chat_frontend_trace: {message}")


def main():
    headers = {"X-OpenWebUI-Chat-Id": "chat-secret-id", "X-OpenWebUI-User-Message-Id": "message-secret-id"}
    history = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "first question about Ostrel"},
               {"role": "assistant", "content": "first answer"}, {"role": "user", "content": "find the review room"}]
    call = lambda i, name: {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": "{}"}}
    answer = chat_frontend.request_trace(headers, history)
    check(answer["kind"] == "answer" and answer["round"] == 0 and "tool_chars" not in answer, f"a new question is an answer request: {answer}")
    check(answer["chat"] == chat_frontend.short_hash("chat-secret-id") and len(answer["chat"]) == 12, "the chat id is hashed")
    rounds = history + [{"role": "assistant", "content": "", "tool_calls": [call(1, "search_web")]}, {"role": "tool", "tool_call_id": "call_1", "content": "x" * 300},
                        {"role": "assistant", "content": "", "tool_calls": [call(2, "fetch_url"), call(3, "fetch_url")]},
                        {"role": "tool", "tool_call_id": "call_2", "content": "page one secret text"}, {"role": "tool", "tool_call_id": "call_3", "name": "named", "content": "y" * 50}]
    second = chat_frontend.request_trace(headers, rounds)
    check(second["kind"] == "round" and second["round"] == 2, f"the round counts tool-call messages of the current exchange: {second}")
    check(second["tool_chars"] == [["fetch_url", 20], ["named", 50]], f"the newest round's results are listed by tool and size: {second}")
    images = rounds + [{"role": "user", "content": [{"type": "text", "text": "Here are the images"}, {"type": "image_url", "image_url": {"url": "data:,"}}]}]
    check(chat_frontend.request_trace(headers, images)["round"] == 2, "a tool-image message does not end the round")
    blob = json.dumps([answer, second])
    for secret in ("chat-secret-id", "message-secret-id", "Ostrel", "page one", "review room"):
        check(secret not in blob, f"the trace carries no content or raw id: {secret}")
    check(chat_frontend.request_trace(None, history)["chat"] is None, "a request without ids has no chat hash")
    print("PASS test_chat_frontend_trace: hashed chat and message ids, tool round and newest results by name and size, no content in the trace")


if __name__ == "__main__":
    main()
