#!/usr/bin/env python3
"""A chat-memory session is opened only for a verified Open WebUI identity on a real chat turn, and its bookkeeping holds across turns: requests without a valid token, with a task, without ids, with a client tool named expand or with Open WebUI's own summary are served raw; the first turns carry the system note and the expand tool but no ledger; after a finished answer the curator runs only once the engine is idle and is cancelled by a foreground request; its line for the current exchange binds on the next turn through the parent message id, older raw exchanges get their lines directly, and the open set follows; the ledger then covers every lined exchange but the newest; follow-ups reuse the frozen layout and replay expansions; two invalid curator replies leave fallback lines; and checkpointed models get a ledger-end stem."""
import asyncio
import base64
import hashlib
import hmac
import importlib.util
import json
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_memory", ROOT / "tools" / "serving" / "chat_memory.py")
cm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cm)
SECRET = "k" * 48


def b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def token(sub="user-1", iss="open-webui", exp=None, secret=SECRET):
    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64(json.dumps({"sub": sub, "iss": iss, "exp": exp or time.time() + 300}).encode())
    return f"{head}.{body}.{b64(hmac.new(secret.encode(), f'{head}.{body}'.encode(), hashlib.sha256).digest())}"


def headers(user_message, parent=None, message=None, chat="chat-1", task="", jwt=None):
    result = {"X-OpenWebUI-User-Jwt": jwt if jwt is not None else token(), "X-OpenWebUI-Chat-Id": chat,
              "X-OpenWebUI-User-Message-Id": user_message, "X-OpenWebUI-Task": task}
    if parent:
        result["X-OpenWebUI-User-Message-Parent-Id"] = parent
    if message:
        result["X-OpenWebUI-Message-Id"] = message
    return result


class Model:
    def __init__(self, checkpoint):
        self.id = "fake-model"
        self.memory = {"curator_suffix": "</think>", "cache_checkpoint_tokens": checkpoint, "open_budget": 1500, "line_tokens": 48}


class Frontend:
    def __init__(self):
        self.ready = True
        self.replies = []
        self.calls = []
        self.cuts = []
        self.block = None

    async def memory_model_ready(self, model):
        return self.ready

    def memory_curator_ids(self, model, messages, tools, kwargs, suffix):
        self.calls.append(messages)
        return [1, 2, 3], False

    def memory_cut_ids(self, model, messages, tools, kwargs, sentinel):
        self.cuts.append(messages)
        return [9, 9]

    async def memory_complete(self, model, ids, max_tokens, reasoning_open=False):
        if self.block is not None:
            await self.block.wait()
        if max_tokens == 1:
            return ""
        return self.replies.pop(0) if self.replies else "not json"


def conversation(n):
    messages = [{"role": "system", "content": "Be helpful."}]
    for i in range(1, n):
        messages += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    messages.append({"role": "user", "content": f"question {n}"})
    return messages


async def settle(memory, model):
    for _ in range(100):
        task = memory.tasks.get(model.id)
        if (task is None or task.done()) and not memory.jobs.get(model.id):
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"background work did not settle: {memory.jobs.get(model.id)}")


async def turn(memory, frontend, model, n, user_message, parent, message, reply):
    session = memory.session(model, headers(user_message, parent, message), conversation(n), [], {})
    assert session is not None, n
    memory.enter(model)
    frontend.replies.append(reply)
    memory.leave(model)
    session.finished(session.messages, f"answer {n}", "", [], "stop")
    await settle(memory, model)
    return session


async def run():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "secret").write_text(SECRET)
        frontend = Frontend()
        memory = cm.Memory({"store": tmp + "/store", "jwt_secret_file": tmp + "/secret", "idle_seconds": 0.01}, frontend)
        model = Model(16384)

        refused = [
            (headers("u1", jwt=""), []), (headers("u1", jwt=token(secret="x" * 48)), []), (headers("u1", jwt=token(iss="other")), []),
            (headers("u1", jwt=token(exp=time.time() - 5)), []), (headers("u1", task="title_generation"), []),
            ({k: v for k, v in headers("u1").items() if k != "X-OpenWebUI-User-Message-Id"}, []),
            (headers("u1", chat="a, b"), []), (headers("u1"), [{"type": "function", "function": {"name": "expand"}}]),
        ]
        for request_headers, tools in refused:
            assert memory.session(model, request_headers, conversation(1), tools, {}) is None, request_headers
        summary = [{"role": "system", "content": "[CONVERSATION SUMMARY]\nold"}, {"role": "user", "content": "hi"}]
        assert memory.session(model, headers("u1"), summary, [], {}) is None
        assert memory.session(type("Plain", (), {"id": "plain", "memory": None})(), headers("u1"), conversation(1), [], {}) is None

        first = await turn(memory, frontend, model, 1, "u1", None, "a1", '{"lines": {"e1": "The user asked question 1; the assistant gave answer 1."}, "open": []}')
        assert frontend.cuts == [], "no stem while the next ledger would be empty"
        assert cm.SYSTEM_NOTE in first.messages[0]["content"] and first.tools[-1]["function"]["name"] == "expand" and first.layout["lined"] == 0
        state = memory.store.load("user-1", "chat-1")
        assert state["pending"]["a1"]["line"].startswith("The user asked question 1"), state["pending"]
        assert "user-1" not in json.dumps(list(Path(tmp, "store").rglob("*")).__repr__())

        second = await turn(memory, frontend, model, 2, "u2", "a1", "a2", '{"lines": {"e2": "The user asked question 2; the assistant gave answer 2."}, "open": ["e1"]}')
        key1 = cm.exchange_key(conversation(2)[1:3])
        assert state["lines"][key1].startswith("The user asked question 1") and second.layout["lined"] == 0
        assert len(frontend.cuts) == 1 and frontend.cuts[0][-1]["content"] == cm.LEDGER_OPEN + "[e1] " + state["lines"][key1] + "\n" + cm.SENTINEL, frontend.cuts
        assert state["open_by_message"]["a2"] == ["e1"]

        third = await turn(memory, frontend, model, 3, "u3", "a2", "a3", '{"lines": {"e3": "The user asked question 3."}, "open": []}')
        assert third.layout["lined"] == 1 and third.layout["open"] == ["e1"], third.layout
        users = [m for m in third.messages if m["role"] == "user"]
        assert users[0]["content"].startswith(cm.LEDGER_OPEN + "[e1] The user asked question 1") and users[0]["content"].endswith("question 2")
        assert users[-1]["content"].startswith(cm.REFERENCE_OPEN) and 'history id="e1"' in users[-1]["content"] and users[-1]["content"].endswith("question 3")
        assert frontend.cuts and frontend.cuts[-1][-1]["content"].startswith(cm.LEDGER_OPEN + "[e1] ") and frontend.cuts[-1][-1]["content"].endswith(cm.SENTINEL)

        cold = memory.session(model, headers("v3", "b2", "b3", chat="chat-2"), conversation(3), [], {})
        assert cold.layout["lined"] == 0
        memory.store.append("user-1", "chat-2", {"type": "line", "key": cm.exchange_key(conversation(3)[1:3]), "line": "arrived mid-exchange", "source": "curator"})
        follow = memory.session(model, headers("v3", "b2", "b3", chat="chat-2"), conversation(3) + [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search_web", "arguments": {"query": "q"}}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "search_web", "content": "found"}], [], {})
        assert follow.layout["lined"] == 0, "a follow-up must reuse its exchange's frozen layout"
        fresh = memory.session(model, headers("v4", "b3", "b4", chat="chat-2"), conversation(4), [], {})
        assert fresh.layout["lined"] == 1, "the next user message sees the new line"
        third_session_expansion = memory.session(model, headers("u3", "a2", "a3"), conversation(3), [], {})
        third_session_expansion.continue_with("", "", [{"id": "x1", "type": "function", "function": {"name": "expand", "arguments": "{\"id\": \"e1\"}"}}])
        replay = memory.session(model, headers("u3", "a2", "a3"), conversation(3) + follow.current[1:], [], {})
        roles = [m["role"] for m in replay.messages]
        expand_at = next(i for i, m in enumerate(replay.messages) if m["role"] == "tool" and m.get("name") == "expand")
        search_at = next(i for i, m in enumerate(replay.messages) if m["role"] == "tool" and m.get("name") == "search_web")
        assert expand_at < search_at and replay.messages[expand_at - 2]["content"].endswith("question 3"), roles
        assert "User: question 1" in replay.messages[expand_at]["content"]

        frontend.block = asyncio.Event()
        session = memory.session(model, headers("u4", "a3", "a4"), conversation(4), [], {})
        session.finished(session.messages, "answer 4", "", [], "stop")
        await asyncio.sleep(0.1)
        running = memory.tasks.get(model.id)
        assert running is not None and not running.done(), "the curator should be running while the engine is idle"
        memory.enter(model)
        await asyncio.sleep(0.05)
        assert running.cancelled() and memory.jobs.get(model.id), "a foreground request must cancel the curator and keep its job"
        frontend.block = None
        frontend.replies[:] = ["garbage", "still garbage"]
        memory.leave(model)
        await settle(memory, model)
        assert state["pending"]["a4"]["line"].startswith("(auto) The user said: question 4"), state["pending"]["a4"]

        session = memory.session(model, headers("u5", "a4", "a5"), conversation(5), [], {})
        session.finished(session.messages, "answer 5", "", [{"id": "t"}], "stop")
        session.finished(session.messages, "answer 5", "", [], "length")
        assert not memory.jobs.get(model.id), "unfinished turns must not be curated"
        frontend.ready = False
        session.finished(session.messages, "answer 5", "", [], "stop")
        await settle(memory, model)
        assert "a5" not in state["pending"], "no background work while the fleet is switching"

        reopened = cm.Memory({"store": tmp + "/store", "jwt_secret_file": tmp + "/secret"}, frontend).store.load("user-1", "chat-1")
        assert reopened["lines"] == state["lines"] and reopened["pending"].keys() == state["pending"].keys()


asyncio.run(run())
print("PASS test_chat_memory_session: verified identity only, system note and expand from the first turn, idle-only cancellable curation, lines bound through parent ids, ledger with open nodes, frozen follow-up layouts with replayed expansions, fallback lines, ledger-end stems, state that survives a restart")
