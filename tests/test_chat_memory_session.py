#!/usr/bin/env python3
"""A chat-memory session is opened only for a verified Open WebUI identity on a real chat turn, and its bookkeeping holds across turns: requests without a valid token, with a task, without ids, with a client tool named like a memory tool or with Open WebUI's own summary are served raw; the first turns carry the system note and the expand and search_history tools but no ledger; a search_history call is answered inside the chat layer; after a finished answer the curator runs only once the engine is idle and is cancelled by a foreground request; its line for the current exchange binds on the next turn through the parent message id, older raw exchanges get their lines directly, and the open set follows; the ledger then covers every lined exchange but the newest; follow-ups reuse the frozen layout and replay expansions; two invalid curator replies leave fallback lines; the first curator attempt continues the answer's own prompt and reply tokens; checkpointed models get a ledger-end stem; the next turn's layout is pre-warmed up to its new question; a ledger over its budget is regrouped by the curator into nested group lines that expand to their children (stub lines when the curator fails), and follow-ups keep their frozen groups; and a prompt over the model's budget is fitted synchronously by dropping the open set, lining raw exchanges and folding the oldest ledger entries, while one that fits is left alone."""
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
        self.continued = []
        self.block = None

    async def memory_model_ready(self, model):
        return self.ready

    def memory_curator_ids(self, model, messages, tools, kwargs, suffix):
        self.calls.append(messages)
        return [1, 2, 3], False

    def memory_continuation_ids(self, model, messages, tools, kwargs, prompt_ids, generated_ids, instruction, suffix):
        self.continued.append((list(prompt_ids), list(generated_ids)))
        return [1, 2, 3, 4], False

    def memory_count(self, model, text):
        return len(text) // 4

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
            (headers("u1"), [{"type": "function", "function": {"name": "search_history"}}]),
        ]
        for request_headers, tools in refused:
            assert memory.session(model, request_headers, conversation(1), tools, {}) is None, request_headers
        summary = [{"role": "system", "content": "[CONVERSATION SUMMARY]\nold"}, {"role": "user", "content": "hi"}]
        assert memory.session(model, headers("u1"), summary, [], {}) is None
        assert memory.session(type("Plain", (), {"id": "plain", "memory": None})(), headers("u1"), conversation(1), [], {}) is None

        first = await turn(memory, frontend, model, 1, "u1", None, "a1", '{"lines": {"e1": "The user asked question 1; the assistant gave answer 1."}, "open": []}')
        assert frontend.cuts == [], "no stem while the next ledger would be empty"
        assert cm.SYSTEM_NOTE in first.messages[0]["content"] and [t["function"]["name"] for t in first.tools[-2:]] == ["expand", "search_history"] and first.layout["lined"] == 0
        state = memory.store.load("user-1", "chat-1")
        assert state["pending"]["a1"]["line"].startswith("The user asked question 1"), state["pending"]
        assert "user-1" not in json.dumps(list(Path(tmp, "store").rglob("*")).__repr__())

        second = await turn(memory, frontend, model, 2, "u2", "a1", "a2", '{"lines": {"e2": "The user asked question 2; the assistant gave answer 2."}, "open": ["e1"]}')
        key1 = cm.exchange_key(conversation(2)[1:3])
        assert state["lines"][key1].startswith("The user asked question 1") and second.layout["lined"] == 0
        assert len(frontend.cuts) == 2 and frontend.cuts[0][-1]["content"] == cm.LEDGER_OPEN + "[e1] " + state["lines"][key1] + "\n" + cm.SENTINEL, frontend.cuts
        assert state["open_by_message"]["a2"] == ["e1"]
        prewarm = frontend.cuts[1]

        third = await turn(memory, frontend, model, 3, "u3", "a2", "a3", '{"lines": {"e3": "The user asked question 3."}, "open": []}')
        assert third.layout["lined"] == 1 and third.layout["open"] == ["e1"], third.layout
        users = [m for m in third.messages if m["role"] == "user"]
        assert users[0]["content"].startswith(cm.LEDGER_OPEN + "[e1] The user asked question 1") and users[0]["content"].endswith("question 2")
        assert users[-1]["content"].startswith(cm.REFERENCE_OPEN) and 'history id="e1"' in users[-1]["content"] and users[-1]["content"].endswith("question 3")
        assert prewarm[:-1] == third.messages[:-1] and prewarm[-1]["content"] == users[-1]["content"][:-len("question 3")] + cm.SENTINEL, \
            "the pre-warm renders the next turn's layout up to its new question"
        stem, warm = frontend.cuts[-2], frontend.cuts[-1]
        assert stem[-1]["content"].startswith(cm.LEDGER_OPEN + "[e1] ") and stem[-1]["content"].endswith(cm.SENTINEL) and cm.REFERENCE_OPEN not in stem[-1]["content"]
        assert warm[-1]["content"].startswith(cm.REFERENCE_OPEN) and warm[-1]["content"].endswith(cm.SENTINEL) and warm[-2] == {"role": "assistant", "content": "answer 3"}

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

        searching = memory.session(model, headers("u3", "a2", "a3"), conversation(3), [], {})
        call = {"id": "s1", "type": "function", "function": {"name": "search_history", "arguments": "{\"query\": \"ANSWER 2\"}"}}
        assert searching.claims(call) and not searching.claims({"function": {"name": "search_web"}})
        searched = searching.continue_with("", "", [call])
        assert searched[-1]["role"] == "tool" and searched[-1]["name"] == "search_history" and "[e2 part 1]" in searched[-1]["content"] and "e1" not in searched[-1]["content"].split(":")[0], searched[-1]
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

        frontend.ready = True
        frontend.continued.clear()
        frontend.calls.clear()
        session = memory.session(model, headers("u6", "a5", "a6", chat="chat-3"), conversation(1), [], {})
        frontend.replies[:] = ["garbage", '{"lines": {"e1": "The user asked question 1."}, "open": []}']
        session.finished(session.messages, "answer 1", "", [], "stop", [5, 6], [7, 8])
        await settle(memory, model)
        assert frontend.continued == [([5, 6], [7, 8])] and len(frontend.calls) == 1, "the first curator attempt continues the answer's own tokens, a retry re-renders"
        assert memory.store.load("user-1", "chat-3")["pending"]["a6"]["source"] == "curator"
        long_chat = memory.session(model, headers("w20", None, "x20", chat="chat-long"), conversation(20), [], {})
        captured = []
        original = memory.curate

        async def capture(model_, job):
            captured.append(list(job["missing"]))
            return await original(model_, job)

        memory.curate = capture
        frontend.replies.append('{"lines": {}, "open": []}')
        long_chat.finished(long_chat.messages, "answer 20", "", [], "stop")
        await settle(memory, model)
        memory.curate = original
        assert captured and captured[0] == [0, 1, 2, 3, 4, 5], captured

        small = Model(16384)
        small.id = "small-model"
        small.memory = dict(small.memory, ledger_tokens=60, group_size=4)
        lined = conversation(12)
        _, past, _ = cm.split(lined)
        for i, messages in enumerate(past):
            memory.store.append("user-1", "chat-group", {"type": "line", "key": cm.exchange_key(messages), "line": f"The user asked question {i + 1}; the assistant gave answer {i + 1}.", "source": "curator"})
        session = memory.session(small, headers("g12", "f11", "f12", chat="chat-group"), lined, [], {})
        assert session.layout["lined"] == 10 and session.layout["groups"] == []
        frontend.replies[:] = ['{"lines": {"e12": "The user asked question 12."}, "open": [], "groups": {"g1": "Questions 1 to 4.", "g2": "Questions 5 to 8.", "g3": "Questions 9 and 10."}}']
        session.finished(session.messages, "answer 12", "", [], "stop")
        await settle(memory, small)
        groups = memory.store.load("user-1", "chat-group")["groups"]
        assert sorted(groups) == ["g1", "g2", "g3"] and all(g["source"] == "curator" for g in groups.values()), groups
        assert groups["g1"]["members"] == ["x:" + cm.exchange_key(m) for m in past[:4]] and groups["g3"]["members"] == ["x:" + cm.exchange_key(m) for m in past[8:10]]
        assert frontend.cuts[-2][-1]["content"].startswith(cm.LEDGER_OPEN + "[g1: e1-e4] Questions 1 to 4.\n[g2: e5-e8] Questions 5 to 8.\n[g3: e9-e10] Questions 9 and 10.\n[e11] ")
        grouped = memory.session(small, headers("g13", "f12", "f13", chat="chat-group"), conversation(13), [], {})
        assert grouped.layout["groups"] == ["g1", "g2", "g3"] and grouped.messages[1]["content"].startswith(cm.LEDGER_OPEN + "[g1: e1-e4] Questions 1 to 4.\n")
        assert grouped.expand({"function": {"name": "expand", "arguments": "{\"id\": \"g1\"}"}}).startswith("g1 (e1-e4):\n[e1] The user asked question 1;")
        assert grouped.expand({"function": {"name": "expand", "arguments": {"id": "g9"}}}).startswith("No group")
        memory.store.append("user-1", "chat-group", {"type": "group", "id": "g4", "members": ["g1", "g2"], "line": "Questions 1 to 8.", "source": "curator"})
        again = memory.session(small, headers("g13", "f12", "f13", chat="chat-group"), conversation(13), [], {})
        assert again.layout["groups"] == ["g1", "g2", "g3"], "a follow-up keeps its frozen groups"

        frontend.replies[:] = ["garbage", "garbage"]
        _, past, _ = cm.split(conversation(14))
        for i, messages in enumerate(past):
            memory.store.append("user-1", "chat-stub", {"type": "line", "key": cm.exchange_key(messages), "line": f"The user asked question {i + 1}; the assistant gave answer {i + 1}.", "source": "curator"})
        session = memory.session(small, headers("s14", "t13", "t14", chat="chat-stub"), conversation(14), [], {})
        session.finished(session.messages, "answer 14", "", [], "stop")
        await settle(memory, small)
        stubs = memory.store.load("user-1", "chat-stub")["groups"]
        assert stubs and all(g["source"] == "fallback" and g["line"].startswith("(auto) ") for g in stubs.values()), stubs

        tight = Model(16384)
        tight.id = "tight-model"
        tight.context_tokens = 32768
        tight.memory = dict(tight.memory, prompt_tokens=700)
        long_messages = conversation(40)
        long_messages[3] = dict(long_messages[3], content="a long reply " * 200)
        fitted = memory.session(tight, headers("t40", None, "y40", chat="chat-fit"), long_messages, [], {})
        fit_state = memory.store.load("user-1", "chat-fit")
        assert memory.prompt_size(tight, fitted.layout, []) <= 700 and fitted.layout["lined"] == 38, fitted.layout["lined"]
        assert all(source == "fit" for source in fit_state["sources"].values()) and len(fit_state["lines"]) == 38
        assert fit_state["groups"] and all(g["source"] == "fit" for g in fit_state["groups"].values()) and len(fitted.layout["nodes"]) < 38
        assert fit_state["layouts"]["t40"]["groups"] == fitted.layout["groups"]
        roomy = Model(16384)
        roomy.id = "roomy-model"
        roomy.context_tokens = 131072
        assert memory.session(roomy, headers("r40", None, "z40", chat="chat-roomy"), conversation(40), [], {}).layout["lined"] == 0, "nothing is folded while the prompt fits"

        reopened = cm.Memory({"store": tmp + "/store", "jwt_secret_file": tmp + "/secret"}, frontend).store.load("user-1", "chat-1")
        assert reopened["lines"] == state["lines"] and reopened["pending"].keys() == state["pending"].keys()


asyncio.run(run())
print("PASS test_chat_memory_session: verified identity only, system note, expand and search_history from the first turn, idle-only cancellable curation, lines bound through parent ids, ledger with open nodes, frozen follow-up layouts with replayed expansions, fallback lines, continued curator prompts, ledger-end stems, next-turn pre-warms, curator regroups, synchronous fits, state that survives a restart")
