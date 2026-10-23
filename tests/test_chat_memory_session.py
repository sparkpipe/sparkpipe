#!/usr/bin/env python3
"""A chat-memory session is opened only for a verified Open WebUI identity on a real chat turn, and its bookkeeping holds across turns: requests without a valid token, with a task, without ids, with a client tool named like a memory tool or with Open WebUI's own summary are served raw; the first turns carry the system note and the expand and search_history tools but no ledger, and every finished exchange stays open with its id; a search_history call is answered inside the chat layer; after a finished answer the curator runs only once the engine is idle and is cancelled by a foreground request; its line for the current exchange binds on the next turn through the parent message id, older raw exchanges get their lines directly, and the open set follows; once the open exchanges pass the model's working budget the curator commits those least likely to be needed in full, with its earlier line or a final line that keeps the earlier facts, the oldest open exchange is committed when it frees too little, and the ledger lists commits in commit order; follow-ups reuse the frozen layout and replay each expansion at the tool round where it ran, with the text streamed before it said once, and only for the attempt that made it; two invalid curator replies leave fallback lines; the first curator attempt continues the answer's own prompt and reply tokens; checkpointed models get a ledger-end stem; the next turn's layout is pre-warmed up to its new question; a ledger over its budget is regrouped by the curator into nested group lines that expand to their children (stub lines when the curator fails), and follow-ups keep their frozen groups; and a prompt over the model's budget is fitted synchronously by dropping the open set, lining raw exchanges and folding the oldest ledger entries, while one that fits is left alone."""
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


BUDGETS = {"open_budget": 1500, "line_tokens": 48, "group_size": 16, "working_tokens": 24576, "working_context_share": 0.1875, "ledger_tokens": 12288,
           "ledger_context_share": 0.1, "part_small_tokens": 1024, "part_cap_tokens": 12288, "keep_tokens": 4096, "target_tolerance": 0.25,
           "target_minimum_tokens": 16384}


class Model:
    def __init__(self):
        self.id = "fake-model"
        self.reasoning = {"start": "<think>", "end": "</think>"}
        self.memory = {}


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

    async def memory_complete(self, model, ids, max_tokens, reasoning_open=False, kind="background", trace=None):
        assert kind in ("curator", "curator_render", "stem", "prewarm") and set(trace) == {"chat", "user_message"}, (kind, trace)
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
        memory = cm.Memory({"store": tmp + "/store", "jwt_secret_file": tmp + "/secret", "idle_seconds": 0.01, "budgets": BUDGETS}, frontend)
        model = Model()

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
        plain = type("Plain", (), {"id": "plain", "reasoning": None})()
        assert memory.session(plain, headers("u1", chat="plain-chat"), conversation(1), [], {}) is not None, "every model gets memory, with the shared budgets"
        assert memory.profile(plain).curator_suffix is None and memory.profile(plain).working_tokens == BUDGETS["working_tokens"]
        assert memory.profile(model).curator_suffix == "</think>"
        small_context = type("Small", (), {"id": "small", "reasoning": None, "context_tokens": 32768})()
        assert (memory.profile(small_context).working_tokens, memory.profile(small_context).ledger_tokens) == (6144, 3276), "budgets shrink with a small context"
        closing = type("Closing", (), {"id": "closing", "reasoning": {"start": "<a>", "end": "</a>", "close": "\n</a><b>"}, "memory": {"ledger_tokens": 99}})()
        assert memory.profile(closing).curator_suffix == "\n</a><b>" and memory.profile(closing).ledger_tokens == 99
        long_context = type("Long", (), {"id": "long", "reasoning": None, "context_tokens": 262144})()
        aimed = memory.profile(long_context, 100000)
        assert (aimed.target, aimed.prompt_tokens, aimed.ledger_tokens, aimed.working_tokens, aimed.working_low) == (100000, 125000, 10000, 113500, 63500), \
            "a target lets the prompt reach the target plus the tolerance and compacts the working set back to the target minus it"
        limit = 262144 - 262144 * 15 // 100
        topped = memory.profile(long_context, 10 ** 7)
        assert (topped.target, topped.prompt_tokens, topped.working_tokens, topped.working_low) == (262144, limit, limit - 13788, int(limit * 0.5) - 13788), \
            "at the true maximum the target is a hard limit with no overshoot"
        assert memory.profile(long_context, 1000).target == BUDGETS["target_minimum_tokens"], "a target below the minimum is raised to it"
        untargeted = memory.profile(long_context)
        assert untargeted.target is None and untargeted.working_low == untargeted.working_tokens // 2, "without a target the shared budgets apply"
        steered = Model()
        steered.id = "steered"
        steered.memory = {"target_minimum_tokens": 1, "open_budget": 0, "ledger_context_share": 0.0}
        long_chat = [dict(m, content=m["content"] + " " + "word " * 80) for m in conversation(5)]
        near = memory.session(steered, headers("t5", chat="chat-target"), long_chat, [], {}, 2000)
        assert near.target == 2000 and near.layout["working"] == [0, 1, 2, 3], ("the prompt is inside the target plus its tolerance", near.layout["working"])
        plan = memory.plan_commit(steered, near, "answer 5 " + "word " * 2000, [])
        assert plan is not None and plan["total"] > 2500 and plan["needed"] == plan["total"] - 1500, plan
        far = memory.session(steered, headers("t5", chat="chat-untargeted"), long_chat, [], {})
        assert memory.plan_commit(steered, far, "answer 5 " + "word " * 2000, []) is None, "the same chat without a target stays inside the shared working budget"
        try:
            memory.profile(type("Bad", (), {"id": "bad", "memory": {"curator_suffix": "x"}})())
            raise AssertionError("an override outside the budget keys must be refused")
        except SystemExit:
            pass
        off = cm.MEMORY_OFF
        cases = [
            ([{"role": "system", "content": off}, {"role": "user", "content": "hi"}], [{"role": "user", "content": "hi"}], True),
            ([{"role": "system", "content": "Be brief." + off}, {"role": "user", "content": "hi"}], [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}], True),
            ([{"role": "user", "content": [{"type": "text", "text": "hi" + off}]}], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], True),
            ([{"role": "user", "content": "hi"}], [{"role": "user", "content": "hi"}], False),
        ]
        for given, cleaned, switched in cases:
            assert cm.memory_switch(given) == (cleaned, switched), given

        working = Model()
        working.id = "working-model"
        working.memory = dict(working.memory, working_tokens=5)
        first = await turn(memory, frontend, working, 1, "u1", None, "a1", '{"lines": {"e1": "The user asked question 1; the assistant gave answer 1."}, "open": []}')
        assert len(frontend.cuts) == 1 and frontend.cuts[0][-1]["content"] == cm.REFERENCE_OPEN + cm.REFERENCE_CLOSE + "[e2] " + cm.SENTINEL and cm.LEDGER_OPEN not in json.dumps(frontend.cuts[0]), \
            "no stem while the ledger is empty, but the next turn is pre-warmed"
        assert cm.SYSTEM_NOTE in first.messages[0]["content"] and [t["function"]["name"] for t in first.tools[-2:]] == ["expand", "search_history"] and first.layout["lined"] == 0
        assert first.messages[-1]["content"].endswith("[e1] question 1")
        state = memory.store.load("user-1", "chat-1")
        assert state["pending"]["a1"]["line"].startswith("The user asked question 1"), state["pending"]
        assert "user-1" not in json.dumps(list(Path(tmp, "store").rglob("*")).__repr__())

        frontend.calls.clear()
        second = await turn(memory, frontend, working, 2, "u2", "a1", "a2", '{"lines": {"e2": "The user asked question 2; the assistant gave answer 2."}, "open": ["e1"], "commit": {"e1": "="}}')
        key1 = cm.exchange_key(conversation(2)[1:3])
        assert state["lines"][key1].startswith("The user asked question 1") and second.layout["lined"] == 0 and second.layout["working"] == [0]
        assert '"commit"' in frontend.calls[-1][-1]["content"] and "e1 " in frontend.calls[-1][-1]["content"].split("Open exchanges")[-1], "an over-budget working set asks the curator what to commit"
        assert state["commits"] == [key1] and key1 not in state["final"], "the curator committed e1 with its earlier line"
        assert len(frontend.cuts) == 3 and frontend.cuts[1][-1]["content"] == cm.LEDGER_OPEN + "[e1] " + state["lines"][key1] + "\n" + cm.SENTINEL, frontend.cuts
        assert state["open_by_message"]["a2"] == ["e1"]
        prewarm = frontend.cuts[2]

        third = await turn(memory, frontend, working, 3, "u3", "a2", "a3", '{"lines": {"e3": "The user asked question 3."}, "open": [], "commit": {"e2": "The user asked question 2, and later questions built on its answer 2."}}')
        assert third.layout["lined"] == 1 and third.layout["open"] == ["e1"] and third.layout["working"] == [1], third.layout
        users = [m for m in third.messages if m["role"] == "user"]
        assert users[0]["content"] == cm.LEDGER_OPEN + "[e1] " + state["lines"][key1] + "\n" + cm.LEDGER_CLOSE + "[e2] question 2"
        assert users[-1]["content"].startswith(cm.REFERENCE_OPEN) and 'history id="e1"' in users[-1]["content"] and users[-1]["content"].endswith("[e3] question 3")
        assert prewarm[:-1] == third.messages[:-1] and prewarm[-1]["content"] == users[-1]["content"][:-len("question 3")] + cm.SENTINEL, \
            "the pre-warm renders the next turn's layout up to its new question"
        key2 = cm.exchange_key(conversation(3)[3:5])
        assert state["commits"] == [key1, key2] and state["final"][key2].startswith("The user asked question 2, and later"), "a final line written with hindsight that keeps the earlier facts is used"
        stem, warm = frontend.cuts[-2], frontend.cuts[-1]
        assert stem[-1]["content"] == cm.LEDGER_OPEN + "[e1] " + state["lines"][key1] + "\n[e2] " + state["final"][key2] + "\n" + cm.SENTINEL
        assert warm[-1]["content"].startswith(cm.REFERENCE_OPEN) and warm[-1]["content"].endswith("[e4] " + cm.SENTINEL) and warm[-2] == {"role": "assistant", "content": "answer 3"}
        fourth = await turn(memory, frontend, working, 4, "u4w", "a3", "a4w", '{"lines": {"e4": "The user asked question 4."}, "open": []}')
        key3 = cm.exchange_key(conversation(4)[5:7])
        assert fourth.layout["working"] == [2] and state["commits"][-1] == key3, "when the curator commits too little, the oldest open exchange is committed"
        record = [json.loads(line) for line in memory.store.path("user-1", "chat-1").read_text().splitlines() if '"commit"' in line][-1]
        assert record["key"] == key3 and record["source"] == "fill"
        refused_final = cm.parse_commits('{"commit": {"e1": "The user asked a question."}}', {"sizes": {0: 5}}, conversation(2)[1:2] and [conversation(2)[1:3]], {key1: "The user asked question 1 about Ostrel-7."})
        assert refused_final == {0: None}, "a final line that drops a name or number of the earlier line falls back to the earlier line"

        cold = memory.session(model, headers("v3", "b2", "b3", chat="chat-2"), conversation(3), [], {})
        assert cold.layout["lined"] == 0
        memory.store.append("user-1", "chat-2", {"type": "line", "key": cm.exchange_key(conversation(3)[1:3]), "line": "arrived mid-exchange", "source": "curator"})
        memory.store.append("user-1", "chat-2", {"type": "commit", "key": cm.exchange_key(conversation(3)[1:3]), "source": "curator"})
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

        searched_web = conversation(3) + [{"role": "assistant", "content": "", "tool_calls": [{"id": "w1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}]},
                                          {"role": "tool", "tool_call_id": "w1", "content": "web results"}]
        looking = memory.session(model, headers("r3", "a2", "r3a", chat="chat-round"), searched_web, [], {})
        looking.continue_with("Let me check the notes. ", "", [{"id": "x7", "type": "function", "function": {"name": "expand", "arguments": "{\"id\": \"e1\"}"}}])
        followed = searched_web + [{"role": "assistant", "content": "Let me check the notes. Now I fetch it.", "tool_calls": [{"id": "f1", "type": "function", "function": {"name": "fetch_url", "arguments": "{}"}}]},
                                   {"role": "tool", "tool_call_id": "f1", "content": "page"}]
        resumed = memory.session(model, headers("r3", "a2", "r3a", chat="chat-round"), followed, [], {})
        tail = [(m["role"], m.get("content"), m.get("name")) for m in resumed.messages[-6:]]
        assert tail[0][0] == "assistant" and tail[1][1].startswith("[e3.t1 search_web · ") and tail[1][1].endswith("]\nweb results") and tail[2][1] == "Let me check the notes. " \
            and tail[3][2] == "expand" and tail[4][:2] == ("assistant", "Now I fetch it.") and tail[5][1].startswith("[e3.t2 fetch_url · ") and tail[5][1].endswith("page"), tail
        retried = memory.session(model, headers("r3", "a2", "r3b", chat="chat-round"), followed, [], {})
        assert not any(m.get("name") == "expand" for m in retried.messages), "a regenerate never replays another attempt's expansions"
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

        small = Model()
        small.id = "small-model"
        small.memory = dict(small.memory, ledger_tokens=60, group_size=4)
        lined = conversation(12)
        _, past, _ = cm.split(lined)
        for i, messages in enumerate(past):
            memory.store.append("user-1", "chat-group", {"type": "line", "key": cm.exchange_key(messages), "line": f"The user asked question {i + 1}; the assistant gave answer {i + 1}.", "source": "curator"})
            if i < 10:
                memory.store.append("user-1", "chat-group", {"type": "commit", "key": cm.exchange_key(messages), "source": "curator"})
        session = memory.session(small, headers("g12", "f11", "f12", chat="chat-group"), lined, [], {})
        assert session.layout["lined"] == 10 and session.layout["groups"] == []
        frontend.replies[:] = ['{"lines": {"e12": "The user asked question 12."}, "open": [], "groups": {"g1": "Questions 1 to 4.", "g2": "Questions 5 to 8.", "g3": "Questions 9 and 10."}}']
        session.finished(session.messages, "answer 12", "", [], "stop")
        await settle(memory, small)
        groups = memory.store.load("user-1", "chat-group")["groups"]
        assert sorted(groups) == ["g1", "g2", "g3"] and all(g["source"] == "curator" for g in groups.values()), groups
        assert groups["g1"]["members"] == ["x:" + cm.exchange_key(m) for m in past[:4]] and groups["g3"]["members"] == ["x:" + cm.exchange_key(m) for m in past[8:10]]
        assert frontend.cuts[-2][-1]["content"] == cm.LEDGER_OPEN + "[g1: e1-e4] Questions 1 to 4.\n[g2: e5-e8] Questions 5 to 8.\n[g3: e9-e10] Questions 9 and 10.\n" + cm.SENTINEL
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
            memory.store.append("user-1", "chat-stub", {"type": "commit", "key": cm.exchange_key(messages), "source": "curator"})
        session = memory.session(small, headers("s14", "t13", "t14", chat="chat-stub"), conversation(14), [], {})
        session.finished(session.messages, "answer 14", "", [], "stop")
        await settle(memory, small)
        stubs = memory.store.load("user-1", "chat-stub")["groups"]
        assert stubs and all(g["source"] == "fallback" and g["line"].startswith("(auto) ") for g in stubs.values()), stubs

        tight = Model()
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
        roomy = Model()
        roomy.id = "roomy-model"
        roomy.context_tokens = 131072
        assert memory.session(roomy, headers("r40", None, "z40", chat="chat-roomy"), conversation(40), [], {}).layout["lined"] == 0, "nothing is folded while the prompt fits"

        page = "".join(f"Paragraph {i}: the scheduler gained {i * 3} percent after fix #{1000 + i}.\n" for i in range(400))
        researched = [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "Read the release notes."},
                      {"role": "assistant", "content": "", "tool_calls": [{"id": "p1", "type": "function", "function": {"name": "fetch_url", "arguments": {"url": "https://a.example/notes"}}}]},
                      {"role": "tool", "tool_call_id": "p1", "content": "[This result is source [1]; cite it inline as [1].]\n" + page,
                       "citation_label": "[This result is source [1]; cite it inline as [1].]\n"}]
        memory.store.append("user-1", "chat-parts", {"type": "line", "key": cm.exchange_key(researched[1:3]), "line": "The user greeted the assistant.", "source": "curator"})
        reading = memory.session(model, headers("k1", None, "k1a", chat="chat-parts"), researched, [], {})
        shown_page = reading.messages[-1]["content"]
        assert shown_page.startswith("[e2.t1 fetch_url ") and "[§1] Paragraph 0:" in shown_page and "[§2] " in shown_page, shown_page[:200]
        frontend.calls.clear()
        frontend.replies[:] = ['{"lines": {"e2": "The user asked to read the release notes; the assistant summarized them."}, "open": [], "keep": {"e2.t1": [2, 3, 999], "e9.t9": [1]}}']
        reading.finished(reading.messages, "The notes list 400 paragraphs.", "", [], "stop")
        await settle(memory, model)
        instruction = frontend.calls[-1][-1]["content"]
        assert "e2.t1 fetch_url" in instruction and '"keep"' in instruction and "§1-§" in instruction, instruction[-400:]
        part = cm.tool_parts(researched)[0]
        assert memory.store.load("user-1", "chat-parts")["forms"][part["key"]] == [2, 3], "only shown sections of known parts are kept"
        assert "[This result is source [1]" in shown_page, "the citation label shows in its own turn"
        after = researched[:-1] + [{"role": "tool", "tool_call_id": "p1", "content": page}, {"role": "assistant", "content": "The notes list 400 paragraphs."},
                                   {"role": "user", "content": "Which fix helped most?"}]
        later = memory.session(model, headers("k2", "k1a", "k2a", chat="chat-parts"), after, [], {})
        collapsed = next(m["content"] for m in later.messages if m["role"] == "tool")
        assert "collapsed; kept §2, §3" in collapsed and "[§2] " in collapsed and "[§3] " in collapsed and "Paragraph 0:" not in collapsed, collapsed[:300]
        assert len(collapsed) < len(page) // 4, "the newest exchange's large result shrinks to a stub plus the kept sections"
        assert frontend.cuts[-1][-3]["role"] == "tool" and frontend.cuts[-1][-3]["content"] == collapsed and "source [1]" not in collapsed, \
            "the pre-warm renders the collapsed form the next turn uses, without the turn's citation label"
        read_back = later.expand({"function": {"name": "expand", "arguments": {"id": "e2.t1", "part": 10}}})
        assert read_back.startswith("e2.t1 (fetch_url), sections §10-") and "[§10] " in read_back, read_back[:120]
        assert later.expand({"function": {"name": "expand", "arguments": {"id": "e2.t7"}}}).startswith("No tool result")

        reopened = cm.Memory({"store": tmp + "/store", "jwt_secret_file": tmp + "/secret", "budgets": BUDGETS}, frontend).store.load("user-1", "chat-1")
        assert reopened["lines"] == state["lines"] and reopened["pending"].keys() == state["pending"].keys()


asyncio.run(run())
print("PASS test_chat_memory_session: every model gets memory with the shared budgets or a context target held within its tolerance (a hard limit at the true maximum) and a curator suffix from its reasoning close (or end) marker, a memory-off marker removed before rendering, verified identity only, system note, expand and search_history from the first turn, idle-only cancellable curation, lines bound through parent ids, ledger with open nodes, frozen follow-up layouts with replayed expansions, fallback lines, continued curator prompts, ledger-end stems, next-turn pre-warms, curator regroups, synchronous fits, tool results with ids and sections, collapsed after their turn to the kept sections, state that survives a restart")
