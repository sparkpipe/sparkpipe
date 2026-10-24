#!/usr/bin/env python3
import asyncio
import base64
import hashlib
import hmac
import importlib.util
import json
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cm = load("chat_memory", ROOT / "tools" / "serving" / "chat_memory.py")
cc = load("chat_commands", ROOT / "tools" / "serving" / "chat_commands.py")
SECRET = "s" * 48
BUDGETS = json.loads((ROOT / "deployment" / "hub" / "chat_frontend.json").read_text())["chat_memory"]["budgets"]


def token(sub="user-1"):
    b = lambda data: base64.urlsafe_b64encode(data).rstrip(b"=").decode()
    head = b(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b(json.dumps({"sub": sub, "iss": "open-webui", "exp": time.time() + 600}).encode())
    return f"{head}.{body}.{b(hmac.new(SECRET.encode(), f'{head}.{body}'.encode(), hashlib.sha256).digest())}"


class Model:
    id = "fake"
    context_tokens = 131072
    reasoning = None
    memory = {}


class Frontend:
    def __init__(self):
        self.reply = ""
        self.asked = []

    def memory_count(self, model, text):
        return len(text) // 4

    def memory_curator_ids(self, model, messages, tools, kwargs, suffix):
        self.asked.append(messages)
        return [1, 2, 3], False

    async def memory_complete(self, model, ids, max_tokens, reasoning_open=False, kind="background", trace=None):
        assert kind == "compact" and max_tokens >= 1024
        return self.reply


def chat(n, words=40):
    messages = [{"role": "system", "content": "Be helpful."}]
    for i in range(1, n + 1):
        messages += [{"role": "user", "content": f"question {i} " + "word " * words}, {"role": "assistant", "content": f"answer {i} " + "word " * words}]
    return messages


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAIL test_chat_commands: {message}")


async def run():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "secret").write_text(SECRET)
        frontend = Frontend()
        memory = cm.Memory({"store": tmp + "/store", "jwt_secret_file": tmp + "/secret", "budgets": BUDGETS}, frontend)
        commands = cc.Commands(memory, cm, frontend)
        model = Model()
        headers = {"X-OpenWebUI-User-Jwt": token(), "X-OpenWebUI-Chat-Id": "chat-1"}
        identity = commands.identity(headers)
        check(identity == ("user-1", "chat-1"), identity)
        check(commands.identity({"X-OpenWebUI-Chat-Id": "chat-1"}) is None, "a request without a verified user has no chat identity")

        check(cc.command_of({"role": "user", "content": "/compact 50%"}) == ("compact", "50%"), "a command and its argument are read from the user message")
        check(cc.command_of({"role": "user", "content": "please /compact"}) is None, "a command starts the message")
        check(cc.command_of({"role": "assistant", "content": "/memory off"}) is None, "only user messages are commands")
        check([cc.size(text, 1000) for text in ("50%", "40k", "1M", "300", "0", "150%", "x")] == [500, 40960, 1048576, 300, None, None, None], "sizes")
        history = chat(2) + [{"role": "user", "content": "/memory off"}, {"role": "assistant", "content": "Memory is off for this chat."},
                             {"role": "user", "content": "next"}]
        check(cc.strip(history) == chat(2) + [{"role": "user", "content": "next"}], "a command and its answer are left out of the prompt")

        check("on" in await commands.run(model, identity, chat(1) + [{"role": "user", "content": "/memory"}], "memory", "", False), "memory reports on by default")
        await commands.run(model, identity, [], "memory", "off", False)
        check(commands.settings(identity).get("memory") == "off", "memory off is a chat setting")
        reply = await commands.run(model, identity, [], "target", "64K", False)
        check(commands.settings(identity).get("target") == 65536 and "64K" in reply, reply)
        await commands.run(model, identity, [], "target", "25%", False)
        check(commands.settings(identity).get("target") == 32768, "a share of the model's context")
        await commands.run(model, identity, [], "target", "off", False)
        check(commands.settings(identity).get("target") is None, "target off clears it")
        check("every 10m" in commands.timer("every 10m check the build") and "Timer box" in commands.timer("every 10m check the build"), "the timer reply echoes the schedule")
        check("off" in commands.timer("off") and "every 10m <prompt>" in commands.timer("whenever"), "timer off and usage")
        report = await commands.run(model, identity, chat(3) + [{"role": "user", "content": "/context"}], "context", "", False)
        check("3 exchanges" in report and "Memory is off" in report, report)
        state = json.loads(await commands.run(model, identity, chat(3) + [{"role": "user", "content": "/context json"}], "context", "json", False))
        check(state["context_tokens"] == 131072 and state["exchanges"] == 3 and state["memory"] == "off" and state["history_tokens"] > 0, state)

        headers2 = {"X-OpenWebUI-User-Jwt": token(), "X-OpenWebUI-Chat-Id": "chat-2"}
        identity2 = commands.identity(headers2)
        refused = await commands.run(model, identity2, chat(4) + [{"role": "user", "content": "/compact 50%"}], "compact", "50%", False)
        check("Memory is on" in refused and memory.store.load(*identity2)["compaction"] is None, "compaction is for chats with memory off")

        long_chat = chat(6)
        frontend.reply = 'Here: {"keep": ["e2", "e5"], "summary": "The third question set the budget at 4,000 euros (e3)."}'
        reply = await commands.run(model, identity, long_chat + [{"role": "user", "content": "/compact 50%"}], "compact", "50%", False)
        check("kept e2, e5, e6" in reply, reply)
        listing = frontend.asked[-1][1]["content"]
        check("[e1]" in listing and "[e6]" in listing and "always keep e6" in listing, "the model sees every exchange and must keep the newest")
        compacted = commands.compacted(identity, long_chat + [{"role": "user", "content": "question 7"}])
        users = [m["content"] for m in compacted if m["role"] == "user"]
        check(len(users) == 4 and users[0].startswith(cc.SUMMARY_OPEN) and "4,000 euros" in users[0] and "question 2" in users[0], users[:1])
        summary = memory.store.load(*identity)["compaction"]["summary"]
        check(all(f"({item})" in summary for item in ("e1", "e3", "e4")), "every dropped exchange is in the summary: the model's (e3) and fallback lines for e1 and e4")
        check([u.split(" ")[1] for u in users[1:]] == ["5", "6", "7"], "kept exchanges stay in order, later turns follow")
        check(compacted[0]["role"] == "system", "the system message stays first")

        frontend.reply = '{"keep": ["e1", "e2", "e3", "e4", "e5", "e6"], "summary": "nothing"}'
        await commands.run(model, identity, long_chat + [{"role": "user", "content": "/compact 30%"}], "compact", "30%", False)
        compaction = memory.store.load(*identity)["compaction"]
        sizes = commands.sizes(model, cm.split(long_chat + [{"role": "user", "content": ""}])[1])
        check(len(compaction["keep"]) < 6 and compaction["target"] == int(sum(sizes) * 0.3), "a choice over budget drops the oldest kept exchanges")
        frontend.reply = '{"keep": [], "summary": "Everything so far in one paragraph."}'
        reply = await commands.run(model, identity, long_chat + [{"role": "user", "content": "/compact full"}], "compact", "full", False)
        compaction = memory.store.load(*identity)["compaction"]
        check(compaction["keep"] == [] and "summarized the other 6" in reply, reply)
        frontend.reply = "not json"
        reply = await commands.run(model, identity, long_chat + [{"role": "user", "content": "/compact 50%"}], "compact", "50%", False)
        check("nothing changed" in reply and memory.store.load(*identity)["compaction"]["summary"].startswith("Everything"), "a bad answer changes nothing")
        await commands.run(model, identity, [], "compact", "undo", False)
        check(commands.compacted(identity, long_chat + [{"role": "user", "content": "q"}]) == long_chat + [{"role": "user", "content": "q"}], "undo restores the full chat")
        check(await commands.run(model, None, [], "context", "", False) == "Commands need a signed-in Open WebUI chat: this request carries no chat identity.", "no identity")


asyncio.run(run())
print("PASS test_chat_commands: /context, /memory, /target and /timer answer without the model and persist per chat; /compact keeps the model's most relevant exchanges within the target (dropping the oldest over budget), always the newest, summarizes the rest, refuses with memory on, ignores a bad answer and undoes; command exchanges never reach a prompt")
