#!/usr/bin/env python3
import asyncio
import importlib.util
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWUI = ROOT / "tools" / "serving" / "open_webui"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAIL test_open_webui_chat_tools: {message}")


cm = load("chat_memory", ROOT / "tools" / "serving" / "chat_memory.py")
cc = load("chat_commands", ROOT / "tools" / "serving" / "chat_commands.py")
status = load("context_status", OWUI / "functions" / "context_status.py")
timer = load("chat_timer", OWUI / "functions" / "chat_timer.py")
toggle = load("timer_toggle", OWUI / "functions" / "timer_toggle.py")
tool = load("chat_timer_tool", OWUI / "tools" / "chat_timer.py")
spec = json.loads((ROOT / "deployment" / "hub" / "open_webui_functions.json").read_text())

secret = "k" * 48
claims = cm.verify_jwt(status.user_token("user-7", secret), secret)
check(claims and claims["sub"] == "user-7" and claims["iss"] == "open-webui", "the status filter signs the user header the chat layer verifies")
check(cc.command_of({"role": "user", "content": status.COMMAND}) == ("context", "json"), "the status filter asks the chat layer's /context json")
state = {"context_tokens": 1048576, "history_tokens": 9000, "memory": "on", "target": 65536, "compaction": None}
line = status.describe(state, {"prompt_tokens": 32613, "prompt_tokens_details": {"cached_tokens": 31164}})
check(line == "Context 32K of 1M (3.1%) · 30K cached · memory on, target 64K", line)
line = status.describe(dict(state, memory="off", target=None, compaction={"kept": 9, "of": 21}), None)
check(line == "Context 8.8K of 1M (0.9%) · memory off · compacted to 9 of 21 exchanges", line)

check(getattr(toggle.Filter(), "toggle", False) is True and timer.TOGGLE in [f["id"] for f in spec["functions"]], "the Timer box is a toggle the chat timer filter looks for")
check(not hasattr(timer.Filter(), "toggle"), "the chat timer filter always runs, so /timer is kept while the box is unticked")
for text in ("every 10m check the build", "every 2h summarize the news", "EVERY 30 s ping"):
    check(bool(cc.TIMER.match(text)) == bool(timer.EVERY.match(text)), f"both sides parse {text!r} the same way")

with tempfile.TemporaryDirectory() as tmp:
    os.environ["DATA_DIR"] = tmp
    check(timer.apply_command("chat-1", "/timer every 10m check the build", "user") == {"prompt": "check the build", "every": "600s", "by": "user"}, "the user's /timer is stored")
    check(timer.apply_command("chat-1", "/timer every 5s hi", "user")["every"] == "60s", "a timer runs at most once a minute")
    check(timer.apply_command("chat-1", "hello", "user") is None and timer.load()["chat-1"]["prompt"] == "hi", "other messages leave the timer alone")
    body = {"messages": [{"role": "user", "content": "/timer every 1h report"}]}
    asyncio.run(timer.Filter().inlet(body, {"chat_id": "chat-2"}))
    check(timer.load()["chat-2"]["every"] == "3600s", "the inlet stores the command of the newest user message")
    reply = asyncio.run(tool.Tools().set_chat_timer("check the deploy", "15m", __chat_id__="chat-1"))
    check(timer.load()["chat-1"] == {"prompt": "check the deploy", "every": "900s", "by": "model"} and "900 s" in reply, "the model's tool writes the same record")
    check("Not set" in asyncio.run(tool.Tools().set_chat_timer("x", "soon", __chat_id__="chat-1")), "the tool refuses an interval it cannot parse")
    asyncio.run(tool.Tools().stop_chat_timer(__chat_id__="chat-1"))
    check(timer.apply_command("chat-2", "/timer off", "user") is None and timer.load() == {}, "stop and off clear the chat's timer")
    unticked = {"chat_id": "chat-3", "filter_ids": []}
    timer.apply_command("chat-3", "/timer every 10m x", "user")
    check(asyncio.run(timer.Filter().outlet(unticked, {"id": "u"}, {}, object())) is unticked, "an unticked box arms nothing")

for entry in spec["functions"] + spec["tools"]:
    check((OWUI / entry["file"]).is_file(), f"{entry['file']} exists")
for prompt in spec["prompts"]:
    check(cc.command_of({"role": "user", "content": prompt["content"].strip() or "/x"}) is not None, f"/{prompt['command']} inserts a command the chat layer answers")
models = json.loads((ROOT / "deployment" / "hub" / "open_webui_models.json").read_text())["defaults"]
check(models["capabilities"].get("usage") is True, "models report token usage, which the status line reads")
check("sparkpipe_chat_timer_tool" in models["toolIds"] and any(entry["id"] == "sparkpipe_chat_timer_tool" for entry in spec["tools"]), "every model may use the timer tool")
print("PASS test_open_webui_chat_tools: the context status signs the user header and reads the chat layer's /context json; the Timer box gates the always-on chat timer filter; /timer and the model's set_chat_timer share one store and one syntax; every slash prompt inserts a chat-layer command; models report usage")
