"""The Full context toggle is a global Open WebUI filter that puts the chat layer's own memory-off marker in front of the request's messages; the chat layer then drops that marker message and serves the turn without memory."""
import asyncio
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cm = load("chat_memory", ROOT / "tools" / "serving" / "chat_memory.py")
toggle = load("memory_off", ROOT / "tools" / "serving" / "open_webui" / "functions" / "memory_off.py")
spec = json.loads((ROOT / "deployment" / "hub" / "open_webui_functions.json").read_text())


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAIL test_open_webui_memory_off: {message}")


check(toggle.MEMORY_OFF == cm.MEMORY_OFF, "the filter and the chat layer share one marker")
entry = [item for item in spec["functions"] if item["id"] == "sparkpipe_memory_off"]
check(len(entry) == 1 and entry[0]["global"] is True and (ROOT / "tools" / "serving" / "open_webui" / entry[0]["file"]).is_file(), "the toggle is installed as a global function for every model")
check(toggle.Filter().toggle is True, "the filter is a user toggle, off unless the user turns it on")
history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}, {"role": "user", "content": "second"}]
body = asyncio.run(toggle.Filter().inlet({"model": "m", "messages": list(history)}))
cleaned, off = cm.memory_switch(body["messages"])
check(off and cleaned == history, f"the chat layer sees memory off and the transcript unchanged: {cleaned}")
print("PASS test_open_webui_memory_off: the Full context toggle adds the chat layer's memory-off marker, which the chat layer removes before serving the turn without memory")
