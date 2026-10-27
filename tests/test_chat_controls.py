#!/usr/bin/env python3
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "deployment" / "vps" / "chat-local" / "index.html"
LOADER = ROOT / "deployment" / "hub" / "open_webui_loader.js"
RUN = ROOT / "deployment" / "hub" / "open_webui_run.sh"
VHOST = ROOT / "deployment" / "vps" / "sparkpipe-chat-local.conf"


def blocks(source):
    found = {}
    for language in ("en", "ko"):
        match = re.search(r"\b" + language + r":\s*\{(.*?)\n\s*\}", source, re.S)
        found[language] = set(re.findall(r"(\w+):\s*'", match.group(1))) if match else set()
    return found


def commands_module():
    spec = importlib.util.spec_from_file_location("chat_commands", ROOT / "tools" / "serving" / "chat_commands.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    failures = []
    page = LOCAL.read_text(encoding="utf-8")
    text = blocks(page)
    if not text["en"] or text["en"] != text["ko"]:
        failures.append(f"local page: English and Korean strings differ: {sorted(text['en'] ^ text['ko'])}")
    used = set(re.findall(r'data-t="(\w+)"', page)) | set(re.findall(r"\bt\('(\w+)'", page))
    if used - text["en"]:
        failures.append(f"local page: strings used but not defined: {sorted(used - text['en'])}")
    fetched = set(re.findall(r"fetch\('([^']+)'", page))
    if fetched != {"/api/models", "/api/chat/completions"}:
        failures.append(f"local page: it must call only Open WebUI's model list and chat completions on its own origin, calls {sorted(fetched)}")
    if re.search(r"<script[^>]+src=", page) or "http://" in page.replace("http://www.w3.org", ""):
        failures.append("local page: no external script and no plain-HTTP address")
    if "indexedDB.open('sparkpipe-local-chat'" not in page or "X-OpenWebUI-Chat-Id" in page:
        failures.append("local page: chats live in the browser's IndexedDB and requests carry no Open WebUI chat id, so the chat layer opens no memory")
    loader = LOADER.read_text(encoding="utf-8")
    words = blocks(loader)
    if not words["en"] or words["en"] != words["ko"]:
        failures.append(f"loader: English and Korean strings differ: {sorted(words['en'] ^ words['ko'])}")
    module = commands_module()
    sent = set(re.findall(r"send\((?:on \? )?'(/[^']+)'", loader)) | set(re.findall(r"'(/compact [^']*)'", loader))
    sent |= {"/compact 50%", "/memory on", "/memory off", "/compact full", "/compact undo", "/context"}
    for command in sorted(sent):
        if module.command_of({"role": "user", "content": command}) is None:
            failures.append(f"loader: the chat layer does not recognize {command!r}")
    if '"$D/branding/loader.js:/app/build/static/loader.js:ro"' not in RUN.read_text(encoding="utf-8"):
        failures.append("open_webui_run.sh: the loader is not mounted where Open WebUI's page loads it")
    vhost = VHOST.read_text(encoding="utf-8")
    if "ProxyPass /local/ !" not in vhost or "Alias /local/ /var/www/chat-local/" not in vhost:
        failures.append("sparkpipe-chat-local.conf: /local/ must be served from /var/www/chat-local and kept out of the Open WebUI proxy")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS chat controls: the toolbar and local page carry English and Korean for every string, the toolbar sends only commands the chat layer answers, and local mode calls only Open WebUI's own API with no chat id")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
