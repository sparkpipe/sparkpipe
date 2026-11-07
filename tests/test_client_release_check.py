#!/usr/bin/env python3
import contextlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import client_release_check as check

KEY = "sk-staging-REALKEY0123456789"
MODELS = {"data": [{"id": "kimi-k3", "context_window": 262144, "max_output_tokens": 32768, "sequences": 8}]}
ALIASES = {"aliases": {"claude-*": "kimi-k3", "claude-haiku-*": "kimi-k3", "gpt-*": "kimi-k3"}, "first_resident": "kimi-k3"}

NPM = r'''
import json, os, sys
args = sys.argv[1:]
with open(LOG, "a") as log:
    log.write(json.dumps({"argv": args, "env": sorted(os.environ), "key": os.environ.get("SPARKPIPE_API_KEY")}) + "\n")
if args[0] == "view":
    print("3.0.0")
    sys.exit(0)
prefix = args[args.index("--prefix") + 1]
package, _, version = args[-1].rpartition("@")
if version.startswith("0.0.0"):
    sys.exit(1)
name = {"@anthropic-ai/claude-code": "claude", "@openai/codex": "codex", "opencode-ai": "opencode"}[package]
target = os.path.join(prefix, "node_modules", ".bin", name)
os.makedirs(os.path.dirname(target), exist_ok=True)
with open(target, "w") as out:
    out.write("#!" + PYTHON + "\n" + open(TEMPLATE).read().replace("__NAME__", name).replace("__VERSION__", version))
os.chmod(target, 0o755)
'''

TOOL = r'''
import http.client, json, os, subprocess, sys, tomllib, urllib.parse, urllib.request, uuid
NAME, VERSION = "__NAME__", "__VERSION__"
HOME = os.environ["HOME"]
args = sys.argv[1:]
with open(os.path.join(HOME, "tool.log"), "a") as log:
    log.write(json.dumps({"name": NAME, "argv": args, "key": os.environ.get("SPARKPIPE_API_KEY")}) + "\n")
if args == ["--version"]:
    print(f"{VERSION} ({NAME})")
    sys.exit(0)
direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, body, headers):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=dict(headers, **{"Content-Type": "application/json"}))
    with direct.open(request, timeout=30) as response:
        return json.loads(response.read() or b"{}")


def mcp(url, headers):
    try:
        post(url, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, headers)
        return "connected"
    except OSError:
        return "failed"


def phone_home():
    proxy = urllib.parse.urlsplit(os.environ["HTTPS_PROXY"])
    connection = http.client.HTTPConnection(proxy.hostname, proxy.port, timeout=10)
    connection.set_tunnel("opencode.ai", 443)
    try:
        connection.connect()
    except OSError:
        pass


def answer(prompt, session):
    path = os.path.join(HOME, "sessions", session + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = json.load(open(path)) if os.path.exists(path) else {}
    if "NOTES.md" in prompt:
        state["word"] = open("NOTES.md").read().split()[0]
        text, tools = state["word"], 1
    elif "git log" in prompt:
        text, tools = subprocess.run(["git", "log", "-1", "--format=%s"], capture_output=True, text=True).stdout.strip(), 1
    else:
        text, tools = state.get("word", "?"), 0
    json.dump(state, open(path, "w"))
    return text, tools


def emit(*items):
    for item in items:
        print(json.dumps(item))


if NAME == "claude":
    prompt = args[args.index("-p") + 1]
    session = args[args.index("--resume") + 1] if "--resume" in args else str(uuid.uuid4())
    env = json.load(open(os.path.join(HOME, ".claude", "settings.json")))["env"]
    server = json.load(open(os.path.join(HOME, ".claude.json")))["mcpServers"]["sparkpipe"]
    status = mcp(server["url"], server["headers"])
    headers = {"Authorization": "Bearer " + env["ANTHROPIC_AUTH_TOKEN"], "X-Claude-Code-Session-Id": session}
    body = {"model": env["ANTHROPIC_MODEL"], "max_tokens": 64, "messages": [{"role": "user", "content": prompt}], "metadata": {"user_id": json.dumps({"session_id": session})}}
    classes = ["auxiliary", "main"] if "--resume" not in args else ["main"]
    for kind in classes:
        reply = post(env["ANTHROPIC_BASE_URL"] + "/v1/messages", body, dict(headers, **({} if VERSION.endswith("-nohints") else {"x-claude-code-request-class": kind})))
    text, tools = answer(prompt, session)
    emit({"type": "system", "subtype": "init", "session_id": session, "mcp_servers": [{"name": "sparkpipe", "status": status}]},
         *([{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": {}}]}}] if tools else []),
         {"type": "result", "subtype": "success", "is_error": False, "result": text, "session_id": session, "usage": reply["usage"]})
elif NAME == "codex":
    home = os.environ["CODEX_HOME"]
    config = tomllib.load(open(os.path.join(home, "config.toml"), "rb"))
    profile = tomllib.load(open(os.path.join(home, args[args.index("--profile") + 1] + ".config.toml"), "rb"))
    provider = config["model_providers"][profile["model_provider"]]
    if "resume" in args:
        session, prompt = args[args.index("resume") + 1], args[args.index("resume") + 2]
    else:
        session, prompt = str(uuid.uuid4()), args[-1]
    server = config["mcp_servers"]["sparkpipe"]
    mcp(server["url"], {"Authorization": "Bearer " + os.environ[server["bearer_token_env_var"]]})
    reply = post(provider["base_url"] + "/responses", {"model": profile["model"], "input": prompt, "prompt_cache_key": session}, {"Authorization": "Bearer " + os.environ[provider["env_key"]]})
    text, tools = answer(prompt, session)
    usage = reply["usage"]
    emit({"type": "thread.started", "thread_id": session}, *([{"type": "item.completed", "item": {"type": "command_execution", "command": "cat"}}] if tools else []),
         {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
         {"type": "turn.completed", "usage": {"input_tokens": usage["input_tokens"], "cached_input_tokens": usage["input_tokens_details"]["cached_tokens"], "output_tokens": 5}})
else:
    config = json.load(open(os.path.join(os.environ["XDG_CONFIG_HOME"], "opencode", "opencode.json")))
    provider = config["provider"]["sparkpipe"]["options"]
    key = os.environ[provider["apiKey"][len("{env:"):-1]]
    session = args[args.index("--session") + 1] if "--session" in args else "ses_" + uuid.uuid4().hex
    prompt = args[-1]
    server = config["mcp"]["sparkpipe"]
    mcp(server["url"], {name: value.replace("{env:SPARKPIPE_API_KEY}", os.environ["SPARKPIPE_API_KEY"]) for name, value in server["headers"].items()})
    if VERSION.endswith("-phone"):
        phone_home()
    reply = post(provider["baseURL"] + "/messages", {"model": config["model"].split("/", 1)[1], "max_tokens": 64, "messages": [{"role": "user", "content": prompt}]},
                 {"x-api-key": key, "x-session-affinity": session, "X-Session-Id": session})
    text, tools = answer(prompt, session)
    usage = reply["usage"]
    emit({"type": "step_start", "sessionID": session, "part": {}}, *([{"type": "tool_use", "sessionID": session, "part": {"tool": "read"}}] if tools else []),
         {"type": "text", "sessionID": session, "part": {"text": text}},
         {"type": "step_finish", "sessionID": session, "part": {"tokens": {"input": usage["input_tokens"], "output": 5, "cache": {"read": usage["cache_read_input_tokens"], "write": 0}}}})
'''


class Fleet(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.server.auth.append(self.headers.get("Authorization"))
        found = {"/v1/models": MODELS, "/v1/models/aliases": ALIASES}.get(self.path)
        self.reply(200 if found else 404, found or {"error": "not found"})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.loads(raw or b"{}")
        self.server.auth.append(self.headers.get("Authorization"))
        if self.path == "/mcp":
            return self.reply(*((200, {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-06-18", "capabilities": {}}}) if self.server.mcp else (404, {"error": "not found"})))
        session = self.headers.get("X-Claude-Code-Session-Id") or self.headers.get("x-session-affinity") or body.get("prompt_cache_key") or "none"
        if self.headers.get("x-claude-code-request-class") != "auxiliary":
            self.server.turns[session] = self.server.turns.get(session, 0) + 1
        cached = 100 * (self.server.turns.get(session, 1) - 1)
        if self.path == "/v1/messages":
            return self.reply(200, {"type": "message", "content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 20, "cache_read_input_tokens": cached,
                                                                                                        "cache_creation_input_tokens": 20, "output_tokens": 5}})
        if self.path == "/v1/responses":
            return self.reply(200, {"object": "response", "output": [], "usage": {"input_tokens": 40 + cached, "input_tokens_details": {"cached_tokens": cached}, "output_tokens": 5}})
        self.reply(404, {"error": "not found"})


@contextlib.contextmanager
def fleet(mcp=True):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fleet)
    server.daemon_threads, server.auth, server.turns, server.mcp = True, [], {}, mcp
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@contextlib.contextmanager
def environment(values):
    saved = dict(os.environ)
    for name in ("SPARKPIPE_API_KEY", "SPARKPIPE_API", "SPARKPIPE_CONFIG", "ANTHROPIC_API_KEY", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME", "HTTPS_PROXY", "HTTP_PROXY"):
        os.environ.pop(name, None)
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def fake_npm(folder):
    template, log, npm = folder / "tool.py", folder / "npm.log", folder / "npm"
    template.write_text(TOOL)
    npm.write_text(f"#!{sys.executable}\nLOG = {str(log)!r}\nTEMPLATE = {str(template)!r}\nPYTHON = {sys.executable!r}\n{NPM}")
    npm.chmod(0o755)
    return npm, log


def run(folder, npm, server, tools, key=KEY, extra=()):
    out = folder / "result.json"
    if out.exists():
        out.unlink()
    printed = io.StringIO()
    with environment({"SPARKPIPE_API_KEY": key} if key else {}), contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
        code = check.main(["--endpoint", f"http://127.0.0.1:{server.server_address[1]}", "--out", str(out), "--npm", str(npm), *extra,
                           *[part for tool in tools for part in ("--tool", tool)]])
    return code, json.loads(out.read_text()) if out.exists() else None, printed.getvalue()


def logged(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def expect(failures, condition, text):
    if not condition:
        failures.append(text)


def check_good_releases(failures):
    with tempfile.TemporaryDirectory() as name, fleet() as server:
        folder = Path(name)
        npm, log = fake_npm(folder)
        code, result, printed = run(folder, npm, server, ["claude-code=2.1.308", "codex=0.134.0", "opencode=1.18.21"], extra=("--keep",))
        tools = {item["tool"]: item for item in (result or {}).get("tools", [])}
        expect(failures, code == 0 and result["ok"] and sorted(tools) == ["claude-code", "codex", "opencode"],
               f"three good releases pass: {code} {printed} {json.dumps({name: item.get('checks') for name, item in tools.items()})}")
        for name, item in tools.items():
            expect(failures, all(item["checks"].values()) and item["version"] == item["requested"] and item["requested"] in item["reported_version"] and len(item["turns"]) == 3,
                   f"{name} passes every check at its release: {item['checks']} {item.get('reported_version')} {[step['stderr'][-300:] for step in item['steps']]}")
            expect(failures, [turn["cached_tokens"] > 0 for turn in item["turns"]] == [False, True, True] and [turn["tool_calls"] > 0 for turn in item["turns"]] == [True, False, True],
                   f"{name} turns report tool calls on the first and third turn and cache hits after the first: {item['turns']}")
        claude = tools["claude-code"]
        expect(failures, claude["summary"]["request_classes"] == ["auxiliary", "main"] and claude["summary"]["background_requests"] == 1
               and all(record["metadata_session"] == record["headers"]["X-Claude-Code-Session-Id"] for record in claude["requests"] if record["path"] == "/v1/messages"),
               f"the relay records Claude Code's request classes, its background requests and the metadata.user_id session: {claude['summary']}")
        expect(failures, all(record["prompt_cache_key"] for record in tools["codex"]["requests"] if record["path"] == "/v1/responses"), "the relay records Codex's prompt_cache_key")
        expect(failures, server.auth and set(server.auth) == {"Bearer " + KEY}, f"the fleet gets the staging key on every request, through the relay: {set(server.auth)}")
        calls = logged(log)
        installs = [call for call in calls if call["argv"][0] == "install"]
        prefixes = [call["argv"][call["argv"].index("--prefix") + 1] for call in installs]
        expect(failures, len(installs) == 3 and all("sparkpipe-release-check-" in prefix for prefix in prefixes)
               and not any(flag in call["argv"] for call in calls for flag in ("-g", "--global", "--location=global")),
               f"npm installs each release into a temporary prefix, never globally: {[call['argv'] for call in calls]}")
        homes = [Path(prefix).parent / "home" / "tool.log" for prefix in prefixes]
        seen = [entry for home in homes for entry in logged(home)]
        everything = json.dumps(result) + printed + json.dumps(calls) + json.dumps(seen)
        expect(failures, KEY not in everything and not any(call["key"] for call in calls) and all(entry["key"] != KEY for entry in seen),
               "the staging key reaches npm, a tool, an argument, the output or the result")
        expect(failures, all(entry["key"] and entry["key"].startswith("spl-") for entry in seen if entry["argv"] != ["--version"]),
               "the tools do not get the relay's own key")


def check_bad_releases(failures):
    with tempfile.TemporaryDirectory() as name, fleet(mcp=False) as server:
        folder = Path(name)
        npm, log = fake_npm(folder)
        code, result, printed = run(folder, npm, server, ["claude-code=2.1.309-nohints", "codex=0.0.0", "opencode=1.18.22-phone"])
        tools = {item["tool"]: item for item in (result or {}).get("tools", [])}
        expect(failures, code == 1 and result and not result["ok"], f"a release that breaks a check fails the run: {code} {printed}")
        claude, codex, opencode = tools.get("claude-code", {}), tools.get("codex", {}), tools.get("opencode", {})
        expect(failures, claude.get("checks", {}).get("hint_headers") is False and claude["checks"]["mcp"] is False and claude["checks"]["answers"] is True
               and set(next((line for line in printed.splitlines() if line.startswith("claude-code ")), "").partition("FAILED ")[2].split(", ")) == {"hint_headers", "mcp"},
               f"a Claude Code release that drops the hint headers, against a fleet without MCP, fails just those checks: {claude.get('checks')} {printed}")
        expect(failures, codex.get("checks", {}).get("installed") is False and codex["checks"]["turns"] is False and codex["turns"] == [],
               f"a Codex release that does not install fails without running a session: {codex.get('checks')}")
        expect(failures, opencode.get("checks", {}).get("no_opencode_ai") is False and "opencode.ai" in opencode["egress"] and opencode["checks"]["turns"] is True,
               f"an OpenCode release that calls opencode.ai fails the check, and the host is listed: {opencode.get('checks')} {opencode.get('egress')}")
        leftover = [call["argv"][call["argv"].index("--prefix") + 1] for call in logged(log) if call["argv"][0] == "install"]
        expect(failures, leftover and not any(Path(prefix).exists() for prefix in leftover), f"the temporary prefix outlives the run: {leftover}")


def check_latest_and_refusals(failures):
    with tempfile.TemporaryDirectory() as name, fleet() as server:
        folder = Path(name)
        npm, log = fake_npm(folder)
        code, result, printed = run(folder, npm, server, ["codex=latest"])
        item = (result or {"tools": [{}]})["tools"][0]
        expect(failures, code == 0 and item.get("version") == "3.0.0" and [call["argv"][:2] for call in logged(log)] == [["view", "@openai/codex@latest"], ["install", "--prefix"]],
               f"latest resolves through npm view and installs that release: {code} {item.get('version')} {printed}")
        before = len(logged(log))
        for tools, key, wanted in ((["claude-code=1.0;rm -rf /"], KEY, "TOOL=VERSION"), (["cursor=1.0.0"], KEY, "TOOL=VERSION"), ([], KEY, "at least one tool"),
                                   (["codex=0.134.0"], None, "SPARKPIPE_API_KEY")):
            code, result, printed = run(folder, npm, server, tools, key=key)
            expect(failures, code == 2 and result is None and wanted in printed and KEY not in printed, f"{tools} with key {bool(key)} is refused: {code} {printed}")
        expect(failures, len(logged(log)) == before, "a refused run calls npm")


def main():
    failures = []
    for test in (check_good_releases, check_bad_releases, check_latest_and_refusals):
        try:
            test(failures)
        except Exception as error:
            failures.append(f"{test.__name__} stopped: {type(error).__name__}: {error}")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("client release check: each release of Claude Code, Codex and OpenCode installs into a temporary prefix, never globally, is set up with sparkpipe setup through "
          "a recording relay and runs a three-turn session; the result JSON reports the answers, the session kept, tool calls, cache hits, MCP, the affinity key, "
          "Claude Code's hint headers and OpenCode's calls to opencode.ai; latest resolves through npm view; the staging key reaches only the fleet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
