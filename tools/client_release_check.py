#!/usr/bin/env python3
import argparse
import contextlib
import http.server
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR / "serving"))

import chat_shapes
import sparkpipe_cli as cli

TOOLS = {
    "claude-code": {"package": "@anthropic-ai/claude-code", "bin": "claude", "path": "/v1/messages", "affinity": ("header", "X-Claude-Code-Session-Id")},
    "codex": {"package": "@openai/codex", "bin": "codex", "path": "/v1/responses", "affinity": ("body", "prompt_cache_key")},
    "opencode": {"package": "opencode-ai", "bin": "opencode", "path": "/v1/messages", "affinity": ("header", "x-session-affinity")},
}
RELEASE = re.compile(r"latest|\d{1,6}\.\d{1,6}\.\d{1,6}(?:-[0-9A-Za-z.]{1,64})?")
WATCHED = ("x-claude-code-request-class", "x-claude-code-agent-id", "x-claude-code-parent-agent-id", "X-Claude-Code-Session-Id", "x-session-affinity",
           "x-parent-session-id", "x-openai-subagent", "User-Agent")
BACKGROUND = {"x-claude-code-request-class": ("compaction", "auxiliary"), "x-openai-subagent": ("compact",)}
KEPT_ENV = ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SYSTEMROOT", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS")
INSTALL_SECONDS = 900
TURN_SECONDS = 900
TAIL = 4000
SHUT = ("opencode.ai",)


class Refused(Exception):
    pass


def number(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def releases(given):
    found = {}
    for item in given:
        name, _, version = item.partition("=")
        if name not in TOOLS or not RELEASE.fullmatch(version):
            raise Refused(f"{item!r} is not TOOL=VERSION with TOOL one of {', '.join(TOOLS)} and VERSION latest or a release such as 2.1.308")
        found[name] = version
    if not found:
        raise Refused("name at least one tool: --tool claude-code=2.1.308")
    return found


def tail(text):
    return text[-TAIL:]


def observe(handler):
    record = {"method": handler.command, "path": handler.path.split("?", 1)[0], "status": None,
              "websocket": (handler.headers.get("Upgrade") or "").lower() == "websocket"}
    record["headers"] = {name: handler.headers.get(name) for name in WATCHED if handler.headers.get(name) is not None}
    return record


def body_facts(raw):
    with contextlib.suppress(ValueError, UnicodeDecodeError):
        body = json.loads(raw)
        if isinstance(body, dict):
            return {"model": body.get("model"), "prompt_cache_key": chat_shapes.clean_affinity(body.get("prompt_cache_key")),
                    "metadata_session": chat_shapes.metadata_session(body.get("metadata"))}
    return {}


class RelayHandler(cli.ProxyHandler):
    def forward(self):
        self.record = observe(self)
        self.server.seen.append(self.record)
        cli.ProxyHandler.forward(self)

    def request_body(self):
        raw = cli.ProxyHandler.request_body(self)
        if raw:
            self.record.update(body_facts(raw))
        return raw

    def log_request(self, code="-", size="-"):
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            self.record["status"] = int(code)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = forward


class EgressHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def refuse(self):
        target = urllib.parse.urlsplit(("//" + self.path) if self.command == "CONNECT" else self.path)
        self.server.hosts.append(target.hostname or self.path[:200])
        self.close_connection = True
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_CONNECT = do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = refuse


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def relay(endpoint, key):
    token = "spl-" + secrets.token_urlsafe(24)
    server = cli.proxy_server(endpoint, key, None, 0, 6, 60, token, log=False, handler=RelayHandler)
    server.seen = []
    return serve(server), token


def egress():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), EgressHandler)
    server.daemon_threads = True
    server.hosts = []
    return serve(server)


def environment(home, extra=None):
    env = {name: os.environ[name] for name in KEPT_ENV if name in os.environ}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), XDG_DATA_HOME=str(home / ".local" / "share"), XDG_STATE_HOME=str(home / ".local" / "state"),
               XDG_CACHE_HOME=str(home / ".cache"), CODEX_HOME=str(home / ".codex"), SPARKPIPE_CONFIG=str(home / ".config" / "sparkpipe" / "config.json"),
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    env.update(extra or {})
    return env


def run(argv, env, cwd, seconds):
    started = time.monotonic()
    try:
        done = subprocess.run(argv, env=env, cwd=str(cwd), capture_output=True, text=True, timeout=seconds, stdin=subprocess.DEVNULL)
        code, out, err = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as error:
        code, out, err = None, error.stdout or "", (error.stderr or "") + f"\nstopped after {seconds} s"
    except OSError as error:
        code, out, err = None, "", f"{type(error).__name__}: {error.strerror or error}"
    out, err = (text.decode("utf-8", "replace") if isinstance(text, bytes) else text for text in (out, err))
    return {"argv": [Path(argv[0]).name] + argv[1:], "exit": code, "seconds": round(time.monotonic() - started, 2), "stdout": out, "stderr": err}


def shown(step):
    return dict(step, stdout=tail(step["stdout"]), stderr=tail(step["stderr"]))


def npm_env(root, home):
    return environment(home, {"npm_config_cache": str(root / "npm-cache"), "npm_config_userconfig": str(home / ".npmrc"), "npm_config_update_notifier": "false",
                              "npm_config_fund": "false", "npm_config_audit": "false"})


def resolve(npm, tool, version, root, home):
    if version != "latest":
        return version, None
    step = run([npm, "view", TOOLS[tool]["package"] + "@latest", "version"], npm_env(root, home), root, INSTALL_SECONDS)
    found = step["stdout"].strip().splitlines()[-1:] if step["exit"] == 0 else []
    return (found[0] if found and RELEASE.fullmatch(found[0]) and found[0] != "latest" else None), step


def install(npm, tool, version, root, home):
    prefix = root / "prefix"
    step = run([npm, "install", "--prefix", str(prefix), "--no-save", "--no-package-lock", "--loglevel=error", f"{TOOLS[tool]['package']}@{version}"],
               npm_env(root, home), root, INSTALL_SECONDS)
    binary = prefix / "node_modules" / ".bin" / TOOLS[tool]["bin"]
    return (binary if step["exit"] == 0 and binary.exists() else None), step


def workspace(root, home):
    work = root / "work"
    work.mkdir()
    word = "QUOKKA" + str(secrets.randbelow(900000) + 100000)
    subject = "Release check " + word.lower()
    (work / "NOTES.md").write_text(word + "\nThe SparkPipe release check reads this file.\n", encoding="utf-8")
    env = environment(home, {"GIT_AUTHOR_NAME": "SparkPipe", "GIT_AUTHOR_EMAIL": "release-check@sparkpipe.invalid", "GIT_COMMITTER_NAME": "SparkPipe",
                             "GIT_COMMITTER_EMAIL": "release-check@sparkpipe.invalid"})
    for argv in (["git", "init", "-q"], ["git", "add", "NOTES.md"], ["git", "commit", "-q", "-m", subject]):
        step = run(argv, env, work, 60)
        if step["exit"] != 0:
            raise Refused(f"cannot make the check's repository: {' '.join(argv)}: {tail(step['stderr'])}")
    return work, word, subject


def script(word, subject):
    return [
        ("Read the file NOTES.md in the current directory with your file-reading tool, then reply with only the code word on its first line.", word, True, False),
        ("What was the code word? Reply with only that word.", word, False, True),
        ("Run the shell command `git log -1 --format=%s` and reply with only what it prints.", subject, True, True),
    ]


def command(tool, binary, prompt, session):
    if tool == "claude-code":
        return [str(binary), "-p", prompt, "--output-format", "stream-json", "--verbose", "--allowedTools", "Read,Bash(git log:*)"] + (["--resume", session] if session else [])
    if tool == "codex":
        head = [str(binary), "exec", "--profile", cli.CODEX_PROFILE, "--json", "--skip-git-repo-check", "--sandbox", "read-only"]
        return head + (["resume", session, prompt] if session else [prompt])
    return [str(binary), "run", "--format", "json"] + (["--session", session] if session else []) + [prompt]


def events(text):
    found = []
    for line in text.splitlines():
        with contextlib.suppress(ValueError):
            item = json.loads(line)
            if isinstance(item, dict):
                found.append(item)
    return found


def claude_turn(items, facts):
    for item in items:
        kind = item.get("type")
        if kind == "system" and item.get("subtype") == "init":
            facts["session"] = item.get("session_id") or facts["session"]
            facts["mcp"] = {server.get("name"): server.get("status") for server in item.get("mcp_servers") or [] if isinstance(server, dict)}
        elif kind == "assistant":
            content = (item.get("message") or {}).get("content") or []
            facts["tool_calls"] += sum(isinstance(block, dict) and block.get("type") == "tool_use" for block in content)
        elif kind == "result":
            usage = item.get("usage") or {}
            facts.update(session=item.get("session_id") or facts["session"], text=str(item.get("result") or ""), cached_tokens=number(usage.get("cache_read_input_tokens")),
                         input_tokens=number(usage.get("input_tokens")) + number(usage.get("cache_read_input_tokens")) + number(usage.get("cache_creation_input_tokens")))
            if item.get("is_error"):
                facts["error"] = str(item.get("result") or item.get("subtype"))[:500]


def codex_turn(items, facts):
    for item in items:
        kind, inner = item.get("type"), item.get("item") or {}
        if kind == "thread.started":
            facts["session"] = item.get("thread_id")
        elif kind == "item.completed" and inner.get("type") == "agent_message":
            facts["text"] += str(inner.get("text") or "")
        elif kind == "item.completed" and inner.get("type") in ("command_execution", "mcp_tool_call", "file_change", "web_search"):
            facts["tool_calls"] += 1
        elif kind == "turn.completed":
            usage = item.get("usage") or {}
            facts["cached_tokens"] += number(usage.get("cached_input_tokens"))
            facts["input_tokens"] += number(usage.get("input_tokens"))
        elif kind in ("turn.failed", "error"):
            facts["error"] = json.dumps(item.get("error") or item.get("message"))[:500]


def opencode_turn(items, facts):
    for item in items:
        kind, part = item.get("type"), item.get("part") or {}
        facts["session"] = item.get("sessionID") or facts["session"]
        if kind == "text":
            facts["text"] += str(part.get("text") or "")
        elif kind == "tool_use":
            facts["tool_calls"] += 1
        elif kind == "step_finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            facts["cached_tokens"] += number(cache.get("read"))
            facts["input_tokens"] += number(tokens.get("input")) + number(cache.get("read")) + number(cache.get("write"))
        elif kind == "error":
            facts["error"] = json.dumps(item.get("error"))[:500]


PARSERS = {"claude-code": claude_turn, "codex": codex_turn, "opencode": opencode_turn}


def turn(tool, step):
    facts = {"exit": step["exit"], "seconds": step["seconds"], "session": None, "text": "", "tool_calls": 0, "input_tokens": 0, "cached_tokens": 0, "error": None, "mcp": None}
    PARSERS[tool](events(step["stdout"]), facts)
    facts["text"] = tail(facts["text"])
    return facts


def affinity(tool, record):
    where, name = TOOLS[tool]["affinity"]
    if where == "header":
        return record["headers"].get(name)
    return record.get(name)


def checks(tool, installed, setup, turns, plan, seen, hosts):
    inference = [record for record in seen if record["path"] == TOOLS[tool]["path"]]
    sessions = [facts["session"] for facts in turns]
    complete = bool(plan) and len(turns) == len(plan)
    found = {
        "installed": installed,
        "setup": setup is not None and setup["exit"] == 0,
        "turns": complete and all(facts["exit"] == 0 and facts["error"] is None for facts in turns),
        "session_kept": complete and sessions[0] is not None and len(set(sessions)) == 1,
        "answers": complete and all(wanted.lower() in facts["text"].lower() for facts, (_, wanted, _, _) in zip(turns, plan)),
        "tool_calls": complete and all(facts["tool_calls"] > 0 for facts, (_, _, tools, _) in zip(turns, plan) if tools),
        "cache_hits": complete and all(facts["cached_tokens"] > 0 for facts, (_, _, _, cached) in zip(turns, plan) if cached),
        "mcp": any(record["path"] == "/mcp" and record["status"] is not None and 200 <= record["status"] < 300 for record in seen),
        "affinity": bool(inference) and all(affinity(tool, record) for record in inference),
    }
    if tool == "claude-code":
        found["hint_headers"] = bool(inference) and all(record["headers"].get("x-claude-code-request-class") for record in inference)
    if tool == "opencode":
        found["no_opencode_ai"] = not any(host == shut or host.endswith("." + shut) for host in hosts for shut in SHUT)
    return found


def summary(seen):
    classes = sorted({record["headers"][name] for record in seen for name in BACKGROUND if record["headers"].get(name)})
    background = sum(record["headers"].get(name, "").lower() in values for record in seen for name, values in BACKGROUND.items())
    return {"request_classes": classes, "background_requests": background, "requests": len(seen)}


def check_tool(tool, version, endpoint, key, npm, base):
    root = base / tool
    home = root / "home"
    home.mkdir(parents=True)
    result = {"tool": tool, "package": TOOLS[tool]["package"], "requested": version, "version": None, "steps": [], "turns": [], "requests": [], "egress": []}
    resolved, step = resolve(npm, tool, version, root, home)
    result["steps"] += [shown(step)] if step else []
    binary = None
    if resolved:
        result["version"] = resolved
        binary, step = install(npm, tool, resolved, root, home)
        result["steps"].append(shown(step))
    plan, setup, turns, seen, hosts = [], None, [], [], []
    if binary:
        result["reported_version"] = run([str(binary), "--version"], environment(home), root, 120)["stdout"].strip()[:200]
        work, word, subject = workspace(root, home)
        plan = script(word, subject)
        server, token = relay(endpoint, key)
        outside = egress()
        try:
            proxy = f"http://127.0.0.1:{outside.server_address[1]}"
            env = environment(home, {"SPARKPIPE_API_KEY": token, "HTTPS_PROXY": proxy, "HTTP_PROXY": proxy, "https_proxy": proxy, "http_proxy": proxy,
                                     "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
            relay_url = f"http://127.0.0.1:{server.server_address[1]}"
            setup = run([sys.executable, str(TOOLS_DIR / "sparkpipe_cli.py"), "setup", tool, "--api", relay_url], env, work, 120)
            result["steps"].append(shown(setup))
            session = None
            for prompt, _, _, _ in (plan if setup["exit"] == 0 else []):
                step = run(command(tool, binary, prompt, session), env, work, TURN_SECONDS)
                result["steps"].append(shown(step))
                facts = turn(tool, step)
                turns.append(facts)
                session = session or facts["session"]
                if facts["exit"] != 0 or not session:
                    break
        finally:
            server.shutdown()
            server.server_close()
            outside.shutdown()
            outside.server_close()
        seen, hosts = list(server.seen), sorted(set(outside.hosts))
    result.update(turns=turns, requests=seen, egress=hosts, summary=summary(seen))
    result["checks"] = checks(tool, binary is not None, setup, turns, plan, seen, hosts)
    result["ok"] = all(result["checks"].values())
    return result


def write_result(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Install releases of Claude Code, Codex and OpenCode in a temporary prefix, set each up with sparkpipe setup against a "
                                                 "staging endpoint, run a scripted three-turn session and write the result as JSON. The staging key is read from "
                                                 "SPARKPIPE_API_KEY and never passed to a tool or written down.")
    parser.add_argument("--endpoint", required=True, help="the staging API address, such as https://staging.example/pay/api")
    parser.add_argument("--tool", action="append", default=[], metavar="TOOL=VERSION", help="claude-code, codex or opencode, at a release or latest; repeat for each tool")
    parser.add_argument("--out", required=True, help="the JSON result file")
    parser.add_argument("--npm", default="npm", help="the npm command that installs each release into the temporary prefix")
    parser.add_argument("--keep", action="store_true", help="keep the temporary prefix, homes and work trees")
    args = parser.parse_args(argv)
    try:
        chosen = releases(args.tool)
        endpoint = cli.api_base(args.endpoint)
        if not os.environ.get("SPARKPIPE_API_KEY"):
            raise Refused("set SPARKPIPE_API_KEY to the staging key; it is never taken as an argument")
        key = cli.credential(os.environ["SPARKPIPE_API_KEY"], "SPARKPIPE_API_KEY")
    except (Refused, cli.Failure) as error:
        print(f"client_release_check: {error}", file=sys.stderr)
        return 2
    base = Path(tempfile.mkdtemp(prefix="sparkpipe-release-check-"))
    try:
        results = [check_tool(tool, version, endpoint, key, args.npm, base) for tool, version in chosen.items()]
    except Refused as error:
        print(f"client_release_check: {error}", file=sys.stderr)
        return 2
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
    data = {"checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "endpoint": endpoint, "ok": all(item["ok"] for item in results), "tools": results}
    write_result(args.out, data)
    for item in results:
        failed = [name for name, passed in item["checks"].items() if not passed]
        print(f"{item['tool']} {item['version'] or item['requested']}: {'ok' if item['ok'] else 'FAILED ' + ', '.join(failed)}")
    return 0 if data["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
