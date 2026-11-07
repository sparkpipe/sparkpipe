#!/usr/bin/env python3
import base64
import contextlib
import hashlib
import http.client
import http.server
import io
import json
import os
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
import urllib.parse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import sparkpipe_cli as cli
import sparkpipe_cli_build as build

KEY = "sk-hc-machine"
SESSION = "sps-SESSIONTOKEN"
TOKEN = "spl-localproxytoken"
SECRET = "hf_SECRETtoken0123456789"
CLEARED = ("SPARKPIPE_API_KEY", "SPARKPIPE_API", "SPARKPIPE_FLEET", "SPARKPIPE_CONFIG", "SPARKPIPE_DEBUG", "ANTHROPIC_API_KEY", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME")
MODELS = {"data": [
    {"id": "kimi-k3+core", "context_window": 1000000, "max_output_tokens": 32768, "sequences": 8},
    {"id": "glm-5.3-flash", "context_window": 200000, "max_output_tokens": 16384, "sequences": 16},
]}
ALIASES = {"aliases": {"claude-haiku-*": "glm-5.3-flash", "claude-*": "kimi-k3+core", "gpt-*": None}, "first_resident": "kimi-k3+core"}
DEVICE = ("sparkpipe CLI on " + socket.gethostname())[:100]

SECURITY = r'''
import json, os, sys
state = os.environ["FAKE_STATE"]
with open(os.path.join(state, "argv.log"), "a") as log:
    log.write(json.dumps(sys.argv) + "\n")
path = os.path.join(state, "keychain.json")
store = json.load(open(path)) if os.path.exists(path) else {}
args = sys.argv[1:]
def save():
    json.dump(store, open(path, "w"))
if args == ["-i"]:
    for line in sys.stdin:
        words = line.split()
        if words and words[0] == "add-generic-password" and "-U" in words:
            store[words[words.index("-s") + 1] + "/" + words[words.index("-a") + 1]] = bytes.fromhex(words[words.index("-X") + 1]).decode()
    save()
    sys.exit(0)
name = args[args.index("-s") + 1] + "/" + args[args.index("-a") + 1]
if args[0] == "add-generic-password":
    store[name] = args[args.index("-w") + 1]
    save()
elif args[0] == "find-generic-password":
    if name not in store:
        sys.exit(44)
    print(store[name])
elif args[0] == "delete-generic-password":
    store.pop(name, None)
    save()
else:
    sys.exit(2)
'''

SECRET_TOOL = r'''
import json, os, sys
state = os.environ["FAKE_STATE"]
with open(os.path.join(state, "argv.log"), "a") as log:
    log.write(json.dumps(sys.argv) + "\n")
path = os.path.join(state, "keychain.json")
store = json.load(open(path)) if os.path.exists(path) else {}
args = sys.argv[1:]
name = args[args.index("service") + 1] + "/" + args[args.index("account") + 1]
if args[0] == "store":
    store[name] = sys.stdin.read()
elif args[0] == "lookup":
    if name not in store:
        sys.exit(1)
    sys.stdout.write(store[name])
elif args[0] == "clear":
    store.pop(name, None)
json.dump(store, open(path, "w"))
'''

AWS = r'''
import json, os, shutil, sys
state = os.environ["FAKE_STATE"]
args = sys.argv[1:]
with open(os.path.join(state, "aws.log"), "a") as log:
    log.write(json.dumps(args) + "\n")
plain, skip = [], False
for arg in args:
    if skip:
        skip = False
    elif arg in ("--metadata", "--bucket", "--key", "--output"):
        skip = True
    elif not arg.startswith("--"):
        plain.append(arg)
def option(name):
    return args[args.index(name) + 1] if name in args else None
def where(url):
    name = url[len("s3://"):]
    return os.path.join(state, "bucket", name), os.path.join(state, "meta", name + ".json")
if plain[:2] == ["s3", "cp"]:
    target, meta = where(plain[3])
    for path in (target, meta):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copyfile(plain[2], target)
    found = option("--metadata")
    json.dump(dict([found.split("=", 1)]) if found else {}, open(meta, "w"))
elif plain[:2] == ["s3", "rm"]:
    for path in where(plain[2]):
        os.unlink(path)
elif plain[:2] == ["s3api", "head-object"]:
    target, meta = where("s3://" + option("--bucket") + "/" + option("--key"))
    if not os.path.exists(target):
        sys.stderr.write("An error occurred (404) when calling the HeadObject operation: Not Found\n")
        sys.exit(254)
    print(json.dumps({"ContentLength": os.path.getsize(target), "Metadata": json.load(open(meta))}))
'''


class FakeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        parts = urllib.parse.urlsplit(self.path)
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        request = {"method": self.command, "path": urllib.parse.unquote(parts.path), "raw_path": parts.path, "query": urllib.parse.parse_qs(parts.query, keep_blank_values=True),
                   "headers": {k.lower(): v for k, v in self.headers.items()}, "raw": raw, "body": body}
        self.fake.requests.append(request)
        route = self.fake.routes.get((self.command, request["path"]))
        status, headers, payload = (route(request) if callable(route) else route) if route else (404, {}, {"error": "Not found."})
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        if status == 101:
            self.end_headers()
            self.close_connection = True
            payload(self)
            return
        if isinstance(payload, list):
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            with contextlib.suppress(OSError):
                for piece in payload:
                    data = piece() if callable(piece) else piece
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            return
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        if "Content-Length" in headers:
            self.close_connection = True
        else:
            self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = answer


class Fake:
    def __init__(self):
        self.routes, self.requests = {}, []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), type("Handler", (FakeHandler,), {"fake": self}))
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def route(self, method, path, answer):
        self.routes[(method, path)] = answer

    def find(self, method, path):
        return [r for r in self.requests if r["method"] == method and r["path"] == path]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@contextlib.contextmanager
def environment(values):
    saved = dict(os.environ)
    for name in CLEARED:
        os.environ.pop(name, None)
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@contextlib.contextmanager
def patched(owner, name, value):
    saved = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, saved)


class Home:
    def __init__(self, root, platform="darwin"):
        self.root = Path(root)
        self.home, self.bin, self.state = self.root / "home", self.root / "bin", self.root / "state"
        for path in (self.home, self.bin, self.state):
            path.mkdir(parents=True, exist_ok=True)
        self.config = self.home / ".config" / "sparkpipe" / "config.json"
        self.platform = platform
        self.env = {"HOME": str(self.home), "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}", "FAKE_STATE": str(self.state)}
        self.prompts = []
        for name, script in (("security", SECURITY), ("secret-tool", SECRET_TOOL), ("aws", AWS)):
            path = self.bin / name
            path.write_text(f"#!{sys.executable}\n{script}")
            path.chmod(0o755)

    def run(self, *argv, answers=(), secrets=(), env=None):
        answers, secrets = list(answers), list(secrets)
        out, err = io.StringIO(), io.StringIO()

        def answer(text):
            self.prompts.append(text)
            found = answers.pop(0)
            if isinstance(found, BaseException):
                raise found
            return found

        cli.prompt, cli.prompt_secret, cli.PLATFORM = answer, lambda text: secrets.pop(0), self.platform
        with environment(dict(self.env, **(env or {}))), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main([str(a) for a in argv])
            except SystemExit as error:
                code = error.code
        return code, out.getvalue(), err.getvalue()

    def signed_in(self, api, **extra):
        self.config.parent.mkdir(parents=True, exist_ok=True)
        self.config.write_text(json.dumps(dict({"api": api, "email": "dev@acme.example", "session": SESSION, "key": KEY, "key_id": "KY1"}, **extra)))

    def cfg(self):
        return json.loads(self.config.read_text()) if self.config.exists() else {}

    def keychain(self):
        path = self.state / "keychain.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def argv_log(self):
        path = self.state / "argv.log"
        return path.read_text() if path.exists() else ""

    def aws_log(self):
        path = self.state / "aws.log"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def scenario(check):
    def run(failures):
        with tempfile.TemporaryDirectory() as folder:
            fake = Fake()
            try:
                check(failures, Home(folder), fake)
            finally:
                fake.close()
    run.__name__ = check.__name__
    return run


def expect(failures, condition, text):
    if not condition:
        failures.append(text)


def session_auth(request):
    return request["headers"].get("authorization") == "Bearer " + SESSION


def key_auth(request):
    return request["headers"].get("authorization") == "Bearer " + KEY


def backups(path):
    return sorted(Path(path).parent.glob(Path(path).name + ".*.bak"))


def refused(failures, call, text):
    try:
        call()
        failures.append(text)
    except cli.Failure as error:
        return str(error)
    return ""


def check_units(failures):
    expect(failures, cli.api_base("https://sparkpipe.ai/pay/api/") == cli.API, "the API address keeps a trailing slash")
    for bad in ("http://sparkpipe.ai/pay/api", "ftp://x", "sparkpipe.ai"):
        refused(failures, lambda: cli.api_base(bad), f"the API address {bad} is accepted")
    expect(failures, cli.api_base("http://127.0.0.1:9") == "http://127.0.0.1:9", "a loopback http API address is refused")
    for text, slot in (("2027-01-05T10", "2027-01-05T10:00Z"), ("2027-01-05T10:00", "2027-01-05T10:00Z"), ("2027-01-05T10:00Z", "2027-01-05T10:00Z")):
        expect(failures, cli.slot(text) == slot, f"the hour {text} is not {slot}")
    for bad in ("2027-13-05T10", "2027-01-05T10:30", "2027-01-05", "tomorrow", "2027-01-05T10\n"):
        refused(failures, lambda: cli.slot(bad), f"the hour {bad!r} is accepted")
    hosts = {"hf://org/model@main": "huggingface.co", "s3://bucket/key": "s3.amazonaws.com", "https://Data.Example.com/x": "data.example.com", "git+https://git.example.com/r.git": "git.example.com"}
    for source, host in hosts.items():
        expect(failures, cli.source_host(source) == host, f"the host of {source} is not {host}")
    expect(failures, cli.host_matches("cdn-lfs.huggingface.co", "huggingface.co") and not cli.host_matches("huggingface.co.evil.example", "huggingface.co")
           and not cli.host_matches("evilhuggingface.co", "huggingface.co"), "a token's host is matched by suffix instead of by domain")
    expect(failures, cli.remote_prefix("/models/") == "models/" and cli.remote_prefix("") == "", "storage prefixes are not normalised")
    for path in ("models/../x", "models/./x", "models//x", "models/sub\\..\\x", "models/"):
        refused(failures, lambda: cli.sync_parts("models/", path), f"the unsafe storage path {path!r} is accepted")
    for bad in (KEY + "\r", KEY + "\n", "sk-hc machine", "sk-hc-\x00", "", None, 7):
        message = refused(failures, lambda: cli.credential(bad, "The key"), f"the credential {bad!r} is accepted")
        expect(failures, "machine" not in message, f"a refused credential is printed: {message}")
    expect(failures, cli.credential(KEY, "The key") == KEY, "a valid key is refused")
    for name in ("hf\n", "a b", ""):
        refused(failures, lambda: cli.token_name(name), f"the token name {name!r} is accepted")
    expect(failures, (cli.percent("0.1"), cli.percent(0.05), cli.percent("0.125"), cli.percent("0"), cli.percent(None)) == ("10%", "5%", "12.5%", None, None),
           "a discount is not shown as a percentage")
    env = cli.claude_env(cli.API, KEY, "kimi-k3+core", "kimi-k3+core", 1000000)
    spec = {"ANTHROPIC_BASE_URL": "https://sparkpipe.ai/pay/api", "ANTHROPIC_AUTH_TOKEN": KEY, "ANTHROPIC_MODEL": "kimi-k3+core", "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-k3+core",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1000000", "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_SUBAGENT_MODEL": "kimi-k3+core", "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1", "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "1800000"}
    expect(failures, env == spec, f"the Claude Code environment differs from FLEET_API.md 3.3 and 3.8: {env}")
    provider = dict(cli.codex_tables(cli.API)[0][1])
    expect(failures, provider == {"name": "SparkPipe", "base_url": "https://sparkpipe.ai/pay/api/v1", "env_key": "SPARKPIPE_API_KEY", "wire_api": "responses", "stream_idle_timeout_ms": 600000,
                                  "model_catalog_url": "https://sparkpipe.ai/pay/api/v1/codex/models"}, f"the Codex provider differs from FLEET_API.md 3.4: {provider}")
    profile = tomllib.loads(cli.codex_profile("kimi-k3+core", {"context_window": 1000000, "max_output_tokens": 32768, "sequences": 8}))
    expect(failures, profile == {"model": "kimi-k3+core", "model_provider": "sparkpipe", "model_context_window": 1000000, "model_auto_compact_token_limit": 967232,
                                 "agents": {"default_subagent_model": "kimi-k3+core", "max_concurrent_threads_per_session": 7}}, f"the Codex profile differs from FLEET_API.md 3.8: {profile}")
    single = tomllib.loads(cli.codex_profile("m", {"context_window": 100, "max_output_tokens": 10, "sequences": 1}))
    expect(failures, single["agents"]["max_concurrent_threads_per_session"] == 1, "a fleet with one sequence gets no subagent thread")


@scenario
def check_login(failures, home, fake):
    keys, sessions = [], {"issued": 0, "live": set()}

    def verify(request):
        body = request["body"]
        if body.get("code") != "123456":
            return 403, {}, {"error": "That code is wrong or has expired. Ask for a new one."}
        if body["email"] == "new@acme.example" and not (body.get("company") and body.get("accept") is True):
            return 404, {}, {"error": "No account uses this email yet.", "reason": "new"}
        summary = {"email": body["email"], "company": "Acme"}
        if body.get("bearer") is not True:
            return 200, {"Set-Cookie": "sp_session=cookie.token; Max-Age=2592000; Path=/pay/api; HttpOnly"}, summary
        sessions["issued"] += 1
        token = f"sps-issued{sessions['issued']}"
        sessions["live"].add(token)
        return 200, {}, dict(summary, session=token, sessionExpires="2027-02-02T00:00:00Z")

    def bearer(request):
        given = request["headers"].get("authorization", "")
        return given[7:] if given.startswith("Bearer ") and given[7:] in sessions["live"] else None

    def state(request):
        return 200, {}, {"account": {"email": "dev@acme.example", "keys": [dict(k) for k in keys]} if bearer(request) else None}

    def create(request):
        if not bearer(request):
            return 401, {}, {"error": "Sign in first."}
        ident = f"KY{len(keys) + 1}"
        keys.append({"id": ident, "key": ident, "note": request["body"]["note"], "hint": "abcd", "created": "2026-11-04T00:00:00Z", "revoked": False, "retires": None})
        return 200, {}, dict(keys[-1], key=f"sk-hc-{ident}")

    def signout(request):
        sessions["live"].discard(bearer(request))
        return 200, {}, {"ok": True}

    fake.route("POST", "/code", (200, {}, {"sent": True}))
    fake.route("POST", "/verify", verify)
    fake.route("POST", "/signout", signout)
    fake.route("GET", "/market/state", state)
    fake.route("POST", "/market/keys", create)
    code, out, err = home.run("login", "--api", fake.url, "--email", "dev@acme.example", answers=["123456"])
    expect(failures, code == 0 and "Created key KY1" in out and "until 2027-02-02T00:00:00Z" in out, f"login does not create a key: {code} {out} {err}")
    sent = fake.find("POST", "/code") + fake.find("POST", "/verify")
    expect(failures, [r["body"] for r in sent] == [{"email": "dev@acme.example"}, {"email": "dev@acme.example", "code": "123456", "bearer": True, "device": DEVICE}]
           and not any("authorization" in r["headers"] for r in sent), f"login does not ask /verify for a bearer session: {[r['body'] for r in sent]}")
    made = fake.find("POST", "/market/keys")
    expect(failures, len(made) == 1 and made[0]["headers"].get("authorization") == "Bearer sps-issued1" and made[0]["body"]["note"] == DEVICE,
           "the key is not created with the bearer session and the device name")
    expect(failures, home.cfg() == {"api": fake.url, "email": "dev@acme.example", "session": "sps-issued1", "session_expires": "2027-02-02T00:00:00Z", "key": "sk-hc-KY1", "key_id": "KY1",
                                    "tokens": {}, "sync": {}}, f"the saved config: {home.cfg()}")
    expect(failures, "cookie.token" not in home.config.read_text(), "the cookie session is saved")
    expect(failures, stat.S_IMODE(home.config.stat().st_mode) == 0o600, "the config holding the key and session is readable by others")
    code, out, _ = home.run("login", "--api", fake.url, "--email", "dev@acme.example", answers=["123456"])
    expect(failures, code == 0 and "Kept key KY1" in out and len(fake.find("POST", "/market/keys")) == 1, f"a second login creates another key: {out}")
    expect(failures, home.cfg()["session"] == "sps-issued2" and sessions["live"] == {"sps-issued2"} and fake.find("POST", "/signout")[-1]["headers"].get("authorization") == "Bearer sps-issued1",
           f"a second login leaves the first session live: {sessions['live']}")
    keys[0]["revoked"] = True
    code, out, _ = home.run("login", "--email", "dev@acme.example", answers=["123456"])
    expect(failures, code == 0 and "Created key KY2" in out and home.cfg()["key"] == "sk-hc-KY2", f"a login after the key was revoked keeps the revoked key: {out}")
    keys[1]["retires"] = "2026-11-04T02:00:00Z"
    code, out, _ = home.run("login", "--email", "dev@acme.example", answers=["123456"])
    expect(failures, code == 0 and "Created key KY3" in out, f"a login after the key was rotated keeps the retiring key: {out}")
    code, _, err = home.run("login", "--email", "dev@acme.example", answers=["654321"])
    expect(failures, code == 1 and "That code is wrong" in err and home.cfg()["key"] == "sk-hc-KY3", f"a wrong code: {code} {err}")
    code, _, err = home.run("login", "--email", "new@acme.example", answers=["123456"])
    expect(failures, code == 1 and "--company" in err and "--accept" in err, f"a new email is not told how to open an account: {err}")
    code, out, _ = home.run("login", "--email", "new@acme.example", "--company", "New Co", "--accept", answers=["123456"])
    opened = fake.find("POST", "/verify")[-1]["body"]
    expect(failures, code == 0 and opened == {"email": "new@acme.example", "code": "123456", "bearer": True, "device": DEVICE, "company": "New Co", "accept": True}, f"opening an account: {opened}")
    fake.route("POST", "/code", (200, {}, {"sent": False, "code": "123456"}))
    code, out, err = home.run("login", "--email", "dev@acme.example")
    expect(failures, code == 0 and "sandbox" in out, f"the sandbox's code is not used: {code} {out} {err}")
    fake.route("POST", "/verify", (200, {"Set-Cookie": "sp_session=cookie.token; Path=/pay/api; HttpOnly"}, {"email": "dev@acme.example"}))
    code, _, err = home.run("login", "--email", "dev@acme.example")
    expect(failures, code == 1 and "without a session" in err and "cookie.token" not in home.config.read_text(), "a sign-in that sets only a cookie is taken as a session")
    fake.route("POST", "/verify", (200, {}, {"email": "dev@acme.example", "session": "sps-broken\r\n"}))
    code, _, err = home.run("login", "--email", "dev@acme.example")
    expect(failures, code == 1 and "sps-broken" not in err + home.config.read_text(), f"a session with a line break is taken or printed: {err}")


@scenario
def check_keys(failures, home, fake):
    home.signed_in(fake.url)
    listed = [{"id": "KY0", "key": "KY0", "note": "laptop", "hint": "zzzz", "created": "2026-10-01T00:00:00Z", "revoked": True},
              {"id": "KY1", "key": "KY1", "note": "sparkpipe CLI on box", "hint": "abcd", "created": "2026-11-01T00:00:00Z", "revoked": False}]
    fake.route("GET", "/market/state", (200, {}, {"account": {"keys": listed}}))
    fake.route("POST", "/market/keys", (200, {}, {"id": "KY2", "key": "sk-hc-new"}))
    fake.route("POST", "/market/keys/revoke", lambda r: (200, {}, {"id": r["body"]["id"], "revoked": True}))
    code, out, _ = home.run("keys", "ls")
    lines = out.splitlines()
    expect(failures, code == 0 and len(lines) == 2 and lines[0].startswith("KY0") and "revoked" in lines[0] and "active" in lines[1] and "this machine" in lines[1], f"keys ls: {out}")
    code, out, _ = home.run("keys", "ls", "--json")
    expect(failures, code == 0 and json.loads(out) == listed, "keys ls --json does not print the keys")
    code, out, _ = home.run("keys", "create", "--note", "ci", "--fleet", "F3", "--allow-swap")
    made = fake.find("POST", "/market/keys")[-1]
    expect(failures, code == 0 and made["body"] == {"note": "ci", "fleet": "F3", "allow_swap": True} and session_auth(made) and "sk-hc-new" in out and home.cfg()["key"] == KEY,
           f"keys create: {made['body']} {out}")
    code, out, _ = home.run("keys", "create")
    expect(failures, code == 0 and fake.find("POST", "/market/keys")[-1]["body"] == {"note": DEVICE}, "keys create without a note does not name the device")
    code, out, _ = home.run("keys", "revoke", "KY0")
    expect(failures, code == 0 and fake.find("POST", "/market/keys/revoke")[-1]["body"] == {"id": "KY0"} and home.cfg()["key"] == KEY, "revoking another key changes this machine's")
    code, out, _ = home.run("keys", "revoke", "KY1")
    expect(failures, code == 0 and "key" not in home.cfg() and "sparkpipe login" in out, f"revoking this machine's key leaves it configured: {home.cfg()}")
    fake.route("GET", "/market/state", (200, {}, {"account": None}))
    code, _, err = home.run("keys", "ls")
    expect(failures, code == 1 and "expired" in err, "an expired session is not reported")
    home.config.write_text(json.dumps({"api": fake.url}))
    code, _, err = home.run("keys", "ls")
    expect(failures, code == 1 and "sparkpipe login" in err, "keys without a session do not ask for a login")


@scenario
def check_status(failures, home, fake):
    home.signed_in(fake.url)
    account = {"email": "dev@acme.example", "company": "Acme", "balance": "300.000000", "locked": "144.000000", "position": 2, "termsCurrent": False,
               "bookings": [{"booking": "BK1", "fleet": "F3", "kind": "week", "start": "2027-01-04T00:00Z", "end": "2027-01-11T00:00Z", "state": "live"},
                            {"booking": "BK0", "fleet": "F1", "kind": "hour", "start": "2026-12-01T00:00Z", "end": "2026-12-01T02:00Z", "state": "ended"}],
               "pinned": [{"fleet": "F3", "weeks": 1, "through": "2027-01-11T00:00Z", "deadline": "2027-01-10T00:00Z", "canExtend": True}], "keys": []}
    state = {"fleet": "F3", "models": [{"id": "kimi-k3"}], "cache": {"used_bytes": 500000000000, "capacity_bytes": 4000000000000}, "pins": [{"pack": "core"}],
             "packs": [{"name": "core", "version": "a41f09", "warm": True}], "runs": [{"id": "run_1", "function": "review", "state": "running"}], "deletes_at": "2027-01-12T00:00Z"}
    fake.route("GET", "/market/state", lambda r: (200, {}, {"account": account if session_auth(r) else None}))
    fake.route("GET", "/v1/state", lambda r: (200, {}, state) if key_auth(r) else (401, {}, {"error": {"message": "bad key", "code": "invalid_key"}}))
    code, out, err = home.run("status", "--fleet", "F3")
    for text in ("300.000000 HC", "Holding now: F3", "BK1", "book the next week by 2027-01-10T00:00Z", "position 2", "Customer Agreement has changed",
                 "Fleet F3", "kimi-k3", "500.0 GB of 4.0 TB", "pins: 1", "core@a41f09 (warm)", "run_1 review running", "2027-01-12T00:00Z"):
        expect(failures, text in out, f"status does not show {text!r}: {out} {err}")
    expect(failures, code == 0 and "BK0" not in out, "status lists ended bookings")
    expect(failures, fake.find("GET", "/v1/state")[-1]["headers"].get("x-sparkpipe-fleet") == "F3", "status --fleet does not send X-SparkPipe-Fleet")
    code, out, _ = home.run("status", "--json")
    expect(failures, code == 0 and json.loads(out) == {"account": account, "fleet": state}, "status --json")
    fake.route("GET", "/v1/state", (403, {}, {"error": {"message": "This account holds no Fleet now.", "code": "no_fleet"}}))
    code, out, _ = home.run("status")
    expect(failures, code == 0 and "Fleet: This account holds no Fleet now." in out and "Account dev@acme.example" in out, f"status without a held fleet: {out}")
    fake.route("GET", "/market/state", (200, {}, {"account": None}))
    code, _, err = home.run("status")
    expect(failures, code == 1 and "expired" in err, "status with an expired session")


@scenario
def check_book(failures, home, fake):
    home.signed_in(fake.url)
    plan = {"fleet": "F3", "weeks": 4, "first": "2027-01-04T00:00Z", "end": "2027-02-01T00:00Z", "price": "540.000000", "discount": "0.1"}

    def booked(request):
        if request["body"].get("accept") is not True and fake.terms:
            return 409, {}, {"error": "The Customer Agreement has changed since you accepted it.", "reason": "terms"}
        return 200, {}, dict(plan, booking="BK9")

    fake.terms = False
    fake.route("GET", "/market/state", (200, {}, {"account": {"email": "dev@acme.example"}, "bookings": {"hourHC": 1, "weekHC": 144}, "fleets": [{"id": "F1", "rate": 1.5}, {"id": "F3", "rate": 1}]}))
    fake.route("POST", "/market/quote-weeks", (200, {}, plan))
    hour_plan = {"fleet": "F1", "start": "2027-01-05T10:00Z", "end": "2027-01-05T12:00Z", "hours": 2, "price": "2.400000", "discount": "0.2", "weeks": 52, "basis": "held"}
    fake.route("POST", "/market/quote-hours", lambda r: (200, {}, hour_plan) if r["body"]["fleet"] in ("F1", "F3") else (400, {}, {"error": "Choose a Fleet on the Fleet List: F1, F3."}))
    fake.route("POST", "/market/book-weeks", booked)
    fake.route("POST", "/market/book-hours", lambda r: (200, {}, dict(r["body"], booking="BK8", end="2027-01-05T12:00Z", hc="3.000000")))
    fake.route("POST", "/market/waitlist", (200, {}, {"locked": "144.000000", "position": 3}))
    fake.route("POST", "/market/waitlist/leave", (200, {}, {"locked": "0.000000", "position": None}))
    code, out, _ = home.run("book", "week", "F3", "4", answers=["y"])
    books = fake.find("POST", "/market/book-weeks")
    question = "".join(home.prompts[-1:])
    expect(failures, code == 0 and "BK9" in out and "for 540.000000 HC (10% off)?" in question and [r["body"] for r in books] == [{"fleet": "F3", "weeks": 4}] and session_auth(books[0])
           and fake.find("POST", "/market/quote-weeks")[0]["body"] == {"fleet": "F3", "weeks": 4}, f"book week: {code} {out} {home.prompts}")
    fake.route("POST", "/market/quote-weeks", (200, {}, dict(plan, weeks=1, discount="0")))
    home.run("book", "week", "F3", "1", answers=["n"])
    expect(failures, "off" not in home.prompts[-1], f"a booking without a discount shows one: {home.prompts[-1]}")
    fake.route("POST", "/market/quote-weeks", (200, {}, plan))
    code, _, err = home.run("book", "week", "F3", "4", answers=["n"])
    expect(failures, code == 1 and len(fake.find("POST", "/market/book-weeks")) == 1, "a declined booking is booked")
    code, _, err = home.run("book", "week", "F3", "4", answers=[EOFError()])
    expect(failures, code == 1 and "--yes" in err and len(fake.find("POST", "/market/book-weeks")) == 1, "a booking without a terminal does not ask for --yes")
    fake.terms = True
    code, _, err = home.run("book", "week", "F3", "4", "--yes")
    expect(failures, code == 1 and "--accept" in err, f"a booking under changed terms: {err}")
    code, out, _ = home.run("book", "week", "F3", "4", "--yes", "--accept")
    expect(failures, code == 0 and fake.find("POST", "/market/book-weeks")[-1]["body"] == {"fleet": "F3", "weeks": 4, "accept": True}, "--accept is not sent with the booking")
    code, out, _ = home.run("book", "hours", "F1", "2027-01-05T10", "2", answers=["y"])
    expect(failures, code == 0 and "F1 for 2 hour(s) from 2027-01-05T10:00Z, for 2.400000 HC (20% off)?" in home.prompts[-1] and "BK8" in out
           and fake.find("POST", "/market/quote-hours")[0]["body"] == {"fleet": "F1", "start": "2027-01-05T10:00Z", "hours": 2}, f"book hours does not show the market's price and discount: {home.prompts[-1]}")
    for start in ("2027-01-05T10", "2027-01-05T10:00Z"):
        code, out, _ = home.run("book", "hours", "F1", start, "2", "--yes")
        expect(failures, code == 0 and fake.find("POST", "/market/book-hours")[-1]["body"] == {"fleet": "F1", "start": "2027-01-05T10:00Z", "hours": 2} and "BK8" in out, f"book hours {start}: {out}")
    count = len(fake.find("POST", "/market/book-hours"))
    code, _, err = home.run("book", "hours", "F1", "2027-01-05T10:30", "2", "--yes")
    expect(failures, code == 1 and len(fake.find("POST", "/market/book-hours")) == count, "a booking at a half hour is sent")
    code, _, err = home.run("book", "hours", "F9", "2027-01-05T10", "2", "--yes")
    expect(failures, code == 1 and "Fleet List" in err and len(fake.find("POST", "/market/book-hours")) == count, "hours of a fleet that is not listed are booked")
    code, out, _ = home.run("waitlist", "lock", "144", "--yes")
    lock = fake.find("POST", "/market/waitlist")
    expect(failures, code == 0 and [r["body"] for r in lock] == [{"lock": "144"}] and session_auth(lock[0]) and "position 3" in out, f"waitlist lock: {out}")
    code, _, _ = home.run("waitlist", "lock", "lots", "--yes")
    expect(failures, code == 1 and len(fake.find("POST", "/market/waitlist")) == 1, "a lock that is not a number is sent")
    code, out, _ = home.run("waitlist", "leave", answers=["yes"])
    expect(failures, code == 0 and fake.find("POST", "/market/waitlist/leave")[0]["body"] == {}, "waitlist leave")


@scenario
def check_setup_claude_code(failures, home, fake):
    home.signed_in(fake.url)
    fake.route("GET", "/v1/models", lambda r: (200, {}, MODELS) if key_auth(r) else (401, {}, {}))
    fake.route("GET", "/v1/models/aliases", (200, {}, ALIASES))
    path = home.home / ".claude" / "settings.json"
    path.parent.mkdir()
    before = {"permissions": {"allow": ["Bash(ls)"]}, "env": {"FOO": "bar", "ANTHROPIC_MODEL": "claude-opus-5-5"}, "modelOverrides": {"claude-opus-4-6": "arn:x"}, "theme": "dark"}
    path.write_text(json.dumps(before))
    state = home.home / ".claude.json"
    state_before = {"numStartups": 41, "projects": {"/repo": {"allowedTools": []}}, "mcpServers": {"github": {"type": "stdio", "command": "gh-mcp"}}}
    state.write_text(json.dumps(state_before))
    state.chmod(0o644)
    code, out, err = home.run("setup", "claude-code")
    after = json.loads(path.read_text())
    server = {"type": "http", "url": fake.url + "/mcp", "headers": {"Authorization": "Bearer " + KEY}}
    written = json.loads(state.read_text())
    expect(failures, code == 0 and written == dict(state_before, mcpServers={"github": {"type": "stdio", "command": "gh-mcp"}, "sparkpipe": server}),
           f"setup claude-code registers the fleet's MCP server in ~/.claude.json beside the other servers and state: {written}")
    expect(failures, stat.S_IMODE(state.stat().st_mode) == 0o600 and [json.loads(p.read_text()) for p in backups(state)] == [state_before],
           "the Claude Code state file holding the MCP key is readable by others, or keeps no backup")
    env = {"ANTHROPIC_BASE_URL": fake.url, "ANTHROPIC_AUTH_TOKEN": KEY, "ANTHROPIC_MODEL": "kimi-k3+core", "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-5.3-flash",
           "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1000000", "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
           "CLAUDE_CODE_SUBAGENT_MODEL": "kimi-k3+core", "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1", "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "1800000"}
    overrides = {"claude-opus-4-6": "arn:x", "claude-opus-5-5": "kimi-k3+core", "claude-sonnet-5-5": "kimi-k3+core", "claude-fable-5-1": "kimi-k3+core", "claude-haiku-5-5": "glm-5.3-flash"}
    expect(failures, code == 0 and after == {"permissions": {"allow": ["Bash(ls)"]}, "env": dict({"FOO": "bar"}, **env), "modelOverrides": overrides, "theme": "dark"},
           f"setup claude-code writes {after} ({code} {err})")
    expect(failures, stat.S_IMODE(path.stat().st_mode) == 0o600, "the Claude Code settings holding the key are readable by others")
    saved = backups(path)
    expect(failures, len(saved) == 1 and json.loads(saved[0].read_text()) == before and stat.S_IMODE(saved[0].stat().st_mode) == 0o600 and saved[0].name in out,
           f"setup claude-code keeps no backup of the settings it replaced: {saved} {out}")
    text = path.read_text()
    code, out, _ = home.run("setup", "claude-code")
    expect(failures, code == 0 and path.read_text() == text and len(backups(path)) == 1 and "already holds" in out, "setup claude-code is not idempotent")
    fake.route("GET", "/v1/models/aliases", (200, {}, {"aliases": {"claude-*": "kimi-k3+core"}, "first_resident": "kimi-k3+core"}))
    elsewhere = home.root / "claude-config"
    code, out, _ = home.run("setup", "claude-code", "--model", "glm-5.3-flash", env={"CLAUDE_CONFIG_DIR": str(elsewhere), "ANTHROPIC_API_KEY": "sk-ant-x"})
    other = json.loads((elsewhere / "settings.json").read_text())["env"]
    expect(failures, code == 0 and other["ANTHROPIC_MODEL"] == other["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == other["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-5.3-flash"
           and other["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "200000" and "ANTHROPIC_API_KEY is also set" in out and not backups(elsewhere / "settings.json"),
           f"setup claude-code --model and CLAUDE_CONFIG_DIR: {other} {out}")
    expect(failures, json.loads((elsewhere / ".claude.json").read_text()) == {"mcpServers": {"sparkpipe": server}} and json.loads(state.read_text())["mcpServers"]["sparkpipe"] == server,
           "with CLAUDE_CONFIG_DIR the MCP server goes into $CLAUDE_CONFIG_DIR/.claude.json")
    legacy = home.root / "legacy-config"
    legacy.mkdir()
    (legacy / ".config.json").write_text(json.dumps({"numStartups": 2}))
    code, _, _ = home.run("setup", "claude-code", env={"CLAUDE_CONFIG_DIR": str(legacy)})
    expect(failures, code == 0 and json.loads((legacy / ".config.json").read_text()) == {"numStartups": 2, "mcpServers": {"sparkpipe": server}} and not (legacy / ".claude.json").exists(),
           "a Claude Code install that still reads the legacy .config.json gets the MCP server there")
    broken_state = home.root / "broken-state"
    broken_state.mkdir()
    (broken_state / ".claude.json").write_text('{"mcpServers": ')
    code, _, err = home.run("setup", "claude-code", env={"CLAUDE_CONFIG_DIR": str(broken_state)})
    expect(failures, code == 1 and "not valid JSON" in err and not (broken_state / "settings.json").exists() and (broken_state / ".claude.json").read_text() == '{"mcpServers": ',
           f"setup claude-code writes the settings although it cannot read .claude.json: {err}")
    fake.route("GET", "/v1/models/aliases", (200, {}, {"aliases": {"claude-*": None, "gpt-*": None}, "first_resident": "glm-5.3-flash"}))
    resident = home.root / "resident.json"
    code, _, err = home.run("setup", "claude-code", "--settings", resident)
    found = json.loads(resident.read_text())["env"] if resident.exists() else {}
    expect(failures, code == 0 and found.get("ANTHROPIC_MODEL") == found.get("ANTHROPIC_DEFAULT_HAIKU_MODEL") == "glm-5.3-flash",
           f"an alias without a target does not resolve to the first resident model: {found} {err}")
    real = home.root / "dotfiles" / "settings.json"
    real.parent.mkdir()
    real.write_text("{}")
    link = home.root / "linked.json"
    link.symlink_to(real)
    code, _, _ = home.run("setup", "claude-code", "--settings", link)
    expect(failures, code == 0 and link.is_symlink() and json.loads(real.read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == KEY and len(backups(real)) == 1,
           "setup claude-code replaces a symlinked settings file, or keeps its backup elsewhere")
    count = len(backups(path))
    path.write_text('{"env": {"FOO": "bar"},')
    code, _, err = home.run("setup", "claude-code")
    expect(failures, code == 1 and path.read_text() == '{"env": {"FOO": "bar"},' and "not valid JSON" in err and len(backups(path)) == count, "setup claude-code overwrites settings it cannot read")
    path.write_text('{"env": ["x"]}')
    code, _, _ = home.run("setup", "claude-code")
    expect(failures, code == 1 and path.read_text() == '{"env": ["x"]}', "setup claude-code replaces an env that is not an object")
    fake.route("GET", "/v1/models/aliases", (200, {}, {"aliases": {}, "first_resident": None}))
    code, _, err = home.run("setup", "claude-code", "--settings", home.root / "x.json")
    expect(failures, code == 1 and "--model" in err and not (home.root / "x.json").exists(), "setup without a main model writes a file")


CODEX_BEFORE = """# my codex config
model = "o4"
approval_policy = "on-request"

[model_providers.sparkpipe]
name = "old"
base_url = "https://old.example/v1"

[model_providers.other]
name = "Other"
base_url = "https://other.example/v1"

[profiles.sparkpipe.agents]
default_subagent_model = "stale"

[profiles.fast]
model = "o4-mini"

[mcp_servers.github]
command = "gh-mcp"
args = ["serve"]
"""


@scenario
def check_setup_codex(failures, home, fake):
    home.signed_in(fake.url)
    fake.route("GET", "/v1/models", (200, {}, MODELS))
    fake.route("GET", "/v1/models/aliases", (200, {}, ALIASES))
    path = home.home / ".codex" / "config.toml"
    profile = home.home / ".codex" / "sparkpipe.config.toml"
    path.parent.mkdir()
    path.write_text(CODEX_BEFORE)
    code, out, err = home.run("setup", "codex")
    text = path.read_text()
    doc = tomllib.loads(text)
    provider = f"""[model_providers.sparkpipe]
name = "SparkPipe"
base_url = "{fake.url}/v1"
env_key = "SPARKPIPE_API_KEY"
wire_api = "responses"
stream_idle_timeout_ms = 600000
model_catalog_url = "{fake.url}/v1/codex/models\""""
    expect(failures, code == 0 and text.startswith('model = "kimi-k3+core"\nmodel_provider = "sparkpipe"\n') and provider in text and "# my codex config" in text, f"setup codex writes:\n{text}\n{err}")
    wanted = {
        "model": "kimi-k3+core", "model_provider": "sparkpipe", "approval_policy": "on-request",
        "model_providers": {"other": {"name": "Other", "base_url": "https://other.example/v1"},
                            "sparkpipe": {"name": "SparkPipe", "base_url": fake.url + "/v1", "env_key": "SPARKPIPE_API_KEY", "wire_api": "responses", "stream_idle_timeout_ms": 600000,
                                          "model_catalog_url": fake.url + "/v1/codex/models"}},
        "profiles": {"sparkpipe": {"agents": {"default_subagent_model": "stale"}}, "fast": {"model": "o4-mini"}},
        "mcp_servers": {"github": {"command": "gh-mcp", "args": ["serve"]}, "sparkpipe": {"url": fake.url + "/mcp", "bearer_token_env_var": "SPARKPIPE_API_KEY"}},
    }
    expect(failures, doc == wanted, f"setup codex parses to {doc}")
    overlay = tomllib.loads(profile.read_text()) if profile.exists() else {}
    expect(failures, overlay == {"model": "kimi-k3+core", "model_provider": "sparkpipe", "model_context_window": 1000000, "model_auto_compact_token_limit": 967232,
                                 "agents": {"default_subagent_model": "kimi-k3+core", "max_concurrent_threads_per_session": 7}}, f"setup codex writes the profile {overlay}")
    expect(failures, "profiles" not in tomllib.loads(profile.read_text()) and "--profile sparkpipe" in out, "the Codex profile is not a top-level overlay")
    saved = backups(path)
    expect(failures, len(saved) == 1 and saved[0].read_text() == CODEX_BEFORE and not backups(profile), f"setup codex keeps no backup of config.toml: {saved}")
    expect(failures, KEY not in text + profile.read_text() and f"export SPARKPIPE_API_KEY={KEY}" in out, "setup codex writes the key into the config, or does not say where Codex reads it")
    overlay_text = profile.read_text()
    home.run("setup", "codex")
    expect(failures, path.read_text() == text and profile.read_text() == overlay_text and len(backups(path)) == 1 and not backups(profile), "setup codex is not idempotent")
    profile.write_text('model = "mine"\n')
    code, out, _ = home.run("setup", "codex", env={"SPARKPIPE_API_KEY": KEY})
    expect(failures, code == 0 and "export" not in out, "setup codex repeats the key reminder when SPARKPIPE_API_KEY is set")
    expect(failures, profile.read_text() == overlay_text and [p.read_text() for p in backups(profile)] == ['model = "mine"\n'], "setup codex replaces a changed profile without a backup")
    fresh = home.root / "codex-home"
    code, _, _ = home.run("setup", "codex", env={"CODEX_HOME": str(fresh)})
    expect(failures, code == 0 and tomllib.loads((fresh / "config.toml").read_text())["model"] == "kimi-k3+core" and (fresh / "sparkpipe.config.toml").exists(), "setup codex into an empty CODEX_HOME")
    for broken in ('model = "o4"\nmodel = \n', 'profiles.sparkpipe.model = "x"\nmodel_providers.sparkpipe.name = "x"\n', 'model_providers = { sparkpipe = { name = "x" } }\n',
                   'model = """\no4\n"""\n'):
        path.write_text(broken)
        count = len(backups(path))
        code, _, err = home.run("setup", "codex")
        expect(failures, code == 1 and path.read_text() == broken and "nothing was changed" in err.lower() and len(backups(path)) == count, f"setup codex over {broken!r}: {code} {err}")
    with patched(cli, "tomllib", None):
        path.write_text(CODEX_BEFORE)
        count = len(backups(path))
        code, _, err = home.run("setup", "codex")
        expect(failures, code == 1 and "Python 3.11" in err and path.read_text() == CODEX_BEFORE and len(backups(path)) == count, f"setup codex without tomllib merges unchecked: {err}")
        older = home.root / "codex-older-python"
        code, _, err = home.run("setup", "codex", env={"CODEX_HOME": str(older)})
        expect(failures, code == 0 and tomllib.loads((older / "config.toml").read_text())["model_providers"]["sparkpipe"]["base_url"] == fake.url + "/v1",
               f"setup codex without tomllib refuses an empty CODEX_HOME: {err}")
    fake.route("GET", "/v1/models", (200, {}, {"data": [{"id": "kimi-k3+core", "context_window": 1000000, "max_output_tokens": 32768}]}))
    path.write_text(CODEX_BEFORE)
    code, _, err = home.run("setup", "codex")
    expect(failures, code == 1 and "sequences" in err and path.read_text() == CODEX_BEFORE, "setup codex without the fleet's sequences")


@scenario
def check_setup_opencode(failures, home, fake):
    home.signed_in(fake.url)
    fake.route("GET", "/v1/models", (200, {}, MODELS))
    fake.route("GET", "/v1/models/aliases", (200, {}, ALIASES))
    path = home.home / ".config" / "opencode" / "opencode.json"
    path.parent.mkdir(parents=True)
    before = {"$schema": "https://opencode.ai/config.json", "theme": "tokyonight", "provider": {"anthropic": {"options": {"apiKey": "{env:ANTHROPIC_API_KEY}"}}},
              "agent": {"general": {"temperature": 0.2}, "reviewer": {"model": "anthropic/claude-sonnet-5-5"}}, "mcp": {"jira": {"type": "remote", "url": "https://jira.example/mcp"}},
              "compaction": {"auto": True}}
    path.write_text(json.dumps(before))
    code, out, err = home.run("setup", "opencode")
    after = json.loads(path.read_text())
    main, small = "sparkpipe/kimi-k3+core", "sparkpipe/glm-5.3-flash"
    wanted = {
        "$schema": "https://opencode.ai/config.json", "theme": "tokyonight",
        "provider": {"anthropic": {"options": {"apiKey": "{env:ANTHROPIC_API_KEY}"}},
                     "sparkpipe": {"npm": "@ai-sdk/anthropic", "name": "SparkPipe", "options": {"baseURL": fake.url + "/v1", "apiKey": "{env:SPARKPIPE_API_KEY}"},
                                   "models": {"kimi-k3+core": {"name": "kimi-k3+core", "limit": {"context": 1000000, "output": 32768}},
                                              "glm-5.3-flash": {"name": "glm-5.3-flash", "limit": {"context": 200000, "output": 16384}}}}},
        "agent": {"general": {"temperature": 0.2, "model": main}, "reviewer": {"model": "anthropic/claude-sonnet-5-5"}, "explore": {"model": main}, "compaction": {"model": main},
                  "title": {"model": small}, "summary": {"model": small}},
        "mcp": {"jira": {"type": "remote", "url": "https://jira.example/mcp"},
                "sparkpipe": {"type": "remote", "url": fake.url + "/mcp", "headers": {"Authorization": "Bearer {env:SPARKPIPE_API_KEY}"}, "enabled": True}},
        "compaction": {"auto": True},
        "model": main, "small_model": small, "share": "disabled",
    }
    expect(failures, code == 0 and after == wanted, f"setup opencode writes {after} ({err})")
    expect(failures, KEY not in path.read_text() and "SPARKPIPE_API_KEY" in out, "setup opencode writes the key into the config")
    saved = backups(path)
    expect(failures, len(saved) == 1 and json.loads(saved[0].read_text()) == before, "setup opencode keeps no backup of the config it replaced")
    text = path.read_text()
    home.run("setup", "opencode")
    expect(failures, path.read_text() == text and len(backups(path)) == 1, "setup opencode is not idempotent")
    path.write_text(json.dumps(dict(before, compaction={"auto": True, "prune": True})))
    code, _, _ = home.run("setup", "opencode")
    expect(failures, code == 0 and json.loads(path.read_text())["compaction"] == {"auto": True, "prune": False}, "setup opencode leaves pruning on")
    project = home.root / "repo" / "opencode.json"
    code, _, _ = home.run("setup", "opencode", "--config", project, "--haiku", "kimi-k3+core")
    written = json.loads(project.read_text())
    expect(failures, code == 0 and written["small_model"] == main and written["agent"]["title"] == {"model": main} and list(written["provider"]["sparkpipe"]["models"]) == ["kimi-k3+core"],
           f"setup opencode --config --haiku: {written}")
    path.write_text('{"mcp": "none"}')
    code, _, _ = home.run("setup", "opencode")
    expect(failures, code == 1 and path.read_text() == '{"mcp": "none"}', "setup opencode replaces an mcp that is not an object")


def conforms(value, shape, where):
    if isinstance(shape, dict):
        if not isinstance(value, dict):
            return [f"{where} is not an object"]
        found = []
        for name, inner in shape.items():
            if name == "*":
                for key, item in value.items():
                    found += conforms(item, inner, f"{where}.{key}")
            elif name not in value:
                found.append(f"{where}.{name} is missing")
            else:
                found += conforms(value[name], inner, f"{where}.{name}")
        return found
    if isinstance(shape, tuple):
        return [] if value in shape else [f"{where} is {value!r}, not one of {shape}"]
    if shape is int:
        return [] if isinstance(value, int) and not isinstance(value, bool) and value > 0 else [f"{where} is not a positive integer"]
    return [] if isinstance(value, shape) else [f"{where} is not a {shape.__name__}"]


CLAUDE_SETTINGS_SHAPE = {"env": {"ANTHROPIC_BASE_URL": str, "ANTHROPIC_AUTH_TOKEN": str, "ANTHROPIC_MODEL": str, "ANTHROPIC_DEFAULT_HAIKU_MODEL": str, "*": str},
                         "modelOverrides": {"*": str}}
CLAUDE_STATE_SHAPE = {"mcpServers": {"sparkpipe": {"type": ("http",), "url": str, "headers": {"*": str}}}}
CODEX_SHAPE = {"model": str, "model_provider": ("sparkpipe",),
               "model_providers": {"sparkpipe": {"name": str, "base_url": str, "env_key": str, "wire_api": ("responses",), "stream_idle_timeout_ms": int, "model_catalog_url": str}},
               "mcp_servers": {"sparkpipe": {"url": str, "bearer_token_env_var": str}}}
CODEX_PROFILE_SHAPE = {"model": str, "model_provider": ("sparkpipe",), "model_context_window": int, "model_auto_compact_token_limit": int,
                       "agents": {"default_subagent_model": str, "max_concurrent_threads_per_session": int}}
OPENCODE_SHAPE = {"$schema": ("https://opencode.ai/config.json",), "model": str, "small_model": str, "share": ("manual", "auto", "disabled"),
                  "provider": {"sparkpipe": {"npm": ("@ai-sdk/anthropic",), "name": str, "options": {"baseURL": str, "apiKey": str},
                                             "models": {"*": {"name": str, "limit": {"context": int, "output": int}}}}},
                  "agent": {name: {"model": str} for name in ("general", "explore", "compaction", "title", "summary")},
                  "mcp": {"sparkpipe": {"type": ("remote",), "url": str, "headers": {"*": str}, "enabled": bool}}}


@scenario
def check_setup_schemas(failures, home, fake):
    home.signed_in(fake.url)
    fake.route("GET", "/v1/models", (200, {}, MODELS))
    fake.route("GET", "/v1/models/aliases", (200, {}, ALIASES))
    for harness in ("claude-code", "codex", "opencode"):
        code, _, err = home.run("setup", harness)
        expect(failures, code == 0, f"setup {harness} on a fresh home: {err}")
    settings = json.loads((home.home / ".claude" / "settings.json").read_text())
    state = json.loads((home.home / ".claude.json").read_text())
    codex = tomllib.loads((home.home / ".codex" / "config.toml").read_text())
    profile = tomllib.loads((home.home / ".codex" / "sparkpipe.config.toml").read_text())
    opencode = json.loads((home.home / ".config" / "opencode" / "opencode.json").read_text())
    problems = (conforms(settings, CLAUDE_SETTINGS_SHAPE, "settings.json") + conforms(state, CLAUDE_STATE_SHAPE, ".claude.json") + conforms(codex, CODEX_SHAPE, "config.toml")
                + conforms(profile, CODEX_PROFILE_SHAPE, "sparkpipe.config.toml") + conforms(opencode, OPENCODE_SHAPE, "opencode.json"))
    models = set(opencode["provider"]["sparkpipe"]["models"])
    named = [opencode["model"], opencode["small_model"]] + [agent["model"] for agent in opencode["agent"].values()]
    problems += [f"opencode.json names {name}, which its provider does not list" for name in named if name.partition("/")[0] != "sparkpipe" or name.partition("/")[2] not in models]
    problems += [] if 0 < profile["model_auto_compact_token_limit"] < profile["model_context_window"] else ["the Codex compaction limit is not below its context window"]
    problems += [] if settings["env"]["ANTHROPIC_BASE_URL"] + "/mcp" == state["mcpServers"]["sparkpipe"]["url"] else ["the Claude Code MCP server is not on the fleet's API"]
    expect(failures, not problems, f"the written configs do not match each tool's documented keys: {problems}")


@scenario
def check_setup_undo(failures, home, fake):
    home.signed_in(fake.url)
    fake.route("GET", "/v1/models", (200, {}, MODELS))
    fake.route("GET", "/v1/models/aliases", (200, {}, ALIASES))
    settings, state = home.home / ".claude" / "settings.json", home.home / ".claude.json"
    codex, profile = home.home / ".codex" / "config.toml", home.home / ".codex" / "sparkpipe.config.toml"
    opencode = home.home / ".config" / "opencode" / "opencode.json"
    for path in (settings, codex, opencode):
        path.parent.mkdir(parents=True, exist_ok=True)
    settings_before = {"env": {"FOO": "bar", "ANTHROPIC_MODEL": "claude-opus-5-5", "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "90000"}, "modelOverrides": {"claude-opus-4-6": "arn:x"}, "theme": "dark"}
    state_before = {"numStartups": 41, "mcpServers": {"github": {"type": "stdio", "command": "gh-mcp"}}}
    opencode_before = {"theme": "tokyonight", "agent": {"general": {"temperature": 0.2}}, "compaction": {"auto": True, "prune": True}}
    settings.write_text(json.dumps(settings_before))
    state.write_text(json.dumps(state_before))
    codex.write_text(CODEX_BEFORE)
    opencode.write_text(json.dumps(opencode_before))
    code, out, _ = home.run("setup", "--undo")
    expect(failures, code == 0 and "nothing to undo" in out and settings.read_text() == json.dumps(settings_before), f"setup --undo before any setup: {out}")
    for harness in ("claude-code", "codex", "opencode"):
        home.run("setup", harness)
        home.run("setup", harness)
    added = json.loads(settings.read_text())
    expect(failures, added["env"]["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "1800000" and added["env"]["FOO"] == "bar",
           f"setup claude-code does not give Claude Code a 30-minute stream idle timeout, so a long prefill sending only pings is cut off: {added['env']}")
    added["permissions"] = {"allow": ["Bash(npm test)"]}
    settings.write_text(json.dumps(added))
    grown = json.loads(state.read_text())
    grown.update(numStartups=42, projects={"/repo": {"hasTrustDialogAccepted": True}})
    state.write_text(json.dumps(grown))
    codex.write_text(codex.read_text() + '\n[mcp_servers.linear]\nurl = "https://linear.example/mcp"\n')
    changed = json.loads(opencode.read_text())
    changed["theme"] = "gruvbox"
    opencode.write_text(json.dumps(changed))
    asked = len(fake.requests)
    code, out, err = home.run("setup", "claude-code", "--undo")
    expect(failures, code == 0 and json.loads(settings.read_text()) == dict(settings_before, permissions={"allow": ["Bash(npm test)"]}),
           f"setup claude-code --undo restores the settings it replaced and keeps the permission added since: {settings.read_text()} {err}")
    expect(failures, json.loads(state.read_text()) == dict(state_before, numStartups=42, projects={"/repo": {"hasTrustDialogAccepted": True}}),
           f"setup claude-code --undo removes the MCP server and keeps the state Claude Code wrote since: {state.read_text()}")
    expect(failures, len(fake.requests) == asked and "Restored" in out, f"setup --undo asks the fleet for something: {fake.requests[asked:]} {out}")
    expect(failures, tomllib.loads(codex.read_text())["model_providers"]["sparkpipe"]["name"] == "SparkPipe", "setup claude-code --undo touches the Codex config")
    code, out, _ = home.run("setup", "claude-code", "--undo")
    expect(failures, code == 0 and "nothing to undo" in out, f"a second setup claude-code --undo: {out}")
    code, out, err = home.run("setup", "--undo")
    wanted = tomllib.loads(CODEX_BEFORE)
    wanted["mcp_servers"]["linear"] = {"url": "https://linear.example/mcp"}
    expect(failures, code == 0 and tomllib.loads(codex.read_text()) == wanted and "# my codex config" in codex.read_text() and not profile.exists(),
           f"setup --undo restores config.toml, keeps the server added since and removes the profile it created:\n{codex.read_text()}\n{err}")
    expect(failures, json.loads(opencode.read_text()) == dict(opencode_before, theme="gruvbox"), f"setup --undo restores opencode.json and keeps the theme changed since: {opencode.read_text()}")
    expect(failures, cli.read_json(home.home / ".config" / "sparkpipe" / "setup.json") == {"files": {}}, "setup --undo leaves records of files it restored")
    fresh = home.root / "fresh-claude"
    home.run("setup", "claude-code", env={"CLAUDE_CONFIG_DIR": str(fresh)})
    made = sorted(p.name for p in fresh.iterdir() if not p.name.endswith(".bak"))
    code, out, _ = home.run("setup", "claude-code", "--undo", env={"CLAUDE_CONFIG_DIR": str(fresh)})
    expect(failures, made == [".claude.json", "settings.json"] and code == 0 and not (fresh / "settings.json").exists() and not (fresh / ".claude.json").exists() and "Removed" in out,
           f"setup claude-code --undo removes the files setup created: {made} {out}")
    home.run("setup", "opencode")
    original = cli.setup_files(cli.read_json(home.home / ".config" / "sparkpipe" / "setup.json"))[os.path.realpath(opencode)]["original"]
    os.unlink(original)
    text = opencode.read_text()
    code, _, err = home.run("setup", "opencode", "--undo")
    expect(failures, code == 1 and "backup" in err and "Nothing was changed" in err and opencode.read_text() == text, f"setup --undo without the backup: {err}")
    code, _, err = home.run("setup")
    expect(failures, code == 1 and "Name a harness" in err, f"setup without a harness: {err}")


def check_tokens(failures):
    for platform in ("darwin", "linux"):
        with tempfile.TemporaryDirectory() as folder:
            tokens_on(failures, Home(folder, platform), platform)
    with tempfile.TemporaryDirectory() as folder:
        code, _, err = Home(folder, "win32").run("tokens", "set", "hf", secrets=[SECRET])
        expect(failures, code == 1 and "keychain" in err, "tokens on a system with no keychain")


def tokens_on(failures, home, platform):
    code, out, err = home.run("tokens", "set", "hf", secrets=[SECRET])
    expect(failures, code == 0 and "fine-grained, read-only" in out and "huggingface.co" in out, f"{platform}: tokens set hf: {code} {out} {err}")
    expect(failures, home.keychain() == {"sparkpipe/hf": SECRET}, f"{platform}: the token is not in the keychain: {home.keychain()}")
    expect(failures, home.cfg().get("tokens") == {"hf": {"host": "huggingface.co"}} and SECRET not in json.dumps(home.cfg()), f"{platform}: the config holds the token or misses its host")
    expect(failures, SECRET not in home.argv_log() and SECRET.encode().hex() not in home.argv_log(), f"{platform}: the token is passed on a command line")
    expect(failures, SECRET not in out + err, f"{platform}: the token is printed")
    code, _, err = home.run("tokens", "set", "r2", secrets=["r2secret"])
    expect(failures, code == 1 and "--host" in err and "sparkpipe/r2" not in home.keychain(), f"{platform}: a token without a host is stored")
    code, out, _ = home.run("tokens", "set", "r2", "--host", "Acct.R2.Cloudflarestorage.com", secrets=["r2secret"])
    expect(failures, code == 0 and home.cfg()["tokens"]["r2"] == {"host": "acct.r2.cloudflarestorage.com"}, f"{platform}: tokens set --host")
    code, _, _ = home.run("tokens", "set", "bad name", "--host", "x.example", secrets=["x"])
    expect(failures, code == 1, f"{platform}: a token name with a space is accepted")
    code, _, _ = home.run("tokens", "set", "host", "--host", "x.example\n", secrets=["x"])
    expect(failures, code == 1 and "host" not in home.cfg()["tokens"], f"{platform}: a host with a line break is accepted")
    code, _, _ = home.run("tokens", "set", "empty", "--host", "x.example", secrets=["  "])
    expect(failures, code == 1 and "empty" not in home.cfg()["tokens"], f"{platform}: an empty token is stored")
    for value in ("hf_split token", "hf_line\nbreak", "hf_bell\x07"):
        code, out, err = home.run("tokens", "set", "odd", "--host", "x.example", secrets=[value])
        expect(failures, code == 1 and "odd" not in home.cfg()["tokens"] and "sparkpipe/odd" not in home.keychain() and value not in out + err, f"{platform}: the token {value!r} is stored")
    code, out, _ = home.run("tokens", "ls")
    expect(failures, code == 0 and out.splitlines() == ["hf  huggingface.co", "r2  acct.r2.cloudflarestorage.com"] and SECRET not in out, f"{platform}: tokens ls: {out}")
    store = home.keychain()
    del store["sparkpipe/r2"]
    (home.state / "keychain.json").write_text(json.dumps(store))
    code, out, _ = home.run("tokens", "ls", "--json")
    expect(failures, code == 0 and json.loads(out) == [{"name": "hf", "host": "huggingface.co", "stored": True}, {"name": "r2", "host": "acct.r2.cloudflarestorage.com", "stored": False}],
           f"{platform}: tokens ls does not notice a token missing from the keychain: {out}")

    def stuck(command, **options):
        raise subprocess.TimeoutExpired(command, options.get("timeout"))

    with patched(cli.subprocess, "run", stuck):
        code, _, err = home.run("tokens", "ls")
    expect(failures, code == 1 and "did not finish" in err and "Traceback" not in err, f"{platform}: a keychain call that hangs: {err}")
    code, out, _ = home.run("tokens", "rm", "hf")
    expect(failures, code == 0 and "sparkpipe/hf" not in home.keychain() and "hf" not in home.cfg()["tokens"], f"{platform}: tokens rm")
    code, _, _ = home.run("tokens", "rm", "hf")
    expect(failures, code == 1, f"{platform}: removing an unknown token succeeds")


@scenario
def check_fetch(failures, home, fake):
    home.signed_in(fake.url)
    home.run("tokens", "set", "hf", secrets=[SECRET])
    record = {"id": "fx_1", "state": "needs_token", "source": "hf://org/model@rev", "to": "models/x/", "bytes": 500000000000, "total_bytes": 1500000000000}
    fake.route("POST", "/v1/fetch", (200, {}, {"id": "fx_1"}))
    fake.route("GET", "/v1/fetch", (200, {}, {"data": [record]}))
    fake.route("GET", "/v1/fetch/fx_1", (200, {}, record))
    fake.route("POST", "/v1/fetch/fx_1/resume", (200, {}, dict(record, state="running")))
    fake.route("DELETE", "/v1/fetch/fx_1", (200, {}, dict(record, state="cancelled")))
    code, out, err = home.run("fetch", "hf://org/model@rev", "models/x/", "--include", "*.safetensors", "--include", "*.json", "--then", "pack")
    sent = fake.find("POST", "/v1/fetch")[-1]
    expect(failures, code == 0 and sent["body"] == {"source": "hf://org/model@rev", "to": "models/x/", "include": ["*.safetensors", "*.json"], "then": "pack"} and key_auth(sent)
           and sent["headers"].get("x-sparkpipe-source-token") == SECRET and "fx_1" in out and SECRET not in out + err, f"fetch with the hf token: {sent['body']} {out} {err}")
    for source, token in (("https://data.example.com/set.tar", None), ("https://huggingface.co.evil.example/x", None), ("https://cdn-lfs.huggingface.co/x", SECRET),
                          ("git+https://huggingface.co/datasets/org/set", SECRET), ("s3://bucket/key", None)):
        code, _, err = home.run("fetch", source, "data/")
        sent = fake.find("POST", "/v1/fetch")[-1]
        expect(failures, code == 0 and sent["body"]["source"] == source and sent["headers"].get("x-sparkpipe-source-token") == token, f"fetch {source} sends the token {sent['headers'].get('x-sparkpipe-source-token')!r}")
    count = len(fake.find("POST", "/v1/fetch"))
    code, _, err = home.run("fetch", "https://data.example.com/x", "data/", "--token", "hf")
    expect(failures, code == 1 and "huggingface.co" in err and len(fake.find("POST", "/v1/fetch")) == count, "a token is sent to a host it does not belong to")
    code, _, _ = home.run("fetch", "hf://org/model", "models/", "--no-token")
    expect(failures, code == 0 and "x-sparkpipe-source-token" not in fake.find("POST", "/v1/fetch")[-1]["headers"], "fetch --no-token sends a token")
    count = len(fake.find("POST", "/v1/fetch"))
    for argv in (("ftp://x/y", "a/"), ("http://data.example.com/x", "a/"), ("hf://org/model",), ("resume",), ("ls", "extra")):
        code, _, _ = home.run("fetch", *argv)
        expect(failures, code == 1, f"fetch {argv} is accepted")
    expect(failures, len(fake.find("POST", "/v1/fetch")) == count, "a refused fetch reaches the API")
    code, out, _ = home.run("fetch", "ls")
    expect(failures, code == 0 and "fx_1  needs_token  hf://org/model@rev to models/x/  500.0 GB of 1.5 TB" in out and key_auth(fake.find("GET", "/v1/fetch")[-1]), f"fetch ls: {out}")
    code, out, _ = home.run("fetch", "resume", "fx_1")
    resumed = fake.find("POST", "/v1/fetch/fx_1/resume")
    expect(failures, code == 0 and len(resumed) == 1 and resumed[0]["headers"].get("x-sparkpipe-source-token") == SECRET and key_auth(resumed[0]), f"fetch resume: {out}")
    code, out, _ = home.run("fetch", "cancel", "fx_1")
    expect(failures, code == 0 and len(fake.find("DELETE", "/v1/fetch/fx_1")) == 1 and "cancelled" in out, "fetch cancel")
    home.run("tokens", "set", "hf2", "--host", "huggingface.co", secrets=["hf_other"])
    code, _, err = home.run("fetch", "hf://org/model", "models/")
    expect(failures, code == 1 and "--token" in err, "fetch picks one of two tokens for the same host")
    code, _, _ = home.run("fetch", "hf://org/model", "models/", "--token", "hf2")
    expect(failures, code == 0 and fake.find("POST", "/v1/fetch")[-1]["headers"].get("x-sparkpipe-source-token") == "hf_other", "fetch --token hf2")
    home.run("tokens", "rm", "hf")
    home.run("tokens", "rm", "hf2")
    fake.route("GET", "/v1/fetch/fx_1", (200, {}, dict(record, source="https://data.example.com/x")))
    code, _, err = home.run("fetch", "resume", "fx_1")
    expect(failures, code == 1 and "tokens set" in err and len(fake.find("POST", "/v1/fetch/fx_1/resume")) == 1, "fetch resume without a token for the host")
    home.run("tokens", "set", "hf", secrets=[SECRET])
    (home.state / "keychain.json").write_text(json.dumps({"sparkpipe/hf": "hf_damaged\rvalue"}))
    count = len(fake.find("POST", "/v1/fetch"))
    code, _, err = home.run("fetch", "hf://org/model", "models/")
    expect(failures, code == 1 and "hf_damaged" not in err and len(fake.find("POST", "/v1/fetch")) == count, f"a keychain token with a control character is sent or printed: {err}")
    elsewhere = Fake()
    try:
        home.run("tokens", "set", "hf", secrets=[SECRET])
        fake.route("POST", "/v1/fetch", (307, {"Location": elsewhere.url + "/v1/fetch"}, {}))
        code, _, err = home.run("fetch", "hf://org/model", "models/")
        expect(failures, code == 1 and "redirect" in err and not elsewhere.requests, "a redirect from the API carries the key and token to another host")
    finally:
        elsewhere.close()


def storage_files(fake, files):
    for path, data in files.items():
        fake.route("GET", "/v1/storage/" + path, (200, {"Content-Type": "application/octet-stream"}, data))


def change(path, data=None, kind="added"):
    found = {"path": path, "change": kind}
    if data is not None:
        found.update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    return found


def tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()}


@scenario
def check_storage_sync(failures, home, fake):
    home.signed_in(fake.url)
    first = {"models/a.bin": b"A" * 70000, "models/sub/b.json": b'{"b": 1}', "models/sub/with space.txt": b"spaced"}
    storage_files(fake, first)
    pages = {
        None: {"data": [change("models/a.bin", first["models/a.bin"]), change("models/sub/b.json", first["models/sub/b.json"])], "until": "c1", "more": True},
        "c1": {"data": [change("models/sub/with space.txt", first["models/sub/with space.txt"])], "until": "c2", "more": False},
        "c2": {"data": [change("models/a.bin", kind="deleted"), change("models/sub/b.json", b'{"b": 2}', "changed")], "until": "c3", "more": False},
        "c3": {"data": [change("models/a.bin", kind="deleted")], "until": "c4", "more": False},
    }
    fake.route("GET", "/v1/storage/changes", lambda r: (200, {}, pages[r["query"].get("since", [None])[0]]))
    dest = home.root / "backup"
    code, out, err = home.run("storage", "sync", "models/", dest)
    calls = fake.find("GET", "/v1/storage/changes")
    expect(failures, code == 0 and all(key_auth(c) for c in calls) and [c["query"] for c in calls] == [{"prefix": ["models/"]}, {"prefix": ["models/"], "since": ["c1"]}],
           f"storage sync pages: {[c['query'] for c in calls]} {err}")
    expect(failures, tree(dest) == {k[len("models/"):]: v for k, v in first.items()}, f"storage sync copies the wrong files: {sorted(tree(dest))}")
    expect(failures, "/v1/storage/models/sub/with%20space.txt" in [r["raw_path"] for r in fake.requests], "a storage path is not quoted")
    expect(failures, home.cfg()["sync"] == {f"{fake.url} models/ {dest.resolve()}": "c2"} and "Copied 3 file(s)" in out, f"storage sync state: {home.cfg()['sync']} {out}")
    fake.route("GET", "/v1/storage/models/sub/b.json", (200, {}, b'{"b": 2}'))
    code, out, _ = home.run("storage", "sync", "models", str(dest))
    expect(failures, code == 0 and fake.find("GET", "/v1/storage/changes")[-1]["query"].get("since") == ["c2"] and (dest / "a.bin").exists()
           and (dest / "sub/b.json").read_bytes() == b'{"b": 2}' and "--delete" in out, f"a second sync: {out}")
    code, out, _ = home.run("storage", "sync", "models/", dest, "--delete")
    expect(failures, code == 0 and not (dest / "a.bin").exists() and (dest / "sub/b.json").exists(), "storage sync --delete")
    for bad, reason in (("models/../../escape.txt", "unsafe"), ("other/x.txt", "outside")):
        fake.route("GET", "/v1/storage/changes", (200, {}, {"data": [change(bad, b"x")], "until": "z", "more": False}))
        fake.route("GET", "/v1/storage/" + bad, (200, {}, b"x"))
        target = home.root / ("bad-" + reason)
        code, _, err = home.run("storage", "sync", "models/", target)
        expect(failures, code == 1 and reason in err and not list(home.root.rglob("escape.txt")) and f"{fake.url} models/ {target.resolve()}" not in home.cfg()["sync"],
               f"storage sync accepts the path {bad}: {err}")
    fake.route("GET", "/v1/storage/models/a.bin", (200, {}, first["models/a.bin"]))
    fake.route("GET", "/v1/storage/changes", (200, {}, {"data": [dict(change("models/a.bin", first["models/a.bin"]), sha256="0" * 64)], "until": "z", "more": False}))
    target = home.root / "bad-sha"
    code, out, err = home.run("storage", "sync", "models/", target)
    expect(failures, code == 0 and "SHA-256" in err and "Skipped 1" in out and not (target / "a.bin").exists() and not list(target.glob(".sparkpipe-*")),
           f"a file with the wrong SHA-256 is kept, or stops the sync: {code} {out} {err}")
    fake.route("GET", "/v1/storage/changes", (200, {}, {"data": [dict(change("models/a.bin", first["models/a.bin"]), sha256="not-a-sha")], "until": "z", "more": False}))
    code, _, err = home.run("storage", "sync", "models/", home.root / "odd-sha")
    expect(failures, code == 1 and "64 hexadecimal" in err, "a listed SHA-256 that is not one is accepted")
    fake.route("GET", "/v1/storage/changes", (200, {}, {"data": [], "until": None, "more": True}))
    code, _, err = home.run("storage", "sync", "models/", home.root / "loop")
    expect(failures, code == 1 and "did not advance" in err, "a change list that never advances loops")
    resumable(failures, home, fake)
    bucket_sync(failures, home, fake, first, pages)
    code, _, _ = home.run("storage", "sync", "models/", "gs://bucket/x")
    expect(failures, code == 1, "storage sync accepts a gs:// destination it cannot write")


def resumable(failures, home, fake):
    files = {"models/p1.bin": b"one" * 100, "models/p2.bin": b"two" * 100}
    storage_files(fake, files)
    fake.route("GET", "/v1/storage/models/gone.bin", (404, {}, {"error": "Not found."}))
    fake.route("GET", "/v1/storage/models/moved.bin", (200, {}, b"the newer content"))
    pages = {
        None: {"data": [change("models/p1.bin", files["models/p1.bin"]), change("models/gone.bin", b"gone"), change("models/moved.bin", b"older")], "until": "r1", "more": True},
        "r1": {"data": [change("models/p2.bin", files["models/p2.bin"]), change("models/../escape", b"x")], "until": "r2", "more": False},
    }
    fake.route("GET", "/v1/storage/changes", lambda r: (200, {}, pages[r["query"].get("since", [None])[0]]))
    dest = home.root / "resumable"
    key = f"{fake.url} models/ {dest.resolve()}"
    code, out, err = home.run("storage", "sync", "models/", dest)
    expect(failures, code == 1 and "unsafe" in err and "gone.bin: deleted on the fleet" in err and "moved.bin" in err and home.cfg()["sync"].get(key) == "r1"
           and tree(dest) == {"p1.bin": files["models/p1.bin"], "p2.bin": files["models/p2.bin"]}, f"a failed sync does not keep the cursor of its last whole page: {home.cfg()['sync'].get(key)} {err}")
    pages["r1"]["data"].pop()
    downloads = len(fake.find("GET", "/v1/storage/models/p2.bin"))
    code, out, err = home.run("storage", "sync", "models/", dest)
    expect(failures, code == 0 and fake.find("GET", "/v1/storage/changes")[-1]["query"].get("since") == ["r1"] and len(fake.find("GET", "/v1/storage/models/p2.bin")) == downloads
           and "1 file(s) were already there" in out and home.cfg()["sync"][key] == "r2", f"a resumed sync downloads a file it already has: {out} {err}")


def bucket_sync(failures, home, fake, first, pages):
    fake.route("GET", "/v1/storage/changes", lambda r: (200, {}, pages[r["query"].get("since", [None])[0]]))
    fake.route("GET", "/v1/storage/models/sub/b.json", (200, {}, first["models/sub/b.json"]))
    timeouts = []
    real = subprocess.run

    def recording(command, **options):
        timeouts.append((command[:3], options.get("timeout")))
        return real(command, **options)

    with patched(cli.subprocess, "run", recording):
        code, out, err = home.run("storage", "sync", "models/", "s3://bucket/backup")
    bucket = home.state / "bucket" / "bucket" / "backup"
    expect(failures, code == 0 and tree(bucket) == {k[len("models/"):]: v for k, v in first.items()}, f"storage sync to s3:// copies the wrong files: {out} {err}")
    copies = [args for args in home.aws_log() if args[:2] == ["s3", "cp"]]
    expect(failures, len(copies) == 3 and all(f"{cli.SHA_METADATA}={hashlib.sha256(first['models/' + args[-1].split('/backup/', 1)[1]]).hexdigest()}" in args for args in copies),
           f"the bucket copies do not carry their SHA-256: {copies}")
    expect(failures, [t for c, t in timeouts if c == ["aws", "s3", "cp"]] == [None, None, None] and all(t == cli.TOOL_TIMEOUT for c, t in timeouts if c == ["aws", "s3api", "head-object"]),
           f"a bucket upload has a time limit: {timeouts}")
    cfg = home.cfg()
    cfg["sync"].pop(f"{fake.url} models/ s3://bucket/backup")
    home.config.write_text(json.dumps(cfg))
    code, out, _ = home.run("storage", "sync", "models/", "s3://bucket/backup/")
    expect(failures, code == 0 and len([a for a in home.aws_log() if a[:2] == ["s3", "cp"]]) == 3 and "3 file(s) were already there" in out, f"a bucket sync copies files it already has: {out}")


def proxy_upstream(fake, released):
    attempts = []

    def messages(request):
        attempts.append(request)
        if len(attempts) == 1:
            return 429, {"retry-after": "0", "x-should-retry": "true"}, {"type": "error", "error": {"type": "rate_limit_error", "message": "busy"}}
        return 200, {"Content-Type": "application/json"}, {"id": "msg_1", "type": "message", "content": [{"type": "text", "text": "hi"}]}

    def later():
        released["streamed"] = released["event"].wait(5)
        return b"data: second\n\n"

    def echo(handler):
        handler.wfile.write(b"ready")
        handler.wfile.write(b"echo:" + handler.rfile.read1(65536))

    def upgrade(request):
        if request["headers"].get("upgrade") != "websocket":
            return 400, {}, {"error": "not an upgrade"}
        return 101, {"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Accept": "accepted"}, echo

    fake.route("POST", "/pay/api/v1/messages", messages)
    fake.route("POST", "/pay/api/v1/messages/count_tokens", (200, {}, {"input_tokens": 3}))
    fake.route("POST", "/pay/api/v1/chat/completions", (200, {"Content-Type": "text/event-stream"}, [b"data: first\n\n", later, b"data: [DONE]\n\n"]))
    fake.route("GET", "/pay/api/v1/models", (429, {"retry-after": "120"}, {"type": "error", "error": {"type": "rate_limit_error", "message": "busy"}}))
    fake.route("GET", "/pay/api/v1/responses", upgrade)
    return attempts


def proxy_request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(method, path, body=body, headers=headers or {})
    return connection, connection.getresponse()


def proxy_status(port, method, path, headers):
    connection, response = proxy_request(port, method, path, b"{}" if method == "POST" else None, headers)
    response.read()
    connection.close()
    return response.status


def received(sock, buffer, marker):
    while marker not in buffer:
        data = sock.recv(65536)
        if not data:
            break
        buffer += data
    return buffer


def proxy_websocket(failures, port, fake):
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        sock.sendall(f"GET /v1/responses HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGVzdA==\r\n"
                     f"Sec-WebSocket-Version: 13\r\nx-api-key: {TOKEN}\r\n\r\n".encode())
        head, _, rest = received(sock, b"", b"ready").partition(b"\r\n\r\n")
        sock.sendall(b"ping")
        tail = received(sock, rest, b"\x00end")
    finally:
        sock.close()
    sent = fake.find("GET", "/pay/api/v1/responses")[-1]["headers"]
    lower = head.lower()
    expect(failures, head.startswith(b"HTTP/1.1 101") and b"upgrade: websocket" in lower and b"sec-websocket-accept: accepted" in lower and tail == b"readyecho:ping",
           f"the proxy does not carry the Responses WebSocket: {head!r} {tail!r}")
    expect(failures, sent.get("authorization") == "Bearer " + KEY and "x-api-key" not in sent and sent.get("upgrade") == "websocket" and sent.get("connection", "").lower() == "upgrade"
           and sent.get("sec-websocket-key") == "dGVzdA==" and sent.get("x-sparkpipe-fleet") == "F3", f"the WebSocket upgrade is forwarded as {sent}")


def proxy_refusals(failures, port, fake):
    for headers, method, path, status in (
        ({"x-api-key": TOKEN, "Host": "rebind.attacker.example"}, "GET", "/v1/models", 403),
        ({"x-api-key": TOKEN, "Host": f"rebind.attacker.example:{port}"}, "POST", "/v1/messages", 403),
        ({"x-api-key": TOKEN, "Host": "rebind.attacker.example"}, "GET", "/v1/storage/models/x", 403),
        ({"x-api-key": TOKEN, "Origin": "https://attacker.example"}, "POST", "/v1/messages", 403),
        ({"x-api-key": TOKEN, "Origin": "null"}, "GET", "/v1/models", 403),
        ({}, "POST", "/v1/messages", 401),
        ({"x-api-key": "anything", "Authorization": "Bearer anything"}, "POST", "/v1/messages", 401),
        ({"Authorization": "Bearer " + KEY}, "POST", "/v1/messages", 401),
        ({"x-api-key": TOKEN}, "GET", "/v1/storage/models/x", 404),
        ({"x-api-key": TOKEN}, "DELETE", "/v1/cache?storage=true", 404),
        ({"x-api-key": TOKEN}, "GET", "/v1/state", 404),
        ({"x-api-key": TOKEN}, "POST", "//v1/messages", 404),
        ({"x-api-key": TOKEN}, "POST", "/v1/messages/../storage/x", 404),
        ({"x-api-key": TOKEN}, "POST", "/v1/functions/x/runs", 404),
    ):
        before = len(fake.requests)
        got = proxy_status(port, method, path, headers)
        expect(failures, got == status and len(fake.requests) == before, f"the proxy answers {got} to {method} {path} with {headers} and sends {len(fake.requests) - before} request(s) upstream")
    got = proxy_status(port, "POST", "/v1/messages/count_tokens", {"x-api-key": TOKEN, "Host": f"localhost:{port}"})
    expect(failures, got == 200 and fake.find("POST", "/pay/api/v1/messages/count_tokens"), f"the proxy refuses Host localhost: {got}")


@scenario
def check_proxy(failures, home, fake):
    released = {"event": threading.Event(), "streamed": None}
    attempts = proxy_upstream(fake, released)
    server = cli.proxy_server(fake.url + "/pay/api", KEY, "F3", 0, 3, 5, TOKEN, log=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        body = json.dumps({"model": "kimi-k3+core", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}).encode()
        connection, response = proxy_request(port, "POST", "/v1/messages", body, {"x-api-key": TOKEN, "X-SparkPipe-Fleet": "F9", "Cookie": "a=b", "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        answer = json.loads(response.read())
        connection.close()
        expect(failures, response.status == 200 and answer["id"] == "msg_1" and len(attempts) == 2, f"the proxy does not retry a 429 with retry-after: {response.status} {len(attempts)}")
        for sent in attempts:
            headers = sent["headers"]
            expect(failures, headers.get("authorization") == "Bearer " + KEY and "x-api-key" not in headers and "cookie" not in headers and headers.get("x-sparkpipe-fleet") == "F3"
                   and sent["raw"] == body and headers.get("anthropic-version") == "2023-06-01", f"the proxy forwards {headers}")
        connection, response = proxy_request(port, "POST", "/v1/chat/completions", b'{"stream": true}', {"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN})
        first = response.read1(65536)
        released["event"].set()
        rest = response.read()
        connection.close()
        expect(failures, response.getheader("Transfer-Encoding") == "chunked" and first == b"data: first\n\n" and rest == b"data: second\n\ndata: [DONE]\n\n" and released["streamed"],
               f"the proxy does not stream: {first!r} {rest!r} {released['streamed']}")
        connection, response = proxy_request(port, "GET", "/v1/models", headers={"x-api-key": TOKEN})
        response.read()
        connection.close()
        expect(failures, response.status == 429 and response.getheader("retry-after") == "120" and len(fake.find("GET", "/pay/api/v1/models")) == 1, "the proxy waits out a retry-after beyond --max-wait")
        connection, response = proxy_request(port, "POST", "/v1/messages", b"x" * 10, {"Transfer-Encoding": "chunked", "x-api-key": TOKEN})
        response.read()
        connection.close()
        expect(failures, response.status == 411, "the proxy accepts a request without a Content-Length")
        proxy_refusals(failures, port, fake)
        proxy_websocket(failures, port, fake)
    finally:
        server.shutdown()
        server.server_close()
    dead = cli.proxy_server("http://127.0.0.1:9", KEY, None, 0, 0, 5, TOKEN, log=False)
    threading.Thread(target=dead.serve_forever, daemon=True).start()
    try:
        connection, response = proxy_request(dead.server_address[1], "GET", "/v1/models", headers={"x-api-key": TOKEN})
        answer = json.loads(response.read())
        connection.close()
        expect(failures, response.status == 502 and answer["error"]["type"] == "upstream_error", "an unreachable API does not answer 502")
    finally:
        dead.shutdown()
        dead.server_close()
    proxy_command(failures, home, fake)


def proxy_command(failures, home, fake):
    tokens = []

    class Stub:
        server_address = ("127.0.0.1", 8790)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    def stub(api, key, fleet, port, retries, max_wait, token, log=True):
        tokens.append(token)
        return Stub()

    home.signed_in(fake.url)
    with patched(cli, "proxy_server", stub):
        runs = [home.run("proxy"), home.run("proxy")]
    expect(failures, [code for code, _, _ in runs] == [0, 0] and len(tokens) == 2 and tokens[0] != tokens[1] and all(t.startswith("spl-") and len(t) > 30 for t in tokens)
           and all(t in out for t, (_, out, _) in zip(tokens, runs)) and not any(KEY in out for _, out, _ in runs), f"the proxy does not print a new local key on each start: {runs}")
    home.config.write_text(json.dumps({"api": fake.url}))
    code, _, err = home.run("proxy", "--port", "0")
    expect(failures, code == 1 and "API key" in err, "the proxy starts without a key")


@scenario
def check_errors(failures, home, fake):
    home.signed_in(fake.url)
    fake.route("GET", "/v1/state", (200, {}, {"fleet": "F3"}))
    fake.route("GET", "/market/state", (200, {}, b"[1, 2]"))
    code, _, err = home.run("keys", "ls")
    expect(failures, code == 1 and "not an object" in err and "Traceback" not in err, f"a JSON answer that is not an object: {err}")
    for env, stored in (({"SPARKPIPE_API_KEY": KEY + "\r"}, {}), ({}, {"key": KEY + "\n"}), ({}, {"session": SESSION + "\r\n"})):
        home.signed_in(fake.url, **stored)
        before = len(fake.requests)
        code, out, err = home.run("status", env=env)
        expect(failures, code == 1 and KEY not in out + err and SESSION not in out + err and "Traceback" not in err and len(fake.requests) == before,
               f"a key or session with a control character is sent or printed: {err}")
    home.signed_in(fake.url)
    code, _, err = home.run("status", "--fleet", "F3\r\n")
    expect(failures, code == 1 and "fleet name" in err, f"a fleet name with a line break is sent: {err}")
    fake.route("GET", "/v1/models", (200, {"Content-Length": "100000"}, b'{"data": ['))
    fake.route("GET", "/v1/models/aliases", (200, {}, ALIASES))
    code, _, err = home.run("setup", "claude-code", "--settings", home.root / "cut.json")
    expect(failures, code == 1 and "broke off (IncompleteRead)" in err and not (home.root / "cut.json").exists(), f"an answer cut short: {err}")

    def slow():
        time.sleep(1.5)
        return b"]}"

    fake.route("GET", "/v1/models", (200, {}, [b'{"data": [', slow]))
    with patched(cli, "TIMEOUT", 0.3):
        code, _, err = home.run("setup", "claude-code", "--settings", home.root / "slow.json")
    expect(failures, code == 1 and "broke off" in err and "Traceback" not in err, f"an answer that stalls: {err}")

    def broken(ctx):
        raise ValueError(f"Invalid header value b'Bearer {KEY}'")

    with patched(cli, "status", broken):
        code, _, err = home.run("status")
        expect(failures, code == 70 and KEY not in err and "SPARKPIPE_DEBUG" in err and "ValueError" in err, f"an unexpected error prints its message: {err}")
        try:
            home.run("status", env={"SPARKPIPE_DEBUG": "1"})
            failures.append("SPARKPIPE_DEBUG does not raise the error")
        except ValueError:
            pass


def check_package(failures):
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "dist"
        with contextlib.redirect_stdout(io.StringIO()):
            code = build.main(["--out", str(out)])
        names = sorted(p.name for p in out.iterdir())
        release = cli.VERSION
        expect(failures, code == 0 and names == [f"sparkpipe-{release}-py3-none-any.whl", f"sparkpipe-{release}.tar.gz"], f"the build makes {names}")
        wheel = zipfile.ZipFile(out / f"sparkpipe-{release}-py3-none-any.whl")
        info = f"sparkpipe-{release}.dist-info"
        expect(failures, wheel.read("sparkpipe_cli.py") == (ROOT / "tools" / "sparkpipe_cli.py").read_bytes(), "the wheel's module is not tools/sparkpipe_cli.py")
        meta = wheel.read(f"{info}/METADATA").decode()
        for line in ("Name: sparkpipe", f"Version: {release}", "License: FSL-1.1-ALv2", "Requires-Python: >=3.9"):
            expect(failures, line in meta.splitlines(), f"the wheel's METADATA misses {line}")
        expect(failures, "[console_scripts]\nsparkpipe = sparkpipe_cli:main" in wheel.read(f"{info}/entry_points.txt").decode(), "the wheel has no sparkpipe command")
        rows = [line.split(",") for line in wheel.read(f"{info}/RECORD").decode().splitlines()]
        for name, digest, length in rows:
            data = wheel.read(name)
            good = digest == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode() and length == str(len(data)) if digest else name.endswith("RECORD")
            expect(failures, good, f"the wheel RECORD is wrong for {name}")
        expect(failures, sorted(r[0] for r in rows) == sorted(wheel.namelist()), "the wheel RECORD does not list every file")
        sdist = tarfile.open(out / f"sparkpipe-{release}.tar.gz")
        top = f"sparkpipe-{release}"
        expect(failures, sorted(sdist.getnames()) == sorted(f"{top}/{n}" for n in ("PKG-INFO", "pyproject.toml", "README.md", "LICENSE", "sparkpipe_cli.py")), f"the sdist holds {sdist.getnames()}")
        project = tomllib.loads(sdist.extractfile(f"{top}/pyproject.toml").read().decode())
        expect(failures, project["project"]["name"] == "sparkpipe" and project["project"]["version"] == release and project["project"]["scripts"] == {"sparkpipe": "sparkpipe_cli:main"}
               and project["tool"]["setuptools"]["py-modules"] == ["sparkpipe_cli"], f"the sdist's pyproject.toml: {project}")
        expect(failures, sdist.extractfile(f"{top}/sparkpipe_cli.py").read() == (ROOT / "tools" / "sparkpipe_cli.py").read_bytes(), "the sdist's module is not tools/sparkpipe_cli.py")
        again = Path(folder) / "again"
        with contextlib.redirect_stdout(io.StringIO()):
            build.main(["--out", str(again)])
        expect(failures, all((out / n).read_bytes() == (again / n).read_bytes() for n in names), "two builds of the same source differ")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            expect(failures, build.main(["--out", str(again), "--tag", "cli-v" + release]) == 0 and build.main(["--out", str(again), "--tag", "v0.0.0"]) == 1, "the release tag is not checked against VERSION")
        target = Path(folder) / "site"
        installed = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-index", "--no-deps", "--target", str(target),
                                    str(out / f"sparkpipe-{release}-py3-none-any.whl")], capture_output=True, text=True)
        ran = subprocess.run([str(target / "bin" / "sparkpipe"), "--version"], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(target))) if installed.returncode == 0 else installed
        expect(failures, ran.returncode == 0 and ran.stdout.strip() == f"sparkpipe {release}", f"pip install of the wheel does not give a sparkpipe command: {ran.stdout} {ran.stderr}")


def main():
    failures = []
    for check in (check_units, check_login, check_keys, check_status, check_book, check_setup_claude_code, check_setup_codex, check_setup_opencode, check_setup_schemas, check_setup_undo,
                  check_tokens, check_fetch, check_storage_sync, check_proxy, check_errors, check_package):
        try:
            check(failures)
        except Exception as error:
            failures.append(f"{check.__name__} stopped: {type(error).__name__}: {error}")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("sparkpipe CLI: login by email code to a revocable bearer session and one key per machine; keys, status, bookings by week and hour priced before confirmation, "
          "and the waitlist with terms; setup of Claude Code (settings and its MCP server), Codex (config.toml and the sparkpipe profile file) and OpenCode merged into existing settings with a backup, idempotent, "
          "matching each tool's documented keys, and undone key by key, keeping settings written since; "
          "refusing files it cannot merge safely; tokens only in the macOS keychain or secret-tool, never on a command line; fetch, ls, resume and cancel sending a token only to its own host "
          "and following no redirect; storage sync by pages of changes, resuming from the last whole page, skipping files it already has, confined to the destination, to a directory or s3://; "
          "the proxy taking only its own local key, local host names and no web pages, forwarding only inference paths and the Responses WebSocket, retrying 429 and streaming; "
          "errors without tracebacks or secrets; a reproducible sdist and wheel that pip installs as the sparkpipe command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
