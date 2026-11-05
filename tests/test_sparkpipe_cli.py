#!/usr/bin/env python3
import base64
import contextlib
import hashlib
import http.client
import http.server
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import tomllib
import urllib.parse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import sparkpipe_cli as cli
import sparkpipe_cli_build as build

KEY = "sk-hc-machine"
SESSION = "SESSIONTOKEN"
SECRET = "hf_SECRETtoken0123456789"
CLEARED = ("SPARKPIPE_API_KEY", "SPARKPIPE_API", "SPARKPIPE_FLEET", "SPARKPIPE_CONFIG", "ANTHROPIC_API_KEY", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME")
MODELS = {"data": [
    {"id": "kimi-k3+core", "context_window": 1000000, "max_output_tokens": 32768, "sequences": 8},
    {"id": "glm-5.3-flash", "context_window": 200000, "max_output_tokens": 16384, "sequences": 16},
]}
ALIASES = {"claude-haiku-*": "glm-5.3-flash", "claude-*": "kimi-k3+core", "gpt-*": "kimi-k3+core"}

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
with open(os.path.join(state, "aws.log"), "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\n")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
bucket = os.path.join(state, "bucket")
if args[:2] == ["s3", "cp"]:
    target = os.path.join(bucket, args[3][len("s3://"):])
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(args[2], target)
elif args[:2] == ["s3", "rm"]:
    os.unlink(os.path.join(bucket, args[2][len("s3://"):]))
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
        if isinstance(payload, list):
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for piece in payload:
                data = piece() if callable(piece) else piece
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            return
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
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


def scenario(check):
    def run(failures):
        with tempfile.TemporaryDirectory() as folder:
            fake = Fake()
            try:
                check(failures, Home(folder), fake)
            finally:
                fake.close()
    return run


def expect(failures, condition, text):
    if not condition:
        failures.append(text)


def session_auth(request):
    return request["headers"].get("authorization") == "Bearer " + SESSION


def key_auth(request):
    return request["headers"].get("authorization") == "Bearer " + KEY


def check_units(failures):
    expect(failures, cli.api_base("https://sparkpipe.ai/pay/api/") == cli.API, "the API address keeps a trailing slash")
    for bad in ("http://sparkpipe.ai/pay/api", "ftp://x", "sparkpipe.ai"):
        try:
            cli.api_base(bad)
            failures.append(f"the API address {bad} is accepted")
        except cli.Failure:
            pass
    expect(failures, cli.api_base("http://127.0.0.1:9") == "http://127.0.0.1:9", "a loopback http API address is refused")
    for text, slot in (("2027-01-05T10", "2027-01-05T10:00Z"), ("2027-01-05T10:00", "2027-01-05T10:00Z"), ("2027-01-05T10:00Z", "2027-01-05T10:00Z")):
        expect(failures, cli.slot(text) == slot, f"the hour {text} is not {slot}")
    for bad in ("2027-13-05T10", "2027-01-05T10:30", "2027-01-05", "tomorrow"):
        try:
            cli.slot(bad)
            failures.append(f"the hour {bad} is accepted")
        except cli.Failure:
            pass
    expect(failures, cli.session_cookie(["other=1; Path=/", "sp_session=abc.def; Max-Age=2592000; Path=/; HttpOnly"]) == "abc.def", "the session cookie is not read")
    expect(failures, cli.session_cookie(["sp_session=; Max-Age=0"]) is None, "an empty session cookie is taken")
    hosts = {"hf://org/model@main": "huggingface.co", "s3://bucket/key": "s3.amazonaws.com", "https://Data.Example.com/x": "data.example.com", "git+https://git.example.com/r.git": "git.example.com"}
    for source, host in hosts.items():
        expect(failures, cli.source_host(source) == host, f"the host of {source} is not {host}")
    expect(failures, cli.host_matches("cdn-lfs.huggingface.co", "huggingface.co") and not cli.host_matches("huggingface.co.evil.example", "huggingface.co")
           and not cli.host_matches("evilhuggingface.co", "huggingface.co"), "a token's host is matched by suffix instead of by domain")
    expect(failures, cli.remote_prefix("/models/") == "models/" and cli.remote_prefix("") == "", "storage prefixes are not normalised")
    for path in ("models/../x", "models/./x", "models//x", "models/sub\\..\\x", "models/"):
        try:
            cli.sync_parts("models/", path)
            failures.append(f"the unsafe storage path {path!r} is accepted")
        except cli.Failure:
            pass
    env = cli.claude_env(cli.API, KEY, "kimi-k3+core", "kimi-k3+core", 1000000)
    spec = {"ANTHROPIC_BASE_URL": "https://sparkpipe.ai/pay/api", "ANTHROPIC_AUTH_TOKEN": KEY, "ANTHROPIC_MODEL": "kimi-k3+core", "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-k3+core",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1000000", "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_SUBAGENT_MODEL": "kimi-k3+core", "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1"}
    expect(failures, env == spec, f"the Claude Code environment differs from FLEET_API.md 3.3 and 3.8: {env}")
    provider = dict(cli.codex_tables(cli.API, "kimi-k3+core", {"context_window": 1000000, "max_output_tokens": 32768, "sequences": 8})[0][1])
    expect(failures, provider == {"name": "SparkPipe", "base_url": "https://sparkpipe.ai/pay/api/v1", "env_key": "SPARKPIPE_API_KEY", "wire_api": "responses", "stream_idle_timeout_ms": 600000},
           f"the Codex provider differs from FLEET_API.md 3.4: {provider}")


@scenario
def check_login(failures, home, fake):
    keys = []

    def verify(request):
        body = request["body"]
        if body.get("code") != "123456":
            return 403, {}, {"error": "That code is wrong or has expired. Ask for a new one."}
        if body["email"] == "new@acme.example" and not (body.get("company") and body.get("accept") is True):
            return 404, {}, {"error": "No account uses this email yet.", "reason": "new"}
        return 200, {"Set-Cookie": f"sp_session={SESSION}; Max-Age=2592000; Path=/; HttpOnly; SameSite=Lax"}, {"email": body["email"], "company": "Acme"}

    def state(request):
        if not session_auth(request):
            return 200, {}, {"account": None}
        return 200, {}, {"account": {"email": "dev@acme.example", "keys": [dict(k) for k in keys]}}

    def create(request):
        ident = f"KY{len(keys) + 1}"
        keys.append({"key": ident, "note": request["body"]["note"], "hint": "abcd", "created": "2026-11-04T00:00:00Z", "revoked": None})
        return 200, {}, {"id": ident, "key": f"sk-hc-{ident}"}

    fake.route("POST", "/code", (200, {}, {"sent": True}))
    fake.route("POST", "/verify", verify)
    fake.route("GET", "/market/state", state)
    fake.route("POST", "/market/keys", create)
    code, out, err = home.run("login", "--api", fake.url, "--email", "dev@acme.example", answers=["123456"])
    expect(failures, code == 0 and "Created key KY1" in out, f"login does not create a key: {code} {out} {err}")
    sent = fake.find("POST", "/code") + fake.find("POST", "/verify")
    expect(failures, [r["body"] for r in sent] == [{"email": "dev@acme.example"}, {"email": "dev@acme.example", "code": "123456"}] and not any("authorization" in r["headers"] for r in sent),
           f"login sends the wrong sign-in requests: {[r['body'] for r in sent]}")
    made = fake.find("POST", "/market/keys")
    expect(failures, len(made) == 1 and session_auth(made[0]) and made[0]["body"]["note"].startswith("sparkpipe CLI on "), "the key is not created with the bearer session and a machine note")
    expect(failures, home.cfg() == {"api": fake.url, "email": "dev@acme.example", "session": SESSION, "key": "sk-hc-KY1", "key_id": "KY1", "tokens": {}, "sync": {}}, f"the saved config: {home.cfg()}")
    expect(failures, stat.S_IMODE(home.config.stat().st_mode) == 0o600, "the config holding the key and session is readable by others")
    code, out, _ = home.run("login", "--api", fake.url, "--email", "dev@acme.example", answers=["123456"])
    expect(failures, code == 0 and "Kept key KY1" in out and len(fake.find("POST", "/market/keys")) == 1, f"a second login creates another key: {out}")
    keys[0]["revoked"] = "2026-11-04T01:00:00Z"
    code, out, _ = home.run("login", "--email", "dev@acme.example", answers=["123456"])
    expect(failures, code == 0 and "Created key KY2" in out and home.cfg()["key"] == "sk-hc-KY2", f"a login after the key was revoked keeps the revoked key: {out}")
    code, _, err = home.run("login", "--email", "dev@acme.example", answers=["654321"])
    expect(failures, code == 1 and "That code is wrong" in err and home.cfg()["key"] == "sk-hc-KY2", f"a wrong code: {code} {err}")
    code, _, err = home.run("login", "--email", "new@acme.example", answers=["123456"])
    expect(failures, code == 1 and "--company" in err and "--accept" in err, f"a new email is not told how to open an account: {err}")
    code, out, _ = home.run("login", "--email", "new@acme.example", "--company", "New Co", "--accept", answers=["123456"])
    opened = fake.find("POST", "/verify")[-1]["body"]
    expect(failures, code == 0 and opened == {"email": "new@acme.example", "code": "123456", "company": "New Co", "accept": True}, f"opening an account: {opened}")
    fake.route("POST", "/code", (200, {}, {"sent": False, "code": "123456"}))
    code, out, err = home.run("login", "--email", "dev@acme.example")
    expect(failures, code == 0 and "sandbox" in out, f"the sandbox's code is not used: {code} {out} {err}")
    fake.route("POST", "/verify", (200, {}, {"email": "dev@acme.example"}))
    code, _, err = home.run("login", "--email", "dev@acme.example")
    expect(failures, code == 1 and "without a session" in err, "a sign-in without a session cookie is accepted")


@scenario
def check_keys(failures, home, fake):
    home.signed_in(fake.url)
    listed = [{"key": "KY0", "note": "laptop", "hint": "zzzz", "created": "2026-10-01T00:00:00Z", "revoked": "2026-10-02T00:00:00Z"},
              {"key": "KY1", "note": "sparkpipe CLI on box", "hint": "abcd", "created": "2026-11-01T00:00:00Z", "revoked": None}]
    fake.route("GET", "/market/state", (200, {}, {"account": {"keys": listed}}))
    fake.route("POST", "/market/keys", (200, {}, {"id": "KY2", "key": "sk-hc-new"}))
    fake.route("POST", "/market/keys/revoke", lambda r: (200, {}, {"id": r["body"]["id"], "revoked": True}))
    code, out, _ = home.run("keys", "ls")
    lines = out.splitlines()
    expect(failures, code == 0 and len(lines) == 2 and "revoked" in lines[0] and "active" in lines[1] and "this machine" in lines[1], f"keys ls: {out}")
    code, out, _ = home.run("keys", "ls", "--json")
    expect(failures, code == 0 and json.loads(out) == listed, "keys ls --json does not print the keys")
    code, out, _ = home.run("keys", "create", "--note", "ci", "--fleet", "F3", "--allow-swap")
    made = fake.find("POST", "/market/keys")[-1]
    expect(failures, code == 0 and made["body"] == {"note": "ci", "fleet": "F3", "allow_swap": True} and session_auth(made) and "sk-hc-new" in out and home.cfg()["key"] == KEY,
           f"keys create: {made['body']} {out}")
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
    fake.route("POST", "/market/quote-weeks", (200, {}, plan))
    fake.route("POST", "/market/book-weeks", booked)
    fake.route("POST", "/market/book-hours", lambda r: (200, {}, dict(r["body"], booking="BK8", end="2027-01-05T12:00Z", hc="6.000000")))
    fake.route("POST", "/market/waitlist", (200, {}, {"locked": "144.000000", "position": 3}))
    fake.route("POST", "/market/waitlist/leave", (200, {}, {"locked": "0.000000", "position": None}))
    code, out, _ = home.run("book", "week", "F3", "4", answers=["y"])
    books = fake.find("POST", "/market/book-weeks")
    expect(failures, code == 0 and "BK9" in out and "540.000000 HC" in "".join(home.prompts[-1:]) and [r["body"] for r in books] == [{"fleet": "F3", "weeks": 4}] and session_auth(books[0])
           and fake.find("POST", "/market/quote-weeks")[0]["body"] == {"fleet": "F3", "weeks": 4}, f"book week: {code} {out} {home.prompts}")
    code, _, err = home.run("book", "week", "F3", "4", answers=["n"])
    expect(failures, code == 1 and len(fake.find("POST", "/market/book-weeks")) == 1, "a declined booking is booked")
    code, _, err = home.run("book", "week", "F3", "4", answers=[EOFError()])
    expect(failures, code == 1 and "--yes" in err and len(fake.find("POST", "/market/book-weeks")) == 1, "a booking without a terminal does not ask for --yes")
    fake.terms = True
    code, _, err = home.run("book", "week", "F3", "4", "--yes")
    expect(failures, code == 1 and "--accept" in err, f"a booking under changed terms: {err}")
    code, out, _ = home.run("book", "week", "F3", "4", "--yes", "--accept")
    expect(failures, code == 0 and fake.find("POST", "/market/book-weeks")[-1]["body"] == {"fleet": "F3", "weeks": 4, "accept": True}, "--accept is not sent with the booking")
    for start in ("2027-01-05T10", "2027-01-05T10:00Z"):
        code, out, _ = home.run("book", "hours", "F1", start, "2", "--yes")
        expect(failures, code == 0 and fake.find("POST", "/market/book-hours")[-1]["body"] == {"fleet": "F1", "start": "2027-01-05T10:00Z", "hours": 2} and "BK8" in out, f"book hours {start}: {out}")
    count = len(fake.find("POST", "/market/book-hours"))
    code, _, err = home.run("book", "hours", "F1", "2027-01-05T10:30", "2", "--yes")
    expect(failures, code == 1 and len(fake.find("POST", "/market/book-hours")) == count, "a booking at a half hour is sent")
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
    code, out, err = home.run("setup", "claude-code")
    after = json.loads(path.read_text())
    env = {"ANTHROPIC_BASE_URL": fake.url, "ANTHROPIC_AUTH_TOKEN": KEY, "ANTHROPIC_MODEL": "kimi-k3+core", "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-5.3-flash",
           "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1000000", "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
           "CLAUDE_CODE_SUBAGENT_MODEL": "kimi-k3+core", "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1"}
    overrides = {"claude-opus-4-6": "arn:x", "claude-opus-5-5": "kimi-k3+core", "claude-sonnet-5-5": "kimi-k3+core", "claude-fable-5-1": "kimi-k3+core", "claude-haiku-5-5": "glm-5.3-flash"}
    expect(failures, code == 0 and after == {"permissions": {"allow": ["Bash(ls)"]}, "env": dict({"FOO": "bar"}, **env), "modelOverrides": overrides, "theme": "dark"},
           f"setup claude-code writes {after} ({code} {err})")
    expect(failures, stat.S_IMODE(path.stat().st_mode) == 0o600, "the Claude Code settings holding the key are readable by others")
    text = path.read_text()
    home.run("setup", "claude-code")
    expect(failures, path.read_text() == text, "setup claude-code is not idempotent")
    fake.route("GET", "/v1/models/aliases", (200, {}, {"claude-*": "kimi-k3+core"}))
    elsewhere = home.root / "claude-config"
    code, out, _ = home.run("setup", "claude-code", "--model", "glm-5.3-flash", env={"CLAUDE_CONFIG_DIR": str(elsewhere), "ANTHROPIC_API_KEY": "sk-ant-x"})
    other = json.loads((elsewhere / "settings.json").read_text())["env"]
    expect(failures, code == 0 and other["ANTHROPIC_MODEL"] == other["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == other["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-5.3-flash"
           and other["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "200000" and "ANTHROPIC_API_KEY is also set" in out, f"setup claude-code --model and CLAUDE_CONFIG_DIR: {other} {out}")
    real = home.root / "dotfiles" / "settings.json"
    real.parent.mkdir()
    real.write_text("{}")
    link = home.root / "linked.json"
    link.symlink_to(real)
    code, _, _ = home.run("setup", "claude-code", "--settings", link)
    expect(failures, code == 0 and link.is_symlink() and json.loads(real.read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == KEY, "setup claude-code replaces a symlinked settings file")
    path.write_text('{"env": {"FOO": "bar"},')
    code, _, err = home.run("setup", "claude-code")
    expect(failures, code == 1 and path.read_text() == '{"env": {"FOO": "bar"},' and "not valid JSON" in err, "setup claude-code overwrites settings it cannot read")
    path.write_text('{"env": ["x"]}')
    code, _, _ = home.run("setup", "claude-code")
    expect(failures, code == 1 and path.read_text() == '{"env": ["x"]}', "setup claude-code replaces an env that is not an object")
    fake.route("GET", "/v1/models/aliases", (200, {}, {}))
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
stream_idle_timeout_ms = 600000"""
    expect(failures, code == 0 and text.startswith('model = "kimi-k3+core"\nmodel_provider = "sparkpipe"\n') and provider in text and "# my codex config" in text, f"setup codex writes:\n{text}\n{err}")
    wanted = {
        "model": "kimi-k3+core", "model_provider": "sparkpipe", "approval_policy": "on-request",
        "model_providers": {"other": {"name": "Other", "base_url": "https://other.example/v1"},
                            "sparkpipe": {"name": "SparkPipe", "base_url": fake.url + "/v1", "env_key": "SPARKPIPE_API_KEY", "wire_api": "responses", "stream_idle_timeout_ms": 600000}},
        "profiles": {"fast": {"model": "o4-mini"},
                     "sparkpipe": {"model": "kimi-k3+core", "model_provider": "sparkpipe", "model_context_window": 1000000, "model_auto_compact_token_limit": 967232,
                                   "model_catalog_url": fake.url + "/v1/codex/models", "agents": {"default_subagent_model": "kimi-k3+core", "max_concurrent_threads_per_session": 8}}},
        "mcp_servers": {"github": {"command": "gh-mcp", "args": ["serve"]}, "sparkpipe": {"url": fake.url + "/mcp", "bearer_token_env_var": "SPARKPIPE_API_KEY"}},
    }
    expect(failures, doc == wanted, f"setup codex parses to {doc}")
    expect(failures, KEY not in text and f"export SPARKPIPE_API_KEY={KEY}" in out, "setup codex writes the key into the config, or does not say where Codex reads it")
    home.run("setup", "codex")
    expect(failures, path.read_text() == text, "setup codex is not idempotent")
    code, out, _ = home.run("setup", "codex", env={"SPARKPIPE_API_KEY": KEY})
    expect(failures, code == 0 and "export" not in out, "setup codex repeats the key reminder when SPARKPIPE_API_KEY is set")
    fresh = home.root / "codex-home"
    code, _, _ = home.run("setup", "codex", env={"CODEX_HOME": str(fresh)})
    expect(failures, code == 0 and tomllib.loads((fresh / "config.toml").read_text())["profiles"]["sparkpipe"]["model"] == "kimi-k3+core", "setup codex into an empty CODEX_HOME")
    for broken in ('model = "o4"\nmodel = \n', 'profiles.sparkpipe.model = "x"\n'):
        path.write_text(broken)
        code, _, err = home.run("setup", "codex")
        expect(failures, code == 1 and path.read_text() == broken and "nothing was changed" in err.lower(), f"setup codex over {broken!r}: {code} {err}")
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
        "agent": {"general": {"temperature": 0.2, "model": main}, "reviewer": {"model": "anthropic/claude-sonnet-5-5"}, "explore": {"model": main}},
        "mcp": {"jira": {"type": "remote", "url": "https://jira.example/mcp"},
                "sparkpipe": {"type": "remote", "url": fake.url + "/mcp", "headers": {"Authorization": "Bearer {env:SPARKPIPE_API_KEY}"}, "enabled": True}},
        "compaction": {"auto": True, "prune": False},
        "model": main, "small_model": small, "share": "disabled",
    }
    expect(failures, code == 0 and after == wanted, f"setup opencode writes {after} ({err})")
    expect(failures, KEY not in path.read_text() and "SPARKPIPE_API_KEY" in out, "setup opencode writes the key into the config")
    text = path.read_text()
    home.run("setup", "opencode")
    expect(failures, path.read_text() == text, "setup opencode is not idempotent")
    project = home.root / "repo" / "opencode.json"
    code, _, _ = home.run("setup", "opencode", "--config", project, "--haiku", "kimi-k3+core")
    written = json.loads(project.read_text())
    expect(failures, code == 0 and written["small_model"] == main and list(written["provider"]["sparkpipe"]["models"]) == ["kimi-k3+core"], f"setup opencode --config --haiku: {written}")
    path.write_text('{"mcp": "none"}')
    code, _, _ = home.run("setup", "opencode")
    expect(failures, code == 1 and path.read_text() == '{"mcp": "none"}', "setup opencode replaces an mcp that is not an object")


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
    code, _, _ = home.run("tokens", "set", "empty", "--host", "x.example", secrets=["  "])
    expect(failures, code == 1 and "empty" not in home.cfg()["tokens"], f"{platform}: an empty token is stored")
    code, out, _ = home.run("tokens", "ls")
    expect(failures, code == 0 and out.splitlines() == ["hf  huggingface.co", "r2  acct.r2.cloudflarestorage.com"] and SECRET not in out, f"{platform}: tokens ls: {out}")
    store = home.keychain()
    del store["sparkpipe/r2"]
    (home.state / "keychain.json").write_text(json.dumps(store))
    code, out, _ = home.run("tokens", "ls", "--json")
    expect(failures, code == 0 and json.loads(out) == [{"name": "hf", "host": "huggingface.co", "stored": True}, {"name": "r2", "host": "acct.r2.cloudflarestorage.com", "stored": False}],
           f"{platform}: tokens ls does not notice a token missing from the keychain: {out}")
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
    expect(failures, code == 0 and "fx_1  needs_token  hf://org/model@rev to models/x/  500.0 GB of 1.5 TB" in out, f"fetch ls: {out}")
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
    expect(failures, {p.relative_to(dest).as_posix(): p.read_bytes() for p in dest.rglob("*") if p.is_file()} == {k[len("models/"):]: v for k, v in first.items()},
           f"storage sync copies the wrong files: {sorted(p.name for p in dest.rglob('*'))}")
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
    fake.route("GET", "/v1/storage/changes", (200, {}, {"data": [dict(change("models/a.bin", first["models/a.bin"]), sha256="0" * 64)], "until": "z", "more": False}))
    target = home.root / "bad-sha"
    code, _, err = home.run("storage", "sync", "models/", target)
    expect(failures, code == 1 and "SHA-256" in err and not (target / "a.bin").exists() and not list(target.glob(".sparkpipe-*")), f"a file with the wrong SHA-256 is kept: {err}")
    fake.route("GET", "/v1/storage/changes", (200, {}, {"data": [], "until": None, "more": True}))
    code, _, err = home.run("storage", "sync", "models/", home.root / "loop")
    expect(failures, code == 1 and "did not advance" in err, "a change list that never advances loops")
    fake.route("GET", "/v1/storage/changes", lambda r: (200, {}, pages[r["query"].get("since", [None])[0]]))
    fake.route("GET", "/v1/storage/models/sub/b.json", (200, {}, first["models/sub/b.json"]))
    code, out, err = home.run("storage", "sync", "models/", "s3://bucket/backup")
    bucket = home.state / "bucket" / "bucket" / "backup"
    expect(failures, code == 0 and {p.relative_to(bucket).as_posix(): p.read_bytes() for p in bucket.rglob("*") if p.is_file()} == {k[len("models/"):]: v for k, v in first.items()},
           f"storage sync to s3:// copies the wrong files: {out} {err}")
    code, _, _ = home.run("storage", "sync", "models/", "gs://bucket/x")
    expect(failures, code == 1, "storage sync accepts a gs:// destination it cannot write")


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

    fake.route("POST", "/pay/api/v1/messages", messages)
    fake.route("POST", "/pay/api/v1/chat/completions", (200, {"Content-Type": "text/event-stream"}, [b"data: first\n\n", later, b"data: [DONE]\n\n"]))
    fake.route("GET", "/pay/api/v1/models", (429, {"retry-after": "120"}, {"type": "error", "error": {"type": "rate_limit_error", "message": "busy"}}))
    return attempts


def proxy_request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(method, path, body=body, headers=headers or {})
    return connection, connection.getresponse()


@scenario
def check_proxy(failures, home, fake):
    released = {"event": threading.Event(), "streamed": None}
    attempts = proxy_upstream(fake, released)
    server = cli.proxy_server(fake.url + "/pay/api", KEY, "F3", 0, 3, 5, log=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        body = json.dumps({"model": "kimi-k3+core", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}).encode()
        connection, response = proxy_request(port, "POST", "/v1/messages", body, {"x-api-key": "anything", "Authorization": "Bearer anything", "X-SparkPipe-Fleet": "F9", "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        answer = json.loads(response.read())
        connection.close()
        expect(failures, response.status == 200 and answer["id"] == "msg_1" and len(attempts) == 2, f"the proxy does not retry a 429 with retry-after: {response.status} {len(attempts)}")
        for sent in attempts:
            headers = sent["headers"]
            expect(failures, headers.get("authorization") == "Bearer " + KEY and "x-api-key" not in headers and headers.get("x-sparkpipe-fleet") == "F3" and sent["raw"] == body
                   and headers.get("anthropic-version") == "2023-06-01", f"the proxy forwards {headers}")
        connection, response = proxy_request(port, "POST", "/v1/chat/completions", b'{"stream": true}', {"Content-Type": "application/json"})
        first = response.read1(65536)
        released["event"].set()
        rest = response.read()
        connection.close()
        expect(failures, response.getheader("Transfer-Encoding") == "chunked" and first == b"data: first\n\n" and rest == b"data: second\n\ndata: [DONE]\n\n" and released["streamed"],
               f"the proxy does not stream: {first!r} {rest!r} {released['streamed']}")
        connection, response = proxy_request(port, "GET", "/v1/models")
        response.read()
        connection.close()
        expect(failures, response.status == 429 and response.getheader("retry-after") == "120" and len(fake.find("GET", "/pay/api/v1/models")) == 1, "the proxy waits out a retry-after beyond --max-wait")
        connection, response = proxy_request(port, "POST", "/v1/messages", b"x" * 10, {"Transfer-Encoding": "chunked"})
        response.read()
        connection.close()
        expect(failures, response.status == 411, "the proxy accepts a request without a Content-Length")
    finally:
        server.shutdown()
        server.server_close()
    dead = cli.proxy_server("http://127.0.0.1:9", KEY, None, 0, 0, 5, log=False)
    threading.Thread(target=dead.serve_forever, daemon=True).start()
    try:
        connection, response = proxy_request(dead.server_address[1], "GET", "/v1/models")
        answer = json.loads(response.read())
        connection.close()
        expect(failures, response.status == 502 and answer["error"]["type"] == "upstream_error", "an unreachable API does not answer 502")
    finally:
        dead.shutdown()
        dead.server_close()
    home.config.parent.mkdir(parents=True, exist_ok=True)
    home.config.write_text(json.dumps({"api": fake.url}))
    code, _, err = home.run("proxy", "--port", "0")
    expect(failures, code == 1 and "API key" in err, "the proxy starts without a key")


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
    for check in (check_units, check_login, check_keys, check_status, check_book, check_setup_claude_code, check_setup_codex, check_setup_opencode,
                  check_tokens, check_fetch, check_storage_sync, check_proxy, check_package):
        check(failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("sparkpipe CLI: login by email code to a bearer session and one key per machine; keys, status, bookings by week and hour and the waitlist with confirmation and terms; "
          "setup of Claude Code, Codex and OpenCode merged into existing settings, idempotent, refusing files it cannot merge; tokens only in the macOS keychain or secret-tool, never on a command line; "
          "fetch, ls, resume and cancel sending a token only to its own host and following no redirect; storage sync by pages of changes, checked, confined to the destination, to a directory or s3://; "
          "the proxy adding the key and fleet, retrying 429 with retry-after and streaming; a reproducible sdist and wheel that pip installs as the sparkpipe command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
