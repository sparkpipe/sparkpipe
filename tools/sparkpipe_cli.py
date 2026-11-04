#!/usr/bin/env python3
import argparse
import contextlib
import getpass
import hashlib
import http.client
import http.server
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:
    import tomllib
except ImportError:
    tomllib = None

VERSION = "0.1.0"
API = "https://sparkpipe.ai/pay/api"
KEYCHAIN_SERVICE = "sparkpipe"
SESSION_COOKIE = "sp_session"
TIMEOUT = 60
UPSTREAM_TIMEOUT = 900
BODY_LIMIT = 32 * 1024 * 1024
CHUNK = 65536
PROXY_PORT = 8790
PLATFORM = sys.platform
TOKEN_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
HOST = re.compile(r"^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$")
SLOT = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2})(?::00)?Z?$")
SOURCES = ("hf://", "s3://", "https://", "git+https://")
SOURCE_HOSTS = {"hf://": "huggingface.co", "s3://": "s3.amazonaws.com"}
TOKEN_HOSTS = {"hf": "huggingface.co"}
LOOPBACK = {"localhost", "127.0.0.1", "::1"}
INSTALL = {"secret-tool": " (Debian and Ubuntu: apt install libsecret-tools)", "aws": " (https://aws.amazon.com/cli/)"}
HF_GUIDANCE = "Use a fine-grained, read-only Hugging Face token, limited to the repositories you need and with an expiry: https://huggingface.co/settings/tokens"
CLAUDE_MODELS = {"claude-opus-5-5": "main", "claude-sonnet-5-5": "main", "claude-fable-5-1": "main", "claude-haiku-5-5": "haiku"}
CODEX_ROOT = ("model", "model_provider")
CODEX_TABLES = (("model_providers", "sparkpipe"), ("profiles", "sparkpipe"), ("mcp_servers", "sparkpipe"))
OPENCODE_SUBAGENTS = ("general", "explore")
MODEL_FIELDS = ("context_window", "max_output_tokens", "sequences")
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te", "trailer", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}
CREDENTIALS = {"authorization", "x-api-key"}
HEADER = re.compile(r"^\s*\[\[?([^\[\]]+)\]\]?\s*(?:#.*)?$")
KEY = re.compile(r"""^\s*(?:"((?:[^"\\]|\\.)*)"|'([^']*)'|([A-Za-z0-9_-]+))\s*=""")
PART = re.compile(r"""\s*(?:"((?:[^"\\]|\\.)*)"|'([^']*)'|([A-Za-z0-9_-]+))\s*(\.|$)""")


class Failure(Exception):
    pass


class ApiError(Failure):
    def __init__(self, status, message, reason=None):
        super().__init__(message)
        self.status = status
        self.reason = reason


def prompt(text):
    return input(text)


def prompt_secret(text):
    return getpass.getpass(text)


def config_path():
    named = os.environ.get("SPARKPIPE_CONFIG")
    if named:
        return Path(named)
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "sparkpipe" / "config.json"


def read_json(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError as error:
        raise Failure(f"{path} is not valid JSON ({error}); fix it and run this again. Nothing was changed.")
    if not isinstance(data, dict):
        raise Failure(f"{path} does not hold a JSON object; fix it and run this again. Nothing was changed.")
    return data


def json_text(data):
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def file_mode(path, default):
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        return default


def write_file(path, text, mode):
    path = Path(os.path.realpath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        os.chmod(temporary, mode)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def api_base(text):
    text = text.rstrip("/")
    parts = urllib.parse.urlsplit(text)
    if parts.scheme == "https" and parts.hostname:
        return text
    if parts.scheme == "http" and parts.hostname in LOOPBACK:
        return text
    raise Failure(f"{text} is not an https:// API address.")


def parse_json(raw):
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        raise Failure("The API answered with something other than JSON.")


def error_fields(data):
    found = data.get("error") if isinstance(data, dict) else None
    if isinstance(found, dict):
        return found.get("message"), found.get("code") or found.get("type")
    if isinstance(found, str):
        return found, data.get("reason")
    return None, None


def refusal(error):
    try:
        data = json.loads(error.read() or b"{}")
    except ValueError:
        data = {}
    message, reason = error_fields(data)
    return ApiError(error.code, message or f"The API answered {error.code} {error.reason}.", reason)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Failure(f"The API redirected {req.full_url} to {newurl}. sparkpipe follows no redirects, so keys and tokens go to no other address.")


class Client:
    def __init__(self, api, session=None, key=None, fleet=None):
        self.api, self.session, self.key, self.fleet = api, session, key, fleet
        handlers = [NoRedirect()]
        if urllib.parse.urlsplit(api).hostname in LOOPBACK:
            handlers.append(urllib.request.ProxyHandler({}))
        self.opener = urllib.request.build_opener(*handlers)

    def credentials(self, auth):
        if auth is None:
            return {}
        if auth == "session":
            if not self.session:
                raise Failure("Sign in first: sparkpipe login")
            return {"Authorization": "Bearer " + self.session}
        if not self.key:
            raise Failure("This machine has no API key: run sparkpipe login, or set SPARKPIPE_API_KEY.")
        found = {"Authorization": "Bearer " + self.key}
        if self.fleet:
            found["X-SparkPipe-Fleet"] = self.fleet
        return found

    def open(self, method, path, body=None, auth="key", headers=None, query=None):
        url = self.api + path + ("?" + urllib.parse.urlencode(query) if query else "")
        sent = {"Accept": "application/json", "User-Agent": "sparkpipe-cli/" + VERSION}
        sent.update(self.credentials(auth))
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            sent["Content-Type"] = "application/json"
        sent.update(headers or {})
        request = urllib.request.Request(url, data=data, method=method, headers=sent)
        try:
            return self.opener.open(request, timeout=TIMEOUT)
        except urllib.error.HTTPError as error:
            raise refusal(error)
        except (urllib.error.URLError, OSError) as error:
            raise Failure(f"Cannot reach {self.api}: {getattr(error, 'reason', error)}")

    def call(self, method, path, body=None, auth="key", headers=None, query=None):
        with self.open(method, path, body, auth, headers, query) as response:
            return parse_json(response.read())


class Context:
    def __init__(self, args):
        self.args = args
        self.cfg = read_json(config_path())
        self.api = api_base(getattr(args, "api", None) or os.environ.get("SPARKPIPE_API") or self.cfg.get("api") or API)

    def mine(self):
        return self.cfg if self.cfg.get("api") == self.api else {}

    def client(self):
        key = os.environ.get("SPARKPIPE_API_KEY") or self.mine().get("key")
        fleet = getattr(self.args, "fleet", None) or os.environ.get("SPARKPIPE_FLEET")
        return Client(self.api, self.mine().get("session"), key, fleet)

    def save(self):
        write_file(config_path(), json_text(self.cfg), 0o600)


def emit(ctx, data, lines):
    if getattr(ctx.args, "json", False):
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    for line in lines:
        print(line)


def size(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000


def confirm(ctx, question):
    if ctx.args.yes:
        return
    try:
        answer = prompt(question + " [y/N] ")
    except EOFError:
        raise Failure("Add --yes to confirm without a terminal.")
    if answer.strip().lower() not in ("y", "yes"):
        raise Failure("Stopped; nothing was changed.")


def account_state(client):
    account = client.call("GET", "/market/state", auth="session").get("account")
    if not account:
        raise Failure("The session has expired: run sparkpipe login.")
    return account


def session_cookie(values):
    for value in values:
        name, _, rest = value.partition("=")
        token = rest.split(";", 1)[0].strip()
        if name.strip() == SESSION_COOKIE and token:
            return token
    return None


def verify(client, email, code, args):
    body = {"email": email, "code": code}
    if args.company:
        body["company"] = args.company
    if args.accept:
        body["accept"] = True
    try:
        with client.open("POST", "/verify", body, auth=None) as response:
            summary = parse_json(response.read())
            session = session_cookie(response.headers.get_all("Set-Cookie") or [])
    except ApiError as error:
        if error.reason == "new":
            raise Failure(f"No account uses {email} yet. To open one, read the Customer Agreement on sparkpipe.ai and run: sparkpipe login --email {email} --company 'Your legal name' --accept")
        if error.reason == "terms":
            raise Failure("Opening an account needs the Customer Agreement accepted: read it on sparkpipe.ai and add --accept.")
        raise
    if not session:
        raise Failure("The sign-in answered without a session.")
    return session, summary


def machine_key(client, previous):
    keys = account_state(client).get("keys", [])
    active = {k.get("key") for k in keys if not k.get("revoked")}
    if previous.get("key") and previous.get("key_id") in active:
        return previous["key"], previous["key_id"], False
    made = client.call("POST", "/market/keys", {"note": "sparkpipe CLI on " + socket.gethostname()[:60]}, auth="session")
    return made["key"], made["id"], True


def login(ctx):
    args = ctx.args
    client = Client(ctx.api)
    email = (args.email or prompt("Email: ")).strip()
    code = client.call("POST", "/code", {"email": email}, auth=None).get("code")
    if code:
        print("The sandbox answered with the code; using it.")
    else:
        code = prompt(f"Code from the email sent to {email}: ").strip()
    session, summary = verify(client, email, code, args)
    previous = ctx.mine() if ctx.mine().get("email") == email else {}
    client.session = session
    key, ident, made = machine_key(client, previous)
    ctx.cfg = {"api": ctx.api, "email": email, "session": session, "key": key, "key_id": ident, "tokens": ctx.cfg.get("tokens", {}), "sync": ctx.cfg.get("sync", {})}
    ctx.save()
    company = f" ({summary['company']})" if summary.get("company") else ""
    print(f"Signed in as {email}{company}. {'Created' if made else 'Kept'} key {ident} for this machine, saved in {config_path()}.")
    return 0


def account_lines(account):
    lines = [f"Account {account.get('email')} ({account.get('company')}): {account.get('balance')} HC, {account.get('locked')} HC locked"]
    bookings = account.get("bookings", [])
    held = sorted({b.get("fleet") for b in bookings if b.get("state") == "live"})
    lines.append("Holding now: " + (", ".join(held) if held else "no fleet"))
    for b in bookings:
        if b.get("state") in ("live", "upcoming"):
            lines.append(f"  {b.get('booking')}  {b.get('fleet')}  {b.get('kind')}  {b.get('start')} to {b.get('end')}  {b.get('state')}")
    for p in account.get("pinned", []):
        lines.append(f"Deadline: {p.get('fleet')} is held through {p.get('through')}; book the next week by {p.get('deadline')}")
    if account.get("position"):
        lines.append(f"Waitlist: position {account['position']}, {account.get('locked')} HC locked")
    if account.get("termsCurrent") is False:
        lines.append("The Customer Agreement has changed: read it on sparkpipe.ai and add --accept to your next booking.")
    return lines


def fleet_lines(state):
    if "error" in state:
        return [f"Fleet: {state['error']}"]
    lines = [("Fleet " + str(state.get("fleet") or "")).rstrip()]
    lines.append("  models: " + (", ".join(str(m.get("id")) for m in state.get("models", [])) or "none resident"))
    cache = state.get("cache") or {}
    if cache:
        lines.append(f"  cache: {size(cache.get('used_bytes'))} of {size(cache.get('capacity_bytes'))}")
    lines.append(f"  pins: {len(state.get('pins', []))}")
    packs = [f"{p.get('name')}@{p.get('version')}" + (" (warm)" if p.get("warm") else " (cold)") for p in state.get("packs", [])]
    lines.append("  packs: " + (", ".join(packs) or "none"))
    runs = [f"{r.get('id')} {r.get('function')} {r.get('state')}" for r in state.get("runs", [])]
    lines.append("  runs: " + (", ".join(runs) or "none"))
    if state.get("deletes_at"):
        lines.append(f"  working state deleted at {state['deletes_at']} if the account stops holding the fleet")
    return lines


def status(ctx):
    client = ctx.client()
    account = account_state(client)
    try:
        fleet = client.call("GET", "/v1/state")
    except Failure as error:
        fleet = {"error": str(error), "reason": getattr(error, "reason", None)}
    emit(ctx, {"account": account, "fleet": fleet}, account_lines(account) + fleet_lines(fleet))
    return 0


def spend(ctx, client, path, body):
    if ctx.args.accept:
        body = dict(body, accept=True)
    try:
        return client.call("POST", path, body, auth="session")
    except ApiError as error:
        if error.reason == "terms":
            raise Failure("The Customer Agreement has changed since you accepted it. Read it on sparkpipe.ai, then run this again with --accept.")
        raise


def slot(text):
    found = SLOT.match(text)
    try:
        datetime.strptime(found.group(1) + found.group(2), "%Y-%m-%d%H")
    except (AttributeError, ValueError):
        raise Failure(f"Give the start as a UTC hour such as 2027-01-05T10, not {text!r}.")
    return f"{found.group(1)}T{found.group(2)}:00Z"


def book_week(ctx):
    client = ctx.client()
    request = {"fleet": ctx.args.fleet_id, "weeks": ctx.args.weeks}
    plan = client.call("POST", "/market/quote-weeks", request, auth="session")
    confirm(ctx, f"Book {ctx.args.fleet_id} for {ctx.args.weeks} week(s), {plan.get('first')} to {plan.get('end')}, for {plan.get('price')} HC ({plan.get('discount')} off)?")
    booked = spend(ctx, client, "/market/book-weeks", request)
    emit(ctx, booked, [f"Booked {booked.get('booking')}: {booked.get('fleet')} from {booked.get('first')} to {booked.get('end')} for {booked.get('price')} HC."])
    return 0


def book_hours(ctx):
    client = ctx.client()
    request = {"fleet": ctx.args.fleet_id, "start": slot(ctx.args.start), "hours": ctx.args.hours}
    confirm(ctx, f"Book {request['fleet']} for {request['hours']} hour(s) from {request['start']}, paid from your HC?")
    booked = spend(ctx, client, "/market/book-hours", request)
    emit(ctx, booked, [f"Booked {booked.get('booking')}: {booked.get('fleet')} from {booked.get('start')} to {booked.get('end')} for {booked.get('hc')} HC."])
    return 0


def waitlist_lock(ctx):
    try:
        amount = Decimal(ctx.args.hc)
    except InvalidOperation:
        amount = Decimal(0)
    if not amount.is_finite() or amount <= 0:
        raise Failure(f"Give the HC to lock as a number above 0, not {ctx.args.hc!r}.")
    client = ctx.client()
    confirm(ctx, f"Lock {ctx.args.hc} HC on the waitlist?")
    found = spend(ctx, client, "/market/waitlist", {"lock": ctx.args.hc})
    emit(ctx, found, [f"Locked {found.get('locked')} HC; position {found.get('position')} on the waitlist."])
    return 0


def waitlist_leave(ctx):
    client = ctx.client()
    confirm(ctx, "Leave the waitlist and unlock your HC? You lose your position.")
    found = client.call("POST", "/market/waitlist/leave", {}, auth="session")
    emit(ctx, found, ["Left the waitlist; your HC is unlocked."])
    return 0


def keys_ls(ctx):
    keys = account_state(ctx.client()).get("keys", [])
    mine = ctx.mine().get("key_id")
    lines = []
    for k in keys:
        marks = ("revoked" if k.get("revoked") else "active") + ("  this machine" if k.get("key") == mine else "")
        lines.append(f"{k.get('key')}  ...{k.get('hint')}  {k.get('created')}  {marks}  {k.get('note') or ''}".rstrip())
    emit(ctx, keys, lines or ["No keys."])
    return 0


def keys_create(ctx):
    body = {"note": ctx.args.note or "sparkpipe CLI on " + socket.gethostname()[:60]}
    if ctx.args.key_fleet:
        body["fleet"] = ctx.args.key_fleet
    if ctx.args.allow_swap:
        body["allow_swap"] = True
    made = ctx.client().call("POST", "/market/keys", body, auth="session")
    emit(ctx, made, [f"Key {made.get('id')}: {made.get('key')}", "It is shown only once: store it now."])
    return 0


def keys_revoke(ctx):
    found = ctx.client().call("POST", "/market/keys/revoke", {"id": ctx.args.id}, auth="session")
    lines = [f"Revoked {ctx.args.id}."]
    if ctx.mine().get("key_id") == ctx.args.id:
        ctx.cfg.pop("key", None)
        ctx.cfg.pop("key_id", None)
        ctx.save()
        lines.append("That was this machine's key: run sparkpipe login for a new one.")
    emit(ctx, found, lines)
    return 0


def run(command, stdin=None, check=True, secret=False):
    try:
        found = subprocess.run(command, input=stdin, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise Failure(f"{command[0]} is not installed{INSTALL.get(command[0], '')}.")
    if check and found.returncode != 0:
        detail = "" if secret else ": " + found.stderr.strip()[:300]
        raise Failure(f"{' '.join(command[:2])} exited {found.returncode}{detail}")
    return found


class MacKeychain:
    name = "the macOS keychain"

    def store(self, account, value):
        run(["security", "-i"], f"add-generic-password -U -s {KEYCHAIN_SERVICE} -a {account} -X {value.encode().hex()}\n", secret=True)
        if self.load(account) != value:
            raise Failure("The macOS keychain did not keep the token.")

    def load(self, account):
        found = run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", account, "-w"], check=False)
        return found.stdout.rstrip("\n") if found.returncode == 0 else None

    def delete(self, account):
        run(["security", "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", account], check=False)


class SecretTool:
    name = "the Secret Service (secret-tool)"

    def store(self, account, value):
        run(["secret-tool", "store", "--label", f"SparkPipe token {account}", "service", KEYCHAIN_SERVICE, "account", account], value, secret=True)
        if self.load(account) != value:
            raise Failure("The Secret Service did not keep the token.")

    def load(self, account):
        found = run(["secret-tool", "lookup", "service", KEYCHAIN_SERVICE, "account", account], check=False)
        return found.stdout.rstrip("\n") if found.returncode == 0 and found.stdout else None

    def delete(self, account):
        run(["secret-tool", "clear", "service", KEYCHAIN_SERVICE, "account", account], check=False)


def keychain():
    if PLATFORM == "darwin":
        return MacKeychain()
    if PLATFORM.startswith("linux"):
        return SecretTool()
    raise Failure("Tokens are kept in the macOS keychain or, on Linux, through secret-tool; this system has neither.")


def token_name(text):
    if not TOKEN_NAME.match(text):
        raise Failure(f"Name a token with letters, digits, '.', '_' or '-', not {text!r}.")
    return text


def host_matches(host, bound):
    return host == bound or host.endswith("." + bound)


def tokens_set(ctx):
    name = token_name(ctx.args.name)
    host = (ctx.args.host or TOKEN_HOSTS.get(name) or "").lower()
    if not HOST.match(host):
        raise Failure("Name the host this token belongs to, for example --host huggingface.co: it is sent only with fetches from that host.")
    store = keychain()
    if host_matches(host, "huggingface.co"):
        print(HF_GUIDANCE)
    value = prompt_secret(f"Token for {name} ({host}), not shown: ").strip()
    if not value:
        raise Failure("No token was entered; nothing was saved.")
    store.store(name, value)
    ctx.cfg.setdefault("tokens", {})[name] = {"host": host}
    ctx.save()
    print(f"Saved {name} in {store.name}. sparkpipe sends it only with fetches from {host}.")
    return 0


def tokens_ls(ctx):
    names = ctx.cfg.get("tokens", {})
    store = keychain() if names else None
    rows = [{"name": n, "host": e.get("host"), "stored": store.load(n) is not None} for n, e in sorted(names.items())]
    lines = [f"{r['name']}  {r['host']}" + ("" if r["stored"] else "  (missing from the keychain)") for r in rows]
    emit(ctx, rows, lines or ["No tokens. Add one with sparkpipe tokens set NAME."])
    return 0


def tokens_rm(ctx):
    name = token_name(ctx.args.name)
    if name not in ctx.cfg.get("tokens", {}):
        raise Failure(f"No token named {name}.")
    keychain().delete(name)
    del ctx.cfg["tokens"][name]
    ctx.save()
    print(f"Removed {name}.")
    return 0


def source_host(source):
    for prefix, host in SOURCE_HOSTS.items():
        if source.startswith(prefix):
            return host
    if not source.startswith(SOURCES):
        raise Failure("Fetch from an hf://, s3://, https:// or git+https:// source.")
    host = urllib.parse.urlsplit(source[4:] if source.startswith("git+") else source).hostname
    if not host:
        raise Failure(f"{source} names no host.")
    return host.lower()


def named_token(tokens, name, host):
    entry = tokens.get(name)
    if not entry:
        raise Failure(f"No token named {name}: add it with sparkpipe tokens set {name}.")
    if not host_matches(host, entry.get("host", "")):
        raise Failure(f"The token {name} belongs to {entry.get('host')}, and this source is on {host}. A token is sent only with fetches from its own host.")
    return name


def source_token(ctx, host):
    tokens = ctx.cfg.get("tokens", {})
    if getattr(ctx.args, "no_token", False):
        return None
    if ctx.args.token:
        name = named_token(tokens, ctx.args.token, host)
    else:
        found = [n for n, e in sorted(tokens.items()) if host_matches(host, e.get("host", ""))]
        if len(found) > 1:
            raise Failure(f"The tokens {', '.join(found)} all belong to {host}: choose one with --token.")
        if not found:
            return None
        name = found[0]
    value = keychain().load(name)
    if value is None:
        raise Failure(f"The token {name} is missing from the keychain: add it again with sparkpipe tokens set {name}.")
    return name, value


def token_header(token):
    return {"X-SparkPipe-Source-Token": token[1]} if token else {}


def fetch_start(ctx, source, path):
    token = source_token(ctx, source_host(source))
    body = {"source": source, "to": path}
    if ctx.args.include:
        body["include"] = ctx.args.include
    if ctx.args.then:
        body["then"] = ctx.args.then
    started = ctx.client().call("POST", "/v1/fetch", body, headers=token_header(token))
    using = f", with the token {token[0]}" if token else ""
    emit(ctx, started, [f"Fetch {started.get('id')} started: {source} to {path}{using}.", "Follow it with sparkpipe fetch ls."])
    return 0


def fetch_ls(ctx):
    found = ctx.client().call("GET", "/v1/fetch").get("data", [])
    lines = [f"{f.get('id')}  {f.get('state')}  {f.get('source')} to {f.get('to')}  {size(f.get('bytes'))} of {size(f.get('total_bytes'))}" for f in found]
    emit(ctx, found, lines or ["No fetches."])
    return 0


def fetch_resume(ctx, ident):
    client = ctx.client()
    path = "/v1/fetch/" + urllib.parse.quote(ident, safe="")
    host = source_host(client.call("GET", path).get("source") or "")
    token = source_token(ctx, host)
    if not token:
        raise Failure(f"No stored token belongs to {host}: add one with sparkpipe tokens set NAME --host {host}.")
    found = client.call("POST", path + "/resume", {}, headers=token_header(token))
    emit(ctx, found, [f"Fetch {ident} resumed with the token {token[0]}."])
    return 0


def fetch_cancel(ctx, ident):
    found = ctx.client().call("DELETE", "/v1/fetch/" + urllib.parse.quote(ident, safe=""))
    emit(ctx, found, [f"Fetch {ident} cancelled."])
    return 0


def fetch(ctx):
    target, rest = ctx.args.target, ctx.args.rest
    actions = {"ls": (0, fetch_ls), "resume": (1, fetch_resume), "cancel": (1, fetch_cancel)}
    count, action = actions.get(target, (1, None))
    if len(rest) != count:
        raise Failure("Usage: sparkpipe fetch SOURCE PATH, fetch ls, fetch resume ID or fetch cancel ID")
    if action:
        return action(ctx, *rest)
    return fetch_start(ctx, target, rest[0])


def remote_prefix(path):
    stripped = path.strip("/")
    if stripped and any(part in ("", ".", "..") for part in stripped.split("/")):
        raise Failure(f"{path} is not a storage path.")
    return stripped + "/" if stripped else ""


def sync_parts(prefix, path):
    if not isinstance(path, str) or not path.startswith(prefix):
        raise Failure(f"The fleet listed {path!r}, outside {prefix or 'the storage root'}.")
    rest = path[len(prefix):]
    parts = rest.split("/")
    if not rest or "\x00" in rest or "\\" in rest or any(part in ("", ".", "..") for part in parts):
        raise Failure(f"The fleet listed an unsafe path {path!r}; nothing was written for it.")
    return parts


def copy_stream(stream, out):
    digest, total = hashlib.sha256(), 0
    while True:
        data = stream.read(CHUNK)
        if not data:
            return total, digest.hexdigest()
        out.write(data)
        digest.update(data)
        total += len(data)


def check_download(parts, written, digest, sha, expected):
    name = "/".join(parts)
    if expected is not None and written != expected:
        raise Failure(f"{name}: received {written} bytes, the fleet listed {expected}; nothing was replaced.")
    if sha and digest != sha.lower():
        raise Failure(f"{name}: the SHA-256 does not match the fleet's; nothing was replaced.")


class LocalTarget:
    def __init__(self, dest):
        self.root = Path(dest).expanduser().resolve()
        self.name = str(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def place(self, parts):
        path = self.root.joinpath(*parts)
        real = Path(os.path.realpath(path.parent))
        if real != self.root and self.root not in real.parents:
            raise Failure(f"{path} leads outside {self.root}; nothing was written for it.")
        return path

    def write(self, parts, stream, sha, expected):
        path = self.place(parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".sparkpipe-", dir=str(path.parent))
        try:
            with os.fdopen(handle, "wb") as out:
                written, digest = copy_stream(stream, out)
            check_download(parts, written, digest, sha, expected)
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise
        return written

    def remove(self, parts):
        path = self.place(parts)
        if path.is_file() or path.is_symlink():
            path.unlink()


class BucketTarget:
    def __init__(self, dest):
        self.name = dest.rstrip("/")
        if not shutil.which("aws"):
            raise Failure("Syncing to s3:// uses the aws command with your own credentials: install and configure it first" + INSTALL["aws"] + ".")

    def write(self, parts, stream, sha, expected):
        with tempfile.TemporaryDirectory(prefix="sparkpipe-sync-") as folder:
            local = os.path.join(folder, "file")
            with open(local, "wb") as out:
                written, digest = copy_stream(stream, out)
            check_download(parts, written, digest, sha, expected)
            run(["aws", "s3", "cp", "--only-show-errors", local, self.name + "/" + "/".join(parts)])
        return written

    def remove(self, parts):
        run(["aws", "s3", "rm", "--only-show-errors", self.name + "/" + "/".join(parts)])


def sync_target(dest):
    if dest.startswith("s3://"):
        return BucketTarget(dest)
    if "://" in dest:
        raise Failure("Sync to a local directory or to an s3:// bucket.")
    return LocalTarget(dest)


def apply_change(client, target, prefix, change, delete, counts):
    parts = sync_parts(prefix, change.get("path"))
    kind = change.get("change")
    if kind == "deleted":
        if delete:
            target.remove(parts)
        counts["deleted" if delete else "kept"] += 1
        return
    if kind not in ("added", "changed"):
        raise Failure(f"The fleet listed {change.get('path')} with an unknown change {kind!r}.")
    with client.open("GET", "/v1/storage/" + urllib.parse.quote(change["path"], safe="/")) as response:
        counts["bytes"] += target.write(parts, response, change.get("sha256"), change.get("size"))
    counts["copied"] += 1


def storage_sync(ctx):
    prefix = remote_prefix(ctx.args.path)
    target = sync_target(ctx.args.dest)
    client = ctx.client()
    key = f"{ctx.api} {prefix} {target.name}"
    since = ctx.cfg.get("sync", {}).get(key)
    counts = {"copied": 0, "bytes": 0, "deleted": 0, "kept": 0}
    while True:
        query = dict({"prefix": prefix}, **({"since": since} if since else {}))
        page = client.call("GET", "/v1/storage/changes", query=query)
        for change in page.get("data", []):
            apply_change(client, target, prefix, change, ctx.args.delete, counts)
        if page.get("more") and page.get("until") in (None, since):
            raise Failure("The fleet's list of changes did not advance; nothing more was copied.")
        since = page.get("until") or since
        if not page.get("more"):
            break
    ctx.cfg.setdefault("sync", {})[key] = since
    ctx.save()
    lines = [f"Copied {counts['copied']} file(s), {size(counts['bytes'])}, from {prefix or 'storage'} to {target.name}."]
    if counts["deleted"]:
        lines.append(f"Deleted {counts['deleted']} file(s) deleted on the fleet.")
    if counts["kept"]:
        lines.append(f"Kept {counts['kept']} file(s) deleted on the fleet; add --delete to remove them here too.")
    emit(ctx, dict(counts, until=since), lines)
    return 0


def fleet_models(ctx, pattern):
    client = ctx.client()
    listing = {m.get("id"): m for m in client.call("GET", "/v1/models").get("data", []) if isinstance(m, dict)}
    aliases = client.call("GET", "/v1/models/aliases")
    main = ctx.args.model or aliases.get(pattern) or aliases.get("claude-*")
    if not main:
        raise Failure(f"The fleet's alias table names no model for {pattern}: pass --model.")
    haiku = getattr(ctx.args, "haiku", None) or aliases.get("claude-haiku-*") or main
    return client, main, haiku, listing


def limits(listing, model, fields=MODEL_FIELDS):
    entry = listing.get(model)
    if not entry:
        raise Failure(f"The fleet does not list {model}; sparkpipe status shows its models.")
    found = {}
    for field in fields:
        value = entry.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise Failure(f"The fleet did not give {field} for {model}.")
        found[field] = value
    if found.get("max_output_tokens", 0) >= found["context_window"]:
        raise Failure(f"The fleet gave {model} an output limit no smaller than its context window.")
    return found


def key_reminder(client, tool):
    if os.environ.get("SPARKPIPE_API_KEY") == client.key:
        return
    print(f"{tool} reads the key from SPARKPIPE_API_KEY. Add this line to your shell profile:")
    print(f"  export SPARKPIPE_API_KEY={client.key}")


def claude_env(api, key, main, haiku, context):
    return {
        "ANTHROPIC_BASE_URL": api,
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_MODEL": main,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": haiku,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context),
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_SUBAGENT_MODEL": main,
        "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1",
    }


def object_fields(path, data, names):
    for name in names:
        if not isinstance(data.get(name, {}), dict):
            raise Failure(f"{path}: {name} is not a JSON object; fix it and run this again. Nothing was changed.")


def setup_claude_code(ctx):
    client, main, haiku, listing = fleet_models(ctx, "claude-*")
    context = limits(listing, main, ("context_window",))["context_window"]
    path = Path(ctx.args.settings or Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "settings.json")
    settings = read_json(path)
    object_fields(path, settings, ("env", "modelOverrides"))
    settings["env"] = dict(settings.get("env", {}), **claude_env(ctx.api, client.key, main, haiku, context))
    settings["modelOverrides"] = dict(settings.get("modelOverrides", {}), **{name: main if role == "main" else haiku for name, role in CLAUDE_MODELS.items()})
    write_file(path, json_text(settings), 0o600)
    background = f", and {haiku} for background calls" if haiku != main else ""
    print(f"Wrote {path}: Claude Code uses {main} on {ctx.api}{background}.")
    if "ANTHROPIC_API_KEY" in settings["env"] or os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is also set: Claude Code would send it to SparkPipe as X-Api-Key. Remove it from your settings and shell.")
    return 0


def toml_value(value):
    return str(value) if isinstance(value, int) else json.dumps(value, ensure_ascii=False)


def toml_block(path, pairs):
    return "\n".join(["[" + ".".join(path) + "]"] + [f"{name} = {toml_value(value)}" for name, value in pairs])


def toml_string(quoted, literal, bare):
    if quoted is None:
        return literal if literal is not None else bare
    try:
        return json.loads('"' + quoted + '"')
    except ValueError:
        return quoted


def toml_path(text):
    parts, index = [], 0
    while index < len(text):
        found = PART.match(text, index)
        if not found:
            return (text.strip(),)
        parts.append(toml_string(found.group(1), found.group(2), found.group(3)))
        index = found.end()
        if not found.group(4):
            break
    return tuple(parts)


def toml_header(line):
    found = HEADER.match(line)
    return toml_path(found.group(1)) if found else None


def toml_key(line):
    found = KEY.match(line)
    return toml_string(found.group(1), found.group(2), found.group(3)) if found else None


def codex_owned(table):
    return any(table[:2] == owned for owned in CODEX_TABLES)


def merge_codex(text, root, tables):
    kept, table = [], ()
    for line in text.splitlines():
        header = toml_header(line)
        if header is not None:
            table = header
        if (table and codex_owned(table)) or (not table and toml_key(line) in CODEX_ROOT):
            continue
        kept.append(line)
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    parts = ["\n".join(f"{name} = {toml_value(value)}" for name, value in root)]
    parts += ["\n".join(kept)] if kept else []
    parts += [toml_block(path, pairs) for path, pairs in tables]
    return "\n\n".join(parts) + "\n"


def codex_tables(api, main, found):
    return [
        (("model_providers", "sparkpipe"), [("name", "SparkPipe"), ("base_url", api + "/v1"), ("env_key", "SPARKPIPE_API_KEY"), ("wire_api", "responses"), ("stream_idle_timeout_ms", 600000)]),
        (("profiles", "sparkpipe"), [("model", main), ("model_provider", "sparkpipe"), ("model_context_window", found["context_window"]),
                                     ("model_auto_compact_token_limit", found["context_window"] - found["max_output_tokens"]), ("model_catalog_url", api + "/v1/codex/models")]),
        (("profiles", "sparkpipe", "agents"), [("default_subagent_model", main), ("max_concurrent_threads_per_session", found["sequences"])]),
        (("mcp_servers", "sparkpipe"), [("url", api + "/mcp"), ("bearer_token_env_var", "SPARKPIPE_API_KEY")]),
    ]


def codex_others(doc):
    rest = {name: value for name, value in doc.items() if name not in CODEX_ROOT}
    for top, name in CODEX_TABLES:
        if isinstance(rest.get(top), dict):
            rest[top] = {k: v for k, v in rest[top].items() if k != name}
            if not rest[top]:
                del rest[top]
    return rest


def dig(doc, path):
    for name in path:
        doc = doc.get(name, {}) if isinstance(doc, dict) else {}
    return doc


def check_codex(path, before, after, root, tables):
    if tomllib is None:
        return
    try:
        old = tomllib.loads(before)
    except tomllib.TOMLDecodeError as error:
        raise Failure(f"{path} is not valid TOML ({error}); fix it and run this again. Nothing was changed.")
    try:
        new = tomllib.loads(after)
    except tomllib.TOMLDecodeError as error:
        raise Failure(f"{path}: adding the SparkPipe settings would break it ({error}); nothing was changed.")
    wanted = [(dig(new, table).get(name), value) for table, pairs in [((), root)] + tables for name, value in pairs]
    if codex_others(old) != codex_others(new) or any(got != value for got, value in wanted):
        raise Failure(f"{path}: the SparkPipe settings could not be merged without changing others; nothing was changed.")


def setup_codex(ctx):
    client, main, _, listing = fleet_models(ctx, "gpt-*")
    found = limits(listing, main)
    path = Path(ctx.args.config or Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml")
    try:
        before = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        before = ""
    root, tables = [("model", main), ("model_provider", "sparkpipe")], codex_tables(ctx.api, main, found)
    after = merge_codex(before, root, tables)
    check_codex(path, before, after, root, tables)
    write_file(path, after, file_mode(path, 0o600))
    print(f"Wrote {path}: Codex uses {main} on {ctx.api}, also as the profile sparkpipe (codex --profile sparkpipe).")
    key_reminder(client, "Codex")
    return 0


def opencode_provider(api, main, haiku, listing):
    models = {}
    for name in dict.fromkeys((main, haiku)):
        found = limits(listing, name, ("context_window", "max_output_tokens"))
        models[name] = {"name": name, "limit": {"context": found["context_window"], "output": found["max_output_tokens"]}}
    return {"npm": "@ai-sdk/anthropic", "name": "SparkPipe", "options": {"baseURL": api + "/v1", "apiKey": "{env:SPARKPIPE_API_KEY}"}, "models": models}


def setup_opencode(ctx):
    client, main, haiku, listing = fleet_models(ctx, "claude-*")
    path = Path(ctx.args.config or Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "opencode" / "opencode.json")
    settings = read_json(path)
    object_fields(path, settings, ("provider", "agent", "mcp", "compaction"))
    settings.setdefault("$schema", "https://opencode.ai/config.json")
    settings.setdefault("provider", {})["sparkpipe"] = opencode_provider(ctx.api, main, haiku, listing)
    settings["model"] = "sparkpipe/" + main
    settings["small_model"] = "sparkpipe/" + haiku
    agents = settings.setdefault("agent", {})
    for name in OPENCODE_SUBAGENTS:
        agents[name] = dict(agents.get(name) if isinstance(agents.get(name), dict) else {}, model="sparkpipe/" + main)
    settings.setdefault("mcp", {})["sparkpipe"] = {"type": "remote", "url": ctx.api + "/mcp", "headers": {"Authorization": "Bearer {env:SPARKPIPE_API_KEY}"}, "enabled": True}
    settings.setdefault("compaction", {})["prune"] = False
    settings["share"] = "disabled"
    write_file(path, json_text(settings), file_mode(path, 0o600))
    print(f"Wrote {path}: OpenCode uses sparkpipe/{main} on {ctx.api}.")
    key_reminder(client, "OpenCode")
    return 0


def retry_wait(response, most):
    try:
        wait = int(response.getheader("retry-after") or "")
    except ValueError:
        return None
    return wait if 0 <= wait <= most else None


def connect(api):
    parts = urllib.parse.urlsplit(api)
    kind = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    return kind(parts.hostname, parts.port, timeout=UPSTREAM_TIMEOUT)


def exchange(settings, method, path, body, headers):
    attempt = 0
    while True:
        connection = connect(settings["api"])
        connection.request(method, settings["base"] + path, body=body, headers=headers)
        response = connection.getresponse()
        wait = retry_wait(response, settings["max_wait"]) if response.status == 429 else None
        if wait is None or attempt >= settings["retries"]:
            return connection, response
        response.read()
        connection.close()
        time.sleep(wait)
        attempt += 1


def outbound(incoming, settings):
    sent = {}
    for name, value in incoming.items():
        lower = name.lower()
        if lower in HOP or lower in CREDENTIALS or (lower == "x-sparkpipe-fleet" and settings["fleet"]):
            continue
        sent[name] = value
    sent["Authorization"] = "Bearer " + settings["key"]
    if settings["fleet"]:
        sent["X-SparkPipe-Fleet"] = settings["fleet"]
    return sent


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "sparkpipe-proxy"

    def log_message(self, fmt, *args):
        if self.server.settings["log"]:
            sys.stderr.write(fmt % args + "\n")

    def fail(self, status, kind, message):
        body = json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def request_body(self):
        if self.headers.get("Transfer-Encoding"):
            return self.fail(411, "invalid_request", "Send the request with a Content-Length.")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > BODY_LIMIT:
            return self.fail(413, "invalid_request", "The request body is missing or larger than 32 MiB.")
        return self.rfile.read(length) if length else b""

    def forward(self):
        settings = self.server.settings
        if not self.path.startswith("/"):
            return self.fail(400, "invalid_request", "Send requests to this proxy with a path such as /v1/messages.")
        body = self.request_body()
        if body is None:
            return
        try:
            connection, response = exchange(settings, self.command, self.path, body or None, outbound(self.headers, settings))
        except (OSError, http.client.HTTPException) as error:
            return self.fail(502, "upstream_error", f"Cannot reach {settings['api']}: {error}")
        try:
            self.relay(response)
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally:
            connection.close()

    def relay(self, response):
        empty = self.command == "HEAD" or response.status in (204, 304)
        length = response.getheader("Content-Length")
        chunked = length is None and not empty
        self.log_request(response.status)
        self.send_response_only(response.status, response.reason)
        for name, value in response.getheaders():
            if name.lower() not in HOP:
                self.send_header(name, value)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", length or "0")
        self.end_headers()
        while not empty:
            data = response.read1(CHUNK)
            self.wfile.write((b"%x\r\n%s\r\n" % (len(data), data)) if chunked else data)
            self.wfile.flush()
            if not data:
                return

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = forward


def proxy_server(api, key, fleet, port, retries, max_wait, log=True):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), ProxyHandler)
    server.daemon_threads = True
    server.settings = {"api": api, "base": urllib.parse.urlsplit(api).path, "key": key, "fleet": fleet, "retries": retries, "max_wait": max_wait, "log": log}
    return server


def proxy(ctx):
    client = ctx.client()
    client.credentials("key")
    server = proxy_server(ctx.api, client.key, client.fleet, ctx.args.port, ctx.args.retries, ctx.args.max_wait)
    port = server.server_address[1]
    fleet = f", fleet {client.fleet}" if client.fleet else ""
    print(f"Proxy for {ctx.api}{fleet} on http://127.0.0.1:{port}. Any API key works; the proxy sends this machine's key.")
    print(f"  Anthropic clients: base URL http://127.0.0.1:{port}")
    print(f"  OpenAI clients: base URL http://127.0.0.1:{port}/v1")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def leaf(group, name, run_, parents, text):
    command = group.add_parser(name, parents=parents, help=text, description=text)
    command.set_defaults(run=run_)
    return command


def spending(command):
    command.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    command.add_argument("--accept", action="store_true", help="accept the current Customer Agreement")


def account_commands(commands, base):
    login_ = leaf(commands, "login", login, [base], "Sign in with an email code and create a key for this machine")
    login_.add_argument("--email")
    login_.add_argument("--company", help="open a new account under this legal name")
    login_.add_argument("--accept", action="store_true", help="accept the Customer Agreement")
    book = commands.add_parser("book", help="Book a fleet by the week or by the hour").add_subparsers(dest="action", metavar="ACTION")
    book.required = True
    week = leaf(book, "week", book_week, [base], "Book weeks of a fleet")
    week.add_argument("fleet_id", metavar="FLEET")
    week.add_argument("weeks", type=int)
    spending(week)
    hours = leaf(book, "hours", book_hours, [base], "Book hours of a fleet nobody holds")
    hours.add_argument("fleet_id", metavar="FLEET")
    hours.add_argument("start", help="the first UTC hour, such as 2027-01-05T10")
    hours.add_argument("hours", type=int)
    spending(hours)
    waitlist = commands.add_parser("waitlist", help="Join or leave the waitlist").add_subparsers(dest="action", metavar="ACTION")
    waitlist.required = True
    lock = leaf(waitlist, "lock", waitlist_lock, [base], "Lock HC on the waitlist")
    lock.add_argument("hc")
    spending(lock)
    leave = leaf(waitlist, "leave", waitlist_leave, [base], "Leave the waitlist")
    leave.add_argument("--yes", action="store_true")
    keys = commands.add_parser("keys", help="API keys").add_subparsers(dest="action", metavar="ACTION")
    keys.required = True
    leaf(keys, "ls", keys_ls, [base], "List the account's keys")
    create = leaf(keys, "create", keys_create, [base], "Create a key")
    create.add_argument("--note")
    create.add_argument("--fleet", dest="key_fleet", help="limit the key to one fleet")
    create.add_argument("--allow-swap", action="store_true", help="let the key replace resident models")
    leaf(keys, "revoke", keys_revoke, [base], "Revoke a key").add_argument("id")


def fleet_commands(commands, base, fleet):
    leaf(commands, "status", status, [base, fleet], "Holdings, deadlines and the fleet's working state")
    setup = commands.add_parser("setup", help="Configure a coding agent for the fleet").add_subparsers(dest="action", metavar="HARNESS")
    setup.required = True
    claude = leaf(setup, "claude-code", setup_claude_code, [base, fleet], "Configure Claude Code")
    claude.add_argument("--settings", help="the settings file (default ~/.claude/settings.json)")
    codex = leaf(setup, "codex", setup_codex, [base, fleet], "Configure Codex")
    codex.add_argument("--config", help="the config file (default ~/.codex/config.toml)")
    opencode = leaf(setup, "opencode", setup_opencode, [base, fleet], "Configure OpenCode")
    opencode.add_argument("--config", help="the config file (default ~/.config/opencode/opencode.json)")
    for command in (claude, codex, opencode):
        command.add_argument("--model", help="the model to use (default: the fleet's alias table)")
    for command in (claude, opencode):
        command.add_argument("--haiku", help="the model for background calls")
    fetch_ = leaf(commands, "fetch", fetch, [base, fleet], "Fetch from a source into the fleet's storage: fetch SOURCE PATH, fetch ls, fetch resume ID, fetch cancel ID")
    fetch_.add_argument("target", metavar="SOURCE|ls|resume|cancel")
    fetch_.add_argument("rest", nargs="*", metavar="PATH|ID")
    fetch_.add_argument("--token", help="the stored token to send (default: the one stored for the source's host)")
    fetch_.add_argument("--no-token", action="store_true", help="send no token")
    fetch_.add_argument("--then", choices=("pack", "warm"))
    fetch_.add_argument("--include", action="append", metavar="GLOB")
    storage = commands.add_parser("storage", help="The fleet's storage").add_subparsers(dest="action", metavar="ACTION")
    storage.required = True
    sync = leaf(storage, "sync", storage_sync, [base, fleet], "Copy what changed under PATH since the last sync to DEST")
    sync.add_argument("path", metavar="PATH")
    sync.add_argument("dest", metavar="DEST", help="a local directory or s3://bucket/prefix")
    sync.add_argument("--delete", action="store_true", help="also delete files deleted on the fleet")
    proxy_ = leaf(commands, "proxy", proxy, [base, fleet], "A local endpoint that adds the key and fleet and retries 429s")
    proxy_.add_argument("--port", type=int, default=PROXY_PORT)
    proxy_.add_argument("--retries", type=int, default=6)
    proxy_.add_argument("--max-wait", type=int, default=60, help="the longest retry-after to wait, in seconds")


def token_commands(commands, base):
    tokens = commands.add_parser("tokens", help="Source tokens, kept in the OS keychain only").add_subparsers(dest="action", metavar="ACTION")
    tokens.required = True
    put = leaf(tokens, "set", tokens_set, [base], "Store a token, read without echo")
    put.add_argument("name")
    put.add_argument("--host", help="the host it belongs to (hf: huggingface.co)")
    leaf(tokens, "ls", tokens_ls, [base], "List stored tokens")
    leaf(tokens, "rm", tokens_rm, [base], "Remove a token").add_argument("name")


def parser():
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--api", help="the API address (default " + API + ")")
    base.add_argument("--json", action="store_true", help="print the API's JSON")
    fleet = argparse.ArgumentParser(add_help=False)
    fleet.add_argument("--fleet", help="the fleet, when the account holds several")
    top = argparse.ArgumentParser(prog="sparkpipe", description="Drive a SparkPipe fleet from your own machine.")
    top.add_argument("--version", action="version", version="sparkpipe " + VERSION)
    commands = top.add_subparsers(dest="command", metavar="COMMAND")
    commands.required = True
    account_commands(commands, base)
    fleet_commands(commands, base, fleet)
    token_commands(commands, base)
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.run(Context(args))
    except Failure as error:
        print(f"sparkpipe: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
