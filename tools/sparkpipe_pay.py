#!/usr/bin/env python3
import base64
import collections
import fcntl
import hashlib
import heapq
import hmac
import http.client
import http.server
import ipaddress
import itertools
import json
import math
import os
import queue
import re
import secrets
import select
import smtplib
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP, Decimal
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr

import sparkpipe_market as market
from sparkpipe_market import Refusal

KST = timezone(timedelta(hours=9))
USD = "5057"
KRW = "KRW"
CENT = Decimal("0.01")
WON = Decimal("1")
LEDGER_IDS = ("order", "invoice", "lot", "key", "ref", "booking", "job")
TERMS = {"ko": "agreement-ko.md", "en": "agreement-en.md"}
EVENTS = ["invoicePending", "invoicePaid", "invoiceCompleted", "invoiceCancelled", "invoiceTimedOut"]
OPEN = {"unpaid", "pending", "paid"}
CLOSED = {"cancelled", "timedout", "deleted", "failed"}
PROCESSORS = ("coinpayments", "toss", "swift")
TOSS_OPEN = ("created", "waiting")
SWIFT_SETTINGS = {"beneficiary": "SPARKPIPE_PAY_SWIFT_BENEFICIARY", "bank": "SPARKPIPE_PAY_SWIFT_BANK", "address": "SPARKPIPE_PAY_SWIFT_BANK_ADDRESS", "account": "SPARKPIPE_PAY_SWIFT_ACCOUNT", "bic": "SPARKPIPE_PAY_SWIFT_BIC"}
BIC = re.compile(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$")
TOSS_SECONDS = 2700
TOSS_KEY = re.compile(r"^[!-~]{1,200}$")
REFUND_CLAUSES = ("3.2", "3.5", "10.4", "16.4", "law")
CASH_METHODS = ("계좌이체", "가상계좌")
METHOD_EN = {"카드": "card", "가상계좌": "virtual account", "계좌이체": "bank transfer", "간편결제": "easy pay", "휴대폰": "mobile phone"}
BANKS = {code: name for codes, name in (
    (("39", "039"), "경남은행"), (("34", "034"), "광주은행"), (("06", "004"), "KB국민은행"), (("03", "003"), "IBK기업은행"), (("11", "011"), "NH농협은행"), (("31", "031"), "iM뱅크"), (("32", "032"), "부산은행"),
    (("07", "007"), "Sh수협은행"), (("88", "088"), "신한은행"), (("20", "020"), "우리은행"), (("71", "071"), "우체국"), (("37", "037"), "전북은행"), (("81", "081"), "하나은행")) for code in codes}
COINS = {
    "USDC.BASE": "51:0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
    "USDC.SOL": "55:EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDC.ARB": "64:0xaf88d065e77c8cc2239327c5edb3a432268e5831",
    "USDC.POL": "54:0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",
    "USDC.BEP20": "35:0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",
    "USDC.ERC20": "4:0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "USDT.TRC20": "9:TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t",
    "USDT.SOL": "55:Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "USDT.ARB": "64:0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",
    "USDT.POL": "54:0xc2132d05d31c914a87c6611c10748aeb04b58e8f",
    "USDT.BEP20": "35:0x55d398326f99059ff775485246999027b3197955",
    "USDT.ERC20": "4:0xdac17f958d2ee523a2206206994597c13d831ec7",
}
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,189}\.[^@\s.]{2,}$")
CODE = re.compile(r"^\d{6}$")
ORDER_PATH = re.compile(r"^/order/(SP[A-Z2-7]{16})$")
TOSS_PATH = re.compile(r"^/toss/(success|fail)/(SP[A-Z2-7]{16})$")
HOLDERS_PATH = re.compile(r"^/market/fleets/([A-Za-z0-9_-]{1,32})/holders$")
BRN = re.compile(r"^\d{3}-\d{2}-\d{5}$")
COOKIE = "sp_session"
SESSION_SECONDS = 30 * 86400
CODE_SECONDS = 600
CODE_TRIES = 5
PENDING_SECONDS = 86400
RECONCILE_DAYS = 14
SKEW_SECONDS = 300
SPENDING = ("/market/book-hours", "/market/book-weeks", "/market/waitlist", "/market/auto-extend")
OWNER_ACTIONS = SPENDING + ("/market/buy", "/market/waitlist/leave")
ACCOUNT_ROLES = {"requests_erase": (("owner",), "Deleting the records of the account's requests"), "webhook_put": (("owner", "admin"), "Changing webhooks"), "webhook_delete": (("owner", "admin"), "Changing webhooks"), "member_put": (("owner",), "Inviting members and changing roles"),
                 "member_delete": (("owner",), "Removing members"), "audit": (("owner", "admin"), "The audit log")}
AUDITED = {"/market/book-hours": ("booking.hours", ("booking", "fleet", "start", "hours", "hc")), "/market/book-weeks": ("booking.weeks", ("booking", "fleet", "first", "weeks", "price")),
           "/market/waitlist": ("waitlist.lock", ("locked",)), "/market/waitlist/leave": ("waitlist.leave", ()), "/market/auto-extend": ("auto_extend.set", ("fleet", "weeks_ahead")),
           "/market/buy": ("hc.bought", ("lot", "hc", "order")), "/market/keys": ("key.created", ("id", "member", "scopes", "fleet")), "/market/keys/revoke": ("key.revoked", ("id",)),
           "/market/keys/rotate": ("key.rotated", ("id", "replaces")), "/market/sessions/revoke": ("session.revoked", ("id",)), "key_create": ("key.created", ("id", "member", "scopes", "fleet")),
           "key_revoke": ("key.revoked", ("id",)), "key_rotate": ("key.rotated", ("id", "replaces")), "webhook_put": ("webhook.set", ("id", "events")), "webhook_delete": ("webhook.deleted", ("id",))}
LIMITS = {"code": (20, 600), "code-email": (5, 3600), "verify": (30, 600), "order": (10, 600), "read": (240, 600), "market": (240, 600), "gateway": (6000, 600)}
BODY_LIMIT = 32 * 1024 * 1024
REPORT_LIMIT = 4 * 1024 * 1024
ACCOUNT_LIMIT = 65536
BEARER = "sps-"
BEARER_SECONDS = 90 * 86400
RECORD_DAYS = 366
RECORDS_PER_REPORT = 10000
RETRY_SECONDS = (60, 300, 1800, 7200, 21600, 43200, 86400)
SWEEP_DAYS = 3
NOTICE_DAYS = 30
TUNNEL_CHECK = 30
GUARD_STEP = 1
WEBHOOK_SECONDS = 20
WEBHOOK_WORKERS = 4
LISTEN_QUEUE = 1024
CHUNK = re.compile(rb"[0-9A-Fa-f]{1,8}")
WEBHOOK_SCHEMES = ("https",)
WEBHOOKS_PER_ACCOUNT = 10
HEALTH_TIMEOUT = 10
UPTIME_GAP_SECONDS = 90
UPTIME_MONTHS_SHOWN = 4
UPTIME_DAYS_SHOWN = 90
UPTIME_YEARS = 5
INVITE_DAYS = 7
MEMBERS_PER_ACCOUNT = 50
HUB_AUDIT = re.compile(r"(model|models|cache|storage|fetch|run|export|restore|pack|repo)\.[a-z_]{1,40}")
AUDIT_LIMIT = 4096
OWN = {}
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te", "trailer", "trailers", "transfer-encoding", "upgrade", "host", "content-length", "expect"}
DROPPED = HOP | {"authorization", "x-api-key", "cookie", "forwarded", "x-real-ip"}
RESPONSE_DROPPED = {"connection", "keep-alive", "proxy-authenticate", "proxy-connection", "te", "trailer", "trailers", "transfer-encoding", "upgrade", "server", "date"}
CUSTOMER_HEADERS = {"x-sparkpipe-priority", "x-sparkpipe-source-token"}
NAME = r"[A-Za-z0-9_.-]+"
TOKEN = re.compile(r"[A-Za-z0-9!#$%&'*+.^_`|~-]+")
HUB_ROUTES = tuple((re.compile(pattern), scope, read) for pattern, scope, read in (
    (r"/v1/messages(/count_tokens)?", "inference", None),
    (rf"/v1/responses(/{NAME}(/[a-z_]+)?)?", "inference", None),
    (r"/v1/chat/completions", "inference", None),
    (r"/v1/(codex/)?models", "inference", None),
    (r"/v1/(tokenize|detokenize)", "inference", None),
    (rf"/v1/requests/{NAME}", "inference", None),
    (rf"/v1/batches(/{NAME})?", "inference", None),
    (r"/api/hello", "inference", None),
    (r"/mcp", "inference", None),
    (r"/v1/state(/move)?", "state", None),
    (rf"/v1/cache(/warm|/pins(/{NAME})?)?", "state", None),
    (rf"/v1/packs(/{NAME}(/refresh)?)?", "state", None),
    (r"/v1/models/(resident|aliases)", "models", "state"),
    (r"/v1/drafters(/active)?", "models", "state"),
    (rf"/v1/custom-models(/{NAME})?", "models", None),
    (rf"/v1/(repos|fetch)(/{NAME}(/resume)?)?", "storage", None),
    (r"/v1/storage(/changes|/.+)?", "storage", None),
    (r"/v1/(export|restore)", "storage", None),
    (rf"/git/{NAME}\.git(/.*)?", "storage", None),
    (rf"/v1/(functions|functions/{NAME}/runs|runs|runs/{NAME}(/log)?|schedules|reports/[0-9-]+)", "functions", None),
))
ACCOUNT_ROUTES = (
    ("GET", r"/v1/requests", "read", "requests"),
    ("DELETE", r"/v1/requests", "write", "requests_erase"),
    ("GET", r"/v1/keys", "read", "keys"),
    ("POST", r"/v1/keys", "write", "key_create"),
    ("DELETE", r"/v1/keys/(KY[A-Z2-7]{16})", "write", "key_revoke"),
    ("POST", r"/v1/keys/(KY[A-Z2-7]{16})/rotate", "write", "key_rotate"),
    ("GET", r"/v1/webhooks", "read", "webhooks"),
    ("PUT", r"/v1/webhooks", "write", "webhook_put"),
    ("DELETE", r"/v1/webhooks/(WH[A-Z2-7]{16})", "write", "webhook_delete"),
    ("GET", r"/v1/members", "read", "members"),
    ("POST", r"/v1/members", "write", "member_put"),
    ("DELETE", r"/v1/members/((?:MB|IV)[A-Z2-7]{16})", "write", "member_delete"),
    ("GET", r"/v1/audit", "read", "audit"),
)
RECORD_TEXT = {"request": 64, "account": 32, "member": 32, "key": 32, "fleet": 32, "model": 128, "pack": 128, "api": 32, "priority": 16, "session": 128, "at": 20}
RECORD_NUMBERS = ("status", "prompt_tokens", "cached_tokens", "prefill_tokens", "queue_ms", "prefill_ms", "ttft_ms", "output_tokens", "duration_ms", "decode_tps", "draft_accept", "cold_prefill_ms_estimate")
TOTALS = ("prompt_tokens", "cached_tokens", "prefill_tokens", "output_tokens", "prefill_ms", "cold_prefill_ms_estimate")
FOOTER = "\n\nChoose which notices you get by email or webhook on your account page or with PUT /v1/webhooks.\n\nHYOR AI Tech Ltd. (SparkPipe), Seoul. Reply to this email with any question."


def clock():
    return datetime.now(timezone.utc)


def log(text):
    print(text, file=sys.stderr, flush=True)


def money(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def text_money(value):
    return format(money(value), "f")


def won(value):
    return Decimal(value).quantize(WON, rounding=ROUND_HALF_UP)


def text_won(value):
    return format(won(value), "f")


def amount_text(value, currency):
    return f"{int(won(value)):,} KRW" if currency == KRW else f"{text_money(value)} USD"


def stamp_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def new_id(prefix):
    return prefix + base64.b32encode(secrets.token_bytes(10)).decode()


def sign(secret, method, url, client, stamp, body):
    message = "﻿".encode() + method.encode() + url.encode() + client.encode() + stamp.encode() + body
    return base64.b64encode(hmac.new(secret, message, hashlib.sha256).digest()).decode()


def verified(cfg, headers, raw, now):
    client = headers.get("X-CoinPayments-Client") or ""
    stamp = headers.get("X-CoinPayments-Timestamp") or ""
    given = headers.get("X-CoinPayments-Signature") or ""
    if not hmac.compare_digest(client.encode(), cfg.client_id.encode()):
        return False
    try:
        sent = datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    if abs((now - sent).total_seconds()) > SKEW_SECONDS:
        return False
    return hmac.compare_digest(sign(cfg.secret, "POST", cfg.webhook_url, client, stamp, raw).encode(), given.encode())


def session_token(key, account, expires):
    body = f"{account}.{expires}"
    return body + "." + hmac.new(key, body.encode(), hashlib.sha256).hexdigest()


def session_account(key, token, now):
    parts = token.split(".") if isinstance(token, str) else []
    if len(parts) != 3 or not parts[1].isdigit() or int(parts[1]) < now:
        return None
    good = hmac.new(key, f"{parts[0]}.{parts[1]}".encode(), hashlib.sha256).hexdigest()
    return parts[0] if hmac.compare_digest(good.encode(), parts[2].encode()) else None


def code_digest(email, code):
    return hashlib.sha256(f"{email}:{code}".encode()).hexdigest()


def hub_scope(method, path):
    for pattern, scope, read in HUB_ROUTES:
        if pattern.fullmatch(path):
            return read if read and method in ("GET", "HEAD") else scope
    return None


def credential(headers):
    given = (headers.get("Authorization") or "").strip()
    if given[:7].lower() == "bearer ":
        return given[7:].strip()
    if given[:6].lower() == "basic ":
        try:
            return base64.b64decode(given[6:].strip(), validate=True).decode().partition(":")[2].strip()
        except (ValueError, UnicodeDecodeError):
            return ""
    return (headers.get("X-Api-Key") or "").strip()


def allowed_address(text):
    found = ipaddress.ip_address(text.split("%")[0])
    return found.is_global and not found.is_multicast


def own_addresses(host):
    found = OWN.get(host)
    if not found or found[0] < time.monotonic():
        try:
            addresses = {info[4][0] for info in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
        except (OSError, UnicodeError):
            addresses = set()
        found = OWN[host] = (time.monotonic() + 600, addresses)
    return found[1]


def webhook_address(url, own):
    parts = urllib.parse.urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        found = sorted({info[4][0] for info in socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)})
    except (OSError, UnicodeError):
        raise Refusal(f"{parts.hostname} does not resolve.", 400, "webhook_address")
    mine = own_addresses(own) if own and own != parts.hostname else set()
    if not found or parts.hostname == own or set(found) & mine or not all(allowed_address(address) for address in found):
        raise Refusal("Webhooks go only to public internet addresses, never to private, loopback or link-local ones or to SparkPipe's own hosts.", 400, "webhook_address")
    return parts, port, found[0]


class Pinned(http.client.HTTPSConnection):
    def __init__(self, host, port, address, secure):
        super().__init__(host, port, timeout=WEBHOOK_SECONDS, context=ssl.create_default_context())
        self.address, self.secure, self.watch = address, secure, None

    def connect(self):
        self.watch = threading.Timer(WEBHOOK_SECONDS, self.cut)
        self.watch.daemon = True
        self.watch.start()
        sock = socket.create_connection((self.address, self.port), self.timeout)
        if not self.secure:
            self.sock = sock
            return
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host, do_handshake_on_connect=False)
        self.sock.do_handshake()

    def cut(self):
        try:
            if self.sock:
                socket.socket.shutdown(self.sock, socket.SHUT_RDWR)
        except OSError:
            pass

    def close(self):
        if self.watch:
            self.watch.cancel()
        super().close()


def webhook_signature(secret, moment, body):
    return f"t={moment},v1=" + hmac.new(secret.encode(), str(moment).encode() + b"." + body, hashlib.sha256).hexdigest()


def post_webhook(url, secret, payload, kind, ident, own):
    parts, port, address = webhook_address(url, own)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "SparkPipe-Webhooks/1", "X-SparkPipe-Event": kind, "X-SparkPipe-Delivery": ident,
               "X-SparkPipe-Signature": webhook_signature(secret, int(time.time()), body)}
    connection = Pinned(parts.hostname, port, address, parts.scheme == "https")
    try:
        connection.request("POST", (parts.path or "/") + ("?" + parts.query if parts.query else ""), body, headers)
        return connection.getresponse().status
    finally:
        connection.close()


def time_field(value, label):
    if value is None:
        return None
    for shape in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, shape).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    raise Refusal(f"Give {label} as a UTC time such as 2027-01-05T09:00:00Z or a date such as 2027-01-05.", 400, "invalid_request")


def upstream_field(text, name):
    found = urllib.parse.urlsplit(text)
    if found.scheme != "http" or not found.hostname:
        raise ValueError(f"{name} names the hub's chat layer over the tunnel, such as http://127.0.0.1:18081/v1")
    return found.hostname, found.port or 80, found.path[:-3] if found.path.endswith("/v1") else found.path.rstrip("/")


def read_secret(path):
    if not path:
        return b""
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip().encode()


class Config:
    def __init__(self, env):
        self.port = int(env.get("SPARKPIPE_PAY_PORT", "18090"))
        self.data = env["SPARKPIPE_PAY_DATA"]
        self.site = env["SPARKPIPE_PAY_SITE"]
        self.public = env.get("SPARKPIPE_PAY_PUBLIC", "https://sparkpipe.ai").rstrip("/")
        parts = urllib.parse.urlsplit(self.public)
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.sandbox = env.get("SPARKPIPE_PAY_SANDBOX") == "1"
        self.prefix = env.get("SPARKPIPE_PAY_PREFIX", "/pay/api")
        self.api = env.get("COINPAYMENTS_API", "https://a-api.coinpayments.net/api/v2").rstrip("/")
        self.client_id = env.get("COINPAYMENTS_CLIENT_ID", "")
        self.secret = read_secret(env.get("COINPAYMENTS_CLIENT_SECRET_FILE"))
        if not self.sandbox and not (self.client_id and self.secret):
            raise ValueError("COINPAYMENTS_CLIENT_ID and COINPAYMENTS_CLIENT_SECRET_FILE are required outside the sandbox")
        self.webhook_url = self.public + self.prefix + "/coinpayments"
        self.toss_api = env.get("TOSS_API", "https://api.tosspayments.com").rstrip("/")
        self.toss_client = env.get("TOSS_CLIENT_KEY", "").strip()
        self.toss_secret = env.get("TOSS_SECRET_KEY", "").strip()
        self.toss_variant = env.get("TOSS_VARIANT_KEY", "").strip() or "DEFAULT"
        mode = "test_" if self.sandbox else "live_"
        if (self.toss_client or self.toss_secret) and not (self.toss_client.startswith(mode) and self.toss_secret.startswith(mode)):
            raise ValueError(f"TOSS_CLIENT_KEY and TOSS_SECRET_KEY must both be Toss Payments {mode[:-1]} keys " + ("in the sandbox" if self.sandbox else "outside the sandbox"))
        self.toss = bool(self.toss_client and self.toss_secret)
        self.pool_upstream = env.get("SPARKPIPE_POOL_UPSTREAM", "").rstrip("/")
        self.pool_key = read_secret(env.get("SPARKPIPE_POOL_KEY_FILE")).decode()
        self.pool = upstream_field(self.pool_upstream, "SPARKPIPE_POOL_UPSTREAM") if self.pool_upstream else None
        self.upstreams = {}
        for item in re.split(r"[\s,]+", env.get("SPARKPIPE_FLEET_UPSTREAMS", "").strip()):
            fleet, _, url = item.partition("=")
            if item and not (re.fullmatch(r"[A-Za-z0-9_-]{1,32}", fleet) and url):
                raise ValueError("SPARKPIPE_FLEET_UPSTREAMS lists FLEET=URL pairs, such as F1B=http://127.0.0.1:18081/v1 F1A=http://127.0.0.1:18083/v1")
            if item:
                self.upstreams[fleet] = upstream_field(url.rstrip("/"), "SPARKPIPE_FLEET_UPSTREAMS")
        self.hub = bool(self.pool or self.upstreams)
        capacity = env.get("SPARKPIPE_GATEWAY_CAPACITY", "").strip()
        if not re.fullmatch(r"[0-9]{1,12}", capacity) or int(capacity) < 1:
            raise ValueError("SPARKPIPE_GATEWAY_CAPACITY is required: the requests the gateway serves in 10 minutes, measured with tools/gateway_capacity_check.py "
                             "(capacity_per_window), a whole number of at least 1. Holders of an active rental share it equally per rented Fleet")
        self.gateway_capacity = int(capacity)
        self.health_seconds = float(env.get("SPARKPIPE_PAY_HEALTH_SECONDS", "30"))
        self.reconcile_seconds = int(env.get("SPARKPIPE_PAY_RECONCILE_SECONDS", "300"))
        self.smtp = env.get("SPARKPIPE_PAY_SMTP", "127.0.0.1:25")
        self.mail_from = env.get("SPARKPIPE_PAY_MAIL_FROM", "SparkPipe <noreply@sparkpipe.ai>")
        self.notify = env.get("SPARKPIPE_PAY_NOTIFY", "hyor@sparkpipe.ai")
        self.swift_bank = {name: env.get(key, "").strip() for name, key in SWIFT_SETTINGS.items()}
        if any(self.swift_bank.values()) and not (all(self.swift_bank.values()) and BIC.match(self.swift_bank["bic"]) and all(len(v) <= 200 and v.isprintable() for v in self.swift_bank.values())):
            raise ValueError("Set all of " + ", ".join(SWIFT_SETTINGS.values()) + " on one line each, with an 8 or 11 character SWIFT/BIC, or leave them all empty")
        self.swift = all(self.swift_bank.values()) and not self.sandbox
        self.terms_seen = {}

    def path(self, name):
        return os.path.join(self.data, name)

    def upstream(self, fleet):
        return self.upstreams.get(fleet) or self.pool

    def pricing(self):
        with open(os.path.join(self.site, "pricing.js"), encoding="utf-8") as handle:
            text = handle.read()
        return json.loads(text[text.index("{"):text.rindex("}") + 1])

    def terms(self, name):
        path = os.path.join(self.site, "terms", name)
        found = os.stat(path)
        mark = (found.st_mtime_ns, found.st_size, found.st_ino)
        if self.terms_seen.get(name, (None,))[0] != mark:
            with open(path, "rb") as handle:
                self.terms_seen[name] = (mark, hashlib.sha256(handle.read()).hexdigest())
        return self.terms_seen[name][1]

    def secret_file(self, name):
        path = self.path(name)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            with open(path, "rb") as handle:
                return handle.read()
        with os.fdopen(descriptor, "wb") as handle:
            key = secrets.token_bytes(32)
            handle.write(key)
            return key


def call(cfg, method, path, body=None):
    url = cfg.api + path
    payload = b"" if body is None else json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
    stamp = clock().strftime("%Y-%m-%dT%H:%M:%S")
    request = urllib.request.Request(url, data=payload if body is not None else None, method=method, headers={
        "Content-Type": "application/json",
        "X-CoinPayments-Client": cfg.client_id,
        "X-CoinPayments-Timestamp": stamp,
        "X-CoinPayments-Signature": sign(cfg.secret, method, url, cfg.client_id, stamp, payload),
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{method} {path}: HTTP {error.code} {error.read()[:300].decode('utf-8', 'replace')}") from None


class Declined(Exception):
    def __init__(self, code, text, status):
        super().__init__(f"{code}: {text}")
        self.code = code
        self.status = status


def toss_call(cfg, method, path, body=None, idempotency=None):
    token = base64.b64encode((cfg.toss_secret + ":").encode()).decode()
    headers = {"Authorization": "Basic " + token, "Content-Type": "application/json"}
    if idempotency:
        headers["Idempotency-Key"] = idempotency
    payload = None if body is None else json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
    request = urllib.request.Request(cfg.toss_api + path, data=payload, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            found = json.loads(error.read() or b"{}")
        except ValueError:
            found = {}
        found = found if isinstance(found, dict) else {}
        raise Declined(str(found.get("code") or "HTTP_" + str(error.code))[:100], str(found.get("message") or "")[:300], error.code) from None


def toss_payment(cfg, key):
    return toss_call(cfg, "GET", "/v1/payments/" + urllib.parse.quote(key, safe=""))


def send_mail(cfg, to, subject, text):
    mail = EmailMessage()
    mail["From"] = cfg.mail_from
    mail["To"] = to
    mail["Reply-To"] = cfg.notify
    mail["Subject"] = subject
    mail["Date"] = formatdate(usegmt=True)
    mail["Message-ID"] = make_msgid(domain=parseaddr(cfg.mail_from)[1].split("@")[-1])
    mail.set_content(text, cte="quoted-printable")
    host, port = cfg.smtp.rsplit(":", 1)
    with smtplib.SMTP(host, int(port), timeout=20) as smtp:
        smtp.send_message(mail)


class Book:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.head = "0" * 64
        self.count = 0
        self.offset = 0
        self.orders = {}
        self.invoices = {}
        self.accounts = {}
        self.emails = {}
        self.members = {}
        self.member_emails = {}
        self.invites = {}
        self.erased = {}
        self.market = market.State()
        if os.path.exists(path):
            with open(path, "rb") as handle:
                self.catch_up(handle)

    def catch_up(self, handle):
        handle.seek(self.offset)
        for raw in handle.read().splitlines(keepends=True):
            if not raw.endswith(b"\n"):
                raise ValueError(f"{self.path}: line {self.count + 1} is incomplete")
            line = json.loads(raw)
            body = {key: line[key] for key in ("seq", "at", "type", "data", "prev")}
            if line["seq"] != self.count + 1 or line["prev"] != self.head or digest(body) != line["hash"]:
                raise ValueError(f"{self.path}: the hash chain breaks at line {self.count + 1}")
            self.apply(line)
            self.offset += len(raw)

    def refresh(self):
        with self.lock:
            if os.path.exists(self.path):
                with open(self.path, "rb") as handle:
                    fcntl.flock(handle, fcntl.LOCK_SH)
                    self.catch_up(handle)

    def append(self, kind, data):
        with self.lock, open(self.path, "a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            self.catch_up(handle)
            body = {"seq": self.count + 1, "at": clock().strftime("%Y-%m-%dT%H:%M:%SZ"), "type": kind, "data": data, "prev": self.head}
            line = dict(body, hash=digest(body))
            raw = (json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            self.apply(line)
            self.offset += len(raw)
            return line

    def apply(self, line):
        self.count, self.head = line["seq"], line["hash"]
        data, kind = line["data"], line["type"]
        order = self.orders.get(data.get("order"))
        if kind in market.KINDS:
            self.market.apply(line)
        elif kind == "account":
            self.accounts[data["account"]] = dict(data, created=line["at"], accepted=None)
            self.emails[data["email"]] = data["account"]
        elif kind == "acceptance":
            self.accounts[data["account"]]["accepted"] = dict({name: data.get(name) for name in TERMS}, at=line["at"])
        elif kind == "records_erased":
            self.erased[data["account"]] = data["through"]
        elif kind == "member_invite":
            self.invites[data["invite"]] = dict(data, created=line["at"], state="invited")
            for other in self.invites.values():
                if other["email"] == data["email"] and other["account"] == data["account"] and other["invite"] != data["invite"] and other["state"] == "invited":
                    other["state"] = "replaced"
        elif kind == "member":
            self.members[data["member"]] = dict(data, created=line["at"], removed=False)
            self.member_emails[data["email"]] = data["member"]
            if data.get("invite") in self.invites:
                self.invites[data["invite"]]["state"] = "accepted"
        elif kind == "member_role":
            self.members[data["member"]]["role"] = data["role"]
        elif kind == "member_remove":
            if data.get("member") in self.members:
                self.members[data["member"]]["removed"] = True
                self.member_emails.pop(self.members[data["member"]]["email"], None)
            if data.get("invite") in self.invites:
                self.invites[data["invite"]]["state"] = "cancelled"
        elif kind == "order":
            self.orders[data["order"]] = dict(data, created=line["at"], state="unpaid" if data.get("processor") == "swift" else "created", invoice=None, link=None, issued=False, paid=None, refunds=[])
        elif kind == "invoice":
            order.update(invoice=data["invoice"], link=data["link"], state="unpaid")
            self.invoices[data["invoice"]] = data["order"]
        elif kind in ("invoice_failed", "invoice_state", "toss_state", "swift_state"):
            order.update({name: data[name] for name in ("state", "received", "reason", "method", "deposit", "expires") if name in data})
            order["state"] = "failed" if kind == "invoice_failed" else order["state"]
        elif kind == "toss_payment":
            order["payment_key"] = data["paymentKey"]
        elif kind == "payment":
            order.update(state="completed", paid=line["at"], received=data["received"], method=data.get("method"), tax_invoice=data.get("taxInvoice"))
        elif kind == "refund":
            order["refunded"] = format(Decimal(order.get("refunded") or "0") + Decimal(data["amount"]), "f")
            order["refunds"].append({"clause": data["clause"], "amount": data["amount"]})
            order["state"] = "refunded" if Decimal(data["balance"]) == 0 else order["state"]
        elif kind == "tax_invoice":
            order["tax_invoice_number"] = data["number"]
        elif kind == "hc_issued":
            order.update(issued=True, issued_hc=data["hc"], paid_net=data["net"])

    def expiry(self, order):
        return stamp_time(order["expires"]) if order.get("expires") else stamp_time(order["created"]) + timedelta(seconds=PENDING_SECONDS)


def private_folder(path):
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


def private_append(path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "a+b")


def trim(handle, path):
    end = handle.seek(0, os.SEEK_END)
    if not end:
        return
    handle.seek(end - 1)
    if handle.read(1) == b"\n":
        return
    keep = end - 1
    while keep > 0:
        start = max(0, keep - 65536)
        handle.seek(start)
        found = handle.read(keep - start).rfind(b"\n")
        if found >= 0:
            keep = start + found + 1
            break
        keep = start
    handle.truncate(keep)
    log(f"{path}: cut a torn last line of {end - keep} bytes before appending")


def parsed(lines, path):
    found = []
    for raw in lines:
        try:
            item = json.loads(raw)
        except ValueError:
            item = None
        if isinstance(item, dict):
            found.append(item)
        else:
            log(f"{path}: skipped a line that does not parse")
    return found


def complete_lines(path):
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except FileNotFoundError:
        return []
    return parsed(data[:data.rfind(b"\n") + 1].splitlines(), path)


class Records:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.items = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                self.items = json.load(handle)

    def get(self, ident):
        with self.lock:
            return self.items.get(ident)

    def values(self):
        with self.lock:
            return list(self.items.values())

    def find(self, test):
        with self.lock:
            return next(((ident, record) for ident, record in self.items.items() if test(record)), (None, None))

    def put(self, ident, record):
        with self.lock:
            self.items[ident] = record
            self.save()

    def drop(self, ident):
        with self.lock:
            found = self.items.pop(ident, None)
            if found is not None:
                self.save()
            return found

    def keep(self, test):
        with self.lock:
            gone = [ident for ident, record in self.items.items() if not test(record)]
            for ident in gone:
                del self.items[ident]
            if gone:
                self.save()

    def save(self):
        temporary = self.path + ".new"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self.items, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)


class Journal:
    def __init__(self, folder, days):
        self.folder, self.days = folder, days
        self.lock = threading.Lock()
        self.offsets = {}
        self.once = set()
        private_folder(folder)
        with self.lock:
            for name in self.names():
                self.catch_up(name)

    def names(self):
        return sorted(name for name in os.listdir(self.folder) if name.endswith(".jsonl"))

    def read_new(self, name, handle):
        handle.seek(self.offsets.get(name, 0))
        data = handle.read()
        done = data.rfind(b"\n") + 1
        for item in parsed(data[:done].splitlines(), os.path.join(self.folder, name)):
            if item.get("once"):
                self.once.add(item["once"])
        self.offsets[name] = self.offsets.get(name, 0) + done

    def catch_up(self, name):
        try:
            with open(os.path.join(self.folder, name), "rb") as handle:
                fcntl.flock(handle, fcntl.LOCK_SH)
                self.read_new(name, handle)
        except FileNotFoundError:
            pass

    def add(self, item, day=None):
        name = (day or item["now"][:10]) + ".jsonl"
        raw = (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        path = os.path.join(self.folder, name)
        with self.lock, private_append(path) as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            self.read_new(name, handle)
            trim(handle, path)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            self.offsets[name] += len(raw)
            if item.get("once"):
                self.once.add(item["once"])

    def sent(self, once):
        with self.lock:
            if once not in self.once:
                for name in self.names()[-SWEEP_DAYS:]:
                    self.catch_up(name)
            return once in self.once

    def items(self, last=None):
        found = []
        for name in self.names()[-last if last else 0:]:
            found += complete_lines(os.path.join(self.folder, name))
        return found

    def prune(self, now):
        cutoff = (now - timedelta(days=self.days)).strftime("%Y-%m-%d")
        with self.lock:
            for name in self.names():
                if name[:-6] < cutoff:
                    os.remove(os.path.join(self.folder, name))
                    self.offsets.pop(name, None)


class Usage:
    def __init__(self, folder, ident="request"):
        self.folder, self.ident = folder, ident
        self.lock = threading.Lock()
        self.ids = collections.OrderedDict()
        private_folder(folder)

    def path(self, account, day):
        return os.path.join(self.folder, account, day + ".jsonl")

    def lines(self, account, day):
        return complete_lines(self.path(account, day))

    def known(self, account, day):
        key = (account, day)
        if key not in self.ids:
            self.ids[key] = {record[self.ident] for record in self.lines(account, day)}
            while len(self.ids) > 256:
                self.ids.popitem(last=False)
        self.ids.move_to_end(key)
        return self.ids[key]

    def add(self, records):
        added = 0
        with self.lock:
            for account, day in sorted({(record["account"], record["at"][:10]) for record in records}):
                seen = self.known(account, day)
                fresh = [record for record in records if record["account"] == account and record["at"][:10] == day and record[self.ident] not in seen]
                unique = list({record[self.ident]: record for record in fresh}.values())
                if not unique:
                    continue
                private_folder(os.path.join(self.folder, account))
                with private_append(self.path(account, day)) as handle:
                    trim(handle, self.path(account, day))
                    handle.write("".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in unique).encode())
                    handle.flush()
                    os.fsync(handle.fileno())
                seen.update(record[self.ident] for record in unique)
                added += len(unique)
        return added

    def erase(self, account):
        folder, count = os.path.join(self.folder, account), 0
        with self.lock:
            for name in os.listdir(folder) if os.path.isdir(folder) else []:
                count += len(complete_lines(os.path.join(folder, name)))
                os.remove(os.path.join(folder, name))
                self.ids.pop((account, name[:-6]), None)
            if os.path.isdir(folder):
                os.rmdir(folder)
        return count

    def prune(self, now):
        cutoff = (now - timedelta(days=RECORD_DAYS)).strftime("%Y-%m-%d")
        with self.lock:
            for account in os.listdir(self.folder):
                folder = os.path.join(self.folder, account)
                for name in os.listdir(folder):
                    if name.endswith(".jsonl") and name[:-6] < cutoff:
                        os.remove(os.path.join(folder, name))
                        self.ids.pop((account, name[:-6]), None)
                if not os.listdir(folder):
                    os.rmdir(folder)

    def query(self, account, since, until, match):
        try:
            names = sorted(os.listdir(os.path.join(self.folder, account)))
        except FileNotFoundError:
            return
        first, last = since.strftime("%Y-%m-%d"), until.strftime("%Y-%m-%d")
        for name in names:
            if name.endswith(".jsonl") and first <= name[:-6] <= last:
                found = [record for record in self.lines(account, name[:-6]) if since <= stamp_time(record["at"]) < until and all(record.get(key) == value for key, value in match.items())]
                yield from sorted(found, key=lambda record: (record["at"], record[self.ident]))


def usage_totals(records):
    totals = {name: 0 for name in TOTALS}
    decode, timed, count = 0, 0.0, 0
    for record in records:
        count += 1
        for name in TOTALS:
            totals[name] += record.get(name) or 0
        if record.get("decode_tps") and record.get("output_tokens"):
            decode += record["output_tokens"]
            timed += record["output_tokens"] / record["decode_tps"]
    shown = {name: round(value, 3) if isinstance(value, float) else value for name, value in totals.items()}
    return dict(shown, requests=count, prefill_avoided_ms=shown["cold_prefill_ms_estimate"], cached_share=round(totals["cached_tokens"] / totals["prompt_tokens"], 4) if totals["prompt_tokens"] else 0,
                decode_tps=round(decode / timed, 2) if timed else None)


class Notifier:
    def __init__(self, app):
        self.app = app
        self.heap = []
        self.cond = threading.Condition()
        self.count = itertools.count()
        self.deliveries = Journal(app.cfg.path("deliveries"), NOTICE_DAYS)
        self.known = {(item["notice"], item["target"]) for item in self.deliveries.items(SWEEP_DAYS + 1)}

    def schedule(self, notice, target, attempt=0, delay=0.0):
        with self.cond:
            self.known.add((notice["notice"], target))
            heapq.heappush(self.heap, (time.monotonic() + delay, next(self.count), notice, target, attempt))
            self.cond.notify()

    def sweep(self, now):
        for notice in self.app.notices.items(SWEEP_DAYS + 1):
            if market.stamp_time(notice["now"]) >= now - timedelta(days=SWEEP_DAYS):
                for target in notice["targets"]:
                    with self.cond:
                        due = target != "mail" and (notice["notice"], target) not in self.known
                    if due and self.app.hooks.get(target):
                        self.schedule(notice, target)

    def finish(self, notice, target, attempt, outcome):
        moment = market.stamp(clock())
        self.deliveries.add({"notice": notice["notice"], "target": target, "attempts": attempt + 1, "outcome": outcome, "at": moment}, moment[:10])

    def run(self):
        while True:
            with self.cond:
                while not self.heap or self.heap[0][0] > time.monotonic():
                    self.cond.wait(self.heap[0][0] - time.monotonic() if self.heap else None)
                _, _, notice, target, attempt = heapq.heappop(self.heap)
            try:
                self.attempt(notice, target, attempt)
            except Exception as error:
                log(f"webhook {target}: {error}")

    def attempt(self, notice, target, attempt):
        hook = self.app.hooks.get(target)
        if not hook or hook["kind"] != "url":
            return self.finish(notice, target, attempt, "deleted")
        payload = {"text": notice["summary"]} if hook.get("slack") else {key: notice[key] for key in ("notice", "type", "account", "summary", "data")}
        if not hook.get("slack"):
            payload = dict(payload, id=payload.pop("notice"), created=notice["now"])
        try:
            status = post_webhook(hook["target"], hook["secret"], payload, notice["type"], notice["notice"], urllib.parse.urlsplit(self.app.cfg.public).hostname)
        except (OSError, ValueError, http.client.HTTPException, Refusal) as error:
            status = 0
            log(f"webhook {target} for {notice['notice']}: {error}")
        if 200 <= status < 300:
            return self.finish(notice, target, attempt, "delivered")
        if attempt < len(RETRY_SECONDS):
            return self.schedule(notice, target, attempt + 1, RETRY_SECONDS[attempt])
        self.finish(notice, target, attempt, f"failed {status}")


class Guard:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}
        self.count = itertools.count()

    def watch(self, check, cut):
        ident = next(self.count)
        with self.lock:
            self.active[ident] = [time.monotonic() + TUNNEL_CHECK, math.inf, check, cut]
        return ident

    def done(self, ident):
        with self.lock:
            self.active.pop(ident, None)

    def sweep(self):
        with self.lock:
            entries = list(self.active.items())
        for ident, entry in entries:
            now = time.monotonic()
            if now >= entry[0]:
                entry[0] = now + TUNNEL_CHECK
                try:
                    left = entry[2]()
                except Exception as error:
                    log(f"guard: {error}")
                    left = None
                entry[1] = now + left if left is not None else now
            if time.monotonic() >= entry[1]:
                self.done(ident)
                entry[3]()

    def run(self):
        while True:
            time.sleep(min(GUARD_STEP, TUNNEL_CHECK))
            self.sweep()


def hub_answer(hub, path):
    connection = http.client.HTTPConnection(hub[0], hub[1], timeout=HEALTH_TIMEOUT)
    try:
        connection.request("GET", hub[2] + path)
        response = connection.getresponse()
        body = response.read(1 << 20)
        try:
            found = json.loads(body or b"{}")
        except ValueError:
            found = None
        return response.status, found if isinstance(found, dict) else None
    finally:
        connection.close()


def fleet_up(health, control):
    status, view = control
    models = (health or {}).get("models")
    answers = {name: isinstance(entry, dict) and entry.get("status") == 200 for name, entry in (models.items() if isinstance(models, dict) else [])}
    named = lambda names: ", ".join(sorted(names)) or "no model"
    if health is None:
        return False, "the hub's /health did not answer"
    if status == 404:
        return health.get("ok") is True, "every model answers" if health.get("ok") is True else f"{named(n for n, up in answers.items() if not up)} not answering, without model control"
    if status != 200 or view is None:
        return any(answers.values()), f"model control unavailable; {named(n for n, up in answers.items() if up)} answering"
    if view.get("state") == "swapping":
        return True, f"swapping to {view.get('target') or 'another model'}"
    resident = [entry.get("id") for entry in view.get("models") or [] if isinstance(entry, dict) and entry.get("active") is True and isinstance(entry.get("id"), str)]
    if not resident:
        return view.get("state") == "idle", "no model resident; a request loads one" if view.get("state") == "idle" else f"no model resident after a {view.get('state')} swap"
    failing = [name for name in resident if not answers.get(name)]
    return not failing, f"{named(failing)} not answering" if failing else f"{named(resident)} answering"


def probe_health(hub):
    try:
        health = hub_answer(hub, "/health")[1]
    except (OSError, http.client.HTTPException) as error:
        return False, f"{type(error).__name__}: {error}"[:200]
    try:
        control = hub_answer(hub, "/v1/sparkpipe/status")
    except (OSError, http.client.HTTPException):
        control = (0, None)
    return fleet_up(health, control)


class Uptime:
    def __init__(self, app):
        self.app = app
        self.folder = app.cfg.path("uptime")
        private_folder(self.folder)
        self.lock = threading.Lock()
        self.state = Records(os.path.join(self.folder, "state.json"))
        self.recent = [span for name in self.months()[-UPTIME_MONTHS_SHOWN:] for span in complete_lines(os.path.join(self.folder, name))]

    def months(self):
        return sorted(name for name in os.listdir(self.folder) if re.fullmatch(r"\d{4}-\d{2}\.jsonl", name))

    def close(self, span, moment):
        span = dict(span, end=market.stamp(moment))
        path = os.path.join(self.folder, span["end"][:7] + ".jsonl")
        with private_append(path) as handle:
            trim(handle, path)
            handle.write((json.dumps(span, separators=(",", ":")) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        self.recent.append(span)

    def observe(self, fleet, up, moment, detail="", watch=True):
        with self.lock:
            entry = self.state.get(fleet) or {"good": None, "open": None, "first": market.stamp(moment)}
            good = market.stamp_time(entry["good"]) if entry["good"] else None
            gap = watch and good is not None and moment - good > timedelta(seconds=max(UPTIME_GAP_SECONDS, 3 * self.app.cfg.health_seconds))
            if up:
                if entry["open"]:
                    self.close(entry["open"], moment)
                elif gap:
                    self.close({"span": new_id("DT"), "fleet": fleet, "start": entry["good"], "cause": "gateway"}, moment)
                entry = dict(entry, good=market.stamp(moment), open=None)
            elif not entry["open"]:
                entry = dict(entry, open={"span": new_id("DT"), "fleet": fleet, "start": entry["good"] if watch and good else market.stamp(moment), "cause": "gateway" if gap else "sandbox" if not watch else "probe"})
            self.state.put(fleet, dict(entry, detail=detail[:200], seen=market.stamp(moment)))

    def spans(self, since):
        with self.lock:
            closed = [span for span in self.recent if span["end"] >= market.stamp(since)]
            return closed + [dict(entry["open"], end=None) for entry in self.state.values() if entry.get("open")]

    def periods(self, since):
        return market.joined(self.spans(since), timedelta(seconds=self.app.cfg.health_seconds))

    def entry(self, fleet):
        return self.state.get(fleet)

    def probe_all(self):
        pricing, seen = self.app.cfg.pricing(), {}
        for fleet in self.app.market.fleets(pricing):
            hub = self.app.cfg.upstream(fleet["id"])
            if fleet["serving"] and hub:
                if hub not in seen:
                    seen[hub] = probe_health(hub)
                self.observe(fleet["id"], seen[hub][0], self.app.market.now(), seen[hub][1])

    def run(self):
        while True:
            time.sleep(self.app.cfg.health_seconds)
            try:
                self.probe_all()
            except Exception as error:
                log(f"uptime probe: {error}")

    def clear(self):
        with self.lock:
            for name in self.months():
                os.remove(os.path.join(self.folder, name))
            self.recent = []
            self.state.keep(lambda record: False)

    def prune(self, now):
        cutoff = market.months_later(now, -12 * UPTIME_YEARS).strftime("%Y-%m")
        with self.lock:
            for name in self.months():
                if name[:7] < cutoff:
                    os.remove(os.path.join(self.folder, name))
            shown = market.stamp(now - timedelta(days=UPTIME_DAYS_SHOWN + 1))
            self.recent = [span for span in self.recent if span["end"] >= shown]


def text_field(value, label, size):
    if not isinstance(value, str) or not value.strip() or len(value) > size or any(ord(c) < 32 for c in value):
        raise Refusal(f"Enter the {label}.")
    return value.strip()


def optional_text(value, size):
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or len(value) > size or any(ord(c) < 32 and c not in "\n\t" for c in value):
        raise Refusal("A field is too long or holds control characters.")
    return value.strip()


def email_field(value):
    if not isinstance(value, str) or len(value) > 254 or not EMAIL.match(value.strip()):
        raise Refusal("Enter a valid email address.")
    return value.strip().lower()


def won_field(value):
    text = str(value) if isinstance(value, (int, str)) and not isinstance(value, bool) else ""
    if not re.fullmatch(r"[0-9]{1,12}", text.strip()):
        raise Refusal("Enter an amount in Korean won, in whole won.")
    return Decimal(text.strip())


def quantity_field(value, least):
    text = str(value).strip() if isinstance(value, (int, float, str)) and not isinstance(value, bool) else ""
    if not re.fullmatch(r"[0-9]{1,9}(\.[0-9]{1,6})?", text) or Decimal(text) < least:
        raise Refusal(f"Buy at least {least} HC, to at most six decimal places.")
    return Decimal(text)


def usd_field(value):
    text = str(value).strip() if isinstance(value, (int, str)) and not isinstance(value, bool) else ""
    if not re.fullmatch(r"[0-9]{1,9}(\.[0-9]{1,2})?", text) or not Decimal(text) > 0:
        raise Refusal("Enter an amount in US dollars, more than 0 and to the cent.")
    return money(text)


def coin_field(value):
    if value not in COINS:
        raise Refusal("Choose USDC or USDT and the network you will pay on.")
    return value


def statement(cfg, company, terms):
    return (f"For {company}, I accept the SparkPipe Customer Agreement and confirm that {company} buys and uses SparkPipe for business, professional, commercial or independent research and development purposes, not as a household consumer. "
            f"Customer Agreement, Korean text, which prevails: SHA-256 {terms['ko']} ({cfg.public}/terms/agreement-ko.md). "
            f"English text: SHA-256 {terms['en']} ({cfg.public}/terms/agreement-en.md).")


def invoice_body(cfg, quote):
    name = f"{quote['hc']} SparkPipe hour-credits"
    back = f"{cfg.public}/account.html?order={quote['order']}"
    return {
        "currency": USD,
        "items": [{"customId": quote["order"], "name": name, "description": f"Customer Agreement; {quote['price']} USD each, usable for 12 months", "quantity": {"value": 1, "type": "quantity"}, "amount": quote["net"], "tax": quote["vat"]}],
        "amount": {"breakdown": {"subtotal": quote["net"], "taxTotal": quote["vat"]}, "total": quote["total"]},
        "invoiceId": quote["order"],
        "description": name,
        "buyer": {"companyName": quote["company"], "emailAddress": quote["email"]},
        "notesToRecipient": f"SparkPipe issues the hour-credits to the account {quote['email']} once this payment completes.",
        "customData": {"order": quote["order"]},
        "webhooks": [{"notificationsUrl": cfg.webhook_url, "notifications": EVENTS}],
        "payment": {"paymentCurrency": COINS[quote["coin"]], "refundEmail": quote["email"]},
        "successUrl": back,
        "cancelUrl": back,
    }


def remote_fields(remote):
    return str(remote["status"]).lower(), remote["amount"]["total"], str(remote["currency"]["id"])


def received_of(remote):
    return money(sum((Decimal(str(p["nativeActualAmount"])) for p in remote["payments"]), Decimal(0)))


def covered(order, paid):
    share = min(Decimal(1), paid / Decimal(order["total"]))
    net = (Decimal(order["net"]) * share).quantize(CENT, rounding=ROUND_DOWN)
    return market.text_hc(net / Decimal(order["price"])), text_money(net)


def paid_with(remote):
    return sorted({str(p["paymentCurrencySymbol"]) for p in remote["payments"]})


def tax_note(order, payment):
    if order.get("currency") != KRW:
        return []
    if payment.get("receiptUrl") and not payment["taxInvoice"]["needed"]:
        return ["", f"Your card slip, which is the VAT document for this payment: {payment['receiptUrl']}"]
    if payment["taxInvoice"]["brn"]:
        return ["", f"We issue an electronic tax invoice (세금계산서) for this payment to business registration number {payment['taxInvoice']['brn']}."]
    return ["", "For an electronic tax invoice (세금계산서) for this payment, reply with your Korean business registration number."]


def swift_invoice(quote, now, back):
    bank = quote["bank"]
    return "\n".join([
        f"Invoice {quote['order']}, {now.astimezone(KST).strftime('%Y-%m-%d')}",
        "Seller: HYOR AI Tech Ltd. (SparkPipe), Seoul, Republic of Korea",
        f"Buyer: {quote['company']} <{quote['email']}>",
        f"Buyer's declaration: {quote['declaration']}",
        "",
        f"{quote['hc']} SparkPipe hour-credits at {quote['price']} USD each, usable for 12 months: {quote['net']} USD",
        f"VAT, 0% on SWIFT payments: {quote['vat']} USD",
        f"Total due: {quote['total']} USD",
        "",
        f"Pay {quote['total']} USD by SWIFT bank transfer, from {quote['company']}'s own account outside the Republic of Korea, to:",
        f"Beneficiary: {bank['beneficiary']}",
        f"Account number: {bank['account']}",
        f"Bank: {bank['bank']}",
        f"Bank address: {bank['address']}",
        f"SWIFT/BIC: {bank['bic']}",
        f"Payment reference: {quote['reference']}",
        "",
        f"Write the payment reference {quote['reference']} in the payment details, so that we can match your transfer. Choose charges OUR: your bank's and any intermediary bank's charges are yours, so that the whole {quote['total']} USD arrives (Customer Agreement clause 3.4).",
        "We issue the hour-credits once the whole amount has reached our account. A transfer that arrives short issues nothing until the rest arrives with the same reference.",
        "VAT is 0% on a SWIFT payment from a customer with no place of business in the Republic of Korea (clause 3.3(b)).",
        "",
        f"Follow the order: {back}",
        "",
        "HYOR AI Tech Ltd. (SparkPipe), Seoul. Reply to this email with any question.",
    ])


def declaration(company):
    return f"{company} has no place of business in the Republic of Korea and pays this order by bank transfer in US dollars from its own account outside the Republic of Korea (Customer Agreement clause 3.3(b))."


def receipt(cfg, order, issued, payment):
    currency, processor = order.get("currency", "USD"), order.get("processor")
    coin = ", ".join(sorted({symbol.split(".")[0] for symbol in payment["paidWith"]}))
    way = f"by {METHOD_EN.get(payment.get('method'), payment.get('method'))} through Toss Payments" if processor == "toss" else "by SWIFT bank transfer" if processor == "swift" else f"in {coin}"
    what = f"{issued['hc']} hour-credits, usable until {market.stamp_time(issued['expires']).astimezone(KST).strftime('%Y-%m-%d')}"
    extra = Decimal(payment["received"]) - Decimal(order["total"]) if processor == "swift" else Decimal(0)
    if issued.get("short"):
        lines = [f"Order {order['order']} was for {order['total']} USD; {payment['received']} USD arrived. We issued what it covers: {what}.", "", f"Received in {coin}: {payment['received']} USD, VAT included"]
    else:
        lines = [f"We received your payment for order {order['order']}: {what}.", "", f"Subtotal: {amount_text(order['net'], currency)}", f"VAT{', 0% on SWIFT payments' if processor == 'swift' else ''}: {amount_text(order['vat'], currency)}",
                 f"Total paid {way}: {amount_text(order['total'], currency)}"] + tax_note(order, payment)
    if extra > 0:
        lines += ["", f"We received {payment['received']} USD: the {text_money(extra)} USD above the total goes back by SWIFT to the account it came from."]
    return "\n".join(lines + ["", f"They are in your account: {cfg.public}/market.html", "", "HYOR AI Tech Ltd. (SparkPipe), Seoul. Reply to this email with any question."])


class App:
    def __init__(self, cfg, book):
        self.cfg = cfg
        self.book = book
        self.key = cfg.secret_file("session_key")
        self.lock = threading.Lock()
        self.codes = {}
        self.hits = collections.defaultdict(collections.deque)
        self.jobs = queue.Queue()
        self.mails = queue.Queue()
        self.market = market.Market(self, cfg.sandbox, lambda: clock())
        self.usage = Usage(cfg.path("requests"))
        self.audits = Usage(cfg.path("audit"), "id")
        self.notices = Journal(cfg.path("notices"), NOTICE_DAYS)
        self.hooks = Records(cfg.path("webhooks.json"))
        self.sessions = Records(cfg.path("sessions.json"))
        self.guard = Guard()
        self.notifier = Notifier(self)
        self.uptime = Uptime(self)

    def throttle(self, kind, who):
        count, window = LIMITS[kind]
        self.limit((kind, who), count, window, "Too many requests. Wait a few minutes and try again.")

    def limit(self, key, count, window, text):
        now = time.monotonic()
        with self.lock:
            hits = self.hits[key]
            while hits and hits[0] < now - window:
                hits.popleft()
            if len(hits) >= count:
                raise Refusal(text, 429, "busy")
            hits.append(now)

    def throttle_gateway(self, account):
        holder, rented = self.market.gateway_share(account)
        if not holder:
            return self.throttle("gateway", account)
        window = LIMITS["gateway"][1]
        rented = max(1, rented)
        share = max(1, self.cfg.gateway_capacity // rented)
        fleets = "1 Fleet is" if rented == 1 else f"{rented} Fleets are"
        self.limit(("gateway-holder", account), share, window,
                   f"This account used its share of the gateway: {share} requests per {window // 60} minutes while {fleets} rented. Wait a minute and try again.")

    def mail(self, to, subject, text):
        if not self.cfg.sandbox:
            self.mails.put((to, subject, text))

    def who(self, account):
        record = self.book.accounts.get(account, {})
        return f"{record.get('company')} <{record.get('email')}> ({account})"

    def reset(self):
        if not self.cfg.sandbox:
            raise Refusal("Not found.", 404)
        with self.book.lock:
            os.replace(self.book.path, self.book.path + "." + clock().strftime("%Y%m%d%H%M%S"))
            self.book = Book(self.book.path)
            self.hooks.keep(lambda record: False)
            self.sessions.keep(lambda record: False)
            self.uptime.clear()
        with self.lock:
            self.codes.clear()
        return {"reset": True}

    def deliver(self):
        while True:
            item = self.mails.get()
            try:
                if item is None:
                    return
                send_mail(self.cfg, *item)
            except Exception as error:
                log(f"mail to {item[0]}: {error}")
            finally:
                self.mails.task_done()

    def current_terms(self):
        return {name: self.cfg.terms(source) for name, source in TERMS.items()}

    def accepted(self, account, terms):
        found = self.book.accounts[account]["accepted"]
        return bool(found) and all(found[name] == terms[name] for name in TERMS)

    def accept(self, account, ip, agent):
        terms = self.current_terms()
        company = self.book.accounts[account]["company"]
        self.book.append("acceptance", dict(terms, account=account, statement=statement(self.cfg, company, terms), ip=ip, agent=agent))

    def terms_current(self, account):
        return account in self.book.accounts and self.accepted(account, self.current_terms())

    def require_terms(self, account, body, ip, agent):
        if not self.accepted(account, self.current_terms()):
            if body.get("accept") is not True:
                raise Refusal("The Customer Agreement has changed since you accepted it. Accept the current text to continue.", 409, "terms")
            self.accept(account, ip, agent)

    def resolve(self, subject):
        if subject in self.book.accounts:
            return subject, subject, "owner"
        found = self.book.members.get(subject)
        if found and not found["removed"] and found["account"] in self.book.accounts:
            return found["account"], subject, found["role"]
        return None

    def actor(self, headers):
        token = credential(headers)
        if token.startswith(BEARER):
            return self.bearer_actor(token)
        jar = {}
        for part in (headers.get("Cookie") or "").split(";"):
            name, _, value = part.strip().partition("=")
            jar[name] = value
        return self.resolve(session_account(self.key, jar.get(COOKIE), int(clock().timestamp())))

    def bearer_actor(self, token):
        record = self.sessions.get(market.key_digest(token))
        if not record or stamp_time(record["expires"]) <= clock():
            return None
        found = self.resolve(record.get("member") or record["account"])
        return found if found and found[0] == record["account"] else None

    def open_bearer(self, account, member, device):
        token = BEARER + secrets.token_urlsafe(32)
        now = clock()
        expires = market.stamp(now + timedelta(seconds=BEARER_SECONDS))
        self.sessions.put(market.key_digest(token), {"session": new_id("SE"), "account": account, "member": member, "device": device, "created": market.stamp(now), "expires": expires})
        return token, expires

    def sign_out(self, headers):
        token, found = credential(headers), self.actor(headers)
        if token.startswith(BEARER):
            self.sessions.drop(market.key_digest(token))
        if found:
            self.audit(found[0], found[1], "session.sign_out", {"via": "bearer" if token.startswith(BEARER) else "cookie"})
        return {"ok": True}

    def revoke_session(self, actor, found):
        account, member, role = actor
        ident, _ = self.sessions.find(lambda record: record["session"] == found and record["account"] == account and (role == "owner" or record.get("member", account) == member))
        if not ident:
            raise Refusal("No active session with that id on this account.", 404)
        self.sessions.drop(ident)
        return {"id": found, "revoked": True}

    def require_role(self, actor, roles, what):
        if actor[2] not in roles:
            raise Refusal(f"{what} is for the account's {' or '.join(roles)}; your role is {actor[2]}.", 403, "role")

    def audit(self, account, member, kind, data, at=None):
        now = at or self.market.now()
        self.audits.add([{"id": new_id("AU"), "account": account, "at": market.stamp(now), "member": member, "type": kind, "data": data}])

    def sessions_of(self, actor):
        now, (account, member, role) = clock(), actor
        return sorted(({"id": r["session"], "member": r.get("member", account), "device": r["device"], "created": r["created"], "expires": r["expires"]} for r in self.sessions.values()
                       if r["account"] == account and (role == "owner" or r.get("member", account) == member) and stamp_time(r["expires"]) > now), key=lambda r: (r["created"], r["id"]))

    def principal(self, headers, ip, scope):
        token = credential(headers)
        if token.startswith(BEARER):
            found, record, via = self.bearer_actor(token), None, "bearer"
            if not found:
                raise Refusal("This session has ended: sign in again.", 401, "invalid_key")
        elif token:
            record, via = self.market.authorize(token, scope, ip), "key"
            found = self.resolve(record["member"])
            if not found or found[0] != record["account"]:
                raise Refusal("This key's member has left the account.", 401, "invalid_key")
        else:
            found, record, via = self.actor(headers), None, "cookie"
            if not found:
                raise Refusal("Send a SparkPipe API key as Authorization: Bearer sk-hc-... or X-Api-Key.", 401, "invalid_key")
        if scope not in market.ROLE_SCOPES[found[2]]:
            raise Refusal(f"A {found[2]} of the account does not have the {scope} scope.", 403, "invalid_key")
        return found[0], record, via, found[1], found[2]

    def notified(self, notice):
        account = self.book.accounts.get(notice["account"])
        for target in notice["targets"]:
            if target != "mail":
                self.notifier.schedule(notice, target)
            elif account:
                self.mail(account["email"], notice["summary"], notice["text"] + FOOTER)

    def hooks_of(self, account):
        return sorted((w for w in self.hooks.values() if w["account"] == account), key=lambda w: w["webhook"])

    def hook_view(self, hook):
        found = {"id": hook["webhook"], "events": hook["events"], "slack": hook["slack"], "created": hook["now"]}
        if hook["kind"] == "email":
            return dict(found, email=self.book.accounts[hook["account"]]["email"], always=list(market.REQUIRED_EMAIL))
        return dict(found, url=hook["target"], hint=hook["hint"])

    def webhooks(self, account):
        mine = [self.hook_view(w) for w in self.hooks_of(account)]
        email = [] if any("email" in w for w in mine) else [{"email": self.book.accounts[account]["email"], "events": list(market.EMAIL_EVENTS), "always": list(market.REQUIRED_EMAIL), "default": True}]
        return {"object": "list", "data": mine + email, "events": list(market.EVENTS), "emailAlways": list(market.REQUIRED_EMAIL)}

    def webhook_put(self, account, body):
        events = body.get("events")
        if not isinstance(events, list) or any(event not in market.EVENTS and event != "*" for event in events) or (not events and not body.get("email")):
            raise Refusal("Give events as a list of " + ", ".join(market.EVENTS) + ", or [\"*\"] for all of them.", 400, "invalid_request")
        if not isinstance(body.get("slack", False), bool) or (body.get("url") is None) == (body.get("email") is None):
            raise Refusal("Give either url, for a signed webhook, or email, with events, and slack as true or false.", 400, "invalid_request")
        record, secret, target = self.book.accounts[account], None, None
        if body.get("email") is not None:
            kind = "email"
            if email_field(body.get("email")) != record["email"]:
                raise Refusal(f"Notices by email go to the account's own address, {record['email']}.", 400, "invalid_request")
        else:
            kind, target = "url", body.get("url")
            if not isinstance(target, str) or len(target) > 2000 or urllib.parse.urlsplit(target).scheme not in WEBHOOK_SCHEMES or not urllib.parse.urlsplit(target).hostname or "@" in urllib.parse.urlsplit(target).netloc:
                raise Refusal("Give url as an https:// address of at most 2000 characters, without a user name or password.", 400, "invalid_request")
            webhook_address(target, urllib.parse.urlsplit(self.cfg.public).hostname)
            secret = body.get("secret")
            if secret is None:
                secret = "whsec_" + secrets.token_urlsafe(32)
            elif not isinstance(secret, str) or not 16 <= len(secret) <= 200 or not secret.isprintable():
                raise Refusal("Give secret as 16 to 200 printable characters, or leave it out and we make one.", 400, "invalid_request")
        with self.hooks.lock:
            mine = self.hooks_of(account)
            same = next((w for w in mine if w["kind"] == kind and w.get("target") == target), None)
            if not same and len(mine) >= WEBHOOKS_PER_ACCOUNT:
                raise Refusal(f"An account has at most {WEBHOOKS_PER_ACCOUNT} webhooks: delete one first.", 409, "too_many")
            ident = same["webhook"] if same else new_id("WH")
            hook = {"webhook": ident, "account": account, "kind": kind, "events": sorted(set(events)), "slack": body.get("slack", False), "now": market.stamp(self.market.now())}
            if secret:
                hook.update(target=target, secret=secret, hint=secret[-4:])
            self.hooks.put(ident, hook)
        found = self.hook_view(hook)
        return dict(found, secret=secret) if secret and body.get("secret") is None else found

    def webhook_delete(self, account, ident):
        with self.hooks.lock:
            hook = self.hooks.get(ident)
            if not hook or hook["account"] != account:
                raise Refusal("No webhook with that id on this account.", 404)
            self.hooks.drop(ident)
        return {"id": ident, "deleted": True}

    def usage_record(self, item, now):
        if not isinstance(item, dict):
            raise Refusal("A record is a JSON object.", 400, "invalid_request")
        found = {}
        for name, size in RECORD_TEXT.items():
            value = item.get(name)
            if value is not None:
                found[name] = market.short_text(value, name, size)
        for name in RECORD_NUMBERS:
            value = item.get(name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise Refusal(f"Give {name} as a number of at least 0.", 400, "invalid_request")
                found[name] = value
        if any(name not in found for name in ("request", "account", "fleet", "at")) or found["account"] not in self.book.accounts:
            raise Refusal("Every record names its request, a known account, its fleet and when it ran (at).", 400, "invalid_request")
        moment = market.stamp_field(found["at"], "time the request ran")
        if found["at"] <= self.book.erased.get(found["account"], ""):
            raise Refusal("The account had the records of its requests up to this time deleted (DPA clause 9.3).", 409, "erased")
        if not now - timedelta(days=RECORD_DAYS) <= moment <= now + timedelta(minutes=5):
            raise Refusal("A record is kept for a year: it must be from the last year, and not from the future.", 400, "invalid_request")
        owner = self.book.market.keys.get(found.get("key"))
        if "member" not in found and owner and owner["account"] == found["account"]:
            found["member"] = owner["member"]
        return found

    def record_usage(self, body):
        items = body.get("records") if isinstance(body.get("records"), list) else [body] if "request" in body else None
        if not items or len(items) > RECORDS_PER_REPORT:
            raise Refusal(f"Send records as a list of 1 to {RECORDS_PER_REPORT}.", 400, "invalid_request")
        now, found, refused = clock(), [], []
        for index, item in enumerate(items):
            try:
                found.append(self.usage_record(item, now))
            except Refusal as error:
                refused.append({"index": index, "error": str(error)})
        added = self.usage.add(found)
        return {"accepted": added, "duplicates": len(found) - added, "refused": refused}

    def requests(self, actor, query):
        account, member, role = actor
        if role == "member":
            query = dict(query, member=[member])
        return self.page(self.usage, account, query, ("key", "fleet", "model", "session", "member"), usage_totals)

    def audit_list(self, account, query):
        return self.page(self.audits, account, query, ("member", "type"), None)

    def page(self, store, account, query, names, summed):
        pick = lambda name: query.get(name, [None])[0]
        until = time_field(pick("until"), "until") or clock() + timedelta(minutes=5)
        since = time_field(pick("since"), "since") or until - timedelta(days=7)
        if not since < until or until - since > timedelta(days=RECORD_DAYS):
            raise Refusal("Give since before until, at most a year apart.", 400, "invalid_request")
        limit = pick("limit") or "1000"
        if not limit.isdigit() or not 1 <= int(limit) <= 10000:
            raise Refusal("Give limit from 1 to 10000.", 400, "invalid_request")
        after, page, more = pick("after"), [], [False]
        cursor = lambda record: f"{record['at']}|{record[store.ident]}"

        def walk():
            for record in store.query(account, since, until, {name: pick(name) for name in names if pick(name)}):
                if not after or cursor(record) > after:
                    if len(page) < int(limit):
                        page.append(record)
                    else:
                        more[0] = True
                yield record

        found = {"object": "list", "since": market.stamp(since), "until": market.stamp(until)}
        if summed:
            found["totals"] = summed(walk())
        else:
            collections.deque(walk(), maxlen=0)
        return dict(found, data=page, has_more=more[0], next=cursor(page[-1]) if more[0] else None)

    def hub_audit(self, body):
        items = body.get("events") if isinstance(body.get("events"), list) else None
        if not items or len(items) > RECORDS_PER_REPORT:
            raise Refusal(f"Send events as a list of 1 to {RECORDS_PER_REPORT}.", 400, "invalid_request")
        now, found, refused = clock(), [], []
        for index, item in enumerate(items):
            try:
                if not isinstance(item, dict) or not isinstance(item.get("type"), str) or not HUB_AUDIT.fullmatch(item["type"]):
                    raise Refusal("Give each event a type such as models.switched, cache.wiped, storage.read, fetch.started or run.started.", 400, "invalid_request")
                entry = {name: market.short_text(item[name], name, 128) for name in ("id", "account", "member", "fleet", "at") if item.get(name) is not None}
                data = item.get("data", {})
                if any(name not in entry for name in ("id", "account", "at")) or entry["account"] not in self.book.accounts or not isinstance(data, dict) or len(json.dumps(data)) > AUDIT_LIMIT:
                    raise Refusal(f"Every event names its id, a known account and when it happened (at), with data as an object of at most {AUDIT_LIMIT} bytes.", 400, "invalid_request")
                moment = market.stamp_field(entry["at"], "time of the event")
                if not now - timedelta(days=RECORD_DAYS) <= moment <= now + timedelta(minutes=5):
                    raise Refusal("An audit event is kept for a year: it must be from the last year, and not from the future.", 400, "invalid_request")
                found.append(dict(entry, id="HB" + entry["id"], type=item["type"], data=data))
            except Refusal as error:
                refused.append({"index": index, "error": str(error)})
        added = self.audits.add(found)
        return {"accepted": added, "duplicates": len(found) - added, "refused": refused}

    def hub_events(self, body):
        items = body.get("events") if isinstance(body.get("events"), list) else [body]
        if not 1 <= len(items) <= 1000:
            raise Refusal("Send 1 to 1000 events.", 400, "invalid_request")
        results = []
        for item in items:
            try:
                results.append(self.market.hub_event(item))
            except Refusal as error:
                results.append({"id": item.get("id") if isinstance(item, dict) else None, "error": str(error), "reason": error.reason})
        return {"events": results}

    def account_call(self, action, actor, match, query, body):
        account, member, role = actor
        calls = {
            "requests": lambda: self.requests(actor, query),
            "requests_erase": lambda: self.erase_records(account, member, "the owner's request through the API"),
            "keys": lambda: {"object": "list", "data": [self.market.key_view(r) for r in self.book.market.keys.values() if r["account"] == account and (role != "member" or r["member"] == member)]},
            "key_create": lambda: self.market.create_key(account, body, member, role),
            "key_revoke": lambda: self.market.revoke_key(account, match.group(1), member, role),
            "key_rotate": lambda: self.market.rotate_key(account, match.group(1), member, role),
            "webhooks": lambda: self.webhooks(account),
            "webhook_put": lambda: self.webhook_put(account, body),
            "webhook_delete": lambda: self.webhook_delete(account, match.group(1)),
            "members": lambda: self.members(account),
            "member_put": lambda: self.member_put(actor, body),
            "member_delete": lambda: self.member_delete(actor, match.group(1)),
            "audit": lambda: self.audit_list(account, query),
        }
        if action in ACCOUNT_ROLES:
            self.require_role(actor, *ACCOUNT_ROLES[action])
        found = calls[action]()
        if action in AUDITED:
            self.audit(account, member, AUDITED[action][0], {name: found.get(name) for name in AUDITED[action][1] if name in found})
        return found

    def need_actor(self, headers):
        found = self.actor(headers)
        if not found:
            raise Refusal("Sign in to continue.", 401)
        return found

    def need_owner(self, headers, roles, what):
        found = self.need_actor(headers)
        self.require_role(found, roles, what)
        return found[0]

    def erase_records(self, account, by, note):
        now = self.market.now()
        with self.book.lock:
            count = self.usage.erase(account)
            line = self.book.append("records_erased", {"account": account, "through": market.stamp(now), "records": count, "by": by, "note": note[:200], "now": market.stamp(now)})
        self.audit(account, by, "records.erased", {"records": count, "through": line["data"]["through"]})
        self.mail(self.cfg.notify, f"Erase the hub's request records of {account} within 30 days",
                  f"{self.who(account)} had the records of its requests deleted on the VPS ({note[:200]}): {count} records up to {line['data']['through']}. "
                  f"DPA clause 9.3 gives us 30 days for the hub's records too. On each hub, as the chat layer's user: python3 chat_frontend.py --config chat_frontend.json --erase-records {account} --note \"{note[:100]}\"")
        self.mail(self.book.accounts[account]["email"], "SparkPipe deleted the records of your requests",
                  f"As you asked ({note[:200]}), we deleted the operational records of the {count} requests your account made up to {market.kst_text(now)} (Data Processing Agreement clause 9.3). "
                  f"Your bookings, payments and HC, which the law has us keep, are unchanged. Records of requests made from now on are kept for a year as before.{FOOTER}")
        return {"erased": count, "through": line["data"]["through"]}

    def members(self, account):
        record, now = self.book.accounts[account], market.stamp(self.market.now())
        owner = {"id": account, "email": record["email"], "role": "owner", "state": "member", "since": record["created"]}
        joined = [{"id": m["member"], "email": m["email"], "role": m["role"], "state": "member", "since": m["created"]} for m in self.book.members.values() if m["account"] == account and not m["removed"]]
        invited = [{"id": i["invite"], "email": i["email"], "role": i["role"], "state": "invited", "invited": i["created"], "expires": i["expires"]} for i in self.book.invites.values() if i["account"] == account and i["state"] == "invited" and i["expires"] > now]
        return {"object": "list", "data": [owner] + joined + invited, "roles": {role: list(market.ROLE_SCOPES[role]) for role in market.ROLES}}

    def pending_invite(self, email, account=None):
        now = market.stamp(self.market.now())
        return next((i for i in self.book.invites.values() if i["email"] == email and i["state"] == "invited" and i["expires"] > now and (account is None or i["account"] == account)), None)

    def member_put(self, actor, body):
        account, member, _ = actor
        email, role = email_field(body.get("email")), body.get("role")
        if role not in ("admin", "member"):
            raise Refusal("Give role as admin or member.", 400, "invalid_request")
        with self.book.lock:
            company = self.book.accounts[account]["company"]
            if email == self.book.accounts[account]["email"]:
                raise Refusal("That is the owner's email; the owner's role does not change.", 400, "invalid_request")
            current = self.book.member_emails.get(email)
            if current and self.book.members[current]["account"] == account:
                if self.book.members[current]["role"] != role:
                    self.book.append("member_role", {"account": account, "member": current, "role": role, "by": member})
                    self.audit(account, member, "member.role", {"member": current, "email": email, "role": role})
                return next(m for m in self.members(account)["data"] if m["id"] == current)
            other = self.pending_invite(email)
            if current or email in self.book.emails or (other and other["account"] != account):
                raise Refusal("That email already belongs to a SparkPipe account or has a pending invitation from one.", 409, "email_taken")
            if not other and len(self.members(account)["data"]) > MEMBERS_PER_ACCOUNT:
                raise Refusal(f"An account has at most {MEMBERS_PER_ACCOUNT} members and invitations.", 409, "too_many")
            now = self.market.now()
            invite = {"invite": new_id("IV"), "account": account, "email": email, "role": role, "by": member, "expires": market.stamp(now + timedelta(days=INVITE_DAYS)), "now": market.stamp(now)}
            self.book.append("member_invite", invite)
            self.audit(account, member, "member.invited", {"invite": invite["invite"], "email": email, "role": role})
        self.mail(email, f"{company} invited you to SparkPipe", "\n".join([
            f"{self.who(account)} invited you to its SparkPipe account as {'an admin' if role == 'admin' else 'a member'}.",
            "",
            f"To join, sign in with this email address at {self.cfg.public}/market.html. The invitation expires at {market.kst_text(market.stamp_time(invite['expires']))}.",
            "An admin manages keys, Functions, models and storage; a member makes its own keys, sends requests and runs its own Functions. Bookings, payments and members stay with the account's owner.",
            "If you did not expect this, ignore it: nothing happens until you sign in.",
        ]))
        return next(m for m in self.members(account)["data"] if m["id"] == invite["invite"])

    def member_delete(self, actor, ident):
        account, member, _ = actor
        with self.book.lock:
            found = self.book.members.get(ident)
            invite = self.book.invites.get(ident)
            if found and found["account"] == account and not found["removed"]:
                self.book.append("member_remove", {"account": account, "member": ident, "by": member})
                for record in [r for r in self.book.market.keys.values() if r["member"] == ident and not r["revoked"]]:
                    self.book.append("api_key_revoke", {"key": record["key"]})
                self.sessions.keep(lambda record: record.get("member") != ident)
                self.audit(account, member, "member.removed", {"member": ident, "email": found["email"]})
            elif invite and invite["account"] == account and invite["state"] == "invited":
                self.book.append("member_remove", {"account": account, "invite": ident, "by": member})
                self.audit(account, member, "member.invite_cancelled", {"invite": ident, "email": invite["email"]})
            else:
                raise Refusal("No member or invitation with that id on this account.", 404)
        return {"id": ident, "removed": True}

    def send_code(self, body, ip):
        self.throttle("code", ip)
        email = email_field(body.get("email"))
        self.throttle("code-email", email)
        code = f"{secrets.randbelow(1000000):06d}"
        with self.lock:
            now = time.monotonic()
            for key in [k for k, v in self.codes.items() if v["expires"] < now]:
                del self.codes[key]
            self.codes[email] = {"hash": code_digest(email, code), "expires": now + CODE_SECONDS, "tries": 0}
        if self.cfg.sandbox:
            return {"sent": False, "code": code}
        try:
            send_mail(self.cfg, email, f"Your SparkPipe code: {code}", "\n".join([
                f"Your SparkPipe sign-in code is {code}.",
                "",
                "Enter it on sparkpipe.ai to continue. It works once and expires in 10 minutes.",
                "If you did not ask for it, ignore this email: nothing happens without the code.",
            ]))
        except Exception as error:
            log(f"code mail to {email}: {error}")
            raise Refusal("We could not send the email. Check the address, or write to hyor@sparkpipe.ai.", 502)
        return {"sent": True}

    def verify(self, body, ip, agent):
        summary, subject = self.check_code(body, ip, agent)
        account, member, _ = self.resolve(subject)
        if body.get("bearer") is True:
            device = optional_text(body.get("device"), 100)
            session, expires = self.open_bearer(account, member, device)
            self.audit(account, member, "session.sign_in", {"via": "bearer", "device": device})
            return dict(summary, session=session, sessionExpires=expires), None
        self.audit(account, member, "session.sign_in", {"via": "cookie"})
        return summary, session_token(self.key, subject, int(clock().timestamp()) + SESSION_SECONDS)

    def check_code(self, body, ip, agent):
        self.throttle("verify", ip)
        email = email_field(body.get("email"))
        code = body.get("code")
        code = code.replace(" ", "") if isinstance(code, str) else ""
        with self.lock:
            entry = self.codes.get(email)
            good = bool(entry) and entry["expires"] >= time.monotonic() and entry["tries"] < CODE_TRIES and CODE.match(code) is not None and hmac.compare_digest(entry["hash"], code_digest(email, code))
            if entry and not good:
                entry["tries"] += 1
            if good:
                del self.codes[email]
        if not good:
            raise Refusal("That code is wrong or has expired. Ask for a new one.", 403)
        with self.book.lock:
            account, member = self.book.emails.get(email), self.book.member_emails.get(email)
            invite = None if account or member else self.pending_invite(email)
            if member:
                return self.summary(self.resolve(member)), member
            if invite:
                member = new_id("MB")
                self.book.append("member", {"member": member, "account": invite["account"], "email": email, "role": invite["role"], "invite": invite["invite"]})
                self.audit(invite["account"], member, "member.joined", {"member": member, "email": email, "role": invite["role"]})
                self.mail(self.book.accounts[invite["account"]]["email"], f"{email} joined your SparkPipe account", f"{email} accepted your invitation and joined {self.book.accounts[invite['account']]['company']} as {invite['role']}. Members and their roles: {self.cfg.public}/market.html")
                return self.summary(self.resolve(member)), member
            if not account:
                if not body.get("company"):
                    raise Refusal("No account uses this email yet. Add your company's name to open one.", 404, "new")
                company = text_field(body.get("company"), "company's legal name, or your full name", 200)
                if body.get("accept") is not True:
                    raise Refusal("Accept the Customer Agreement to open an account.", 409, "terms")
                account = new_id("AC")
                self.book.append("account", {"account": account, "email": email, "company": company, "ip": ip, "agent": agent})
                self.accept(account, ip, agent)
                self.mail(self.cfg.notify, f"New SparkPipe account: {company}", f"{company} <{email}> opened account {account}.")
                log(f"account {account} opened")
            elif body.get("accept") is True and not self.accepted(account, self.current_terms()):
                self.accept(account, ip, agent)
        return self.summary((account, account, "owner")), account

    def own_email(self, account, member):
        return self.book.accounts[account]["email"] if member == account else self.book.members[member]["email"]

    def summary(self, actor):
        account, member, role = actor
        with self.book.lock:
            record = self.book.accounts[account]
            mine = [o for o in reversed(list(self.book.orders.values())) if o["account"] == account] if role == "owner" else []
            return {
                "email": self.own_email(account, member),
                "company": record["company"],
                "member": member,
                "role": role,
                "since": record["created"][:10],
                "termsCurrent": self.accepted(account, self.current_terms()),
                "hc": market.text_hc(self.book.market.balance(account, market.stamp(self.market.now()))),
                "locked": market.text_hc(self.book.market.locked(account)),
                "orders": [self.shown(o) for o in mine[:50]],
            }

    def credit_terms(self, pricing, body, currency, rate=None):
        offer = pricing["hc"]
        count = quantity_field(body.get("hc"), offer["min"])
        step = max((p for p in offer["prices"] if p["from"] <= count), key=lambda p: p["from"])
        price = won(step["krw"]) if currency == KRW else money(step["usd"])
        rate = Decimal(str(pricing["vatRate"] if rate is None else rate))
        net = (price * count).quantize(WON if currency == KRW else CENT, rounding=ROUND_HALF_UP)
        vat = (net * rate).quantize(WON if currency == KRW else CENT, rounding=ROUND_HALF_UP)
        shown = text_won if currency == KRW else text_money
        return {"product": "credits", "currency": "KRW" if currency == KRW else "USD", "hc": format(count.normalize(), "f"), "price": shown(price), "net": shown(net), "vatRate": format(rate.normalize(), "f"), "vat": shown(vat), "total": shown(net + vat)}

    def priced(self, pricing, body, account, currency, ip, agent, rate=None):
        if body.get("product") != "credits":
            raise Refusal("Choose hour-credits: product credits with the number of HC.")
        quote = self.credit_terms(pricing, body, currency, rate)
        self.require_terms(account, body, ip, agent)
        return quote

    def buy(self, body, account, ip, agent):
        if self.cfg.sandbox and body.get("processor") != "toss":
            return self.market.buy(account, quantity_field(body.get("hc"), self.cfg.pricing()["hc"]["min"]))
        return self.order(dict(body, product="credits"), account, ip, agent)

    def order(self, body, account, ip, agent):
        self.throttle("order", ip)
        processor = body.get("processor", "coinpayments")
        if processor not in PROCESSORS:
            raise Refusal("Pay by card or bank transfer through Toss Payments, in USDC or USDT, or in US dollars by SWIFT.")
        if processor == "toss":
            return self.toss_order(body, account, ip, agent)
        if self.cfg.sandbox:
            raise Refusal("The sandbox sells HC instantly at no charge, or through Toss Payments in test mode.", 404)
        if processor == "swift":
            return self.swift_order(body, account, ip, agent)
        coin = coin_field(body.get("coin"))
        pricing = self.cfg.pricing()
        with self.book.lock:
            record = self.book.accounts[account]
            quote = self.priced(pricing, body, account, "USD", ip, agent)
            quote.update(order=new_id("SP"), account=account, email=record["email"], company=record["company"], processor="coinpayments", coin=coin)
            self.book.append("order", dict(quote, ip=ip, agent=agent))
        try:
            invoice = call(self.cfg, "POST", "/merchant/invoices", invoice_body(self.cfg, quote))["invoices"][0]
            link = invoice.get("checkoutLink") or invoice["link"]
        except Exception as error:
            self.book.append("invoice_failed", {"order": quote["order"], "error": str(error)[:500]})
            log(f"order {quote['order']}: invoice failed: {error}")
            raise Refusal("The payment processor did not create the invoice. Nothing was charged; try again in a minute.", 502)
        self.book.append("invoice", {"order": quote["order"], "invoice": str(invoice["id"]), "link": link})
        log(f"order {quote['order']}: invoice {invoice['id']} for {quote['total']} USD in {coin}")
        return {"order": quote["order"], "checkout": link}

    def shown(self, order):
        shown = {key: order.get(key) for key in ("order", "product", "hc", "price", "net", "vatRate", "vat", "total", "coin", "state", "issued", "created", "paid", "received", "processor", "method", "deposit", "refunded", "reason", "expires", "reference", "bank")}
        shown["currency"] = order.get("currency", "USD")
        shown["issuedHC"] = order.get("issued_hc")
        shown["checkout"] = order["link"] if order["state"] in ("unpaid", "pending") else None
        return shown

    def status(self, order_id):
        with self.book.lock:
            order = self.book.orders.get(order_id)
            if not order:
                raise Refusal("No such order.", 404)
            return self.shown(order)

    def market_state(self, actor):
        state = self.market.public()
        state["account"] = None
        if actor:
            account, member, role = actor
            record, view = self.book.accounts[account], self.market.view(account)
            if role == "member":
                view["keys"] = [k for k in view["keys"] if k["member"] == member]
            state["account"] = dict(view, email=self.own_email(account, member), company=record["company"], member=member, role=role, termsCurrent=self.accepted(account, self.current_terms()), sessions=self.sessions_of(actor))
        return state

    def market_action(self, path, body, actor, ip, agent):
        self.throttle("market", ip)
        account, member, role = actor
        if path in OWNER_ACTIONS:
            self.require_role(actor, ("owner",), "Booking, the waitlist, auto-extend and buying HC")
        if path in SPENDING:
            with self.book.lock:
                self.require_terms(account, body, ip, agent)
        found = self.market_call(path, body, actor, ip, agent)
        if path in AUDITED:
            self.audit(account, member, AUDITED[path][0], {name: found.get(name) for name in AUDITED[path][1] if name in found})
        return found

    def market_call(self, path, body, actor, ip, agent):
        account, member, role = actor
        actions = {
            "/market/buy": lambda: self.buy(body, account, ip, agent),
            "/market/quote-weeks": lambda: self.market.quote_weeks(account, body),
            "/market/quote-hours": lambda: self.market.quote_hours(account, body),
            "/market/book-hours": lambda: self.market.book_hours(account, body),
            "/market/book-weeks": lambda: self.market.book_weeks(account, body),
            "/market/waitlist": lambda: self.market.join(account, body),
            "/market/waitlist/leave": lambda: self.market.leave(account),
            "/market/auto-extend": lambda: self.market.auto_extend(account, body),
            "/market/keys": lambda: self.market.create_key(account, body, member, role),
            "/market/keys/revoke": lambda: self.market.revoke_key(account, body.get("id"), member, role),
            "/market/keys/rotate": lambda: self.market.rotate_key(account, body.get("id"), member, role),
            "/market/sessions/revoke": lambda: self.revoke_session(actor, body.get("id")),
        }
        sandbox = {
            "/market/sandbox/clock": lambda: self.market.move_clock(market.count_field(body.get("hours"), "number of hours", 9600)),
            "/market/sandbox/fleet": lambda: self.market.sandbox_fleet(body),
            "/market/sandbox/reset": self.reset,
        }
        if path in actions:
            return actions[path]()
        if path in sandbox and self.cfg.sandbox:
            return sandbox[path]()
        raise Refusal("Not found.", 404)

    def ledger(self, account):
        lines, mine = [], {account}
        with self.book.lock:
            if os.path.exists(self.book.path):
                with open(self.book.path, encoding="utf-8") as handle:
                    for raw in handle:
                        line = json.loads(raw)
                        data = line["data"]
                        ids = {data[name] for name in LEDGER_IDS if isinstance(data.get(name), str)}
                        if ("account" in data and data["account"] == account) or ("account" not in data and ids & mine):
                            lines.append(line)
                            mine |= ids
            return {"head": self.book.head, "count": self.book.count, "lines": lines}

    def webhook(self, headers, raw):
        if not verified(self.cfg, headers, raw, clock()):
            log("webhook: signature rejected")
            return False
        try:
            event = json.loads(raw)
        except ValueError:
            return True
        invoice = event.get("invoice") if isinstance(event.get("invoice"), dict) else {}
        key = invoice.get("id")
        if key:
            self.jobs.put(str(key))
        return True

    def settle(self, order_id, remote, released=False):
        state, total, currency = remote_fields(remote)
        received = received_of(remote)
        with self.book.lock:
            order = self.book.orders[order_id]
            if order["issued"] or (order["state"] == "mismatch" and not released):
                return None
            due = money(order["total"])
            arrived = text_money(received)
            if state != "completed":
                before = Decimal(order.get("received") or "0")
                moved = (before == 0 and received > 0) or abs(received - before) >= Decimal(1)
                if (state != order["state"] and order["state"] != "completed") or moved:
                    change = {"order": order_id, "invoice": order["invoice"], "state": state}
                    if received:
                        change["received"] = arrived
                    self.book.append("invoice_state", change)
                return None
            matches = currency == USD and money(total) == due
            full = matches and received + CENT >= due
            short = matches and received + CENT < due
            if not full and not short and not released:
                self.book.append("invoice_state", {"order": order_id, "invoice": order["invoice"], "state": "mismatch", "total": str(total), "currency": currency, "received": arrived})
                log(f"order {order_id}: completed invoice does not match the order ({total} {currency}, {arrived} received); HC not issued")
                self.mail(self.cfg.notify, f"Payment to review: {order_id}", f"Order {order_id} for {order['total']} USD completed at CoinPayments as {total} {currency} with {arrived} USD received. No HC was issued. If it is right, `sparkpipe_pay.py release {order_id}` issues the HC the amount received covers.")
                return None
            credits, net = (order["hc"], order["net"]) if full else covered(order, received)
            if not Decimal(credits) > 0:
                log(f"order {order_id}: {arrived} USD received covers no HC; refund it in CoinPayments")
                self.mail(self.cfg.notify, f"Payment to review: {order_id}", f"Order {order_id} completed at CoinPayments with {arrived} USD received, which covers no HC. Refund it in CoinPayments.")
                return None
            payment = {"order": order_id, "invoice": order["invoice"], "total": order["total"], "received": arrived, "currency": currency, "paidWith": paid_with(remote)}
            return self.issue(order, payment, credits, net, released, not full)

    def issue(self, order, payment, credits, net, released=False, short=False):
        order_id, currency = order["order"], order.get("currency", "USD")
        if order["state"] != "completed":
            self.book.append("payment", payment)
        issued = {"order": order_id, "account": order["account"], "hc": credits, "net": net, "released": released, "short": short}
        self.book.append("hc_issued", issued)
        issued["expires"] = self.market.issue(order["account"], Decimal(credits), order_id)["expires"]
        log(f"order {order_id}: paid, {credits} HC issued to {order['account']}")
        self.mail(order["email"], f"Payment received: SparkPipe order {order_id}", receipt(self.cfg, order, issued, payment))
        received, total = amount_text(payment["received"], currency), amount_text(order["total"], currency)
        subject = f"Paid in part: {order['company']}, {payment['received']} of {total}" if short else f"Paid: {order['company']}, {total}"
        self.mail(self.cfg.notify, subject, f"Order {order_id}: {credits} of {order['hc']} HC for {order['company']} <{order['email']}>, {received} received of {total} in {', '.join(payment['paidWith'])}.")
        return issued

    def swift_config(self):
        return {"enabled": self.cfg.swift}

    def swift_reference(self):
        taken = {o.get("reference") for o in self.book.orders.values()}
        while True:
            found = new_id("SW")[:10]
            if found not in taken:
                return found

    def swift_order(self, body, account, ip, agent):
        if not self.cfg.swift:
            raise Refusal("SWIFT payments are not open yet. Pay through Toss Payments or in USDC or USDT, or write to hyor@sparkpipe.ai.", 503, "swift_closed")
        if body.get("declare") is not True:
            raise Refusal("A USD bank transfer (SWIFT) is for a customer with no place of business in the Republic of Korea, paying from its own account outside the Republic of Korea. Confirm both to pay this way.", 400, "declaration")
        now, pricing = clock(), self.cfg.pricing()
        with self.book.lock:
            record = self.book.accounts[account]
            quote = self.priced(pricing, body, account, "USD", ip, agent, pricing["swiftVatRate"])
            quote.update(order=new_id("SP"), account=account, email=record["email"], company=record["company"], processor="swift", reference=self.swift_reference(), bank=dict(self.cfg.swift_bank), declaration=declaration(record["company"]))
            self.book.append("order", dict(quote, ip=ip, agent=agent))
        self.mail(quote["email"], f"Invoice for SparkPipe order {quote['order']}: {quote['total']} USD by SWIFT", swift_invoice(quote, now, self.back(quote["order"])))
        self.mail(self.cfg.notify, f"SWIFT order: {quote['company']}, {quote['total']} USD", f"Order {quote['order']}, payment reference {quote['reference']}: {quote['hc']} HC for {self.who(account)}, {quote['total']} USD by SWIFT with {quote['vat']} USD VAT. "
                  f"When it reaches the account: sparkpipe_pay.py swift-confirm {quote['reference']} USD")
        log(f"order {quote['order']}: SWIFT for {quote['total']} USD, reference {quote['reference']}")
        return {key: quote[key] for key in ("order", "processor", "reference", "hc", "price", "net", "vatRate", "vat", "total", "currency", "bank")}

    def swift_find(self, key):
        order = self.book.orders.get(key) or next((o for o in self.book.orders.values() if o.get("reference") == key), None)
        if not order or order.get("processor") != "swift":
            raise Refusal("No SWIFT order has that order number or payment reference.")
        return order

    def swift_confirm(self, key, amount, note=""):
        amount = usd_field(amount)
        with self.book.lock:
            order = self.swift_find(key)
            total = money(order["total"])
            if order["issued"] or order["state"] != "unpaid":
                raise Refusal(f"{order['order']} is {order['state']}: nothing recorded. Send what arrived for it back by SWIFT to the account it came from.")
            if amount < total:
                raise Refusal(f"{text_money(amount)} USD is less than the invoice total of {order['total']} USD: nothing recorded and no HC issued. The customer pays the whole amount (clause 3.4): ask for the remaining {text_money(total - amount)} USD, and once it arrives confirm the whole amount received.")
            self.book.append("swift_state", {"order": order["order"], "account": order["account"], "state": "paid", "received": text_money(amount), "note": note})
            payment = {"order": order["order"], "processor": "swift", "reference": order["reference"], "total": order["total"], "received": text_money(amount), "currency": "USD", "paidWith": ["SWIFT"]}
            issued = self.issue(order, payment, order["hc"], order["net"])
        if amount > total:
            self.mail(self.cfg.notify, f"Overpaid by SWIFT: {order['order']}", f"Order {order['order']} (reference {order['reference']}) for {order['total']} USD received {text_money(amount)} USD. Send the {text_money(amount - total)} USD above the total back by SWIFT to the account it came from, "
                      f"then record it: sparkpipe_pay.py refund {order['order']} 3.5 {text_money(amount - total)} REASON")
        return {"order": order["order"], "received": text_money(amount), "issued": issued["hc"], "excess": text_money(amount - total)}

    def swift_cancel(self, key, reason):
        if not reason:
            raise Refusal("Give the reason for cancelling.")
        with self.book.lock:
            order = self.swift_find(key)
            if order["issued"] or order["state"] != "unpaid":
                raise Refusal(f"{order['order']} is {order['state']}: only an unpaid SWIFT order can be cancelled.")
            data = {"order": order["order"], "account": order["account"], "state": "cancelled", "reason": reason}
            self.book.append("swift_state", data)
        self.mail(order["email"], f"Cancelled: SparkPipe order {order['order']}", "\n".join([
            f"We cancelled order {order['order']} (payment reference {order['reference']}): {reason}",
            "",
            "No hour-credits were issued for it. Do not send a transfer for this order; place a new order to buy HC. A transfer that still arrives for it goes back by SWIFT to the account it came from.",
            "",
            "HYOR AI Tech Ltd. (SparkPipe), Seoul. Reply to this email with any question.",
        ]))
        return data

    def toss_config(self):
        return {"enabled": self.cfg.toss, "clientKey": self.cfg.toss_client or None, "variantKey": self.cfg.toss_variant, "test": self.cfg.sandbox}

    def toss_order(self, body, account, ip, agent):
        if not self.cfg.toss:
            raise Refusal("Card and bank payments are not open yet. Pay in USDC or USDT, or write to hyor@sparkpipe.ai.", 503, "cards_closed")
        brn = optional_text(body.get("brn"), 20)
        if brn and not BRN.match(brn):
            raise Refusal("Give the Korean business registration number as 123-45-67890, or leave it empty.")
        now, pricing = clock(), self.cfg.pricing()
        with self.book.lock:
            record = self.book.accounts[account]
            quote = self.priced(pricing, body, account, KRW, ip, agent)
            quote.update(order=new_id("SP"), account=account, email=record["email"], company=record["company"], processor="toss", brn=brn, expires=market.stamp(now + timedelta(seconds=TOSS_SECONDS)))
            self.book.append("order", dict(quote, ip=ip, agent=agent))
        log(f"order {quote['order']}: Toss Payments for {quote['total']} KRW")
        return {"order": quote["order"], "processor": "toss", "toss": self.toss_request(quote)}

    def toss_request(self, quote):
        base = self.cfg.public + self.cfg.prefix + "/toss/"
        found = {
            "clientKey": self.cfg.toss_client, "variantKey": self.cfg.toss_variant, "test": self.cfg.sandbox, "orderId": quote["order"], "orderName": f"SparkPipe hour-credits, {quote['hc']} HC"[:100],
            "amount": int(quote["total"]), "currency": KRW, "customerName": quote["company"][:100], "successUrl": base + "success/" + quote["order"], "failUrl": base + "fail/" + quote["order"],
        }
        if len(quote["email"]) <= 100:
            found["customerEmail"] = quote["email"]
        return found

    def back(self, order_id):
        return f"{self.cfg.public}/account.html?{'sandbox&' if self.cfg.sandbox else ''}order={order_id}"

    def toss_success(self, order_id, query):
        key, given, amount = (query.get(name, [""])[0] for name in ("paymentKey", "orderId", "amount"))
        with self.book.lock:
            order = self.book.orders.get(order_id)
            if not order or order.get("processor") != "toss" or given != order_id or not TOSS_KEY.match(key):
                return self.back(order_id)
            if order["issued"] or order["state"] != "created" or order.get("payment_key") not in (None, key):
                return self.back(order_id)
            if amount != text_won(order["total"]):
                self.book.append("toss_state", {"order": order_id, "state": "failed", "reason": "amount"})
                log(f"order {order_id}: Toss Payments authorised {amount[:20]!r} KRW for an order of {order['total']} KRW; not confirmed")
                return self.back(order_id)
            if not order.get("payment_key"):
                self.book.append("toss_payment", {"order": order_id, "paymentKey": key})
        reason = None
        try:
            remote = toss_call(self.cfg, "POST", "/v1/payments/confirm", {"paymentKey": key, "orderId": order_id, "amount": int(order["total"])}, "confirm-" + order_id)
        except Declined as error:
            log(f"order {order_id}: confirm refused: {error.code}")
            reason = error.code
            try:
                remote = toss_payment(self.cfg, key)
            except (Declined, urllib.error.URLError, OSError, ValueError) as again:
                log(f"order {order_id}: {again}")
                return self.back(order_id)
        except (urllib.error.URLError, OSError, ValueError) as error:
            log(f"order {order_id}: confirm: {error}")
            return self.back(order_id)
        self.toss_apply(order_id, remote, reason)
        return self.back(order_id)

    def toss_fail(self, order_id, query):
        code = query.get("code", [""])[0]
        code = code if re.fullmatch(r"[A-Z0-9_]{1,100}", code) else "PAY_PROCESS_CANCELED"
        with self.book.lock:
            order = self.book.orders.get(order_id)
            if order and order.get("processor") == "toss" and order["state"] == "created" and not order.get("payment_key") and not order["issued"]:
                self.book.append("toss_state", {"order": order_id, "state": "cancelled", "reason": code})
        return self.back(order_id)

    def toss_webhook(self, raw):
        try:
            event = json.loads(raw)
        except ValueError:
            return
        data = event.get("data") if isinstance(event, dict) and isinstance(event.get("data"), dict) else event
        found = data.get("orderId") if isinstance(data, dict) else None
        if isinstance(found, str) and self.book.orders.get(found, {}).get("processor") == "toss":
            self.jobs.put(("toss", found))

    def toss_paid(self, order, remote):
        method, easy = str(remote.get("method") or ""), remote.get("easyPay") or {}
        provider = easy.get("provider")
        cash = method in CASH_METHODS or (method == "간편결제" and Decimal(str(easy.get("amount") or 0)) > 0)
        return {
            "order": order["order"], "processor": "toss", "paymentKey": order["payment_key"], "total": order["total"], "received": text_won(Decimal(str(remote["totalAmount"]))), "currency": KRW,
            "method": method, "paidWith": [method + (f":{provider}" if provider else "")], "approvedAt": remote.get("approvedAt"), "receiptUrl": (remote.get("receipt") or {}).get("url"),
            "taxInvoice": {"needed": cash, "brn": order.get("brn") or None, "supply": order["net"], "vat": order["vat"]},
        }

    def toss_apply(self, order_id, remote, reason=None):
        status = str(remote.get("status") or "")
        with self.book.lock:
            order = self.book.orders[order_id]
            if order["issued"] or remote.get("orderId") != order_id or remote.get("paymentKey") != order.get("payment_key"):
                return None
            matches = remote.get("currency") == KRW and str(remote.get("totalAmount")) == text_won(order["total"])
            if status in ("DONE", "WAITING_FOR_DEPOSIT") and not matches:
                if order.get("reason") == "amount" and order["state"] in ("cancelled", "mismatch"):
                    return None
            elif status == "DONE":
                return self.issue(order, self.toss_paid(order, remote), order["hc"], order["net"])
            elif status == "WAITING_FOR_DEPOSIT":
                if order["state"] == "created":
                    self.toss_waiting(order, remote)
                return None
            elif status in ("ABORTED", "EXPIRED", "CANCELED", "PARTIAL_CANCELED"):
                if order["state"] in TOSS_OPEN:
                    state = "timedout" if status == "EXPIRED" else "cancelled" if status.endswith("CANCELED") else "failed"
                    self.book.append("toss_state", {"order": order_id, "state": state, "reason": reason or str((remote.get("failure") or {}).get("code") or status)[:100]})
                return None
            else:
                return None
        self.toss_refuse(order, status)
        return None

    def toss_waiting(self, order, remote):
        account = remote.get("virtualAccount") or {}
        deposit = {"bank": BANKS.get(str(account.get("bankCode")), str(account.get("bankCode") or "")), "accountNumber": str(account.get("accountNumber") or ""), "dueDate": str(account.get("dueDate") or "")}
        expires = stamp_time(order["created"]) + timedelta(seconds=PENDING_SECONDS)
        try:
            due = datetime.fromisoformat(deposit["dueDate"])
            expires = min(expires, due if due.tzinfo else due.replace(tzinfo=KST))
        except ValueError:
            pass
        self.book.append("toss_state", {"order": order["order"], "state": "waiting", "method": str(remote.get("method") or ""), "deposit": deposit, "expires": market.stamp(expires)})
        self.mail(order["email"], f"Deposit for SparkPipe order {order['order']}", "\n".join([
            f"Deposit {amount_text(order['total'], KRW)} to {deposit['bank']} account {deposit['accountNumber']}, the virtual account Toss Payments issued for order {order['order']}, by {market.kst_text(expires)}.",
            "",
            "We issue the hour-credits as soon as Toss Payments reports the deposit. If nothing arrives by then, the account expires with the order and nothing is owed.",
            "",
            f"Follow it in your account: {self.back(order['order'])}",
            "",
            "HYOR AI Tech Ltd. (SparkPipe), Seoul. Reply to this email with any question.",
        ]))

    def toss_refuse(self, order, status):
        order_id, state = order["order"], "cancelled"
        try:
            toss_call(self.cfg, "POST", "/v1/payments/" + urllib.parse.quote(order["payment_key"], safe="") + "/cancel", {"cancelReason": "The amount differs from the order."}, "mismatch-" + order_id)
        except (Declined, urllib.error.URLError, OSError, ValueError) as error:
            log(f"order {order_id}: cancelling a payment that does not match its order: {error}")
            state = "mismatch"
        with self.book.lock:
            self.book.append("toss_state", {"order": order_id, "state": state, "reason": "amount"})
        log(f"order {order_id}: Toss Payments reports {status} for another amount than the order; no HC issued, payment {'cancelled' if state == 'cancelled' else 'left for review'}")
        self.mail(self.cfg.notify, f"Payment to review: {order_id}", f"Order {order_id} for {order['total']} KRW: Toss Payments reported {status} for a different amount or currency. No HC was issued. " + ("We cancelled it at Toss Payments." if state == "cancelled" else "Cancelling it failed: cancel it in the Toss Payments console."))

    def toss_check(self, order_id):
        order = self.book.orders.get(order_id)
        if not order or order.get("processor") != "toss":
            return
        if order.get("payment_key"):
            self.toss_apply(order_id, toss_payment(self.cfg, order["payment_key"]))
        with self.book.lock:
            if order["issued"] or order["state"] not in TOSS_OPEN or clock() < self.book.expiry(order):
                return
        if order["state"] == "waiting":
            try:
                remote = toss_call(self.cfg, "POST", "/v1/payments/" + urllib.parse.quote(order["payment_key"], safe="") + "/cancel", {"cancelReason": "The deposit deadline passed unpaid."}, "expire-" + order_id)
            except Declined as error:
                log(f"order {order_id}: cancelling the unpaid virtual account: {error}")
                remote = toss_payment(self.cfg, order["payment_key"])
            if remote.get("status") == "DONE":
                self.toss_apply(order_id, remote)
                return
        with self.book.lock:
            if not order["issued"] and order["state"] in TOSS_OPEN:
                self.book.append("toss_state", {"order": order_id, "state": "timedout", "reason": "expired"})

    def lapse_due(self, order_id):
        owed = sum((Decimal(r["refund"]["amount"]) for refunds in self.book.market.refunds.values() for r in refunds if r["refund"] and r["refund"]["order"] == order_id), Decimal(0))
        return owed - sum((Decimal(r["amount"]) for r in self.book.orders[order_id]["refunds"] if r["clause"] == "16.4"), Decimal(0))

    def refund(self, order_id, clause, amount, reason, account=None):
        order = self.book.orders.get(order_id)
        swift = bool(order) and order.get("processor") == "swift"
        if not order or not order["issued"] or not (swift or (order.get("processor") == "toss" and order.get("payment_key"))):
            raise Refusal(f"{order_id} is not a Toss Payments or SWIFT payment that issued HC.")
        if clause not in REFUND_CLAUSES or not reason:
            raise Refusal("Name the clause of the Customer Agreement that allows the refund (" + ", ".join(REFUND_CLAUSES) + ") and the reason.")
        currency, unit, shown = (order["currency"], CENT, text_money) if swift else (KRW, WON, text_won)
        remote = None if swift else toss_payment(self.cfg, order["payment_key"])
        if remote is not None and remote.get("orderId") != order_id:
            raise Refusal(f"Toss Payments does not report {order_id} for this payment key.")
        balance = Decimal(order["received"]) - Decimal(order.get("refunded") or "0") if swift else Decimal(str(remote.get("balanceAmount") or 0))
        amount = balance if amount == "all" else usd_field(amount) if swift else won_field(amount)
        if not 0 < amount <= balance:
            raise Refusal(f"Refund from {amount_text(unit, currency)} to {amount_text(balance, currency)}, what is left of this payment" + ("." if swift else " at Toss Payments."))
        if clause == "16.4" and amount > self.lapse_due(order_id):
            raise Refusal(f"Clause 16.4 refunds locked HC that lapsed: {amount_text(max(self.lapse_due(order_id), Decimal(0)), currency)} of this order is due.")
        excess = Decimal(order["received"]) - Decimal(order["total"]) - sum((Decimal(r["amount"]) for r in order["refunds"] if r["clause"] == "3.5"), Decimal(0)) if swift else Decimal(0)
        if swift and clause == "3.5" and amount > excess:
            raise Refusal(f"Clause 3.5 returns what arrived above the invoice total: {amount_text(max(excess, Decimal(0)), currency)} of this order is left to return.")
        if remote is not None and remote.get("method") == "가상계좌" and not account:
            raise Refusal("A virtual account payment is refunded to a bank account: give the bank code, the account number and the holder's name.")
        debit, lot, moment = Decimal(0), None, market.stamp(self.market.now())
        if clause not in ("3.2", "16.4") and not (swift and clause == "3.5"):
            debit = (Decimal(order["hc"]) * amount / Decimal(order["total"])).quantize(market.HC_STEP, rounding=ROUND_CEILING)
            lot = next((found for found in self.book.market.lots[order["account"]] if found["ref"] == order_id and found["expires"] > moment), None)
            left = lot["hc"] if lot else Decimal(0)
            if debit > left:
                most = (left * Decimal(order["total"]) / Decimal(order["hc"])).quantize(unit, rounding=ROUND_DOWN)
                raise Refusal(f"Only {market.text_hc(left)} HC of this order is unspent, unlocked and not lapsed, which refunds at most {amount_text(most, currency)}.")
        rest = balance - amount
        if not swift:
            body = {"cancelReason": f"Customer Agreement clause {clause}: {reason}"[:200], "refundableAmount": int(balance)}
            if amount < balance:
                body["cancelAmount"] = int(amount)
            if account:
                body["refundReceiveAccount"] = account
            after = toss_call(self.cfg, "POST", "/v1/payments/" + urllib.parse.quote(order["payment_key"], safe="") + "/cancel", body, f"refund-{order_id}-{int(balance)}")
            rest = Decimal(str(after.get("balanceAmount")))
            if rest != balance - amount:
                raise Refusal(f"Toss Payments left {amount_text(rest, KRW)} of this payment instead of {amount_text(balance - amount, KRW)}; check it in the console before recording anything.", 502)
        data = {"order": order_id, "account": order["account"], "processor": order["processor"], "amount": shown(amount), "balance": shown(rest), "clause": clause, "reason": reason, "hc": market.text_hc(debit)}
        if not swift:
            data["paymentKey"] = order["payment_key"]
        with self.book.lock:
            line = self.book.append("refund", data)
            if lot:
                self.book.append("hc_debit", {"account": order["account"], "hc": market.text_hc(debit), "reason": "refund", "ref": order_id, "lot": lot["lot"], "now": moment})
        way = "It goes back by SWIFT to the account it came from" if swift else "It goes back by the method you paid with, through Toss Payments"
        self.mail(order["email"], f"Refund: SparkPipe order {order_id}", "\n".join([
            f"We refunded {amount_text(amount, currency)} of order {order_id} under clause {clause} of the Customer Agreement: {reason}",
            "",
            way + (f"; {market.text_hc(debit)} HC of the order left your account." if lot else "."),
            "",
            "HYOR AI Tech Ltd. (SparkPipe), Seoul. Reply to this email with any question.",
        ]))
        return line["data"]

    def tax_invoices(self):
        return [o for o in self.book.orders.values() if o.get("processor") == "toss" and (o.get("tax_invoice") or {}).get("needed") and not o.get("tax_invoice_number")]

    def work(self):
        while True:
            key = self.jobs.get()
            try:
                if key is None:
                    return
                if isinstance(key, tuple):
                    self.toss_check(key[1])
                    continue
                order_id = self.book.invoices.get(key)
                if order_id:
                    self.settle(order_id, call(self.cfg, "GET", "/merchant/invoices/" + key))
            except Exception as error:
                log(f"invoice {key}: {error}")
            finally:
                self.jobs.task_done()

    def reconcile(self):
        now = clock()
        with self.book.lock:
            keys = [o["invoice"] for o in self.book.orders.values() if o["invoice"] and (o["state"] in OPEN or (o["state"] == "completed" and not o["issued"])) and (now - stamp_time(o["created"])).days < RECONCILE_DAYS]
            keys += [("toss", o["order"]) for o in self.book.orders.values() if o.get("processor") == "toss" and not o["issued"] and o["state"] in TOSS_OPEN]
        for key in keys:
            self.jobs.put(key)

    def housekeeping(self):
        self.book.refresh()
        self.usage.prune(clock())
        self.audits.prune(clock())
        self.notices.prune(self.market.now())
        self.notifier.deliveries.prune(clock())
        self.uptime.prune(self.market.now())
        self.sessions.keep(lambda record: stamp_time(record["expires"]) > clock())
        self.notifier.sweep(self.market.now())

    def reconcile_forever(self):
        while True:
            time.sleep(self.cfg.reconcile_seconds)
            self.reconcile()
            try:
                self.housekeeping()
            except Exception as error:
                log(f"housekeeping: {error}")


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "sparkpipe-pay"

    def log_message(self, fmt, *args):
        pass

    def reply(self, status, data, cookie=None, headers=None):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if cookie is not None:
            cfg = self.server.app.cfg
            secure = "; Secure" if cfg.public.startswith("https://") else ""
            self.send_header("Set-Cookie", f"{COOKIE}={cookie}; Max-Age={SESSION_SECONDS if cookie else 0}; Path={cfg.prefix}; HttpOnly; SameSite=Lax{secure}")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def reply_text(self, text):
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def redirect(self, location):
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def refuse(self, error):
        data = {"error": str(error)}
        if error.reason:
            data["reason"] = error.reason
        self.reply(error.status, data)

    def api_error(self, error, shape="openai", path=""):
        code = error.reason or "refused"
        body = {"type": "error", "error": {"type": code, "message": str(error)}} if shape == "anthropic" else {"error": {"message": str(error), "type": "invalid_request_error", "code": code}}
        headers = {"Retry-After": "60"} if error.status == 429 else {}
        if error.status == 401 and path.startswith("/git/"):
            headers["WWW-Authenticate"] = 'Basic realm="SparkPipe"'
        self.reply(error.status, body, headers=headers)

    def client_ip(self):
        forwarded = self.headers.get("X-Forwarded-For")
        return forwarded.split(",")[-1].strip() if forwarded else self.client_address[0]

    def agent(self):
        return (self.headers.get("User-Agent") or "")[:300]

    def bearer(self):
        header = self.headers.get("Authorization") or ""
        return header[7:].strip() if header.startswith("Bearer ") else ""

    def cross_site(self):
        origin = self.headers.get("Origin")
        return bool(origin) and origin != self.server.app.cfg.origin

    def do_GET(self):
        self.route("GET")

    def do_HEAD(self):
        self.route("HEAD")

    def do_POST(self):
        self.route("POST")

    def do_PUT(self):
        self.route("PUT")

    def do_DELETE(self):
        self.route("DELETE")

    def route(self, method):
        parts = urllib.parse.urlsplit(self.path)
        scope = hub_scope(method, parts.path)
        if scope:
            return self.hub(method, parts, scope)
        if parts.path.startswith("/v1/"):
            return self.account_api(method, parts)
        if parts.path.startswith("/hub/") and method == "POST":
            return self.hub_report(parts.path)
        if method == "GET":
            return self.get(parts)
        if method == "POST" or (method == "PUT" and parts.path == "/market/auto-extend"):
            return self.post(parts.path)
        self.reply(404, {"error": "Not found."})

    def get(self, parts):
        app = self.server.app
        path, query = parts.path, urllib.parse.parse_qs(parts.query)
        if path == "/health":
            return self.reply(200, {"ok": True})
        match = HOLDERS_PATH.match(path)
        if match:
            return self.holders(match.group(1), query)
        routes = {
            "/account": lambda: app.summary(app.need_actor(self.headers)),
            "/market/state": lambda: app.market_state(app.actor(self.headers)),
            "/market/ledger": lambda: app.ledger(app.need_owner(self.headers, ("owner", "admin"), "The ledger export")),
            "/status": app.market.status_page,
            "/toss/config": app.toss_config,
            "/swift/config": app.swift_config,
        }
        try:
            app.throttle("read", self.client_ip())
            if path == "/availability.txt":
                return self.reply_text(app.market.availability_text())
            if path in routes:
                return self.reply(200, routes[path]())
            match = TOSS_PATH.match(path)
            if match:
                return self.redirect((app.toss_success if match.group(1) == "success" else app.toss_fail)(match.group(2), query))
            match = ORDER_PATH.match(path)
            if not match:
                return self.reply(404, {"error": "Not found."})
            self.reply(200, app.status(match.group(1)))
        except Refusal as error:
            self.refuse(error)

    def json_body(self, limit):
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = -1
        if length <= 0 or length > limit:
            raise Refusal("The request body is missing or too large.", 413 if length > limit else 400, "invalid_request")
        try:
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            raise Refusal("The request is not a JSON object.", 400, "invalid_request")
        return body

    def post(self, path):
        app = self.server.app
        limit = 262144 if path in ("/coinpayments", "/toss/webhook") else 16384
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = -1
        if length <= 0 or length > limit:
            return self.reply(413 if length > limit else 400, {"error": "The request body is missing or too large."})
        raw = self.rfile.read(length)
        if path == "/coinpayments":
            return self.reply(200 if app.webhook(self.headers, raw) else 401, {})
        if path == "/toss/webhook":
            app.toss_webhook(raw)
            return self.reply(200, {})
        if self.cross_site():
            return self.reply(403, {"error": "Cross-site requests are refused."})
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self.reply(400, {"error": "The request is not a JSON object."})
        try:
            if path == "/code":
                return self.reply(200, app.send_code(body, self.client_ip()))
            if path == "/verify":
                summary, token = app.verify(body, self.client_ip(), self.agent())
                return self.reply(200, summary, cookie=token)
            if path == "/signout":
                return self.reply(200, app.sign_out(self.headers), cookie=None if credential(self.headers).startswith(BEARER) else "")
            if path == "/order":
                account = app.need_owner(self.headers, ("owner",), "Buying HC")
                found = app.order(body, account, self.client_ip(), self.agent())
                app.audit(account, account, "order.created", {"order": found.get("order"), "processor": body.get("processor", "coinpayments")})
                return self.reply(200, found)
            if path.startswith("/market/"):
                return self.reply(200, app.market_action(path, body, app.need_actor(self.headers), self.client_ip(), self.agent()))
            return self.reply(404, {"error": "Not found."})
        except Refusal as error:
            self.refuse(error)

    def hub_key(self):
        app = self.server.app
        return bool(app.cfg.pool_key) and hmac.compare_digest(self.bearer().encode(), app.cfg.pool_key.encode())

    def holders(self, fleet, query):
        app = self.server.app
        if not self.hub_key():
            return self.refuse(Refusal("Send the upstream key as Authorization: Bearer.", 401, "invalid_key"))
        try:
            start, end = (market.slot_field(query.get(name, [""])[0], name) for name in ("from", "to"))
            self.reply(200, app.market.holder_hours(fleet, start, end))
        except Refusal as error:
            self.refuse(error)

    def hub_report(self, path):
        app = self.server.app
        actions = {"/hub/requests": app.record_usage, "/hub/events": app.hub_events, "/hub/pair-loss": app.market.pair_loss, "/hub/audit": app.hub_audit}
        if path not in actions:
            return self.reply(404, {"error": "Not found."})
        if not self.hub_key():
            return self.refuse(Refusal("Send the upstream key as Authorization: Bearer.", 401, "invalid_key"))
        try:
            self.reply(200, actions[path](self.json_body(REPORT_LIMIT)))
        except Refusal as error:
            self.refuse(error)

    def account_api(self, method, parts):
        app = self.server.app
        route = next(((need, action, match) for verb, pattern, need, action in ACCOUNT_ROUTES for match in [re.fullmatch(pattern, parts.path)] if match and verb == method), None)
        try:
            if not route:
                raise Refusal("Not found.", 404, "not_found")
            need, action, match = route
            account, record, via, member, role = app.principal(self.headers, self.client_ip(), "account:read")
            if need == "write" and record:
                raise Refusal("Managing keys, webhooks and members needs a session: sign in on the account page or with sparkpipe login.", 403, "invalid_key")
            if need == "write" and via == "cookie" and self.cross_site():
                raise Refusal("Cross-site requests are refused.", 403, "invalid_request")
            app.throttle("market", account)
            body = self.json_body(ACCOUNT_LIMIT) if method in ("POST", "PUT") and self.headers.get("Content-Length") not in (None, "0") else {}
            self.reply(200, app.account_call(action, (account, member, role), match, urllib.parse.parse_qs(parts.query), body))
        except Refusal as error:
            self.api_error(error)

    def chunked(self):
        found = bytearray()
        while True:
            line = self.rfile.readline(1024).split(b";")[0].strip()
            if not CHUNK.fullmatch(line):
                raise Refusal("The chunked request body is malformed.", 400, "invalid_request")
            size = int(line, 16)
            if size == 0:
                while self.rfile.readline(1024) not in (b"\r\n", b"\n", b""):
                    pass
                return bytes(found)
            if len(found) + size > BODY_LIMIT:
                raise Refusal("Request bodies are limited to 32 MiB.", 413, "invalid_request")
            chunk = self.rfile.read(size)
            if len(chunk) < size:
                raise Refusal("The chunked request body is malformed.", 400, "invalid_request")
            found += chunk
            self.rfile.readline(1024)

    def hub_body(self, path):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            return self.chunked()
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = -1
        if length < 0:
            raise Refusal("The Content-Length is not a number.", 400, "invalid_request")
        if length > BODY_LIMIT:
            raise Refusal("Request bodies are limited to 32 MiB.", 413, "invalid_request")
        if not length:
            return None
        return self.rfile.read(length) if path in ("/v1/chat/completions", "/v1/state/move") else length

    def body_object(self, raw):
        try:
            found = json.loads(raw or b"")
            if not isinstance(found, dict):
                raise ValueError
        except ValueError:
            raise Refusal("The request is not a JSON object.", 400, "invalid_request")
        return found

    def chat_body(self, raw):
        found = self.body_object(raw)
        if "sparkpipe" not in found:
            return raw, None
        extra = found["sparkpipe"]
        if not isinstance(extra, dict) or not isinstance(extra.get("fleet", ""), str) or set(extra) - {"fleet"}:
            raise Refusal("Give sparkpipe as an object whose only field is fleet, a Fleet id.", 400, "invalid_sparkpipe")
        rest = {name: value for name, value in found.items() if name != "sparkpipe"}
        return json.dumps(rest, ensure_ascii=False).encode(), extra.get("fleet") or None

    def move_fleets(self, raw):
        found = self.body_object(raw)
        source, target = found.get("from"), found.get("to")
        if not isinstance(source, str) or not isinstance(target, str) or source == target:
            raise Refusal("Give from and to as two different Fleets the account holds.", 400, "invalid_request")
        return source, (target,)

    def admitted(self, method, path, account, record):
        app = self.server.app
        requested = (self.headers.get("X-SparkPipe-Fleet") or "").strip() or None
        if method == "POST" and path == "/v1/state/move":
            if record and record["fleet"]:
                raise Refusal(f"This key reaches only {record['fleet']}: moving working state needs a key without a fleet, or the account session.", 403, "invalid_key")
            app.market.admit_first(account, None, None)
            body = self.hub_body(path)
            source, also = self.move_fleets(body)
            return app.market.admit(account, record, source, also), body
        if method == "POST" and path == "/v1/chat/completions":
            fleet = app.market.admit_first(account, record, requested)
            body, named = self.chat_body(self.hub_body(path))
            return fleet or app.market.admit(account, record, named), body
        return app.market.admit(account, record, requested), self.hub_body(path)

    def hub(self, method, parts, scope):
        app = self.server.app
        path = parts.path
        shape = "anthropic" if path.startswith("/v1/messages") else "openai"
        try:
            if any(part in (".", "..") for part in urllib.parse.unquote(path).split("/")):
                raise Refusal("Not found.", 404, "not_found")
            account, record, via, member, role = app.principal(self.headers, self.client_ip(), scope)
            if via == "cookie" and method not in ("GET", "HEAD") and self.cross_site():
                raise Refusal("Cross-site requests are refused.", 403, "invalid_request")
            app.throttle_gateway(account)
            fleet, body = self.admitted(method, path, account, record)
            served, origin = app.market.route(account, fleet) if app.cfg.hub or app.cfg.sandbox else (fleet, None)
        except Refusal as error:
            return self.api_error(error, shape, path)
        if app.cfg.sandbox:
            return self.sandboxed(method, path, body, served, shape, origin)
        while True:
            hub = app.cfg.upstream(served)
            if not hub:
                return self.api_error(Refusal("The API opens once it is connected to the Fleets.", 503, "closed"), shape, path)
            target = hub[2] + path + ("?" + parts.query if parts.query else "")
            headers = self.upstream_headers(account, record, served, origin, member, role)
            ip, upstream = self.client_ip(), []
            self.cutoff = False
            watch = app.guard.watch(lambda: self.still(account, fleet, scope, ip, served), lambda: self.cut(upstream))
            try:
                if method == "GET" and "websocket" in (self.headers.get("Upgrade") or "").lower():
                    return self.tunnel(hub, target, headers, shape, upstream)
                if self.forward(hub, method, target, headers, body, shape, path, upstream, origin):
                    return
            finally:
                app.guard.done(watch)
            if origin:
                return self.api_error(Refusal("The Fleet did not answer. Try again in a minute.", 502, "upstream_error"), shape, path)
            app.uptime.observe(fleet, False, app.market.now(), "the gateway could not connect")
            try:
                served, origin = app.market.route(account, fleet)
            except Refusal as error:
                return self.api_error(error, shape, path)

    def still(self, account, fleet, scope, ip, served):
        app = self.server.app
        try:
            found = app.principal(self.headers, ip, scope)[0]
        except Refusal:
            return None
        if found != account:
            return None
        left = app.market.hold_left(account, fleet)
        return left if served == fleet else min(left, app.market.failover_left(served, app.market.now()))

    def cut(self, upstream):
        self.cutoff = True
        for sock in [self.connection] + upstream:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def upstream_headers(self, account, record, fleet, origin=None, member=None, role="owner"):
        app = self.server.app
        found = [(name, value) for name, value in self.headers.items()
                 if TOKEN.fullmatch(name) and "\r" not in value and "\n" not in value and name.lower() not in DROPPED and not name.lower().startswith("x-forwarded-")
                 and (not name.lower().startswith("x-sparkpipe-") or name.lower() in CUSTOMER_HEADERS)]
        scopes = [scope for scope in (record["scopes"] if record else market.SCOPES) if scope in market.ROLE_SCOPES[role]]
        found += [("Authorization", f"Bearer {app.cfg.pool_key}"), ("X-SparkPipe-Public", "1"), ("X-SparkPipe-Account", account), ("X-SparkPipe-Fleet", fleet),
                  ("X-SparkPipe-Key", record["key"] if record else "session"), ("X-SparkPipe-Member", member or account), ("X-SparkPipe-Role", role), ("X-SparkPipe-Scopes", ",".join(scopes)),
                  ("X-SparkPipe-Allow-Swap", "1" if (not record or record["allow_swap"]) and role != "member" else "0")]
        if record and record["priority"] and not self.headers.get("X-SparkPipe-Priority"):
            found.append(("X-SparkPipe-Priority", record["priority"]))
        if origin:
            found.append(("X-SparkPipe-Failover-From", origin))
        return found

    def forward(self, hub, method, target, headers, body, shape, path, upstream, origin=None):
        connection = http.client.HTTPConnection(hub[0], hub[1], timeout=900)
        started = False
        try:
            connection.connect()
        except OSError as error:
            log(f"fleet upstream {method} {'/'.join(path.split('/')[:3])}: {error}")
            connection.close()
            return False
        try:
            upstream.append(connection.sock)
            connection.putrequest(method, target, skip_accept_encoding=True)
            for name, value in headers:
                connection.putheader(name, value)
            if body is not None:
                connection.putheader("Content-Length", str(len(body) if isinstance(body, bytes) else body))
            connection.endheaders()
            if isinstance(body, bytes):
                connection.send(body)
            elif isinstance(body, int):
                left = body
                while left:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        raise ConnectionError("the client stopped sending its body")
                    connection.send(chunk)
                    left -= len(chunk)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                if name.lower() not in RESPONSE_DROPPED:
                    self.send_header(name, value)
            if origin:
                self.send_header("X-SparkPipe-Failover-From", origin)
            self.end_headers()
            started = True
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (OSError, http.client.HTTPException) as error:
            log(f"fleet upstream {method} {'/'.join(path.split('/')[:3])}: {error}")
            if not started and not self.cutoff:
                self.api_error(Refusal("The Fleet did not answer. Try again in a minute.", 502, "upstream_error"), shape, path)
        finally:
            connection.close()
            self.close_connection = True
        return True

    def tunnel(self, hub, target, headers, shape, sockets):
        try:
            upstream = socket.create_connection((hub[0], hub[1]), timeout=30)
        except OSError as error:
            log(f"fleet upstream websocket: {error}")
            return self.api_error(Refusal("The Fleet did not answer. Try again in a minute.", 502, "upstream_error"), shape)
        sockets.append(upstream)
        try:
            lines = [f"GET {target} HTTP/1.1", f"Host: {hub[0]}:{hub[1]}", "Connection: Upgrade", f"Upgrade: {self.headers.get('Upgrade')}"]
            upstream.sendall(("\r\n".join(lines + [f"{name}: {value}" for name, value in headers]) + "\r\n\r\n").encode("latin-1"))
            reply = b""
            while b"\r\n\r\n" not in reply and len(reply) < 65536:
                chunk = upstream.recv(65536)
                if not chunk:
                    break
                reply += chunk
            head = reply.partition(b"\r\n\r\n")[0]
            if not head.startswith(b"HTTP/1.1 101"):
                if not head.startswith(b"HTTP/"):
                    return self.api_error(Refusal("The Fleet did not answer. Try again in a minute.", 502, "upstream_error"), shape)
                found = re.search(rb"(?im)^content-length:\s*(\d+)", head)
                while found and len(reply) - len(head) - 4 < int(found.group(1)):
                    chunk = upstream.recv(65536)
                    if not chunk:
                        break
                    reply += chunk
                self.connection.sendall(reply)
                return
            self.connection.sendall(reply)
            self.pipe(upstream)
        except OSError as error:
            log(f"fleet upstream websocket: {error}")
        finally:
            upstream.close()
            self.close_connection = True

    def pipe(self, upstream):
        client = self.connection
        client.settimeout(None)
        upstream.settimeout(None)
        while True:
            ready = select.select([client, upstream], [], [])[0]
            for side in ready:
                data = side.recv(65536)
                if not data:
                    return
                (upstream if side is client else client).sendall(data)

    def sandboxed(self, method, path, body, fleet, shape, origin=None):
        app = self.server.app
        pricing = app.cfg.pricing()
        if method == "GET" and path == "/v1/models":
            return self.reply(200, {"object": "list", "data": [{"id": m["id"], "object": "model", "owned_by": "sparkpipe"} for m in pricing["models"]]})
        if method != "POST" or path != "/v1/chat/completions":
            return self.api_error(Refusal("The sandbox answers /v1/chat/completions and /v1/models; the other paths need a held Fleet on the live service.", 503, "closed"), shape, path)
        try:
            found = self.body_object(body)
            if not isinstance(found.get("messages"), list) or not found["messages"]:
                raise Refusal("Send the conversation as a non-empty messages list.", 400, "invalid_messages")
            if found.get("model") not in [m["id"] for m in pricing["models"]]:
                raise Refusal("Choose one of the models: " + ", ".join(m["id"] for m in pricing["models"]) + ".", 404, "model_not_found")
        except Refusal as error:
            return self.api_error(error, shape, path)
        self.simulated(found, found["messages"], {"model": found["model"], "fleet": fleet, "failover_from": origin})

    def stream_head(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def event(self, data):
        self.wfile.write(b"data: " + (data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()) + b"\n\n")
        self.wfile.flush()

    def simulated(self, body, messages, ticket):
        limit = body.get("max_tokens") if isinstance(body.get("max_tokens"), int) and not isinstance(body.get("max_tokens"), bool) else 128
        text, tokens_in, tokens_out = market.simulate(ticket["model"], ticket["fleet"], messages, min(max(limit, 1), 512))
        made, ident = int(time.time()), "chatcmpl-" + secrets.token_hex(12)
        usage = {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": tokens_in + tokens_out}
        note = dict({"fleet": ticket["fleet"]}, **({"failover_from": ticket["failover_from"]} if ticket.get("failover_from") else {}))
        if body.get("stream") is True:
            self.stream_head()
            for word in text.split(" "):
                self.event({"id": ident, "object": "chat.completion.chunk", "created": made, "model": ticket["model"], "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]})
                time.sleep(0.025)
            self.event({"id": ident, "object": "chat.completion.chunk", "created": made, "model": ticket["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage, "sparkpipe": note})
            return self.event(b"[DONE]")
        time.sleep(min(1.0, tokens_out / 40))
        self.reply(200, {"id": ident, "object": "chat.completion", "created": made, "model": ticket["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage, "sparkpipe": note})


class Server(http.server.ThreadingHTTPServer):
    request_queue_size = LISTEN_QUEUE
    daemon_threads = True


def start(cfg, port=None):
    os.makedirs(cfg.data, exist_ok=True)
    app = App(cfg, Book(cfg.path("ledger.jsonl")))
    threading.Thread(target=app.work, daemon=True).start()
    threading.Thread(target=app.deliver, daemon=True).start()
    threading.Thread(target=app.market.run, args=(log,), daemon=True).start()
    for _ in range(WEBHOOK_WORKERS):
        threading.Thread(target=app.notifier.run, daemon=True).start()
    threading.Thread(target=app.guard.run, daemon=True).start()
    if cfg.hub and cfg.health_seconds > 0 and not cfg.sandbox:
        threading.Thread(target=app.uptime.run, daemon=True).start()
    app.usage.prune(clock())
    app.audits.prune(clock())
    app.notices.prune(app.market.now())
    app.notifier.deliveries.prune(clock())
    app.notifier.sweep(app.market.now())
    server = Server(("127.0.0.1", cfg.port if port is None else port), Handler)
    server.app = app
    return server


def flush(app):
    threading.Thread(target=app.deliver, daemon=True).start()
    app.mails.join()


def release(cfg, app, order_id):
    order = app.book.orders.get(order_id)
    if not order or not order["invoice"]:
        print(f"{order_id}: no order with an invoice", file=sys.stderr)
        return 1
    remote = call(cfg, "GET", "/merchant/invoices/" + order["invoice"])
    state, total, currency = remote_fields(remote)
    print(f"{order_id}: CoinPayments says {state}, {total} {currency}, {received_of(remote)} USD received; the order is {order['total']} USD for {order['hc']} HC")
    if state != "completed":
        print("not completed at CoinPayments: nothing issued. Accept the part payment there first, or refund it.", file=sys.stderr)
        return 1
    issued = app.settle(order_id, remote, released=True)
    flush(app)
    print(f"{order_id}: " + (f"issued {issued['hc']} HC for {issued['net']} USD before VAT" if issued else "nothing issued"))
    return 0 if issued else 1


def maintenance(app, argv):
    try:
        found = app.market.schedule_maintenance(argv[2], argv[3], int(argv[4]) if len(argv) == 5 else None)
    except (Refusal, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    flush(app)
    print(f"maintenance {found['maintenance']}: {found['fleet']} for {found['hours']} hours from {found['start']}; the booked hours it overlaps are credited when it ends")
    return 0


def credit(app, argv):
    account = app.book.emails.get(argv[2].lower())
    try:
        credits = market.hc_field(argv[3], "credit")
    except Refusal as error:
        print(error, file=sys.stderr)
        return 2
    if not account or not argv[4].strip():
        print("usage: sparkpipe_pay.py credit EMAIL HC NOTE, for an existing account", file=sys.stderr)
        return 2
    lot = app.market.downtime_credit(account, credits, argv[4].strip()[:200])
    flush(app)
    print(f"credited {lot['hc']} HC to {argv[2]} under clause 4.4, lot {lot['lot']}, lapsing {lot['expires']}; the customer is told by downtime.credited")
    return 0


def uptime(app, argv):
    try:
        start, end = (market.stamp_field(value if "T" in value else value + "T00:00:00Z", "time") for value in argv[3:5])
        app.market.fleet_field(app.cfg.pricing(), argv[2])
    except Refusal as error:
        print(error, file=sys.stderr)
        return 2
    found = [s for s in app.uptime.spans(start) if s["fleet"] == argv[2] and s["start"] < market.stamp(end)]
    entry = app.uptime.entry(argv[2])
    print(f"{argv[2]}: records since {entry['first'] if entry else 'never'}, last probe {entry.get('seen') if entry else 'none'}; {len(found)} periods without a good probe from {market.stamp(start)} to {market.stamp(end)}")
    for s in sorted(found, key=lambda s: s["start"]):
        print(f"{s['span']} {s['start']} to {s['end'] or 'now'} ({s['cause']})")
    return 0


def erase_records(app, argv):
    account = app.book.emails.get(argv[2].lower())
    if not account or not argv[3].strip():
        print("usage: sparkpipe_pay.py erase-records EMAIL NOTE, for an existing account's owner email, with the written request as NOTE", file=sys.stderr)
        return 2
    found = app.erase_records(account, "operator", argv[3].strip())
    flush(app)
    print(f"deleted {found['erased']} request records of {argv[2]} up to {found['through']} (DPA clause 9.3); the customer is emailed, and the hub's records are deleted on the hub")
    return 0


def refund(app, argv):
    account = {"bank": argv[6], "accountNumber": argv[7], "holderName": argv[8]} if len(argv) == 9 else None
    try:
        found = app.refund(argv[2], argv[3], argv[4], argv[5].strip()[:150], account)
    except (Refusal, Declined) as error:
        print(f"{argv[2]}: {error}", file=sys.stderr)
        return 2
    flush(app)
    if found["processor"] == "swift":
        print(f"{argv[2]}: recorded a refund of {found['amount']} USD under clause {found['clause']}, {found['balance']} USD left, {found['hc']} HC debited. Send {found['amount']} USD by SWIFT to the account the payment came from.")
    else:
        print(f"{argv[2]}: refunded {found['amount']} KRW under clause {found['clause']}, {found['balance']} KRW left at Toss Payments, {found['hc']} HC debited")
    return 0


def swift_confirm(app, argv):
    try:
        found = app.swift_confirm(argv[2], argv[3], argv[4].strip()[:200] if len(argv) == 5 else "")
    except Refusal as error:
        print(f"{argv[2]}: {error}", file=sys.stderr)
        return 2
    flush(app)
    extra = f"; {found['excess']} USD above the total: send it back by SWIFT and record it with refund {found['order']} 3.5 {found['excess']} REASON" if Decimal(found["excess"]) > 0 else ""
    print(f"{found['order']}: {found['received']} USD received, {found['issued']} HC issued{extra}")
    return 0


def swift_cancel(app, argv):
    try:
        found = app.swift_cancel(argv[2], argv[3].strip()[:200])
    except Refusal as error:
        print(f"{argv[2]}: {error}", file=sys.stderr)
        return 2
    flush(app)
    print(f"{found['order']}: cancelled, the customer is emailed")
    return 0


def tax_invoices(app, argv):
    for order in app.tax_invoices():
        invoice = order["tax_invoice"]
        day = stamp_time(order["paid"]).astimezone(KST).strftime("%Y-%m-%d")
        refunded = f", refunded {order['refunded']} KRW: issue an amended invoice" if order.get("refunded") else ""
        print(f"{order['order']} {day} {order['company']} <{order['email']}> BRN {invoice['brn'] or 'missing: ask the customer'}: supply {invoice['supply']} KRW, VAT {invoice['vat']} KRW, total {order['total']} KRW, {METHOD_EN.get(order.get('method'), order.get('method'))}{refunded}")
    return 0


def tax_invoice(app, argv):
    order = app.book.orders.get(argv[2])
    if not order or order not in app.tax_invoices() or not argv[3].strip():
        print("usage: sparkpipe_pay.py tax-invoice ORDER NUMBER, for an order that tax-invoices lists", file=sys.stderr)
        return 2
    app.book.append("tax_invoice", {"order": argv[2], "account": order["account"], "number": argv[3].strip()[:40]})
    print(f"{argv[2]}: electronic tax invoice {argv[3].strip()[:40]} recorded")
    return 0


COMMANDS = {
    ("release", 3): lambda cfg, app, argv: release(cfg, app, argv[2]),
    ("maintenance", 4): lambda cfg, app, argv: maintenance(app, argv),
    ("maintenance", 5): lambda cfg, app, argv: maintenance(app, argv),
    ("credit", 5): lambda cfg, app, argv: credit(app, argv),
    ("refund", 6): lambda cfg, app, argv: refund(app, argv),
    ("uptime", 5): lambda cfg, app, argv: uptime(app, argv),
    ("erase-records", 4): lambda cfg, app, argv: erase_records(app, argv),
    ("refund", 9): lambda cfg, app, argv: refund(app, argv),
    ("tax-invoices", 2): lambda cfg, app, argv: tax_invoices(app, argv),
    ("tax-invoice", 4): lambda cfg, app, argv: tax_invoice(app, argv),
    ("swift-confirm", 4): lambda cfg, app, argv: swift_confirm(app, argv),
    ("swift-confirm", 5): lambda cfg, app, argv: swift_confirm(app, argv),
    ("swift-cancel", 4): lambda cfg, app, argv: swift_cancel(app, argv),
}


def main(argv):
    cfg = Config(os.environ)
    if argv[1:] == ["verify"]:
        book = Book(cfg.path("ledger.jsonl"))
        print(f"{book.count} lines, head {book.head}, {len(book.accounts)} accounts, {len(book.orders)} orders")
        return 0
    command = COMMANDS.get((argv[1] if len(argv) > 1 else "", len(argv)))
    if command:
        return command(cfg, App(cfg, Book(cfg.path("ledger.jsonl"))), argv)
    if argv[1:] not in ([], ["serve"]):
        print("usage: sparkpipe_pay.py [serve|verify|maintenance FLEET YYYY-MM-DD [HOURS]|credit EMAIL HC NOTE|release ORDER|refund ORDER CLAUSE KRW|USD|all REASON [BANK ACCOUNT HOLDER]|tax-invoices|tax-invoice ORDER NUMBER|swift-confirm ORDER|REFERENCE USD [NOTE]|swift-cancel ORDER|REFERENCE REASON|uptime FLEET FROM TO|erase-records EMAIL NOTE]", file=sys.stderr)
        return 2
    server = start(cfg)
    app = server.app
    threading.Thread(target=app.reconcile_forever, daemon=True).start()
    app.reconcile()
    log(f"sparkpipe-pay on 127.0.0.1:{server.server_address[1]}, {app.book.count} ledger lines, webhook {cfg.webhook_url}, Toss Payments " + (("test" if cfg.sandbox else "live") + f" at {cfg.public}{cfg.prefix}/toss/webhook" if cfg.toss else "off") + ", SWIFT " + ("on" if cfg.swift else "off"))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
