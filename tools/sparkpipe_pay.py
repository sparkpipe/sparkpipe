#!/usr/bin/env python3
import base64
import calendar
import collections
import fcntl
import hashlib
import hmac
import http.server
import itertools
import json
import os
import queue
import re
import secrets
import smtplib
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import sparkpipe_market as market
from sparkpipe_market import Refusal

KST = timezone(timedelta(hours=9))
USD = "5057"
CENT = Decimal("0.01")
HOUR_STEP = Decimal("0.0001")
LEDGER_IDS = ("order", "invoice", "reservation", "ask", "bid", "lot", "key", "ref", "plan")
TERMS = {"ko": "agreement-ko.md", "en": "agreement-en.md", "rules_ko": "market-rules-ko.md", "rules_en": "market-rules-en.md"}
DAYS = ("every", "weekdays")
PAYMENTS = ("upfront", "gates")
EVENTS = ["invoicePending", "invoicePaid", "invoiceCompleted", "invoiceCancelled", "invoiceTimedOut"]
OPEN = {"unpaid", "pending", "paid"}
CLOSED = {"cancelled", "timedout", "deleted", "failed"}
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
PLAN_PATH = re.compile(r"^/plan/(PO[A-Z2-7]{16})$")
PLAN_ID = re.compile(r"^PO[A-Z2-7]{16}$")
BRN = re.compile(r"^\d{3}-\d{2}-\d{5}$")
THRESHOLD = 80
PARTIES = ("company", "individual")
WINDOW_KO = {"americas": "미주", "asia": "아시아", "europe": "유럽"}
COOKIE = "sp_session"
SESSION_SECONDS = 30 * 86400
CODE_SECONDS = 600
CODE_TRIES = 5
PENDING_SECONDS = 86400
RECONCILE_DAYS = 14
SKEW_SECONDS = 300
LIMITS = {"code": (20, 600), "code-email": (5, 3600), "verify": (30, 600), "order": (10, 600), "lead": (5, 3600), "plan": (60, 3600), "read": (240, 600), "market": (240, 600), "pool": (600, 600)}


def clock():
    return datetime.now(timezone.utc)


def log(text):
    print(text, file=sys.stderr, flush=True)


def money(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def text_money(value):
    return format(money(value), "f")


def stamp_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def local_day(moment):
    return moment.astimezone(KST).date()


def months_later(start, months):
    total = start.month - 1 + months
    year, month = start.year + total // 12, total % 12 + 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(start.day, last)), start.day <= last


def term_end(start, months):
    moved, exists = months_later(start, months)
    return moved - timedelta(days=1) if exists else moved


def plan_fields(pricing, plan):
    fleets, months, lock, chosen = plan.get("fleets", 1), plan.get("months"), plan.get("lock") or [], plan.get("windows")
    named = {"F" + f["name"].split()[-1] for f in pricing["fleets"]}
    whole = lambda n: isinstance(n, int) and not isinstance(n, bool) and n >= 1
    good = isinstance(chosen, list) and bool(chosen) and all(
        isinstance(w, dict) and isinstance(w.get("id"), str) and whole(w.get("to")) and isinstance(w.get("from"), int) and not isinstance(w.get("from"), bool)
        and 0 <= w["from"] < w["to"] <= 24 and w.get("localDay") in (-1, 0, 1) for w in chosen)
    hours = [h for w in chosen for h in range(w["from"], w["to"])] if good else []
    try:
        fee, allowance = Decimal(str(plan.get("releaseFee"))), plan.get("maintenanceHoursPerWeek")
    except InvalidOperation:
        fee, allowance = None, None
    if (not good or len(set(hours)) != len(hours) or len({w["id"] for w in chosen}) != len(chosen)
            or plan.get("days") not in DAYS or not whole(fleets) or not whole(months) or plan.get("payment") not in PAYMENTS
            or fee is None or not fee.is_finite() or not 0 <= fee < 1 or not isinstance(allowance, int) or isinstance(allowance, bool) or not 0 <= allowance <= 24
            or not isinstance(lock, list) or (lock and (len(lock) != fleets or len(set(lock)) != len(lock) or any(f not in named for f in lock)
                                                       or len(hours) != 24 or plan.get("days") != "every"))):
        raise ValueError(f"plan {plan.get('id')}: windows as non-overlapping {{id, from, to, localDay}} of the Order, days {' or '.join(DAYS)}, fleets and months whole numbers from 1, payment {' or '.join(PAYMENTS)}, releaseFee from 0 to below 1, maintenanceHoursPerWeek from 0 to 24, and lock none, or one named fleet per fleet on a plan of the whole day every day")
    return chosen, fleets


def day_hours(windows, days, fleets, day):
    return sum((w["to"] - w["from"]) * fleets for w in windows if days == "every" or (day + timedelta(days=w["localDay"])).weekday() < 5)


def plan_pattern(pricing, plan):
    windows, fleets = plan_fields(pricing, plan)
    monday = date(2026, 11, 2)
    return tuple(sum(fleets for w in windows if w["from"] <= hour < w["to"] and (plan["days"] == "every" or (monday + timedelta(days=weekday + w["localDay"])).weekday() < 5)) for weekday in range(7) for hour in range(24))


def tranche_spans(pricing, plan):
    windows, fleets = plan_fields(pricing, plan)
    start = date.fromisoformat(plan["start"])
    end = term_end(start, plan["months"])
    found, cursor = [], start
    while cursor <= end:
        last = min(date(cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1]), end)
        found.append((cursor, last, sum(day_hours(windows, plan["days"], fleets, cursor + timedelta(days=n)) for n in range((last - cursor).days + 1))))
        cursor = last + timedelta(days=1)
    return found


def tranches(pricing, plan):
    return [hours for _, _, hours in tranche_spans(pricing, plan)]


def hours_before(pricing, plan, first, last, moment):
    windows, fleets = plan_fields(pricing, plan)
    local = moment.astimezone(KST)
    day, whole = local.date(), min(last, local.date() - timedelta(days=1))
    total = sum(day_hours(windows, plan["days"], fleets, first + timedelta(days=n)) for n in range((whole - first).days + 1)) if whole >= first else 0
    if first <= day <= last:
        upto = local.hour + (1 if (local.minute, local.second, local.microsecond) != (0, 0, 0) else 0)
        total += sum(max(0, min(w["to"], upto) - w["from"]) * fleets for w in windows if plan["days"] == "every" or (day + timedelta(days=w["localDay"])).weekday() < 5)
    return total


def kst_start(day):
    return datetime(day.year, day.month, day.day, tzinfo=KST).astimezone(timezone.utc)


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
        self.pool_upstream = env.get("SPARKPIPE_POOL_UPSTREAM", "").rstrip("/")
        self.pool_key = read_secret(env.get("SPARKPIPE_POOL_KEY_FILE")).decode()
        self.sandbox_accelerators = json.loads(env.get("SPARKPIPE_PAY_SANDBOX_ACCELERATORS") or "[]") if self.sandbox else []
        self.reconcile_seconds = int(env.get("SPARKPIPE_PAY_RECONCILE_SECONDS", "300"))
        self.smtp = env.get("SPARKPIPE_PAY_SMTP", "127.0.0.1:25")
        self.mail_from = env.get("SPARKPIPE_PAY_MAIL_FROM", "SparkPipe <noreply@sparkpipe.ai>")
        self.notify = env.get("SPARKPIPE_PAY_NOTIFY", "hyor@sparkpipe.ai")
        self.supplier = {"ko": env.get("SPARKPIPE_PAY_SUPPLIER_ADDRESS_KO", "").strip(), "en": env.get("SPARKPIPE_PAY_SUPPLIER_ADDRESS_EN", "").strip()}

    def path(self, name):
        return os.path.join(self.data, name)

    def pricing(self):
        with open(os.path.join(self.site, "pricing.js"), encoding="utf-8") as handle:
            text = handle.read()
        return json.loads(text[text.index("{"):text.rindex("}") + 1])

    def terms(self, name):
        with open(os.path.join(self.site, "terms", name), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    def version(self, lang):
        with open(os.path.join(self.site, "agreement-ko.html" if lang == "ko" else "agreement.html"), encoding="utf-8") as handle:
            found = re.search(r'<p class="effective">([^<]+)</p>', handle.read())
        if not found:
            raise ValueError(f"the published Customer Agreement ({lang}) states no version")
        return found.group(1)

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

    def session_key(self):
        return self.secret_file("session_key")

    def signing_key(self):
        return Ed25519PrivateKey.from_private_bytes(self.secret_file("signing_key"))


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
        self.plans = {}
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
        if kind == "account":
            self.accounts[data["account"]] = dict(data, created=line["at"], accepted=None)
            self.emails[data["email"]] = data["account"]
        elif kind == "acceptance":
            self.accounts[data["account"]]["accepted"] = dict({name: data.get(name) for name in TERMS}, at=line["at"])
        elif kind == "order":
            self.orders[data["order"]] = dict(data, created=line["at"], state="created", invoice=None, link=None, issued=False, paid=None)
        elif kind == "invoice":
            self.orders[data["order"]].update(invoice=data["invoice"], link=data["link"], state="unpaid")
            self.invoices[data["invoice"]] = data["order"]
        elif kind == "invoice_failed":
            self.orders[data["order"]]["state"] = "failed"
        elif kind == "invoice_state":
            self.orders[data["order"]]["state"] = data["state"]
            if "received" in data:
                self.orders[data["order"]]["received"] = data["received"]
        elif kind == "payment":
            self.orders[data["order"]].update(state="completed", paid=line["at"], received=data["received"])
        elif kind == "hours_issued":
            self.orders[data["order"]].update(issued=True, issued_hours=data["hours"], paid_net=data["net"])
        elif kind == "plan_order":
            self.plans[data["plan"]] = dict(data, id=data["plan"], created=line["at"], ended=None)
        elif kind == "plan_end":
            self.plans[data["plan"]]["ended"] = {"day": data["day"], "reason": data["reason"]}
        elif kind in market.KINDS:
            self.market.apply(line)

    def waiting(self, order, now):
        return order["state"] not in CLOSED and order["state"] != "completed" and not order["issued"] and (now - stamp_time(order["created"])).total_seconds() < PENDING_SECONDS

    def prepaid(self, plan):
        return sorted((stamp_time(order["paid"]), Decimal(order["paid_net"])) for order in self.orders.values() if order.get("plan") == plan and order["issued"])

    def pending(self, plan, now):
        return sum((Decimal(order["net"]) for order in self.orders.values() if order.get("plan") == plan and self.waiting(order, now)), Decimal(0))

    def paid_hours(self, plan, without=None):
        return sum((Decimal(order["issued_hours"]) for order in self.orders.values() if order.get("plan") == plan and order["issued"] and order["order"] != without), Decimal(0))


def standing(pricing, book, plan, now):
    windows, fleets = plan_fields(pricing, plan)
    start = date.fromisoformat(plan["start"])
    price = money(plan["hourPrice"])
    hours = tranches(pricing, plan)
    end = term_end(start, plan["months"])
    today = local_day(now)
    paid = book.prepaid(plan["id"])
    paid_net = sum((net for _, net in paid), Decimal(0))
    pending = book.pending(plan["id"], now)
    total = price * sum(hours)
    if plan["payment"] == "upfront":
        gates = [{"name": "Payment", "tranches": None, "dueDay": 1}]
    else:
        gates = pricing["gates"]
    ended, stop, missed = None, plan.get("ended"), False
    if today > end:
        ended = f"The term ended on {end.isoformat()}."
    elif stop and today >= date.fromisoformat(stop["day"]):
        ended = f"The plan ended on {stop['day']}: {stop['reason']}"
    elif plan["payment"] == "gates" and not paid and today > start + timedelta(days=pricing["phaseADays"] - 1):
        ended = "Phase A ended with no payment, so the plan ended with nothing owed."
    elif plan["payment"] == "upfront" and today > start and paid_net < total:
        ended = "The plan was not paid in full by its Start Date, so it ended."
    next_gate = None
    for gate in gates:
        due = start + timedelta(days=gate["dueDay"] - 1)
        need = price * sum(hours if gate["tranches"] is None else hours[:gate["tranches"]])
        cutoff = datetime(due.year, due.month, due.day, 23, 59, 59, tzinfo=KST)
        have = sum((net for moment, net in paid if moment <= cutoff), Decimal(0))
        if today > due:
            if plan["payment"] == "gates" and paid and have < need and not ended:
                ended, missed = f"{gate['name']} was missed on {due.isoformat()}, so no further prepayment can be made.", True
        elif next_gate is None and paid_net + pending < need:
            next_gate = {"name": gate["name"], "due": due.isoformat(), "amount": text_money(need - paid_net - pending)}
    held = None if ended else end
    if missed:
        covered, cursor = book.paid_hours(plan["id"]), Decimal(0)
        for _, last, count in tranche_spans(pricing, plan):
            held, cursor = last, cursor + count
            if cursor >= covered:
                break
        held = held if held >= today else None
    if held and stop:
        held = min(held, date.fromisoformat(stop["day"]) - timedelta(days=1))
    return {
        "plan": plan["id"],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "months": plan["months"],
        "windows": [w["id"] for w in windows],
        "days": plan["days"],
        "fleets": fleets,
        "lock": list(plan.get("lock") or []),
        "payment": plan["payment"],
        "price": text_money(price),
        "hours": sum(hours),
        "total": text_money(total),
        "paid": text_money(paid_net),
        "pending": text_money(pending),
        "remaining": text_money(max(Decimal(0), total - paid_net - pending)),
        "nextGate": None if ended else next_gate,
        "ended": ended,
        "heldUntil": held.isoformat() if held and held >= start else None,
    }


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


def money_field(value):
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        amount = None
    if amount is None or not amount.is_finite() or amount.as_tuple().exponent < -2:
        raise Refusal("Enter an amount in US dollars, to the cent.")
    return money(amount)


def coin_field(value):
    if value not in COINS:
        raise Refusal("Choose USDC or USDT and the network you will pay on.")
    return value


def statement(cfg, company, terms):
    return (f"For {company}, I accept the SparkPipe Customer Agreement, which includes the Hour Market Rules, and confirm that {company} buys and uses SparkPipe for business, professional, commercial or independent research and development purposes, not as a household consumer. "
            f"Customer Agreement, Korean text, which prevails: SHA-256 {terms['ko']} ({cfg.public}/terms/agreement-ko.md). "
            f"English text: SHA-256 {terms['en']} ({cfg.public}/terms/agreement-en.md). "
            f"Hour Market Rules, Korean text, which prevails: SHA-256 {terms['rules_ko']} ({cfg.public}/terms/market-rules-ko.md). "
            f"English text: SHA-256 {terms['rules_en']} ({cfg.public}/terms/market-rules-en.md).")


def party_field(value, label, size):
    found = text_field(value, label, size)
    if "|" in found:
        raise Refusal(f"The {label} cannot hold a | character.")
    return found


def long_date(day, lang):
    day = date.fromisoformat(day) if isinstance(day, str) else day
    return f"{day.year}년 {day.month}월 {day.day}일" if lang == "ko" else f"{day.day} {calendar.month_name[day.month]} {day.year}"


def fleet_name(fleet, lang):
    return ("플릿 " if lang == "ko" else "Fleet ") + fleet[1:]


def window_names(windows, lang):
    ko = lang == "ko"
    named = [(f"{WINDOW_KO.get(w['id'], w['name'])} 시간창" if ko else f"{w['name']} Window ") + f"({w['from']:02d}:00–{w['to']:02d}:00 KST)" for w in windows]
    full = sum(w["to"] - w["from"] for w in windows) == 24
    return ("종일: " if ko else "Full day: ") * full + ", ".join(named)


def plan_line(plan):
    windows = "Full day" if sum(w["to"] - w["from"] for w in plan["windows"]) == 24 else " + ".join(w.get("name", w["id"]) for w in plan["windows"])
    lock = [f"locked {', '.join(plan['lock'])}"] if plan.get("lock") else []
    return " · ".join([f"{plan['months']} months", windows, "Every day" if plan["days"] == "every" else "Weekdays", f"{plan['fleets']} fleet" + "s" * (plan["fleets"] > 1)] + lock)


def order_rows(record, lang):
    ko = lang == "ko"
    c, a, s = record["customer"], record["agreement"], record["supplier"]
    usd = lambda value: format(Decimal(value), ",.2f")
    if c["kind"] == "company":
        customer = f"{c['name']}. {c['jurisdiction']}의 법률에 따라 설립된 회사로서, 등록번호 {c['registration']}, 소재지 {c['address']}" if ko else f"{c['name']}, a company organised under the laws of {c['jurisdiction']}, registration number {c['registration']}, registered address {c['address']}"
    else:
        customer = f"{c['name']}. 개인, 주소 {c['address']}" if ko else f"{c['name']}, an individual, of {c['address']}"
    gates = record["schedule"]
    fee, allowance = format((Decimal(record["releaseFee"]) * 100).normalize(), "f"), record["maintenanceHoursPerWeek"]
    rates = ("; ", ", ")[ko].join(f"{fleet_name(fleet, lang)}: {rate}" for fleet, rate in record["rates"])
    models = ", ".join(m["name"] for m in record["models"])
    if ko:
        payment = "일시 선납: 개시일 전까지 플랜 기간의 대금" if record["payment"] == "upfront" else "게이트 선납: " + ", ".join(f"게이트 {g['name'].split()[-1]} {long_date(g['due'], lang)}" for g in gates) + "까지"
        return [
            ("주문번호", record["plan"]),
            ("고객", customer),
            ("고객 계약(제9.1조)", f"{a['version']['ko']}. 국문 SHA-256 {a['ko']}, 영문 SHA-256 {a['en']}. 시간 마켓 규칙 국문 SHA-256 {a['rules_ko']}, 영문 SHA-256 {a['rules_en']}"),
            ("개시일(제16.3조)", long_date(record["start"], lang)),
            ("플랜 기간(제16.3조)", f"{record['months']}개월, {long_date(record['end'], lang)} 종료"),
            ("시간창(제17.1조)", window_names(record["windows"], lang)),
            ("이용 요일(제17.2조)", "매일" if record["days"] == "every" else "평일"),
            ("플릿 수(제18.1조)", str(record["fleets"])),
            ("플릿 고정(제19조)", ", ".join(fleet_name(f, lang) for f in record["lock"]) or "없음"),
            ("시간당 가격(제16.4조)", f"플릿시간당 미화 {usd(record['hourPrice'])}달러(USD), 부가가치세 별도, 플릿 고정 포함, 플랜 기간 동안 고정"),
            ("플랜 기간의 시간(제17.3조)", f"{record['hours']:,}시간"),
            ("플랜 기간의 대금(제16.4조)", f"미화 {usd(record['total'])}달러(USD), 부가가치세 별도"),
            ("납부 방식(제20조)", payment),
            ("선택 모델(제7.2조)", models),
            ("성능 기준(제23.2조)", f"기준치의 {record['threshold']}%"),
            ("계정 이메일(제2.2조)", record["email"]),
            ("공급자 통지 주소 및 이메일(제11.7조)", f"{s['ko']}, {s['email']}"),
            ("고객 통지 주소 및 이메일(제11.7조)", f"{c['address']}, {c['notice']}"),
            ("고객의 부가가치세 지위(제3.2조)", f"대한민국 사업자등록번호 {c['brn']}" if c["brn"] else "대한민국 내 사업장 없음"),
            ("지급 통화 및 계좌(제3.3조)", "미국 달러 스테이블코인, 제3.3조 (다)목"),
            ("공개 설정(제9.3조)", f"플랜 주문서 성립 시 공개된 설정: 반납 수수료 {fee}%, 점검 허용 시간 플릿당 매주 {allowance}시간, 시간창 시각은 위 시간창란에 기재된 바에 따름"),
            ("플릿 요율(제16.5조)", f"플랜 주문서 성립 시 플릿 목록에 공개된 요율: {rates}"),
        ]
    payment = "Upfront: the Price of the Term before the Start Date" if record["payment"] == "upfront" else "Gated: " + ", ".join(f"{g['name']} by {long_date(g['due'], lang)}" for g in gates)
    return [
        ("Order number", record["plan"]),
        ("Customer", customer),
        ("Customer Agreement (clause 9.1)", f"{a['version']['en']}; SHA-256 of the Korean text {a['ko']} and of the English text {a['en']}; Hour Market Rules, SHA-256 of the Korean text {a['rules_ko']} and of the English text {a['rules_en']}"),
        ("Start Date (clause 16.3)", long_date(record["start"], lang)),
        ("Term (clause 16.3)", f"{record['months']} months, ending on {long_date(record['end'], lang)}"),
        ("Windows (clause 17.1)", window_names(record["windows"], lang)),
        ("Days (clause 17.2)", "Every day" if record["days"] == "every" else "Weekdays"),
        ("Fleets (clause 18.1)", str(record["fleets"])),
        ("Fleet lock (clause 19)", ", ".join(fleet_name(f, lang) for f in record["lock"]) or "None"),
        ("Hour Price (clause 16.4)", f"USD {usd(record['hourPrice'])} per fleet-hour, exclusive of VAT, including any Fleet lock, fixed for the Term"),
        ("Hours in the Term (clause 17.3)", f"{record['hours']:,}"),
        ("Price of the Term (clause 16.4)", f"USD {usd(record['total'])}, exclusive of VAT"),
        ("Payment schedule (clause 20)", payment),
        ("Chosen Models (clause 7.2)", models),
        ("Performance Threshold (clause 23.2)", f"{record['threshold']}% of the Baseline"),
        ("Account email (clause 2.2)", record["email"]),
        ("Supplier notice address and email (clause 11.7)", f"{s['en']}; {s['email']}"),
        ("Customer notice address and email (clause 11.7)", f"{c['address']}; {c['notice']}"),
        ("Customer VAT status (clause 3.2)", f"Korean business registration number {c['brn']}" if c["brn"] else "No place of business in the Republic of Korea"),
        ("Payment currency and account (clause 3.3)", "US dollar stablecoin, clause 3.3(c)"),
        ("Published settings (clause 9.3)", f"As published when this Plan Order is made: release fee {fee}%; maintenance allowance {allowance} hours a week per Fleet; window times as stated under Windows"),
        ("Fleet Rates (clause 16.5)", f"As published on the Fleet List when this Plan Order is made: {rates}"),
    ]


def order_text(record):
    blocks = []
    for lang, title, head in (("ko", "플랜 주문서", "| 항목 | 기재 사항 |"), ("en", "Plan Order", "| Item | Entry |")):
        lead = "본 플랜 주문서는 SparkPipe 고객 계약에 따라 체결된다." if lang == "ko" else "This Plan Order is made under the SparkPipe Customer Agreement."
        blocks.append("\n".join([f"# {title} {record['plan']}", "", lead, "", head, "| --- | --- |"] + [f"| {item} | {entry} |" for item, entry in order_rows(record, lang)]))
    return "\n\n".join(blocks) + "\n"


def text_rows(text):
    found, rows = {}, None
    for line in text.splitlines():
        if line.startswith("# "):
            rows = found.setdefault("ko" if not found else "en", [])
        elif line.startswith("| ") and rows is not None and not line.startswith("| --- "):
            rows.append(line[2:-2].split(" | ", 1))
    return {lang: rows[1:] for lang, rows in found.items()}


def shortfall(cover, life, pending):
    low = cover.low()
    return max(Decimal(0), -low if low is not None else Decimal(0)) + max(Decimal(0), pending - cover.headroom(life))


def plan_mail(cfg, record, pricing):
    dates = ", ".join(f"{g['name']} by {long_date(g['due'], 'en')}" for g in record["schedule"])
    return "\n".join([
        f"{record['company']} placed Plan Order {record['plan']} under the SparkPipe Customer Agreement.",
        "",
        f"{plan_line(record)}: {record['hours']:,} hours from {long_date(record['start'], 'en')} to {long_date(record['end'], 'en')}, at {record['hourPrice']} USD an hour.",
        f"Price of the Term: {record['total']} USD before VAT, paid " + ("in full by the Start Date." if record["payment"] == "upfront" else f"by gates: {dates}. Phase A runs to day {pricing['phaseADays']}: nothing is due, and you withdraw by not paying."),
        f"Pay toward it from your account: {cfg.public}/account.html",
        f"The Plan Order, ready to print: {cfg.public}/plan-order.html?plan={record['plan']}",
        "",
        f"You accepted the text below while signed in to {record['email']}. Its SHA-256 is {record['hash']}. We signed that hash with our Ed25519 key {record['signature']['key']}; the signature is {record['signature']['value']}.",
        "",
        record["text"],
        "HYOR AI Tech Ltd. (SparkPipe), Seoul. Reply to this email with any question.",
    ])


def invoice_body(cfg, quote):
    if quote["product"] == "credits":
        name = f"{quote['hours']} SparkPipe hour-credits"
        detail = f"Customer Agreement; {quote['price']} USD each, usable for 12 months"
        quantity = {"value": 1, "type": "quantity"}
    else:
        name = f"Prepayment, plan {quote['plan']}"
        detail = f"{quote['hours']} fleet-hours at {quote['price']} USD"
        quantity = {"value": 1, "type": "quantity"}
    back = f"{cfg.public}/account.html?order={quote['order']}"
    return {
        "currency": USD,
        "items": [{"customId": quote["order"], "name": name, "description": detail, "quantity": quantity, "amount": quote["net"], "tax": quote["vat"]}],
        "amount": {"breakdown": {"subtotal": quote["net"], "taxTotal": quote["vat"]}, "total": quote["total"]},
        "invoiceId": quote["order"],
        "description": name,
        "buyer": {"companyName": quote["company"], "emailAddress": quote["email"]},
        "notesToRecipient": f"SparkPipe issues the hours to the account {quote['email']} once this payment completes.",
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


def hours_number(value):
    value = Decimal(str(value))
    return int(value) if value == value.to_integral_value() else format(value.normalize(), "f")


def covered(order, paid):
    share = min(Decimal(1), paid / Decimal(order["total"]))
    net = (Decimal(order["net"]) * share).quantize(CENT, rounding=ROUND_DOWN)
    if order["product"] == "credits":
        return market.text_hc(net / Decimal(order["price"])), text_money(net)
    return format((net / Decimal(order["price"])).quantize(HOUR_STEP, rounding=ROUND_DOWN), "f"), text_money(net)


def paid_with(remote):
    return sorted({str(p["paymentCurrencySymbol"]) for p in remote["payments"]})


def receipt(cfg, order, issued, received, paid):
    coin = ", ".join(sorted({symbol.split(".")[0] for symbol in paid}))
    if order["product"] == "credits":
        what = f"{issued['hours']} hour-credits, usable until {market.stamp_time(issued['expires']).astimezone(KST).strftime('%Y-%m-%d')}"
    else:
        what = f"a prepayment of {issued['net']} USD under plan {order['plan']}: {issued['hours']} fleet-hours"
    if issued.get("short"):
        lines = [
            f"Order {order['order']} was for {order['total']} USD; {received} USD arrived. We issued what it covers: {what}.",
            "",
            f"Received in {coin}: {received} USD, VAT included",
        ]
    else:
        lines = [
            f"We received your payment for order {order['order']}: {what}.",
            "",
            f"Subtotal: {order['net']} USD",
            f"VAT: {order['vat']} USD",
            f"Total paid in {coin}: {order['total']} USD",
        ]
    return "\n".join(lines + [
        "",
        f"They are in your account: {cfg.public}/market.html" if order["product"] == "credits" else f"They are in your account: {cfg.public}/account.html",
        "",
        "HYOR AI Tech Ltd. (SparkPipe), Seoul. Reply to this email with any question.",
    ])


class App:
    def __init__(self, cfg, book):
        self.cfg = cfg
        self.book = book
        self.key = cfg.session_key()
        self.lock = threading.Lock()
        self.codes = {}
        self.hits = collections.defaultdict(collections.deque)
        self.jobs = queue.Queue()
        self.mails = queue.Queue()
        self.committed = None
        self.market = market.Market(self, cfg.sandbox, lambda: clock())

    def throttle(self, kind, who):
        count, window = LIMITS[kind]
        now = time.monotonic()
        with self.lock:
            hits = self.hits[(kind, who)]
            while hits and hits[0] < now - window:
                hits.popleft()
            if len(hits) >= count:
                raise Refusal("Too many requests. Wait a few minutes and try again.", 429)
            hits.append(now)

    def mail(self, to, subject, text):
        if not self.cfg.sandbox:
            self.mails.put((to, subject, text))

    def reset(self):
        if not self.cfg.sandbox:
            raise Refusal("Not found.", 404)
        with self.book.lock:
            os.replace(self.book.path, self.book.path + "." + clock().strftime("%Y%m%d%H%M%S"))
            self.book = Book(self.book.path)
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

    def signed_in(self, headers):
        jar = {}
        for part in (headers.get("Cookie") or "").split(";"):
            name, _, value = part.strip().partition("=")
            jar[name] = value
        account = session_account(self.key, jar.get(COOKIE), int(clock().timestamp()))
        return account if account in self.book.accounts else None

    def need(self, headers):
        account = self.signed_in(headers)
        if not account:
            raise Refusal("Sign in to continue.", 401)
        return account

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
            account = self.book.emails.get(email)
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
        expires = int(clock().timestamp()) + SESSION_SECONDS
        return self.summary(account), session_token(self.key, account, expires)

    def plan_terms(self, account, start):
        pricing, day = self.cfg.pricing(), local_day(start)
        for plan in self.plans(account):
            plan_fields(pricing, plan)
            begins = date.fromisoformat(plan["start"])
            if begins <= day <= term_end(begins, plan["months"]):
                return {"advanceMultiplier": market.published_terms(pricing["market"])["advanceMultiplier"], "releaseFee": format(Decimal(str(plan["releaseFee"])), "f")}
        return None

    def plans(self, account):
        return [p for p in self.book.plans.values() if p["account"] == account]

    def commitment(self, pricing, plan, until):
        return (
            plan.get("account") or self.book.emails.get(str(plan.get("email", "")).lower()),
            kst_start(date.fromisoformat(plan["start"])),
            kst_start(until + timedelta(days=1)),
            plan_pattern(pricing, plan),
            tuple(plan.get("lock") or ()),
            plan["maintenanceHoursPerWeek"],
            plan["id"],
        )

    def commitments(self, now, extra=None):
        key = (local_day(now), self.book.count)
        if extra is None and self.committed and self.committed[0] == key:
            return self.committed[1]
        pricing, found = self.cfg.pricing(), []
        for plan in list(self.book.plans.values()) + list(extra or []):
            until = standing(pricing, self.book, plan, now)["heldUntil"]
            if until:
                found.append(self.commitment(pricing, plan, date.fromisoformat(until)))
        found = tuple(found)
        if extra is None:
            self.committed = (key, found)
        return found

    def plans_open(self):
        if self.cfg.sandbox:
            raise Refusal("Plan Orders are placed on the live service, not in the sandbox.", 404)
        if not (self.cfg.supplier["ko"] and self.cfg.supplier["en"]):
            raise Refusal("Online Plan Orders are not open yet. Ask us for a quote, or write to hyor@sparkpipe.ai.", 503, "plans_closed")

    def customer_field(self, value, record):
        value = value if isinstance(value, dict) else {}
        if value.get("kind") not in PARTIES:
            raise Refusal("Say whether the Customer is a company or an individual.")
        found = {"kind": value["kind"], "name": record["company"], "address": party_field(value.get("address"), "address", 300), "notice": email_field(value.get("notice") or record["email"])}
        if value["kind"] == "company":
            found.update(jurisdiction=party_field(value.get("jurisdiction"), "country or state the company is organised in", 100), registration=party_field(value.get("registration"), "company registration number", 60))
        found["brn"] = optional_text(value.get("brn"), 20)
        if found["brn"] and not BRN.match(found["brn"]):
            raise Refusal("Give the Korean business registration number as 123-45-67890, or leave it empty.")
        return found

    def plan_choice(self, pricing, body, account, now, ident):
        offer, record = pricing["plans"], self.book.accounts[account]
        months = body.get("months")
        term = next((t for t in offer["terms"] if not isinstance(months, bool) and t["months"] == months), None)
        if not term:
            raise Refusal("Choose one of the terms we offer.")
        try:
            start = date.fromisoformat(body["start"])
        except (KeyError, TypeError, ValueError):
            raise Refusal("Give the Start Date as YYYY-MM-DD.")
        today = local_day(now)
        first = max(today, date.fromisoformat(offer["earliestStart"]))
        last = max(first, today + timedelta(days=offer["startWithinDays"]))
        if not first <= start <= last:
            raise Refusal(f"Choose a Start Date from {long_date(first, 'en')} to {long_date(last, 'en')}.")
        published, chosen = {w["id"]: w for w in pricing["windows"]}, body.get("windows")
        if not isinstance(chosen, list) or not chosen or not all(isinstance(w, str) and w in published for w in chosen) or len(set(chosen)) != len(chosen):
            raise Refusal("Choose one or more of the windows.")
        fleets = body.get("fleets")
        if isinstance(fleets, bool) or not isinstance(fleets, int) or not 1 <= fleets <= offer["maxFleets"]:
            raise Refusal(f"Choose 1 to {offer['maxFleets']} fleets. For more, ask us for a quote.")
        if body.get("days") not in DAYS:
            raise Refusal("Choose Every day or Weekdays.")
        lock = body.get("lock") is True
        if lock and (len(chosen) != len(published) or body["days"] != "every"):
            raise Refusal("A fleet lock comes only with a Full day on Every day.")
        listed, models = {m["id"]: m for m in pricing["models"]}, body.get("models")
        if not isinstance(models, list) or not models or not all(isinstance(m, str) and m in listed for m in models) or len(set(models)) != len(models):
            raise Refusal("Choose at least one of the models.")
        price = money(Decimal(str(term["hourPrice"])) + (Decimal(str(offer["lockPremium"])) if lock else 0))
        plan = {
            "id": ident, "plan": ident, "account": account, "email": record["email"], "company": record["company"],
            "start": start.isoformat(), "months": months, "windows": [{key: w[key] for key in ("id", "name", "from", "to", "localDay")} for w in pricing["windows"] if w["id"] in chosen],
            "days": body["days"], "fleets": fleets, "lock": [], "hourPrice": text_money(price), "payment": term["payment"],
            "releaseFee": format(Decimal(str(pricing["market"]["releaseFee"])), "f"), "maintenanceHoursPerWeek": offer["maintenanceHoursPerWeek"],
        }
        details = {
            "customer": self.customer_field(body.get("customer"), record), "models": [{"id": m, "name": listed[m]["name"]} for m in listed if m in models],
            "threshold": THRESHOLD, "currency": "stablecoin", "rates": [[f["id"], format(f["rate"].quantize(CENT), "f")] for f in self.market.fleets(pricing)],
        }
        return plan, details, lock

    def plan_room(self, pricing, plan, now, current):
        added = self.commitment(pricing, plan, term_end(date.fromisoformat(plan["start"]), plan["months"]))
        before, after = self.market.cover(now, pricing, current), self.market.cover(now, pricing, current + (added,))
        moment, pattern = max(added[1], after.first), added[3]
        while moment < added[2]:
            local = moment.astimezone(KST)
            if pattern[local.weekday() * 24 + local.hour] and after.clash(moment) and not before.clash(moment):
                return Refusal(f"We cannot keep the capacity this plan needs at {market.kst_text(moment)}: it is already reserved or kept for another plan, or the fleets are not serving yet. Choose a later Start Date, other windows or fewer fleets.", 409, "capacity")
            moment += market.HOUR
        life, pending = market.year_later(now), self.market.pending()
        if shortfall(after, life, pending) > shortfall(before, life, pending):
            return Refusal("Hour-credits already sold need the free fleet-hours this plan would keep. Choose a later Start Date, other windows or fewer fleets.", 409, "capacity")
        return None

    def placed(self, pricing, plan, lock, now):
        current = self.commitments(now)
        if not lock:
            refusal = self.plan_room(pricing, plan, now, current)
            if refusal:
                raise refusal
            return plan
        refusal = Refusal(f"We have fewer than {plan['fleets']} serving fleets to lock. Choose fewer fleets, or a plan without a lock.", 409, "capacity")
        for fleets in itertools.combinations([f["id"] for f in self.market.fleets(pricing) if f["serving"]], plan["fleets"]):
            trial = dict(plan, lock=list(fleets))
            found = self.plan_room(pricing, trial, now, current)
            if not found:
                return trial
            refusal = found
        raise refusal

    def plan_record(self, pricing, plan, details):
        spans, start = tranche_spans(pricing, plan), date.fromisoformat(plan["start"])
        hours = sum(count for _, _, count in spans)
        schedule = [{"name": "Payment", "due": plan["start"]}] if plan["payment"] == "upfront" else [{"name": g["name"], "due": (start + timedelta(days=g["dueDay"] - 1)).isoformat()} for g in pricing["gates"]]
        record = dict({key: value for key, value in plan.items() if key != "id"}, **details)
        record.update(
            end=term_end(start, plan["months"]).isoformat(), hours=hours, total=text_money(money(plan["hourPrice"]) * hours), schedule=schedule,
            agreement=dict(self.current_terms(), version={"ko": self.cfg.version("ko"), "en": self.cfg.version("en")}),
            supplier={"ko": self.cfg.supplier["ko"], "en": self.cfg.supplier["en"], "email": self.cfg.notify},
        )
        record["text"] = order_text(record)
        record["hash"] = hashlib.sha256(record["text"].encode()).hexdigest()
        return record

    def draft(self, pricing, body, account, now, ident):
        plan, details, lock = self.plan_choice(pricing, body, account, now, ident)
        return self.plan_record(pricing, self.placed(pricing, plan, lock, now), details)

    def preview(self, pricing, record):
        net = Decimal(record["total"])
        vat = money(net * Decimal(str(pricing["vatRate"])))
        return {
            "plan": record["plan"], "hash": record["hash"], "text": record["text"], "rows": text_rows(record["text"]), "line": plan_line(record),
            "start": record["start"], "end": record["end"], "hours": record["hours"], "price": record["hourPrice"], "total": record["total"],
            "vat": text_money(vat), "withVat": text_money(net + vat), "lock": record["lock"], "payment": record["payment"], "schedule": record["schedule"],
        }

    def quote_plan(self, body, account, ip):
        self.throttle("plan", ip)
        self.plans_open()
        pricing = self.cfg.pricing()
        with self.book.lock:
            return self.preview(pricing, self.draft(pricing, body, account, clock(), new_id("PO")))

    def place_plan(self, body, account, ip, agent):
        self.throttle("plan", ip)
        self.plans_open()
        ident = body.get("plan")
        if not isinstance(ident, str) or not PLAN_ID.match(ident):
            raise Refusal("Review the Plan Order before placing it.")
        if body.get("agree") is not True:
            raise Refusal("Accept the Plan Order to place it.", 409, "agree")
        pricing, now = self.cfg.pricing(), clock()
        with self.book.lock:
            if ident in self.book.plans:
                raise Refusal("This Plan Order is already placed. It is in your account.", 409)
            waiting = [p for p in self.plans(account) if not standing(pricing, self.book, p, now)["ended"] and not self.book.paid_hours(p["id"])]
            if waiting:
                raise Refusal(f"Plan Order {waiting[0]['id']} has no payment yet. Pay toward it from your account, or wait until it lapses unpaid, before ordering another plan.", 409, "unpaid")
            if not self.accepted(account, self.current_terms()):
                if body.get("accept") is not True:
                    raise Refusal("The Customer Agreement has changed since you accepted it. Accept the current text to continue.", 409, "terms")
                self.accept(account, ip, agent)
            record = self.draft(pricing, body, account, now, ident)
            if not hmac.compare_digest(record["hash"].encode(), str(body.get("hash")).encode()):
                raise Refusal("The Plan Order changed since you reviewed it: the capacity, the fleets it locks or a published text moved. Review the new Order and accept it again.", 409, "changed")
            key = self.cfg.signing_key()
            record.update(source="account", accepted={"ip": ip, "agent": agent}, signature={
                "algorithm": "Ed25519", "key": key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex(), "value": key.sign(bytes.fromhex(record["hash"])).hex()})
            self.book.append("plan_order", record)
        self.mail(record["email"], f"Plan Order {ident}: your SparkPipe plan", plan_mail(self.cfg, record, pricing))
        self.mail(self.cfg.notify, f"New Plan Order: {record['company']}, {plan_line(record)}", f"{record['company']} <{record['email']}> placed Plan Order {ident}: {plan_line(record)}, from {record['start']}, {record['hours']:,} hours at {record['hourPrice']} USD, {record['total']} USD before VAT, {record['payment']}.")
        log(f"plan order {ident} for {account}")
        return self.plan_view(account, ident)

    def plan_key(self):
        return {"algorithm": "Ed25519", "key": self.cfg.signing_key().public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()}

    def plan_view(self, account, ident):
        pricing, now = self.cfg.pricing(), clock()
        with self.book.lock:
            plan = self.book.plans.get(ident)
            if not plan or plan["account"] != account:
                raise Refusal("No such plan on this account.", 404)
            view = standing(pricing, self.book, plan, now)
            view.update(
                line=plan_line(plan), company=plan["company"], created=plan["created"], source=plan.get("source"), hash=plan.get("hash"),
                rows=text_rows(plan["text"]) if plan.get("text") else None, text=plan.get("text"), signature=plan.get("signature"),
                accepted=plan["created"] if plan.get("source") == "account" else None, schedule=plan.get("schedule"),
                tranches=[{"from": first.isoformat(), "to": last.isoformat(), "hours": count} for first, last, count in tranche_spans(pricing, plan)],
            )
            return view

    def record_signed(self, pricing, plan, now):
        account = self.book.emails.get(str(plan.get("email", "")).lower())
        ident = plan.get("id")
        if not account or not isinstance(ident, str) or not ident or ident in self.book.plans:
            raise Refusal("A signed Plan Order needs an unused id and the email of an existing account.")
        record = {key: plan[key] for key in ("start", "months", "windows", "days", "fleets", "hourPrice", "payment", "releaseFee", "maintenanceHoursPerWeek")}
        record.update(id=ident, plan=ident, account=account, email=self.book.accounts[account]["email"], company=self.book.accounts[account]["company"], lock=list(plan.get("lock") or []), source="signed")
        tranches(pricing, record)
        with self.book.lock:
            refusal = self.plan_room(pricing, record, now, self.commitments(now))
            if refusal:
                raise refusal
            self.book.append("plan_order", {key: value for key, value in record.items() if key != "id"})
        return record

    def end_plan(self, ident, day, reason):
        plan = self.book.plans.get(ident)
        if not plan or plan["ended"] or not reason:
            raise Refusal("Name a Plan Order that has not ended, the first day it no longer holds, and the reason.")
        self.book.append("plan_end", {"plan": ident, "account": plan["account"], "day": day.isoformat(), "reason": reason})

    def grant_paid(self, ident, order_id, now):
        plan = self.book.plans.get(ident)
        if not plan:
            return
        pricing = self.cfg.pricing()
        spans = tranche_spans(pricing, plan)
        before, after = int(self.book.paid_hours(ident, order_id)), int(self.book.paid_hours(ident))
        if plan["payment"] == "upfront":
            if after < sum(count for _, _, count in spans):
                return
            before = 0
        cursor = 0
        for first, last, count in spans:
            old, new = min(count, max(0, before - cursor)), min(count, max(0, after - cursor))
            cursor += count
            gone = max(0, min(new, hours_before(pricing, plan, first, last, now)) - old)
            if new - old - gone > 0:
                self.book.append("tranche_grant", {"account": plan["account"], "month": first.strftime("%Y-%m"), "hours": new - old - gone, "plan": ident, "source": "payment", "order": order_id})

    def summary(self, account):
        pricing = self.cfg.pricing()
        now = clock()
        with self.book.lock:
            record = self.book.accounts[account]
            mine = [o for o in reversed(list(self.book.orders.values())) if o["account"] == account]
            return {
                "email": record["email"],
                "company": record["company"],
                "since": record["created"][:10],
                "termsCurrent": self.accepted(account, self.current_terms()),
                "hc": market.text_hc(self.book.market.balance(account, market.stamp(self.market.now()))),
                "orders"
: [self.shown(o) for o in mine[:50]],
                "plans": [dict(standing(pricing, self.book, p, now), line=plan_line(p), form=bool(p.get("text"))) for p in self.plans(account)],
            }

    def prepayment_terms(self, pricing, body, account, now):
        number = body.get("plan")
        found = next((p for p in self.plans(account) if isinstance(number, str) and p["id"] == number), None)
        if not found:
            raise Refusal("No plan with this number belongs to your account.", 404)
        net = money_field(body.get("amount"))
        if net < Decimal("1.00"):
            raise Refusal("The smallest prepayment is 1.00 USD.")
        state = standing(pricing, self.book, found, now)
        if state["ended"]:
            raise Refusal(state["ended"], 409)
        if net > Decimal(state["remaining"]):
            raise Refusal(f"At most {state['remaining']} USD remains to prepay, counting unpaid orders.", 409)
        price = money(found["hourPrice"])
        vat = money(net * Decimal(str(pricing["vatRate"])))
        return {
            "product": "plan", "plan": found["id"],
            "price": text_money(price), "hours": format((net / price).quantize(HOUR_STEP, rounding=ROUND_DOWN), "f"),
            "net": text_money(net), "vat": text_money(vat), "total": text_money(net + vat),
        }

    def credit_terms(self, pricing, body):
        offer = pricing["market"]
        net = money_field(body.get("amount"))
        if net < Decimal(str(offer["minPurchase"])) or net > 100000:
            raise Refusal(f"Buy at least {offer['minPurchase']} USD of HC, and at most 100,000 USD at a time.")
        price = money(Decimal(str(offer["hcPrice"])))
        vat = money(net * Decimal(str(pricing["vatRate"])))
        return {"product": "credits", "hours": market.text_hc(net / price), "price": text_money(price), "net": text_money(net), "vat": text_money(vat), "total": text_money(net + vat), "terms": market.published_terms(offer)}

    def buy(self, body, account, ip, agent):
        if self.cfg.sandbox:
            self.throttle("market", ip)
            pricing = self.cfg.pricing()
            quote = self.credit_terms(pricing, body)
            return self.market.buy(account, Decimal(quote["net"]), pricing)
        return self.order(dict(body, product="credits"), account, ip, agent)

    def order(self, body, account, ip, agent):
        self.throttle("order", ip)
        if self.cfg.sandbox:
            raise Refusal("The sandbox sells HC only, through the Hour Market test page.", 404)
        now = clock()
        coin = coin_field(body.get("coin"))
        pricing = self.cfg.pricing()
        with self.book.lock:
            record = self.book.accounts[account]
            if body.get("product") == "credits":
                quote = self.credit_terms(pricing, body)
                self.market.check_cover(Decimal(quote["hours"]), self.market.now(), pricing)
                if not self.accepted(account, self.current_terms()):
                    if body.get("accept") is not True:
                        raise Refusal("The Customer Agreement has changed since you accepted it. Accept the current text to continue.", 409, "terms")
                    self.accept(account, ip, agent)
            elif body.get("product") == "plan":
                quote = self.prepayment_terms(pricing, body, account, now)
            else:
                raise Refusal("Choose hour-credits or a plan prepayment.")
            quote.update(order=new_id("SP"), account=account, email=record["email"], company=record["company"], coin=coin)
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
        shown = {key: order.get(key) for key in ("order", "product", "hours", "net", "vat", "total", "plan", "coin", "state", "issued", "created", "paid", "received")}
        shown["issuedHours"] = None if order.get("issued_hours") is None else hours_number(order["issued_hours"])
        shown["checkout"] = order["link"] if order["state"] in ("unpaid", "pending") else None
        return shown

    def status(self, order_id):
        with self.book.lock:
            order = self.book.orders.get(order_id)
            if not order:
                raise Refusal("No such order.", 404)
            return self.shown(order)

    def market_state(self, account):
        state = self.market.public()
        state["account"] = None
        if account:
            record = self.book.accounts[account]
            state["account"] = dict(self.market.view(account), email=record["email"], company=record["company"])
        return state

    def market_action(self, path, body, account, ip):
        self.throttle("market", ip)
        actions = {
            "/market/reserve": lambda: self.market.reserve(account, body),
            "/market/release": lambda: self.market.release(account, body),
            "/market/cancel": lambda: self.market.cancel(account, body),
            "/market/bid": lambda: self.market.bid(account, body),
            "/market/keys": lambda: self.market.create_key(account, optional_text(body.get("note"), 100)),
            "/market/keys/revoke": lambda: self.market.revoke_key(account, body.get("id")),
        }
        sandbox = {
            "/market/sandbox/clock": lambda: self.market.move_clock(market.count_field(body.get("hours"), "number of hours", 9600)),
            "/market/sandbox/tranche": lambda: self.market.demo_tranche(account),
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
                        owners = [data[name] for name in ("account", "seller") if name in data]
                        ids = {data[name] for name in LEDGER_IDS if isinstance(data.get(name), str)}
                        if (owners and account in owners) or (not owners and ids & mine):
                            lines.append(line)
                            mine |= ids
            return {"head": self.book.head, "count": self.book.count, "lines": lines}


    def lead(self, body, ip, agent):
        self.throttle("lead", ip)
        lead = {
            "company": text_field(body.get("company"), "company name", 200),
            "email": email_field(body.get("email")),
            "name": optional_text(body.get("name"), 200),
            "fleets": optional_text(str(body.get("fleets") or ""), 20),
            "start": optional_text(body.get("start"), 40),
            "message": optional_text(body.get("message"), 4000),
        }
        line = dict(lead, at=clock().strftime("%Y-%m-%dT%H:%M:%SZ"), ip=ip, agent=agent)
        with self.lock, open(self.cfg.path("leads.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.mail(self.cfg.notify, f"Plan quote request: {lead['company']}", "\n".join(f"{key}: {value}" for key, value in lead.items()))
        log(f"lead from {lead['company']}")
        return {"ok": True}

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
                first = before == 0 and received > 0
                moved = first or abs(received - before) >= Decimal(1)
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
                log(f"order {order_id}: completed invoice does not match the order ({total} {currency}, {arrived} received); hours not issued")
                self.mail(self.cfg.notify, f"Payment to review: {order_id}", f"Order {order_id} for {order['total']} USD completed at CoinPayments as {total} {currency} with {arrived} USD received. No hours were issued. If it is right, `sparkpipe_pay.py release {order_id}` issues the hours the amount received covers.")
                return None
            hours, net = (order["hours"], order["net"]) if full else covered(order, received)
            if not Decimal(hours) > 0:
                log(f"order {order_id}: {arrived} USD received covers no hours; refund it in CoinPayments")
                self.mail(self.cfg.notify, f"Payment to review: {order_id}", f"Order {order_id} completed at CoinPayments with {arrived} USD received, which covers no hours. Refund it in CoinPayments.")
                return None
            paid = paid_with(remote)
            if order["state"] != "completed":
                self.book.append("payment", {"order": order_id, "invoice": order["invoice"], "total": order["total"], "received": arrived, "currency": currency, "paidWith": paid})
            issued = {"order": order_id, "account": order["account"], "product": order["product"], "hours": hours, "net": net, "released": released, "short": not full}
            if order["product"] == "plan":
                issued.update(plan=order["plan"])
            self.book.append("hours_issued", issued)
            if order["product"] == "plan":
                self.grant_paid(order["plan"], order_id, clock())
            if order["product"] == "credits":
                issued["expires"] = self.market.issue(order["account"], Decimal(hours), order_id, order["terms"])["expires"]
            log(f"order {order_id}: paid, {hours} hours issued to {order['account']}")
            self.mail(order["email"], f"Payment received: SparkPipe order {order_id}", receipt(self.cfg, order, issued, arrived, paid))
            subject = f"Paid in part: {order['company']}, {arrived} of {order['total']} USD" if not full else f"Paid: {order['company']}, {order['total']} USD"
            self.mail(self.cfg.notify, subject, f"Order {order_id}: {hours} of {order['hours']} hours for {order['company']} <{order['email']}>, {arrived} USD received of {order['total']} USD in {', '.join(paid)}.")
            return issued

    def work(self):
        while True:
            key = self.jobs.get()
            try:
                if key is None:
                    return
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
        for key in keys:
            self.jobs.put(key)

    def reconcile_forever(self):
        while True:
            time.sleep(self.cfg.reconcile_seconds)
            self.reconcile()


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "sparkpipe-pay"

    def log_message(self, fmt, *args):
        pass

    def reply(self, status, data, cookie=None):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if cookie is not None:
            cfg = self.server.app.cfg
            secure = "; Secure" if cfg.public.startswith("https://") else ""
            age = SESSION_SECONDS if cookie else 0
            self.send_header("Set-Cookie", f"{COOKIE}={cookie}; Max-Age={age}; Path={cfg.prefix}; HttpOnly; SameSite=Lax{secure}")
        self.end_headers()
        self.wfile.write(body)

    def reply_text(self, text):
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def refuse(self, error):
        data = {"error": str(error)}
        if error.reason:
            data["reason"] = error.reason
        self.reply(error.status, data)

    def pool_error(self, error):
        self.reply(error.status, {"error": {"message": str(error), "type": "invalid_request_error", "code": error.reason or "refused"}})

    def client_ip(self):
        forwarded = self.headers.get("X-Forwarded-For")
        return forwarded.split(",")[-1].strip() if forwarded else self.client_address[0]

    def agent(self):
        return (self.headers.get("User-Agent") or "")[:300]

    def bearer(self):
        header = self.headers.get("Authorization") or ""
        return header[7:].strip() if header.startswith("Bearer ") else ""

    def do_GET(self):
        app = self.server.app
        parts = urllib.parse.urlsplit(self.path)
        path, query = parts.path, urllib.parse.parse_qs(parts.query)
        if path == "/health":
            return self.reply(200, {"ok": True})
        if path == "/pool/v1/models":
            try:
                app.market.key_account(self.bearer())
            except Refusal as error:
                return self.pool_error(error)
            return self.reply(200, {"object": "list", "data": [{"id": m["id"], "object": "model", "owned_by": "sparkpipe"} for m in app.cfg.pricing()["market"]["pool"]["models"]]})
        try:
            app.throttle("read", self.client_ip())
            if path == "/account":
                return self.reply(200, app.summary(app.need(self.headers)))
            if path == "/market/state":
                return self.reply(200, app.market_state(app.signed_in(self.headers)))
            if path == "/plans/key":
                return self.reply(200, app.plan_key())
            match = PLAN_PATH.match(path)
            if match:
                return self.reply(200, app.plan_view(app.need(self.headers), match.group(1)))
            if path == "/market/calendar":
                days = query.get("days", ["7"])[0]
                return self.reply(200, app.market.calendar(app.signed_in(self.headers), int(days) if days.isdigit() else 7))
            if path == "/market/book":
                return self.reply(200, app.market.book_view())
            if path == "/market/ledger":
                return self.reply(200, app.ledger(app.need(self.headers)))
            if path == "/market/market.txt":
                return self.reply_text(app.market.market_text())
            if path == "/market/pool.txt":
                return self.reply_text(app.market.pool_text())
            match = ORDER_PATH.match(path)
            if not match:
                return self.reply(404, {"error": "Not found."})
            self.reply(200, app.status(match.group(1)))
        except Refusal as error:
            self.refuse(error)

    def do_POST(self):
        app = self.server.app
        path = urllib.parse.urlsplit(self.path).path
        limit = 262144 if path == "/coinpayments" else 8388608 if path.startswith("/pool/") else 16384
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = -1
        if length <= 0 or length > limit:
            return self.reply(413 if length > limit else 400, {"error": "The request body is missing or too large."})
        raw = self.rfile.read(length)
        if path == "/coinpayments":
            return self.reply(200 if app.webhook(self.headers, raw) else 401, {})
        origin = self.headers.get("Origin")
        if origin and origin != app.cfg.origin:
            return self.reply(403, {"error": "Cross-site requests are refused."})
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self.reply(400, {"error": "The request is not a JSON object."})
        if path == "/pool/v1/chat/completions":
            return self.pool(body)
        try:
            if path == "/code":
                return self.reply(200, app.send_code(body, self.client_ip()))
            if path == "/verify":
                summary, token = app.verify(body, self.client_ip(), self.agent())
                return self.reply(200, summary, cookie=token)
            if path == "/signout":
                return self.reply(200, {"ok": True}, cookie="")
            if path == "/order":
                return self.reply(200, app.order(body, app.need(self.headers), self.client_ip(), self.agent()))
            if path == "/lead":
                return self.reply(200, app.lead(body, self.client_ip(), self.agent()))
            if path == "/plans/quote":
                return self.reply(200, app.quote_plan(body, app.need(self.headers), self.client_ip()))
            if path == "/plans/order":
                return self.reply(200, app.place_plan(body, app.need(self.headers), self.client_ip(), self.agent()))
            if path == "/market/buy":
                return self.reply(200, app.buy(body, app.need(self.headers), self.client_ip(), self.agent()))
            if path.startswith("/market/"):
                return self.reply(200, app.market_action(path, body, app.need(self.headers), self.client_ip()))
            return self.reply(404, {"error": "Not found."})
        except Refusal as error:
            self.refuse(error)

    def stream_head(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def event(self, data):
        self.wfile.write(b"data: " + (data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()) + b"\n\n")
        self.wfile.flush()

    def pool(self, body):
        app = self.server.app
        try:
            app.throttle("pool", self.bearer()[-16:] or self.client_ip())
            if not app.cfg.sandbox and not app.cfg.pool_upstream:
                raise Refusal("The live shared pool opens once it is connected to the fleets.", 503, "pool_closed")
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise Refusal("Send the conversation as a non-empty messages list.", 400, "invalid_messages")
            extra = body.get("sparkpipe", {})
            if not isinstance(extra, dict):
                raise Refusal("Give sparkpipe as an object.", 400, "invalid_accelerator")
            options = market.accelerator_field(extra.get("accelerator"), app.cfg.pricing()["market"])
            ticket = app.market.pool_open(self.bearer(), body.get("model"), accelerated=options is not None)
            accel = app.market.accelerator_join(ticket, options, market.estimate_tokens(messages)) if options else None
        except Refusal as error:
            return self.pool_error(error)
        body = {k: v for k, v in body.items() if k != "sparkpipe"}
        if app.cfg.sandbox:
            return self.simulated(body, messages, ticket, accel)
        return self.forwarded(body, ticket, accel)

    def simulated(self, body, messages, ticket, accel):
        app = self.server.app
        limit = body.get("max_tokens") if isinstance(body.get("max_tokens"), int) and not isinstance(body.get("max_tokens"), bool) else 128
        text, tokens_in, tokens_out = market.simulate(ticket["model"], messages, min(max(limit, 1), 512))
        if accel and "job" in accel:
            seconds = Decimal(tokens_in) / accel["accelerator"]["speed"]
            time.sleep(float(min(seconds, 30)))
            accel = app.market.accelerator_finish(accel, float(seconds))
        app.market.pool_seat(ticket)
        made, ident = int(time.time()), "chatcmpl-" + secrets.token_hex(12)
        usage = {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": tokens_in + tokens_out}
        if body.get("stream") is True:
            self.stream_head()
            try:
                for word in text.split(" "):
                    self.event({"id": ident, "object": "chat.completion.chunk", "created": made, "model": ticket["model"], "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]})
                    time.sleep(0.025)
            finally:
                charge = dict(app.market.pool_close(ticket, tokens_in, tokens_out), accelerator=accel)
            self.event({"id": ident, "object": "chat.completion.chunk", "created": made, "model": ticket["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage, "sparkpipe": charge})
            return self.event(b"[DONE]")
        time.sleep(min(2.0, tokens_out / 40))
        charge = dict(app.market.pool_close(ticket, tokens_in, tokens_out), accelerator=accel)
        self.reply(200, {"id": ident, "object": "chat.completion", "created": made, "model": ticket["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage, "sparkpipe": charge})

    def forwarded(self, body, ticket, accel):
        app = self.server.app
        job = accel if accel and "job" in accel else None
        note = [accel]
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {app.cfg.pool_key}", "X-SparkPipe-Public": "1"}
        if job:
            headers["X-SparkPipe-Accelerator"] = job["accelerator"]["id"]
        else:
            app.market.pool_seat(ticket)

        def handed(usage, finished=True):
            if job and note[0] is job:
                reported = usage.get("sparkpipe_accelerator_seconds") if finished else None
                measured = min(time.monotonic() - job["started"], float(job["tokens"] / job["accelerator"]["speed"]))
                seconds = float(reported) if isinstance(reported, (int, float)) and not isinstance(reported, bool) and reported >= 0 else measured
                note[0] = app.market.accelerator_finish(job, seconds if finished else None)
                app.market.pool_seat(ticket)

        def closing():
            return dict(app.market.pool_close(ticket, tokens_in, tokens_out), accelerator=note[0])

        payload = json.dumps(dict(body, model=ticket["model"]), ensure_ascii=False).encode()
        request = urllib.request.Request(app.cfg.pool_upstream + "/chat/completions", data=payload, method="POST", headers=headers)
        tokens_in = tokens_out = 0
        closed = started = False
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                if body.get("stream") is not True:
                    data = json.loads(response.read())
                    usage = data.get("usage") or {}
                    handed(usage)
                    tokens_in, tokens_out = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
                    data["sparkpipe"] = closing()
                    closed = True
                    return self.reply(200, data)
                self.stream_head()
                started = True
                for line in response:
                    text = line.strip()
                    if not text.startswith(b"data: "):
                        continue
                    if text == b"data: [DONE]":
                        handed({})
                        closed = True
                        self.event({"object": "chat.completion.chunk", "choices": [], "sparkpipe": closing()})
                        return self.event(b"[DONE]")
                    try:
                        chunk = json.loads(text[6:])
                        usage = chunk.get("usage") or {}
                        handed(usage)
                        tokens_in = int(usage.get("prompt_tokens") or tokens_in)
                        tokens_out = int(usage.get("completion_tokens") or tokens_out + sum(1 for c in chunk.get("choices") or [] if (c.get("delta") or {}).get("content")))
                    except (ValueError, AttributeError):
                        pass
                    self.event(text[6:])
        except urllib.error.HTTPError as error:
            handed({}, finished=False)
            closed = True
            app.market.pool_close(ticket, tokens_in, tokens_out)
            detail = error.read()[:4096]
            self.send_response(error.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(detail)))
            self.end_headers()
            self.wfile.write(detail)
        except (urllib.error.URLError, OSError, ValueError) as error:
            log(f"pool upstream: {error}")
            handed({}, finished=False)
            if not closed:
                closed = True
                app.market.pool_close(ticket, tokens_in, tokens_out)
            if not started:
                self.pool_error(Refusal("The fleets did not answer. Try again in a minute.", 502, "upstream_error"))
        finally:
            handed({}, finished=False)
            if not closed:
                app.market.pool_close(ticket, tokens_in, tokens_out)


def start(cfg, port=None):
    os.makedirs(cfg.data, exist_ok=True)
    app = App(cfg, Book(cfg.path("ledger.jsonl")))
    threading.Thread(target=app.work, daemon=True).start()
    threading.Thread(target=app.deliver, daemon=True).start()
    threading.Thread(target=app.market.run, args=(log,), daemon=True).start()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", cfg.port if port is None else port), Handler)
    server.daemon_threads = True
    server.app = app
    return server


def main(argv):
    cfg = Config(os.environ)
    if argv[1:] == ["verify"]:
        book = Book(cfg.path("ledger.jsonl"))
        print(f"{book.count} lines, head {book.head}, {len(book.accounts)} accounts, {len(book.orders)} orders")
        return 0
    if len(argv) == 3 and argv[1] == "release":
        app = App(cfg, Book(cfg.path("ledger.jsonl")))
        order = app.book.orders.get(argv[2])
        if not order or not order["invoice"]:
            print(f"{argv[2]}: no order with an invoice", file=sys.stderr)
            return 1
        remote = call(cfg, "GET", "/merchant/invoices/" + order["invoice"])
        state, total, currency = remote_fields(remote)
        print(f"{argv[2]}: CoinPayments says {state}, {total} {currency}, {received_of(remote)} USD received; the order is {order['total']} USD for {order['hours']} hours")
        if state != "completed":
            print("not completed at CoinPayments: nothing issued. Accept the part payment there first, or refund it.", file=sys.stderr)
            return 1
        threading.Thread(target=app.deliver, daemon=True).start()
        issued = app.settle(argv[2], remote, released=True)
        app.mails.join()
        print(f"{argv[2]}: " + (f"issued {issued['hours']} hours for {issued['net']} USD before VAT" if issued else "nothing issued"))
        return 0 if issued else 1
    if len(argv) == 5 and argv[1] == "tranche":
        app = App(cfg, Book(cfg.path("ledger.jsonl")))
        account = app.book.emails.get(argv[2].lower())
        if not account or not re.match(r"^\d{4}-\d{2}$", argv[3]) or not argv[4].isdigit():
            print("usage: sparkpipe_pay.py tranche EMAIL YYYY-MM HOURS, for an existing account", file=sys.stderr)
            return 2
        print(app.market.grant_tranche(account, argv[3], int(argv[4])))
        return 0
    if len(argv) in (2, 3) and argv[1] == "cover":
        app = App(cfg, Book(cfg.path("ledger.jsonl")))
        listing = None
        if len(argv) == 3:
            with open(argv[2], encoding="utf-8") as handle:
                listing = json.load(handle)
        found = app.market.report(None if listing is None else app.commitments(app.market.now(), listing))
        tight = found["tightest"]
        print(f"cover at {found['now']}: serving {', '.join(found['fleets']) or 'no fleet'} from {found['first']}, {found['plans']} plans")
        print(f"free fleet-hours in the next 12 months, every day, after plan set-asides and the maintenance allowance: {found['free']} HC")
        print(f"unspent HC: {found['unspent']}; in unpaid orders: {found['pending']}; HC on sale now: {found['onSale']}")
        print("tightest lapse: none" if not tight else f"tightest lapse: {tight['lapse']}, {tight['slack']} HC of free hours to spare")
        print(f"hours where a set-aside cannot be kept: {found['clashes']}" + (f", first {found['firstClash']}" if found["firstClash"] else ""))
        return 1 if found["clashes"] or (tight and Decimal(tight["slack"]) < 0) else 0
    if len(argv) in (4, 5) and argv[1] == "maintenance":
        app = App(cfg, Book(cfg.path("ledger.jsonl")))
        try:
            found = app.market.schedule_maintenance(argv[2], argv[3], int(argv[4]) if len(argv) == 5 else None)
        except (Refusal, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        threading.Thread(target=app.deliver, daemon=True).start()
        app.mails.join()
        print(f"maintenance {found['maintenance']}: {found['fleet']} for {found['hours']} hours from {found['start']}; credited {', '.join(found['credited']) or 'nothing'}")
        return 0
    if len(argv) == 3 and argv[1] == "plan":
        app = App(cfg, Book(cfg.path("ledger.jsonl")))
        with open(argv[2], encoding="utf-8") as handle:
            listing = json.load(handle)
        for plan in listing if isinstance(listing, list) else [listing]:
            try:
                found = app.record_signed(cfg.pricing(), plan, app.market.now())
            except (Refusal, ValueError, KeyError, TypeError) as error:
                print(f"{plan.get('id') if isinstance(plan, dict) else plan}: {error}", file=sys.stderr)
                return 2
            print(f"{found['plan']}: recorded for {found['email']}, {plan_line(found)}, from {found['start']}")
        return 0
    if len(argv) == 5 and argv[1] == "end-plan":
        app = App(cfg, Book(cfg.path("ledger.jsonl")))
        try:
            app.end_plan(argv[2], date.fromisoformat(argv[3]), argv[4].strip()[:300])
        except (Refusal, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        print(f"{argv[2]}: holds no capacity from {argv[3]}")
        return 0
    if len(argv) == 5 and argv[1] == "credit":
        app = App(cfg, Book(cfg.path("ledger.jsonl")))
        account = app.book.emails.get(argv[2].lower())
        try:
            credits = market.hc_field(argv[3], "credit")
        except Refusal as error:
            print(error, file=sys.stderr)
            return 2
        if not account or not argv[4].strip():
            print("usage: sparkpipe_pay.py credit EMAIL HC NOTE, for an existing account", file=sys.stderr)
            return 2
        lot = app.market.credit(account, credits, argv[4].strip()[:200])
        print(f"credited {lot['hc']} HC to {argv[2]}, lot {lot['lot']}, lapsing {lot['expires']}")
        return 0
    if argv[1:] not in ([], ["serve"]):
        print("usage: sparkpipe_pay.py [serve|verify|cover [PLANS.json]|plan PLANS.json|end-plan PLAN YYYY-MM-DD REASON|maintenance FLEET YYYY-MM-DD [HOURS]|credit EMAIL HC NOTE|release ORDER|tranche EMAIL YYYY-MM HOURS]", file=sys.stderr)
        return 2
    server = start(cfg)
    app = server.app
    threading.Thread(target=app.reconcile_forever, daemon=True).start()
    app.reconcile()
    log(f"sparkpipe-pay on 127.0.0.1:{server.server_address[1]}, {app.book.count} ledger lines, webhook {cfg.webhook_url}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
