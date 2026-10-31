#!/usr/bin/env python3
import base64
import calendar
import collections
import fcntl
import hashlib
import hmac
import http.server
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

import sparkpipe_market as market
from sparkpipe_market import Refusal

KST = timezone(timedelta(hours=9))
USD = "5057"
CENT = Decimal("0.01")
HOUR_STEP = Decimal("0.0001")
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
ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CODE = re.compile(r"^\d{6}$")
ORDER_PATH = re.compile(r"^/order/(SP[A-Z2-7]{16})$")
COOKIE = "sp_session"
SESSION_SECONDS = 30 * 86400
CODE_SECONDS = 600
CODE_TRIES = 5
PENDING_SECONDS = 86400
RECONCILE_DAYS = 14
SKEW_SECONDS = 300
LIMITS = {"code": (20, 600), "code-email": (5, 3600), "verify": (30, 600), "order": (10, 600), "lead": (5, 3600), "read": (240, 600), "market": (240, 600), "pool": (600, 600)}


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


def served_days(first, last):
    return sum(1 for n in range((last - first).days + 1) if (first + timedelta(days=n)).weekday() != 6)


def tranches(signed, months):
    end = term_end(signed, months)
    plan, cursor = [], signed
    while cursor <= end:
        last = min(date(cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1]), end)
        plan.append(24 * served_days(cursor, last))
        cursor = last + timedelta(days=1)
    return plan


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
        self.reconcile_seconds = int(env.get("SPARKPIPE_PAY_RECONCILE_SECONDS", "300"))
        self.smtp = env.get("SPARKPIPE_PAY_SMTP", "127.0.0.1:25")
        self.mail_from = env.get("SPARKPIPE_PAY_MAIL_FROM", "SparkPipe <noreply@sparkpipe.ai>")
        self.notify = env.get("SPARKPIPE_PAY_NOTIFY", "hyor@sparkpipe.ai")

    def path(self, name):
        return os.path.join(self.data, name)

    def pricing(self):
        with open(os.path.join(self.site, "pricing.js"), encoding="utf-8") as handle:
            text = handle.read()
        return json.loads(text[text.index("{"):text.rindex("}") + 1])

    def terms(self, name):
        with open(os.path.join(self.site, "terms", name), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    def listing(self, name):
        path = self.path(name)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def session_key(self):
        path = self.path("session_key")
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
            self.accounts[data["account"]]["accepted"] = {"ko": data["ko"], "en": data["en"], "at": line["at"]}
        elif kind == "order":
            self.orders[data["order"]] = dict(data, created=line["at"], state="created", invoice=None, link=None, issued=False, paid=None, child=None)
            if "parent" in data:
                self.orders[data["parent"]]["child"] = data["order"]
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
        elif kind in market.KINDS:
            self.market.apply(line)

    def waiting(self, order, now):
        return order["state"] not in CLOSED and order["state"] != "completed" and not order["issued"] and (now - stamp_time(order["created"])).total_seconds() < PENDING_SECONDS

    def held(self, account, now):
        total = 0
        for order in self.orders.values():
            if order["account"] != account or order["product"] != "hourly":
                continue
            if order["issued"] or order["state"] == "mismatch":
                if date.fromisoformat(order["last"]) >= local_day(now):
                    total += Decimal(str(order["issued_hours"] if order["issued"] else order["hours"]))
            elif self.waiting(order, now):
                total += Decimal(str(order["hours"]))
        return total

    def prepaid(self, agreement):
        return sorted((stamp_time(order["paid"]), Decimal(order["paid_net"])) for order in self.orders.values() if order.get("agreement") == agreement and order["issued"])

    def pending(self, agreement, now):
        return sum((Decimal(order["net"]) for order in self.orders.values() if order.get("agreement") == agreement and self.waiting(order, now)), Decimal(0))


def standing(pricing, book, agreement, now):
    signed = date.fromisoformat(agreement["signed"])
    price = money(agreement["hourPrice"])
    plan = tranches(signed, pricing["termMonths"])
    end = term_end(signed, pricing["termMonths"])
    today = local_day(now)
    paid = book.prepaid(agreement["id"])
    paid_net = sum((net for _, net in paid), Decimal(0))
    pending = book.pending(agreement["id"], now)
    total = price * sum(plan)
    ended = None
    if today > end:
        ended = f"The term ended on {end.isoformat()}."
    elif not paid and today > signed + timedelta(days=pricing["phaseADays"] - 1):
        ended = "Phase A ended with no payment, so the agreement ended with nothing owed (clause 5.2)."
    next_gate = None
    for gate in pricing["gates"]:
        due = signed + timedelta(days=gate["dueDay"] - 1)
        need = price * sum(plan if gate["tranches"] is None else plan[:gate["tranches"]])
        cutoff = datetime(due.year, due.month, due.day, 23, 59, 59, tzinfo=KST)
        have = sum((net for moment, net in paid if moment <= cutoff), Decimal(0))
        if today > due:
            if paid and have < need and not ended:
                ended = f"{gate['name']} was missed on {due.isoformat()}, so no further prepayment can be made (clause 6.4)."
        elif next_gate is None and paid_net + pending < need:
            next_gate = {"name": gate["name"], "due": due.isoformat(), "amount": text_money(need - paid_net - pending)}
    return {
        "agreement": agreement["id"],
        "signed": signed.isoformat(),
        "end": end.isoformat(),
        "price": text_money(price),
        "hours": sum(plan),
        "total": text_money(total),
        "paid": text_money(paid_net),
        "pending": text_money(pending),
        "remaining": text_money(max(Decimal(0), total - paid_net - pending)),
        "nextGate": None if ended else next_gate,
        "ended": ended,
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


def whole_field(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value).isdigit():
        raise Refusal("Enter a whole number of hours.")
    return int(value)


def day_field(value):
    try:
        if isinstance(value, str) and ISO_DAY.match(value):
            return date.fromisoformat(value)
    except ValueError:
        pass
    raise Refusal("Enter a start date.")


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
    return (f"On behalf of {company}, I accept the SparkPipe Hourly Terms. "
            f"Korean text, which prevails: SHA-256 {terms['ko']} ({cfg.public}/terms/hourly-ko.md). "
            f"English text: SHA-256 {terms['en']} ({cfg.public}/terms/hourly-en.md).")


def invoice_body(cfg, quote):
    if quote.get("parent"):
        name = f"Remaining {quote['hours']} SparkPipe fleet-hours of order {quote['parent']}"
        detail = f"Hourly Terms 3.5; usable {quote['start']} to {quote['last']}"
        quantity = {"value": 1, "type": "quantity"}
    elif quote["product"] == "credits":
        name = f"{quote['hours']} SparkPipe hour-credits"
        detail = f"Hour Market; {quote['price']} USD each, usable for 12 months"
        quantity = {"value": 1, "type": "quantity"}
    elif quote["product"] == "hourly":
        name = f"{quote['hours']} SparkPipe fleet-hours"
        detail = f"Hourly Terms; usable {quote['start']} to {quote['last']}"
        quantity = {"value": quote["hours"], "type": "hours"}
    else:
        name = f"Prepayment, agreement {quote['agreement']}"
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
    if order["product"] == "hourly":
        return format((Decimal(order["hours"]) * share).quantize(HOUR_STEP, rounding=ROUND_DOWN), "f"), text_money(net)
    if order["product"] == "credits":
        return market.text_hc(net / Decimal(order["price"])), text_money(net)
    return format((net / Decimal(order["price"])).quantize(HOUR_STEP, rounding=ROUND_DOWN), "f"), text_money(net)


def paid_with(remote):
    return sorted({str(p["paymentCurrencySymbol"]) for p in remote["payments"]})


def left_of(order):
    if order["product"] != "hourly" or not order["issued"]:
        return None
    hours = Decimal(str(order["hours"])) - Decimal(str(order["issued_hours"]))
    if hours <= 0:
        return None
    total = money(order["total"]) - money(order["received"])
    net = money(order["net"]) - money(order["paid_net"])
    if total <= 0:
        return None
    return {"hours": hours, "total": total, "net": net, "vat": total - net}


def receipt(cfg, order, issued, received, paid):
    coin = ", ".join(sorted({symbol.split(".")[0] for symbol in paid}))
    if order["product"] == "hourly":
        what = f"{issued['hours']} fleet-hours, usable from {order['start']} to {order['last']}"
    elif order["product"] == "credits":
        what = f"{issued['hours']} hour-credits, usable for 12 months"
    else:
        what = f"a prepayment of {issued['net']} USD under agreement {order['agreement']}: {issued['hours']} fleet-hours"
    if issued.get("short"):
        lines = [
            f"Order {order['order']} was for {order['total']} USD; {received} USD arrived. We issued what it covers: {what}.",
            "",
            f"Received in {coin}: {received} USD, VAT included",
            "",
            f"To get the other hours of this order at the same price, pay the remaining amount from your account: {cfg.public}/account.html?order={order['order']}",
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
        f"The hours are in your account: {cfg.public}/account.html",
        "We will email you when the fleet calendar opens for reservations.",
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
        return {"ko": self.cfg.terms("hourly-ko.md"), "en": self.cfg.terms("hourly-en.md")}

    def accepted(self, account, terms):
        found = self.book.accounts[account]["accepted"]
        return bool(found) and found["ko"] == terms["ko"] and found["en"] == terms["en"]

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
                    raise Refusal("No account uses this email yet. Buy hours to open one.", 404, "new")
                company = text_field(body.get("company"), "company's legal name", 200)
                if body.get("accept") is not True:
                    raise Refusal("Accept the Hourly Terms to open an account.", 409, "terms")
                account = new_id("AC")
                self.book.append("account", {"account": account, "email": email, "company": company, "ip": ip, "agent": agent})
                self.accept(account, ip, agent)
                self.mail(self.cfg.notify, f"New SparkPipe account: {company}", f"{company} <{email}> opened account {account}.")
                log(f"account {account} opened")
            elif body.get("accept") is True and not self.accepted(account, self.current_terms()):
                self.accept(account, ip, agent)
        expires = int(clock().timestamp()) + SESSION_SECONDS
        return self.summary(account), session_token(self.key, account, expires)

    def agreements(self, account):
        email = self.book.accounts[account]["email"]
        return [a for a in self.cfg.listing("agreements.json") if str(a.get("email", "")).lower() == email]

    def limit(self, pricing, account):
        approved = {str(e).lower() for e in self.cfg.listing("approved.json")}
        return pricing["verifiedMaxHours"] if self.book.accounts[account]["email"] in approved else pricing["hourlyMaxHours"]

    def summary(self, account):
        pricing = self.cfg.pricing()
        now = clock()
        with self.book.lock:
            record = self.book.accounts[account]
            mine = [o for o in reversed(list(self.book.orders.values())) if o["account"] == account]
            blocks = [{"order": o["order"], "hours": hours_number(o["issued_hours"]), "start": o["start"], "last": o["last"]} for o in mine if o["product"] == "hourly" and o["issued"] and date.fromisoformat(o["last"]) >= local_day(now)]
            limit = self.limit(pricing, account)
            held = self.book.held(account, now)
            return {
                "email": record["email"],
                "company": record["company"],
                "since": record["created"][:10],
                "termsCurrent": self.accepted(account, self.current_terms()),
                "limit": limit,
                "held": hours_number(held),
                "room": hours_number(max(0, limit - held)),
                "hours": hours_number(sum((Decimal(str(b["hours"])) for b in blocks), Decimal(0))),
                "blocks": blocks,
                "orders": [self.shown(o) for o in mine[:50]],
                "agreements": [standing(pricing, self.book, a, now) for a in self.agreements(account)],
            }

    def hourly_terms(self, pricing, body, account, now):
        hours = whole_field(body.get("hours"))
        start = day_field(body.get("start"))
        earliest = max(local_day(now), date.fromisoformat(pricing["hourlyEarliestStart"]))
        if start < earliest:
            raise Refusal(f"Hours can start no earlier than {earliest.isoformat()}.")
        if start > earliest + timedelta(days=365):
            raise Refusal("Choose a start date within the next year.")
        limit = self.limit(pricing, account)
        room = limit - self.book.held(account, now)
        if hours < 1 or hours > room:
            raise Refusal(f"This account may buy at most {hours_number(max(0, int(room)))} more hours now: the holding limit is {limit} hours, counting unused hours and unpaid orders (clause 4.1).", 409)
        net = money(Decimal(pricing["hourlyPrice"]) * hours)
        vat = money(net * Decimal(str(pricing["vatRate"])))
        return {
            "product": "hourly", "hours": hours,
            "start": start.isoformat(), "last": term_end(start, pricing["hourlyValidityMonths"]).isoformat(),
            "net": text_money(net), "vat": text_money(vat), "total": text_money(net + vat),
        }

    def prepayment_terms(self, pricing, body, account, now):
        number = body.get("agreement")
        found = next((a for a in self.agreements(account) if isinstance(number, str) and a.get("id") == number), None)
        if not found:
            raise Refusal("No agreement with this number belongs to your account.", 404)
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
            "product": "two-year", "agreement": found["id"],
            "price": text_money(price), "hours": format((net / price).quantize(HOUR_STEP, rounding=ROUND_DOWN), "f"),
            "net": text_money(net), "vat": text_money(vat), "total": text_money(net + vat),
        }

    def rest_terms(self, pricing, body, account, now):
        parent = self.book.orders.get(body.get("parent")) if isinstance(body.get("parent"), str) else None
        if not parent or parent["account"] != account:
            raise Refusal("No such order belongs to your account.", 404)
        child = self.book.orders.get(parent["child"]) if parent["child"] else None
        if child and (child["issued"] or self.book.waiting(child, now)):
            raise Refusal("A payment for the rest of this order is already open or paid; see your orders.", 409)
        left = left_of(parent)
        if not left:
            raise Refusal("Nothing is left to pay on this order.", 409)
        if date.fromisoformat(parent["last"]) < local_day(now):
            raise Refusal("The hours of this order have expired.", 409)
        limit = self.limit(pricing, account)
        if left["hours"] > limit - self.book.held(account, now):
            raise Refusal(f"The rest of this order would take the account over its holding limit of {limit} hours (clause 4.1). Use some hours first, or write to us.", 409)
        return {
            "product": "hourly", "parent": parent["order"], "hours": format(left["hours"], "f"),
            "start": parent["start"], "last": parent["last"],
            "net": text_money(left["net"]), "vat": text_money(left["vat"]), "total": text_money(left["total"]),
        }

    def credit_terms(self, pricing, body):
        offer = pricing["market"]
        net = money_field(body.get("amount"))
        if net < Decimal(str(offer["minPurchase"])) or net > 100000:
            raise Refusal(f"Buy at least {offer['minPurchase']} USD of HC, and at most 100,000 USD at a time.")
        price = money(Decimal(str(offer["hcPrice"])))
        vat = money(net * Decimal(str(pricing["vatRate"])))
        return {"product": "credits", "hours": market.text_hc(net / price), "price": text_money(price), "net": text_money(net), "vat": text_money(vat), "total": text_money(net + vat)}

    def buy(self, body, account, ip, agent):
        if self.cfg.sandbox:
            self.throttle("market", ip)
            pricing = self.cfg.pricing()
            quote = self.credit_terms(pricing, body)
            return self.market.buy(account, Decimal(quote["net"]), pricing["market"])
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
            if body.get("product") in ("hourly", "rest", "credits"):
                if body["product"] == "credits":
                    quote = self.credit_terms(pricing, body)
                elif body["product"] == "hourly":
                    quote = self.hourly_terms(pricing, body, account, now)
                else:
                    quote = self.rest_terms(pricing, body, account, now)
                if not self.accepted(account, self.current_terms()):
                    if body.get("accept") is not True:
                        raise Refusal("The Hourly Terms have changed since you accepted them. Accept the current text to continue.", 409, "terms")
                    self.accept(account, ip, agent)
            elif body.get("product") == "two-year":
                quote = self.prepayment_terms(pricing, body, account, now)
            else:
                raise Refusal("Choose hourly hours or a two-year prepayment.")
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
        shown = {key: order.get(key) for key in ("order", "product", "hours", "net", "vat", "total", "start", "last", "agreement", "coin", "state", "issued", "created", "paid", "received")}
        shown["issuedHours"] = None if order.get("issued_hours") is None else hours_number(order["issued_hours"])
        shown["parent"] = order.get("parent")
        child = self.book.orders.get(order["child"]) if order.get("child") else None
        left = left_of(order) if not child or not (child["issued"] or self.book.waiting(child, clock())) else None
        shown["left"] = text_money(left["total"]) if left else None
        shown["leftHours"] = hours_number(left["hours"]) if left else None
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
        lines = []
        with self.book.lock:
            if os.path.exists(self.book.path):
                with open(self.book.path, encoding="utf-8") as handle:
                    lines = [json.loads(raw) for raw in handle if account in raw]
            return {"head": self.book.head, "count": self.book.count, "lines": lines[-300:]}

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
        self.mail(self.cfg.notify, f"Two-year quote request: {lead['company']}", "\n".join(f"{key}: {value}" for key, value in lead.items()))
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
            if order["product"] == "hourly":
                issued.update(start=order["start"], last=order["last"])
            elif order["product"] == "two-year":
                issued.update(agreement=order["agreement"])
            self.book.append("hours_issued", issued)
            if order["product"] == "credits":
                self.market.issue(order["account"], Decimal(hours), order_id)
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
            ticket = app.market.pool_open(self.bearer(), body.get("model"))
        except Refusal as error:
            return self.pool_error(error)
        if app.cfg.sandbox:
            return self.simulated(body, messages, ticket)
        return self.forwarded(body, ticket)

    def simulated(self, body, messages, ticket):
        app = self.server.app
        limit = body.get("max_tokens") if isinstance(body.get("max_tokens"), int) and not isinstance(body.get("max_tokens"), bool) else 128
        text, tokens_in, tokens_out = market.simulate(ticket["model"], messages, min(max(limit, 1), 512))
        made, ident = int(time.time()), "chatcmpl-" + secrets.token_hex(12)
        usage = {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": tokens_in + tokens_out}
        if body.get("stream") is True:
            self.stream_head()
            try:
                for word in text.split(" "):
                    self.event({"id": ident, "object": "chat.completion.chunk", "created": made, "model": ticket["model"], "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]})
                    time.sleep(0.025)
            finally:
                charge = app.market.pool_close(ticket, tokens_in, tokens_out)
            self.event({"id": ident, "object": "chat.completion.chunk", "created": made, "model": ticket["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage, "sparkpipe": charge})
            return self.event(b"[DONE]")
        time.sleep(min(2.0, tokens_out / 40))
        charge = app.market.pool_close(ticket, tokens_in, tokens_out)
        self.reply(200, {"id": ident, "object": "chat.completion", "created": made, "model": ticket["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage, "sparkpipe": charge})

    def forwarded(self, body, ticket):
        app = self.server.app
        payload = json.dumps(dict(body, model=ticket["model"]), ensure_ascii=False).encode()
        request = urllib.request.Request(app.cfg.pool_upstream + "/chat/completions", data=payload, method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {app.cfg.pool_key}", "X-SparkPipe-Public": "1"})
        tokens_in = tokens_out = 0
        closed = started = False
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                if body.get("stream") is not True:
                    data = json.loads(response.read())
                    usage = data.get("usage") or {}
                    tokens_in, tokens_out = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
                    data["sparkpipe"] = app.market.pool_close(ticket, tokens_in, tokens_out)
                    closed = True
                    return self.reply(200, data)
                self.stream_head()
                started = True
                for line in response:
                    text = line.strip()
                    if not text.startswith(b"data: "):
                        continue
                    if text == b"data: [DONE]":
                        closed = True
                        self.event({"object": "chat.completion.chunk", "choices": [], "sparkpipe": app.market.pool_close(ticket, tokens_in, tokens_out)})
                        return self.event(b"[DONE]")
                    try:
                        chunk = json.loads(text[6:])
                        usage = chunk.get("usage") or {}
                        tokens_in = int(usage.get("prompt_tokens") or tokens_in)
                        tokens_out = int(usage.get("completion_tokens") or tokens_out + sum(1 for c in chunk.get("choices") or [] if (c.get("delta") or {}).get("content")))
                    except (ValueError, AttributeError):
                        pass
                    self.event(text[6:])
        except urllib.error.HTTPError as error:
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
            if not closed:
                closed = True
                app.market.pool_close(ticket, tokens_in, tokens_out)
            if not started:
                self.pool_error(Refusal("The fleets did not answer. Try again in a minute.", 502, "upstream_error"))
        finally:
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
    if argv[1:] not in ([], ["serve"]):
        print("usage: sparkpipe_pay.py [serve|verify|release ORDER|tranche EMAIL YYYY-MM HOURS]", file=sys.stderr)
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
