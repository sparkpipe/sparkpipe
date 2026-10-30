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
LIMITS = {"code": (20, 600), "code-email": (5, 3600), "verify": (30, 600), "order": (10, 600), "lead": (5, 3600), "read": (240, 600)}


def clock():
    return datetime.now(timezone.utc)


def log(text):
    print(text, file=sys.stderr, flush=True)


class Refusal(Exception):
    def __init__(self, text, status=400, reason=None):
        super().__init__(text)
        self.status = status
        self.reason = reason


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


class Config:
    def __init__(self, env):
        self.port = int(env.get("SPARKPIPE_PAY_PORT", "18090"))
        self.data = env["SPARKPIPE_PAY_DATA"]
        self.site = env["SPARKPIPE_PAY_SITE"]
        self.public = env.get("SPARKPIPE_PAY_PUBLIC", "https://sparkpipe.ai").rstrip("/")
        parts = urllib.parse.urlsplit(self.public)
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.api = env.get("COINPAYMENTS_API", "https://a-api.coinpayments.net/api/v2").rstrip("/")
        self.client_id = env["COINPAYMENTS_CLIENT_ID"]
        with open(env["COINPAYMENTS_CLIENT_SECRET_FILE"], encoding="utf-8") as handle:
            self.secret = handle.read().strip().encode()
        self.webhook_url = self.public + "/pay/api/coinpayments"
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
            self.orders[data["order"]].update(state="completed", paid=line["at"])
        elif kind == "hours_issued":
            self.orders[data["order"]].update(issued=True, issued_hours=data["hours"], paid_net=data.get("net", self.orders[data["order"]]["net"]))

    def waiting(self, order, now):
        return order["state"] not in CLOSED and order["state"] != "completed" and not order["issued"] and (now - stamp_time(order["created"])).total_seconds() < PENDING_SECONDS

    def held(self, account, now):
        total = 0
        for order in self.orders.values():
            if order["account"] != account or order["product"] != "hourly":
                continue
            if order["issued"] or order["state"] == "mismatch":
                if date.fromisoformat(order["last"]) >= local_day(now):
                    total += order["issued_hours"] if order["issued"] else order["hours"]
            elif self.waiting(order, now):
                total += order["hours"]
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
    if quote["product"] == "hourly":
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
    if isinstance(remote.get("invoice"), dict):
        remote = remote["invoice"]
    state = str(remote.get("status") or remote.get("state") or "").lower()
    amount = remote.get("amount")
    total = amount.get("total") if isinstance(amount, dict) else amount
    currency = remote.get("currency")
    currency = currency.get("id") if isinstance(currency, dict) else currency
    return state, total, None if currency is None else str(currency)


def received_of(remote):
    if isinstance(remote.get("invoice"), dict):
        remote = remote["invoice"]
    payments = remote.get("payments")
    if not isinstance(payments, list):
        return None
    try:
        return money(sum((Decimal(str(p.get("nativeActualAmount") or "0")) for p in payments if isinstance(p, dict)), Decimal(0)))
    except InvalidOperation:
        return None


def covered(order, paid):
    share = min(Decimal(1), paid / Decimal(order["total"]))
    if order["product"] == "hourly":
        hours = int((Decimal(order["hours"]) * share).to_integral_value(rounding=ROUND_DOWN))
        return hours, text_money(Decimal(order["net"]) * hours / order["hours"])
    net = (Decimal(order["net"]) * share).quantize(CENT, rounding=ROUND_DOWN)
    return format((net / Decimal(order["price"])).quantize(HOUR_STEP, rounding=ROUND_DOWN), "f"), text_money(net)


def paid_with(remote):
    payments = remote.get("payments") if isinstance(remote.get("payments"), list) else []
    return sorted({str(p.get("paymentCurrencySymbol")) for p in payments if isinstance(p, dict) and p.get("paymentCurrencySymbol")})


def receipt(cfg, order, issued, received):
    if order["product"] == "hourly":
        what = f"{issued['hours']} fleet-hours, usable from {order['start']} to {order['last']}"
    else:
        what = f"a prepayment of {issued['net']} USD under agreement {order['agreement']}: {issued['hours']} fleet-hours"
    if issued.get("short"):
        lines = [
            f"Order {order['order']} was for {order['total']} USD; {received} USD arrived. We issued what it covers: {what}.",
            "",
            f"Received in {order['coin'].split('.')[0]}: {received} USD, VAT included",
        ]
    else:
        lines = [
            f"We received your payment for order {order['order']}: {what}.",
            "",
            f"Subtotal: {order['net']} USD",
            f"VAT: {order['vat']} USD",
            f"Total paid in {order['coin'].split('.')[0]}: {order['total']} USD",
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
        self.mails.put((to, subject, text))

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
            blocks = [{"order": o["order"], "hours": o["issued_hours"], "start": o["start"], "last": o["last"]} for o in mine if o["product"] == "hourly" and o["issued"] and date.fromisoformat(o["last"]) >= local_day(now)]
            limit = self.limit(pricing, account)
            held = self.book.held(account, now)
            return {
                "email": record["email"],
                "company": record["company"],
                "since": record["created"][:10],
                "termsCurrent": self.accepted(account, self.current_terms()),
                "limit": limit,
                "held": held,
                "room": max(0, limit - held),
                "hours": sum(b["hours"] for b in blocks),
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
            raise Refusal(f"This account may buy at most {max(0, room)} more hours now: the holding limit is {limit} hours, counting unused hours and unpaid orders (clause 4.1).", 409)
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

    def order(self, body, account, ip, agent):
        self.throttle("order", ip)
        now = clock()
        coin = coin_field(body.get("coin"))
        pricing = self.cfg.pricing()
        with self.book.lock:
            record = self.book.accounts[account]
            if body.get("product") == "hourly":
                quote = self.hourly_terms(pricing, body, account, now)
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
        shown["issuedHours"] = order.get("issued_hours")
        shown["checkout"] = order["link"] if order["state"] in ("unpaid", "pending") else None
        return shown

    def status(self, order_id):
        with self.book.lock:
            order = self.book.orders.get(order_id)
            if not order:
                raise Refusal("No such order.", 404)
            return self.shown(order)

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
        key = invoice.get("id") or event.get("id")
        if key:
            self.jobs.put(str(key))
        return True

    def settle(self, order_id, remote, released=False):
        state, total, currency = remote_fields(remote)
        received = received_of(remote)
        with self.book.lock:
            order = self.book.orders[order_id]
            if order["issued"] or not state or (order["state"] == "mismatch" and not released):
                return None
            due = money(order["total"])
            arrived = None if received is None else text_money(received)
            if state != "completed":
                moved = arrived is not None and arrived != order.get("received", "0.00")
                if (state != order["state"] and order["state"] != "completed") or moved:
                    change = {"order": order_id, "invoice": order["invoice"], "state": state}
                    if arrived is not None:
                        change["received"] = arrived
                    self.book.append("invoice_state", change)
                if moved and Decimal(0) < received < due:
                    self.mail(self.cfg.notify, f"Part payment: {order_id}", f"Order {order_id} for {order['total']} USD has received {arrived} USD at CoinPayments ({order['company']} <{order['email']}>). It completes only if the rest arrives, you accept the part payment in CoinPayments, or you refund it.")
                return None
            full = total is not None and currency == USD and money(total) == due and (received is None or received + CENT >= due)
            if not full and not released:
                self.book.append("invoice_state", {"order": order_id, "invoice": order["invoice"], "state": "mismatch", "total": str(total), "currency": currency, "received": arrived})
                log(f"order {order_id}: completed invoice does not match the order ({total} {currency}, {arrived} received); hours not issued")
                self.mail(self.cfg.notify, f"Payment to review: {order_id}", f"Order {order_id} for {order['total']} USD completed at CoinPayments as {total} {currency} with {arrived} USD received. No hours were issued. If it is right, `sparkpipe_pay.py release {order_id}` issues the hours the amount received covers.")
                return None
            hours, net = (order["hours"], order["net"]) if full else covered(order, received if received is not None else money(total))
            if not Decimal(hours) > 0:
                log(f"order {order_id}: {arrived} USD received covers no hours; refund it in CoinPayments")
                return None
            if order["state"] != "completed":
                self.book.append("payment", {"order": order_id, "invoice": order["invoice"], "total": order["total"], "received": arrived, "currency": currency, "paidWith": paid_with(remote)})
            issued = {"order": order_id, "account": order["account"], "product": order["product"], "hours": hours, "net": net, "released": released, "short": not full}
            if order["product"] == "hourly":
                issued.update(start=order["start"], last=order["last"])
            else:
                issued.update(agreement=order["agreement"])
            self.book.append("hours_issued", issued)
            log(f"order {order_id}: paid, {hours} hours issued to {order['account']}")
            self.mail(order["email"], f"Payment received: SparkPipe order {order_id}", receipt(self.cfg, order, issued, arrived))
            self.mail(self.cfg.notify, f"Paid: {order['company']}, {arrived or order['total']} USD", f"Order {order_id}: {hours} of {order['hours']} hours for {order['company']} <{order['email']}>, {arrived or order['total']} USD received of {order['total']} USD in {order['coin']}.")
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
            secure = "; Secure" if self.server.app.cfg.public.startswith("https://") else ""
            age = SESSION_SECONDS if cookie else 0
            self.send_header("Set-Cookie", f"{COOKIE}={cookie}; Max-Age={age}; Path=/pay/api; HttpOnly; SameSite=Lax{secure}")
        self.end_headers()
        self.wfile.write(body)

    def refuse(self, error):
        data = {"error": str(error)}
        if error.reason:
            data["reason"] = error.reason
        self.reply(error.status, data)

    def client_ip(self):
        forwarded = self.headers.get("X-Forwarded-For")
        return forwarded.split(",")[-1].strip() if forwarded else self.client_address[0]

    def agent(self):
        return (self.headers.get("User-Agent") or "")[:300]

    def do_GET(self):
        app = self.server.app
        if self.path == "/health":
            return self.reply(200, {"ok": True})
        try:
            app.throttle("read", self.client_ip())
            if self.path == "/account":
                return self.reply(200, app.summary(app.need(self.headers)))
            match = ORDER_PATH.match(self.path)
            if not match:
                return self.reply(404, {"error": "Not found."})
            self.reply(200, app.status(match.group(1)))
        except Refusal as error:
            self.refuse(error)

    def do_POST(self):
        app = self.server.app
        limit = 262144 if self.path == "/coinpayments" else 16384
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = -1
        if length <= 0 or length > limit:
            return self.reply(413 if length > limit else 400, {"error": "The request body is missing or too large."})
        raw = self.rfile.read(length)
        if self.path == "/coinpayments":
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
        try:
            if self.path == "/code":
                return self.reply(200, app.send_code(body, self.client_ip()))
            if self.path == "/verify":
                summary, token = app.verify(body, self.client_ip(), self.agent())
                return self.reply(200, summary, cookie=token)
            if self.path == "/signout":
                return self.reply(200, {"ok": True}, cookie="")
            if self.path == "/order":
                return self.reply(200, app.order(body, app.need(self.headers), self.client_ip(), self.agent()))
            if self.path == "/lead":
                return self.reply(200, app.lead(body, self.client_ip(), self.agent()))
            return self.reply(404, {"error": "Not found."})
        except Refusal as error:
            self.refuse(error)


def start(cfg, port=None):
    os.makedirs(cfg.data, exist_ok=True)
    app = App(cfg, Book(cfg.path("ledger.jsonl")))
    threading.Thread(target=app.work, daemon=True).start()
    threading.Thread(target=app.deliver, daemon=True).start()
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
    if argv[1:] not in ([], ["serve"]):
        print("usage: sparkpipe_pay.py [serve|verify|release ORDER]", file=sys.stderr)
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
