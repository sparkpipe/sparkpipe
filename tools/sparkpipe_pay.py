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
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eth_signature import personal_recover

KST = timezone(timedelta(hours=9))
USD = "5057"
CENT = Decimal("0.01")
HOUR_STEP = Decimal("0.0001")
EVENTS = ["invoicePending", "invoicePaid", "invoiceCompleted", "invoiceCancelled", "invoiceTimedOut"]
OPEN = {"unpaid", "pending", "paid"}
CLOSED = {"cancelled", "timedout", "deleted", "failed"}
ACCOUNT = re.compile(r"^0x[0-9a-fA-F]{40}$")
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,189}\.[^@\s.]{2,}$")
ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ORDER_PATH = re.compile(r"^/order/(SP[A-Z2-7]{16})$")
QUOTE_SECONDS = 900
PENDING_SECONDS = 86400
RECONCILE_DAYS = 14
SKEW_SECONDS = 300
LIMITS = {"quote": (30, 600), "order": (10, 600), "agreement": (30, 600)}


def clock():
    return datetime.now(timezone.utc)


def log(text):
    print(text, file=sys.stderr, flush=True)


class Refusal(Exception):
    def __init__(self, text, status=400):
        super().__init__(text)
        self.status = status


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


class Config:
    def __init__(self, env):
        self.port = int(env.get("SPARKPIPE_PAY_PORT", "18090"))
        self.data = env["SPARKPIPE_PAY_DATA"]
        self.site = env["SPARKPIPE_PAY_SITE"]
        self.public = env.get("SPARKPIPE_PAY_PUBLIC", "https://sparkpipe.ai").rstrip("/")
        self.api = env.get("COINPAYMENTS_API", "https://a-api.coinpayments.net/api/v2").rstrip("/")
        self.client_id = env["COINPAYMENTS_CLIENT_ID"]
        with open(env["COINPAYMENTS_CLIENT_SECRET_FILE"], encoding="utf-8") as handle:
            self.secret = handle.read().strip().encode()
        self.webhook_url = self.public + "/pay/api/coinpayments"
        self.reconcile_seconds = int(env.get("SPARKPIPE_PAY_RECONCILE_SECONDS", "300"))

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


class Book:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.head = "0" * 64
        self.count = 0
        self.offset = 0
        self.orders = {}
        self.invoices = {}
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
        if kind == "order":
            self.orders[data["order"]] = dict(data, created=line["at"], state="created", invoice=None, link=None, issued=False, paid=None)
        elif kind == "invoice":
            self.orders[data["order"]].update(invoice=data["invoice"], link=data["link"], state="unpaid")
            self.invoices[data["invoice"]] = data["order"]
        elif kind == "invoice_failed":
            self.orders[data["order"]]["state"] = "failed"
        elif kind == "invoice_state":
            self.orders[data["order"]]["state"] = data["state"]
        elif kind == "payment":
            self.orders[data["order"]].update(state="completed", paid=line["at"])
        elif kind == "hours_issued":
            self.orders[data["order"]]["issued"] = True

    def waiting(self, order, now):
        return order["state"] not in CLOSED and order["state"] != "completed" and not order["issued"] and (now - stamp_time(order["created"])).total_seconds() < PENDING_SECONDS

    def held(self, account, now):
        total = 0
        for order in self.orders.values():
            if order["account"] != account or order["product"] != "hourly":
                continue
            if order["issued"] or order["state"] == "mismatch":
                if date.fromisoformat(order["last"]) >= local_day(now):
                    total += order["hours"]
            elif self.waiting(order, now):
                total += order["hours"]
        return total

    def prepaid(self, agreement):
        return sorted((stamp_time(order["paid"]), Decimal(order["net"])) for order in self.orders.values() if order.get("agreement") == agreement and order["issued"])

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


def account_field(value):
    if not isinstance(value, str) or not ACCOUNT.match(value):
        raise Refusal("Connect a wallet first: the account is its Ethereum address.")
    return value.lower()


def text_field(value, label, size):
    if not isinstance(value, str) or not value.strip() or len(value) > size or any(ord(c) < 32 for c in value):
        raise Refusal(f"Enter the {label}.")
    return value.strip()


def email_field(value):
    if not isinstance(value, str) or len(value) > 254 or not EMAIL.match(value.strip()):
        raise Refusal("Enter a valid email address.")
    return value.strip()


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


def message(cfg, quote):
    if quote["product"] == "hourly":
        return "\n".join([
            f"SparkPipe order {quote['order']}",
            f"I buy {quote['hours']} SparkPipe fleet-hours under the SparkPipe Hourly Terms, usable from {quote['start']} to {quote['last']}.",
            f"Price: {quote['net']} USD + VAT {quote['vat']} USD = {quote['total']} USD, paid in USDC.",
            f"Account: {quote['account']}",
            f"Company: {quote['company']}",
            f"Email: {quote['email']}",
            "I accept the SparkPipe Hourly Terms, whose texts have these SHA-256 hashes:",
            f"Korean, which prevails: {quote['terms']['ko']} ({cfg.public}/terms/hourly-ko.md)",
            f"English: {quote['terms']['en']} ({cfg.public}/terms/hourly-en.md)",
        ])
    return "\n".join([
        f"SparkPipe order {quote['order']}",
        f"I prepay {quote['net']} USD under SparkPipe 2-Year Reserved Fleet Service Agreement {quote['agreement']}, for {quote['hours']} hours at {quote['price']} USD per hour.",
        f"Price: {quote['net']} USD + VAT {quote['vat']} USD = {quote['total']} USD, paid in USDC.",
        f"Account: {quote['account']}",
        f"Company: {quote['company']}",
    ])


def invoice_body(cfg, quote):
    if quote["product"] == "hourly":
        name = f"{quote['hours']} SparkPipe fleet-hours"
        detail = f"Hourly Terms; usable {quote['start']} to {quote['last']}"
        quantity = {"value": quote["hours"], "type": "hours"}
    else:
        name = f"Prepayment, agreement {quote['agreement']}"
        detail = f"{quote['hours']} fleet-hours at {quote['price']} USD"
        quantity = {"value": 1, "type": "quantity"}
    back = f"{cfg.public}/buy.html?order={quote['order']}"
    return {
        "currency": USD,
        "items": [{"customId": quote["order"], "name": name, "description": detail, "quantity": quantity, "amount": quote["net"], "tax": quote["vat"]}],
        "amount": {"breakdown": {"subtotal": quote["net"], "taxTotal": quote["vat"]}, "total": quote["total"]},
        "invoiceId": quote["order"],
        "description": name,
        "buyer": {"companyName": quote["company"], "emailAddress": quote["email"]},
        "notesToRecipient": f"SparkPipe issues the hours to account {quote['account']} once this payment completes.",
        "customData": {"order": quote["order"]},
        "webhooks": [{"notificationsUrl": cfg.webhook_url, "notifications": EVENTS}],
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


class App:
    def __init__(self, cfg, book):
        self.cfg = cfg
        self.book = book
        self.lock = threading.Lock()
        self.quotes = {}
        self.hits = collections.defaultdict(collections.deque)
        self.jobs = queue.Queue()

    def throttle(self, kind, ip):
        count, window = LIMITS[kind]
        now = time.monotonic()
        with self.lock:
            hits = self.hits[(kind, ip)]
            while hits and hits[0] < now - window:
                hits.popleft()
            if len(hits) >= count:
                raise Refusal("Too many requests. Wait a few minutes and try again.", 429)
            hits.append(now)

    def agreement_for(self, body, account):
        number = body.get("agreement")
        found = next((a for a in self.cfg.listing("agreements.json") if isinstance(number, str) and a.get("id") == number), None)
        if not found or found["account"].lower() != account:
            raise Refusal("No agreement with this number belongs to the connected account.", 404)
        return found

    def agreement(self, body, ip):
        self.throttle("agreement", ip)
        account = account_field(body.get("account"))
        found = self.agreement_for(body, account)
        with self.book.lock:
            return standing(self.cfg.pricing(), self.book, found, clock())

    def hourly_terms(self, pricing, body, account, now):
        company = text_field(body.get("company"), "company name", 200)
        email = email_field(body.get("email"))
        hours = whole_field(body.get("hours"))
        start = day_field(body.get("start"))
        earliest = max(local_day(now), date.fromisoformat(pricing["hourlyEarliestStart"]))
        if start < earliest:
            raise Refusal(f"Hours can start no earlier than {earliest.isoformat()}.")
        if start > earliest + timedelta(days=365):
            raise Refusal("Choose a start date within the next year.")
        approved = {a.lower() for a in self.cfg.listing("approved.json")}
        limit = pricing["verifiedMaxHours"] if account in approved else pricing["hourlyMaxHours"]
        room = limit - self.book.held(account, now)
        if hours < 1 or hours > room:
            raise Refusal(f"This account may buy at most {max(0, room)} more hours now: the holding limit is {limit} hours, counting unused hours and unpaid orders (clause 4.1).", 409)
        net = money(Decimal(pricing["hourlyPrice"]) * hours)
        vat = money(net * Decimal(str(pricing["vatRate"])))
        return {
            "product": "hourly", "company": company, "email": email, "hours": hours,
            "start": start.isoformat(), "last": term_end(start, pricing["hourlyValidityMonths"]).isoformat(),
            "net": text_money(net), "vat": text_money(vat), "total": text_money(net + vat),
            "terms": {"ko": self.cfg.terms("hourly-ko.md"), "en": self.cfg.terms("hourly-en.md")},
        }

    def prepayment_terms(self, pricing, body, account, now):
        found = self.agreement_for(body, account)
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
            "product": "two-year", "agreement": found["id"], "company": found["company"], "email": found["email"],
            "price": text_money(price), "hours": format((net / price).quantize(HOUR_STEP, rounding=ROUND_DOWN), "f"),
            "net": text_money(net), "vat": text_money(vat), "total": text_money(net + vat),
        }

    def terms_for(self, quote, now):
        pricing = self.cfg.pricing()
        if quote["product"] == "hourly":
            return self.hourly_terms(pricing, quote, quote["account"], now)
        return self.prepayment_terms(pricing, {"agreement": quote["agreement"], "amount": quote["net"]}, quote["account"], now)

    def quote(self, body, ip):
        self.throttle("quote", ip)
        account = account_field(body.get("account"))
        now = clock()
        if body.get("product") not in ("hourly", "two-year"):
            raise Refusal("Choose the hourly or the two-year product.")
        pricing = self.cfg.pricing()
        with self.book.lock:
            terms = self.hourly_terms(pricing, body, account, now) if body["product"] == "hourly" else self.prepayment_terms(pricing, body, account, now)
        quote = dict(terms, order="SP" + base64.b32encode(secrets.token_bytes(10)).decode(), account=account)
        quote["message"] = message(self.cfg, quote)
        quote["expires"] = now.timestamp() + QUOTE_SECONDS
        with self.lock:
            for key in [k for k, q in self.quotes.items() if q["expires"] < now.timestamp()]:
                del self.quotes[key]
            while len(self.quotes) >= 10000:
                del self.quotes[next(iter(self.quotes))]
            self.quotes[quote["order"]] = quote
        return {key: quote[key] for key in quote if key not in ("expires", "terms")}

    def order(self, body, ip, agent):
        self.throttle("order", ip)
        now = clock()
        with self.lock:
            quote = self.quotes.get(body.get("order")) if isinstance(body.get("order"), str) else None
        if not quote or quote["expires"] < now.timestamp():
            raise Refusal("This quote has expired. Start the order again.", 410)
        if personal_recover(quote["message"], body.get("signature")) != quote["account"]:
            raise Refusal("The signature does not come from the connected account.", 403)
        with self.lock:
            if self.quotes.pop(quote["order"], None) is None:
                raise Refusal("This order was already placed.", 409)
        with self.book.lock:
            fresh = self.terms_for(quote, now)
            if any(fresh[key] != quote[key] for key in ("net", "vat", "total", "hours")):
                raise Refusal("The price or the room for this order changed. Start the order again.", 409)
            record = {key: quote[key] for key in quote if key != "expires"}
            record.update(signature=body["signature"], ip=ip, agent=agent)
            self.book.append("order", record)
        try:
            invoice = call(self.cfg, "POST", "/merchant/invoices", invoice_body(self.cfg, quote))["invoices"][0]
            link = invoice.get("checkoutLink") or invoice["link"]
        except Exception as error:
            self.book.append("invoice_failed", {"order": quote["order"], "error": str(error)[:500]})
            log(f"order {quote['order']}: invoice failed: {error}")
            raise Refusal("The payment processor did not create the invoice. Nothing was charged; try again later.", 502)
        self.book.append("invoice", {"order": quote["order"], "invoice": str(invoice["id"]), "link": link})
        log(f"order {quote['order']}: invoice {invoice['id']} for {quote['total']} USD")
        return {"order": quote["order"], "checkout": link}

    def status(self, order_id):
        with self.book.lock:
            order = self.book.orders.get(order_id)
            if not order:
                raise Refusal("No such order.", 404)
            shown = {key: order.get(key) for key in ("order", "product", "hours", "net", "vat", "total", "start", "last", "agreement", "state", "issued", "created", "paid")}
            shown["checkout"] = order["link"] if order["state"] == "unpaid" else None
            return shown

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
        with self.book.lock:
            order = self.book.orders[order_id]
            if order["issued"] or not state or (order["state"] == "mismatch" and not released):
                return
            if state != "completed":
                if state != order["state"] and order["state"] != "completed":
                    self.book.append("invoice_state", {"order": order_id, "invoice": order["invoice"], "state": state})
                return
            if not released and (total is None or currency != USD or money(total) != money(order["total"])):
                self.book.append("invoice_state", {"order": order_id, "invoice": order["invoice"], "state": "mismatch", "total": str(total), "currency": currency})
                log(f"order {order_id}: completed invoice does not match the order ({total} {currency}, fields {sorted(remote)}); hours not issued")
                return
            if order["state"] != "completed":
                self.book.append("payment", {"order": order_id, "invoice": order["invoice"], "total": order["total"], "currency": currency})
            issued = {"order": order_id, "account": order["account"], "product": order["product"], "hours": order["hours"], "released": released}
            if order["product"] == "hourly":
                issued.update(start=order["start"], last=order["last"])
            else:
                issued.update(agreement=order["agreement"], net=order["net"])
            self.book.append("hours_issued", issued)
            log(f"order {order_id}: paid, {order['hours']} hours issued to {order['account']}")

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

    def reply(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def client_ip(self):
        forwarded = self.headers.get("X-Forwarded-For")
        return forwarded.split(",")[-1].strip() if forwarded else self.client_address[0]

    def do_GET(self):
        app = self.server.app
        if self.path == "/health":
            return self.reply(200, {"ok": True})
        match = ORDER_PATH.match(self.path)
        if not match:
            return self.reply(404, {"error": "Not found."})
        try:
            self.reply(200, app.status(match.group(1)))
        except Refusal as error:
            self.reply(error.status, {"error": str(error)})

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
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self.reply(400, {"error": "The request is not a JSON object."})
        try:
            if self.path == "/quote":
                result = app.quote(body, self.client_ip())
            elif self.path == "/order":
                result = app.order(body, self.client_ip(), (self.headers.get("User-Agent") or "")[:300])
            elif self.path == "/agreement":
                result = app.agreement(body, self.client_ip())
            else:
                return self.reply(404, {"error": "Not found."})
        except Refusal as error:
            return self.reply(error.status, {"error": str(error)})
        self.reply(200, result)


def start(cfg, port=None):
    os.makedirs(cfg.data, exist_ok=True)
    app = App(cfg, Book(cfg.path("ledger.jsonl")))
    threading.Thread(target=app.work, daemon=True).start()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", cfg.port if port is None else port), Handler)
    server.daemon_threads = True
    server.app = app
    return server


def main(argv):
    cfg = Config(os.environ)
    if argv[1:] == ["verify"]:
        book = Book(cfg.path("ledger.jsonl"))
        print(f"{book.count} lines, head {book.head}")
        return 0
    if len(argv) == 3 and argv[1] == "release":
        app = App(cfg, Book(cfg.path("ledger.jsonl")))
        order = app.book.orders.get(argv[2])
        if not order or not order["invoice"]:
            print(f"{argv[2]}: no order with an invoice", file=sys.stderr)
            return 1
        remote = call(cfg, "GET", "/merchant/invoices/" + order["invoice"])
        state, total, currency = remote_fields(remote)
        print(f"{argv[2]}: CoinPayments says {state}, {total} {currency}; the order is {order['total']} USD")
        if state != "completed":
            print("not completed at CoinPayments: nothing issued", file=sys.stderr)
            return 1
        app.settle(argv[2], remote, released=True)
        print(f"{argv[2]}: issued {app.book.orders[argv[2]]['issued']}")
        return 0
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
