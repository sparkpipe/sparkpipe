#!/usr/bin/env python3
import base64
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
SPENDING = ("/market/book-hours", "/market/book-weeks", "/market/waitlist")
LIMITS = {"code": (20, 600), "code-email": (5, 3600), "verify": (30, 600), "order": (10, 600), "read": (240, 600), "market": (240, 600), "gateway": (600, 600)}


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
        self.reconcile_seconds = int(env.get("SPARKPIPE_PAY_RECONCILE_SECONDS", "300"))
        self.smtp = env.get("SPARKPIPE_PAY_SMTP", "127.0.0.1:25")
        self.mail_from = env.get("SPARKPIPE_PAY_MAIL_FROM", "SparkPipe <noreply@sparkpipe.ai>")
        self.notify = env.get("SPARKPIPE_PAY_NOTIFY", "hyor@sparkpipe.ai")
        self.swift_bank = {name: env.get(key, "").strip() for name, key in SWIFT_SETTINGS.items()}
        if any(self.swift_bank.values()) and not (all(self.swift_bank.values()) and BIC.match(self.swift_bank["bic"]) and all(len(v) <= 200 and v.isprintable() for v in self.swift_bank.values())):
            raise ValueError("Set all of " + ", ".join(SWIFT_SETTINGS.values()) + " on one line each, with an 8 or 11 character SWIFT/BIC, or leave them all empty")
        self.swift = all(self.swift_bank.values()) and not self.sandbox

    def path(self, name):
        return os.path.join(self.data, name)

    def pricing(self):
        with open(os.path.join(self.site, "pricing.js"), encoding="utf-8") as handle:
            text = handle.read()
        return json.loads(text[text.index("{"):text.rindex("}") + 1])

    def terms(self, name):
        with open(os.path.join(self.site, "terms", name), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

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
        order = self.orders.get(data.get("order"))
        if kind in market.KINDS:
            self.market.apply(line)
        elif kind == "account":
            self.accounts[data["account"]] = dict(data, created=line["at"], accepted=None)
            self.emails[data["email"]] = data["account"]
        elif kind == "acceptance":
            self.accounts[data["account"]]["accepted"] = dict({name: data.get(name) for name in TERMS}, at=line["at"])
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

    def who(self, account):
        record = self.book.accounts.get(account, {})
        return f"{record.get('company')} <{record.get('email')}> ({account})"

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

    def require_terms(self, account, body, ip, agent):
        if not self.accepted(account, self.current_terms()):
            if body.get("accept") is not True:
                raise Refusal("The Customer Agreement has changed since you accepted it. Accept the current text to continue.", 409, "terms")
            self.accept(account, ip, agent)

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

    def summary(self, account):
        with self.book.lock:
            record = self.book.accounts[account]
            mine = [o for o in reversed(list(self.book.orders.values())) if o["account"] == account]
            return {
                "email": record["email"],
                "company": record["company"],
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

    def market_state(self, account):
        state = self.market.public()
        state["account"] = None
        if account:
            record = self.book.accounts[account]
            state["account"] = dict(self.market.view(account), email=record["email"], company=record["company"], termsCurrent=self.accepted(account, self.current_terms()))
        return state

    def market_action(self, path, body, account, ip, agent):
        self.throttle("market", ip)
        if path in SPENDING:
            with self.book.lock:
                self.require_terms(account, body, ip, agent)
        actions = {
            "/market/buy": lambda: self.buy(body, account, ip, agent),
            "/market/quote-weeks": lambda: self.market.quote_weeks(account, body),
            "/market/book-hours": lambda: self.market.book_hours(account, body),
            "/market/book-weeks": lambda: self.market.book_weeks(account, body),
            "/market/waitlist": lambda: self.market.join(account, body),
            "/market/waitlist/leave": lambda: self.market.leave(account),
            "/market/keys": lambda: self.market.create_key(account, optional_text(body.get("note"), 100)),
            "/market/keys/revoke": lambda: self.market.revoke_key(account, body.get("id")),
        }
        sandbox = {
            "/market/sandbox/clock": lambda: self.market.move_clock(market.count_field(body.get("hours"), "number of hours", 9600)),
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
            self.send_header("Set-Cookie", f"{COOKIE}={cookie}; Max-Age={SESSION_SECONDS if cookie else 0}; Path={cfg.prefix}; HttpOnly; SameSite=Lax{secure}")
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

    def api_error(self, error):
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
        if path == "/v1/models":
            return self.models()
        match = HOLDERS_PATH.match(path)
        if match:
            return self.holders(match.group(1), query)
        routes = {
            "/account": lambda: app.summary(app.need(self.headers)),
            "/market/state": lambda: app.market_state(app.signed_in(self.headers)),
            "/market/ledger": lambda: app.ledger(app.need(self.headers)),
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

    def do_POST(self):
        app = self.server.app
        path = urllib.parse.urlsplit(self.path).path
        limit = 262144 if path in ("/coinpayments", "/toss/webhook") else 8388608 if path.startswith("/v1/") else 16384
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
        origin = self.headers.get("Origin")
        if origin and origin != app.cfg.origin:
            return self.reply(403, {"error": "Cross-site requests are refused."})
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self.reply(400, {"error": "The request is not a JSON object."})
        if path == "/v1/chat/completions":
            return self.chat(body)
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
            if path.startswith("/market/"):
                return self.reply(200, app.market_action(path, body, app.need(self.headers), self.client_ip(), self.agent()))
            return self.reply(404, {"error": "Not found."})
        except Refusal as error:
            self.refuse(error)

    def holders(self, fleet, query):
        app = self.server.app
        if not app.cfg.pool_key or not hmac.compare_digest(self.bearer().encode(), app.cfg.pool_key.encode()):
            return self.refuse(Refusal("Send the upstream key as Authorization: Bearer.", 401, "invalid_key"))
        try:
            start, end = (market.slot_field(query.get(name, [""])[0], name) for name in ("from", "to"))
            self.reply(200, app.market.holder_hours(fleet, start, end))
        except Refusal as error:
            self.refuse(error)

    def models(self):
        app = self.server.app
        try:
            app.throttle("gateway", self.bearer()[-16:] or self.client_ip())
            app.market.gateway(self.bearer(), None, choose=False)
        except Refusal as error:
            return self.api_error(error)
        self.reply(200, {"object": "list", "data": [{"id": m["id"], "object": "model", "owned_by": "sparkpipe"} for m in app.cfg.pricing()["models"]]})

    def stream_head(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def event(self, data):
        self.wfile.write(b"data: " + (data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()) + b"\n\n")
        self.wfile.flush()

    def chat(self, body):
        app = self.server.app
        try:
            app.throttle("gateway", self.bearer()[-16:] or self.client_ip())
            pricing = app.cfg.pricing()
            if not app.cfg.sandbox and not app.cfg.pool_upstream:
                raise Refusal("The API opens once it is connected to the Fleets.", 503, "closed")
            messages, extra = body.get("messages"), body.get("sparkpipe", {})
            if not isinstance(messages, list) or not messages:
                raise Refusal("Send the conversation as a non-empty messages list.", 400, "invalid_messages")
            if not isinstance(extra, dict) or not isinstance(extra.get("fleet", ""), str) or set(extra) - {"fleet"}:
                raise Refusal("Give sparkpipe as an object whose only field is fleet, a Fleet id.", 400, "invalid_sparkpipe")
            if body.get("model") not in [m["id"] for m in pricing["models"]]:
                raise Refusal("Choose one of the models: " + ", ".join(m["id"] for m in pricing["models"]) + ".", 404, "model_not_found")
            record, fleet = app.market.gateway(self.bearer(), extra.get("fleet"))
            ticket = {"account": record["account"], "key": record["key"], "model": body["model"], "fleet": fleet}
        except Refusal as error:
            return self.api_error(error)
        body = {k: v for k, v in body.items() if k != "sparkpipe"}
        if app.cfg.sandbox:
            return self.simulated(body, messages, ticket)
        return self.forwarded(body, ticket)

    def simulated(self, body, messages, ticket):
        limit = body.get("max_tokens") if isinstance(body.get("max_tokens"), int) and not isinstance(body.get("max_tokens"), bool) else 128
        text, tokens_in, tokens_out = market.simulate(ticket["model"], ticket["fleet"], messages, min(max(limit, 1), 512))
        made, ident = int(time.time()), "chatcmpl-" + secrets.token_hex(12)
        usage = {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": tokens_in + tokens_out}
        note = {"fleet": ticket["fleet"]}
        if body.get("stream") is True:
            self.stream_head()
            for word in text.split(" "):
                self.event({"id": ident, "object": "chat.completion.chunk", "created": made, "model": ticket["model"], "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]})
                time.sleep(0.025)
            self.event({"id": ident, "object": "chat.completion.chunk", "created": made, "model": ticket["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage, "sparkpipe": note})
            return self.event(b"[DONE]")
        time.sleep(min(1.0, tokens_out / 40))
        self.reply(200, {"id": ident, "object": "chat.completion", "created": made, "model": ticket["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage, "sparkpipe": note})

    def forwarded(self, body, ticket):
        app = self.server.app
        note = {"fleet": ticket["fleet"]}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {app.cfg.pool_key}", "X-SparkPipe-Public": "1", "X-SparkPipe-Fleet": ticket["fleet"], "X-SparkPipe-Account": ticket["account"]}
        payload = json.dumps(body, ensure_ascii=False).encode()
        request = urllib.request.Request(app.cfg.pool_upstream + "/chat/completions", data=payload, method="POST", headers=headers)
        started = False
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                if body.get("stream") is not True:
                    data = json.loads(response.read())
                    data["sparkpipe"] = note
                    return self.reply(200, data)
                self.stream_head()
                started = True
                for line in response:
                    text = line.strip()
                    if not text.startswith(b"data: "):
                        continue
                    if text == b"data: [DONE]":
                        self.event({"object": "chat.completion.chunk", "choices": [], "sparkpipe": note})
                        return self.event(b"[DONE]")
                    self.event(text[6:])
        except urllib.error.HTTPError as error:
            detail = error.read()[:4096]
            self.send_response(error.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(detail)))
            self.end_headers()
            self.wfile.write(detail)
        except (urllib.error.URLError, OSError, ValueError) as error:
            log(f"fleet upstream: {error}")
            if not started:
                self.api_error(Refusal("The Fleet did not answer. Try again in a minute.", 502, "upstream_error"))


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
    lot = app.market.credit(account, credits, argv[4].strip()[:200])
    print(f"credited {lot['hc']} HC to {argv[2]}, lot {lot['lot']}, lapsing {lot['expires']}")
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
        print("usage: sparkpipe_pay.py [serve|verify|maintenance FLEET YYYY-MM-DD [HOURS]|credit EMAIL HC NOTE|release ORDER|refund ORDER CLAUSE KRW|USD|all REASON [BANK ACCOUNT HOLDER]|tax-invoices|tax-invoice ORDER NUMBER|swift-confirm ORDER|REFERENCE USD [NOTE]|swift-cancel ORDER|REFERENCE REASON]", file=sys.stderr)
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
