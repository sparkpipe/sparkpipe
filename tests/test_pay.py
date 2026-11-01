#!/usr/bin/env python3

import base64
import contextlib
import email
import io
import email.policy
import hashlib
import hmac
import http.server
import json
import os
import re
import shutil
import socketserver
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import sparkpipe_pay as sp

SECRET = b"test-client-secret"
CLIENT = "test-client-id"
TOSS_LIVE = ("live_gck_fake_client", "live_gsk_fake_secret")
TOSS_TEST = ("test_gck_fake_client", "test_gsk_fake_secret")
NOW = [datetime(2026, 11, 2, 3, 0, 0, tzinfo=timezone.utc)]


def docs_signature(method, url, stamp, body):
    message = "﻿" + method + url + CLIENT + stamp
    return base64.b64encode(hmac.new(SECRET, message.encode("utf-8") + body, hashlib.sha256).digest()).decode()


WINDOWS = {"americas": {"id": "americas", "from": 0, "to": 8, "localDay": -1}, "asia": {"id": "asia", "from": 8, "to": 16, "localDay": 0}, "europe": {"id": "europe", "from": 16, "to": 24, "localDay": 0}}


def plan(number, email, windows, days="every", fleets=1, months=24, start="2026-12-15", payment="gates", lock=None):
    found = {"id": f"SP-PLAN-{number}", "email": email, "company": "Acme", "start": start, "months": months, "windows": [WINDOWS[w] for w in windows], "days": days, "fleets": fleets,
             "hourPrice": "15.00", "payment": payment, "releaseFee": "0.10", "maintenanceHoursPerWeek": 3}
    if lock:
        found["lock"] = lock
    return found


def record(app, found):
    data = {key: value for key, value in found.items() if key != "id"}
    app.book.append("plan_order", dict(data, plan=found["id"], account=app.book.emails.get(found["email"].lower()), email=found["email"].lower(), lock=found.get("lock", []), source="signed"))


def set_plans(app, plans):
    keep = {found["id"] for found in plans}
    today = sp.local_day(app.market.now()).isoformat()
    for ident, held in list(app.book.plans.items()):
        if ident not in keep and not held["ended"]:
            app.book.append("plan_end", {"plan": ident, "account": held["account"], "day": today, "reason": "test"})
    for found in plans:
        record(app, found)


def check_dates(failures):
    pricing = json.loads((ROOT / "site" / "pricing.js").read_text().split("=", 1)[1].strip().rstrip(";"))
    hours = sp.tranches(pricing, plan(1, "a@x", ["asia"]))
    if (len(hours), hours[0], hours[1], hours[14], hours[-1], sum(hours)) != (25, 136, 248, 232, 112, 5848):
        failures.append(f"tranches of the Customer Agreement's example plan: {hours}")
    if (sum(hours[:6]), sum(hours[:12])) != (1344, 2808):
        failures.append("the example plan's gates differ from the Customer Agreement")
    if sum(sp.tranches(pricing, plan(1, "a@x", ["asia"], "weekdays"))) != 4184 or sum(sp.tranches(pricing, plan(1, "a@x", ["europe"], months=3))) != 720:
        failures.append("a weekday plan or a 3-month plan holds the wrong hours")
    if sum(sp.tranches(pricing, plan(1, "a@x", ["americas", "asia", "europe"], fleets=2))) != 2 * 17544:
        failures.append("a full-day plan on two fleets holds the wrong hours")
    week = sp.plan_pattern(pricing, plan(1, "a@x", ["americas"], "weekdays"))
    if [sum(week[d * 24:d * 24 + 24]) for d in range(7)] != [0, 8, 8, 8, 8, 8, 0]:
        failures.append(f"an Americas weekday plan does not hold KST Tuesday to Saturday, the Americas' Monday to Friday: {week}")
    overlapping = dict(plan(1, "a@x", ["asia"]), windows=[WINDOWS["asia"], dict(WINDOWS["europe"], **{"from": 15})])
    for bad in (plan(1, "a@x", ["asia"], lock=["F1"]), plan(1, "a@x", ["americas", "asia", "europe"], "weekdays", lock=["F1"]), plan(1, "a@x", ["asia"], "weekends"), plan(1, "a@x", []), overlapping, dict(plan(1, "a@x", ["asia"]), releaseFee="1.5")):
        try:
            sp.tranches(pricing, bad)
            failures.append(f"an invalid plan is accepted: {bad}")
        except ValueError:
            pass
    asia = plan(1, "a@x", ["asia"])
    first, last = date(2026, 12, 15), date(2026, 12, 31)
    if [sp.hours_before(pricing, asia, first, last, datetime(2026, 12, 20, hour, minute, tzinfo=timezone.utc)) for hour, minute in ((3, 0), (3, 30), (8, 0), (15, 0))] != [44, 45, 48, 48]:
        failures.append("Plan hours that occur before a payment are counted wrongly")
    if sp.term_end(date(2028, 2, 29), 24) != date(2030, 2, 28) or sp.term_end(date(2026, 12, 1), 12) != date(2027, 11, 30):
        failures.append("term_end differs from the Customer Agreement")


def check_covered(failures):
    credits = {"product": "credits", "hours": "168.000000", "price": "20.00", "net": "3360.00", "total": "3696.00"}
    if sp.covered(credits, sp.money("22.99")) != ("1.045000", "20.90") or sp.covered(credits, sp.money("0.01")) != ("0.000000", "0.00") or sp.covered(credits, sp.money("9999.00")) != ("168.000000", "3360.00"):
        failures.append(f"HC covered by a part payment: {sp.covered(credits, sp.money('22.99'))}")
    prepay = {"product": "plan", "hours": "6.6666", "net": "100.00", "total": "110.00", "price": "15.00"}
    if sp.covered(prepay, sp.money("55.00")) != ("3.3333", "50.00"):
        failures.append(f"prepayment covered by a part payment: {sp.covered(prepay, sp.money('55.00'))}")


def check_sessions(failures):
    key = b"k" * 32
    token = sp.session_token(key, "ACX", 2000)
    if sp.session_account(key, token, 1999) != "ACX":
        failures.append("a fresh session token is refused")
    if sp.session_account(key, token, 2001) is not None:
        failures.append("an expired session token is accepted")
    if sp.session_account(b"j" * 32, token, 1999) is not None or sp.session_account(key, token.replace("ACX", "ACY"), 1999) is not None:
        failures.append("a forged session token is accepted")
    for bad in (None, "", "a.b", "ACX.x.y", "ACX.2000.00"):
        if sp.session_account(key, bad, 1999) is not None:
            failures.append(f"a malformed session token {bad!r} is accepted")


class FakeCoinPayments(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def answer(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def signed(self, body):
        url = f"http://127.0.0.1:{self.server.server_address[1]}{self.path}"
        expect = docs_signature(self.command, url, self.headers["X-CoinPayments-Timestamp"], body)
        return self.headers["X-CoinPayments-Client"] == CLIENT and hmac.compare_digest(expect, self.headers["X-CoinPayments-Signature"])

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if not self.signed(body):
            return self.answer(401, {"error": "signature"})
        if self.server.down:
            return self.answer(503, {"error": "down"})
        data = json.loads(body)
        key = f"inv{len(self.server.invoices) + 1}"
        self.server.invoices[key] = {"status": "unpaid", "total": data["amount"]["total"], "body": data}
        self.answer(200, {"invoices": [{"id": key, "link": f"https://pay.example/i/{key}", "checkoutLink": f"https://pay.example/c/{key}"}]})

    def do_GET(self):
        if not self.signed(b""):
            return self.answer(401, {"error": "signature"})
        key = self.path.rsplit("/", 1)[1]
        invoice = self.server.invoices[key]
        arrived = invoice.get("received", invoice["total"] if invoice["status"] in ("paid", "completed") else "0")
        self.answer(200, {"id": key, "status": invoice["status"], "amount": {"breakdown": {}, "total": invoice["total"]}, "currency": {"id": "5057", "symbol": "USD"}, "payments": [{"paymentCurrencySymbol": invoice.get("paidIn", "USDC.BASE"), "nativeActualAmount": arrived}]})


class FakeToss(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def answer(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorised(self):
        return self.headers.get("Authorization") == "Basic " + base64.b64encode((self.server.secret + ":").encode()).decode()

    def view(self, key):
        found = self.server.payments[key]
        return {name: value for name, value in found.items() if not name.startswith("_")}

    def do_GET(self):
        key = urllib.parse.unquote(self.path.rsplit("/", 1)[1])
        self.server.calls.append(("GET", key, None, None))
        if not self.authorised():
            return self.answer(401, {"code": "UNAUTHORIZED_KEY", "message": "key"})
        if key not in self.server.payments:
            return self.answer(404, {"code": "NOT_FOUND_PAYMENT", "message": "none"})
        self.answer(200, self.view(key))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        idem = self.headers.get("Idempotency-Key")
        parts = self.path.split("/")
        key = body.get("paymentKey") if self.path == "/v1/payments/confirm" else urllib.parse.unquote(parts[3])
        self.server.calls.append(("POST", key, body, idem))
        if not self.authorised():
            return self.answer(401, {"code": "UNAUTHORIZED_KEY", "message": "key"})
        if idem in self.server.done:
            return self.answer(*self.server.done[idem])
        found = self.server.payments.get(key)
        if found is None:
            result = (404, {"code": "NOT_FOUND_PAYMENT_SESSION", "message": "none"})
        elif self.path == "/v1/payments/confirm":
            if body["amount"] != found["_authorised"] or body["orderId"] != found["orderId"]:
                result = (400, {"code": "INVALID_REQUEST", "message": "amount"})
            elif found.get("_decline"):
                found["status"] = "ABORTED"
                result = (400, {"code": found["_decline"], "message": "declined"})
            else:
                found["status"] = found.pop("_after", "DONE")
                result = (200, self.view(key))
        else:
            balance = found["balanceAmount"]
            amount = body.get("cancelAmount", balance)
            if "refundableAmount" in body and body["refundableAmount"] != balance:
                result = (400, {"code": "NOT_MATCHES_REFUNDABLE_AMOUNT", "message": "balance"})
            elif found["method"] == "가상계좌" and found["status"] == "DONE" and "refundReceiveAccount" not in body:
                result = (400, {"code": "INVALID_REFUND_ACCOUNT_INFO", "message": "account"})
            else:
                found["balanceAmount"] = balance - amount
                found["status"] = "CANCELED" if found["balanceAmount"] == 0 else "PARTIAL_CANCELED"
                found.setdefault("cancels", []).append({"cancelAmount": amount, "cancelReason": body["cancelReason"]})
                result = (200, self.view(key))
        if idem:
            self.server.done[idem] = result
        self.answer(*result)


def toss_payment(fake, key, order, amount, method="카드", after="DONE", total=None, **extra):
    found = {"paymentKey": key, "orderId": order, "status": "IN_PROGRESS", "totalAmount": amount if total is None else total, "balanceAmount": amount if total is None else total,
             "currency": "KRW", "method": method, "approvedAt": "2026-11-02T12:00:00+09:00", "receipt": {"url": f"https://dashboard.tosspayments.com/receipt/{key}"}, "_authorised": amount, "_after": after}
    if method == "가상계좌":
        found["virtualAccount"] = {"accountNumber": "X12345678901", "bankCode": "88", "dueDate": "2026-11-03T12:00:00+09:00"}
    found.update(extra)
    fake.payments[key] = found
    return found


class FakeSMTP(socketserver.StreamRequestHandler):
    def handle(self):
        def say(text):
            self.wfile.write((text + "\r\n").encode())
        say("220 fake")
        lines, rcpt = None, []
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            text = raw.decode("utf-8", "replace").rstrip("\r\n")
            if lines is not None:
                if text == ".":
                    self.server.mail.append((rcpt, email.message_from_string("\n".join(lines), policy=email.policy.default)))
                    lines, rcpt = None, []
                    say("250 queued")
                else:
                    lines.append(text[1:] if text.startswith("..") else text)
                continue
            verb = text.split(" ", 1)[0].upper()
            if verb == "DATA":
                lines = []
                say("354 go on")
            elif verb == "RCPT":
                rcpt.append(text.split(":", 1)[1].strip(" <>"))
                say("250 ok")
            elif verb == "QUIT":
                say("221 bye")
                return
            else:
                say("250 ok")


class Client:
    def __init__(self, port, origin=None):
        self.port = port
        self.cookie = None
        self.origin = origin

    def __call__(self, method, path, body=None, headers=None, raw=None):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        sent = {"Content-Type": "application/json"}
        if self.cookie:
            sent["Cookie"] = self.cookie
        if self.origin:
            sent["Origin"] = self.origin
        sent.update(headers or {})
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers=sent)
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                status, payload, given = response.status, response.read(), response.headers.get("Set-Cookie")
        except urllib.error.HTTPError as error:
            status, payload, given = error.code, error.read(), error.headers.get("Set-Cookie")
        if given:
            value = given.split(";", 1)[0]
            self.cookie = None if value.endswith("=") else value
            self.last_cookie = given
        return status, json.loads(payload or b"{}")


def last_code(mailbox, to):
    for rcpt, message in reversed(mailbox.mail):
        if to in rcpt:
            found = re.search(r"\b(\d{6})\b", message["Subject"])
            if found:
                return found.group(1)
    return None


def sign_in(client, mailbox, address, **extra):
    status, _ = client("POST", "/code", {"email": address})
    if status != 200:
        return status, {}
    return client("POST", "/verify", dict({"email": address, "code": last_code(mailbox, address.lower())}, **extra))


def webhook(client, cfg, invoice, kind="InvoiceCompleted", stamp=None, tamper=False):
    stamp = stamp or NOW[0].strftime("%Y-%m-%dT%H:%M:%S")
    raw = json.dumps({"id": invoice, "type": kind, "invoice": {"id": invoice, "state": kind[7:]}}).encode()
    signature = docs_signature("POST", cfg.webhook_url, stamp, raw)
    headers = {"X-CoinPayments-Client": CLIENT, "X-CoinPayments-Timestamp": stamp, "X-CoinPayments-Signature": signature}
    return client("POST", "/coinpayments", headers=headers, raw=raw + (b" " if tamper else b""))


def ledger_types(cfg):
    return [json.loads(line)["type"] for line in open(cfg.path("ledger.jsonl"), encoding="utf-8")]


def check_service(failures):
    work = Path(tempfile.mkdtemp(prefix="sparkpipe-pay-"))
    try:
        (work / "site" / "terms").mkdir(parents=True)
        shutil.copy(ROOT / "site" / "pricing.js", work / "site" / "pricing.js")
        for name in sp.TERMS.values():
            shutil.copy(ROOT / "site" / "terms" / name, work / "site" / "terms" / name)
        for name in ("agreement.html", "agreement-ko.html"):
            shutil.copy(ROOT / "site" / name, work / "site" / name)
        (work / "secret").write_text("test-client-secret\n")
        fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeCoinPayments)
        fake.invoices, fake.down = {}, False
        threading.Thread(target=fake.serve_forever, daemon=True).start()
        mailbox = socketserver.ThreadingTCPServer(("127.0.0.1", 0), FakeSMTP)
        mailbox.daemon_threads = True
        mailbox.mail = []
        threading.Thread(target=mailbox.serve_forever, daemon=True).start()
        cfg = sp.Config({
            "SPARKPIPE_PAY_DATA": str(work / "data"), "SPARKPIPE_PAY_SITE": str(work / "site"),
            "COINPAYMENTS_API": f"http://127.0.0.1:{fake.server_address[1]}/api/v2",
            "COINPAYMENTS_CLIENT_ID": CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(work / "secret"),
            "SPARKPIPE_PAY_SMTP": f"127.0.0.1:{mailbox.server_address[1]}",
            "SPARKPIPE_PAY_SUPPLIER_ADDRESS_KO": "서울특별시 용산구 시험로 1", "SPARKPIPE_PAY_SUPPLIER_ADDRESS_EN": "1 Test-ro, Yongsan-gu, Seoul",
        })
        sp.clock = lambda: NOW[0]
        sp.LIMITS["order"] = (40, 600)
        sp.LIMITS["plan"] = (200, 600)
        server = sp.start(cfg, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port, app = server.server_address[1], server.app
        buyer = Client(port, origin="https://sparkpipe.ai")
        account_flow(failures, cfg, app, buyer, mailbox, port)
        credits_flow(failures, cfg, app, fake, buyer, mailbox)
        part_payment_flow(failures, cfg, app, fake, buyer, mailbox)
        plan_order_flow(failures, cfg, app, fake, buyer, mailbox, port, work)
        prepayment_flow(failures, cfg, app, fake, buyer, mailbox, port)
        lead_flow(failures, app, buyer, mailbox, work)
        book = sp.Book(cfg.path("ledger.jsonl"))
        if book.count != app.book.count or book.head != app.book.head or book.emails != app.book.emails:
            failures.append("a reloaded ledger differs from the live one")
        second = sp.Book(cfg.path("ledger.jsonl"))
        second.append("note", {"by": "second writer"})
        app.book.append("note", {"by": "service"})
        if sp.Book(cfg.path("ledger.jsonl")).count != second.count + 1:
            failures.append("two writers break the ledger chain")
        lines = open(cfg.path("ledger.jsonl"), encoding="utf-8").read().splitlines()
        index = next(i for i, line in enumerate(lines) if '"type":"account"' in line)
        lines[index] = lines[index].replace('Acme', 'Acmf', 1)
        (work / "tampered.jsonl").write_text("\n".join(lines) + "\n")
        try:
            sp.Book(str(work / "tampered.jsonl"))
            failures.append("a tampered ledger loads")
        except ValueError:
            pass
        if oct((work / "data" / "session_key").stat().st_mode & 0o777) != "0o600":
            failures.append("the session key is readable by others")
        server.shutdown()
        fake.shutdown()
        mailbox.shutdown()
    finally:
        shutil.rmtree(work)


def account_flow(failures, cfg, app, buyer, mailbox, port):
    if buyer("GET", "/account")[0] != 401:
        failures.append("the account is readable without signing in")
    status, _ = buyer("POST", "/code", {"email": "Ops@Acme.example"})
    code = last_code(mailbox, "ops@acme.example")
    if status != 200 or not code:
        failures.append(f"no sign-in code was emailed: {status} {[m[0] for m in mailbox.mail]}")
        return
    wrong = "000000" if code != "000000" else "111111"
    if buyer("POST", "/verify", {"email": "ops@acme.example", "code": wrong, "company": "Acme", "accept": True})[0] != 403:
        failures.append("a wrong code signs in")
    status, refused = buyer("POST", "/verify", {"email": "ops@acme.example", "code": code})
    if status != 404 or refused.get("reason") != "new":
        failures.append(f"a code for an unknown email without a company opens an account: {status} {refused}")
    sign_in(buyer, mailbox, "ops@acme.example", company="Acme 주식회사")
    if "ops@acme.example" in app.book.emails:
        failures.append("an account opens without accepting the Hourly Terms")
    status, summary = sign_in(buyer, mailbox, "Ops@Acme.example", company="Acme 주식회사", accept=True)
    if status != 200 or summary.get("company") != "Acme 주식회사" or not summary.get("termsCurrent") or summary.get("hc") != "0.000000":
        failures.append(f"opening an account: {status} {summary}")
        return
    if "HttpOnly" not in buyer.last_cookie or "SameSite=Lax" not in buyer.last_cookie or "Path=/pay/api" not in buyer.last_cookie:
        failures.append(f"session cookie attributes: {buyer.last_cookie}")
    record = [json.loads(line) for line in open(cfg.path("ledger.jsonl"), encoding="utf-8") if '"acceptance"' in line][-1]["data"]
    if record["ko"] != cfg.terms("agreement-ko.md") or record["rules_ko"] != cfg.terms("market-rules-ko.md") or record["rules_en"] != cfg.terms("market-rules-en.md") or "Hour Market Rules, Korean text, which prevails" not in record["statement"] or "Acme 주식회사" not in record["statement"]:
        failures.append(f"acceptance record: {record}")
    if buyer("GET", "/account")[0] != 200:
        failures.append("a signed-in account cannot read itself")
    if buyer("POST", "/verify", {"email": "ops@acme.example", "code": last_code(mailbox, "ops@acme.example")})[0] != 403:
        failures.append("a sign-in code works twice")
    stranger = Client(port, origin="https://evil.example")
    if stranger("POST", "/code", {"email": "ops@acme.example"})[0] != 403:
        failures.append("a cross-site POST is accepted")
    forged = Client(port)
    forged.cookie = buyer.cookie.replace(buyer.cookie.split("=", 1)[1].split(".")[0], "ACAAAAAAAAAAAAAAAA")
    if forged("GET", "/account")[0] != 401:
        failures.append("a forged session reads an account")
    other = Client(port)
    status, _ = sign_in(other, mailbox, "ops@acme.example")
    if status != 200 or other("POST", "/signout", {})[0] != 200 or other.cookie is not None:
        failures.append("signing in again or signing out fails")


def credits_flow(failures, cfg, app, fake, buyer, mailbox):
    base = {"product": "credits", "amount": "80", "coin": "USDC.BASE"}
    for coin in ("BTC", "TUSD.TRC20", "base", None):
        if buyer("POST", "/order", dict(base, coin=coin))[0] != 400:
            failures.append(f"an order in {coin} is accepted")
    for amount in ("9.99", "100000.01", "12.345", "ten"):
        if buyer("POST", "/order", dict(base, amount=amount))[0] != 400:
            failures.append(f"an HC order of {amount} USD is accepted")
    if buyer("POST", "/order", {"product": "hourly", "hours": 4, "start": "2026-12-01", "coin": "USDC.BASE"})[0] != 400:
        failures.append("fleet-hours are still sold by the hour")
    anonymous = Client(buyer.port)
    if anonymous("POST", "/order", base)[0] != 401:
        failures.append("an order without signing in is accepted")
    status, placed = buyer("POST", "/order", base)
    if status != 200 or not placed.get("checkout", "").startswith("https://pay.example/c/"):
        failures.append(f"HC order: {status} {placed}")
        return
    invoice = fake.invoices["inv1"]["body"]
    expected = {"currency": "5057", "invoiceId": placed["order"], "amount": {"breakdown": {"subtotal": "80.00", "taxTotal": "8.00"}, "total": "88.00"}}
    if any(invoice.get(key) != value for key, value in expected.items()):
        failures.append(f"invoice body: {invoice}")
    if invoice["payment"] != {"paymentCurrency": sp.COINS["USDC.BASE"], "refundEmail": "ops@acme.example"} or invoice["buyer"] != {"companyName": "Acme 주식회사", "emailAddress": "ops@acme.example"}:
        failures.append(f"invoice payment or buyer: {invoice['payment']} {invoice['buyer']}")
    if invoice["items"][0]["name"] != "4.000000 SparkPipe hour-credits" or invoice["webhooks"][0]["notificationsUrl"] != "https://sparkpipe.ai/pay/api/coinpayments":
        failures.append(f"invoice items or webhooks: {invoice['items']} {invoice['webhooks']}")
    if invoice["successUrl"] != f"https://sparkpipe.ai/account.html?order={placed['order']}":
        failures.append(f"invoice return address: {invoice['successUrl']}")
    status, shown = buyer("GET", "/order/" + placed["order"])
    if status != 200 or shown["state"] != "unpaid" or shown["checkout"] != placed["checkout"] or "email" in shown:
        failures.append(f"unpaid order status: {status} {shown}")
    if webhook(buyer, cfg, "inv1", tamper=True)[0] != 401 or webhook(buyer, cfg, "inv1", stamp="2026-11-02T02:50:00")[0] != 401:
        failures.append("a tampered or stale webhook is accepted")
    fake.invoices["inv1"]["status"] = "paid"
    webhook(buyer, cfg, "inv1", "InvoicePaid")
    app.jobs.join()
    if buyer("GET", "/order/" + placed["order"])[1]["state"] != "paid":
        failures.append("a paid invoice does not move the order to paid")
    fake.invoices["inv1"]["status"] = "completed"
    if webhook(buyer, cfg, "inv1")[0] != 200:
        failures.append("a signed webhook is refused")
    app.jobs.join()
    webhook(buyer, cfg, "inv1")
    app.jobs.join()
    app.mails.join()
    summary = buyer("GET", "/account")[1]
    if summary["hc"] != "4.000000" or ledger_types(cfg).count("hours_issued") != 1 or ledger_types(cfg).count("hc_lot") != 1:
        failures.append(f"completed order: {summary} {ledger_types(cfg)}")
    receipts = [m for rcpt, m in mailbox.mail if "ops@acme.example" in rcpt and placed["order"] in m["Subject"]]
    if len(receipts) != 1 or "88.00" not in receipts[0].get_content() or "4.000000 hour-credits" not in receipts[0].get_content() or "/market.html" not in receipts[0].get_content():
        failures.append(f"receipt mail: {[m['Subject'] for _, m in mailbox.mail]}")
    if not any(cfg.notify in rcpt and m["Subject"].startswith("Paid:") for rcpt, m in mailbox.mail):
        failures.append("we are not told of a payment")
    payment = [json.loads(line) for line in open(cfg.path("ledger.jsonl"), encoding="utf-8") if '"payment"' in line][-1]["data"]
    if payment["paidWith"] != ["USDC.BASE"]:
        failures.append(f"the payment line does not record the coin: {payment}")
    if buyer("POST", "/order", dict(base, amount="3280", coin="USDT.TRC20"))[0] != 200:
        failures.append("a second HC order is refused")
    fake.invoices["inv2"]["status"] = "completed"
    fake.invoices["inv2"]["total"] = "1.00"
    webhook(buyer, cfg, "inv2")
    app.jobs.join()
    if fake.invoices["inv2"]["body"]["payment"]["paymentCurrency"] != "9:TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t":
        failures.append("a USDT order on Tron does not preset USDT on Tron")
    second = next(o for o in buyer("GET", "/account")[1]["orders"] if o["net"] == "3280.00")
    if second["coin"] != "USDT.TRC20":
        failures.append(f"the order does not show its coin: {second}")
    if second["state"] != "mismatch" or second["issued"] or buyer("GET", "/account")[1]["hc"] != "4.000000":
        failures.append(f"an invoice that does not match its order issues HC: {second}")
    fake.down = True
    status, refused = buyer("POST", "/order", dict(base, amount="20"))
    fake.down = False
    if status != 502 or buyer("GET", "/account")[1]["orders"][0]["state"] != "failed":
        failures.append(f"a failed invoice is not recorded: {status} {refused}")
    for name in ("agreement-en.md", "market-rules-ko.md"):
        terms = Path(cfg.site) / "terms" / name
        terms.write_text(terms.read_text() + "\n")
        status, refused = buyer("POST", "/order", dict(base, amount="20"))
        if status != 409 or refused.get("reason") != "terms":
            failures.append(f"an order after {name} changed does not ask to accept the terms again: {status} {refused}")
        if buyer("POST", "/order", dict(base, amount="20", accept=True))[0] != 200 or not buyer("GET", "/account")[1]["termsCurrent"]:
            failures.append(f"accepting the terms again after {name} changed is not recorded")


def part_payment_flow(failures, cfg, app, fake, buyer, mailbox):
    status, placed = buyer("POST", "/order", {"product": "credits", "amount": "80", "coin": "USDC.ERC20"})
    if status != 200:
        failures.append(f"part payment order: {status} {placed}")
        return
    key = max(fake.invoices, key=lambda k: int(k[3:]))
    fake.invoices[key].update(status="pending", received="22.99", paidIn="DAI.ERC20")
    webhook(buyer, cfg, key, "InvoicePending")
    app.jobs.join()
    app.mails.join()
    shown = buyer("GET", "/order/" + placed["order"])[1]
    if shown["state"] != "pending" or shown["received"] != "22.99":
        failures.append(f"a part payment is not shown: {shown}")
    lines = app.book.count
    for value in ("22.97", "22.99", "22.98"):
        fake.invoices[key]["received"] = value
        app.reconcile()
        app.jobs.join()
    app.mails.join()
    if app.book.count != lines:
        failures.append("a part payment revalued by cents writes ledger lines")
    before = sp.Decimal(buyer("GET", "/account")[1]["hc"])
    fake.invoices[key]["status"] = "completed"
    webhook(buyer, cfg, key)
    app.jobs.join()
    app.mails.join()
    shown = buyer("GET", "/order/" + placed["order"])[1]
    after = sp.Decimal(buyer("GET", "/account")[1]["hc"])
    if shown["state"] != "completed" or not shown["issued"] or shown["issuedHours"] != "1.0445" or after != before + sp.Decimal("1.0445"):
        failures.append(f"a completed part payment does not issue HC in proportion: {shown} {before} {after}")
    note = [m for rcpt, m in mailbox.mail if "ops@acme.example" in rcpt and placed["order"] in m["Subject"]]
    if not note or "22.98 USD arrived" not in note[-1].get_content() or "1.044500 hour-credits" not in note[-1].get_content():
        failures.append("the receipt for a part payment does not say what arrived and what it bought")
    if not note or "Received in DAI: 22.98 USD" not in note[-1].get_content() or "account.html?order=" in note[-1].get_content():
        failures.append("the receipt for a part payment names the wrong coin or offers to pay the rest")
    payments = [json.loads(line)["data"] for line in open(cfg.path("ledger.jsonl"), encoding="utf-8") if '"payment"' in line]
    if [e["paidWith"] for e in payments if e["order"] == placed["order"]] != [["DAI.ERC20"]]:
        failures.append("the ledger does not record the stablecoin that was paid")
    if not any(cfg.notify in rcpt and m["Subject"].startswith("Paid in part:") for rcpt, m in mailbox.mail):
        failures.append("we are not told of a part payment that settled")
    if buyer("POST", "/order", {"product": "rest", "parent": placed["order"], "coin": "USDC.ERC20"})[0] != 400:
        failures.append("the rest of a part-paid order can still be ordered")


CHOICE = {"start": "2026-12-15", "months": 24, "windows": ["asia"], "days": "every", "fleets": 1, "lock": False, "models": ["kimi-k3"],
          "customer": {"kind": "company", "jurisdiction": "Delaware, United States", "registration": "7781234", "address": "1 Main St, Dover, DE"}}


def pay_plan(app, fake, client, plan, amount):
    status, placed = client("POST", "/order", {"product": "plan", "plan": plan, "amount": amount, "coin": "USDC.BASE"})
    key = max(fake.invoices, key=lambda k: int(k[3:]))
    fake.invoices[key]["status"] = "completed"
    app.reconcile()
    app.jobs.join()
    return status, placed


def place(client, choice):
    status, quote = client("POST", "/plans/quote", choice)
    if status != 200:
        return status, quote, quote
    status, placed = client("POST", "/plans/order", dict(choice, plan=quote["plan"], hash=quote["hash"], agree=True))
    return status, placed, quote


def plan_order_flow(failures, cfg, app, fake, buyer, mailbox, port, work):
    team, other = Client(port, origin="https://sparkpipe.ai"), Client(port, origin="https://sparkpipe.ai")
    sign_in(team, mailbox, "cto@plan.example", company="Plan Co Ltd", accept=True)
    sign_in(other, mailbox, "dev@solo.example", company="Jane Kim", accept=True)
    supplier = dict(cfg.supplier)
    cfg.supplier["ko"] = ""
    status, closed = team("POST", "/plans/quote", CHOICE)
    cfg.supplier.update(supplier)
    if status != 503 or closed.get("reason") != "plans_closed":
        failures.append(f"Plan Orders open without our notice address: {status} {closed}")
    if Client(port, origin="https://sparkpipe.ai")("POST", "/plans/quote", CHOICE)[0] != 401:
        failures.append("a Plan Order is quoted without signing in")
    customer = CHOICE["customer"]
    for bad in (dict(CHOICE, months=5), dict(CHOICE, start="2026-11-20"), dict(CHOICE, start="2027-01-02"), dict(CHOICE, start="15 Dec"), dict(CHOICE, windows=[]), dict(CHOICE, windows=["mars"]),
                dict(CHOICE, windows=["asia", "asia"]), dict(CHOICE, fleets=0), dict(CHOICE, fleets=4), dict(CHOICE, fleets=True), dict(CHOICE, days="weekends"), dict(CHOICE, lock=True),
                dict(CHOICE, models=[]), dict(CHOICE, models=["gpt"]), dict(CHOICE, customer=dict(customer, kind="trust")), dict(CHOICE, customer=dict(customer, registration="")),
                dict(CHOICE, customer=dict(customer, address="1 | 2")), dict(CHOICE, customer=dict(customer, brn="12345"))):
        status, refused = team("POST", "/plans/quote", bad)
        if status != 400:
            failures.append(f"an invalid Plan Order is quoted: {status} {refused} for {bad}")
    status, locked = team("POST", "/plans/quote", dict(CHOICE, windows=["americas", "asia", "europe"], lock=True))
    if status != 200 or locked.get("lock") != ["F1"] or locked.get("price") != "16.00" or locked.get("hours") != 17544:
        failures.append(f"a locked full-day plan is not assigned the free serving fleet at the lock price: {status} {locked}")
    status, refused = team("POST", "/plans/quote", dict(CHOICE, fleets=2))
    if status != 409 or refused.get("reason") != "capacity":
        failures.append(f"a plan for more fleets than serve is quoted: {status} {refused}")
    status, quote = team("POST", "/plans/quote", CHOICE)
    rows, ko = dict(map(tuple, quote.get("rows", {}).get("en", []))), dict(map(tuple, quote.get("rows", {}).get("ko", [])))
    if (status != 200 or quote["hours"] != 5848 or quote["total"] != "87720.00" or quote["withVat"] != "96492.00" or rows.get("Hours in the Term (clause 17.3)") != "5,848"
            or rows.get("Payment schedule (clause 20)") != "Gated: Gate 1 by 13 January 2027, Gate 2 by 12 February 2027, Gate 3 by 14 March 2027"
            or not rows.get("Customer", "").startswith("Plan Co Ltd, a company organised under the laws of Delaware, United States, registration number 7781234")
            or cfg.terms("agreement-ko.md") not in rows.get("Customer Agreement (clause 9.1)", "") or cfg.terms("market-rules-en.md") not in rows.get("Customer Agreement (clause 9.1)", "")
            or not rows.get("Customer Agreement (clause 9.1)", "").startswith("Version of ") or rows.get("Supplier notice address and email (clause 11.7)") != "1 Test-ro, Yongsan-gu, Seoul; hyor@sparkpipe.ai"
            or ko.get("플랜 기간의 시간(제17.3조)") != "5,848시간" or ko.get("시간창(제17.1조)") != "아시아 시간창(08:00–16:00 KST)" or ko.get("공급자 통지 주소 및 이메일(제11.7조)") != "서울특별시 용산구 시험로 1, hyor@sparkpipe.ai"
            or hashlib.sha256(quote["text"].encode()).hexdigest() != quote["hash"] or not quote["text"].index("# 플랜 주문서") < quote["text"].index("# Plan Order")):
        failures.append(f"the Plan Order quote differs from the Customer Agreement's example and form: {status} {quote.get('hours')} {quote.get('total')} {rows} {ko}")
        return
    order = dict(CHOICE, plan=quote["plan"], hash=quote["hash"])
    for body, reason in ((order, "agree"), (dict(order, hash="0" * 64, agree=True), "changed"), (dict(order, customer=dict(customer, address="2 Main St"), agree=True), "changed")):
        status, refused = team("POST", "/plans/order", body)
        if status != 409 or refused.get("reason") != reason:
            failures.append(f"a Plan Order is placed without accepting its exact text: {status} {refused}")
    sale = app.market.on_sale(app.market.now(), cfg.pricing())
    status, placed = team("POST", "/plans/order", dict(order, agree=True))
    app.mails.join()
    line = [json.loads(raw) for raw in open(cfg.path("ledger.jsonl"), encoding="utf-8") if '"plan_order"' in raw][-1]["data"]
    signature = line.get("signature") or {}
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(signature["key"])).verify(bytes.fromhex(signature["value"]), bytes.fromhex(quote["hash"]))
        signed = True
    except Exception:
        signed = False
    if status != 200 or placed.get("plan") != quote["plan"] or placed.get("hours") != 5848 or line["hash"] != quote["hash"] or line["text"] != quote["text"] or line["accepted"]["ip"] != "127.0.0.1" or not signed or team("GET", "/plans/key")[1].get("key") != signature.get("key"):
        failures.append(f"placing a Plan Order does not record the accepted text with our signature of its hash: {status} {placed} {signed}")
        return
    told = next((m for rcpt, m in mailbox.mail if "cto@plan.example" in rcpt and m["Subject"] == f"Plan Order {quote['plan']}: your SparkPipe plan"), None)
    if not told or quote["hash"] not in told.get_content() or "| Hours in the Term (clause 17.3) | 5,848 |" not in told.get_content() or "| 플랜 기간의 시간(제17.3조) | 5,848시간 |" not in told.get_content():
        failures.append("the customer is not emailed the Plan Order it accepted")
    if not any(cfg.notify in rcpt and m["Subject"].startswith("New Plan Order: Plan Co Ltd") for rcpt, m in mailbox.mail):
        failures.append("we are not told of a new Plan Order")
    if not app.market.on_sale(app.market.now(), cfg.pricing()) < sale - 2000:
        failures.append(f"an unpaid Plan Order does not take its hours off the HC on sale: {sale}")
    if team("POST", "/plans/order", dict(order, agree=True))[0] != 409:
        failures.append("a Plan Order is placed twice")
    status, refused, _ = place(team, dict(CHOICE, windows=["europe"]))
    if status != 409 or refused.get("reason") != "unpaid":
        failures.append(f"an account places a second plan while its first is unpaid: {status} {refused}")
    solo = {"kind": "individual", "address": "3 Side St, Busan"}
    status, refused = other("POST", "/plans/quote", dict(CHOICE, customer=solo))
    if status != 409 or refused.get("reason") != "capacity":
        failures.append(f"a plan is quoted on capacity another Plan Order keeps: {status} {refused}")
    status, refused = other("POST", "/plans/quote", dict(CHOICE, windows=["americas", "asia", "europe"], lock=True, customer=solo))
    if status != 409:
        failures.append(f"a fleet lock is quoted on a fleet another Plan Order needs: {status} {refused}")
    status, view = team("GET", "/plan/" + quote["plan"])
    if status != 200 or len(view.get("tranches", [])) != 25 or view.get("rows") != quote["rows"] or view.get("signature", {}).get("key") != signature["key"] or view.get("heldUntil") != "2028-12-14":
        failures.append(f"the Plan Order view: {status} {view.get('tranches', [])[:1]} {view.get('heldUntil')}")
    if other("GET", "/plan/" + quote["plan"])[0] != 404:
        failures.append("another account reads a Plan Order")
    pay_plan(app, fake, team, quote["plan"], "20160.00")
    holder = app.book.emails["cto@plan.example"]
    granted = [app.book.market.tranches[(holder, month)] for month in ("2026-12", "2027-01", "2027-05", "2027-06")]
    if granted != [136, 248, 248, 0]:
        failures.append(f"paying Gate 1 does not issue Tranches 1 to 6: {granted}")
    status, upfront, upfront_quote = place(other, dict(CHOICE, months=3, windows=["americas"], customer=solo))
    rows = dict(map(tuple, upfront_quote.get("rows", {}).get("en", [])))
    if status != 200 or upfront.get("payment") != "upfront" or upfront.get("hours") != 720 or upfront.get("total") != "12600.00" or rows.get("Customer") != "Jane Kim, an individual, of 3 Side St, Busan" or rows.get("Payment schedule (clause 20)") != "Upfront: the Price of the Term before the Start Date":
        failures.append(f"an individual's upfront plan: {status} {upfront} {rows}")
    pay_plan(app, fake, other, upfront.get("plan"), "6300.00")
    if app.book.market.tranches[(app.book.emails["dev@solo.example"], "2026-12")] != 0:
        failures.append("an upfront plan paid in part issues Tranche hours")
    status, europe, europe_quote = place(team, dict(CHOICE, windows=["europe"]))
    if status != 200:
        failures.append(f"a paid customer cannot order another plan on free capacity: {status} {europe}")
        return
    prices = work / "site" / "pricing.js"
    text = prices.read_text()
    prices.write_text(text.replace('{ "name": "Fleet 2", "kind": "M5U", "state": "building"', '{ "name": "Fleet 2", "kind": "M5U", "state": "serving"'))
    status, held = buyer("POST", "/market/reserve", {"fleet": "F1", "start": "2026-12-02T01:00Z", "model": "kimi-k3", "max": "2"})
    status, moved = team("POST", "/plans/quote", dict(CHOICE, start="2026-12-01", windows=["americas", "asia", "europe"], lock=True))
    prices.write_text(text)
    if not held.get("reservation") or status != 200 or moved.get("lock") != ["F2"]:
        failures.append(f"a fleet lock is not moved to another serving fleet when the first is reserved in its term: {held} {status} {moved}")
    start = NOW[0]
    NOW[0] = datetime(2026, 12, 16, 3, 0, 0, tzinfo=timezone.utc)
    kept = [found[6] for found in app.commitments(NOW[0])]
    if upfront.get("plan") in kept or quote["plan"] not in kept or europe["plan"] not in kept:
        failures.append(f"an unpaid upfront plan keeps capacity after its Start Date, or an unpaid plan in Phase A loses it: {kept}")
    NOW[0] = datetime(2026, 12, 20, 3, 0, 0, tzinfo=timezone.utc)
    sign_in(team, mailbox, "cto@plan.example")
    pay_plan(app, fake, team, europe["plan"], "20160.00")
    if app.book.market.tranches[(holder, "2026-12")] != 136 + 96:
        failures.append(f"Plan hours that occurred before they were paid for are issued: {app.book.market.tranches[(holder, '2026-12')]}")
    NOW[0] = start


def prepayment_flow(failures, cfg, app, fake, buyer, mailbox, port):
    plans = [
        plan("0001", "ops@acme.example", ["asia"], start="2026-11-01"),
        plan("0002", "OPS@acme.example", ["europe"], fleets=2, start="2026-11-01"),
        plan("0003", "else@other.example", ["americas"], start="2026-11-01"),
        plan("0004", "ops@acme.example", ["asia"], months=3, start="2026-11-10", payment="upfront"),
    ]
    set_plans(app, plans)
    hours = sp.tranches(cfg.pricing(), plans[0])
    found = buyer("GET", "/account")[1]["plans"]
    if [a["plan"] for a in found] != ["SP-PLAN-0001", "SP-PLAN-0002", "SP-PLAN-0004"]:
        failures.append(f"the account lists other customers' plans: {found}")
        return
    if found[1]["hours"] != 2 * sum(hours):
        failures.append(f"a plan for two fleets holds twice the hours: {found[1]}")
    if found[0]["hours"] != sum(hours) or found[0]["nextGate"] != {"name": "Gate 1", "due": "2026-11-30", "amount": f"{15 * sum(hours[:6])}.00"}:
        failures.append(f"plan standing: {found[0]}")
    if found[2]["payment"] != "upfront" or found[2]["nextGate"] != {"name": "Payment", "due": "2026-11-10", "amount": f"{15 * found[2]['hours']}.00"}:
        failures.append(f"an upfront plan is not due in full on its Start Date: {found[2]}")
    pay = {"product": "plan", "plan": "SP-PLAN-0001", "coin": "USDT.SOL"}
    if buyer("POST", "/order", dict(pay, plan="SP-PLAN-0003", amount="100.00"))[0] != 404:
        failures.append("an account can prepay another account's plan")
    if buyer("POST", "/order", dict(pay, amount="999999999.00"))[0] != 409:
        failures.append("a prepayment above the remaining price is ordered")
    status, placed = buyer("POST", "/order", dict(pay, amount="100.00"))
    key = max(fake.invoices, key=lambda k: int(k[3:]))
    invoice = fake.invoices[key]["body"]
    if status != 200 or invoice["items"][0]["quantity"] != {"value": 1, "type": "quantity"} or invoice["amount"]["total"] != "110.00":
        failures.append(f"prepayment order: {status} {placed} {invoice['amount']}")
        return
    fake.invoices[key]["status"] = "completed"
    app.reconcile()
    app.jobs.join()
    state = buyer("GET", "/account")[1]["plans"][0]
    if state["paid"] != "100.00" or not buyer("GET", "/order/" + placed["order"])[1]["issued"] or state["heldUntil"] != "2028-10-31":
        failures.append(f"reconciling a missed webhook does not issue the prepayment: {state}")
    account = app.book.emails["ops@acme.example"]
    if app.book.market.tranches[(account, "2026-11")] != 0:
        failures.append(f"a payment smaller than the Plan hours that already occurred issues Tranche hours: {app.book.market.tranches[(account, '2026-11')]}")
    pay_plan(app, fake, buyer, "SP-PLAN-0001", "3515.00")
    if app.book.market.tranches[(account, "2026-11")] != 240 - 12 or app.book.market.tranches[(account, "2026-12")] != 0:
        failures.append(f"a payment does not issue the rest of Tranche 1 less the 12 Plan hours that occurred before it was paid for: {app.book.market.tranches[(account, '2026-11')]} {app.book.market.tranches[(account, '2026-12')]}")
    NOW[0] = datetime(2026, 12, 1, 3, 0, 0, tzinfo=timezone.utc)
    state = buyer("GET", "/account")[1]["plans"]
    status, _ = buyer("POST", "/order", dict(pay, amount="100.00"))
    if "Gate 1 was missed" not in (state[0]["ended"] or "") or status != 409:
        failures.append(f"a missed gate does not stop prepayment: {state[0]} {status}")
    kept = [found[6] for found in app.commitments(NOW[0])]
    if state[0]["heldUntil"] != "2026-12-31" or "SP-PLAN-0001" not in kept or "SP-PLAN-0002" in kept:
        failures.append(f"after a missed gate, a plan keeps capacity beyond its last paid Tranche, or a plan that ended in Phase A keeps any: {state[0]['heldUntil']} {kept}")
    app.end_plan("SP-PLAN-0001", date(2026, 12, 10), "ended under clause 10.2")
    if buyer("GET", "/account")[1]["plans"][0]["heldUntil"] != "2026-12-09":
        failures.append("a Plan Order we end keeps capacity after its last day")
    for bad in (plan("0005", "ops@acme.example", ["asia"], fleets=2, start="2026-12-15"), plan("0006", "nobody@acme.example", ["europe"]), plan("0001", "ops@acme.example", ["europe"])):
        try:
            app.record_signed(cfg.pricing(), bad, NOW[0])
            failures.append(f"a signed Plan Order is recorded without capacity, an account or an unused number: {bad['id']}")
        except sp.Refusal:
            pass
    signed = app.record_signed(cfg.pricing(), plan("0007", "ops@acme.example", ["europe"], months=3, start="2027-01-01", payment="upfront"), NOW[0])
    if app.book.plans.get("SP-PLAN-0007", {}).get("source") != "signed" or signed["account"] != account:
        failures.append(f"a signed Plan Order that fits is not recorded: {signed}")
    pay_plan(app, fake, buyer, "SP-PLAN-0007", "10800.00")
    if [app.book.market.tranches[(account, month)] for month in ("2027-01", "2027-02", "2027-03")] != [248, 224, 248]:
        failures.append(f"an upfront plan paid in full does not issue every Tranche: {[app.book.market.tranches[(account, month)] for month in ('2027-01', '2027-02', '2027-03')]}")
    app.market.credit(account, app.market.on_sale(app.market.now(), cfg.pricing()), "cover test")
    try:
        app.record_signed(cfg.pricing(), plan("0008", "ops@acme.example", ["americas"], months=3, start="2027-01-01", payment="upfront"), NOW[0])
        failures.append("a Plan Order takes free fleet-hours that hour-credits already sold need")
    except sp.Refusal as error:
        if "Hour-credits already sold" not in str(error):
            failures.append(f"a Plan Order is refused for the wrong reason when sold HC needs its hours: {error}")
    if "Phase A ended" not in (state[1]["ended"] or ""):
        failures.append(f"an unpaid plan survives Phase A: {state[1]}")
    if "not paid in full by its Start Date" not in (state[2]["ended"] or ""):
        failures.append(f"an unpaid upfront plan survives its Start Date: {state[2]}")


def lead_flow(failures, app, buyer, mailbox, work):
    status, _ = buyer("POST", "/lead", {"company": "Big Co", "email": "cto@big.example", "fleets": 2, "start": "2027-01", "message": "Two fleets, East window.\nCall me."})
    app.mails.join()
    saved = [json.loads(line) for line in open(work / "data" / "leads.jsonl", encoding="utf-8")]
    if status != 200 or saved[-1]["company"] != "Big Co" or saved[-1]["fleets"] != "2":
        failures.append(f"quote request: {status} {saved}")
    if not any("Big Co" in m["Subject"] for _, m in mailbox.mail):
        failures.append("we are not told of a quote request")
    opened = next(m for _, m in mailbox.mail if m["Subject"].startswith("New SparkPipe account"))
    if "Acme 주식회사 <ops@acme.example>" not in opened.get_content() or opened["Content-Transfer-Encoding"] != "quoted-printable":
        failures.append(f"a Korean company name is garbled in mail: {opened.get_content()[:80]!r}")
    if buyer("POST", "/lead", {"company": "", "email": "cto@big.example"})[0] != 400:
        failures.append("a quote request without a company is accepted")


def site_copy(work):
    (work / "site" / "terms").mkdir(parents=True)
    shutil.copy(ROOT / "site" / "pricing.js", work / "site" / "pricing.js")
    for name in sp.TERMS.values():
        shutil.copy(ROOT / "site" / "terms" / name, work / "site" / "terms" / name)
    for name in ("agreement.html", "agreement-ko.html"):
        shutil.copy(ROOT / "site" / name, work / "site" / name)
    (work / "secret").write_text("test-client-secret\n")


def toss_env(work, fake, mailbox, keys, name="data", sandbox=False):
    env = {
        "SPARKPIPE_PAY_DATA": str(work / name), "SPARKPIPE_PAY_SITE": str(work / "site"),
        "COINPAYMENTS_API": "http://127.0.0.1:9/api/v2", "COINPAYMENTS_CLIENT_ID": CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(work / "secret"),
        "SPARKPIPE_PAY_SMTP": f"127.0.0.1:{mailbox.server_address[1]}", "TOSS_API": f"http://127.0.0.1:{fake.server_address[1]}",
        "SPARKPIPE_PAY_SUPPLIER_ADDRESS_KO": "서울특별시 용산구 시험로 1", "SPARKPIPE_PAY_SUPPLIER_ADDRESS_EN": "1 Test-ro, Yongsan-gu, Seoul",
    }
    if keys:
        env.update(TOSS_CLIENT_KEY=keys[0], TOSS_SECRET_KEY=keys[1])
    if sandbox:
        env.update(SPARKPIPE_PAY_SANDBOX="1", SPARKPIPE_PAY_PREFIX="/pay/sandbox/api")
    return env


def redirected(port, path):
    class Stop(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    try:
        urllib.request.build_opener(Stop).open(f"http://127.0.0.1:{port}{path}", timeout=20)
        return None
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Location")


def succeed(port, order, key, amount):
    return redirected(port, f"/toss/success/{order}?" + urllib.parse.urlencode({"paymentKey": key, "orderId": order, "amount": amount, "paymentType": "NORMAL"}))


def toss_webhook(client, body):
    return client("POST", "/toss/webhook", raw=json.dumps(body, ensure_ascii=False).encode(), headers={"tosspayments-webhook-transmission-time": "2026-11-02T12:00:00+09:00"})


def lines_of(cfg, kind):
    return [json.loads(line)["data"] for line in open(cfg.path("ledger.jsonl"), encoding="utf-8") if json.loads(line)["type"] == kind]


def check_toss_keys(failures, work, fake, mailbox):
    for keys, sandbox in ((TOSS_TEST, False), (TOSS_LIVE, True), ((TOSS_LIVE[0], ""), False), (("", TOSS_TEST[1]), True)):
        try:
            sp.Config(toss_env(work, fake, mailbox, keys, sandbox=sandbox))
            failures.append(f"Toss Payments keys {'live' if keys[0].startswith('live') else 'test'} are accepted {'in the sandbox' if sandbox else 'on the live service'}, or one key without the other")
        except ValueError as error:
            if "test_" in str(error) or "live_" in str(error) or "fake" in str(error):
                failures.append(f"the key error names a key: {error}")
    cfg = sp.Config(toss_env(work, fake, mailbox, None, "closed"))
    server = sp.start(cfg, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    buyer = Client(server.server_address[1], origin="https://sparkpipe.ai")
    try:
        if buyer("GET", "/toss/config")[1] != {"enabled": False, "clientKey": None, "variantKey": "DEFAULT", "test": False}:
            failures.append(f"the live service offers cards without live Toss Payments keys: {buyer('GET', '/toss/config')[1]}")
        sign_in(buyer, mailbox, "nokeys@acme.example", company="No Keys", accept=True)
        status, refused = buyer("POST", "/order", {"product": "credits", "amount": "130000", "processor": "toss"})
        if status != 503 or refused.get("reason") != "cards_closed" or server.app.book.orders:
            failures.append(f"a card order is taken without live Toss Payments keys: {status} {refused}")
    finally:
        server.shutdown()


def check_toss_sandbox(failures, work, fake, mailbox):
    cfg = sp.Config(toss_env(work, fake, mailbox, TOSS_TEST, "sandbox", sandbox=True))
    server = sp.start(cfg, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, app = server.server_address[1], server.app
    tester = Client(port, origin="https://sparkpipe.ai")
    try:
        if tester("GET", "/toss/config")[1] != {"enabled": True, "clientKey": TOSS_TEST[0], "variantKey": "DEFAULT", "test": True}:
            failures.append(f"the sandbox does not run Toss Payments in test mode: {tester('GET', '/toss/config')[1]}")
        code = tester("POST", "/code", {"email": "try@acme.example"})[1].get("code")
        tester("POST", "/verify", {"email": "try@acme.example", "code": code, "company": "Try Co", "accept": True})
        status, placed = tester("POST", "/market/buy", {"amount": "26000", "processor": "toss"})
        request = placed.get("toss", {})
        if status != 200 or request.get("clientKey") != TOSS_TEST[0] or not request.get("test") or request.get("amount") != 28600 or not request.get("successUrl", "").startswith("https://sparkpipe.ai/pay/sandbox/api/toss/success/"):
            failures.append(f"a sandbox card purchase is not a Toss Payments test payment: {status} {placed}")
            return
        toss_payment(fake, "sandbox-key-1", request["orderId"], 28600)
        hop = succeed(port, request["orderId"], "sandbox-key-1", "28600")
        state = tester("GET", "/market/state")[1]["account"]
        if hop != (303, f"https://sparkpipe.ai/account.html?sandbox&order={request['orderId']}") or Decimal(state["balance"]) != 1:
            failures.append(f"a sandbox test card payment does not issue its HC: {hop} {state['balance']}")
        if Decimal(tester("POST", "/market/buy", {"amount": "40"})[1].get("hc", "0")) != 2:
            failures.append("the sandbox's free HC purchase stops working beside the test card payments")
        if tester("POST", "/order", {"product": "plan", "plan": "PO", "amount": "1000", "processor": "toss"})[0] != 404:
            failures.append("the sandbox takes a plan payment")
    finally:
        server.shutdown()


def check_toss(failures):
    work = Path(tempfile.mkdtemp(prefix="sparkpipe-toss-"))
    fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeToss)
    fake.payments, fake.calls, fake.done, fake.secret = {}, [], {}, TOSS_LIVE[1]
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    mailbox = socketserver.ThreadingTCPServer(("127.0.0.1", 0), FakeSMTP)
    mailbox.daemon_threads, mailbox.mail = True, []
    threading.Thread(target=mailbox.serve_forever, daemon=True).start()
    start = NOW[0]
    NOW[0] = datetime(2026, 11, 2, 3, 0, 0, tzinfo=timezone.utc)
    try:
        site_copy(work)
        sp.clock = lambda: NOW[0]
        sp.LIMITS["order"] = (100, 600)
        check_toss_keys(failures, work, fake, mailbox)
        cfg = sp.Config(toss_env(work, fake, mailbox, TOSS_LIVE))
        server = sp.start(cfg, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port, app = server.server_address[1], server.app
        buyer = Client(port, origin="https://sparkpipe.ai")
        sign_in(buyer, mailbox, "card@acme.example", company="Card Co", accept=True)
        try:
            toss_flows(failures, cfg, app, fake, buyer, mailbox, port)
        finally:
            server.shutdown()
        fake.secret = TOSS_TEST[1]
        check_toss_sandbox(failures, work, fake, mailbox)
    finally:
        NOW[0] = start
        fake.shutdown()
        mailbox.shutdown()
        shutil.rmtree(work)


def order_toss(client, body):
    status, placed = client("POST", "/order", dict(body, processor="toss"))
    return status, placed, placed.get("toss", {})


def toss_flows(failures, cfg, app, fake, buyer, mailbox, port):
    if buyer("GET", "/toss/config")[1] != {"enabled": True, "clientKey": TOSS_LIVE[0], "variantKey": "DEFAULT", "test": False}:
        failures.append(f"the live service hides cards with live Toss Payments keys set: {buyer('GET', '/toss/config')[1]}")
    for amount in ("12999", "130000001", "130000.5", "ten", "", True):
        if buyer("POST", "/order", {"product": "credits", "amount": amount, "processor": "toss"})[0] != 400:
            failures.append(f"a card order of {amount!r} KRW is accepted")
    if buyer("POST", "/order", {"product": "credits", "amount": "130000", "processor": "toss", "brn": "12345"})[0] != 400:
        failures.append("a card order with a malformed business registration number is accepted")
    if buyer("POST", "/order", {"product": "credits", "amount": "130000", "processor": "paypal"})[0] != 400:
        failures.append("an order through an unknown processor is accepted")
    status, placed, request = order_toss(buyer, {"product": "credits", "amount": "130000", "total": "1"})
    order = placed.get("order", "")
    expected = {"amount": 143000, "currency": "KRW", "orderId": order, "customerEmail": "card@acme.example", "customerName": "Card Co", "clientKey": TOSS_LIVE[0],
                "successUrl": f"https://sparkpipe.ai/pay/api/toss/success/{order}", "failUrl": f"https://sparkpipe.ai/pay/api/toss/fail/{order}", "orderName": "SparkPipe hour-credits, 5 HC"}
    if status != 200 or any(request.get(key) != value for key, value in expected.items()):
        failures.append(f"a card order does not fix its won amount on the server: {status} {placed}")
        return
    shown = buyer("GET", "/order/" + order)[1]
    if (shown["state"], shown["total"], shown["vat"], shown["hours"], shown["currency"], shown["processor"]) != ("created", "143000", "13000", "5.000000", "KRW", "toss"):
        failures.append(f"a card order's status: {shown}")
    toss_payment(fake, "pay-card-1", order, 143000)
    hop = succeed(port, order, "pay-card-1", "143000")
    app.mails.join()
    confirms = [c for c in fake.calls if c[0] == "POST" and c[1] == "pay-card-1"]
    account = app.book.emails["card@acme.example"]
    if hop != (303, f"https://sparkpipe.ai/account.html?order={order}") or confirms != [("POST", "pay-card-1", {"paymentKey": "pay-card-1", "orderId": order, "amount": 143000}, f"confirm-{order}")]:
        failures.append(f"the success redirect does not confirm the server's amount once with an idempotency key: {hop} {confirms}")
    if buyer("GET", "/account")[1]["hc"] != "5.000000" or ledger_types(cfg).count("hours_issued") != 1:
        failures.append(f"a confirmed card payment does not issue its HC: {buyer('GET', '/account')[1]['hc']}")
    payment = lines_of(cfg, "payment")[-1]
    if payment.get("processor") != "toss" or payment.get("method") != "카드" or payment.get("taxInvoice", {}).get("needed") is not False or payment.get("received") != "143000":
        failures.append(f"the card payment line: {payment}")
    note = next((m.get_content() for rcpt, m in mailbox.mail if "card@acme.example" in rcpt and order in m["Subject"]), "")
    if "143,000 KRW" not in note or "by card through Toss Payments" not in note or "card slip" not in note or "receipt/pay-card-1" not in note:
        failures.append(f"the card receipt: {note!r}")
    succeed(port, order, "pay-card-1", "143000")
    toss_webhook(buyer, {"eventType": "PAYMENT_STATUS_CHANGED", "createdAt": "2026-11-02T12:00:00.000000", "data": {"orderId": order, "paymentKey": "pay-card-1", "status": "DONE"}})
    app.jobs.join()
    app.reconcile()
    app.jobs.join()
    if buyer("GET", "/account")[1]["hc"] != "5.000000" or ledger_types(cfg).count("hours_issued") != 1 or len([c for c in fake.calls if c[0] == "POST" and c[1] == "pay-card-1"]) != 1:
        failures.append("a repeated success redirect or webhook issues HC twice or confirms twice")

    status, placed, request = order_toss(buyer, {"product": "credits", "amount": "130000"})
    toss_payment(fake, "pay-cheap", request["orderId"], 100)
    hop = succeed(port, request["orderId"], "pay-cheap", "100")
    shown = buyer("GET", "/order/" + request["orderId"])[1]
    if any(c[1] == "pay-cheap" and c[0] == "POST" for c in fake.calls) or shown["state"] != "failed" or shown["reason"] != "amount" or buyer("GET", "/account")[1]["hc"] != "5.000000":
        failures.append(f"a payment authorised for another amount is confirmed: {shown}")
    status, placed, request = order_toss(buyer, {"product": "credits", "amount": "130000"})
    toss_payment(fake, "pay-odd", request["orderId"], 143000, total=1000)
    succeed(port, request["orderId"], "pay-odd", "143000")
    shown = buyer("GET", "/order/" + request["orderId"])[1]
    cancels = [c for c in fake.calls if c[1] == "pay-odd" and c[0] == "POST" and c[3] == f"mismatch-{request['orderId']}"]
    if shown["state"] != "cancelled" or shown["reason"] != "amount" or len(cancels) != 1 or fake.payments["pay-odd"]["status"] != "CANCELED" or buyer("GET", "/account")[1]["hc"] != "5.000000":
        failures.append(f"a payment Toss Payments reports for another amount is not refused and cancelled: {shown} {cancels}")
    status, placed, request = order_toss(buyer, {"product": "credits", "amount": "130000"})
    toss_payment(fake, "pay-declined", request["orderId"], 143000, _decline="REJECT_CARD_COMPANY")
    succeed(port, request["orderId"], "pay-declined", "143000")
    shown = buyer("GET", "/order/" + request["orderId"])[1]
    if shown["state"] != "failed" or shown["reason"] != "REJECT_CARD_COMPANY":
        failures.append(f"a declined card is not recorded: {shown}")
    status, placed, request = order_toss(buyer, {"product": "credits", "amount": "26000"})
    hop = redirected(port, f"/toss/fail/{request['orderId']}?code=PAY_PROCESS_CANCELED&message=x")
    if hop != (303, f"https://sparkpipe.ai/account.html?order={request['orderId']}") or buyer("GET", "/order/" + request["orderId"])[1]["state"] != "cancelled":
        failures.append(f"a payment the buyer cancels stays open: {hop}")

    status, placed, request = order_toss(buyer, {"product": "credits", "amount": "260000", "brn": "123-45-67890"})
    deposit = request["orderId"]
    toss_payment(fake, "pay-va-1", deposit, 286000, method="가상계좌", after="WAITING_FOR_DEPOSIT", secret="va-secret")
    succeed(port, deposit, "pay-va-1", "286000")
    app.mails.join()
    shown = buyer("GET", "/order/" + deposit)[1]
    if shown["state"] != "waiting" or shown["deposit"] != {"bank": "신한은행", "accountNumber": "X12345678901", "dueDate": "2026-11-03T12:00:00+09:00"} or shown["expires"] != "2026-11-03T03:00:00Z" or buyer("GET", "/account")[1]["hc"] != "5.000000":
        failures.append(f"an issued virtual account is not shown awaiting its deposit, or issues HC before it: {shown}")
    if not any("신한은행 account X12345678901" in m.get_content() for rcpt, m in mailbox.mail if "card@acme.example" in rcpt and deposit in m["Subject"]):
        failures.append("the buyer is not emailed the virtual account to pay")
    for forged in ({"createdAt": "2026-11-02T12:00:00.000000", "secret": "va-secret", "status": "DONE", "transactionKey": "t", "orderId": deposit},
                   {"eventType": "PAYMENT_STATUS_CHANGED", "data": {"orderId": deposit, "paymentKey": "pay-va-1", "status": "DONE", "totalAmount": 286000, "currency": "KRW", "method": "가상계좌"}}):
        if toss_webhook(buyer, forged)[0] != 200:
            failures.append("a Toss Payments webhook is not acknowledged")
        app.jobs.join()
    if buyer("GET", "/account")[1]["hc"] != "5.000000" or buyer("GET", "/order/" + deposit)[1]["state"] != "waiting":
        failures.append("a webhook body that says DONE issues HC while Toss Payments still waits for the deposit")
    fake.payments["pay-va-1"]["status"] = "DONE"
    toss_webhook(buyer, {"createdAt": "2026-11-02T13:00:00.000000", "secret": "va-secret", "status": "DONE", "transactionKey": "t", "orderId": deposit})
    app.jobs.join()
    toss_webhook(buyer, {"createdAt": "2026-11-02T13:00:00.000000", "secret": "va-secret", "status": "DONE", "transactionKey": "t", "orderId": deposit})
    app.jobs.join()
    app.mails.join()
    if buyer("GET", "/account")[1]["hc"] != "15.000000" or ledger_types(cfg).count("hours_issued") != 2:
        failures.append(f"a virtual account deposit verified with Toss Payments does not issue its HC exactly once: {buyer('GET', '/account')[1]['hc']}")
    invoice = lines_of(cfg, "payment")[-1].get("taxInvoice")
    if invoice != {"needed": True, "brn": "123-45-67890", "supply": "260000", "vat": "26000"} or [o["order"] for o in app.tax_invoices()] != [deposit]:
        failures.append(f"a virtual account payment is not recorded as needing an electronic tax invoice: {invoice}")
    note = next((m.get_content() for rcpt, m in mailbox.mail if "card@acme.example" in rcpt and m["Subject"] == f"Payment received: SparkPipe order {deposit}"), "")
    if "electronic tax invoice" not in note or "123-45-67890" not in note:
        failures.append(f"the virtual account receipt does not promise the tax invoice: {note!r}")

    status, placed, request = order_toss(buyer, {"product": "credits", "amount": "130000"})
    late = request["orderId"]
    toss_payment(fake, "pay-va-2", late, 143000, method="가상계좌", after="WAITING_FOR_DEPOSIT")
    succeed(port, late, "pay-va-2", "143000")
    status, placed, request = order_toss(buyer, {"product": "credits", "amount": "130000"})
    abandoned = request["orderId"]
    held = app.market.pending()
    NOW[0] = datetime(2026, 11, 3, 4, 0, 0, tzinfo=timezone.utc)
    if held != 10 or app.market.pending() != 0:
        failures.append(f"open card orders do not hold their HC off sale until they expire, and no longer: {held} {app.market.pending()}")
    app.reconcile()
    app.jobs.join()
    expired = [c for c in fake.calls if c[1] == "pay-va-2" and c[0] == "POST" and c[3] == f"expire-{late}"]
    states = [buyer("GET", "/order/" + ident)[1]["state"] for ident in (late, abandoned)]
    if states != ["timedout", "timedout"] or len(expired) != 1 or fake.payments["pay-va-2"]["status"] != "CANCELED" or buyer("GET", "/account")[1]["hc"] != "15.000000":
        failures.append(f"an unpaid virtual account or an abandoned card order does not expire with its order: {states} {expired}")
    toss_flows_plans(failures, cfg, app, fake, buyer, mailbox, port)
    toss_refunds(failures, cfg, app, fake, buyer, mailbox, order, deposit)


def toss_flows_plans(failures, cfg, app, fake, buyer, mailbox, port):
    status, quote = buyer("POST", "/plans/quote", CHOICE)
    rows = dict(map(tuple, quote.get("rows", {}).get("en", [])))
    ko = dict(map(tuple, quote.get("rows", {}).get("ko", [])))
    if status != 200 or quote.get("priceKRW") != "20000" or rows.get("Hour Price (clause 16.4)") != "KRW 20,000 or USD 15.00 per fleet-hour, in the payment currency below, exclusive of VAT, including any Fleet lock, fixed for the Term" \
            or rows.get("Price of the Term (clause 16.4)") != "KRW 116,960,000 or USD 87,720.00, exclusive of VAT" or rows.get("Payment currency and account (clause 3.3)") != "KRW, clause 3.3(a): Toss Payments / US dollar stablecoin, clause 3.3(c)" \
            or ko.get("시간당 가격(제16.4조)") != "아래 지급 통화에 따라 플릿시간당 20,000원(KRW) 또는 미화 15.00달러(USD), 부가가치세 별도, 플릿 고정 포함, 플랜 기간 동안 고정":
        failures.append(f"a Plan Order does not state its Hour Price in won for payment through Toss Payments: {status} {quote.get('priceKRW')} {rows} {ko}")
    holder = app.book.emails["card@acme.example"]
    record(app, dict(plan("TOSS", "card@acme.example", ["asia"]), hourPriceKRW="20000"))
    record(app, plan("USD", "card@acme.example", ["europe"]))
    found = {p["plan"]: p for p in buyer("GET", "/account")[1]["plans"]}
    toss_plan = found.get("SP-PLAN-TOSS", {})
    if toss_plan.get("priceKRW") != "20000" or toss_plan.get("nextGateKRW") != "26880000" or toss_plan.get("remainingKRW") != "116960000" or found.get("SP-PLAN-USD", {}).get("remainingKRW") is not None:
        failures.append(f"a plan does not show what is due in won: {toss_plan}")
    if buyer("POST", "/order", {"product": "plan", "plan": "SP-PLAN-USD", "amount": "2720000", "processor": "toss"})[0] != 409:
        failures.append("a plan whose Plan Order has no won Hour Price is paid in won")
    if buyer("POST", "/order", {"product": "plan", "plan": "SP-PLAN-TOSS", "amount": "116960001", "processor": "toss"})[0] != 409:
        failures.append("a won prepayment above what is left of the plan is ordered")
    status, placed, request = order_toss(buyer, {"product": "plan", "plan": "SP-PLAN-TOSS", "amount": "2720000"})
    if status != 200 or request.get("amount") != 2992000 or request.get("orderName") != "SparkPipe plan SP-PLAN-TOSS, prepayment":
        failures.append(f"a plan payment through Toss Payments: {status} {placed}")
        return
    toss_payment(fake, "pay-plan-1", request["orderId"], 2992000, method="계좌이체")
    succeed(port, request["orderId"], "pay-plan-1", "2992000")
    app.mails.join()
    after = {p["plan"]: p for p in buyer("GET", "/account")[1]["plans"]}["SP-PLAN-TOSS"]
    if app.book.market.tranches[(holder, "2026-12")] != 136 or after["paid"] != "2040.00" or after["remainingKRW"] != "114240000" or after["nextGateKRW"] != "24160000":
        failures.append(f"a plan payment through Toss Payments does not issue its Tranche: {app.book.market.tranches[(holder, '2026-12')]} {after}")
    if lines_of(cfg, "payment")[-1]["taxInvoice"] != {"needed": True, "brn": None, "supply": "2720000", "vat": "272000"}:
        failures.append(f"a won bank transfer for a plan is not recorded as needing a tax invoice: {lines_of(cfg, 'payment')[-1]}")


def run_main(cfg, args, fake, mailbox):
    saved = dict(os.environ)
    os.environ.update({"SPARKPIPE_PAY_DATA": cfg.data, "SPARKPIPE_PAY_SITE": cfg.site, "COINPAYMENTS_CLIENT_ID": CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(Path(cfg.site).parent / "secret"),
                       "SPARKPIPE_PAY_SMTP": cfg.smtp, "TOSS_API": cfg.toss_api, "TOSS_CLIENT_KEY": cfg.toss_client, "TOSS_SECRET_KEY": cfg.toss_secret})
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return sp.main(["sparkpipe_pay.py"] + args)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def toss_refunds(failures, cfg, app, fake, buyer, mailbox, card, deposit):
    account = app.book.emails["card@acme.example"]
    if run_main(cfg, ["refund", card, "9.9", "71500", "goodwill"], fake, mailbox) != 2 or run_main(cfg, ["refund", card, "10.4", "143001", "too much"], fake, mailbox) != 2:
        failures.append("a refund outside the Customer Agreement's clauses, or above the payment, is made")
    if run_main(cfg, ["refund", deposit, "10.4", "all", "service ended"], fake, mailbox) != 2:
        failures.append("a virtual account payment is refunded without the account to refund it to")
    before = len(fake.calls)
    if run_main(cfg, ["refund", card, "10.4", "71500", "we stopped offering HC"], fake, mailbox) != 0:
        failures.append("the refund command fails")
        return
    cancel = [c for c in fake.calls[before:] if c[0] == "POST"]
    with app.book.lock, open(app.book.path, "rb") as handle:
        app.book.catch_up(handle)
    refund = lines_of(cfg, "refund")
    if cancel != [("POST", "pay-card-1", {"cancelReason": "Customer Agreement clause 10.4: we stopped offering HC", "refundableAmount": 143000, "cancelAmount": 71500}, f"refund-{card}-143000")] \
            or refund != [dict(refund[0], order=card, amount="71500", balance="71500", clause="10.4", hc="2.500000")] or buyer("GET", "/account")[1]["hc"] != "12.500000":
        failures.append(f"the refund command does not refund through the cancel API and debit the order's HC: {cancel} {refund} {buyer('GET', '/account')[1]['hc']}")
    if not any(m["Subject"] == f"Refund: SparkPipe order {card}" for rcpt, m in mailbox.mail if "card@acme.example" in rcpt):
        failures.append("the customer is not told of a refund")
    app.market.reserve(account, {"fleet": "F1", "start": "2026-12-02T01:00Z", "model": "kimi-k3", "max": "2"})
    if run_main(cfg, ["refund", card, "10.4", "all", "rest"], fake, mailbox) != 2:
        failures.append("a refund pays back HC that was already spent")
    if run_main(cfg, ["refund", deposit, "3.2", "100", "VAT not owed", "088", "110123456789", "Card Co"], fake, mailbox) != 0 or fake.calls[-1][2].get("refundReceiveAccount") != {"bank": "088", "accountNumber": "110123456789", "holderName": "Card Co"}:
        failures.append("a virtual account payment is not refunded to the account given")
    if run_main(cfg, ["tax-invoice", deposit, "20261102-41000000-00000001"], fake, mailbox) != 0:
        failures.append("an issued tax invoice cannot be recorded")
    with app.book.lock, open(app.book.path, "rb") as handle:
        app.book.catch_up(handle)
    if deposit in [o["order"] for o in app.tax_invoices()]:
        failures.append("a recorded tax invoice is still listed")


def main():
    failures = []
    check_dates(failures)
    check_sessions(failures)
    check_covered(failures)
    check_service(failures)
    check_toss(failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("pay: plan hours and gates match the Customer Agreement for every window, day choice, term and fleet count, invalid plans fail; email codes, sessions, terms acceptance, USDC and USDT invoices on the chosen network, payments in another stablecoin, signed webhooks, receipts, HC orders and part payments, reconciliation, gates, self-serve Plan Orders (the bilingual Order Form, the accepted hash and our Ed25519 signature, capacity, HC cover and fleet-lock checks, holds that lapse unpaid or end at the last paid Tranche, Tranche hours issued on payment less hours that already occurred), quote requests and the hash-chained ledger behave; Toss Payments: live keys only on the live service and test keys only in the sandbox, no card option without them, won amounts fixed on the server and confirmed once, other amounts refused and cancelled, webhooks only trigger a check with Toss Payments, virtual accounts issued on a verified deposit and expiring with their order, plan payments in won issuing Tranches, refunds under the named clause through the cancel API, and the tax invoice list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
