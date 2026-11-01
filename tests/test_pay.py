#!/usr/bin/env python3

import base64
import contextlib
import email
import email.policy
import hashlib
import hmac
import http.server
import io
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
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

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


def check_units(failures):
    credits = {"hc": "168", "price": "20.00", "net": "3360.00", "total": "3696.00"}
    if sp.covered(credits, sp.money("22.99")) != ("1.045000", "20.90") or sp.covered(credits, sp.money("0.01")) != ("0.000000", "0.00") or sp.covered(credits, sp.money("9999.00")) != ("168.000000", "3360.00"):
        failures.append(f"HC covered by a part payment: {sp.covered(credits, sp.money('22.99'))}")
    key = b"k" * 32
    token = sp.session_token(key, "ACX", 2000)
    if sp.session_account(key, token, 1999) != "ACX" or sp.session_account(key, token, 2001) is not None:
        failures.append("a fresh session token is refused or an expired one accepted")
    if sp.session_account(b"j" * 32, token, 1999) is not None or sp.session_account(key, token.replace("ACX", "ACY"), 1999) is not None:
        failures.append("a forged session token is accepted")
    for bad in (None, "", "a.b", "ACX.x.y", "ACX.2000.00"):
        if sp.session_account(key, bad, 1999) is not None:
            failures.append(f"a malformed session token {bad!r} is accepted")
    if set(sp.TERMS.values()) != {"agreement-ko.md", "agreement-en.md"} or "Market Rules" in sp.statement(type("C", (), {"public": "https://sparkpipe.ai"}), "Acme", {"ko": "k", "en": "e"}):
        failures.append("the accepted terms are not just the Korean and English Customer Agreement")


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
        return {name: value for name, value in self.server.payments[key].items() if not name.startswith("_")}

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
        key = body.get("paymentKey") if self.path == "/v1/payments/confirm" else urllib.parse.unquote(self.path.split("/")[3])
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


def serve(handler, **fields):
    server = (socketserver.ThreadingTCPServer if handler is FakeSMTP else http.server.ThreadingHTTPServer)(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    for name, value in fields.items():
        setattr(server, name, value)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


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
        try:
            return status, json.loads(payload or b"{}")
        except ValueError:
            return status, payload.decode()


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
    headers = {"X-CoinPayments-Client": CLIENT, "X-CoinPayments-Timestamp": stamp, "X-CoinPayments-Signature": docs_signature("POST", cfg.webhook_url, stamp, raw)}
    return client("POST", "/coinpayments", headers=headers, raw=raw + (b" " if tamper else b""))


def lines_of(cfg, kind=None):
    found = [json.loads(line) for line in open(cfg.path("ledger.jsonl"), encoding="utf-8")]
    return [line["type"] for line in found] if kind is None else [line["data"] for line in found if line["type"] == kind]


def site_copy(work):
    (work / "site" / "terms").mkdir(parents=True)
    shutil.copy(ROOT / "site" / "pricing.js", work / "site" / "pricing.js")
    for name in sp.TERMS.values():
        shutil.copy(ROOT / "site" / "terms" / name, work / "site" / "terms" / name)
    (work / "secret").write_text("test-client-secret\n")


def chain(rows):
    lines, head = [], "0" * 64
    for seq, (kind, data) in enumerate(rows, 1):
        body = {"seq": seq, "at": f"2026-10-2{seq % 10}T01:00:00Z", "type": kind, "data": data, "prev": head}
        head = sp.digest(body)
        lines.append(json.dumps(dict(body, hash=head), ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n"


LEGACY = [
    ("account", {"account": "ACLIVE0000000000000", "email": "first@acme.example", "company": "First Co", "ip": "1.2.3.4", "agent": "test"}),
    ("acceptance", {"account": "ACLIVE0000000000000", "ko": "a" * 64, "en": "b" * 64, "rules_ko": "c" * 64, "rules_en": "d" * 64, "statement": "old", "ip": "1.2.3.4", "agent": "test"}),
    ("order", {"order": "SPLIVEAAAAAAAAAAAAAA", "account": "ACLIVE0000000000000", "email": "first@acme.example", "company": "First Co", "product": "hourly", "hours": 4, "start": "2026-12-01", "price": "15.00", "net": "60.00", "vat": "6.00", "total": "66.00", "coin": "USDC.BASE"}),
    ("invoice", {"order": "SPLIVEAAAAAAAAAAAAAA", "invoice": "CPLIVE1", "link": "https://pay.example/c/CPLIVE1"}),
    ("invoice_state", {"order": "SPLIVEAAAAAAAAAAAAAA", "invoice": "CPLIVE1", "state": "pending"}),
    ("invoice_state", {"order": "SPLIVEAAAAAAAAAAAAAA", "invoice": "CPLIVE1", "state": "pending", "received": "1.00"}),
    ("invoice_state", {"order": "SPLIVEAAAAAAAAAAAAAA", "invoice": "CPLIVE1", "state": "unpaid"}),
    ("invoice_state", {"order": "SPLIVEAAAAAAAAAAAAAA", "invoice": "CPLIVE1", "state": "timedout"}),
    ("invoice_state", {"order": "SPLIVEAAAAAAAAAAAAAA", "invoice": "CPLIVE1", "state": "cancelled"}),
]


def run_main(cfg, args, capture=True):
    saved = dict(os.environ)
    os.environ.update({"SPARKPIPE_PAY_DATA": cfg.data, "SPARKPIPE_PAY_SITE": cfg.site, "COINPAYMENTS_CLIENT_ID": CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(Path(cfg.site).parent / "secret"),
                       "SPARKPIPE_PAY_SMTP": cfg.smtp, "TOSS_API": cfg.toss_api, "TOSS_CLIENT_KEY": cfg.toss_client, "TOSS_SECRET_KEY": cfg.toss_secret})
    try:
        if not capture:
            return sp.main(["sparkpipe_pay.py"] + args)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return sp.main(["sparkpipe_pay.py"] + args)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def started(cfg):
    server = sp.start(cfg, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, Client(server.server_address[1], origin="https://sparkpipe.ai")


def check_legacy(failures):
    work = Path(tempfile.mkdtemp(prefix="sparkpipe-legacy-"))
    mailbox = serve(FakeSMTP, mail=[])
    try:
        site_copy(work)
        (work / "data").mkdir()
        (work / "data" / "ledger.jsonl").write_text(chain(LEGACY))
        cfg = sp.Config({"SPARKPIPE_PAY_DATA": str(work / "data"), "SPARKPIPE_PAY_SITE": str(work / "site"), "COINPAYMENTS_CLIENT_ID": CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(work / "secret"),
                         "SPARKPIPE_PAY_SMTP": f"127.0.0.1:{mailbox.server_address[1]}"})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run_main(cfg, ["verify"], capture=False)
        if code != 0 or not out.getvalue().startswith("9 lines, head ") or "1 accounts, 1 orders" not in out.getvalue():
            failures.append(f"the live ledger of 9 lines does not replay: {out.getvalue()!r}")
        server, client = started(cfg)
        try:
            status, summary = sign_in(client, mailbox, "first@acme.example")
            order = (summary.get("orders") or [{}])[0]
            if status != 200 or summary.get("termsCurrent") is not False or order.get("state") != "cancelled" or order.get("product") != "hourly" or summary.get("hc") != "0.000000":
                failures.append(f"an account from the live ledger signs in, sees its cancelled hourly order and is asked to accept the new Agreement: {status} {summary}")
            status, state = client("GET", "/market/state")
            if status != 200 or state["account"]["termsCurrent"] is not False:
                failures.append(f"the market state of a legacy account: {status} {state}")
            if sign_in(client, mailbox, "first@acme.example", accept=True)[1].get("termsCurrent") is not True or sp.Book(cfg.path("ledger.jsonl")).count != 10:
                failures.append("a legacy account cannot accept the new Agreement on top of the old ledger")
        finally:
            server.shutdown()
    finally:
        mailbox.shutdown()
        shutil.rmtree(work)


def check_service(failures):
    work = Path(tempfile.mkdtemp(prefix="sparkpipe-pay-"))
    fake = serve(FakeCoinPayments, invoices={}, down=False)
    mailbox = serve(FakeSMTP, mail=[])
    try:
        site_copy(work)
        cfg = sp.Config({
            "SPARKPIPE_PAY_DATA": str(work / "data"), "SPARKPIPE_PAY_SITE": str(work / "site"),
            "COINPAYMENTS_API": f"http://127.0.0.1:{fake.server_address[1]}/api/v2",
            "COINPAYMENTS_CLIENT_ID": CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(work / "secret"),
            "SPARKPIPE_PAY_SMTP": f"127.0.0.1:{mailbox.server_address[1]}",
        })
        sp.clock = lambda: NOW[0]
        sp.LIMITS["order"] = (40, 600)
        server, buyer = started(cfg)
        port, app = server.server_address[1], server.app
        account_flow(failures, cfg, app, buyer, mailbox, port)
        credits_flow(failures, cfg, app, fake, buyer, mailbox)
        part_payment_flow(failures, cfg, app, fake, buyer, mailbox)
        for path, body in (("/lead", {"company": "Big Co", "email": "cto@big.example"}), ("/plans/quote", {}), ("/plans/order", {}), ("/plans/stop", {}), ("/market/reserve", {}), ("/market/bid", {}), ("/market/release", {}), ("/market/cancel", {}), ("/market/sandbox/tranche", {})):
            if buyer("POST", path, body)[0] != 404:
                failures.append(f"a removed route still answers: {path}")
        for path in ("/plans/key", "/market/market.txt", "/market/pool.txt", "/market/calendar", "/market/book", "/pool/v1/models"):
            if buyer("GET", path)[0] != 404:
                failures.append(f"a removed route still answers: {path}")
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
        lines[index] = lines[index].replace("Acme", "Acmf", 1)
        (work / "tampered.jsonl").write_text("\n".join(lines) + "\n")
        try:
            sp.Book(str(work / "tampered.jsonl"))
            failures.append("a tampered ledger loads")
        except ValueError:
            pass
        if oct((work / "data" / "session_key").stat().st_mode & 0o777) != "0o600":
            failures.append("the session key is readable by others")
        server.shutdown()
    finally:
        fake.shutdown()
        mailbox.shutdown()
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
        failures.append("an account opens without accepting the Customer Agreement")
    status, summary = sign_in(buyer, mailbox, "Ops@Acme.example", company="Acme 주식회사", accept=True)
    if status != 200 or summary.get("company") != "Acme 주식회사" or not summary.get("termsCurrent") or summary.get("hc") != "0.000000" or summary.get("locked") != "0.000000":
        failures.append(f"opening an account: {status} {summary}")
        return
    if "HttpOnly" not in buyer.last_cookie or "SameSite=Lax" not in buyer.last_cookie or "Path=/pay/api" not in buyer.last_cookie:
        failures.append(f"session cookie attributes: {buyer.last_cookie}")
    record = lines_of(cfg, "acceptance")[-1]
    if (record["ko"], record["en"]) != (cfg.terms("agreement-ko.md"), cfg.terms("agreement-en.md")) or "rules_ko" in record or "Market Rules" in record["statement"] or "Korean text, which prevails" not in record["statement"] or "Acme 주식회사" not in record["statement"]:
        failures.append(f"acceptance record: {record}")
    if buyer("POST", "/verify", {"email": "ops@acme.example", "code": last_code(mailbox, "ops@acme.example")})[0] != 403:
        failures.append("a sign-in code works twice")
    if Client(port, origin="https://evil.example")("POST", "/code", {"email": "ops@acme.example"})[0] != 403:
        failures.append("a cross-site POST is accepted")
    forged = Client(port)
    forged.cookie = buyer.cookie.replace(buyer.cookie.split("=", 1)[1].split(".")[0], "ACAAAAAAAAAAAAAAAA")
    if forged("GET", "/account")[0] != 401:
        failures.append("a forged session reads an account")
    other = Client(port)
    status, _ = sign_in(other, mailbox, "ops@acme.example")
    if status != 200 or other("POST", "/signout", {})[0] != 200 or other.cookie is not None:
        failures.append("signing in again or signing out fails")
    app.mails.join()
    opened = next(m for _, m in mailbox.mail if m["Subject"].startswith("New SparkPipe account"))
    if "Acme 주식회사 <ops@acme.example>" not in opened.get_content() or opened["Content-Transfer-Encoding"] != "quoted-printable":
        failures.append(f"a Korean company name is garbled in mail: {opened.get_content()[:80]!r}")


def credits_flow(failures, cfg, app, fake, buyer, mailbox):
    base = {"product": "credits", "hc": 4, "coin": "USDC.BASE"}
    for coin in ("BTC", "TUSD.TRC20", "base", None):
        if buyer("POST", "/order", dict(base, coin=coin))[0] != 400:
            failures.append(f"an order in {coin} is accepted")
    for count in (0, "0.5", -1, "ten", True, "1.0000001", "1234567890", None):
        if buyer("POST", "/order", dict(base, hc=count))[0] != 400:
            failures.append(f"an order for {count!r} HC is accepted")
    status, placed = buyer("POST", "/order", dict(base, hc="2.5"))
    if status != 200 or fake.invoices[max(fake.invoices, key=lambda k: int(k[3:]))]["body"]["amount"]["total"] != "55.00" or buyer("GET", "/order/" + placed["order"])[1]["hc"] != "2.5":
        failures.append(f"an order for a quantity of HC that is not whole is refused or mispriced: {status} {placed}")
    for body in ({"product": "hourly", "hours": 4, "coin": "USDC.BASE"}, {"product": "plan", "plan": "PO", "amount": "100", "coin": "USDC.BASE"}, {"product": "credits", "amount": "80", "coin": "USDC.BASE"}):
        if buyer("POST", "/order", body)[0] != 400:
            failures.append(f"a removed product or an order by amount is accepted: {body}")
    if Client(buyer.port)("POST", "/order", base)[0] != 401:
        failures.append("an order without signing in is accepted")
    ladder = {}
    for count in (143, 144):
        status, placed = buyer("POST", "/order", dict(base, hc=count))
        invoice = fake.invoices[max(fake.invoices, key=lambda k: int(k[3:]))]["body"]
        ladder[count] = (status, invoice["amount"], invoice["items"][0]["description"])
    if ladder[143] != (200, {"breakdown": {"subtotal": "2860.00", "taxTotal": "286.00"}, "total": "3146.00"}, "Customer Agreement; 20.00 USD each, usable for 12 months") \
            or ladder[144] != (200, {"breakdown": {"subtotal": "2160.00", "taxTotal": "216.00"}, "total": "2376.00"}, "Customer Agreement; 15.00 USD each, usable for 12 months"):
        failures.append(f"the US dollar price ladder at 143 and 144 HC: {ladder}")
    status, placed = buyer("POST", "/order", base)
    if status != 200 or not placed.get("checkout", "").startswith("https://pay.example/c/"):
        failures.append(f"HC order: {status} {placed}")
        return
    key = max(fake.invoices, key=lambda k: int(k[3:]))
    invoice = fake.invoices[key]["body"]
    expected = {"currency": "5057", "invoiceId": placed["order"], "amount": {"breakdown": {"subtotal": "80.00", "taxTotal": "8.00"}, "total": "88.00"}}
    if any(invoice.get(name) != value for name, value in expected.items()):
        failures.append(f"invoice body: {invoice}")
    if invoice["payment"] != {"paymentCurrency": sp.COINS["USDC.BASE"], "refundEmail": "ops@acme.example"} or invoice["buyer"] != {"companyName": "Acme 주식회사", "emailAddress": "ops@acme.example"}:
        failures.append(f"invoice payment or buyer: {invoice['payment']} {invoice['buyer']}")
    if invoice["items"][0]["name"] != "4 SparkPipe hour-credits" or invoice["webhooks"][0]["notificationsUrl"] != "https://sparkpipe.ai/pay/api/coinpayments" or invoice["successUrl"] != f"https://sparkpipe.ai/account.html?order={placed['order']}":
        failures.append(f"invoice items, webhooks or return address: {invoice['items']} {invoice['webhooks']} {invoice['successUrl']}")
    status, shown = buyer("GET", "/order/" + placed["order"])
    if status != 200 or shown["state"] != "unpaid" or shown["checkout"] != placed["checkout"] or shown["hc"] != "4" or shown["price"] != "20.00" or "email" in shown:
        failures.append(f"unpaid order status: {status} {shown}")
    if webhook(buyer, cfg, key, tamper=True)[0] != 401 or webhook(buyer, cfg, key, stamp="2026-11-02T02:50:00")[0] != 401:
        failures.append("a tampered or stale webhook is accepted")
    fake.invoices[key]["status"] = "paid"
    webhook(buyer, cfg, key, "InvoicePaid")
    app.jobs.join()
    if buyer("GET", "/order/" + placed["order"])[1]["state"] != "paid":
        failures.append("a paid invoice does not move the order to paid")
    fake.invoices[key]["status"] = "completed"
    if webhook(buyer, cfg, key)[0] != 200:
        failures.append("a signed webhook is refused")
    app.jobs.join()
    webhook(buyer, cfg, key)
    app.jobs.join()
    app.mails.join()
    summary = buyer("GET", "/account")[1]
    lots = buyer("GET", "/market/state")[1]["account"]["lots"]
    if summary["hc"] != "4.000000" or lines_of(cfg).count("hc_issued") != 1 or lines_of(cfg).count("hc_lot") != 1 or lots[0]["expires"] != "2027-11-02T03:00:00Z" or lots[0]["ref"] != placed["order"]:
        failures.append(f"completed order: {summary} {lots} {lines_of(cfg)}")
    receipts = [m for rcpt, m in mailbox.mail if "ops@acme.example" in rcpt and placed["order"] in m["Subject"]]
    if len(receipts) != 1 or "88.00" not in receipts[0].get_content() or "4 hour-credits, usable until 2027-11-02" not in receipts[0].get_content():
        failures.append(f"receipt mail: {[m['Subject'] for _, m in mailbox.mail]}")
    if not any(cfg.notify in rcpt and m["Subject"].startswith("Paid:") for rcpt, m in mailbox.mail):
        failures.append("we are not told of a payment")
    if lines_of(cfg, "payment")[-1]["paidWith"] != ["USDC.BASE"]:
        failures.append("the payment line does not record the coin")
    status, placed = buyer("POST", "/order", dict(base, hc=164, coin="USDT.TRC20"))
    key = max(fake.invoices, key=lambda k: int(k[3:]))
    fake.invoices[key].update(status="completed", total="1.00")
    webhook(buyer, cfg, key)
    app.jobs.join()
    if fake.invoices[key]["body"]["payment"]["paymentCurrency"] != "9:TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t":
        failures.append("a USDT order on Tron does not preset USDT on Tron")
    second = buyer("GET", "/order/" + placed["order"])[1]
    if second["coin"] != "USDT.TRC20" or second["state"] != "mismatch" or second["issued"] or buyer("GET", "/account")[1]["hc"] != "4.000000":
        failures.append(f"an invoice that does not match its order issues HC: {second}")
    fake.down = True
    status, refused = buyer("POST", "/order", dict(base, hc=1))
    fake.down = False
    if status != 502 or buyer("GET", "/account")[1]["orders"][0]["state"] != "failed":
        failures.append(f"a failed invoice is not recorded: {status} {refused}")
    terms = Path(cfg.site) / "terms" / "agreement-en.md"
    terms.write_text(terms.read_text() + "\n")
    status, refused = buyer("POST", "/order", dict(base, hc=1))
    if status != 409 or refused.get("reason") != "terms":
        failures.append(f"an order after the Agreement changed does not ask to accept it again: {status} {refused}")
    if buyer("POST", "/order", dict(base, hc=1, accept=True))[0] != 200 or not buyer("GET", "/account")[1]["termsCurrent"]:
        failures.append("accepting the Agreement again is not recorded")


def part_payment_flow(failures, cfg, app, fake, buyer, mailbox):
    status, placed = buyer("POST", "/order", {"product": "credits", "hc": 4, "coin": "USDC.ERC20"})
    if status != 200:
        failures.append(f"part payment order: {status} {placed}")
        return
    key = max(fake.invoices, key=lambda k: int(k[3:]))
    fake.invoices[key].update(status="pending", received="22.99", paidIn="DAI.ERC20")
    webhook(buyer, cfg, key, "InvoicePending")
    app.jobs.join()
    shown = buyer("GET", "/order/" + placed["order"])[1]
    if shown["state"] != "pending" or shown["received"] != "22.99":
        failures.append(f"a part payment is not shown: {shown}")
    lines = app.book.count
    for value in ("22.97", "22.99", "22.98"):
        fake.invoices[key]["received"] = value
        app.reconcile()
        app.jobs.join()
    if app.book.count != lines:
        failures.append("a part payment revalued by cents writes ledger lines")
    before = Decimal(buyer("GET", "/account")[1]["hc"])
    fake.invoices[key]["status"] = "completed"
    webhook(buyer, cfg, key)
    app.jobs.join()
    app.mails.join()
    shown = buyer("GET", "/order/" + placed["order"])[1]
    if shown["state"] != "completed" or not shown["issued"] or shown["issuedHC"] != "1.044500" or Decimal(buyer("GET", "/account")[1]["hc"]) != before + Decimal("1.0445"):
        failures.append(f"a completed part payment does not issue HC in proportion: {shown}")
    note = [m.get_content() for rcpt, m in mailbox.mail if "ops@acme.example" in rcpt and placed["order"] in m["Subject"]]
    if not note or "22.98 USD arrived" not in note[-1] or "1.044500 hour-credits" not in note[-1] or "Received in DAI: 22.98 USD" not in note[-1]:
        failures.append(f"the receipt for a part payment: {note[-1:]}")
    if [e["paidWith"] for e in lines_of(cfg, "payment") if e["order"] == placed["order"]] != [["DAI.ERC20"]]:
        failures.append("the ledger does not record the stablecoin that was paid")
    if not any(cfg.notify in rcpt and m["Subject"].startswith("Paid in part:") for rcpt, m in mailbox.mail):
        failures.append("we are not told of a part payment that settled")


def toss_env(work, fake, mailbox, keys, name="data", sandbox=False):
    env = {
        "SPARKPIPE_PAY_DATA": str(work / name), "SPARKPIPE_PAY_SITE": str(work / "site"),
        "COINPAYMENTS_API": "http://127.0.0.1:9/api/v2", "COINPAYMENTS_CLIENT_ID": CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(work / "secret"),
        "SPARKPIPE_PAY_SMTP": f"127.0.0.1:{mailbox.server_address[1]}", "TOSS_API": f"http://127.0.0.1:{fake.server_address[1]}",
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


def check_toss_keys(failures, work, fake, mailbox):
    for keys, sandbox in ((TOSS_TEST, False), (TOSS_LIVE, True), ((TOSS_LIVE[0], ""), False), (("", TOSS_TEST[1]), True)):
        try:
            sp.Config(toss_env(work, fake, mailbox, keys, sandbox=sandbox))
            failures.append(f"Toss Payments keys {keys} are accepted {'in the sandbox' if sandbox else 'on the live service'}, or one key without the other")
        except ValueError as error:
            if "test_" in str(error) or "live_" in str(error) or "fake" in str(error):
                failures.append(f"the key error names a key: {error}")
    server, buyer = started(sp.Config(toss_env(work, fake, mailbox, None, "closed")))
    try:
        if buyer("GET", "/toss/config")[1] != {"enabled": False, "clientKey": None, "variantKey": "DEFAULT", "test": False}:
            failures.append("the live service offers cards without live Toss Payments keys")
        sign_in(buyer, mailbox, "nokeys@acme.example", company="No Keys", accept=True)
        status, refused = buyer("POST", "/order", {"product": "credits", "hc": 5, "processor": "toss"})
        if status != 503 or refused.get("reason") != "cards_closed" or server.app.book.orders:
            failures.append(f"a card order is taken without live Toss Payments keys: {status} {refused}")
    finally:
        server.shutdown()


def check_toss_sandbox(failures, work, fake, mailbox):
    server, tester = started(sp.Config(toss_env(work, fake, mailbox, TOSS_TEST, "sandbox", sandbox=True)))
    port = server.server_address[1]
    try:
        if tester("GET", "/toss/config")[1] != {"enabled": True, "clientKey": TOSS_TEST[0], "variantKey": "DEFAULT", "test": True}:
            failures.append("the sandbox does not run Toss Payments in test mode")
        code = tester("POST", "/code", {"email": "try@acme.example"})[1].get("code")
        tester("POST", "/verify", {"email": "try@acme.example", "code": code, "company": "Try Co", "accept": True})
        status, placed = tester("POST", "/market/buy", {"hc": 1, "processor": "toss"})
        request = placed.get("toss", {})
        if status != 200 or request.get("clientKey") != TOSS_TEST[0] or not request.get("test") or request.get("amount") != 28600 or not request.get("successUrl", "").startswith("https://sparkpipe.ai/pay/sandbox/api/toss/success/"):
            failures.append(f"a sandbox card purchase is not a Toss Payments test payment: {status} {placed}")
            return
        toss_payment(fake, "sandbox-key-1", request["orderId"], 28600)
        hop = succeed(port, request["orderId"], "sandbox-key-1", "28600")
        if hop != (303, f"https://sparkpipe.ai/account.html?sandbox&order={request['orderId']}") or Decimal(tester("GET", "/market/state")[1]["account"]["balance"]) != 1:
            failures.append(f"a sandbox test card payment does not issue its HC: {hop}")
        if tester("POST", "/market/buy", {"hc": 2})[1].get("hc") != "2.000000" or Decimal(tester("GET", "/market/state")[1]["account"]["balance"]) != 3:
            failures.append("the sandbox's free HC purchase stops working beside the test card payments")
        if tester("POST", "/order", {"product": "credits", "hc": 2, "coin": "USDC.BASE"})[0] != 404:
            failures.append("the sandbox opens a CoinPayments invoice")
        if tester("GET", "/swift/config")[1] != {"enabled": False} or tester("POST", "/order", {"product": "credits", "hc": 2, "processor": "swift", "declare": True})[0] != 404:
            failures.append("the sandbox takes a SWIFT order")
    finally:
        server.shutdown()


def check_toss(failures):
    work = Path(tempfile.mkdtemp(prefix="sparkpipe-toss-"))
    fake = serve(FakeToss, payments={}, calls=[], done={}, secret=TOSS_LIVE[1])
    mailbox = serve(FakeSMTP, mail=[])
    begin = NOW[0]
    try:
        site_copy(work)
        sp.clock = lambda: NOW[0]
        sp.LIMITS["order"] = (100, 600)
        check_toss_keys(failures, work, fake, mailbox)
        cfg = sp.Config(toss_env(work, fake, mailbox, TOSS_LIVE))
        server, buyer = started(cfg)
        sign_in(buyer, mailbox, "card@acme.example", company="Card Co", accept=True)
        try:
            card, deposit = toss_flows(failures, cfg, server.app, fake, buyer, mailbox, server.server_address[1])
            if card:
                toss_refunds(failures, cfg, server.app, fake, buyer, mailbox, card, deposit)
        finally:
            server.shutdown()
        NOW[0] = begin
        fake.secret = TOSS_TEST[1]
        check_toss_sandbox(failures, work, fake, mailbox)
    finally:
        NOW[0] = begin
        fake.shutdown()
        mailbox.shutdown()
        shutil.rmtree(work)


def order_toss(client, body):
    status, placed = client("POST", "/order", dict(body, processor="toss", product="credits"))
    return status, placed, placed.get("toss", {})


def toss_flows(failures, cfg, app, fake, buyer, mailbox, port):
    if buyer("GET", "/toss/config")[1] != {"enabled": True, "clientKey": TOSS_LIVE[0], "variantKey": "DEFAULT", "test": False}:
        failures.append("the live service hides cards with live Toss Payments keys set")
    for count in (0, "0.9", "1.0000001", "ten", "", True):
        if buyer("POST", "/order", {"product": "credits", "hc": count, "processor": "toss"})[0] != 400:
            failures.append(f"a card order of {count!r} HC is accepted")
    if buyer("POST", "/order", {"product": "credits", "hc": 5, "processor": "toss", "brn": "12345"})[0] != 400:
        failures.append("a card order with a malformed business registration number is accepted")
    if buyer("POST", "/order", {"product": "credits", "hc": 5, "processor": "paypal"})[0] != 400:
        failures.append("an order through an unknown processor is accepted")
    amounts = {count: order_toss(buyer, {"hc": count})[2].get("amount") for count in (143, 144)}
    shown = {count: buyer("GET", "/order/" + order_toss(buyer, {"hc": count})[2]["orderId"])[1] for count in (143, 144)}
    if amounts != {143: 4089800, 144: 3088800} or (shown[143]["price"], shown[143]["net"], shown[143]["vat"]) != ("26000", "3718000", "371800") or (shown[144]["price"], shown[144]["net"], shown[144]["vat"]) != ("19500", "2808000", "280800"):
        failures.append(f"the won price ladder at 143 and 144 HC: {amounts} {shown}")
    status, placed, request = order_toss(buyer, {"hc": 5, "total": "1"})
    order = placed.get("order", "")
    expected = {"amount": 143000, "currency": "KRW", "orderId": order, "customerEmail": "card@acme.example", "customerName": "Card Co", "clientKey": TOSS_LIVE[0],
                "successUrl": f"https://sparkpipe.ai/pay/api/toss/success/{order}", "failUrl": f"https://sparkpipe.ai/pay/api/toss/fail/{order}", "orderName": "SparkPipe hour-credits, 5 HC"}
    if status != 200 or any(request.get(name) != value for name, value in expected.items()):
        failures.append(f"a card order does not fix its won amount on the server: {status} {placed}")
        return None, None
    shown = buyer("GET", "/order/" + order)[1]
    if (shown["state"], shown["total"], shown["vat"], shown["hc"], shown["currency"], shown["processor"]) != ("created", "143000", "13000", "5", "KRW", "toss"):
        failures.append(f"a card order's status: {shown}")
    toss_payment(fake, "pay-card-1", order, 143000)
    hop = succeed(port, order, "pay-card-1", "143000")
    app.mails.join()
    confirms = [c for c in fake.calls if c[0] == "POST" and c[1] == "pay-card-1"]
    if hop != (303, f"https://sparkpipe.ai/account.html?order={order}") or confirms != [("POST", "pay-card-1", {"paymentKey": "pay-card-1", "orderId": order, "amount": 143000}, f"confirm-{order}")]:
        failures.append(f"the success redirect does not confirm the server's amount once with an idempotency key: {hop} {confirms}")
    if buyer("GET", "/account")[1]["hc"] != "5.000000" or lines_of(cfg).count("hc_issued") != 1:
        failures.append("a confirmed card payment does not issue its HC")
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
    if buyer("GET", "/account")[1]["hc"] != "5.000000" or lines_of(cfg).count("hc_issued") != 1 or len([c for c in fake.calls if c[0] == "POST" and c[1] == "pay-card-1"]) != 1:
        failures.append("a repeated success redirect or webhook issues HC twice or confirms twice")

    status, placed, request = order_toss(buyer, {"hc": 5})
    toss_payment(fake, "pay-cheap", request["orderId"], 100)
    succeed(port, request["orderId"], "pay-cheap", "100")
    shown = buyer("GET", "/order/" + request["orderId"])[1]
    if any(c[1] == "pay-cheap" and c[0] == "POST" for c in fake.calls) or shown["state"] != "failed" or shown["reason"] != "amount" or buyer("GET", "/account")[1]["hc"] != "5.000000":
        failures.append(f"a payment authorised for another amount is confirmed: {shown}")
    status, placed, request = order_toss(buyer, {"hc": 5})
    toss_payment(fake, "pay-odd", request["orderId"], 143000, total=1000)
    succeed(port, request["orderId"], "pay-odd", "143000")
    shown = buyer("GET", "/order/" + request["orderId"])[1]
    cancels = [c for c in fake.calls if c[1] == "pay-odd" and c[0] == "POST" and c[3] == f"mismatch-{request['orderId']}"]
    if shown["state"] != "cancelled" or shown["reason"] != "amount" or len(cancels) != 1 or fake.payments["pay-odd"]["status"] != "CANCELED" or buyer("GET", "/account")[1]["hc"] != "5.000000":
        failures.append(f"a payment Toss Payments reports for another amount is not refused and cancelled: {shown} {cancels}")
    status, placed, request = order_toss(buyer, {"hc": 5})
    toss_payment(fake, "pay-declined", request["orderId"], 143000, _decline="REJECT_CARD_COMPANY")
    succeed(port, request["orderId"], "pay-declined", "143000")
    shown = buyer("GET", "/order/" + request["orderId"])[1]
    if shown["state"] != "failed" or shown["reason"] != "REJECT_CARD_COMPANY":
        failures.append(f"a declined card is not recorded: {shown}")
    status, placed, request = order_toss(buyer, {"hc": 1})
    hop = redirected(port, f"/toss/fail/{request['orderId']}?code=PAY_PROCESS_CANCELED&message=x")
    if hop != (303, f"https://sparkpipe.ai/account.html?order={request['orderId']}") or buyer("GET", "/order/" + request["orderId"])[1]["state"] != "cancelled":
        failures.append(f"a payment the buyer cancels stays open: {hop}")

    status, placed, request = order_toss(buyer, {"hc": 10, "brn": "123-45-67890"})
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
    for _ in range(2):
        toss_webhook(buyer, {"createdAt": "2026-11-02T13:00:00.000000", "secret": "va-secret", "status": "DONE", "transactionKey": "t", "orderId": deposit})
        app.jobs.join()
    app.mails.join()
    if buyer("GET", "/account")[1]["hc"] != "15.000000" or lines_of(cfg).count("hc_issued") != 2:
        failures.append("a virtual account deposit verified with Toss Payments does not issue its HC exactly once")
    if lines_of(cfg, "payment")[-1].get("taxInvoice") != {"needed": True, "brn": "123-45-67890", "supply": "260000", "vat": "26000"} or [o["order"] for o in app.tax_invoices()] != [deposit]:
        failures.append("a virtual account payment is not recorded as needing an electronic tax invoice")
    note = next((m.get_content() for rcpt, m in mailbox.mail if "card@acme.example" in rcpt and m["Subject"] == f"Payment received: SparkPipe order {deposit}"), "")
    if "electronic tax invoice" not in note or "123-45-67890" not in note:
        failures.append(f"the virtual account receipt does not promise the tax invoice: {note!r}")

    status, placed, request = order_toss(buyer, {"hc": 5})
    late = request["orderId"]
    toss_payment(fake, "pay-va-2", late, 143000, method="가상계좌", after="WAITING_FOR_DEPOSIT")
    succeed(port, late, "pay-va-2", "143000")
    abandoned = order_toss(buyer, {"hc": 5})[2]["orderId"]
    NOW[0] = datetime(2026, 11, 3, 4, 0, 0, tzinfo=timezone.utc)
    app.reconcile()
    app.jobs.join()
    expired = [c for c in fake.calls if c[1] == "pay-va-2" and c[0] == "POST" and c[3] == f"expire-{late}"]
    states = [buyer("GET", "/order/" + ident)[1]["state"] for ident in (late, abandoned)]
    if states != ["timedout", "timedout"] or len(expired) != 1 or fake.payments["pay-va-2"]["status"] != "CANCELED" or buyer("GET", "/account")[1]["hc"] != "15.000000":
        failures.append(f"an unpaid virtual account or an abandoned card order does not expire with its order: {states} {expired}")
    return order, deposit


def toss_refunds(failures, cfg, app, fake, buyer, mailbox, card, deposit):
    def caught_up():
        with app.book.lock, open(app.book.path, "rb") as handle:
            app.book.catch_up(handle)
    if run_main(cfg, ["refund", card, "20.2", "71500", "old clause"]) != 2 or run_main(cfg, ["refund", card, "10.4", "143001", "too much"]) != 2:
        failures.append("a refund outside the Customer Agreement's clauses, or above the payment, is made")
    if run_main(cfg, ["refund", deposit, "10.4", "all", "service ended"]) != 2:
        failures.append("a virtual account payment is refunded without the account to refund it to")
    if run_main(cfg, ["refund", card, "16.4", "1000", "nothing lapsed"]) != 2:
        failures.append("a clause 16.4 refund is made with no locked HC lapsed")
    before = len(fake.calls)
    if run_main(cfg, ["refund", card, "10.4", "71500", "we stopped offering HC"]) != 0:
        failures.append("the refund command fails")
        return
    cancel = [c for c in fake.calls[before:] if c[0] == "POST"]
    caught_up()
    refund = lines_of(cfg, "refund")
    if cancel != [("POST", "pay-card-1", {"cancelReason": "Customer Agreement clause 10.4: we stopped offering HC", "refundableAmount": 143000, "cancelAmount": 71500}, f"refund-{card}-143000")] \
            or refund != [dict(refund[0], order=card, amount="71500", balance="71500", clause="10.4", hc="2.500000")] or buyer("GET", "/account")[1]["hc"] != "12.500000":
        failures.append(f"the refund command does not refund through the cancel API and debit the order's HC: {cancel} {refund}")
    if not any(m["Subject"] == f"Refund: SparkPipe order {card}" for rcpt, m in mailbox.mail if "card@acme.example" in rcpt):
        failures.append("the customer is not told of a refund")
    account = app.book.emails["card@acme.example"]
    app.market.credit(account, Decimal(200), "test")
    if buyer("POST", "/market/waitlist", {"lock": 144})[1].get("locked") != "144.000000" or run_main(cfg, ["refund", card, "10.4", "all", "rest"]) != 2:
        failures.append("a refund pays back HC that is locked on the waitlist")
    NOW[0] = datetime(2027, 11, 2, 3, 0, 1, tzinfo=timezone.utc)
    app.market.tick()
    app.mails.join()
    due = {r["refund"]["order"]: r["refund"]["amount"] for r in app.book.market.refunds[account] if r["refund"]}
    if due != {card: "71500", deposit: "286000"} or account in app.book.market.locks:
        failures.append(f"locked HC that lapses is not recorded as a refund due at the price paid: {due}")
    mail = next((m.get_content() for rcpt, m in mailbox.mail if cfg.notify in rcpt and m["Subject"] == "Refund due: locked HC lapsed"), "")
    if f"refund 71500 KRW of order {card} through toss under clause 16.4" not in mail or "Card Co <card@acme.example>" not in mail:
        failures.append(f"we are not emailed the refunds due for lapsed locked HC: {mail!r}")
    if run_main(cfg, ["refund", card, "16.4", "71501", "locked HC lapsed"]) != 2 or run_main(cfg, ["refund", card, "16.4", "71500", "locked HC lapsed"]) != 0:
        failures.append("a clause 16.4 refund is not limited to what lapsed while locked")
    if run_main(cfg, ["refund", deposit, "3.2", "100", "VAT not owed", "088", "110123456789", "Card Co"]) != 0 or fake.calls[-1][2].get("refundReceiveAccount") != {"bank": "088", "accountNumber": "110123456789", "holderName": "Card Co"}:
        failures.append("a virtual account payment is not refunded to the account given")
    if run_main(cfg, ["tax-invoice", deposit, "20261102-41000000-00000001"]) != 0:
        failures.append("an issued tax invoice cannot be recorded")
    caught_up()
    if deposit in [o["order"] for o in app.tax_invoices()]:
        failures.append("a recorded tax invoice is still listed")


BANK = {"SPARKPIPE_PAY_SWIFT_BENEFICIARY": "Test Beneficiary Co", "SPARKPIPE_PAY_SWIFT_BANK": "Test Foreign Exchange Bank", "SPARKPIPE_PAY_SWIFT_BANK_ADDRESS": "1 Test-ro, Jung-gu, Seoul",
        "SPARKPIPE_PAY_SWIFT_ACCOUNT": "000-000000-00-000", "SPARKPIPE_PAY_SWIFT_BIC": "TESTKRSE"}
SHOWN_BANK = {"beneficiary": "Test Beneficiary Co", "bank": "Test Foreign Exchange Bank", "address": "1 Test-ro, Jung-gu, Seoul", "account": "000-000000-00-000", "bic": "TESTKRSE"}


def swift_env(work, mailbox, name, bank=True):
    env = {"SPARKPIPE_PAY_DATA": str(work / name), "SPARKPIPE_PAY_SITE": str(work / "site"), "COINPAYMENTS_API": "http://127.0.0.1:9/api/v2", "COINPAYMENTS_CLIENT_ID": CLIENT,
           "COINPAYMENTS_CLIENT_SECRET_FILE": str(work / "secret"), "SPARKPIPE_PAY_SMTP": f"127.0.0.1:{mailbox.server_address[1]}"}
    return dict(env, **(BANK if bank else {}))


def check_swift(failures):
    work = Path(tempfile.mkdtemp(prefix="sparkpipe-swift-"))
    mailbox = serve(FakeSMTP, mail=[])
    begin = NOW[0]
    try:
        site_copy(work)
        sp.clock = lambda: NOW[0]
        sp.LIMITS["order"] = (100, 600)
        for broken in ({"SPARKPIPE_PAY_SWIFT_BIC": ""}, {"SPARKPIPE_PAY_SWIFT_BIC": "TESTKRS"}, {"SPARKPIPE_PAY_SWIFT_ACCOUNT": "000\n111"}):
            try:
                sp.Config(dict(swift_env(work, mailbox, "broken"), **broken))
                failures.append(f"incomplete or malformed SWIFT bank settings are accepted: {broken}")
            except ValueError:
                pass
        server, buyer = started(sp.Config(swift_env(work, mailbox, "closed", bank=False)))
        try:
            sign_in(buyer, mailbox, "closed@acme.example", company="Closed Co", accept=True)
            status, refused = buyer("POST", "/order", {"product": "credits", "hc": 5, "processor": "swift", "declare": True})
            if buyer("GET", "/swift/config")[1] != {"enabled": False} or status != 503 or refused.get("reason") != "swift_closed" or server.app.book.orders:
                failures.append(f"a SWIFT order is taken without the bank settings: {status} {refused}")
        finally:
            server.shutdown()
        cfg = sp.Config(swift_env(work, mailbox, "data"))
        server, buyer = started(cfg)
        try:
            sign_in(buyer, mailbox, "intl@acme.example", company="Intl Co", accept=True)
            swift_flows(failures, cfg, server.app, buyer, mailbox)
        finally:
            server.shutdown()
    finally:
        NOW[0] = begin
        mailbox.shutdown()
        shutil.rmtree(work)


def mail_to(mailbox, address, subject):
    return next((m.get_content() for rcpt, m in mailbox.mail if address in rcpt and m["Subject"] == subject), "")


def swift_flows(failures, cfg, app, buyer, mailbox):
    def caught_up():
        with app.book.lock, open(app.book.path, "rb") as handle:
            app.book.catch_up(handle)
    swift = {"product": "credits", "processor": "swift", "declare": True}
    if buyer("GET", "/swift/config")[1] != {"enabled": True}:
        failures.append("the live service does not offer SWIFT with the bank settings set")
    pricing = cfg.pricing()
    rates = {name: app.credit_terms(pricing, {"hc": 143}, currency, rate)["vat"] for name, currency, rate in (("swift", "USD", pricing["swiftVatRate"]), ("coin", "USD", None), ("toss", sp.KRW, None))}
    if rates != {"swift": "0.00", "coin": "286.00", "toss": "371800"}:
        failures.append(f"VAT is not 0 on SWIFT payments and only on them: {rates}")
    status, refused = buyer("POST", "/order", dict(swift, hc=143, declare=None))
    if status != 400 or refused.get("reason") != "declaration" or app.book.orders:
        failures.append(f"a SWIFT order is taken without the customer's declaration of clause 3.3(b): {status} {refused}")
    (status, first), (_, second) = (buyer("POST", "/order", dict(swift, hc=count)) for count in (143, 144))
    if status != 200 or (first["total"], first["net"], first["vat"], first["vatRate"], first["price"], first["currency"], first["processor"]) != ("2860.00", "2860.00", "0.00", "0", "20.00", "USD", "swift") \
            or (second["total"], second["vat"], second["price"]) != ("2160.00", "0.00", "15.00") or first["bank"] != SHOWN_BANK:
        failures.append(f"a SWIFT order is not priced on the US dollar ladder with 0% VAT, or lacks the bank details: {status} {first} {second}")
        return
    if not re.fullmatch(r"SW[A-Z2-7]{8}", first["reference"]) or first["reference"] == second["reference"] or first["reference"] in first["order"]:
        failures.append(f"SWIFT orders do not get unique payment references of their own: {first['reference']} {second['reference']}")
    shown = buyer("GET", "/order/" + first["order"])[1]
    if (shown["state"], shown["processor"], shown["reference"], shown["vat"], shown["checkout"], shown["issued"], shown["bank"]) != ("unpaid", "swift", first["reference"], "0.00", None, False, SHOWN_BANK):
        failures.append(f"an unpaid SWIFT order's status: {shown}")
    if next(o for o in lines_of(cfg, "order") if o["order"] == first["order"])["declaration"] != sp.declaration("Intl Co"):
        failures.append("a SWIFT order does not store the customer's declaration")
    app.mails.join()
    invoice = mail_to(mailbox, "intl@acme.example", f"Invoice for SparkPipe order {first['order']}: 2860.00 USD by SWIFT")
    for text in ("Total due: 2860.00 USD", "VAT, 0% on SWIFT payments: 0.00 USD", f"Payment reference: {first['reference']}", "Beneficiary: Test Beneficiary Co", "Account number: 000-000000-00-000", "Bank: Test Foreign Exchange Bank",
                 "Bank address: 1 Test-ro, Jung-gu, Seoul", "SWIFT/BIC: TESTKRSE", "Intl Co has no place of business in the Republic of Korea", "Choose charges OUR"):
        if text not in invoice:
            failures.append(f"the SWIFT invoice email lacks {text!r}: {invoice[:200]!r}")
    if first["reference"] not in mail_to(mailbox, cfg.notify, "SWIFT order: Intl Co, 2860.00 USD"):
        failures.append("we are not told of a SWIFT order and its reference")
    for args in (["SWNOTANORDER", "2860.00"], [first["reference"], "0"], [first["reference"], "12.345"], [first["reference"], "abc"], [first["reference"], "-5"], [first["reference"], "2859.99"]):
        if run_main(cfg, ["swift-confirm"] + args) != 2:
            failures.append(f"swift-confirm {' '.join(args)} is accepted")
    caught_up()
    if buyer("GET", "/order/" + first["order"])[1]["state"] != "unpaid" or buyer("GET", "/account")[1]["hc"] != "0.000000" or "swift_state" in lines_of(cfg):
        failures.append("a SWIFT transfer below the invoice total is recorded or issues HC")
    if run_main(cfg, ["swift-confirm", first["reference"], "2860.00", "MT103 20250101"]) != 0:
        failures.append("the operator cannot confirm a SWIFT transfer by its payment reference")
        return
    caught_up()
    shown = buyer("GET", "/order/" + first["order"])[1]
    if (shown["state"], shown["issued"], shown["issuedHC"], shown["received"]) != ("completed", True, "143", "2860.00") or buyer("GET", "/account")[1]["hc"] != "143.000000":
        failures.append(f"a confirmed SWIFT transfer does not issue its HC: {shown}")
    if lines_of(cfg, "payment")[-1] != {"order": first["order"], "processor": "swift", "reference": first["reference"], "total": "2860.00", "received": "2860.00", "currency": "USD", "paidWith": ["SWIFT"]} \
            or lines_of(cfg, "swift_state")[-1].get("note") != "MT103 20250101":
        failures.append(f"the SWIFT payment line: {lines_of(cfg, 'payment')[-1]}")
    note = mail_to(mailbox, "intl@acme.example", f"Payment received: SparkPipe order {first['order']}")
    if "Total paid by SWIFT bank transfer: 2860.00 USD" not in note or "VAT, 0% on SWIFT payments: 0.00 USD" not in note or "143 hour-credits" not in note:
        failures.append(f"the SWIFT receipt: {note!r}")
    if run_main(cfg, ["swift-confirm", first["order"], "2860.00"]) != 2 or lines_of(cfg).count("hc_issued") != 1:
        failures.append("a SWIFT order issues its HC twice")
    status, small = buyer("POST", "/order", dict(swift, hc=1))
    if run_main(cfg, ["swift-confirm", small["order"], "25.00"]) != 0:
        failures.append("an overpaid SWIFT order cannot be confirmed")
    caught_up()
    if buyer("GET", "/account")[1]["hc"] != "144.000000" or "Send the 5.00 USD above the total back by SWIFT" not in mail_to(mailbox, cfg.notify, f"Overpaid by SWIFT: {small['order']}") \
            or "the 5.00 USD above the total goes back by SWIFT" not in mail_to(mailbox, "intl@acme.example", f"Payment received: SparkPipe order {small['order']}"):
        failures.append("an overpaid SWIFT order does not issue its HC and have the excess sent back")
    if run_main(cfg, ["refund", small["order"], "3.5", "5.01", "above the total"]) != 2 or run_main(cfg, ["refund", small["order"], "3.5", "5.00", "above the total"]) != 0:
        failures.append("a clause 3.5 SWIFT refund is not limited to what arrived above the total")
    caught_up()
    if lines_of(cfg, "refund")[-1] != {"order": small["order"], "account": app.book.emails["intl@acme.example"], "processor": "swift", "amount": "5.00", "balance": "20.00", "clause": "3.5", "reason": "above the total", "hc": "0.000000"} \
            or buyer("GET", "/account")[1]["hc"] != "144.000000":
        failures.append(f"returning what arrived above a SWIFT order's total debits HC or is not recorded: {lines_of(cfg, 'refund')[-1:]}")
    if run_main(cfg, ["refund", first["order"], "10.4", "2860.01", "too much"]) != 2 or run_main(cfg, ["refund", first["order"], "10.4", "1430.00", "we stopped offering HC"]) != 0:
        failures.append("a SWIFT refund under clause 10.4 is refused, or allowed above the payment")
    caught_up()
    if buyer("GET", "/account")[1]["hc"] != "72.500000" or lines_of(cfg, "refund")[-1]["hc"] != "71.500000" \
            or "It goes back by SWIFT to the account it came from; 71.500000 HC of the order left your account." not in mail_to(mailbox, "intl@acme.example", f"Refund: SparkPipe order {first['order']}"):
        failures.append("a SWIFT refund does not debit the order's HC and tell the customer it goes back to the account it came from")
    if run_main(cfg, ["swift-cancel", second["reference"], "asked by the customer"]) != 0 or run_main(cfg, ["swift-cancel", second["reference"], "again"]) != 2 or run_main(cfg, ["swift-confirm", second["reference"], "2160.00"]) != 2:
        failures.append("a SWIFT order cannot be cancelled, or a cancelled one is cancelled or confirmed again")
    caught_up()
    shown = buyer("GET", "/order/" + second["order"])[1]
    if (shown["state"], shown["reason"], shown["issued"]) != ("cancelled", "asked by the customer", False) or buyer("GET", "/account")[1]["hc"] != "72.500000" \
            or "Do not send a transfer for this order" not in mail_to(mailbox, "intl@acme.example", f"Cancelled: SparkPipe order {second['order']}"):
        failures.append(f"a cancelled SWIFT order: {shown}")
    if run_main(cfg, ["swift-cancel", first["order"], "late"]) != 2:
        failures.append("a SWIFT order that issued its HC is cancelled")
    listed = {o["order"]: o["reference"] for o in buyer("GET", "/account")[1]["orders"]}
    if listed != {first["order"]: first["reference"], second["order"]: second["reference"], small["order"]: small["reference"]}:
        failures.append(f"the account does not list SWIFT orders with their references: {listed}")


def main():
    failures = []
    check_units(failures)
    check_legacy(failures)
    check_service(failures)
    check_toss(failures)
    check_swift(failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("pay: sessions, email codes, acceptance of the Korean and English Customer Agreement only, HC orders by quantity on the price ladder (143 and 144 HC in US dollars and won), USDC and USDT invoices on the chosen network, signed webhooks, receipts, part payments, mismatches, reconciliation, the removed routes, the hash-chained ledger with two writers and the 9-line live ledger replayed; Toss Payments key gating, won amounts fixed on the server and confirmed once, other amounts refused and cancelled, webhooks that only trigger a check, virtual accounts and their expiry, refunds under the named clause (clause 16.4 only up to what lapsed while locked), the tax invoice list and the sandbox's test payments; USD bank transfers (SWIFT) gated on complete bank settings and the customer's clause 3.3(b) declaration, priced on the US dollar ladder with 0% VAT only for them, invoiced by email with the bank details and a unique payment reference, confirmed by the operator only for the whole amount, overpayments sent back under clause 3.5, cancellation and refunds back to the account the payment came from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
