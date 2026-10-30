#!/usr/bin/env python3

import base64
import email
import email.policy
import hashlib
import hmac
import http.server
import json
import re
import shutil
import socketserver
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import sparkpipe_pay as sp

SECRET = b"test-client-secret"
CLIENT = "test-client-id"
NOW = [datetime(2026, 11, 2, 3, 0, 0, tzinfo=timezone.utc)]


def docs_signature(method, url, stamp, body):
    message = "﻿" + method + url + CLIENT + stamp
    return base64.b64encode(hmac.new(SECRET, message.encode("utf-8") + body, hashlib.sha256).digest()).decode()


def check_dates(failures):
    plan = sp.tranches(date(2026, 12, 15), 24)
    if (len(plan), plan[0], plan[1], plan[14], plan[-1], sum(plan)) != (25, 360, 624, 600, 288, 15048):
        failures.append(f"tranches of a 15 Dec 2026 term: {plan}")
    if (sum(plan[:6]), sum(plan[:12])) != (3456, 7224):
        failures.append("the 15 Dec 2026 gates differ from agreement clause 6.6")
    if sp.term_end(date(2028, 2, 29), 24) != date(2030, 2, 28) or sp.term_end(date(2026, 12, 1), 12) != date(2027, 11, 30):
        failures.append("term_end differs from agreement clause 3.1")
    if sp.served_days(date(2026, 12, 15), date(2026, 12, 21)) != 6:
        failures.append("served_days counts a Sunday")


def check_covered(failures):
    hourly = {"product": "hourly", "hours": 168, "net": "3360.00", "total": "3696.00"}
    if sp.covered(hourly, sp.money("22.99")) != (1, "20.00") or sp.covered(hourly, sp.money("3696.00")) != (168, "3360.00") or sp.covered(hourly, sp.money("21.99"))[0] != 0:
        failures.append(f"hours covered by a part payment: {sp.covered(hourly, sp.money('22.99'))}")
    prepay = {"product": "two-year", "hours": "6.6666", "net": "100.00", "total": "110.00", "price": "15.00"}
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
        self.answer(200, {"id": key, "status": invoice["status"], "amount": {"breakdown": {}, "total": invoice["total"]}, "currency": {"id": "5057", "symbol": "USD"}, "payments": [{"paymentCurrencySymbol": "USDC.BASE", "nativeActualAmount": arrived}]})


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
        for name in ("hourly-en.md", "hourly-ko.md"):
            shutil.copy(ROOT / "site" / "terms" / name, work / "site" / "terms" / name)
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
        })
        sp.clock = lambda: NOW[0]
        sp.LIMITS["order"] = (40, 600)
        server = sp.start(cfg, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port, app = server.server_address[1], server.app
        buyer = Client(port, origin="https://sparkpipe.ai")
        account_flow(failures, cfg, app, buyer, mailbox, port)
        hourly_flow(failures, cfg, app, fake, buyer, mailbox)
        part_payment_flow(failures, cfg, app, fake, buyer, mailbox)
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
        index = next(i for i, line in enumerate(lines) if '"hours":4,' in line)
        lines[index] = lines[index].replace('"hours":4,', '"hours":40,', 1)
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
    if status != 200 or summary.get("company") != "Acme 주식회사" or not summary.get("termsCurrent") or summary.get("room") != 168:
        failures.append(f"opening an account: {status} {summary}")
        return
    if "HttpOnly" not in buyer.last_cookie or "SameSite=Lax" not in buyer.last_cookie or "Path=/pay/api" not in buyer.last_cookie:
        failures.append(f"session cookie attributes: {buyer.last_cookie}")
    record = [json.loads(line) for line in open(cfg.path("ledger.jsonl"), encoding="utf-8") if '"acceptance"' in line][-1]["data"]
    if record["ko"] != cfg.terms("hourly-ko.md") or "Korean text, which prevails" not in record["statement"] or "Acme 주식회사" not in record["statement"]:
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


def hourly_flow(failures, cfg, app, fake, buyer, mailbox):
    base = {"product": "hourly", "hours": 4, "start": "2026-12-01", "coin": "USDC.BASE"}
    if buyer("POST", "/order", dict(base, start="2026-11-30"))[0] != 400:
        failures.append("hours starting before the first sale date are ordered")
    for coin in ("BTC", "TUSD.TRC20", "base", None):
        if buyer("POST", "/order", dict(base, coin=coin))[0] != 400:
            failures.append(f"an order in {coin} is accepted")
    anonymous = Client(buyer.port)
    if anonymous("POST", "/order", base)[0] != 401:
        failures.append("an order without signing in is accepted")
    status, placed = buyer("POST", "/order", base)
    if status != 200 or not placed.get("checkout", "").startswith("https://pay.example/c/"):
        failures.append(f"hourly order: {status} {placed}")
        return
    invoice = fake.invoices["inv1"]["body"]
    expected = {"currency": "5057", "invoiceId": placed["order"], "amount": {"breakdown": {"subtotal": "80.00", "taxTotal": "8.00"}, "total": "88.00"}}
    if any(invoice.get(key) != value for key, value in expected.items()):
        failures.append(f"invoice body: {invoice}")
    if invoice["payment"] != {"paymentCurrency": sp.COINS["USDC.BASE"], "refundEmail": "ops@acme.example"} or invoice["buyer"] != {"companyName": "Acme 주식회사", "emailAddress": "ops@acme.example"}:
        failures.append(f"invoice payment or buyer: {invoice['payment']} {invoice['buyer']}")
    if invoice["items"][0]["quantity"] != {"value": 4, "type": "hours"} or invoice["webhooks"][0]["notificationsUrl"] != "https://sparkpipe.ai/pay/api/coinpayments":
        failures.append(f"invoice items or webhooks: {invoice['items']} {invoice['webhooks']}")
    if invoice["successUrl"] != f"https://sparkpipe.ai/account.html?order={placed['order']}":
        failures.append(f"invoice return address: {invoice['successUrl']}")
    status, shown = buyer("GET", "/order/" + placed["order"])
    if status != 200 or shown["state"] != "unpaid" or shown["checkout"] != placed["checkout"] or "email" in shown:
        failures.append(f"unpaid order status: {status} {shown}")
    if buyer("POST", "/order", dict(base, hours=165))[0] != 409:
        failures.append("an unpaid order does not count toward the 168-hour limit")
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
    if summary["hours"] != 4 or summary["blocks"] != [{"order": placed["order"], "hours": 4, "start": "2026-12-01", "last": "2027-11-30"}] or ledger_types(cfg).count("hours_issued") != 1:
        failures.append(f"completed order: {summary} {ledger_types(cfg)}")
    receipts = [m for rcpt, m in mailbox.mail if "ops@acme.example" in rcpt and placed["order"] in m["Subject"]]
    if len(receipts) != 1 or "88.00" not in receipts[0].get_content():
        failures.append(f"receipt mail: {[m['Subject'] for _, m in mailbox.mail]}")
    if not any(cfg.notify in rcpt and m["Subject"].startswith("Paid:") for rcpt, m in mailbox.mail):
        failures.append("we are not told of a payment")
    payment = [json.loads(line) for line in open(cfg.path("ledger.jsonl"), encoding="utf-8") if '"payment"' in line][-1]["data"]
    if payment["paidWith"] != ["USDC.BASE"]:
        failures.append(f"the payment line does not record the coin: {payment}")
    if buyer("POST", "/order", dict(base, hours=165))[0] != 409 or buyer("POST", "/order", dict(base, hours=164, coin="USDT.TRC20"))[0] != 200:
        failures.append("issued hours and the holding limit")
    fake.invoices["inv2"]["status"] = "completed"
    fake.invoices["inv2"]["total"] = "1.00"
    webhook(buyer, cfg, "inv2")
    app.jobs.join()
    if fake.invoices["inv2"]["body"]["payment"]["paymentCurrency"] != "9:TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t":
        failures.append("a USDT order on Tron does not preset USDT on Tron")
    second = next(o for o in buyer("GET", "/account")[1]["orders"] if o["hours"] == 164)
    if second["coin"] != "USDT.TRC20":
        failures.append(f"the order does not show its coin: {second}")
    if second["state"] != "mismatch" or second["issued"]:
        failures.append(f"an underpaid invoice issues hours: {second}")
    fake.down = True
    Path(cfg.path("approved.json")).write_text(json.dumps(["OPS@acme.example"]))
    status, refused = buyer("POST", "/order", dict(base, hours=1))
    fake.down = False
    if status != 502 or buyer("GET", "/account")[1]["orders"][0]["state"] != "failed":
        failures.append(f"a failed invoice is not recorded: {status} {refused}")
    terms = Path(cfg.site) / "terms" / "hourly-en.md"
    terms.write_text(terms.read_text() + "\n")
    status, refused = buyer("POST", "/order", dict(base, hours=1))
    if status != 409 or refused.get("reason") != "terms":
        failures.append(f"an order under changed terms does not ask to accept them again: {status} {refused}")
    if buyer("POST", "/order", dict(base, hours=1, accept=True))[0] != 200 or not buyer("GET", "/account")[1]["termsCurrent"]:
        failures.append("accepting changed terms at the order does not record it")


def part_payment_flow(failures, cfg, app, fake, buyer, mailbox):
    status, placed = buyer("POST", "/order", {"product": "hourly", "hours": 4, "start": "2026-12-01", "coin": "USDC.ERC20"})
    if status != 200:
        failures.append(f"part payment order: {status} {placed}")
        return
    key = max(fake.invoices, key=lambda k: int(k[3:]))
    fake.invoices[key].update(status="pending", received="22.99")
    webhook(buyer, cfg, key, "InvoicePending")
    app.jobs.join()
    app.mails.join()
    shown = buyer("GET", "/order/" + placed["order"])[1]
    if shown["state"] != "pending" or shown["received"] != "22.99":
        failures.append(f"a part payment is not shown: {shown}")
    if not any(m["Subject"] == f"Part payment: {placed['order']}" for _, m in mailbox.mail):
        failures.append("we are not told of a part payment")
    fake.invoices[key]["status"] = "completed"
    webhook(buyer, cfg, key)
    app.jobs.join()
    shown = buyer("GET", "/order/" + placed["order"])[1]
    if shown["state"] != "mismatch" or shown["issued"]:
        failures.append(f"an accepted part payment issues the whole order: {shown}")
    account = app.book.orders[placed["order"]]["account"]
    held = app.book.held(account, sp.clock())
    before = buyer("GET", "/account")[1]["hours"]
    issued = app.settle(placed["order"], {"status": "completed", "amount": {"total": "88.00"}, "currency": {"id": "5057"}, "payments": [{"nativeActualAmount": "22.99"}]}, released=True)
    app.mails.join()
    after = buyer("GET", "/account")[1]
    if not issued or issued["hours"] != 1 or issued["net"] != "20.00" or after["hours"] != before + 1:
        failures.append(f"releasing a part payment does not issue the hour it covers: {issued} {before} {after['hours']}")
    note = [m for rcpt, m in mailbox.mail if "ops@acme.example" in rcpt and placed["order"] in m["Subject"]]
    if not note or "22.99 USD arrived" not in note[-1].get_content():
        failures.append("the receipt for a part payment does not say what arrived")
    if app.book.held(account, sp.clock()) != held - 3:
        failures.append("the holding limit counts the ordered hours of a part payment, not the issued ones")


def prepayment_flow(failures, cfg, app, fake, buyer, mailbox, port):
    agreements = [
        {"id": "SP2Y-TEST-0001", "email": "ops@acme.example", "company": "Acme", "signed": "2026-11-01", "hourPrice": "15.00"},
        {"id": "SP2Y-TEST-0002", "email": "OPS@acme.example", "company": "Acme", "signed": "2026-11-01", "hourPrice": "15.00"},
        {"id": "SP2Y-TEST-0003", "email": "else@other.example", "company": "Other", "signed": "2026-11-01", "hourPrice": "15.00"},
    ]
    Path(cfg.path("agreements.json")).write_text(json.dumps(agreements))
    plan = sp.tranches(date(2026, 11, 1), 24)
    found = buyer("GET", "/account")[1]["agreements"]
    if [a["agreement"] for a in found] != ["SP2Y-TEST-0001", "SP2Y-TEST-0002"]:
        failures.append(f"the account lists other agreements: {found}")
        return
    if found[0]["hours"] != sum(plan) or found[0]["nextGate"] != {"name": "Gate 1", "due": "2026-11-30", "amount": f"{15 * sum(plan[:6])}.00"}:
        failures.append(f"agreement standing: {found[0]}")
    pay = {"product": "two-year", "agreement": "SP2Y-TEST-0001", "coin": "USDT.SOL"}
    if buyer("POST", "/order", dict(pay, agreement="SP2Y-TEST-0003", amount="100.00"))[0] != 404:
        failures.append("an account can prepay another account's agreement")
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
    state = buyer("GET", "/account")[1]["agreements"][0]
    if state["paid"] != "100.00" or not buyer("GET", "/order/" + placed["order"])[1]["issued"]:
        failures.append(f"reconciling a missed webhook does not issue the prepayment: {state}")
    NOW[0] = datetime(2026, 12, 1, 3, 0, 0, tzinfo=timezone.utc)
    state = buyer("GET", "/account")[1]["agreements"]
    status, _ = buyer("POST", "/order", dict(pay, amount="100.00"))
    if "Gate 1 was missed" not in (state[0]["ended"] or "") or status != 409:
        failures.append(f"a missed gate does not stop prepayment: {state[0]} {status}")
    if "Phase A ended" not in (state[1]["ended"] or ""):
        failures.append(f"an unpaid agreement survives Phase A: {state[1]}")


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


def main():
    failures = []
    check_dates(failures)
    check_sessions(failures)
    check_covered(failures)
    check_service(failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("pay: tranches match the agreement; email codes, sessions, terms acceptance, USDC and USDT invoices on the chosen network, signed webhooks, receipts, part payments, reconciliation, the holding limit, gates, quote requests and the hash-chained ledger behave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
