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


WINDOWS = {"americas": {"id": "americas", "from": 0, "to": 8, "localDay": -1}, "asia": {"id": "asia", "from": 8, "to": 16, "localDay": 0}, "europe": {"id": "europe", "from": 16, "to": 24, "localDay": 0}}


def plan(number, email, windows, days="every", fleets=1, months=24, start="2026-12-15", payment="gates", lock=None):
    found = {"id": f"SP-PLAN-{number}", "email": email, "company": "Acme", "start": start, "months": months, "windows": [WINDOWS[w] for w in windows], "days": days, "fleets": fleets,
             "hourPrice": "15.00", "payment": payment, "releaseFee": "0.10", "maintenanceHoursPerWeek": 3}
    if lock:
        found["lock"] = lock
    return found


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
        credits_flow(failures, cfg, app, fake, buyer, mailbox)
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


def prepayment_flow(failures, cfg, app, fake, buyer, mailbox, port):
    plans = [
        plan("0001", "ops@acme.example", ["asia"], start="2026-11-01"),
        plan("0002", "OPS@acme.example", ["europe"], fleets=2, start="2026-11-01"),
        plan("0003", "else@other.example", ["americas"], start="2026-11-01"),
        plan("0004", "ops@acme.example", ["asia"], months=3, start="2026-11-10", payment="upfront"),
    ]
    Path(cfg.path("plans.json")).write_text(json.dumps(plans))
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
    if state["paid"] != "100.00" or not buyer("GET", "/order/" + placed["order"])[1]["issued"]:
        failures.append(f"reconciling a missed webhook does not issue the prepayment: {state}")
    NOW[0] = datetime(2026, 12, 1, 3, 0, 0, tzinfo=timezone.utc)
    state = buyer("GET", "/account")[1]["plans"]
    status, _ = buyer("POST", "/order", dict(pay, amount="100.00"))
    if "Gate 1 was missed" not in (state[0]["ended"] or "") or status != 409:
        failures.append(f"a missed gate does not stop prepayment: {state[0]} {status}")
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
    print("pay: plan hours and gates match the Customer Agreement for every window, day choice, term and fleet count, invalid plans fail; email codes, sessions, terms acceptance, USDC and USDT invoices on the chosen network, payments in another stablecoin, signed webhooks, receipts, HC orders and part payments, reconciliation, gates, quote requests and the hash-chained ledger behave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
