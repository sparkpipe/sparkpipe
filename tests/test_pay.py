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
from decimal import Decimal
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import sparkpipe_pay as sp

SECRET = b"test-client-secret"
CLIENT = "test-client-id"
NOW = [datetime(2026, 11, 2, 3, 0, 0, tzinfo=timezone.utc)]


def docs_signature(method, url, stamp, body):
    message = "﻿" + method + url + CLIENT + stamp
    return base64.b64encode(hmac.new(SECRET, message.encode("utf-8") + body, hashlib.sha256).digest()).decode()


OFFICES = {
    "day": {"zone": "Asia/Seoul", "from": 9, "to": 18},
    "asia": {"zone": "Asia/Seoul", "from": 8, "to": 16},
    "europe": {"zone": "Asia/Seoul", "from": 16, "to": 0},
    "americas": {"zone": "Asia/Seoul", "from": 0, "to": 8},
    "full": {"zone": "Asia/Seoul", "from": 0, "to": 0},
    "newyork": {"zone": "America/New_York", "from": 9, "to": 17},
    "night": {"zone": "America/New_York", "from": 22, "to": 6},
    "dawn": {"zone": "America/New_York", "from": 0, "to": 8},
    "berlin": {"zone": "Europe/Berlin", "from": 9, "to": 18},
    "kolkata": {"zone": "Asia/Kolkata", "from": 9, "to": 18},
}


def plan(number, email, office, days="every", fleets=1, months=24, start="2026-12-15", payment="gates", lock=None):
    found = {"id": f"SP-PLAN-{number}", "email": email, "company": "Acme", "start": start, "months": None if payment == "monthly" else months, "office": dict(OFFICES[office] if isinstance(office, str) else office),
             "days": days, "fleets": fleets, "hourPrice": "15.00", "payment": payment, "releaseFee": "0.10", "maintenanceHoursPerWeek": 3}
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


def kst_hours(found, first, last):
    return sorted(datetime.fromtimestamp(key * 3600, sp.KST) for key in sp.plan_slots(found, first, last))


def local_hours(found, first, last):
    zone = sp.zoneinfo.ZoneInfo(found["office"]["zone"])
    return sorted(moment.astimezone(zone) for moment in kst_hours(found, first, last))


def check_dates(failures):
    pricing = json.loads((ROOT / "site" / "pricing.js").read_text().split("=", 1)[1].strip().rstrip(";"))
    hours = sp.tranches(pricing, plan(1, "a@x", "day"))
    if (len(hours), hours[0], hours[1], hours[14], hours[-1], sum(hours)) != (25, 153, 279, 261, 126, 6579):
        failures.append(f"tranches of the Customer Agreement's example plan: {hours}")
    if (sum(hours[:6]), sum(hours[:12])) != (1512, 3159):
        failures.append("the example plan's gates differ from the Customer Agreement")
    if sum(sp.tranches(pricing, plan(1, "a@x", "day", "weekdays"))) != 4707:
        failures.append("the example plan on Weekdays holds the wrong hours")
    if sum(sp.tranches(pricing, plan(1, "a@x", "full", fleets=2))) != 2 * 17544:
        failures.append("a full-day plan on two fleets holds the wrong hours")
    if pricing["plans"]["office"]["minHours"] != sp.MIN_OFFICE_HOURS or [t["id"] for t in pricing["plans"]["terms"]] != ["monthly", "1y", "2y", "3y"]:
        failures.append(f"the published terms or the shortest office hours differ from the Customer Agreement: {pricing['plans']['terms']}")
    night = plan(1, "a@x", "night", "weekdays", start="2027-01-04")
    week = [sum(1 for moment in kst_hours(night, date(2027, 1, 4), date(2027, 1, 10)) if moment.weekday() == day) for day in range(7)]
    if week != [0, 8, 8, 8, 8, 8, 0]:
        failures.append(f"New York office hours from 22:00 to 06:00 on Weekdays do not run from KST Tuesday to Saturday, the hours New York's Monday to Friday start in: {week}")
    if [moment.strftime("%a %H") for moment in local_hours(night, date(2027, 1, 9), date(2027, 1, 9))][:1] != ["Fri 22"]:
        failures.append("office hours that start on a local Friday stop at local midnight instead of running to their end")
    dawn = plan(1, "a@x", "dawn", start="2027-01-01", months=12)
    spring, autumn = local_hours(dawn, date(2027, 3, 14), date(2027, 3, 15)), local_hours(dawn, date(2027, 11, 7), date(2027, 11, 8))
    spring = [m for m in spring if m.date() == date(2027, 3, 14)]
    autumn = [m for m in autumn if m.date() == date(2027, 11, 7)]
    if len(spring) != 7 or len(autumn) != 9 or [m.hour for m in spring] != [0, 1, 3, 4, 5, 6, 7]:
        failures.append(f"office hours on the days New York changes its clock are not the hours that elapse between 00:00 and 08:00: {[m.strftime('%H%z') for m in spring]} {[m.strftime('%H%z') for m in autumn]}")
    berlin = plan(1, "a@x", "berlin", start="2027-01-01", months=12)
    winter, summer = kst_hours(berlin, date(2027, 1, 13), date(2027, 1, 13)), kst_hours(berlin, date(2027, 7, 14), date(2027, 7, 14))
    if [m.hour for m in winter] != [0, 1, 17, 18, 19, 20, 21, 22, 23] or [m.hour for m in summer] != [0, 16, 17, 18, 19, 20, 21, 22, 23]:
        failures.append(f"Berlin office hours do not move in KST when Berlin changes its clock: {[m.hour for m in winter]} {[m.hour for m in summer]}")
    kolkata = kst_hours(plan(1, "a@x", "kolkata", start="2027-01-01", months=12), date(2027, 1, 13), date(2027, 1, 13))
    if [m.hour for m in kolkata] != list(range(13, 22)):
        failures.append(f"office hours in a zone half an hour off KST do not hold the nine KST hours that start within them: {[m.hour for m in kolkata]}")
    seoul = plan(1, "a@x", "day", "weekdays", start="2027-01-01", months=12)
    if [m.strftime("%a") for m in kst_hours(seoul, date(2027, 1, 8), date(2027, 1, 11))][::9] != ["Fri", "Mon"]:
        failures.append("Seoul Weekdays hold a weekend day")
    for bad in (plan(1, "a@x", "day", lock=["F1"]), plan(1, "a@x", "full", "weekdays", lock=["F1"]), plan(1, "a@x", "day", "weekends"),
                plan(1, "a@x", {"zone": "Asia/Seoul", "from": 9, "to": 16}), plan(1, "a@x", {"zone": "Mars/Olympus", "from": 9, "to": 18}),
                plan(1, "a@x", {"zone": "Asia/Seoul", "from": 9, "to": 24}), plan(1, "a@x", {"zone": "Asia/Seoul", "from": "9", "to": 18}),
                dict(plan(1, "a@x", "day"), payment="monthly"), dict(plan(1, "a@x", "day"), months=None), dict(plan(1, "a@x", "day"), releaseFee="1.5")):
        try:
            sp.tranches(pricing, bad)
            failures.append(f"an invalid plan is accepted: {bad}")
        except ValueError:
            pass
    try:
        sp.tranches(pricing, plan(1, "a@x", {"zone": "Asia/Seoul", "from": 20, "to": 4}))
        sp.tranches(pricing, plan(1, "a@x", "full", lock=["F1"]))
    except ValueError as error:
        failures.append(f"eight office hours across midnight, or a locked full day, are refused: {error}")
    first, last = date(2026, 12, 15), date(2026, 12, 31)
    if [sp.hours_before(pricing, plan(1, "a@x", "day"), first, last, datetime(2026, 12, 20, hour, minute, tzinfo=timezone.utc)) for hour, minute in ((3, 0), (3, 30), (8, 0), (15, 0))] != [48, 49, 53, 54]:
        failures.append("Plan hours that occur before a payment are counted wrongly")
    if sp.term_end(date(2028, 2, 29), 24) != date(2030, 2, 28) or sp.term_end(date(2026, 12, 1), 12) != date(2027, 11, 30):
        failures.append("term_end differs from the Customer Agreement")
    month = plan(1, "a@x", "newyork", "weekdays", payment="monthly")
    spans = sp.tranche_spans(pricing, month, date(2027, 1, 31))
    if [(a.isoformat(), b.isoformat(), n) for a, b, n in spans] != [("2026-12-15", "2026-12-31", 104), ("2027-01-01", "2027-01-31", 175)]:
        failures.append(f"a Monthly plan's Tranches are not its calendar months in KST, each holding the KST hours that start in it: {spans}")


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


CHOICE = {"start": "2026-12-15", "term": "2y", "office": OFFICES["day"], "days": "every", "fleets": 1, "lock": False, "models": ["kimi-k3"],
          "customer": {"kind": "company", "jurisdiction": "Delaware, United States", "registration": "7781234", "address": "1 Main St, Dover, DE"}}
MONTHLY = dict(CHOICE, term="monthly", office=OFFICES["newyork"], days="weekdays", customer={"kind": "individual", "address": "3 Side St, Busan"})


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


def kept(app):
    return [found.plan for found in app.commitments(NOW[0])]


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
    customer, office = CHOICE["customer"], CHOICE["office"]
    for bad in (dict(CHOICE, term="3m"), dict(CHOICE, term=None), dict({k: v for k, v in CHOICE.items() if k != "term"}, months=3), dict(CHOICE, term=24),
                dict(CHOICE, start="2026-11-20"), dict(CHOICE, start="2027-01-02"), dict(CHOICE, start="15 Dec"),
                dict(CHOICE, office=None), dict(CHOICE, office=dict(office, zone="Mars/Olympus")), dict(CHOICE, office=dict(office, to=16)), dict(CHOICE, office=dict(office, **{"from": "9"})),
                dict(CHOICE, office=dict(office, to=24)), dict(CHOICE, office={"zone": "Asia/Seoul"}), dict(CHOICE, fleets=0), dict(CHOICE, fleets=4), dict(CHOICE, fleets=True),
                dict(CHOICE, days="weekends"), dict(CHOICE, lock=True), dict(CHOICE, office=OFFICES["full"], days="weekdays", lock=True),
                dict(CHOICE, models=[]), dict(CHOICE, models=["gpt"]), dict(CHOICE, customer=dict(customer, kind="trust")), dict(CHOICE, customer=dict(customer, registration="")),
                dict(CHOICE, customer=dict(customer, address="1 | 2")), dict(CHOICE, customer=dict(customer, brn="12345"))):
        status, refused = team("POST", "/plans/quote", bad)
        if status != 400:
            failures.append(f"an invalid Plan Order is quoted: {status} {refused} for {bad}")
    status, locked = team("POST", "/plans/quote", dict(CHOICE, office=OFFICES["full"], lock=True))
    if status != 200 or locked.get("lock") != ["F1"] or locked.get("price") != "16.00" or locked.get("hours") != 17544:
        failures.append(f"a locked full-day plan is not assigned the free serving fleet at the lock price: {status} {locked}")
    status, refused = team("POST", "/plans/quote", dict(CHOICE, fleets=2))
    if status != 409 or refused.get("reason") != "capacity":
        failures.append(f"a plan for more fleets than serve is quoted: {status} {refused}")
    status, quote = team("POST", "/plans/quote", CHOICE)
    rows, ko = dict(map(tuple, quote.get("rows", {}).get("en", []))), dict(map(tuple, quote.get("rows", {}).get("ko", [])))
    if (status != 200 or quote["hours"] != 6579 or quote["total"] != "98685.00" or quote["withVat"] != "108553.50" or rows.get("Hours in the Term (clause 17.3)") != "6,579"
            or rows.get("Payment schedule (clause 20)") != "Gated: Gate 1 by 13 January 2027, Gate 2 by 12 February 2027, Gate 3 by 14 March 2027"
            or rows.get("Term (clause 16.3)") != "24 months, ending on 14 December 2028" or rows.get("Time zone (clause 17.1)") != "Asia/Seoul"
            or rows.get("Office hours (clause 17.1)") != "09:00 to 18:00, local time in Asia/Seoul, 9 hours" or rows.get("Days (clause 17.2)") != "Every day"
            or "Windows (clause 17.1)" in rows or not rows.get("Published settings (clause 9.3)", "").endswith("maintenance windows (clause 4.6) 00:00–08:00, 08:00–16:00 and 16:00–24:00 KST")
            or not rows.get("Customer", "").startswith("Plan Co Ltd, a company organised under the laws of Delaware, United States, registration number 7781234")
            or cfg.terms("agreement-ko.md") not in rows.get("Customer Agreement (clause 9.1)", "") or cfg.terms("market-rules-en.md") not in rows.get("Customer Agreement (clause 9.1)", "")
            or not rows.get("Customer Agreement (clause 9.1)", "").startswith("Version of ") or rows.get("Supplier notice address and email (clause 11.7)") != "1 Test-ro, Yongsan-gu, Seoul; hyor@sparkpipe.ai"
            or ko.get("플랜 기간의 시간(제17.3조)") != "6,579시간" or ko.get("업무 시간(제17.1조)") != "Asia/Seoul 현지 시각 09:00부터 18:00까지(9시간)" or ko.get("표준시간대(제17.1조)") != "Asia/Seoul"
            or ko.get("공급자 통지 주소 및 이메일(제11.7조)") != "서울특별시 용산구 시험로 1, hyor@sparkpipe.ai"
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
    if status != 200 or placed.get("plan") != quote["plan"] or placed.get("hours") != 6579 or line["hash"] != quote["hash"] or line["text"] != quote["text"] or line["accepted"]["ip"] != "127.0.0.1" or not signed or team("GET", "/plans/key")[1].get("key") != signature.get("key"):
        failures.append(f"placing a Plan Order does not record the accepted text with our signature of its hash: {status} {placed} {signed}")
        return
    told = next((m for rcpt, m in mailbox.mail if "cto@plan.example" in rcpt and m["Subject"] == f"Plan Order {quote['plan']}: your SparkPipe plan"), None)
    if not told or quote["hash"] not in told.get_content() or "| Hours in the Term (clause 17.3) | 6,579 |" not in told.get_content() or "| 플랜 기간의 시간(제17.3조) | 6,579시간 |" not in told.get_content():
        failures.append("the customer is not emailed the Plan Order it accepted")
    if not any(cfg.notify in rcpt and m["Subject"].startswith("New Plan Order: Plan Co Ltd") for rcpt, m in mailbox.mail):
        failures.append("we are not told of a new Plan Order")
    if not app.market.on_sale(app.market.now(), cfg.pricing()) < sale - 2000:
        failures.append(f"an unpaid Plan Order does not take its hours off the HC on sale: {sale}")
    if team("POST", "/plans/order", dict(order, agree=True))[0] != 409:
        failures.append("a Plan Order is placed twice")
    status, refused, _ = place(team, dict(CHOICE, office=OFFICES["americas"]))
    if status != 409 or refused.get("reason") != "unpaid":
        failures.append(f"an account places a second plan while its first is unpaid: {status} {refused}")
    status, refused = other("POST", "/plans/quote", dict(MONTHLY, office=OFFICES["asia"]))
    if status != 409 or refused.get("reason") != "capacity":
        failures.append(f"a plan is quoted on capacity another Plan Order keeps: {status} {refused}")
    status, refused = other("POST", "/plans/quote", dict(CHOICE, office=OFFICES["full"], lock=True, customer=MONTHLY["customer"]))
    if status != 409:
        failures.append(f"a fleet lock is quoted on a fleet another Plan Order needs: {status} {refused}")
    status, view = team("GET", "/plan/" + quote["plan"])
    if status != 200 or len(view.get("tranches", [])) != 25 or view.get("rows") != quote["rows"] or view.get("signature", {}).get("key") != signature["key"] or view.get("heldUntil") != "2028-12-14":
        failures.append(f"the Plan Order view: {status} {view.get('tranches', [])[:1]} {view.get('heldUntil')}")
    if other("GET", "/plan/" + quote["plan"])[0] != 404:
        failures.append("another account reads a Plan Order")
    start = NOW[0]
    NOW[0] = datetime(2026, 12, 16, 3, 0, 0, tzinfo=timezone.utc)
    if quote["plan"] not in kept(app):
        failures.append("an unpaid gated plan loses its capacity in Phase A")
    NOW[0] = start
    pay_plan(app, fake, team, quote["plan"], "22680.00")
    holder = app.book.emails["cto@plan.example"]
    granted = [app.book.market.tranches[(holder, month)] for month in ("2026-12", "2027-01", "2027-05", "2027-06")]
    if granted != [153, 279, 279, 0]:
        failures.append(f"paying Gate 1 does not issue Tranches 1 to 6: {granted}")
    status, refused = team("POST", "/plans/stop", {"plan": quote["plan"]})
    if status != 409 or refused.get("reason") != "fixed":
        failures.append(f"a plan with a fixed term can be stopped: {status} {refused}")
    prices = work / "site" / "pricing.js"
    text = prices.read_text()
    prices.write_text(text.replace('{ "name": "Fleet 2", "kind": "M5U", "state": "building"', '{ "name": "Fleet 2", "kind": "M5U", "state": "serving"'))
    status, held = buyer("POST", "/market/reserve", {"fleet": "F1", "start": "2026-12-02T01:00Z", "model": "kimi-k3", "max": "2"})
    status, moved = team("POST", "/plans/quote", dict(CHOICE, start="2026-12-01", office=OFFICES["full"], lock=True))
    prices.write_text(text)
    if not held.get("reservation") or status != 200 or moved.get("lock") != ["F2"]:
        failures.append(f"a fleet lock is not moved to another serving fleet when the first is reserved in its term: {held} {status} {moved}")
    monthly_flow(failures, cfg, app, fake, team, other, mailbox)


def monthly_flow(failures, cfg, app, fake, team, other, mailbox):
    pricing = cfg.pricing()
    status, placed, quote = place(other, MONTHLY)
    rows, ko = dict(map(tuple, quote.get("rows", {}).get("en", []))), dict(map(tuple, quote.get("rows", {}).get("ko", [])))
    if (status != 200 or placed.get("payment") != "monthly" or placed.get("end") is not None or quote.get("hours") != 104 or quote.get("total") != "1820.00"
            or rows.get("Customer") != "Jane Kim, an individual, of 3 Side St, Busan" or rows.get("Term (clause 16.3)") != "Monthly, with no fixed end"
            or rows.get("Payment schedule (clause 20)") != "Monthly: each Tranche before it starts, Tranche 1 before the Start Date"
            or not rows.get("Tranche 1 (clause 17.3)", "").startswith("15 December 2026 to 31 December 2026: 104 Plan hours, USD 1,820.00, exclusive of VAT")
            or rows.get("Office hours (clause 17.1)") != "09:00 to 17:00, local time in America/New_York, 8 hours; on the Start Date, the Plan hours are 23:00–07:00 KST"
            or rows.get("Days (clause 17.2)") != "Weekdays" or not rows.get("Hour Price (clause 16.4)", "").endswith("fixed for as long as the Plan runs")
            or "Hours in the Term (clause 17.3)" in rows or ko.get("플랜 기간(제16.3조)") != "월간, 종료일 없음" or not ko.get("트랜치 1(제17.3조)", "").startswith("2026년 12월 15일부터 2026년 12월 31일까지: 플랜 시간 104시간")):
        failures.append(f"a Monthly Plan Order: {status} {placed} {quote.get('hours')} {rows} {ko}")
        return
    ident, account = placed["plan"], app.book.emails["dev@solo.example"]
    state = next(p for p in other("GET", "/account")[1]["plans"] if p["plan"] == ident)
    if state["nextGate"] != {"name": "December 2026", "due": "2026-12-15", "amount": "1820.00", "hours": 104, "from": "2026-12-15", "to": "2026-12-31"} or state["heldUntil"] != "2027-01-31" or ident not in kept(app):
        failures.append(f"an unpaid Monthly plan is not due Tranche 1 by its Start Date, or does not keep Tranches 1 and 2: {state}")
    late = plan("0099", "cto@plan.example", "newyork", "weekdays", months=12, start="2027-06-01")
    try:
        app.record_signed(pricing, late, NOW[0])
        failures.append("a Plan Order takes the hours a running Monthly plan keeps in a later month")
    except sp.Refusal:
        pass
    if other("POST", "/order", {"product": "plan", "plan": ident, "amount": "1820.01", "coin": "USDC.BASE"})[0] != 409:
        failures.append("a Monthly plan takes a payment beyond its next Tranche")
    pay_plan(app, fake, other, ident, "900.00")
    if app.book.market.tranches[(account, "2026-12")] != 0:
        failures.append("a Tranche paid in part issues Plan hours")
    pay_plan(app, fake, other, ident, "920.00")
    state = next(p for p in other("GET", "/account")[1]["plans"] if p["plan"] == ident)
    if app.book.market.tranches[(account, "2026-12")] != 104 or state["paidThrough"] != "2026-12-31" or state["nextGate"]["name"] != "January 2027" or state["nextGate"]["amount"] != "3062.50" or state["nextGate"]["due"] != "2027-01-01":
        failures.append(f"paying Tranche 1 in full does not issue its hours and ask for January before it starts: {app.book.market.tranches[(account, '2026-12')]} {state}")
    start = NOW[0]
    NOW[0] = datetime(2027, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
    sign_in(other, mailbox, "dev@solo.example")
    state = next(p for p in other("GET", "/account")[1]["plans"] if p["plan"] == ident)
    if state["ended"] or state["heldUntil"] != "2027-01-31" or ident not in kept(app):
        failures.append(f"a Monthly plan ends before the end of the first day of its unpaid Tranche: {state}")
    NOW[0] = datetime(2027, 1, 2, 3, 0, 0, tzinfo=timezone.utc)
    state = next(p for p in other("GET", "/account")[1]["plans"] if p["plan"] == ident)
    if "January 2027 was not paid by its first day, so the plan ended on 2026-12-31 with nothing owed" not in (state["ended"] or "") or state["heldUntil"] is not None or ident in kept(app) or state["nextGate"] is not None:
        failures.append(f"a Monthly plan whose next Tranche is unpaid on its first day does not end after its last paid Tranche, freeing the next month: {state} {kept(app)}")
    status, _ = other("POST", "/order", {"product": "plan", "plan": ident, "amount": "3062.50", "coin": "USDC.BASE"})
    if status != 409:
        failures.append("a Monthly plan that ended takes a payment")
    try:
        app.record_signed(pricing, late, NOW[0])
    except sp.Refusal as error:
        failures.append(f"the hours of a Monthly plan that ended stay kept: {error}")
    if "SP-PLAN-0099" in app.book.plans:
        app.end_plan("SP-PLAN-0099", sp.local_day(NOW[0]), "test")
    sign_in(team, mailbox, "cto@plan.example")
    status, placed, quote = place(team, dict(MONTHLY, start="2027-01-15", customer=CHOICE["customer"]))
    if status != 200 or quote.get("hours") != 95:
        failures.append(f"a second Monthly plan: {status} {placed} {quote.get('hours')}")
        NOW[0] = start
        return
    second = placed["plan"]
    pay_plan(app, fake, team, second, "1662.50")
    status, stopped = team("POST", "/plans/stop", {"plan": second})
    if status != 200 or not stopped.get("stopped") or stopped.get("end") != "2027-01-31" or stopped.get("heldUntil") != "2027-01-31" or stopped.get("nextGate") is not None or stopped.get("ended"):
        failures.append(f"stopping a Monthly plan does not end it after its paid month: {status} {stopped}")
    if [found.ends for found in app.commitments(NOW[0]) if found.plan == second] != [sp.kst_start(date(2027, 2, 1))]:
        failures.append("a stopped Monthly plan keeps capacity after its last paid Tranche")
    status, refused = team("POST", "/order", {"product": "plan", "plan": second, "amount": "100.00", "coin": "USDC.BASE"})
    if status != 409 or refused.get("reason") != "stopped" or team("POST", "/plans/stop", {"plan": second})[0] != 409 or other("POST", "/plans/stop", {"plan": second})[0] != 404:
        failures.append(f"a stopped plan takes a payment, is stopped twice or by another account: {status} {refused}")
    app.mails.join()
    if not any("cto@plan.example" in rcpt and m["Subject"] == f"Plan {second} is stopped" for rcpt, m in mailbox.mail):
        failures.append("stopping a plan is not confirmed by email")
    NOW[0] = datetime(2027, 2, 2, 3, 0, 0, tzinfo=timezone.utc)
    sign_in(team, mailbox, "cto@plan.example")
    state = next(p for p in team("GET", "/account")[1]["plans"] if p["plan"] == second)
    if "stopped and ended on 2027-01-31" not in (state["ended"] or "") or second in kept(app):
        failures.append(f"a stopped Monthly plan does not end after its paid month: {state}")
    NOW[0] = start


def prepayment_flow(failures, cfg, app, fake, buyer, mailbox, port):
    plans = [
        plan("0001", "ops@acme.example", "day", start="2026-11-01"),
        plan("0002", "OPS@acme.example", "europe", fleets=2, start="2026-11-01"),
        plan("0003", "else@other.example", "americas", start="2026-11-01"),
        plan("0004", "ops@acme.example", "asia", months=12, start="2026-11-10", payment="upfront"),
    ]
    set_plans(app, plans)
    hours = sp.tranches(cfg.pricing(), plans[0])
    found = buyer("GET", "/account")[1]["plans"]
    if [a["plan"] for a in found] != ["SP-PLAN-0001", "SP-PLAN-0002", "SP-PLAN-0004"]:
        failures.append(f"the account lists other customers' plans: {found}")
        return
    if found[1]["hours"] != 2 * sum(sp.tranches(cfg.pricing(), dict(plans[1], fleets=1))):
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
    pay_plan(app, fake, buyer, "SP-PLAN-0001", "4000.00")
    if app.book.market.tranches[(account, "2026-11")] != 270 - 12 or app.book.market.tranches[(account, "2026-12")] != 3:
        failures.append(f"a payment does not issue the rest of Tranche 1 less the 12 Plan hours that occurred before it was paid for, and the start of Tranche 2: {app.book.market.tranches[(account, '2026-11')]} {app.book.market.tranches[(account, '2026-12')]}")
    NOW[0] = datetime(2026, 12, 1, 3, 0, 0, tzinfo=timezone.utc)
    state = buyer("GET", "/account")[1]["plans"]
    status, _ = buyer("POST", "/order", dict(pay, amount="100.00"))
    if "Gate 1 was missed" not in (state[0]["ended"] or "") or status != 409:
        failures.append(f"a missed gate does not stop prepayment: {state[0]} {status}")
    kept = [found.plan for found in app.commitments(NOW[0])]
    if state[0]["heldUntil"] != "2026-12-31" or "SP-PLAN-0001" not in kept or "SP-PLAN-0002" in kept:
        failures.append(f"after a missed gate, a plan keeps capacity beyond its last paid Tranche, or a plan that ended in Phase A keeps any: {state[0]['heldUntil']} {kept}")
    app.end_plan("SP-PLAN-0001", date(2026, 12, 10), "ended under clause 10.2")
    if buyer("GET", "/account")[1]["plans"][0]["heldUntil"] != "2026-12-09":
        failures.append("a Plan Order we end keeps capacity after its last day")
    for bad in (plan("0005", "ops@acme.example", "day", fleets=2, start="2026-12-15"), plan("0006", "nobody@acme.example", "europe"), plan("0001", "ops@acme.example", "europe")):
        try:
            app.record_signed(cfg.pricing(), bad, NOW[0])
            failures.append(f"a signed Plan Order is recorded without capacity, an account or an unused number: {bad['id']}")
        except sp.Refusal:
            pass
    signed = app.record_signed(cfg.pricing(), plan("0007", "ops@acme.example", "europe", months=12, start="2027-01-01", payment="upfront"), NOW[0])
    if app.book.plans.get("SP-PLAN-0007", {}).get("source") != "signed" or signed["account"] != account:
        failures.append(f"a signed Plan Order that fits is not recorded: {signed}")
    pay_plan(app, fake, buyer, "SP-PLAN-0007", "6300.00")
    if app.book.market.tranches[(account, "2027-01")] != 0:
        failures.append("an upfront plan paid in part issues Tranche hours")
    pay_plan(app, fake, buyer, "SP-PLAN-0007", "37500.00")
    if [app.book.market.tranches[(account, month)] for month in ("2027-01", "2027-02", "2027-03")] != [248, 224, 248]:
        failures.append(f"an upfront plan paid in full does not issue every Tranche: {[app.book.market.tranches[(account, month)] for month in ('2027-01', '2027-02', '2027-03')]}")
    paper = app.record_signed(cfg.pricing(), plan("0009", "ops@acme.example", "asia", "weekdays", start="2027-01-01", payment="monthly"), NOW[0])
    shown = next((p for p in buyer("GET", "/account")[1]["plans"] if p["plan"] == "SP-PLAN-0009"), {})
    if paper.get("months") is not None or shown.get("nextGate", {}).get("due") != "2027-01-01" or shown.get("nextGate", {}).get("hours") != 8 * 21 or "Monthly" not in shown.get("line", ""):
        failures.append(f"a signed Monthly Plan Order is not recorded with Tranche 1 due by its Start Date: {paper} {shown}")
    app.market.credit(account, app.market.on_sale(app.market.now(), cfg.pricing()), "cover test")
    try:
        app.record_signed(cfg.pricing(), plan("0008", "ops@acme.example", "americas", months=12, start="2027-01-01", payment="upfront"), NOW[0])
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
    print("pay: plan hours, Tranches and gates match the Customer Agreement for office hours in Seoul, New York, Berlin and Kolkata, across midnight and daylight-saving changes, Weekdays by the local date, Full days, terms and fleet counts, and invalid plans fail; email codes, sessions, terms acceptance, USDC and USDT invoices on the chosen network, payments in another stablecoin, signed webhooks, receipts, HC orders and part payments, reconciliation, gates, self-serve Plan Orders (the bilingual Order Form, the accepted hash and our Ed25519 signature, capacity, HC cover and fleet-lock checks, the removed 3-month term refused, holds that lapse unpaid or end at the last paid Tranche, Tranche hours issued on payment less hours that already occurred), Monthly plans (each Tranche paid before it starts, ending after the last paid Tranche when the next is unpaid on its first day, the next month's capacity freed, Stop after this month, a running Monthly plan's later months kept from other Plan Orders), quote requests and the hash-chained ledger behave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
