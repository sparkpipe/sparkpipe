#!/usr/bin/env python3

import base64
import hashlib
import hmac
import http.server
import json
import random
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import eth_signature as es
import sparkpipe_pay as sp

ORACLE = [
    ("hello", "0xa5d58782075bdf09490159d634d1aae66a8f6777c7247d2f233e9511cfd7c64c34f288cdbcea5370e4863fdbe9f4d86654c2ba1d86589e9ebb64494c649008591b"),
    ("SparkPipe order SPX\nI buy 4 fleet-hours.\n한국어 본문 sha256 ab12", "0x818cda2ef8dc52d95920105b48dfa9284dc3e21b972d2a0b4afa1d52de61cb195dd379231b454cea57959ea50a317cccef2ccf99e04630ba1d00dbc274e6aaaa1b"),
    ("", "0x8a68b4e66cd2b575338e16069d7b65f6f67c7ceae8945dccf8cb7bdb06278d933dd9c888f3444ca4698464079a067ad3cfffe96d493b8ecf56885169d0fdfe7d1b"),
]
ORACLE_ADDRESS = "0x2c7536e3605d9c16a7a3d7b1898e529396a65c23"
SECRET = b"test-client-secret"
CLIENT = "test-client-id"
NOW = [datetime(2026, 11, 2, 3, 0, 0, tzinfo=timezone.utc)]


def personal_sign(key, text):
    z = int.from_bytes(es.personal_digest(text), "big")
    k = int.from_bytes(hashlib.sha256(key.to_bytes(32, "big") + text.encode()).digest(), "big") % es.N or 1
    point = es.multiply(es.G, k)
    r = point[0] % es.N
    s = pow(k, -1, es.N) * (z + r * key) % es.N
    parity = point[1] & 1
    if s > es.N // 2:
        s, parity = es.N - s, parity ^ 1
    return "0x" + r.to_bytes(32, "big").hex() + s.to_bytes(32, "big").hex() + bytes([27 + parity]).hex()


def docs_signature(method, url, stamp, body):
    message = "﻿" + method + url + CLIENT + stamp
    return base64.b64encode(hmac.new(SECRET, message.encode("utf-8") + body, hashlib.sha256).digest()).decode()


def check_keccak(failures):
    if es.keccak256(b"").hex() != "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470":
        failures.append("keccak256 of the empty string")
    if es.keccak256(b"abc").hex() != "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45":
        failures.append("keccak256 of abc")
    rng = random.Random(3)
    for size in list(range(0, 300)) + [1000]:
        data = bytes(rng.getrandbits(8) for _ in range(size))
        if es._sponge(data, 0x06) != hashlib.sha3_256(data).digest():
            failures.append(f"the sponge with SHA3 padding differs from hashlib at {size} bytes")
            break


def check_recover(failures):
    for text, signature in ORACLE:
        if es.personal_recover(text, signature) != ORACLE_ADDRESS:
            failures.append(f"eth-account signature over {text[:20]!r} does not recover its address")
    text, signature = ORACLE[0]
    if es.personal_recover(text + "!", signature) == ORACLE_ADDRESS:
        failures.append("a changed message still recovers the signer")
    raw = bytes.fromhex(signature[2:])
    high = (es.N - int.from_bytes(raw[32:64], "big")).to_bytes(32, "big")
    flipped = "0x" + (raw[:32] + high + bytes([raw[64] ^ 1])).hex()
    if es.personal_recover(text, flipped) is not None:
        failures.append("a high-s signature is accepted")
    for bad in ("", "0x00", signature[:-2], signature[:-2] + "1d", None):
        if es.personal_recover(text, bad) is not None:
            failures.append(f"a malformed signature {bad!r} is accepted")
    key = 0x1234567890ABCDEF
    if es.personal_recover("order", personal_sign(key, "order")) != es.address_of(es.multiply(es.G, key)):
        failures.append("the test signer does not round-trip")


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
        data = json.loads(body)
        key = f"inv{len(self.server.invoices) + 1}"
        self.server.invoices[key] = {"status": "unpaid", "total": data["amount"]["total"], "body": data}
        self.answer(200, {"invoices": [{"id": key, "link": f"https://pay.example/i/{key}", "checkoutLink": f"https://pay.example/c/{key}"}]})

    def do_GET(self):
        if not self.signed(b""):
            return self.answer(401, {"error": "signature"})
        key = self.path.rsplit("/", 1)[1]
        invoice = self.server.invoices[key]
        self.answer(200, {"id": key, "status": invoice["status"], "amount": {"breakdown": {}, "total": invoice["total"]}, "currency": {"id": "5057", "symbol": "USD"}})


def request(port, method, path, body=None, headers=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=dict({"Content-Type": "application/json"}, **(headers or {})))
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def webhook(port, cfg, invoice, kind="InvoiceCompleted", stamp=None, tamper=False):
    stamp = stamp or NOW[0].strftime("%Y-%m-%dT%H:%M:%S")
    raw = json.dumps({"id": invoice, "type": kind, "invoice": {"id": invoice, "state": kind[7:]}}).encode()
    signature = docs_signature("POST", cfg.webhook_url, stamp, raw)
    headers = {"X-CoinPayments-Client": CLIENT, "X-CoinPayments-Timestamp": stamp, "X-CoinPayments-Signature": signature}
    return request(port, "POST", "/coinpayments", headers=headers, raw=raw + (b" " if tamper else b""))


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
        fake.invoices = {}
        threading.Thread(target=fake.serve_forever, daemon=True).start()
        cfg = sp.Config({
            "SPARKPIPE_PAY_DATA": str(work / "data"), "SPARKPIPE_PAY_SITE": str(work / "site"),
            "COINPAYMENTS_API": f"http://127.0.0.1:{fake.server_address[1]}/api/v2",
            "COINPAYMENTS_CLIENT_ID": CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(work / "secret"),
        })
        sp.clock = lambda: NOW[0]
        server = sp.start(cfg, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port, app = server.server_address[1], server.app
        buyer, other = 0xB0B, 0xC0FFEE
        account = es.address_of(es.multiply(es.G, buyer))
        hourly_flow(failures, cfg, port, app, fake, buyer, other, account)
        prepayment_flow(failures, cfg, port, app, fake, buyer, account)
        book = sp.Book(cfg.path("ledger.jsonl"))
        if book.count != app.book.count or book.head != app.book.head:
            failures.append("a reloaded ledger differs from the live one")
        second = sp.Book(cfg.path("ledger.jsonl"))
        second.append("note", {"by": "second writer"})
        app.book.append("note", {"by": "service"})
        if sp.Book(cfg.path("ledger.jsonl")).count != second.count + 1:
            failures.append("two writers break the ledger chain")
        lines = open(cfg.path("ledger.jsonl"), encoding="utf-8").read().splitlines()
        lines[0] = lines[0].replace('"hours":4', '"hours":40', 1)
        (work / "tampered.jsonl").write_text("\n".join(lines) + "\n")
        try:
            sp.Book(str(work / "tampered.jsonl"))
            failures.append("a tampered ledger loads")
        except ValueError:
            pass
        server.shutdown()
        fake.shutdown()
    finally:
        shutil.rmtree(work)


def hourly_flow(failures, cfg, port, app, fake, buyer, other, account):
    status, quote = request(port, "POST", "/quote", {"product": "hourly", "account": account, "company": "Acme 주식회사", "email": "ops@acme.example", "hours": 4, "start": "2026-12-01"})
    if status != 200 or (quote.get("net"), quote.get("vat"), quote.get("total"), quote.get("last")) != ("80.00", "8.00", "88.00", "2027-11-30"):
        failures.append(f"hourly quote: {status} {quote}")
        return
    if cfg.terms("hourly-ko.md") not in quote["message"] or "Korean, which prevails" not in quote["message"]:
        failures.append("the order message does not carry the Korean terms hash")
    status, early = request(port, "POST", "/quote", {"product": "hourly", "account": account, "company": "Acme", "email": "ops@acme.example", "hours": 1, "start": "2026-11-30"})
    if status != 400:
        failures.append(f"hours starting before the first sale date are quoted: {status}")
    status, _ = request(port, "POST", "/order", {"order": quote["order"], "signature": personal_sign(other, quote["message"])})
    if status != 403:
        failures.append(f"an order signed by another key is accepted: {status}")
    status, placed = request(port, "POST", "/order", {"order": quote["order"], "signature": personal_sign(buyer, quote["message"])})
    if status != 200 or not placed.get("checkout", "").startswith("https://pay.example/c/"):
        failures.append(f"hourly order: {status} {placed}")
        return
    status, again = request(port, "POST", "/order", {"order": quote["order"], "signature": personal_sign(buyer, quote["message"])})
    if status != 410:
        failures.append(f"a quote can be ordered twice: {status}")
    invoice = fake.invoices["inv1"]["body"]
    expected = {"currency": "5057", "invoiceId": quote["order"], "amount": {"breakdown": {"subtotal": "80.00", "taxTotal": "8.00"}, "total": "88.00"}}
    if any(invoice.get(key) != value for key, value in expected.items()):
        failures.append(f"invoice body: {invoice}")
    if invoice["items"][0]["quantity"] != {"value": 4, "type": "hours"} or invoice["webhooks"][0]["notificationsUrl"] != "https://sparkpipe.ai/pay/api/coinpayments":
        failures.append(f"invoice items or webhooks: {invoice['items']} {invoice['webhooks']}")
    status, shown = request(port, "GET", "/order/" + quote["order"])
    if status != 200 or shown["state"] != "unpaid" or shown["checkout"] != placed["checkout"]:
        failures.append(f"unpaid order status: {status} {shown}")
    status, _ = request(port, "POST", "/quote", {"product": "hourly", "account": account, "company": "Acme", "email": "ops@acme.example", "hours": 165, "start": "2026-12-01"})
    if status != 409:
        failures.append(f"an unpaid order does not count toward the 168-hour limit: {status}")
    if webhook(port, cfg, "inv1", tamper=True)[0] != 401 or webhook(port, cfg, "inv1", stamp="2026-11-02T02:50:00")[0] != 401:
        failures.append("a tampered or stale webhook is accepted")
    fake.invoices["inv1"]["status"] = "paid"
    webhook(port, cfg, "inv1", "InvoicePaid")
    app.jobs.join()
    if request(port, "GET", "/order/" + quote["order"])[1]["state"] != "paid":
        failures.append("a paid invoice does not move the order to paid")
    fake.invoices["inv1"]["status"] = "completed"
    if webhook(port, cfg, "inv1")[0] != 200:
        failures.append("a signed webhook is refused")
    app.jobs.join()
    webhook(port, cfg, "inv1")
    app.jobs.join()
    shown = request(port, "GET", "/order/" + quote["order"])[1]
    if shown["state"] != "completed" or not shown["issued"] or ledger_types(cfg).count("hours_issued") != 1:
        failures.append(f"completed order: {shown} {ledger_types(cfg)}")
    status, _ = request(port, "POST", "/quote", {"product": "hourly", "account": account, "company": "Acme", "email": "ops@acme.example", "hours": 165, "start": "2026-12-01"})
    status_ok, room = request(port, "POST", "/quote", {"product": "hourly", "account": account, "company": "Acme", "email": "ops@acme.example", "hours": 164, "start": "2026-12-01"})
    if status != 409 or status_ok != 200:
        failures.append(f"issued hours and the holding limit: {status} {status_ok}")
    request(port, "POST", "/order", {"order": room["order"], "signature": personal_sign(buyer, room["message"])})
    fake.invoices["inv2"]["status"] = "completed"
    fake.invoices["inv2"]["total"] = "1.00"
    webhook(port, cfg, "inv2")
    app.jobs.join()
    shown = request(port, "GET", "/order/" + room["order"])[1]
    if shown["state"] != "mismatch" or shown["issued"]:
        failures.append(f"an underpaid invoice issues hours: {shown}")


def prepayment_flow(failures, cfg, port, app, fake, buyer, account):
    agreements = [
        {"id": "SP2Y-TEST-0001", "account": account, "company": "Acme", "email": "ops@acme.example", "signed": "2026-11-01", "hourPrice": "15.00"},
        {"id": "SP2Y-TEST-0002", "account": account, "company": "Acme", "email": "ops@acme.example", "signed": "2026-11-01", "hourPrice": "15.00"},
    ]
    Path(cfg.path("agreements.json")).write_text(json.dumps(agreements))
    plan = sp.tranches(date(2026, 11, 1), 24)
    status, state = request(port, "POST", "/agreement", {"agreement": "SP2Y-TEST-0001", "account": account})
    if status != 200 or state["hours"] != sum(plan) or state["nextGate"] != {"name": "Gate 1", "due": "2026-11-30", "amount": f"{15 * sum(plan[:6])}.00"}:
        failures.append(f"agreement standing: {status} {state}")
    if request(port, "POST", "/agreement", {"agreement": "SP2Y-TEST-0001", "account": "0x" + "1" * 40})[0] != 404:
        failures.append("another account can read an agreement")
    if request(port, "POST", "/quote", {"product": "two-year", "account": account, "agreement": "SP2Y-TEST-0001", "amount": "999999999.00"})[0] != 409:
        failures.append("a prepayment above the remaining price is quoted")
    status, quote = request(port, "POST", "/quote", {"product": "two-year", "account": account, "agreement": "SP2Y-TEST-0001", "amount": "100.00"})
    if status != 200 or (quote["hours"], quote["vat"], quote["total"]) != ("6.6666", "10.00", "110.00"):
        failures.append(f"prepayment quote: {status} {quote}")
        return
    status, placed = request(port, "POST", "/order", {"order": quote["order"], "signature": personal_sign(buyer, quote["message"])})
    key = max(fake.invoices, key=lambda k: int(k[3:]))
    if status != 200 or fake.invoices[key]["body"]["items"][0]["quantity"] != {"value": 1, "type": "quantity"}:
        failures.append(f"prepayment order: {status} {placed}")
        return
    fake.invoices[key]["status"] = "completed"
    app.reconcile()
    app.jobs.join()
    state = request(port, "POST", "/agreement", {"agreement": "SP2Y-TEST-0001", "account": account})[1]
    if state["paid"] != "100.00" or not request(port, "GET", "/order/" + quote["order"])[1]["issued"]:
        failures.append(f"reconciling a missed webhook does not issue the prepayment: {state}")
    NOW[0] = datetime(2026, 12, 1, 3, 0, 0, tzinfo=timezone.utc)
    state = request(port, "POST", "/agreement", {"agreement": "SP2Y-TEST-0001", "account": account})[1]
    status, refused = request(port, "POST", "/quote", {"product": "two-year", "account": account, "agreement": "SP2Y-TEST-0001", "amount": "100.00"})
    if "Gate 1 was missed" not in (state["ended"] or "") or status != 409:
        failures.append(f"a missed gate does not stop prepayment: {state} {status} {refused}")
    state = request(port, "POST", "/agreement", {"agreement": "SP2Y-TEST-0002", "account": account})[1]
    if "Phase A ended" not in (state["ended"] or ""):
        failures.append(f"an unpaid agreement survives Phase A: {state}")


def main():
    failures = []
    check_keccak(failures)
    check_recover(failures)
    check_dates(failures)
    check_service(failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("pay: keccak and signature recovery match eth-account, tranches match the agreement, quotes, signed orders, CoinPayments invoices, signed webhooks, reconciliation, the holding limit, gates and the hash-chained ledger behave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
