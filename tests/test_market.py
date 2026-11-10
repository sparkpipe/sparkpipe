#!/usr/bin/env python3

import base64
import contextlib
import hashlib
import hmac
import http.client
import http.server
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import sparkpipe_market as market
import sparkpipe_pay as sp
import test_pay as tp

NOW = [datetime(2026, 11, 28, 3, 0, 0, tzinfo=timezone.utc)]
POOL_KEY = "sk-pool-test"


def kst(*parts):
    return datetime(*parts, tzinfo=market.KST).astimezone(timezone.utc)


def slot(moment):
    return market.slot_text(moment)


def refused(call):
    try:
        call()
    except market.Refusal as error:
        return error.status, error.reason
    return None


def site(work, serving=True):
    tp.site_copy(work)
    path = work / "site" / "pricing.js"
    pricing = tp.read_pricing(path)
    if serving:
        for fleet in pricing["fleets"]:
            fleet["state"] = "serving"
        pricing["fleets"] += [dict(pricing["fleets"][-1], id=f"F{n}", name=f"Fleet {n}") for n in (4, 5)]
    tp.write_pricing(path, pricing)
    (work / "pool_key").write_text(POOL_KEY + "\n")


def live(work, name, mailbox, upstream=None, extra=None):
    env = {"SPARKPIPE_GATEWAY_CAPACITY": "100000", "SPARKPIPE_PAY_DATA": str(work / name), "SPARKPIPE_PAY_SITE": str(work / "site"), "COINPAYMENTS_CLIENT_ID": tp.CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(work / "secret"),
           "SPARKPIPE_PAY_SMTP": f"127.0.0.1:{mailbox.server_address[1]}", "SPARKPIPE_POOL_KEY_FILE": str(work / "pool_key"), "SPARKPIPE_PAY_HEALTH_SECONDS": "0"}
    env.update(extra or {})
    if upstream:
        env["SPARKPIPE_POOL_UPSTREAM"] = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    sp.clock = lambda: NOW[0]
    server = sp.start(sp.Config(env), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.app, server.server_address[1]


def person(app, name, credits=0):
    account = sp.new_id("AC")
    app.book.append("account", {"account": account, "email": f"{name}@example.com", "company": name.title(), "ip": "127.0.0.1", "agent": "test"})
    app.accept(account, "127.0.0.1", "test")
    if credits:
        app.market.credit(account, Decimal(credits), "test")
    return account


def client(app, port, account):
    found = tp.Client(port, origin="https://sparkpipe.ai")
    found.cookie = f"{sp.COOKIE}={sp.session_token(app.key, account, 10 ** 10)}"
    return found


def call(port, method, path, body=None, headers=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(), method=method, headers=dict({"Content-Type": "application/json"}, **(headers or {})))
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.headers.get("Content-Type"), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type"), error.read()


def chat(port, key, body, headers=None):
    status, _, raw = call(port, "POST", "/v1/chat/completions", dict({"model": "kimi-k3", "messages": [{"role": "user", "content": "hello"}]}, **body), dict({"Authorization": f"Bearer {key}"}, **(headers or {})))
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode()


def feed(port, fleet, first, last, key=POOL_KEY):
    status, _, raw = call(port, "GET", f"/market/fleets/{fleet}/holders?from={first}&to={last}", headers={"Authorization": f"Bearer {key}"})
    return status, json.loads(raw)


def balance(app, account):
    return app.book.market.balance(account, market.stamp(app.market.now()))


class FakeUpstream(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def record(self, body):
        self.server.calls.append({"method": self.command, "path": self.path, "items": list(self.headers.items()), "body": body if len(body) < 1 << 20 else (len(body), hashlib.sha256(body).hexdigest())})

    def answer(self, status, data):
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("X-Hub", "1")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def events(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()
                time.sleep(0.01)
        except OSError:
            self.close_connection = True

    def generic(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path in ("/health", "/v1/sparkpipe/status") and self.command == "GET":
            getattr(self.server, "probes", []).append(self.path)
            if self.path == "/v1/sparkpipe/status":
                control = getattr(self.server, "control", None)
                return self.answer(*control) if control else self.answer(404, {"error": {"type": "not_found", "message": "this front end has no model control"}})
            models = getattr(self.server, "models", None) or {"kimi-k3": 200 if getattr(self.server, "healthy", True) else 503}
            ok = all(code == 200 for code in models.values())
            return self.answer(200 if ok else 503, {"ok": ok, "models": {name: {"status": code} for name, code in models.items()}})
        self.record(body)
        fleet = self.headers.get("X-SparkPipe-Fleet")
        if self.path == "/v1/chat/completions":
            data = json.loads(body)
            self.server.seen.append(({name: self.headers.get(name) for name in ("Authorization", "X-SparkPipe-Fleet", "X-SparkPipe-Account")}, self.headers.get_all("X-SparkPipe-Account"), data))
            if data.get("stream"):
                chunks = ({"choices": [{"index": 0, "delta": {"content": "hi"}}]}, {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 1}, "sparkpipe": {"fleet": fleet}})
                return self.events([b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks] + [b"data: [DONE]\n\n"])
            return self.answer(200, {"model": data.get("model"), "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 5}, "sparkpipe": {"fleet": fleet}})
        if self.path == "/v1/models/resident" and self.command == "PUT" and self.headers.get("X-SparkPipe-Allow-Swap") != "1":
            return self.answer(409, {"type": "error", "error": {"type": "model_not_loaded", "message": "kimi-k3 is resident"}})
        if self.path == "/v1/responses" and body and json.loads(body).get("slow"):
            return self.events([b"data: tick\n\n"] * 1000)
        if self.path == "/v1/messages" and body and json.loads(body).get("stream"):
            return self.events([f"event: {name}\ndata: {json.dumps({'type': name})}\n\n".encode() for name in ("message_start", "content_block_delta", "message_delta", "message_stop")])
        self.answer(200, {"method": self.command, "path": self.path, "fleet": fleet})

    def do_GET(self):
        if (self.headers.get("Upgrade") or "").lower() != "websocket":
            return self.generic()
        self.record(b"")
        accept = base64.b64encode(hashlib.sha1((self.headers["Sec-WebSocket-Key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        self.wfile.write(f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n".encode())
        while True:
            data = self.connection.recv(65536)
            if not data:
                break
            self.connection.sendall(b"echo:" + data)
        self.close_connection = True

    do_POST = do_PUT = do_DELETE = do_HEAD = generic


class Drip(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.server.posts.append(time.monotonic())
        try:
            for _ in range(200):
                self.wfile.write(b"H")
                self.wfile.flush()
                time.sleep(0.05)
        except OSError:
            pass
        self.close_connection = True


class Receiver(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.posts.append(({name.lower(): value for name, value in self.headers.items()}, body))
        failing = self.server.fail > 0
        self.server.fail -= 1
        self.send_response(500 if failing else 200)
        self.send_header("Content-Length", "0")
        self.end_headers()


def websocket(port, key, extra=""):
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.sendall(("GET /v1/responses HTTP/1.1\r\nHost: sparkpipe.ai\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                  f"Sec-WebSocket-Version: 13\r\nAuthorization: Bearer {key}\r\nX-SparkPipe-Account: ACFAKE\r\n{extra}\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = sock.recv(4096)
        if not chunk:
            break
        head += chunk
    return sock, head


def hub(port, path, body, key=POOL_KEY):
    status, _, raw = call(port, "POST", path, body, {"Authorization": f"Bearer {key}"})
    return status, json.loads(raw)


def notices(app, account, kind=None):
    return [n for n in app.notices.items() if n["account"] == account and (kind is None or n["type"] == kind)]


def ledger_lines(app):
    return [json.loads(raw) for raw in Path(app.cfg.path("ledger.jsonl")).read_text().splitlines()]


def private(path):
    return os.stat(path).st_mode & 0o777 == 0o600


def until(test, seconds=10):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if test():
            return True
        time.sleep(0.05)
    return test()


def check_units(failures):
    if market.text_hc("1.23456789") != "1.234567" or market.months_later(datetime(2028, 2, 29, 5, tzinfo=timezone.utc), 12) != datetime(2029, 2, 28, 5, tzinfo=timezone.utc):
        failures.append("HC rounds down to 6 places, and a lot issued on 29 February lapses on 28 February a year later")
    if market.week_start(kst(2027, 1, 10, 23, 59)) != kst(2027, 1, 4) or market.week_start(kst(2027, 1, 11)) != kst(2027, 1, 11) or market.sunday_after(kst(2027, 1, 6, 12)) != kst(2027, 1, 10) or market.sunday_after(kst(2027, 1, 10)) != kst(2027, 1, 17):
        failures.append("Weeks run from Monday 00:00 to Sunday 24:00 KST, and the round is at Sunday 00:00 KST")
    conf = (ROOT / "deployment" / "vps" / "sparkpipe-pay.conf").read_text()
    if '<Location "/availability.txt">' not in conf or 'ProxyPass "http://127.0.0.1:18090/availability.txt"' not in conf:
        failures.append("Apache does not map https://sparkpipe.ai/availability.txt to the live service")
    if 'ProxyPass "http://127.0.0.1:18090/" retry=0 timeout=900 flushpackets=on upgrade=websocket' not in conf or 'ProxyPass "http://127.0.0.1:18091/" retry=0 timeout=900 flushpackets=on upgrade=websocket' not in conf:
        failures.append("Apache does not pass the WebSocket upgrade of /v1/responses through to the live service and the sandbox")
    for unit in ("sparkpipe-pay.service", "sparkpipe-pay-sandbox.service"):
        if "\nUMask=0077\n" not in (ROOT / "deployment" / "vps" / unit).read_text():
            failures.append(f"{unit} does not keep the service's files private with UMask=0077")


def live_flows(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    upstream = tp.serve(FakeUpstream, seen=[], calls=[])
    server, app, port = live(work, "live", mailbox, upstream)
    m = app.market
    try:
        opening(failures, app, m)
        state = weeks_and_hours(failures, app, m, port)
        waitlist(failures, app, m, port, mailbox, state)
        gateway(failures, app, m, port, upstream, state)
        passthrough(failures, app, m, port, upstream, state)
        key_options(failures, app, m, port, upstream, state)
        usage(failures, app, m, port, state)
        holders_and_maintenance(failures, app, m, port, state)
        deadline(failures, app, m, mailbox, state)
        replay = sp.Book(app.cfg.path("ledger.jsonl"))
        moment = market.stamp(m.now())
        if [replay.market.balance(a, moment) for a in app.book.accounts] != [app.book.market.balance(a, moment) for a in app.book.accounts] or set(replay.market.bookings) != set(app.book.market.bookings) or replay.market.round != app.book.market.round:
            failures.append("a replayed ledger gives different balances, bookings or rounds")
        lines = client(app, port, state["H"])("GET", "/market/ledger")[1]["lines"]
        if not lines or any(line["data"].get("account") not in (None, state["H"]) for line in lines) or "booking" not in {line["type"] for line in lines}:
            failures.append("the ledger export holds other accounts' lines or misses the account's bookings")
    finally:
        server.shutdown()
        upstream.shutdown()
        mailbox.shutdown()


def opening(failures, app, m):
    NOW[0] = kst(2026, 11, 28, 12)
    early = person(app, "early", 50)
    if refused(lambda: m.book_hours(early, {"fleet": "F1", "start": slot(kst(2026, 11, 30, 23)), "hours": 1})) != (409, "not_open") or refused(lambda: m.book_hours(early, {"fleet": "F1", "start": slot(kst(2026, 12, 1)), "hours": 1})):
        failures.append("Hour Fleet hours open on hoursFrom and not before")
    if refused(lambda: m.book_hours(early, {"fleet": "F4", "start": slot(kst(2026, 11, 29, 12)), "hours": 1})) != (409, "not_open"):
        failures.append("Hours on a Fleet that is not an Hour Fleet are booked before hoursFrom")
    NOW[0] = kst(2026, 12, 25, 12)
    if refused(lambda: m.quote_weeks(early, {"fleet": "F3", "weeks": 1})) != (409, "not_open"):
        failures.append("a Week before weeksFrom is booked")
    for moment, first in ((kst(2027, 1, 1, 12), kst(2027, 1, 4)), (kst(2027, 1, 3, 23), kst(2027, 1, 4)), (kst(2027, 1, 4, 0, 30), kst(2027, 1, 11))):
        NOW[0] = moment
        quote = m.quote_weeks(early, {"fleet": "F4", "weeks": 2})
        if (quote["first"], quote["last"], quote["end"]) != (slot(first), slot(first + market.WEEK), slot(first + 2 * market.WEEK)):
            failures.append(f"a first Week asked at {market.kst_text(moment)} does not start on the next Monday 00:00 KST and run to Sunday 24:00: {quote}")


def weeks_and_hours(failures, app, m, port):
    NOW[0] = kst(2027, 1, 1, 12)
    state = {name: person(app, name.lower(), credits) for name, credits in (("H", 10000), ("C", 5000), ("D", 100), ("G", 1000), ("K", 1000))}
    H, C, D, G, K = (state[name] for name in "HCDGK")
    first = m.book_weeks(H, {"fleet": "F3", "weeks": 1})
    if (first["first"], first["end"], first["price"], first["discount"], first["extends"]) != (slot(kst(2027, 1, 4)), slot(kst(2027, 1, 11)), "144.000000", "0", False) or balance(app, H) != 10000 - 144:
        failures.append(f"a first Week is booked from the next Monday at weekHC times Rate: {first}")
    turned = [line["data"] for line in ledger_lines(app) if line["type"] == "auto_extend" and line["data"]["account"] == H]
    if [(t["fleet"], t["weeks"], t.get("default")) for t in turned] != [("F3", 1, True)] or m.view(H)["pinned"][0]["autoExtend"] != 1:
        failures.append(f"a first booked Week does not turn auto-extend on with one Week ahead: {turned}")
    status, off = client(app, port, H)("PUT", "/market/auto-extend", {"fleet": "F3", "weeks_ahead": 0})
    if status != 200 or off != {"fleet": "F3", "weeks_ahead": 0, "through": slot(kst(2027, 1, 11))} or m.view(H)["pinned"][0]["autoExtend"] != 0:
        failures.append(f"auto-extend is not turned off with one call: {status} {off}")
    NOW[0] = kst(2027, 1, 6, 12)
    public = m.public()
    fleets = {f["id"]: f for f in public["fleets"]}
    if [fleets[f]["status"] for f in ("F1", "F3", "F4", "F5")] != ["hour-fleet", "pinned", "free", "free"] or fleets["F4"]["firstWeek"] != slot(kst(2027, 1, 11)) or fleets["F3"]["firstWeek"] is not None or fleets["F3"]["booked"] != [[slot(kst(2027, 1, 4)), slot(kst(2027, 1, 11))]]:
        failures.append(f"the public Fleet states: {fleets}")
    if public["waitlist"] != {"locks": 0} or fleets["F3"]["storageTB"] != 4 or public["hc"]["prices"][1] != {"from": 144, "usd": 15, "krw": 19500} or public["bookings"]["weekHC"] != 144:
        failures.append("the public state shows more of an empty waitlist than its count of locks, or lacks the published settings")
    for who in (C, H):
        if refused(lambda: m.book_hours(who, {"fleet": "F3", "start": slot(kst(2027, 1, 6, 14)), "hours": 1})) != (409, "pinned"):
            failures.append("an Hour is booked on a pinned Fleet")
    if refused(lambda: m.quote_weeks(C, {"fleet": "F3", "weeks": 1})) != (409, "pinned"):
        failures.append("another customer books the next Week of a pinned Fleet before its holder's deadline")
    for weeks, off, price in ((3, "0", "432.000000"), (4, "0.05", "547.200000"), (13, "0.1", "1684.800000"), (26, "0.15", "3182.400000"), (52, "0.2", "5990.400000")):
        quote = m.quote_weeks(C, {"fleet": "F4", "weeks": weeks})
        if (quote["discount"], quote["price"]) != (off, price):
            failures.append(f"{weeks} Weeks are priced at {quote['price']} HC, {quote['discount']} off, not {price} at {off}")
    if refused(lambda: m.quote_weeks(C, {"fleet": "F4", "weeks": 53})) != (400, None) or m.quote_weeks(C, {"fleet": "F4", "weeks": 52})["last"] != slot(kst(2027, 1, 4) + 52 * market.WEEK):
        failures.append("the last Week may start more than 52 Weeks after the current Week")
    quote = m.quote_weeks(H, {"fleet": "F3", "weeks": 3})
    if (quote["first"], quote["count"], quote["discount"], quote["price"], quote["extends"]) != (slot(kst(2027, 1, 11)), 4, "0.05", "410.400000", True):
        failures.append(f"an extension does not follow the booked Week, or its discount does not count the current Week: {quote}")
    stale = (refused(lambda: m.book_weeks(H, {"fleet": "F3", "weeks": 3, "first": slot(kst(2027, 1, 4)), "price": quote["price"]})),
             refused(lambda: m.book_weeks(H, {"fleet": "F3", "weeks": 3, "first": quote["first"], "price": "1"})),
             refused(lambda: m.book_hours(C, {"fleet": "F4", "start": slot(kst(2027, 1, 6, 13)), "hours": 2, "price": "1"})))
    if stale != ((409, "quote_changed"),) * 3 or balance(app, H) != 10000 - 144:
        failures.append(f"a booking whose quoted first Week or price no longer holds is charged: {stale}")
    m.book_weeks(H, {"fleet": "F3", "weeks": 3, "first": quote["first"], "price": "410.4"})
    if refused(lambda: m.quote_weeks(H, {"fleet": "F3", "weeks": 50})) != (409, "too_far") or m.quote_weeks(H, {"fleet": "F3", "weeks": 49})["first"] != slot(kst(2027, 2, 1)):
        failures.append("an extension is not contiguous, or reaches past 52 Weeks after the current Week")
    if m.view(H)["pinned"] != [{"fleet": "F3", "weeks": 4, "through": slot(kst(2027, 2, 1)), "deadline": slot(kst(2027, 1, 31)), "canExtend": True, "autoExtend": 0, "autoExtendWaiting": None}]:
        failures.append(f"the account's pinned Fleets: {m.view(H)['pinned']}")
    if refused(lambda: m.book_weeks(C, {"fleet": "F3", "weeks": 1}))[0] != 409 or refused(lambda: m.book_weeks(C, {"fleet": "F1", "weeks": 1})) != (409, "not_weekly"):
        failures.append("another customer books a Week of a pinned Fleet, or the Demo Fleet is booked by the Week")
    hours = m.book_hours(C, {"fleet": "F4", "start": slot(kst(2027, 1, 6, 13)), "hours": 2})
    if hours["hc"] != "2.000000" or hours["state"] != "upcoming" or refused(lambda: m.book_hours(C, {"fleet": "F4", "start": slot(kst(2027, 1, 10, 23)), "hours": 1})):
        failures.append(f"free-Fleet Hours to the end of the current Week: {hours}")
    if refused(lambda: m.book_hours(C, {"fleet": "F5", "start": slot(kst(2027, 1, 10, 23)), "hours": 2})) != (409, "too_far") or refused(lambda: m.book_hours(D, {"fleet": "F4", "start": slot(kst(2027, 1, 6, 14)), "hours": 1})) != (409, "taken"):
        failures.append("a free-Fleet Hour reaches past the current Week, or takes a booked hour")
    if refused(lambda: m.book_hours(C, {"fleet": "F5", "start": slot(kst(2027, 1, 6, 12)), "hours": 1})) != (409, "started"):
        failures.append("an hour that has started is booked")
    hour_fleet(failures, m, C, D)
    m.book_weeks(G, {"fleet": "F4", "weeks": 1})
    m.book_weeks(K, {"fleet": "F5", "weeks": 1})
    for who, fleet in ((G, "F4"), (K, "F5")):
        m.auto_extend(who, {"fleet": fleet, "weeks_ahead": 0})
    return state


def hour_fleet(failures, m, C, D):
    book = lambda who, start, hours: refused(lambda: m.book_hours(who, {"fleet": "F1", "start": slot(start), "hours": hours}))
    if book(C, kst(2027, 1, 6, 13), 24) or book(C, kst(2027, 1, 7, 13), 1) != (409, "hour_cap") or book(D, kst(2027, 1, 7, 13), 1):
        failures.append("the Hour Fleets' weekly cap is not per account")
    if book(C, kst(2027, 1, 10, 23), 2) != (409, "hour_cap") or book(C, kst(2027, 1, 11), 3):
        failures.append("Hour Fleet hours are not capped by the Week they fall in")
    if book(C, kst(2027, 1, 13, 12), 1) != (409, "too_far") or book(C, kst(2027, 1, 13, 11), 1):
        failures.append("Hour Fleet hours are not bookable exactly up to 7 days ahead")


def waitlist(failures, app, m, port, mailbox, state):
    for name, credits in (("A", 1000), ("B", 1000), ("W", 1000), ("E", 50), ("X", 144)):
        state[name] = person(app, name.lower(), credits)
    A, B, W, E, X = (state[name] for name in "ABWEX")
    app.mails.join()
    sent = len(mailbox.mail)
    if refused(lambda: m.join(E, {"lock": 144})) != (402, "insufficient_hc") or refused(lambda: m.join(A, {"lock": 100})) != (409, "lock_too_small"):
        failures.append("a lock below weekHC, or above the unlocked balance, joins the waitlist")
    order = [m.join(A, {"lock": 200})["position"], m.join(B, {"lock": 300})["position"], m.position(A), m.join(W, {"lock": 200})["position"]]
    order += [m.join(A, {"lock": "50"})["position"], m.join(W, {"lock": 100})["position"], m.position(B), m.position(A)]
    if order != [1, 1, 2, 3, 2, 2, 1, 3]:
        failures.append(f"the waitlist is not ordered by largest lock, then earliest join, keeping the join time on an added lock: {order}")
    app.mails.join()
    notes = [(rcpt, message["Subject"], message.get_content()) for rcpt, message in mailbox.mail[sent:]]
    expect = [([app.cfg.notify], f"Waitlist lock: {who}, {lock} HC (+{change})") for who, lock, change in (("A", "200.000000", "200.000000"), ("B", "300.000000", "300.000000"), ("W", "200.000000", "200.000000"), ("A", "250.000000", "50.000000"), ("W", "300.000000", "100.000000"))]
    if [(rcpt, subject) for rcpt, subject, _ in notes] != expect:
        failures.append(f"the operator is not emailed once for each lock created or increased, and never for a refused one: {[(rcpt, subject) for rcpt, subject, _ in notes]}")
    elif f"A <a@example.com> ({A}) added 50.000000 HC to its lock of 200.000000 HC: the lock is now 250.000000 HC (+50.000000 HC), position 2 of 3" not in notes[3][2] or "joined the waitlist with 200.000000 HC" not in notes[0][2]:
        failures.append(f"the operator's lock email does not give the account, the new lock and the change: {notes[3][2]!r}")
    smtp, app.cfg.smtp = app.cfg.smtp, "127.0.0.1:9"
    logged = io.StringIO()
    try:
        with contextlib.redirect_stderr(logged):
            joined = m.join(X, {"lock": 144})
            app.mails.join()
    finally:
        app.cfg.smtp = smtp
    if joined != {"locked": "144.000000", "position": 4} or app.book.market.locked(X) != 144 or f"mail to {app.cfg.notify}" not in logged.getvalue():
        failures.append(f"a lock fails, or is not recorded, when the operator's email cannot be sent: {joined} {logged.getvalue()!r}")
    if m.public()["waitlist"] != {"locks": 4}:
        failures.append(f"the public state does not count the accounts that hold a lock: {m.public()['waitlist']}")
    if refused(lambda: m.book_hours(X, {"fleet": "F5", "start": slot(kst(2027, 1, 6, 15)), "hours": 1})) != (402, "insufficient_hc") or m.leave(X) != {"locked": "0.000000", "position": None, "repaid": "0.000000"} or balance(app, X) != 144:
        failures.append("locked HC can be spent, or leaving does not unlock all of it")
    if refused(lambda: m.quote_weeks(E, {"fleet": "F5", "weeks": 1})) != (409, "waitlist") or [f["firstWeek"] for f in m.public()["fleets"] if f["id"] == "F5"] != [None]:
        failures.append("a first Week jumps ahead of a waitlist lock that could pay for it")
    shown = client(app, port, A)("GET", "/market/state")[1]
    text = json.dumps(shown)
    if (shown["account"]["position"], shown["account"]["locked"], shown["account"]["balance"]) != (3, "250.000000", "750.000000") or shown["waitlist"] != {"locks": 3} or any(other in text for other in (B, W, "b@example.com", "300.000000")):
        failures.append(f"a customer sees more of the waitlist than its own position: {shown['account']}")


def gateway(failures, app, m, port, upstream, state):
    H, C, D = state["H"], state["C"], state["D"]
    NOW[0] = kst(2027, 1, 6, 13, 30)
    keys = {who: m.create_key(who, {"note": "test"})["key"] for who in (H, C, D)}
    state["keys"] = keys
    if chat(port, keys[D], {})[0] != 403 or chat(port, keys[D], {})[1]["error"]["code"] != "fleet_not_held" or call(port, "GET", "/v1/models", headers={"Authorization": f"Bearer {keys[D]}"})[0] != 403:
        failures.append("the gateway serves an account that holds no Fleet")
    if chat(port, "sk-hc-wrong", {})[0] != 401 or chat(port, keys[C], {}) != (409, {"error": {"message": "This account holds F1, F4 now: name one in the X-SparkPipe-Fleet header.", "type": "invalid_request_error", "code": "fleet_required"}}) or chat(port, keys[C], {"sparkpipe": {"fleet": "F3"}})[1].get("error", {}).get("code") != "fleet_not_held":
        failures.append("the gateway does not make an account holding two Fleets name one it holds")
    seen = len(upstream.seen)
    if chat(port, keys[C], {"sparkpipe": {"fleet": "F4"}}, {"X-SparkPipe-Fleet": "F9"})[1].get("error", {}).get("code") != "fleet_not_held" or chat(port, keys[C], {"sparkpipe": {"fleet": "F9"}}, {"X-SparkPipe-Fleet": "F4"})[0] != 200:
        failures.append("the X-SparkPipe-Fleet header does not come before the body's sparkpipe.fleet")
    status, answer = chat(port, keys[C], {"sparkpipe": {"fleet": "F4"}}, {"X-SparkPipe-Account": "ACFAKE"})
    headers, accounts, body = upstream.seen[-1] if len(upstream.seen) > seen else ({}, [], {})
    if status != 200 or answer.get("sparkpipe", {}).get("fleet") != "F4" or headers != {"Authorization": f"Bearer {POOL_KEY}", "X-SparkPipe-Fleet": "F4", "X-SparkPipe-Account": C} or accounts != [C] or "sparkpipe" in body:
        failures.append(f"the gateway does not forward a holder with its Fleet and account, or forwards the customer's own copies: {status} {headers} {accounts} {body}")
    status, answer = chat(port, keys[H], {})
    if status != 200 or upstream.seen[-1][0]["X-SparkPipe-Fleet"] != "F3" or upstream.seen[-1][0]["X-SparkPipe-Account"] != H:
        failures.append(f"a Week holder is not forwarded to its pinned Fleet: {status} {answer}")
    status, _, raw = call(port, "GET", "/v1/models", headers={"Authorization": f"Bearer {keys[C]}", "X-SparkPipe-Fleet": "F4"})
    if status != 200 or json.loads(raw) != {"method": "GET", "path": "/v1/models", "fleet": "F4"}:
        failures.append("a holder's model listing does not come from the hub of the Fleet it names")
    before, seen = balance(app, C), len(upstream.seen)
    for extra in ({"fleet": "F1", "priority": 2}, {"bid": 2}, ["F1"], {"fleet": 1}):
        status, answer = chat(port, keys[C], {"sparkpipe": extra})
        if status != 400 or answer["error"]["code"] != "invalid_sparkpipe":
            failures.append(f"a sparkpipe field other than fleet is accepted: {extra} {status} {answer}")
    if len(upstream.seen) != seen or balance(app, C) != before:
        failures.append("a refused sparkpipe field reaches the Fleet or charges HC")
    status, _, raw = call(port, "POST", "/v1/chat/completions", {"model": "kimi-k3", "messages": [{"role": "user", "content": "stream"}], "stream": True}, {"Authorization": f"Bearer {keys[H]}"})
    events = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
    if status != 200 or events[-1] != "[DONE]" or json.loads(events[-2])["sparkpipe"]["fleet"] != "F3":
        failures.append(f"a streamed request is not passed through: {events}")
    if chat(port, keys[C], {"model": "gpt-9"}, {"X-SparkPipe-Fleet": "F4"}) != (200, {"model": "gpt-9", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 5}, "sparkpipe": {"fleet": "F4"}}):
        failures.append("a model name the hub's alias table resolves does not reach the hub, or the X-SparkPipe-Fleet header does not choose the Fleet")


HUB_CALLS = [
    ("POST", "/v1/messages", {"model": "kimi-k3+core", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}),
    ("POST", "/v1/messages/count_tokens", {"model": "kimi-k3", "messages": []}),
    ("POST", "/v1/responses", {"model": "kimi-k3", "input": "hi"}),
    ("GET", "/v1/responses/resp_1", None),
    ("GET", "/v1/models", None),
    ("GET", "/v1/codex/models", None),
    ("HEAD", "/api/hello", None),
    ("POST", "/v1/tokenize", {"text": "hi"}),
    ("POST", "/v1/detokenize", {"tokens": [1]}),
    ("DELETE", "/v1/requests/r_7Q2", None),
    ("GET", "/v1/batches", None), ("POST", "/v1/batches", {"input": "jobs.jsonl"}), ("GET", "/v1/batches/b_1", None), ("DELETE", "/v1/batches/b_1", None),
    ("POST", "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    ("GET", "/v1/state", None),
    ("POST", "/v1/cache/warm", {"format": "anthropic"}),
    ("GET", "/v1/cache/pins", None), ("POST", "/v1/cache/pins", {"pack": "core"}), ("DELETE", "/v1/cache/pins/pin_1", None),
    ("DELETE", "/v1/cache?storage=true", None),
    ("GET", "/v1/models/resident", None), ("GET", "/v1/models/aliases", None), ("PUT", "/v1/models/aliases", {"claude-*": "kimi-k3"}),
    ("GET", "/v1/packs", None), ("POST", "/v1/packs", {"name": "core"}), ("GET", "/v1/packs/core", None), ("DELETE", "/v1/packs/core", None), ("POST", "/v1/packs/core/refresh", {}),
    ("GET", "/v1/repos", None), ("POST", "/v1/repos", {"name": "main"}), ("DELETE", "/v1/repos/main", None),
    ("GET", "/git/main.git/info/refs?service=git-receive-pack", None),
    ("GET", "/v1/storage?prefix=models/", None), ("PUT", "/v1/storage/models/x/config.json", {"a": 1}), ("GET", "/v1/storage/models/x/config.json", None), ("DELETE", "/v1/storage/models/x/config.json", None),
    ("GET", "/v1/storage/changes?since=2027-01-05T00:00:00Z", None),
    ("POST", "/v1/export", {"paths": ["models/"]}), ("POST", "/v1/restore", {"path": "exports/a.tar"}),
    ("POST", "/v1/fetch", {"source": "hf://org/model@main", "to": "models/x/"}), ("GET", "/v1/fetch/f_1", None), ("DELETE", "/v1/fetch/f_1", None), ("POST", "/v1/fetch/f_1/resume", {}),
    ("GET", "/v1/functions", None), ("POST", "/v1/functions/review/runs", {}), ("GET", "/v1/runs", None), ("GET", "/v1/runs/run_1", None), ("GET", "/v1/runs/run_1/log", None), ("DELETE", "/v1/runs/run_1", None),
    ("GET", "/v1/schedules", None), ("PUT", "/v1/schedules", {"function": "review", "cron": "0 2 * * *", "tz": "Asia/Seoul"}), ("DELETE", "/v1/schedules", None), ("GET", "/v1/reports/2027-01-06", None),
    ("GET", "/v1/drafters", None), ("PUT", "/v1/drafters/active", {"drafter": "acct-1"}),
    ("GET", "/v1/custom-models", None), ("POST", "/v1/custom-models", {"name": "mine"}), ("DELETE", "/v1/custom-models/cm_1", None),
]


def raw_call(port, method, path, body=None, headers=None, chunks=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        if chunks is not None:
            connection.request(method, path, body=iter(chunks), headers=dict(headers or {}, **{"Transfer-Encoding": "chunked"}), encode_chunked=True)
        else:
            connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, {name.lower(): value for name, value in response.getheaders()}, response.read()
    finally:
        connection.close()


def basic(key):
    return {"Authorization": "Basic " + base64.b64encode(f"git:{key}".encode()).decode()}


def sent(call, name):
    return [value for key, value in call["items"] if key.lower() == name]


def passthrough(failures, app, m, port, upstream, state):
    H, C, D, keys = state["H"], state["C"], state["D"], state["keys"]
    record = m.key_account(keys[H])
    forged = {"X-SparkPipe-Account": "ACFAKE", "X-SparkPipe-Key": "KYFAKE", "X-SparkPipe-Allow-Swap": "1", "X-SparkPipe-Public": "0", "anthropic-version": "2023-06-01", "X-SparkPipe-Priority": "background", "Cookie": "other=1"}
    for n, (method, path, body) in enumerate(HUB_CALLS):
        raw = None if body is None else json.dumps(body).encode()
        auth = basic(keys[H]) if path.startswith("/git/") else {"Authorization": f"Bearer {keys[H]}"} if n % 2 else {"X-Api-Key": keys[H]}
        before = len(upstream.calls)
        status, headers, answer = raw_call(port, method, path, raw, dict(forged, **auth))
        got = upstream.calls[before] if len(upstream.calls) == before + 1 else {}
        expect = {"authorization": f"Bearer {POOL_KEY}", "x-sparkpipe-account": H, "x-sparkpipe-fleet": "F3", "x-sparkpipe-key": record["key"], "x-sparkpipe-allow-swap": "0", "x-sparkpipe-public": "1",
                  "x-sparkpipe-priority": "background", "anthropic-version": "2023-06-01"}
        if (got.get("method"), got.get("path"), got.get("body")) != (method, path, raw or b"") or any(sent(got, name) != [value] for name, value in expect.items()) or sent(got, "x-api-key") or sent(got, "cookie"):
            failures.append(f"{method} {path} does not reach the hub unchanged with the gateway's own account, fleet, key and swap headers: {got}")
        if status != 200 or headers.get("x-hub") != "1" or answer != (b"" if method == "HEAD" else json.dumps({"method": method, "path": path, "fleet": "F3"}).encode()):
            failures.append(f"the hub's answer to {method} {path} does not come back unchanged: {status} {headers} {answer[:200]!r}")
    before = len(upstream.calls)
    for method, path, body in HUB_CALLS + [("POST", "/v1/chat/completions", {"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]}), ("POST", "/v1/state/move", {"from": "F3", "to": "F4"})]:
        status, _, answer = raw_call(port, method, path, None if body is None else json.dumps(body).encode(), basic(keys[D]) if path.startswith("/git/") else {"X-Api-Key": keys[D]})
        if status != 403 or (method != "HEAD" and "fleet_not_held" not in answer.decode()):
            failures.append(f"{method} {path} is not refused to an account that holds no Fleet: {status} {answer[:200]!r}")
    status, _, answer = raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": keys[D]})
    if json.loads(answer) != {"type": "error", "error": {"type": "fleet_not_held", "message": "This account holds no Fleet now. Book an Hour or a Week to send requests."}}:
        failures.append(f"a refusal on /v1/messages does not use Anthropic's error body: {answer!r}")
    for method, path in (("POST", "/v1/sparkpipe/switch-now"), ("GET", "/v1/sparkpipe/status"), ("GET", "/v1/storage/../sparkpipe/status"), ("GET", "/v1/storage/%2e%2e/%2E%2E/sparkpipe/status"), ("GET", "/v1/teams"), ("GET", "/git/../v1/sparkpipe/status")):
        status, _, _ = raw_call(port, method, path, b"{}" if method == "POST" else None, {"Authorization": f"Bearer {keys[H]}"})
        if status != 404:
            failures.append(f"{method} {path}, which is not in the API, is not refused: {status}")
    if len(upstream.calls) != before:
        failures.append(f"a refused request reaches the hub: {upstream.calls[before:]}")
    status, headers, _ = raw_call(port, "GET", "/git/main.git/info/refs?service=git-receive-pack")
    if status != 401 or headers.get("www-authenticate") != 'Basic realm="SparkPipe"' or len(upstream.calls) != before:
        failures.append(f"git without credentials is not asked for the key with Basic authentication: {status} {headers}")
    big = bytes(range(256)) * (sp.BODY_LIMIT // 256)
    status, _, _ = raw_call(port, "PUT", "/v1/storage/models/x/part-1", big, {"Authorization": f"Bearer {keys[H]}", "Content-Range": f"bytes 0-{len(big) - 1}/*"})
    got = upstream.calls[-1]
    if status != 200 or got["path"] != "/v1/storage/models/x/part-1" or got["body"] != (len(big), hashlib.sha256(big).hexdigest()) or sent(got, "content-range") != [f"bytes 0-{len(big) - 1}/*"]:
        failures.append(f"a 32 MiB storage chunk does not reach the hub whole: {status} {got['path']} {got['body'] if isinstance(got['body'], tuple) else len(got['body'])}")
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.putrequest("PUT", "/v1/storage/models/x/part-2")
    connection.putheader("Authorization", f"Bearer {keys[H]}")
    connection.putheader("Content-Length", str(sp.BODY_LIMIT + 1))
    connection.endheaders()
    status = connection.getresponse().status
    connection.close()
    if status != 413 or upstream.calls[-1]["path"] != "/v1/storage/models/x/part-1":
        failures.append(f"a body over 32 MiB is not refused before it reaches the hub: {status}")
    early = []
    for method, path in (("POST", "/v1/chat/completions"), ("PUT", "/v1/storage/models/x/part-3"), ("POST", "/v1/state/move"), ("POST", "/git/main.git/git-receive-pack")):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.putrequest(method, path)
        for name, value in dict(basic(keys[D]) if path.startswith("/git/") else {"X-Api-Key": keys[D]}, **{"Content-Length": str(20 << 20)}).items():
            connection.putheader(name, value)
        connection.endheaders()
        try:
            early.append(connection.getresponse().status)
        except OSError as error:
            early.append(str(error))
        connection.close()
    if early != [403, 403, 403, 403] or len(upstream.calls) != before + 1:
        failures.append(f"the body of a request from an account that holds no Fleet is read before it is refused: {early}")
    for size in (b"-1", b"+5", b"0x10", b"zz", b"123456789"):
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(b"POST /git/main.git/git-receive-pack HTTP/1.1\r\nHost: sparkpipe.ai\r\nAuthorization: " + basic(keys[H])["Authorization"].encode() + b"\r\nTransfer-Encoding: chunked\r\n\r\n" + size + b"\r\nabcde\r\n0\r\n\r\n")
        try:
            head = sock.recv(100)
        except OSError:
            head = b"timeout"
        sock.close()
        if not head.startswith(b"HTTP/1.0 400") or upstream.calls[-1]["path"] != "/v1/storage/models/x/part-1":
            failures.append(f"a chunk size of {size!r} is not refused: {head!r}")
    status, _, _ = raw_call(port, "POST", "/git/main.git/git-receive-pack", headers=dict(basic(keys[H]), **{"Content-Type": "application/x-git-receive-pack-request"}), chunks=[b"0032want ", b"abc", b"\x00" * 70000])
    got = upstream.calls[-1]
    if status != 200 or got["path"] != "/git/main.git/git-receive-pack" or got["body"] != b"0032want abc" + b"\x00" * 70000 or sent(got, "content-length") != [str(70012)] or sent(got, "transfer-encoding"):
        failures.append(f"a chunked git push does not reach the hub as one body: {status} {got['path']} {len(got['body'])}")
    status, headers, answer = raw_call(port, "POST", "/v1/messages", json.dumps({"model": "kimi-k3", "stream": True, "messages": []}).encode(), {"X-Api-Key": keys[H]})
    if status != 200 or headers.get("content-type") != "text/event-stream" or [line[7:] for line in answer.decode().splitlines() if line.startswith("event: ")] != ["message_start", "content_block_delta", "message_delta", "message_stop"]:
        failures.append(f"a Messages stream does not pass through: {status} {headers} {answer!r}")
    status, _, answer = raw_call(port, "PUT", "/v1/models/resident", json.dumps({"models": ["glm-5.3"]}).encode(), {"Authorization": f"Bearer {keys[H]}"})
    if status != 409 or json.loads(answer)["error"]["type"] != "model_not_loaded":
        failures.append(f"the hub's refusal of a swap from a key without allow_swap does not come back unchanged: {status} {answer!r}")
    sock, head = websocket(port, keys[H])
    got = upstream.calls[-1]
    sock.sendall(b"frame-1")
    echoed = sock.recv(100)
    sock.close()
    if not head.startswith(b"HTTP/1.1 101") or echoed != b"echo:frame-1" or got["path"] != "/v1/responses" or [sent(got, name) for name in ("x-sparkpipe-account", "x-sparkpipe-fleet", "authorization", "sec-websocket-key")] != [[H], ["F3"], [f"Bearer {POOL_KEY}"], ["dGhlIHNhbXBsZSBub25jZQ=="]]:
        failures.append(f"the WebSocket transport of /v1/responses does not reach the hub with the account and fleet: {head!r} {echoed!r} {got}")
    sock, head = websocket(port, keys[D])
    sock.close()
    if not head.startswith(b"HTTP/1.0 403"):
        failures.append(f"a WebSocket from an account that holds no Fleet is not refused: {head!r}")
    status, _, answer = raw_call(port, "POST", "/v1/state/move", json.dumps({"from": "F4", "to": "F1"}).encode(), {"X-Api-Key": keys[C]})
    moved = upstream.calls[-1]
    refusal = raw_call(port, "POST", "/v1/state/move", json.dumps({"from": "F4", "to": "F3"}).encode(), {"X-Api-Key": keys[C]})
    if status != 200 or sent(moved, "x-sparkpipe-fleet") != ["F4"] or moved["path"] != "/v1/state/move" or refusal[0] != 403:
        failures.append(f"moving working state does not go to the source Fleet with both Fleets held: {status} {moved} {refusal}")
    session = client(app, port, C)
    status, answer = session("GET", "/v1/state", headers={"X-SparkPipe-Fleet": "F4"})
    got = upstream.calls[-1]
    if status != 200 or [sent(got, name) for name in ("x-sparkpipe-account", "x-sparkpipe-fleet", "x-sparkpipe-key", "x-sparkpipe-allow-swap")] != [[C], ["F4"], ["session"], ["1"]] or sent(got, "cookie"):
        failures.append(f"the account session does not reach the hub as the account with allow_swap: {status} {got}")
    session.origin = "https://evil.example"
    if session("PUT", "/v1/models/resident", {"models": []}, headers={"X-SparkPipe-Fleet": "F4"})[0] != 403 or upstream.calls[-1] is not got:
        failures.append("a cross-site request with the account's cookie reaches the hub")


def key_options(failures, app, m, port, upstream, state):
    H, C, keys = state["H"], state["C"], state["keys"]
    owner = client(app, port, H)
    made = lambda body, who=owner: who("POST", "/v1/keys", body)
    status, narrow = made({"scopes": ["inference"], "note": "ci"})
    if status != 200 or narrow["scopes"] != ["inference"] or not narrow["key"].startswith("sk-hc-") or narrow["note"] != "ci":
        failures.append(f"a key with scopes is not created: {status} {narrow}")
        return
    if raw_call(port, "GET", "/v1/storage", headers={"X-Api-Key": narrow["key"]})[0] != 403 or raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": narrow["key"]})[0] != 200 or raw_call(port, "GET", "/v1/requests", headers={"X-Api-Key": narrow["key"]})[0] != 403:
        failures.append("a key's scopes are not enforced")
    for scope, path in (("state", "/v1/state"), ("storage", "/v1/repos"), ("functions", "/v1/runs"), ("models", "/v1/custom-models"), ("inference", "/mcp")):
        found = made({"scopes": [scope]})[1]["key"]
        other = next(p for s_, p in (("state", "/v1/state"), ("storage", "/v1/repos"), ("functions", "/v1/runs"), ("models", "/v1/custom-models"), ("inference", "/mcp")) if s_ != scope)
        if raw_call(port, "GET" if path != "/mcp" else "POST", path, b"{}" if path == "/mcp" else None, {"X-Api-Key": found})[0] != 200 or raw_call(port, "GET" if other != "/mcp" else "POST", other, b"{}" if other == "/mcp" else None, {"X-Api-Key": found})[0] != 403:
            failures.append(f"the {scope} scope does not open {path} alone")
    pinned = made({"fleet": "F4"}, client(app, port, C))[1]["key"]
    raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": pinned, "X-SparkPipe-Fleet": "F1"})
    if sent(upstream.calls[-1], "x-sparkpipe-fleet") != ["F4"]:
        failures.append("a key with a fleet does not send every request to that Fleet")
    elsewhere = made({"fleet": "F3"}, client(app, port, C))[1]["key"]
    if raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": elsewhere})[0] != 403:
        failures.append("a key limited to a Fleet the account does not hold reaches the hub")
    swapper = made({"allow_swap": True})[1]
    status, _, _ = raw_call(port, "PUT", "/v1/models/resident", b"{}", {"X-Api-Key": swapper["key"]})
    if status != 200 or sent(upstream.calls[-1], "x-sparkpipe-allow-swap") != ["1"] or swapper["allow_swap"] is not True:
        failures.append(f"a key with allow_swap does not tell the hub: {status}")
    walled = made({"ips": ["203.0.113.0/24", "2001:db8::/32"]})[1]
    if walled["ips"] != ["203.0.113.0/24", "2001:db8::/32"] or raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": walled["key"]})[0] != 403 or raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": walled["key"], "X-Forwarded-For": "203.0.113.9, 198.51.100.1"})[0] != 403:
        failures.append("a key's IP allowlist admits an address outside it, or trusts anything but the address the proxy added")
    if raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": walled["key"], "X-Forwarded-For": "198.51.100.1, 203.0.113.9"})[0] != 200 or raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": walled["key"], "X-Forwarded-For": "2001:db8::7"})[0] != 200:
        failures.append("a key's IP allowlist refuses an address inside it")
    background = made({"priority": "background"})[1]["key"]
    raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": background})
    first = sent(upstream.calls[-1], "x-sparkpipe-priority")
    raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": background, "X-SparkPipe-Priority": "interactive"})
    if first != ["background"] or sent(upstream.calls[-1], "x-sparkpipe-priority") != ["interactive"]:
        failures.append("a key's default priority is not sent, or overrides the request's own")
    for bad in ({"scopes": ["root"]}, {"scopes": []}, {"fleet": "F9"}, {"expires": "2020-01-01T00:00:00Z"}, {"expires": "soon"}, {"ips": ["300.1.1.1"]}, {"ips": "10.0.0.0/8"}, {"allow_swap": "yes"}, {"priority": "urgent"}, {"note": "a\nb"}):
        status, answer = made(bad)
        if status != 400 or answer["error"]["code"] != "invalid_request":
            failures.append(f"a bad key option is accepted: {bad} {status} {answer}")
    listed = raw_call(port, "GET", "/v1/keys", headers={"X-Api-Key": keys[H]})
    ids = [k["id"] for k in json.loads(listed[2])["data"]]
    if listed[0] != 200 or swapper["id"] not in ids or any("hash" in k or k.get("key", "").startswith("sk-") for k in json.loads(listed[2])["data"]) or raw_call(port, "GET", "/v1/keys", headers={"X-Api-Key": narrow["key"]})[0] != 403:
        failures.append(f"keys are not listed to account:read, without secrets: {listed}")
    if raw_call(port, "POST", "/v1/keys", b"{}", {"X-Api-Key": keys[H]})[0] != 403 or raw_call(port, "DELETE", f"/v1/keys/{narrow['id']}", headers={"X-Api-Key": keys[H]})[0] != 403:
        failures.append("an API key creates or revokes keys")
    if owner("DELETE", f"/v1/keys/{narrow['id']}")[1] != {"id": narrow["id"], "revoked": True} or raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": narrow["key"]})[0] != 401:
        failures.append("a revoked key still works")
    page = owner("POST", "/market/keys", {"note": "page", "scopes": ["inference", "state"], "fleet": "F3"})[1]
    shown = next(k for k in owner("GET", "/market/state")[1]["account"]["keys"] if k["key"] == page["id"])
    if (shown["scopes"], shown["fleet"], shown["note"]) != (["inference", "state"], "F3", "page"):
        failures.append(f"the account page does not create keys with options: {shown}")


def usage(failures, app, m, port, state):
    H, C, keys = state["H"], state["C"], state["keys"]
    key_id = m.key_account(keys[H])["key"]
    moment = m.now()
    base = {"account": H, "fleet": "F3", "model": "kimi-k3", "key": key_id, "api": "messages", "session": "s-1"}
    records = [dict(base, request=f"r_{n}", at=market.stamp(moment - timedelta(minutes=n)), prompt_tokens=1000 * (n + 1), cached_tokens=900 * (n + 1), prefill_tokens=100 * (n + 1), output_tokens=50,
                    decode_tps=15.0 + n, cold_prefill_ms_estimate=5000 * (n + 1), prefill_ms=200, status=200) for n in range(3)]
    records[0].update(prompt="the customer's secret prompt", messages=[{"role": "user", "content": "the customer's secret prompt"}], reply="the secret reply")
    other = dict(base, account=C, fleet="F4", request="r_c", at=market.stamp(moment), prompt_tokens=10)
    status, found = hub(port, "/hub/requests", {"records": records + [other, {"request": "r_x", "fleet": "F3", "at": market.stamp(moment)}, dict(base, request="r_old", at=market.stamp(moment - timedelta(days=400)))]})
    if status != 200 or found["accepted"] != 4 or found["duplicates"] != 0 or [r["index"] for r in found["refused"]] != [4, 5]:
        failures.append(f"the hub's usage records are not taken in, with refusals for incomplete or year-old ones: {status} {found}")
    if hub(port, "/hub/requests", {"records": records})[1] != {"accepted": 0, "duplicates": 3, "refused": []} or hub(port, "/hub/requests", {"records": records}, "wrong")[0] != 401:
        failures.append("a resent usage record is stored twice, or a wrong upstream key is accepted")
    owner = client(app, port, H)
    status, listed = owner("GET", f"/v1/requests?since={market.stamp(moment - timedelta(hours=1))}")
    text = json.dumps(listed)
    if status != 200 or [r["request"] for r in listed["data"]] != ["r_2", "r_1", "r_0"] or "secret" in text or any({"prompt", "messages", "reply"} & set(r) for r in listed["data"]) or "r_c" in text:
        failures.append(f"GET /v1/requests does not list the account's own records without prompts or replies: {status} {text[:400]}")
    elif listed["totals"] != {"prompt_tokens": 6000, "cached_tokens": 5400, "prefill_tokens": 600, "output_tokens": 150, "prefill_ms": 600, "cold_prefill_ms_estimate": 30000, "requests": 3,
                              "prefill_avoided_ms": 30000, "cached_share": 0.9, "decode_tps": round(150 / (50 / 15 + 50 / 16 + 50 / 17), 2)}:
        failures.append(f"the usage records do not sum into the stats: {listed['totals']}")
    page = owner("GET", "/v1/requests?limit=2")[1]
    rest = owner("GET", f"/v1/requests?limit=2&after={page['next']}")[1]
    if [r["request"] for r in page["data"]] != ["r_2", "r_1"] or not page["has_more"] or [r["request"] for r in rest["data"]] != ["r_0"] or rest["has_more"] or rest["totals"]["requests"] != 3:
        failures.append(f"usage records do not page: {page} {rest}")
    by_key = raw_call(port, "GET", f"/v1/requests?key={key_id}&model=kimi-k3", headers={"X-Api-Key": keys[H]})
    if by_key[0] != 200 or json.loads(by_key[2])["totals"]["requests"] != 3 or owner("GET", "/v1/requests?key=KYNONE")[1]["data"] != [] or owner("GET", "/v1/requests?since=2027-01-01&until=2027-01-02")[1]["data"] != []:
        failures.append(f"usage records are not filtered by key, model and window: {by_key}")
    if client(app, port, C)("GET", "/v1/requests")[1]["totals"]["requests"] != 1 or owner("GET", "/v1/requests?since=2027-01-06&until=2026-01-06")[0] != 400 or owner("GET", "/v1/requests?limit=0")[0] != 400:
        failures.append("another account's records are shown, or a bad window or limit is accepted")
    day = moment.strftime("%Y-%m-%d")
    folder = Path(app.cfg.path("requests"))
    if sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*.jsonl")) != sorted([f"{H}/{day}.jsonl", f"{C}/{day}.jsonl"]) or "r_c" in (folder / H / f"{day}.jsonl").read_text():
        failures.append(f"usage records are not filed per account and day: {sorted(str(path) for path in folder.rglob('*.jsonl'))}")
    app.usage.prune(moment + timedelta(days=366))
    kept = (folder / H / f"{day}.jsonl").exists()
    app.usage.prune(moment + timedelta(days=367))
    if not kept or (folder / H / f"{day}.jsonl").exists() or list(app.usage.query(H, moment - timedelta(days=1), moment + timedelta(days=1), {})) or list(folder.iterdir()):
        failures.append("usage records are not kept for a year and then deleted")


def holders_and_maintenance(failures, app, m, port, state):
    H, C = state["H"], state["C"]
    status, found = feed(port, "F3", slot(kst(2027, 1, 6)), slot(kst(2027, 1, 7)))
    if status != 200 or found["fleet"] != "F3" or [h["start"] for h in found["hours"]] != [slot(kst(2027, 1, 6) + market.HOUR * n) for n in range(24)] or {h["account"] for h in found["hours"]} != {H}:
        failures.append(f"the holder feed does not list every hour of a booked Week: {status} {found}")
    status, found = feed(port, "F4", slot(kst(2027, 1, 6, 12)), slot(kst(2027, 1, 6, 16)))
    if found.get("hours") != [{"start": slot(kst(2027, 1, 6, 13)), "account": C}, {"start": slot(kst(2027, 1, 6, 14)), "account": C}]:
        failures.append(f"the holder feed does not list HC-booked Hours: {found}")
    refusals = [feed(port, "F3", slot(kst(2027, 1, 6)), slot(kst(2027, 1, 7)), "wrong")[0], feed(port, "F9", slot(kst(2027, 1, 6)), slot(kst(2027, 1, 7)))[0],
                feed(port, "F3", slot(kst(2027, 1, 6)), slot(kst(2027, 1, 14, 1)))[0], feed(port, "F3", "", slot(kst(2027, 1, 7)))[0]]
    if refusals != [401, 404, 400, 400]:
        failures.append(f"the holder feed does not refuse a bad key, an unknown Fleet or a bad window: {refusals}")
    planned = m.schedule_maintenance("F3", "2027-01-18")
    if planned["start"] != slot(kst(2027, 1, 18)) or planned["hours"] != 3 or refused(lambda: m.schedule_maintenance("F3", "2027-01-18", 1)) != (400, None) or refused(lambda: m.schedule_maintenance("F3", "2027-01-19")) != (400, None):
        failures.append(f"maintenance is not limited to the weekly allowance in a Week starting on Monday: {planned}")
    state["maintenance"] = planned["maintenance"]
    status, found = feed(port, "F3", slot(kst(2027, 1, 17, 21)), slot(kst(2027, 1, 18, 5)))
    if [h["start"] for h in found["hours"]] != [slot(kst(2027, 1, 17, h)) for h in (21, 22, 23)] + [slot(kst(2027, 1, 18, h)) for h in (3, 4)]:
        failures.append(f"the holder feed lists maintenance hours as held: {found}")
    status, kind, raw = call(port, "GET", "/availability.txt")
    if status != 200 or not kind.startswith("text/plain") or not raw.decode().endswith("F1 serving hour-fleet\nF3 serving pinned\nF4 serving available\nF5 serving available\nwaitlist locks 3\n"):
        failures.append(f"availability.txt: {status} {kind} {raw[-120:]!r}")
    if client(app, port, C)("POST", "/market/sandbox/clock", {"hours": 1})[0] != 404 or client(app, port, C)("POST", "/market/sandbox/fleet", {"fleet": "F3", "up": False})[0] != 404:
        failures.append("the live service moves its clock or takes a Fleet down on request")


def deadline(failures, app, m, mailbox, state):
    H, C, G, K, A, B, W = (state[name] for name in "HCGKABW")
    NOW[0] = kst(2027, 1, 10)
    m.tick()
    if [m.position(x) for x in (B, W, A)] != [1, 2, 3] or app.book.market.round != market.stamp(kst(2027, 1, 10)):
        failures.append("a round with no free Fleet changes the waitlist")
    NOW[0] = kst(2027, 1, 14, 12)
    m.leave(W)
    if balance(app, W) != 1000 or [m.position(x) for x in (B, A)] != [1, 2] or refused(lambda: m.quote_weeks(C, {"fleet": "F4", "weeks": 1})) != (409, "pinned"):
        failures.append("leaving the waitlist, or the holder's deadline before Sunday 00:00")
    NOW[0] = kst(2027, 1, 16, 23, 59)
    if refused(lambda: m.quote_weeks(G, {"fleet": "F4", "weeks": 1})):
        failures.append("the holder cannot extend before Sunday 00:00")
    NOW[0] = kst(2027, 1, 17)
    m.tick()
    app.mails.join()
    granted = {b["account"]: b for b in app.book.market.bookings.values() if b["source"] == "waitlist"}
    expect = {B: ("F4", slot(kst(2027, 1, 18)), slot(kst(2027, 2, 1)), "288.000000", "12.000000"), A: ("F5", slot(kst(2027, 1, 18)), slot(kst(2027, 1, 25)), "144.000000", "106.000000")}
    if {who: (b["fleet"], b["start"], b["end"], b["hc"], b["unlocked"]) for who, b in granted.items()} != expect or balance(app, B) != 712 or balance(app, A) != 856 or app.book.market.locks:
        failures.append(f"Sunday 00:00 does not allocate each free Fleet to the waitlist in order, as many Weeks as each lock pays for with the rest unlocked: {granted}")
    if refused(lambda: m.book_weeks(G, {"fleet": "F4", "weeks": 1})) != (409, "taken") or refused(lambda: m.book_weeks(K, {"fleet": "F5", "weeks": 1})) != (409, "taken"):
        failures.append("a holder past its Sunday deadline keeps a Fleet the waitlist was allocated")
    if not any("b@example.com" in rcpt and m_["Subject"].startswith("SparkPipe Fleet 4 is yours") and "Auto-extend is on: it keeps 1 Week booked ahead" in m_.get_content() for rcpt, m_ in mailbox.mail):
        failures.append("the allocated customer is not emailed that the Fleet is its own, with auto-extend on")
    if [app.book.market.extend.get((who, fleet), {}).get("weeks") for who, fleet in ((B, "F4"), (A, "F5"))] != [1, 1]:
        failures.append("a Week granted by the waitlist does not turn auto-extend on with one Week ahead")
    for who, fleet in ((B, "F4"), (A, "F5")):
        m.auto_extend(who, {"fleet": fleet, "weeks_ahead": 0})
    NOW[0] = kst(2027, 1, 30, 23)
    if refused(lambda: m.book_weeks(C, {"fleet": "F4", "weeks": 1})) != (409, "pinned"):
        failures.append("another customer books before the holder's Sunday deadline")
    NOW[0] = kst(2027, 1, 31, 0, 30)
    if refused(lambda: m.book_weeks(H, {"fleet": "F3", "weeks": 1})) or refused(lambda: m.book_weeks(C, {"fleet": "F4", "weeks": 1})):
        failures.append("with nobody allocated at Sunday 00:00, the holder or another customer cannot book the next Week before it starts")
    credited = [lot for lot in app.book.market.lots[H] if lot["source"] == "maintenance"]
    if [(lot["first"], lot["ref"]) for lot in credited] != [("3.000000", state["maintenance"])]:
        failures.append(f"booked hours under maintenance are not credited at the Fleet's Rate: {credited}")


def order_lot(app, account, count, price, total):
    ident = sp.new_id("SP")
    app.book.append("order", {"order": ident, "account": account, "email": "x@example.com", "company": "X", "product": "credits", "processor": "toss", "currency": "KRW", "hc": str(count), "price": price, "net": "0", "vat": "0", "total": total})
    app.market.issue(account, Decimal(count), ident)
    return ident


def discount_flows(failures, work):
    root = work / "discounts"
    site(root)
    path = root / "site" / "pricing.js"
    pricing = tp.read_pricing(path)
    pricing["fleets"] += [dict(pricing["fleets"][0], id="F6", name="Fleet 6"), dict(pricing["fleets"][1], id="F7", name="Fleet 7")]
    tp.write_pricing(path, pricing)
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    server, app, port = live(root, "data", mailbox)
    m = app.market
    try:
        NOW[0] = kst(2027, 1, 6, 12)
        H, S, P, W, N, V = (person(app, name, credits) for name, credits in (("h", 6100), ("s", 1000), ("p", 7000), ("w", 3300), ("n", 100), ("v", 3300)))
        for who, fleet, weeks in ((H, "F3", 52), (S, "F4", 3), (S, "F7", 2), (P, "F5", 4)):
            m.book_weeks(who, {"fleet": fleet, "weeks": weeks})
        for fleet in ("F4", "F7"):
            m.auto_extend(S, {"fleet": fleet, "weeks_ahead": 0})
        m.join(P, {"lock": 6000})
        m.join(W, {"lock": 3200})
        start = slot(kst(2027, 1, 7, 10))
        quotes = {who: m.quote_hours(who, {"fleet": "F1", "start": start, "hours": 2}) for who in (H, S, P, W, N)}
        got = {who: (q["price"], q["discount"], q["weeks"], q["basis"]) for who, q in quotes.items()}
        expect = {H: ("1.600000", "0.2", 52, "held"), S: ("2.000000", "0", 3, "held"), P: ("1.600000", "0.2", 52, "waitlist"), W: ("1.700000", "0.15", 26, "waitlist"), N: ("2.000000", "0", 0, None)}
        if got != expect:
            failures.append(f"an Hour is not priced at the most Weeks held on one Fleet, or the Weeks a waitlist lock books, whichever discount is larger: {got}")
        booked = m.book_hours(H, {"fleet": "F1", "start": start, "hours": 2})
        if (booked["hc"], booked["discount"]) != ("1.600000", "0.2") or m.view(H)["discount"] != {"discount": "0.2", "weeks": 52, "basis": "held", "fleet": "F3"} or balance(app, H) != Decimal("108.0"):
            failures.append(f"a holder's Hours are not charged and shown at its discount: {booked} {m.view(H)['discount']}")
        if m.view(N)["discount"] != {"discount": "0", "weeks": 0, "basis": None, "fleet": None}:
            failures.append(f"an account with no Weeks and no lock shows a discount: {m.view(N)['discount']}")
        if refused(lambda: m.book_hours(H, {"fleet": "F6", "start": slot(kst(2027, 1, 8)), "hours": 22})) or refused(lambda: m.book_hours(H, {"fleet": "F1", "start": slot(kst(2027, 1, 9)), "hours": 1})) != (409, "hour_cap") \
                or refused(lambda: m.book_hours(N, {"fleet": "F6", "start": slot(kst(2027, 1, 9)), "hours": 1})):
            failures.append("the weekly hour cap does not count an account's hours on every Hour Fleet together, or binds another account")
        if refused(lambda: m.quote_weeks(N, {"fleet": "F6", "weeks": 1})) != (409, "not_weekly") or [f["status"] for f in m.public()["fleets"] if f["hourFleet"]] != ["hour-fleet", "hour-fleet"]:
            failures.append("a second Hour Fleet can be pinned or is not shown as an Hour Fleet")
        m.join(V, {"lock": 3200})
        kept = m.book_hours(W, {"fleet": "F6", "start": slot(kst(2027, 1, 9, 10)), "hours": 2})
        taken = m.book_hours(V, {"fleet": "F6", "start": slot(kst(2027, 1, 9, 12)), "hours": 2})
        owed = m.view(V)["waitlistRepay"]
        left = m.leave(V)
        if (taken["hc"], taken["waitlist_saving"], owed, left["repaid"], balance(app, V)) != ("1.700000", "0.300000", "0.300000", "0.300000", Decimal("3298.0")) \
                or [line["data"]["ref"] for line in ledger_lines(app) if line["type"] == "hc_debit" and line["data"]["account"] == V] != [taken["booking"]]:
            failures.append(f"leaving the waitlist before it allocates a Fleet does not repay the waitlist discount of the Hours booked while waiting: {taken} {owed} {left} {balance(app, V)}")
        if m.view(H)["waitlistRepay"] is not None or booked.get("waitlist_saving") is not None or m.view(W)["waitlistRepay"] != "0.300000":
            failures.append("a holder's Hours carry a waitlist saving, or an account off the waitlist shows a repayment")
        NOW[0] = kst(2027, 1, 10, 0, 1)
        m.tick()
        if any(b["source"] == "waitlist" for b in app.book.market.bookings.values()) or [m.position(who) for who in (P, W)] != [1, 2]:
            failures.append("the Sunday round allocates an Hour Fleet to the waitlist")
        NOW[0] = kst(2027, 1, 31, 0, 1)
        m.tick()
        granted = {b["account"]: (b["fleet"], b["weeks"]) for b in app.book.market.bookings.values() if b["source"] == "waitlist"}
        repaid = [line for line in ledger_lines(app) if line["type"] == "hc_debit" and line["data"]["account"] in (P, W)]
        if granted != {P: ("F7", 52), W: ("F4", 26)} or repaid or kept["waitlist_saving"] != "0.300000" or balance(app, W) != Decimal("115.9"):
            failures.append(f"a lock the Sunday round turns into Weeks repays its waitlist discount: {granted} {repaid} {balance(app, W)}")
    finally:
        server.shutdown()
        mailbox.shutdown()


def pair_flows(failures, work):
    root = work / "pairs"
    site(root)
    path = root / "site" / "pricing.js"
    pricing = tp.read_pricing(path)
    base = dict(pricing["fleets"][1])
    pricing["fleets"] = [dict(base, id=f"F{n}{side}", name=f"Fleet {n}{side}", state="serving", hourFleet=side == "B", pair=f"F{n}{'B' if side == 'A' else 'A'}")
                         for n in range(1, 5) for side in "AB"]
    tp.write_pricing(path, pricing)
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    server, app, port = live(root, "data", mailbox)
    m = app.market
    through = lambda who, fleet: m.weeks_left(who, fleet, NOW[0])[1]
    try:
        NOW[0] = kst(2027, 1, 6, 12)
        A1, A2, A3, O = (person(app, name, credits) for name, credits in (("a1", 2000), ("a2", 2000), ("a3", 2000), ("o", 1000)))
        for who, fleet in ((A1, "F1A"), (A2, "F2A"), (A3, "F3A")):
            m.book_weeks(who, {"fleet": fleet, "weeks": 4})
        quote = m.quote_weeks(A1, {"fleet": "F1B", "weeks": 4})
        if refused(lambda: m.quote_weeks(O, {"fleet": "F1B", "weeks": 1})) != (409, "pair_not_held") or refused(lambda: m.quote_weeks(A1, {"fleet": "F1B", "weeks": 5})) != (409, "pair_not_held") \
                or (quote["first"], quote["end"], quote["price"], quote["discount"], quote["pair"]) != (slot(kst(2027, 1, 11)), slot(kst(2027, 2, 8)), "547.200000", "0.05", "F1A"):
            failures.append(f"a pair is not pinned only by its partner's holder, within the partner's Weeks and at the partner's discount: {quote}")
        m.book_weeks(A1, {"fleet": "F1B", "weeks": 4})
        m.book_weeks(A2, {"fleet": "F2B", "weeks": 4})
        if refused(lambda: m.quote_weeks(A3, {"fleet": "F3B", "weeks": 1})) != (409, "pair_limit") or m.public()["pairs"] != {"open": 0, "limit": 2}:
            failures.append(f"more pairs are pinned than leave the published number of Hour Fleets open: {m.public()['pairs']}")
        halves = dict(m.settings(), fleets=[f for f in m.settings()["fleets"] if f["id"][1] in "12"])
        if m.pair_room(halves, kst(2027, 3, 1))[1] != 1:
            failures.append("with two pairs (64 Sparks) one B half may be pinned while a quarter of the Fleets stays open for hours")
        if any(f["firstWeek"] for f in m.public()["fleets"] if f["id"].endswith("B")):
            failures.append("an Hour Fleet offers a first Week to a new customer")
        NOW[0] = kst(2027, 1, 12, 9)
        states = {f["id"]: f["status"] for f in m.public()["fleets"]}
        board = {f["id"]: f for f in m.public()["fleets"]}
        if board["F1B"]["pinned"] != [[slot(kst(2027, 1, 11)), slot(kst(2027, 2, 8))]] or board["F4B"]["pinned"] != [] or board["F1A"]["booked"] != board["F1A"]["pinned"]:
            failures.append(f"the public state does not show pinned Weeks apart from booked hours: {board['F1B']}")
        if (states["F1B"], states["F3B"], states["F4B"]) != ("pinned", "hour-fleet", "hour-fleet") or refused(lambda: m.book_hours(O, {"fleet": "F1B", "start": slot(kst(2027, 1, 12, 12)), "hours": 1})) != (409, "pinned") \
                or refused(lambda: m.book_hours(O, {"fleet": "F4B", "start": slot(kst(2027, 1, 12, 12)), "hours": 2})):
            failures.append(f"a pinned pair still takes Hours, or an open Hour Fleet does not: {states}")
        m.auto_extend(A2, {"fleet": "F2A", "weeks_ahead": 0})
        NOW[0] = kst(2027, 2, 1, 0, 1)
        m.tick()
        if (through(A1, "F1A"), through(A1, "F1B"), through(A2, "F2B"), through(A3, "F3A")) != (kst(2027, 2, 15), kst(2027, 2, 15), kst(2027, 2, 8), kst(2027, 2, 15)) \
                or app.book.market.extend[(A2, "F2B")]["weeks"] != 1:
            failures.append(f"auto-extend does not keep a pair with its partner, or extends a pair past the partner's Weeks: {through(A1, 'F1B')} {through(A2, 'F2B')}")
        NOW[0] = kst(2027, 2, 7, 12)
        pinned = m.book_weeks(A3, {"fleet": "F3B", "weeks": 1})
        if (pinned["first"], pinned["end"], pinned["price"]) != (slot(kst(2027, 2, 8)), slot(kst(2027, 2, 15)), "144.000000"):
            failures.append(f"a pair that frees up is not open to another partner's holder: {pinned}")
        NOW[0] = kst(2027, 2, 8, 0, 1)
        m.tick()
        stopped = [n for n in notices(app, A2, "booking.deadline") if n["data"].get("reason") == "pair_not_held"]
        if app.book.market.extend.get((A2, "F2B"), {}).get("weeks", 0) != 0 or not stopped:
            failures.append("auto-extend on a pair does not stop when its partner's Weeks end")
    finally:
        server.shutdown()
        mailbox.shutdown()


def pair_deadline_flows(failures, work):
    root = work / "pair-deadline"
    site(root)
    path = root / "site" / "pricing.js"
    pricing = tp.read_pricing(path)
    base = dict(pricing["fleets"][1])
    pricing["fleets"] = [dict(base, id=f"F{n}{side}", name=f"Fleet {n}{side}", state="serving", hourFleet=side == "B", pair=f"F{n}{'B' if side == 'A' else 'A'}")
                         for n in range(1, 5) for side in "AB"]
    tp.write_pricing(path, pricing)
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    server, app, port = live(root, "data", mailbox)
    m = app.market
    try:
        NOW[0] = kst(2027, 1, 6, 12)
        A1, A2, O = (person(app, name, credits) for name, credits in (("a1", 2000), ("a2", 288), ("o", 1000)))
        for fleet in ("F1A", "F1B"):
            m.book_weeks(A1, {"fleet": fleet, "weeks": 2})
            m.auto_extend(A1, {"fleet": fleet, "weeks_ahead": 0})
        NOW[0] = kst(2027, 1, 12, 9)
        if refused(lambda: m.book_hours(A1, {"fleet": "F2B", "start": slot(kst(2027, 1, 12, 12)), "hours": 1})):
            failures.append("the Hours cap on Hour Fleets counts the account's pinned pair Weeks")
        NOW[0] = kst(2027, 1, 13, 12)
        for fleet in ("F3A", "F3B"):
            m.book_weeks(A2, {"fleet": fleet, "weeks": 1})
        NOW[0] = kst(2027, 1, 19, 12)
        m.tick()
        strangers = [refused(lambda: m.book_hours(O, {"fleet": fleet, "start": slot(kst(2027, 1, 25)), "hours": 1})) for fleet in ("F1B", "F3B")]
        if strangers != [(409, "pinned"), (409, "pinned")]:
            failures.append("another customer books Hours on a pinned B half in the Week after its holder's last Week, before the holder's Sunday 00:00 deadline")
        quote = m.quote_weeks(A1, {"fleet": "F1A", "weeks": 1})
        m.book_weeks(A1, {"fleet": "F1A", "weeks": 1, "first": quote["first"]})
        if refused(lambda: m.book_weeks(A1, {"fleet": "F1A", "weeks": 1, "first": quote["first"]})) != (409, "quote_changed") or through(app, A1, "F1A") != kst(2027, 2, 1):
            failures.append("a repeated Week booking with the quoted first Week books a second Week")
        if refused(lambda: m.book_weeks(A1, {"fleet": "F1B", "weeks": 1})) or through(app, A1, "F1B") != kst(2027, 2, 1):
            failures.append("the holder cannot extend its pinned B half before its Sunday 00:00 deadline")
        app.market.credit(A2, Decimal(288), "top up")
        m.tick()
        if through(app, A2, "F3A") != kst(2027, 2, 1) or through(app, A2, "F3B") != kst(2027, 2, 1):
            failures.append(f"auto-extend waiting on a short balance loses the pinned B half before the Sunday 00:00 deadline: {through(app, A2, 'F3B')}")
        NOW[0] = kst(2027, 1, 31, 0, 30)
        m.tick()
        if refused(lambda: m.book_hours(O, {"fleet": "F1B", "start": slot(kst(2027, 2, 1, 1)), "hours": 1})):
            failures.append("a B half whose holder did not book its next Week by the Sunday 00:00 deadline stays closed to Hours")
    finally:
        server.shutdown()
        mailbox.shutdown()


def allocation_flows(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    server, app, port = live(work, "allocation", mailbox)
    m = app.market
    try:
        NOW[0] = kst(2027, 1, 6, 12)
        S, X, Y, Z, Q = (person(app, name, credits) for name, credits in (("s", 6000), ("x", 3200), ("y", 600), ("z", 144), ("q", 10)))
        L1, L2, L3 = (person(app, name) for name in ("l1", "l2", "l3"))
        first = order_lot(app, L1, 200, "26000", "5720000")
        third = order_lot(app, L3, 100, "26000", "2860000")
        app.market.credit(L2, Decimal(100), "a")
        key = m.create_key(S, {"note": "s"})["key"]
        m.book_weeks(S, {"fleet": "F3", "weeks": 52})
        for who, lock in ((X, 3200), (Y, 600), (Z, 144)):
            m.join(who, {"lock": lock})
        NOW[0] = kst(2027, 1, 7, 12)
        app.market.credit(L2, Decimal(150), "b")
        app.market.credit(L3, Decimal(100), "d")
        NOW[0] = kst(2027, 1, 10, 0, 1)
        m.tick()
        granted = {b["account"]: (b["fleet"], b["weeks"], b["hc"], b["unlocked"], b["discount"]) for b in app.book.market.bookings.values() if b["source"] == "waitlist"}
        if granted != {X: ("F4", 26, "3182.400000", "17.600000", "0.15"), Y: ("F5", 4, "547.200000", "52.800000", "0.05")} or m.position(Z) != 1:
            failures.append(f"an allocated lock does not book as many Weeks as it pays for at the discount for that count: {granted} {m.position(Z)}")
        NOW[0] = kst(2028, 1, 6, 11, 59, 59)
        if balance(app, Q) != 10 or balance(app, S) != Decimal("9.6"):
            failures.append("HC lapses before 12 months after issue")
        for who, lock in ((L1, 200), (L2, 250), (L3, 200)):
            m.join(who, {"lock": lock})
        NOW[0] = kst(2028, 1, 6, 12)
        m.tick()
        app.mails.join()
        view = m.view(S)
        if balance(app, Q) != 0 or balance(app, S) != 0 or not view["lots"][0]["expired"] or m.held(S, NOW[0]) != ["F3"] or m.admit(S, m.key_account(key), None) != "F3" or view["bookings"][0]["state"] != "live":
            failures.append(f"HC lapses 12 months after issue, but HC spent on a Week that runs past it must not: {view['bookings']} {m.held(S, NOW[0])}")
        refunds = {who: [r["refund"] for r in app.book.market.refunds[who]] for who in (L1, L2, L3)}
        expect = {L1: [{"order": first, "processor": "toss", "currency": "KRW", "amount": "5720000"}], L2: [None], L3: [{"order": third, "processor": "toss", "currency": "KRW", "amount": "2860000"}]}
        if refunds != expect or [m.position(who) for who in (L1, L2, L3)] != [None, 1, None] or app.book.market.locked(L2) != 150 or balance(app, L3) != 100:
            failures.append(f"locked HC that lapses is not unlocked and recorded as a refund due, or a lock below weekHC stays: {refunds} {[m.position(who) for who in (L1, L2, L3)]}")
        note = next((m_.get_content() for rcpt, m_ in mailbox.mail if app.cfg.notify in rcpt and m_["Subject"] == "Refund due: locked HC lapsed"), "")
        if f"refund 5720000 KRW of order {first} through toss under clause 16.4" not in note or "nothing was paid for it" not in note:
            failures.append(f"we are not emailed the refunds due: {note!r}")
    finally:
        server.shutdown()
        mailbox.shutdown()


def bearer_flows(failures, app, m, port, mailbox):
    P = person(app, "pilot", 2000)
    cli = tp.Client(port)
    status, _ = cli("POST", "/code", {"email": "pilot@example.com"})
    status, signed = cli("POST", "/verify", {"email": "pilot@example.com", "code": tp.last_code(mailbox, "pilot@example.com"), "bearer": True, "device": "laptop"})
    token = signed.get("session", "")
    if status != 200 or not token.startswith("sps-") or cli.cookie is not None or signed["email"] != "pilot@example.com":
        failures.append(f"a CLI sign-in does not return a bearer session without a cookie: {status} {signed}")
        return P
    auth = {"Authorization": f"Bearer {token}"}
    booked = cli("POST", "/market/book-weeks", {"fleet": "F4", "weeks": 1}, headers=auth)
    joined = cli("POST", "/market/waitlist", {"lock": 144}, headers=auth)
    shown = cli("GET", "/market/state", headers=auth)[1]["account"]
    if booked[0] != 200 or booked[1]["first"] != slot(kst(2027, 1, 4)) or joined[1] != {"locked": "144.000000", "position": 1} or shown["email"] != "pilot@example.com" or [s_["device"] for s_ in shown["sessions"]] != ["laptop"]:
        failures.append(f"the CLI does not book a Week and join the waitlist with a bearer session: {booked} {joined}")
    if shown["pinned"][0]["autoExtend"] != 1 or cli("PUT", "/market/auto-extend", {"fleet": "F4", "weeks_ahead": 0}, headers=auth) != (200, {"fleet": "F4", "weeks_ahead": 0, "through": slot(kst(2027, 1, 11))}):
        failures.append("the CLI's first Week does not turn auto-extend on, or the CLI cannot turn it off with its bearer session")
    cli("POST", "/market/waitlist/leave", {}, headers=auth)
    if cli("POST", "/market/book-weeks", {"fleet": "F4", "weeks": 1}, headers={"Authorization": "Bearer sps-wrong"})[0] != 401 or cli("GET", "/market/state", headers={"Authorization": "Bearer sps-wrong"})[1]["account"] is not None:
        failures.append("a wrong bearer session is accepted")
    second = tp.Client(port)
    second("POST", "/code", {"email": "pilot@example.com"})
    other = second("POST", "/verify", {"email": "pilot@example.com", "code": tp.last_code(mailbox, "pilot@example.com"), "bearer": True, "device": "desktop"})[1]["session"]
    if cli("POST", "/signout", {}, headers=auth)[0] != 200 or cli("GET", "/market/state", headers=auth)[1]["account"] is not None or cli("POST", "/market/waitlist", {"lock": 144}, headers=auth)[0] != 401:
        failures.append("signing out does not end the bearer session")
    kept = Path(app.cfg.path("sessions.json"))
    if "laptop" in kept.read_text() or "desktop" not in kept.read_text() or not private(kept) or any("device" in json.dumps(line) or line["type"].startswith("session") for line in ledger_lines(app)):
        failures.append("bearer sessions are kept in the ledger, or a signed-out session is kept, or the session file is readable by others")
    found = second("GET", "/market/state", headers={"Authorization": f"Bearer {other}"})[1]["account"]["sessions"]
    if [s_["device"] for s_ in found] != ["desktop"] or second("POST", "/market/sessions/revoke", {"id": found[0]["id"]}, headers={"Authorization": f"Bearer {other}"})[0] != 200 or second("GET", "/market/state", headers={"Authorization": f"Bearer {other}"})[1]["account"] is not None:
        failures.append("an account cannot see and revoke its own bearer sessions")
    third = tp.Client(port)
    third("POST", "/code", {"email": "pilot@example.com"})
    late = third("POST", "/verify", {"email": "pilot@example.com", "code": tp.last_code(mailbox, "pilot@example.com"), "bearer": True})[1]["session"]
    saved = NOW[0]
    NOW[0] = saved + timedelta(days=89)
    alive = third("GET", "/market/state", headers={"Authorization": f"Bearer {late}"})[1]["account"] is not None
    NOW[0] = saved + timedelta(days=90, seconds=1)
    if not alive or third("GET", "/market/state", headers={"Authorization": f"Bearer {late}"})[1]["account"] is not None:
        failures.append("a bearer session does not last 90 days and then end")
    NOW[0] = saved
    return P


def through(app, account, fleet):
    return max((b["ends"] for b in app.book.market.bookings.values() if b["account"] == account and b["fleet"] == fleet and b["kind"] == "week"), default=None)


def extension_flows(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    upstream = tp.serve(FakeUpstream, seen=[], calls=[])
    server, app, port = live(work, "extension", mailbox, upstream)
    m = app.market
    try:
        NOW[0] = kst(2027, 1, 1, 12)
        P = bearer_flows(failures, app, m, port, mailbox)
        app.market.credit(P, Decimal(8000), "year")
        owner = client(app, port, P)
        if owner("PUT", "/market/auto-extend", {"fleet": "F5", "weeks_ahead": 4}) != (409, {"error": "Auto-extend keeps Weeks you hold: book a Week of Fleet 5 first.", "reason": "not_pinned"}) or owner("PUT", "/market/auto-extend", {"fleet": "F4", "weeks_ahead": 53})[0] != 400:
            failures.append("auto-extend is set on a Fleet the account holds no Week of, or beyond 52 Weeks")
        spent = balance(app, P)
        status, setting = owner("PUT", "/market/auto-extend", {"fleet": "F4", "weeks_ahead": 4})
        if status != 200 or setting != {"fleet": "F4", "weeks_ahead": 4, "through": slot(kst(2027, 2, 1))} or m.view(P)["pinned"][0]["autoExtend"] != 4:
            failures.append(f"auto-extend does not book the Weeks ahead at once: {status} {setting}")
        drift = []
        for week in range(52):
            NOW[0] = kst(2027, 1, 1, 12) + market.WEEK * (week + 1)
            m.tick()
            if through(app, P, "F4") != market.week_start(NOW[0]) + 5 * market.WEEK:
                drift.append((market.kst_text(NOW[0]), through(app, P, "F4")))
        autos = [b for b in app.book.market.bookings.values() if b["account"] == P and b.get("auto")]
        if drift or len(autos) != 53 or any(b["weeks"] != 1 for b in autos[1:]) or {b["discount"] for b in autos[1:]} != {"0.05"} or notices(app, P, "booking.deadline"):
            failures.append(f"auto-extend does not keep 4 Weeks booked ahead across a year, one Week at a time at the discount for the Weeks held: {drift[:3]} {len(autos)} {[(b['weeks'], b['discount']) for b in autos[:3]]}")
        if spent - balance(app, P) != sum(Decimal(b["hc"]) for b in autos) or len(notices(app, P, "booking.confirmed")) != 54:
            failures.append("auto-extended Weeks are not paid from the account's HC, or not confirmed")
        S = person(app, "short", 200)
        m.book_weeks(S, {"fleet": "F5", "weeks": 1})
        first = through(app, S, "F5")
        found = client(app, port, S)("PUT", "/market/auto-extend", {"fleet": "F5", "weeks_ahead": 2})
        warned = notices(app, S, "booking.deadline")
        data = warned[0]["data"] if warned else {}
        if found[1].get("through") != slot(first) or len(warned) != 1 or (data.get("early"), data.get("hc_needed"), data.get("price"), data.get("balance")) != (True, "88.000000", "144.000000", "56.000000") or market.stamp_time(warned[0]["now"]) >= first - market.DAY - 48 * market.HOUR:
            failures.append(f"a short balance does not send booking.deadline early with the HC needed: {found} {warned}")
        m.tick()
        if len(notices(app, S, "booking.deadline")) != 1:
            failures.append("the early warning repeats while the balance stays short")
        app.mails.join()
        if not any(rcpt == ["short@example.com"] and mail["Subject"] == "Auto-extend on Fleet 5 needs 88.000000 more HC" for rcpt, mail in mailbox.mail):
            failures.append("the early warning is not emailed to the account")
        app.market.credit(S, Decimal(100), "top up")
        m.tick()
        if through(app, S, "F5") != first + market.WEEK:
            failures.append("auto-extend does not book once the balance is topped up")
        R = person(app, "ender", 300)
        m.book_weeks(R, {"fleet": "F3", "weeks": 1})
        client(app, port, R)("PUT", "/market/auto-extend", {"fleet": "F3", "weeks_ahead": 1})
        app.book.append("hc_debit", {"account": R, "hc": market.text_hc(balance(app, R)), "reason": "test", "ref": "test", "now": market.stamp(m.now())})
        NOW[0] += market.WEEK
        m.tick()
        warned = [n["data"].get("early") for n in notices(app, R, "booking.deadline")]
        NOW[0] += 3 * market.WEEK
        m.tick()
        if (R, "F3") in app.book.market.extend or warned != [True, None]:
            failures.append("auto-extend does not stop once the pin it keeps has lapsed")
        stop = notices(app, R, "booking.deadline")[-1]
        if (stop["data"].get("stopped"), stop["data"].get("reason"), stop["data"].get("auto_extend")) != (True, "insufficient_hc", 0) or stop["targets"] != ["mail"] or not stop["summary"].startswith("SparkPipe auto-extend stopped on Fleet 3"):
            failures.append(f"the auto-extend stop is not a notice sent like the others and by email: {stop}")
        app.mails.join()
        if not any(rcpt == ["ender@example.com"] and mail["Subject"] == "SparkPipe auto-extend stopped on Fleet 3" for rcpt, mail in mailbox.mail):
            failures.append("the auto-extend stop is not emailed")
        time_based_keys(failures, app, m, port, upstream, P)
        tunnel_end(failures, app, m, port)
        changed_terms(failures, app, m, port, mailbox)
        default_keeps(failures, app, m)
        replay = sp.Book(app.cfg.path("ledger.jsonl"))
        if set(replay.market.bookings) != set(app.book.market.bookings) or replay.market.extend != app.book.market.extend:
            failures.append("a replayed ledger gives different auto-extended bookings or settings")
    finally:
        server.shutdown()
        upstream.shutdown()
        mailbox.shutdown()


def default_keeps(failures, app, m):
    V = person(app, "keeper", 1000)
    m.book_weeks(V, {"fleet": "F5", "weeks": 1})
    first, start, held = through(app, V, "F5"), NOW[0], []
    for week in (1, 2, 3):
        NOW[0] = start + week * market.WEEK
        m.tick()
        held.append((through(app, V, "F5") - first, m.held(V, NOW[0])))
    autos = [b for b in app.book.market.bookings.values() if b["account"] == V and b.get("auto")]
    if held != [(market.WEEK * n, ["F5"]) for n in (1, 2, 3)] or len(autos) != 3 or notices(app, V, "booking.deadline"):
        failures.append(f"with auto-extend on by default, a holder who does nothing does not keep its Fleet week after week without a deadline notice: {held} {len(autos)}")


def changed_terms(failures, app, m, port, mailbox):
    T = person(app, "terms", 1000)
    m.book_weeks(T, {"fleet": "F3", "weeks": 1})
    first = through(app, T, "F3")
    terms = Path(app.cfg.site) / "terms" / "agreement-en.md"
    text = terms.read_text()
    try:
        set_up = client(app, port, T)("PUT", "/market/auto-extend", {"fleet": "F3", "weeks_ahead": 1})
        terms.write_text(text + "\n9.9. A clause added after the auto-extend setting was saved.\n")
        if client(app, port, T)("PUT", "/market/auto-extend", {"fleet": "F3", "weeks_ahead": 1})[1].get("reason") != "terms":
            failures.append("auto-extend is set again without accepting the new terms")
        NOW[0] = first - market.WEEK + market.HOUR
        m.tick()
        m.tick()
        warned = notices(app, T, "booking.deadline")
        waited = [line["data"] for line in ledger_lines(app) if line["type"] == "auto_extend_wait" and line["data"]["account"] == T]
        if set_up[0] != 200 or through(app, T, "F3") != first or len(warned) != 1 or (warned[0]["data"].get("reason"), warned[0]["data"].get("early")) != ("terms", True) or [w["reason"] for w in waited] != ["terms"]:
            failures.append(f"auto-extend books for an account that has not accepted the current terms, or does not warn it once: {through(app, T, 'F3')} {warned} {waited}")
        app.mails.join()
        if not any(rcpt == ["terms@example.com"] and " ".join(mail["Subject"].split()) == "Auto-extend on Fleet 3 waits for you to accept the new Customer Agreement" for rcpt, mail in mailbox.mail):
            failures.append("the account is not emailed that auto-extend waits for its acceptance of the new terms")
        app.accept(T, "127.0.0.1", "test")
        m.tick()
        if through(app, T, "F3") != first + market.WEEK or not next(b for b in app.book.market.bookings.values() if b["account"] == T and b["ends"] == first + market.WEEK).get("auto"):
            failures.append("auto-extend does not book once the account accepts the new terms")
    finally:
        terms.write_text(text)


def time_based_keys(failures, app, m, port, upstream, P):
    owner = client(app, port, P)
    expiring = owner("POST", "/v1/keys", {"expires": market.stamp(m.now() + market.HOUR)})[1]
    works = raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": expiring["key"]})[0]
    NOW[0] += 2 * market.HOUR
    if works != 200 or raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": expiring["key"]})[0] != 401:
        failures.append("a key works after its expiry, or not before it")
    old = owner("POST", "/v1/keys", {"scopes": ["inference", "storage"], "fleet": "F4", "allow_swap": True, "ips": ["127.0.0.1/32"], "priority": "background", "note": "ci"})[1]
    status, new = owner("POST", f"/v1/keys/{old['id']}/rotate")
    same = all(new.get(name) == old[name] for name in ("scopes", "fleet", "allow_swap", "ips", "priority", "note"))
    before = [raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": key})[0] for key in (old["key"], new.get("key"))]
    if status != 200 or not same or new["replaces"] != old["id"] or new["retires"] != market.stamp(m.now() + market.HOUR) or before != [200, 200] or owner("POST", f"/v1/keys/{old['id']}/rotate")[0] != 409:
        failures.append(f"a rotated key does not keep its options, or the old key stops before an hour: {status} {new} {before}")
    NOW[0] += market.HOUR - timedelta(seconds=1)
    still = raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": old["key"]})[0]
    NOW[0] += timedelta(seconds=1)
    after = [raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": key})[0] for key in (old["key"], new["key"])]
    if still != 200 or after != [401, 200]:
        failures.append(f"the old key of a rotation is not valid for exactly an hour: {still} {after}")


def tunnel_end(failures, app, m, port):
    Q = person(app, "quick", 100)
    start = market.slot_time(market.slot_text(m.now())) + market.HOUR
    m.book_hours(Q, {"fleet": "F1", "start": slot(start), "hours": 1})
    m.book_hours(Q, {"fleet": "F1", "start": slot(start + 2 * market.HOUR), "hours": 1})
    key = m.create_key(Q, {})["key"]
    NOW[0] = start + timedelta(minutes=30)
    saved, sp.TUNNEL_CHECK = sp.TUNNEL_CHECK, 0.2
    try:
        sock, head = websocket(port, key)
        sock.sendall(b"ping")
        first = sock.recv(100)
        NOW[0] = start + market.HOUR + timedelta(seconds=10)
        time.sleep(0.8)
        sock.sendall(b"late")
        late = sock.recv(100)
        NOW[0] = start + market.HOUR + timedelta(seconds=31)
        sock.settimeout(5)
        try:
            closed = sock.recv(100) == b""
        except OSError:
            closed = False
        sock.close()
        if not head.startswith(b"HTTP/1.1 101") or first != b"echo:ping" or late != b"echo:late" or not closed:
            failures.append(f"a WebSocket is not given 30 seconds after the hold ends and then closed: {head[:40]!r} {first!r} {late!r} {closed}")
        NOW[0] = start + 2 * market.HOUR + timedelta(minutes=30)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request("POST", "/v1/responses", json.dumps({"slow": True}).encode(), {"X-Api-Key": key, "Content-Type": "application/json"})
        response = connection.getresponse()
        began = response.read1(64)
        NOW[0] = start + 3 * market.HOUR + timedelta(seconds=31)
        moment = time.monotonic()
        try:
            while response.read1(65536):
                pass
        except (OSError, http.client.HTTPException):
            pass
        took = time.monotonic() - moment
        connection.close()
        if not began.startswith(b"data: tick") or took > 3:
            failures.append(f"an HTTP stream runs on past the end of the hold and its 30 seconds: {began!r} {took:.1f}s")
        NOW[0] = start + 2 * market.HOUR + timedelta(minutes=40)
        revoked = m.create_key(Q, {})
        sock, head = websocket(port, revoked["key"])
        sock.sendall(b"one")
        echoed = sock.recv(100)
        m.revoke_key(Q, revoked["id"])
        sock.settimeout(5)
        try:
            cut = sock.recv(100) == b""
        except OSError:
            cut = False
        sock.close()
        if echoed != b"echo:one" or not cut:
            failures.append(f"a WebSocket stays open after its key is revoked: {echoed!r} {cut}")
    finally:
        sp.TUNNEL_CHECK = saved


def signed(secret, headers, body):
    stamp, _, given = (headers.get("x-sparkpipe-signature") or "").partition(",v1=")
    moment = stamp[2:] if stamp.startswith("t=") else ""
    return moment.isdigit() and abs(int(moment) - time.time()) < 300 and hmac.compare_digest(given, hmac.new(secret.encode(), moment.encode() + b"." + body, hashlib.sha256).hexdigest())


def notice_flows(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    good, slack, flaky, stopper = (tp.serve(Receiver, posts=[], fail=fail) for fail in (0, 0, 2, 0))
    drip = tp.serve(Drip, posts=[])
    saved = sp.WEBHOOK_SCHEMES, sp.allowed_address, sp.RETRY_SECONDS, sp.WEBHOOK_SECONDS
    sp.WEBHOOK_SCHEMES, sp.allowed_address, sp.RETRY_SECONDS, sp.WEBHOOK_SECONDS = ("https", "http"), lambda text: True, (0.05,) * 7, 1.5
    server, app, port = live(work, "notices", mailbox)
    m = app.market
    try:
        NOW[0] = kst(2027, 1, 6, 12)
        W, O = person(app, "watcher", 1000), person(app, "other", 1000)
        owner = client(app, port, W)
        url = lambda receiver: f"http://127.0.0.1:{receiver.server_address[1]}/hook"
        status, everything = owner("PUT", "/v1/webhooks", {"url": url(good), "events": ["*"], "secret": "s" * 24})
        status2, chat_hook = owner("PUT", "/v1/webhooks", {"url": url(slack), "events": ["booking.confirmed"], "slack": True})
        status3, retried = owner("PUT", "/v1/webhooks", {"url": url(flaky), "events": ["booking.confirmed", "working_state.lost"], "secret": "f" * 24})
        if [status, status2, status3] != [200, 200, 200] or "secret" in everything or not chat_hook.get("secret", "").startswith("whsec_") or everything["hint"] != "ssss":
            failures.append(f"webhooks are not set with a given or generated secret: {everything} {chat_hook} {retried}")
            return
        listed = owner("GET", "/v1/webhooks")[1]
        if [h.get("url") or h.get("email") for h in listed["data"]] != sorted([url(good), url(slack), url(flaky)], key=lambda u: [everything, chat_hook, retried][[url(good), url(slack), url(flaky)].index(u)]["id"]) + ["watcher@example.com"] or "secret" in json.dumps(listed) or listed["events"] != list(market.EVENTS):
            failures.append(f"the webhook list shows secrets, or not the default email notices: {listed}")
        stuck = owner("PUT", "/v1/webhooks", {"url": url(drip), "events": ["booking.confirmed"], "secret": "d" * 24})[1]
        m.book_hours(W, {"fleet": "F1", "start": slot(kst(2027, 1, 6, 14)), "hours": 1})
        confirmed = lambda: any(json.loads(body)["type"] == "booking.confirmed" for _, body in good.posts)
        if not until(confirmed, 1) or not until(lambda: len(drip.posts) >= 2, 8) or drip.posts[1] - drip.posts[0] > 2.5:
            failures.append(f"a slow webhook receiver holds up other deliveries, or one delivery runs past its deadline: {confirmed()} {[round(t - drip.posts[0], 2) for t in drip.posts]}")
        owner("DELETE", f"/v1/webhooks/{stuck['id']}")
        N = person(app, "lapsing", 144)
        client(app, port, N)("PUT", "/v1/webhooks", {"url": url(stopper), "events": ["booking.deadline"], "secret": "n" * 24})
        m.book_weeks(N, {"fleet": "F5", "weeks": 1})
        m.join(O, {"lock": 200})
        m.join(W, {"lock": 300})
        m.join(O, {"lock": 200})
        NOW[0] = kst(2027, 1, 10)
        m.tick()
        mine = next(b for b in app.book.market.bookings.values() if b["account"] == W and b["source"] == "waitlist")
        if owner("PUT", "/market/auto-extend", {"fleet": mine["fleet"], "weeks_ahead": 0})[1].get("weeks_ahead") != 0:
            failures.append("the holder of a Week granted by the waitlist cannot turn auto-extend off")
        m.schedule_maintenance(mine["fleet"], "2027-01-18")
        NOW[0] = kst(2027, 1, 18, 3)
        m.tick()
        deadline = mine["ends"] - market.DAY
        NOW[0] = deadline - 73 * market.HOUR
        m.tick()
        early = len(notices(app, W, "booking.deadline"))
        for hours_before in (71, 47, 7, 5, 1):
            NOW[0] = deadline - hours_before * market.HOUR
            m.tick()
        m.tick()
        hours = [n["data"]["hours"] for n in notices(app, W, "booking.deadline")]
        if early or hours != [72, 48, 6, 2] or notices(app, W, "booking.deadline")[0]["data"]["deadline"] != slot(deadline) or {n["data"]["auto_extend"] for n in notices(app, W, "booking.deadline")} != {0}:
            failures.append(f"booking.deadline does not fire once each 72, 48, 6 and 2 hours before the Sunday deadline: {early} {hours}")
        until(lambda: len(stopper.posts) >= 2, 5)
        sent_stop = [payload["data"] for payload in sorted((json.loads(body) for _, body in stopper.posts), key=lambda payload: payload["created"])]
        if [(d.get("early"), d.get("reason"), d.get("stopped")) for d in sent_stop] != [(True, "insufficient_hc", None), (None, "insufficient_hc", True)] or not all(signed("n" * 24, headers, body) for headers, body in stopper.posts):
            failures.append(f"a webhook does not receive the auto-extend wait and stop notices, signed, in the order they were created: {sent_stop}")
        pair_loss(failures, app, m, port, W, O, mine["fleet"])
        app.uptime.observe("F5", True, m.now(), watch=False)
        app.uptime.observe(mine["fleet"], False, m.now(), watch=False)
        if m.route(W, mine["fleet"]) != ("F5", mine["fleet"]):
            failures.append("a holder whose Fleet is down is not routed to a free Fleet")
        app.uptime.observe(mine["fleet"], True, m.now(), watch=False)
        m.tick()
        events = [{"id": f"e{n}", "account": W, "fleet": mine["fleet"], "type": kind, "data": {"run": "run_1", "function": "review"}} for n, kind in enumerate(market.HUB_EVENTS)]
        status, answer = hub(port, "/hub/events", {"events": events})
        again = hub(port, "/hub/events", {"events": events[:1] + [{"id": "x", "account": W, "fleet": "F1", "type": "booking.confirmed"}]})[1]
        if status != 200 or any(item.get("duplicate") is not False for item in answer["events"]) or again["events"][0] != {"id": "e0", "duplicate": True} or again["events"][1].get("reason") != "invalid_request" or hub(port, "/hub/events", {"events": events}, "wrong")[0] != 401:
            failures.append(f"the hub's events are not taken in once each, with only the hub's event types and the upstream key: {answer} {again}")
        types = lambda receiver: {json.loads(body).get("type") for _, body in receiver.posts}
        if not until(lambda: types(good) >= set(market.EVENTS)):
            failures.append(f"not every event of §13 reaches the webhook: {set(market.EVENTS) - types(good)}")
        bad = [(headers.get("x-sparkpipe-event"), body[:80]) for headers, body in good.posts if not signed("s" * 24, headers, body) or headers.get("x-sparkpipe-event") != json.loads(body)["type"]]
        if bad or {json.loads(body)["account"] for _, body in good.posts} != {W}:
            failures.append(f"webhook deliveries are not signed with HMAC-SHA256 in X-SparkPipe-Signature, or carry other accounts: {bad[:3]}")
        sample = next(json.loads(body) for _, body in good.posts if json.loads(body)["type"] == "working_state.lost")
        if set(sample) != {"id", "type", "created", "account", "summary", "data"} or sample["data"]["contexts"] != ["conv-a", "conv-b"] or sample["data"]["hc"] != "24.000000":
            failures.append(f"a webhook body is not the event with its data: {sample}")
        until(lambda: len(slack.posts) >= 1 and len(flaky.posts) >= 5)
        if [json.loads(body) for _, body in slack.posts] != [{"text": "Booked Fleet 1: 1 hours from 2027-01-06 14:00 KST"}] or not all(signed(chat_hook["secret"], headers, body) for headers, body in slack.posts):
            failures.append(f"a Slack-compatible webhook does not get the event's text, signed: {slack.posts}")
        delivered = [json.loads(body)["type"] for headers, body in flaky.posts]
        if len(flaky.posts) != 5 or sorted(set(delivered)) != ["booking.confirmed", "working_state.lost"] or not all(signed("f" * 24, headers, body) for headers, body in flaky.posts):
            failures.append(f"a failed delivery is not retried until it succeeds: {delivered}")
        app.mails.join()
        subjects = sorted(" ".join(mail["Subject"].split()) for rcpt, mail in mailbox.mail if rcpt == ["watcher@example.com"])
        expect = ["Book the next Week of Fleet 4 by " + market.kst_text(deadline)] * 4 + ["SparkPipe Fleet 4 is yours from 2027-01-11", "SparkPipe credited 3.000000 HC for maintenance on Fleet 4"] + ["SparkPipe Fleet 4 lost part of your working state: 24.000000 HC credited"] * 2 + [
                  "SparkPipe serves you on Fleet 5 while Fleet 4 is down", "SparkPipe stopped serving you on Fleet 5"] + [
                  "SparkPipe maintenance on Fleet 4, 2027-01-18 00:00 KST"]
        if mine["fleet"] != "F4" or subjects != sorted(expect):
            failures.append(f"the default email notices are not booking.deadline, fleet.allocated, downtime.credited and working_state.lost: {subjects}")
        status, quiet = owner("PUT", "/v1/webhooks", {"email": "watcher@example.com", "events": []})
        listed = owner("GET", "/v1/webhooks")[1]["data"]
        if status != 200 or quiet.get("email") != "watcher@example.com" or any(h.get("default") for h in listed) or m.targets(W, "booking.deadline") != [everything["id"]] or m.targets(W, "booking.confirmed") != sorted([everything["id"], chat_hook["id"], retried["id"]]):
            failures.append(f"email notices cannot be turned off: {quiet} {listed}")
        if any(m.targets(W, kind)[-1:] != ["mail"] for kind in ("fleet.allocated", "working_state.lost", "downtime.credited")) or quiet.get("always") != ["fleet.allocated", "downtime.credited", "working_state.lost"]:
            failures.append(f"the notices the Agreement requires are not emailed once email notices are off: {[m.targets(W, kind) for kind in market.REQUIRED_EMAIL]} {quiet}")
        before = len(mailbox.mail)
        hub(port, "/hub/pair-loss", {"fleet": mine["fleet"], "event": "loss-quiet", "account": W, "contexts": ["conv-q"]})
        if sp.credit(app, ["sparkpipe_pay.py", "credit", "watcher@example.com", "5", "Fleet 4 unreachable from 09:00 to 09:40 KST on 19 January"]) != 0:
            failures.append("the operator credit command fails")
        app.mails.join()
        quiet_subjects = sorted(" ".join(mail["Subject"].split()) for rcpt, mail in mailbox.mail[before:] if rcpt == ["watcher@example.com"])
        credited = notices(app, W, "downtime.credited")[-1]["data"]
        if quiet_subjects != ["SparkPipe Fleet 4 lost part of your working state: 24.000000 HC credited", "SparkPipe credited 5.000000 HC for downtime"] or (credited.get("clause"), credited.get("hc")) != ("4.4", "5.000000"):
            failures.append(f"with email notices off, a pair loss and an operator credit under clause 4.4 are not emailed with downtime.credited: {quiet_subjects} {credited}")
        if owner("PUT", "/v1/webhooks", {"email": "someone@else.example", "events": ["booking.deadline"]})[0] != 400 or owner("PUT", "/v1/webhooks", {"url": url(good), "events": ["nope"]})[0] != 400:
            failures.append("notices go to another address than the account's, or to an unknown event")
        hooks = Path(app.cfg.path("webhooks.json"))
        if owner("DELETE", f"/v1/webhooks/{retried['id']}")[1] != {"id": retried["id"], "deleted": True} or retried["id"] in m.targets(W, "working_state.lost") or retried["id"] in hooks.read_text() or url(flaky) in hooks.read_text() or not private(hooks):
            failures.append("a deleted webhook keeps its secret, address or events, or the webhook file is readable by others")
        if "watcher@example.com" in hooks.read_text():
            failures.append("an email subscription stores the address")
        text = Path(app.cfg.path("ledger.jsonl")).read_text()
        kinds = {line["type"] for line in ledger_lines(app)}
        mailed = {line["type"] for line in ledger_lines(app) if "watcher@example.com" in json.dumps(line)}
        if any(found in text for found in ("conv-a", "conv-q", "models/x/", "sparkg", "run_rebuild", "/hook", "whsec_", "run_1")) or kinds & {"notice", "webhook", "webhook_delete", "pair_loss", "session"} or mailed - {"account"}:
            failures.append(f"the ledger holds notices, webhooks, lost paths, context names or hub event data: {sorted(kinds)} {sorted(mailed)}")
        if raw_call(port, "PUT", "/v1/webhooks", json.dumps({"url": url(good), "events": ["*"]}).encode(), {"X-Api-Key": m.create_key(W, {})["key"]})[0] != 403:
            failures.append("an API key manages webhooks")
        until(lambda: any(o["target"] == stuck["id"] for o in app.notifier.deliveries.items()) and sum(o["attempts"] for o in app.notifier.deliveries.items() if o["target"] == retried["id"]) >= 6, 5)
        outcomes = [o for o in app.notifier.deliveries.items() if o["target"] != stuck["id"]]
        if {o["outcome"] for o in outcomes} != {"delivered"} or sum(o["attempts"] for o in outcomes if o["target"] == retried["id"]) != 6 or {o["attempts"] for o in outcomes if o["target"] != retried["id"]} != {1}:
            failures.append(f"webhook outcomes are not recorded: {[o for o in outcomes if o['target'] == retried['id']]}")
        if [o["outcome"] for o in app.notifier.deliveries.items() if o["target"] == stuck["id"]] != ["deleted"]:
            failures.append("a delivery to a deleted webhook is not dropped")
        later = m.now() + timedelta(days=sp.NOTICE_DAYS + 1)
        app.notices.prune(later - timedelta(days=1))
        kept = len(app.notices.items())
        app.notices.prune(later)
        app.notifier.deliveries.prune(later)
        if not kept or app.notices.items() or app.notifier.deliveries.items():
            failures.append(f"notices and delivery outcomes are not kept {sp.NOTICE_DAYS} days and then deleted")
    finally:
        server.shutdown()
        for receiver in (good, slack, flaky, drip, stopper):
            receiver.shutdown()
        mailbox.shutdown()
        sp.WEBHOOK_SCHEMES, sp.allowed_address, sp.RETRY_SECONDS, sp.WEBHOOK_SECONDS = saved


def pair_loss(failures, app, m, port, W, O, fleet):
    before = balance(app, W)
    report = {"fleet": fleet, "event": "loss-1", "account": W, "pair": ["sparkg", "sparkh"], "contexts": ["conv-a", "conv-b"], "storage": ["models/x/"], "run": "run_rebuild"}
    status, found = hub(port, "/hub/pair-loss", report)
    lots = [lot for lot in app.book.market.lots[W] if lot["source"] == "rebuild"]
    if status != 200 or found["hc"] != "24.000000" or balance(app, W) - before != 24 or [lot["ref"] for lot in lots] != [f"{fleet}:loss-1"] or found["duplicate"]:
        failures.append(f"a pair-loss report does not credit 24 hours of the Fleet at its Rate at once: {status} {found}")
    again = hub(port, "/hub/pair-loss", report)[1]
    if not again.get("duplicate") or again["lot"] != found["lot"] or balance(app, W) - before != 24 or len(notices(app, W, "working_state.lost")) != 1:
        failures.append(f"the same pair-loss event is credited twice: {again}")
    lost = notices(app, W, "working_state.lost")[0]["data"]
    if (lost["contexts"], lost["storage"], lost["hc"], lost["run"]) != (["conv-a", "conv-b"], ["models/x/"], "24.000000", "run_rebuild"):
        failures.append(f"working_state.lost does not say what was lost, the credit and the rebuild run: {lost}")
    refusals = [hub(port, "/hub/pair-loss", dict(report, event="loss-2"), "wrong")[0], hub(port, "/hub/pair-loss", dict(report, event="loss-3", account=O))[0], hub(port, "/hub/pair-loss", dict(report, event="loss-4", fleet="F9"))[0],
                hub(port, "/hub/pair-loss", dict(report, event="loss-5", at="2099-01-01T00:00:00Z"))[0], hub(port, "/hub/pair-loss", {"fleet": fleet, "account": W})[0]]
    if refusals != [401, 409, 404, 400, 400] or balance(app, W) - before != 24:
        failures.append(f"a pair-loss report is credited with a wrong key, for an account that did not hold the Fleet, an unknown Fleet, a future time or no event: {refusals}")
    crash = dict(report, event="loss-crash")

    def crashed(*args, **kwargs):
        raise RuntimeError("the service stopped after writing the credit")

    m.notify = crashed
    try:
        m.pair_loss(crash)
    except RuntimeError:
        pass
    finally:
        del m.notify
    replayed = [lot for lot in sp.Book(app.cfg.path("ledger.jsonl")).market.lots[W] if lot["ref"] == f"{fleet}:loss-crash"]
    middle = balance(app, W)
    status, again = hub(port, "/hub/pair-loss", crash)
    resent = hub(port, "/hub/pair-loss", crash)[1]
    told = [n for n in notices(app, W, "working_state.lost") if n["data"]["event"] == "loss-crash"]
    if len(replayed) != 1 or middle - before != 48 or status != 200 or not again["duplicate"] or not resent["duplicate"] or balance(app, W) != middle or len(told) != 1 or told[0]["data"]["contexts"] != ["conv-a", "conv-b"]:
        failures.append(f"a pair loss reported again after a crash is credited twice, or its notice is lost: {replayed} {again} {told}")


def probe(app, moment, *states):
    NOW[0] = moment
    for upstream, healthy in states:
        upstream.healthy = healthy
    app.uptime.probe_all()
    app.market.tick()


def downtime_lots(app, account):
    return [(lot["first"], [market.kst_text(market.slot_time(hour))[5:16] for hour in lot["hours"]]) for lot in app.book.market.lots[account] if lot["source"] == "downtime"]


def public_status(port):
    status, _, raw = call(port, "GET", "/status")
    return status, {f["id"]: f for f in json.loads(raw)["fleets"]} if status == 200 else {}


def uptime_flows(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    one, two = (tp.serve(FakeUpstream, seen=[], calls=[], healthy=True) for _ in range(2))
    extra = {"SPARKPIPE_FLEET_UPSTREAMS": f"F4=http://127.0.0.1:{two.server_address[1]}/v1"}
    server, app, port = live(work, "uptime", mailbox, one, extra)
    saved, sp.UPTIME_GAP_SECONDS = sp.UPTIME_GAP_SECONDS, 10 ** 9
    try:
        NOW[0] = kst(2027, 1, 8, 12)
        H, R = person(app, "held", 1000), person(app, "renter", 100)
        app.market.book_weeks(H, {"fleet": "F3", "weeks": 3})
        planned = app.market.schedule_maintenance("F3", "2027-01-25")
        NOW[0] = kst(2027, 1, 11, 12)
        app.market.book_hours(R, {"fleet": "F4", "start": slot(kst(2027, 1, 12, 10)), "hours": 3})
        probe(app, kst(2027, 1, 12, 10, 50))
        probe(app, kst(2027, 1, 12, 10, 55), (one, False), (two, False))
        probe(app, kst(2027, 1, 12, 11, 5))
        early = downtime_lots(app, H)
        probe(app, kst(2027, 1, 12, 11, 6))
        first, shown = downtime_lots(app, H), public_status(port)[1]
        probe(app, kst(2027, 1, 12, 12, 59))
        middle = downtime_lots(app, H)
        probe(app, kst(2027, 1, 12, 13), (one, True), (two, True))
        if early or first != [("2.000000", ["01-12 10:00", "01-12 11:00"])] or middle != first or downtime_lots(app, H) != first + [("1.000000", ["01-12 12:00"])]:
            failures.append(f"a held Fleet down for more than 15 minutes does not credit each booked hour the period touches, as soon as it passes 15 minutes and at its end: {early} {first} {middle} {downtime_lots(app, H)}")
        if downtime_lots(app, R) != [("2.000000", ["01-12 10:00", "01-12 11:00"]), ("1.000000", ["01-12 12:00"])]:
            failures.append(f"an Hour holder on a Fleet with its own upstream is not credited from that Fleet's records: {downtime_lots(app, R)}")
        if (shown["F3"]["status"], shown["F3"]["since"], shown["F4"]["status"], shown["F1"]["status"]) != ("down", market.stamp(kst(2027, 1, 12, 10, 50)), "down", "down"):
            failures.append(f"the status page does not show a Fleet down since its last good probe: {shown}")
        told = notices(app, H, "downtime.credited")
        app.mails.join()
        if [n["data"]["hours"] for n in told] != [[slot(kst(2027, 1, 12, h)) for h in (10, 11)], [slot(kst(2027, 1, 12, 12))]] or {n["data"]["clause"] for n in told} != {"4.4"} or not all(n["targets"][-1:] == ["mail"] for n in told):
            failures.append(f"each downtime credit does not send downtime.credited with its hours, by email: {told}")
        subjects = [" ".join(mail["Subject"].split()) for rcpt, mail in mailbox.mail if rcpt == [app.cfg.notify] and " F3 " in mail["Subject"]]
        if not any(s.startswith("Fleet down: F3 since") for s in subjects) or not any(s.startswith("Fleet back: F3 from") for s in subjects):
            failures.append(f"we are not emailed when a Fleet goes down for more than 15 minutes and when it is back: {subjects}")
        probe(app, kst(2027, 1, 12, 14))
        probe(app, kst(2027, 1, 12, 14, 1), (one, False))
        blip = public_status(port)[1]
        probe(app, kst(2027, 1, 12, 14, 10), (one, True))
        probe(app, kst(2027, 1, 12, 15))
        probe(app, kst(2027, 1, 12, 15, 1), (one, False))
        probe(app, kst(2027, 1, 12, 15, 20), (one, True))
        probe(app, kst(2027, 1, 12, 15, 30), (one, False))
        probe(app, kst(2027, 1, 12, 15, 50), (one, True))
        if (blip["F3"]["status"], blip["F4"]["status"]) != ("down", "up") or downtime_lots(app, H)[2:] != [("1.000000", ["01-12 15:00"])]:
            failures.append(f"a period of 15 minutes or less is credited, two periods in one hour credit it twice, or Fleets on different upstreams share a status: {downtime_lots(app, H)} {blip}")
        probe(app, kst(2027, 1, 12, 16))
        server.shutdown()
        sp.UPTIME_GAP_SECONDS = saved
        server, app, port = live(work, "uptime", mailbox, one, extra)
        probe(app, kst(2027, 1, 12, 16, 40))
        gap = [s for s in app.uptime.spans(kst(2027, 1, 12)) if s["fleet"] == "F3" and s["cause"] == "gateway"]
        if downtime_lots(app, H)[3:] != [("1.000000", ["01-12 16:00"])] or [(s["start"], s["end"]) for s in gap] != [(market.stamp(kst(2027, 1, 12, 16)), market.stamp(kst(2027, 1, 12, 16, 40)))]:
            failures.append(f"40 minutes in which the service saw no good probe, across a restart, are not a period the Fleet was unreachable: {gap} {downtime_lots(app, H)}")
        sp.UPTIME_GAP_SECONDS = 10 ** 9
        probe(app, kst(2027, 1, 24, 23, 50))
        probe(app, kst(2027, 1, 24, 23, 55), (one, False))
        probe(app, kst(2027, 1, 25, 3, 20), (one, True))
        maintained = [lot["first"] for lot in app.book.market.lots[H] if lot["source"] == "maintenance"]
        if planned["start"] != slot(kst(2027, 1, 25)) or downtime_lots(app, H)[4:] != [("2.000000", ["01-24 23:00", "01-25 03:00"])] or maintained != ["3.000000"]:
            failures.append(f"booked hours under maintenance are credited twice, or the hours around it are not credited: {planned} {downtime_lots(app, H)[4:]} {maintained}")
        probe(app, kst(2027, 1, 26, 10))
        probe(app, kst(2027, 1, 26, 10, 5), (one, False))
        probe(app, kst(2027, 1, 26, 10, 20))
        probe(app, kst(2027, 1, 27, 10, 30))
        probe(app, kst(2027, 1, 27, 12), (one, True))
        long = [(hc, len(hours)) for hc, hours in downtime_lots(app, H)[5:]]
        if long != [("1.000000", 1), ("24.000000", 24), ("1.000000", 1)]:
            failures.append(f"a period that goes on for a day is not credited once it passes 15 minutes, again each day, and at its end: {long}")
        status, fleets = public_status(port)
        f3 = fleets.get("F3", {})
        if status != 200 or (f3.get("status"), f3.get("since")) != ("up", market.stamp(kst(2027, 1, 27, 12))) or [i["creditable"] for i in f3.get("incidents", [])] != [True, True, True, True, True, False, True]:
            failures.append(f"the status page does not list each period, newest first, saying which are over 15 minutes: {f3}")
        if f3.get("uptime", {}).get("24h") != 0.0 or f3["uptime"]["7d"] != 84.226 or fleets["F4"]["uptime"]["7d"] != 100.0 or fleets["F4"]["since"] != market.stamp(kst(2027, 1, 12, 16, 40)) or H in json.dumps(fleets) or R in json.dumps(fleets):
            failures.append(f"the status page uptime is not measured from the records, without maintenance, or it names accounts: {f3.get('uptime')} {fleets.get('F4', {}).get('uptime')}")
        two.shutdown()
        two.server_close()
        probe(app, kst(2027, 1, 27, 12, 1))
        refused_state = app.uptime.entry("F4")
        if public_status(port)[1]["F4"]["status"] != "down" or "ConnectionRefused" not in refused_state.get("detail", ""):
            failures.append(f"a hub that does not answer is not recorded as down: {refused_state}")
        lines = client(app, port, H)("GET", "/market/ledger")[1]["lines"]
        if sum(1 for line in lines if line["type"] == "hc_lot" and line["data"]["source"] == "downtime") != 8 or any(line["type"] == "downtime" for line in lines):
            failures.append("the account's ledger does not show its downtime credits, or shows other accounts' credit records")
        replay = sp.Book(app.cfg.path("ledger.jsonl"))
        if replay.market.downtime_hours != app.book.market.downtime_hours or set(replay.market.downtime) != set(app.book.market.downtime):
            failures.append("a replayed ledger gives different downtime credits")
        server.shutdown()
        server, app, port = live(work, "uptime", mailbox, one, extra)
        before = balance(app, H)
        app.market.tick()
        if balance(app, H) != before or len(downtime_lots(app, H)) != 8:
            failures.append("a restarted service credits downtime again")
        if sp.uptime(app, ["sparkpipe_pay.py", "uptime", "F3", "2027-01-12", "2027-01-13"]) != 0:
            failures.append("the operator cannot list a Fleet's uptime records")
    finally:
        sp.UPTIME_GAP_SECONDS = saved
        server.shutdown()
        for upstream in (one, two):
            upstream.shutdown()
        mailbox.shutdown()


def ask(port, key):
    status, headers, raw = raw_call(port, "POST", "/v1/chat/completions", json.dumps({"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]}).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        found = json.loads(raw)
    except ValueError:
        found = {}
    return status, headers.get("x-sparkpipe-failover-from"), (found.get("sparkpipe") or {}).get("fleet") or (found.get("error") or {}).get("code")


def failover_flows(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    hubs = {fleet: tp.serve(FakeUpstream, seen=[], calls=[], healthy=True) for fleet in ("F1", "F3", "F4", "F5")}
    extra = {"SPARKPIPE_FLEET_UPSTREAMS": " ".join(f"{fleet}=http://127.0.0.1:{hub.server_address[1]}/v1" for fleet, hub in hubs.items())}
    server, app, port = live(work, "failover", mailbox, hubs["F1"], extra)
    m = app.market
    try:
        NOW[0] = kst(2027, 1, 8, 12)
        H, G, R = person(app, "holder", 1000), person(app, "second", 1000), person(app, "renter", 100)
        m.book_weeks(H, {"fleet": "F3", "weeks": 1})
        m.book_weeks(G, {"fleet": "F4", "weeks": 1})
        keys = {who: m.create_key(who, {})["key"] for who in (H, G, R)}
        probe(app, kst(2027, 1, 12, 10))
        normal = ask(port, keys[H])
        probe(app, kst(2027, 1, 12, 10, 5), (hubs["F3"], False))
        moved = ask(port, keys[H])
        got = hubs["F5"].calls[-1] if hubs["F5"].calls else {"items": []}
        if normal != (200, None, "F3") or moved != (200, "F3", "F5") or [sent(got, name) for name in ("x-sparkpipe-fleet", "x-sparkpipe-account", "x-sparkpipe-failover-from")] != [["F5"], [H], ["F3"]]:
            failures.append(f"a holder's requests are not served on a free Fleet's hub while its own Fleet is down, as that Fleet with the Fleet it replaces: {normal} {moved} {got.get('items')}")
        if feed(port, "F5", slot(kst(2027, 1, 12, 9)), slot(kst(2027, 1, 12, 12)))[1]["hours"] != [{"start": slot(kst(2027, 1, 12, 10)), "account": H}]:
            failures.append(f"the free Fleet's holder schedule does not name the account it serves under clause 15.7, so the hub cannot keep or wipe its working state: {feed(port, 'F5', slot(kst(2027, 1, 12, 9)), slot(kst(2027, 1, 12, 12)))}")
        probe(app, kst(2027, 1, 12, 10, 10), (hubs["F4"], False))
        second = ask(port, keys[G])
        m.book_hours(R, {"fleet": "F5", "start": slot(kst(2027, 1, 12, 11)), "hours": 1})
        if second != (200, "F4", "F1"):
            failures.append(f"a second holder whose Fleet is down is not served on another free Fleet, the Demo Fleet last: {second}")
        probe(app, kst(2027, 1, 12, 11, 5))
        stranded, renter = ask(port, keys[H]), ask(port, keys[R])
        if stranded != (503, None, "fleet_down") or renter != (200, None, "F5") or feed(port, "F5", slot(kst(2027, 1, 12, 10)), slot(kst(2027, 1, 12, 12)))[1]["hours"] != [{"start": slot(kst(2027, 1, 12, 10)), "account": H}, {"start": slot(kst(2027, 1, 12, 11)), "account": R}]:
            failures.append(f"a failover keeps a Fleet someone has booked, or a holder with no free Fleet left is not told so: {stranded} {renter}")
        probe(app, kst(2027, 1, 12, 11, 30), (hubs["F3"], True))
        back = ask(port, keys[H])
        probe(app, kst(2027, 1, 12, 11, 40), (hubs["F4"], True))
        home = ask(port, keys[G])
        events = [(n["data"]["state"], n["data"]["to"], n["data"].get("reason")) for n in notices(app, H, "fleet.failover") + notices(app, G, "fleet.failover")]
        if back != (200, None, "F3") or home != (200, None, "F4") or events != [("started", "F5", None), ("ended", "F5", "taken"), ("started", "F1", None), ("ended", "F1", "recovered")]:
            failures.append(f"requests do not go back to the held Fleet once it serves again, or fleet.failover is not sent when serving on another Fleet starts and ends: {back} {home} {events}")
        app.mails.join()
        if not any(rcpt == ["holder@example.com"] and mail["Subject"] == "SparkPipe serves you on Fleet 5 while Fleet 3 is down" and "without your working state" in mail.get_content() for rcpt, mail in mailbox.mail):
            failures.append("the holder is not emailed that it is served on another Fleet without its working state")
        hubs["F3"].shutdown()
        hubs["F3"].server_close()
        NOW[0] = kst(2027, 1, 12, 12, 10)
        refused_hub = ask(port, keys[H])
        if refused_hub != (200, "F3", "F5") or not m.fleet_down("F3"):
            failures.append(f"a held Fleet whose hub refuses the connection is not marked down and its request served on a free Fleet: {refused_hub}")
        lots = [l for l in app.book.market.lots[H] if l["source"] == "downtime"]
        if [l["hours"] for l in lots] != [[slot(kst(2027, 1, 12, 10))], [slot(kst(2027, 1, 12, 11))]]:
            failures.append(f"serving on another Fleet takes away the downtime credit of the held Fleet's hours: {lots}")
        replay = sp.Book(app.cfg.path("ledger.jsonl"))
        if {k: (f["to"], f["end"]) for k, f in replay.market.failovers.items()} != {k: (f["to"], f["end"]) for k, f in app.book.market.failovers.items()}:
            failures.append("a replayed ledger gives different failovers")
    finally:
        server.shutdown()
        for hub in hubs.values():
            hub.shutdown()
        mailbox.shutdown()


def sign_in(port, mailbox, email, bearer=False):
    found = tp.Client(port, origin="https://sparkpipe.ai")
    found("POST", "/code", {"email": email})
    status, summary = found("POST", "/verify", dict({"email": email, "code": tp.last_code(mailbox, email)}, **({"bearer": True} if bearer else {})))
    if bearer and status == 200:
        found.cookie, found.bearer = None, {"Authorization": f"Bearer {summary['session']}"}
    return found, status, summary


def team_flows(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    upstream = tp.serve(FakeUpstream, seen=[], calls=[])
    server, app, port = live(work, "teams", mailbox, upstream)
    m = app.market
    try:
        NOW[0] = kst(2027, 1, 8, 12)
        O, X = person(app, "owner", 2000), person(app, "outsider", 10)
        owner = client(app, port, O)
        m.book_weeks(O, {"fleet": "F3", "weeks": 2})
        owner_key = m.create_key(O, {"note": "owner"})["key"]
        invited = [owner("POST", "/v1/members", {"email": email, "role": role}) for email, role in (("ann@example.com", "admin"), ("bob@example.com", "member"))]
        refused_invites = [owner("POST", "/v1/members", body)[0] for body in ({"email": "owner@example.com", "role": "admin"}, {"email": "outsider@example.com", "role": "member"}, {"email": "cat@example.com", "role": "owner"})]
        refused_invites.append(raw_call(port, "POST", "/v1/members", json.dumps({"email": "cat@example.com", "role": "member"}).encode(), {"X-Api-Key": owner_key, "Content-Type": "application/json"})[0])
        app.mails.join()
        if [(status, found.get("state"), found.get("role")) for status, found in invited] != [(200, "invited", "admin"), (200, "invited", "member")] or refused_invites != [400, 409, 400, 403]:
            failures.append(f"the owner cannot invite members by email with a role, or invites the owner, another account's email, a second owner, or with an API key: {invited} {refused_invites}")
        if not any(rcpt == ["ann@example.com"] and mail["Subject"] == "Owner invited you to SparkPipe" and "as an admin" in mail.get_content() for rcpt, mail in mailbox.mail):
            failures.append("an invited member is not emailed the invitation")
        ann, status_a, summary_a = sign_in(port, mailbox, "ann@example.com")
        bob, status_b, summary_b = sign_in(port, mailbox, "bob@example.com", bearer=True)
        A, B = summary_a.get("member"), summary_b.get("member")
        listed = owner("GET", "/v1/members")[1]["data"]
        if (status_a, summary_a.get("role"), summary_a.get("company"), status_b, summary_b.get("role")) != (200, "admin", "Owner", 200, "member") or not (A or "").startswith("MB") or [(x["email"], x["role"], x["state"]) for x in listed] != [("owner@example.com", "owner", "member"), ("ann@example.com", "admin", "member"), ("bob@example.com", "member", "member")]:
            failures.append(f"an invited member does not join by signing in with its own email code: {summary_a} {summary_b} {listed}")
        NOW[0] = kst(2027, 1, 12, 10)
        denied = [bob("POST", path, body, headers=bob.bearer)[1].get("reason") for path, body in (("/market/book-weeks", {"fleet": "F3", "weeks": 1}), ("/market/buy", {"hc": 10}), ("/market/waitlist", {"lock": 144}), ("/order", {"product": "credits", "hc": 144, "processor": "toss"}))]
        denied.append(bob("PUT", "/market/auto-extend", {"fleet": "F3", "weeks_ahead": 0}, headers=bob.bearer)[1].get("reason"))
        if denied != ["role"] * 5:
            failures.append(f"a member books, pays or joins the waitlist: {denied}")
        status, made = bob("POST", "/v1/keys", {"note": "bob laptop"}, headers=bob.bearer)
        wide = [bob("POST", "/v1/keys", body, headers=bob.bearer)[1].get("error", {}).get("code") for body in ({"scopes": ["storage"]}, {"allow_swap": True})]
        if status != 200 or made.get("member") != B or made.get("scopes") != ["inference", "functions", "account:read"] or wide != ["role", "role"]:
            failures.append(f"a member's keys are not its own, limited to inference, functions and account:read: {made} {wide}")
        before = len(upstream.calls)
        raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": made["key"]})
        got = upstream.calls[-1] if len(upstream.calls) > before else {"items": []}
        if [sent(got, name) for name in ("x-sparkpipe-account", "x-sparkpipe-member", "x-sparkpipe-role", "x-sparkpipe-scopes", "x-sparkpipe-allow-swap")] != [[O], [B], ["member"], ["inference,functions,account:read"], ["0"]]:
            failures.append(f"the hub is not told which member sends a request, its role and scopes, for fair share per member: {got['items']}")
        blocked = [raw_call(port, "GET", "/v1/storage", headers={"X-Api-Key": made["key"]})[0], raw_call(port, "GET", "/v1/state", headers=bob.bearer)[0]]
        if blocked != [403, 403]:
            failures.append(f"a member reaches storage or state: {blocked}")
        own = [k["id"] for k in bob("GET", "/v1/keys", headers=bob.bearer)[1]["data"]]
        owner_id = m.key_account(owner_key)["key"]
        if own != [made["id"]] or bob("DELETE", f"/v1/keys/{owner_id}", headers=bob.bearer)[0] != 404:
            failures.append(f"a member sees or revokes keys other than its own: {own}")
        admin_key = ann("POST", "/v1/keys", {"scopes": ["inference", "state", "storage", "models"], "allow_swap": True})[1]
        hook = ann("PUT", "/v1/webhooks", {"email": "owner@example.com", "events": ["booking.deadline", "fleet.failover"]})
        if admin_key.get("member") != A or ann("DELETE", f"/v1/keys/{made['id']}")[0] != 200 or hook[0] != 200 or ann("POST", "/v1/members", {"email": "cat@example.com", "role": "member"})[1].get("error", {}).get("code") != "role":
            failures.append(f"an admin cannot manage every key and the webhooks, or can manage members: {admin_key} {hook}")
        reads = [bob("GET", "/v1/audit", headers=bob.bearer)[0], bob("PUT", "/v1/webhooks", {"email": "owner@example.com", "events": []}, headers=bob.bearer)[0], bob("GET", "/market/ledger", headers=bob.bearer)[0], ann("GET", "/market/ledger")[0]]
        if reads != [403, 403, 403, 200] or bob("GET", "/account", headers=bob.bearer)[1].get("orders") != [] or owner("GET", "/account")[1].get("role") != "owner":
            failures.append(f"a member reads the audit log or ledger, changes webhooks or sees payments: {reads}")
        second = bob("POST", "/v1/keys", {"note": "bob ci"}, headers=bob.bearer)[1]
        moment = market.stamp(m.now())
        records = [{"request": "t1", "account": O, "fleet": "F3", "at": moment, "key": second["id"]}, {"request": "t2", "account": O, "fleet": "F3", "at": moment, "key": owner_id}, {"request": "t3", "account": O, "fleet": "F3", "at": moment, "key": "session", "member": A}]
        hub(port, "/hub/requests", {"records": records})
        by_member = {who: [r["request"] for r in owner("GET", f"/v1/requests?member={who}")[1]["data"]] for who in (O, A, B)}
        if by_member != {O: ["t2"], A: ["t3"], B: ["t1"]} or [r["request"] for r in bob("GET", "/v1/requests", headers=bob.bearer)[1]["data"]] != ["t1"]:
            failures.append(f"usage records are not kept per member, or a member sees others' usage: {by_member}")
        status, took = hub(port, "/hub/audit", {"events": [{"id": "w1", "account": O, "member": B, "fleet": "F3", "type": "cache.wiped", "at": moment, "data": {"reason": "user"}}, {"id": "w2", "account": O, "type": "prompt.seen", "at": moment}]})
        again = hub(port, "/hub/audit", {"events": [{"id": "w1", "account": O, "type": "cache.wiped", "at": moment}]})[1]
        if status != 200 or (took["accepted"], [r["index"] for r in took["refused"]], again["duplicates"]) != (1, [1], 1) or hub(port, "/hub/audit", {"events": []}, "wrong")[0] != 401:
            failures.append(f"the hub's audit events are not taken in once each, by type, with the upstream key: {took} {again}")
        log = owner("GET", "/v1/audit?limit=100")[1]["data"]
        kinds = [(e["type"], e["member"]) for e in log]
        expected = [("member.invited", O), ("member.invited", O), ("member.joined", A), ("session.sign_in", A), ("member.joined", B), ("session.sign_in", B), ("key.created", B), ("key.created", A), ("webhook.set", A), ("key.revoked", A), ("key.created", B), ("cache.wiped", B)]
        if sorted(k for k in kinds if k[0] != "booking.weeks") != sorted(expected) or [e["at"] for e in log] != sorted(e["at"] for e in log) or any("prompt" in json.dumps(e) for e in log):
            failures.append(f"the audit log does not list sign-ins, member, key and webhook changes and the hub's events, each with the member who did it: {kinds}")
        paged = owner("GET", "/v1/audit?limit=2")[1]
        if len(paged["data"]) != 2 or not paged["has_more"] or owner("GET", f"/v1/audit?type=cache.wiped")[1]["data"][0]["id"] != "HBw1":
            failures.append(f"the audit log does not page or filter: {paged}")
        owner("POST", "/v1/members", {"email": "ann@example.com", "role": "member"})
        demoted = raw_call(port, "GET", "/v1/state", headers={"X-Api-Key": admin_key["key"]})[0]
        removed = owner("DELETE", f"/v1/members/{B}")
        after = [raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": second["key"]})[0], bob("GET", "/market/state", headers=bob.bearer)[1]["account"]]
        rejoin = sign_in(port, mailbox, "bob@example.com")[1]
        if demoted != 403 or removed != (200, {"id": B, "removed": True}) or after != [401, None] or rejoin != 404:
            failures.append(f"a member moved to the member role keeps admin scopes, or a removed member keeps its keys, sessions or sign-in: {demoted} {removed} {after} {rejoin}")
        late = owner("POST", "/v1/members", {"email": "cat@example.com", "role": "member"})[1]
        gone = owner("POST", "/v1/members", {"email": "dan@example.com", "role": "member"})[1]
        cancelled = owner("DELETE", f"/v1/members/{gone['id']}")[0]
        NOW[0] += timedelta(days=8)
        if sign_in(port, mailbox, "cat@example.com")[1] != 404 or sign_in(port, mailbox, "dan@example.com")[1] != 404 or cancelled != 200 or any(x["state"] == "invited" for x in owner("GET", "/v1/members")[1]["data"]):
            failures.append(f"an invitation works after 7 days or after it was cancelled: {late} {gone}")
        flips = [owner("POST", "/v1/members", {"email": "eve@example.com", "role": role})[0] for role in ("admin", "member", "admin", "member")]
        app.mails.join()
        if flips != [200, 200, 200, 429] or sum(rcpt == ["eve@example.com"] for rcpt, _ in mailbox.mail) != 3:
            failures.append(f"one email receives more than 3 invitations a day: {flips}")
        replay = sp.Book(app.cfg.path("ledger.jsonl"))
        if {k: (v["role"], v["removed"]) for k, v in replay.members.items()} != {k: (v["role"], v["removed"]) for k, v in app.book.members.items()} or {k: v["state"] for k, v in replay.invites.items()} != {k: v["state"] for k, v in app.book.invites.items()}:
            failures.append("a replayed ledger gives different members or invitations")
        exported = {line["type"] for line in owner("GET", "/market/ledger")[1]["lines"]}
        if not {"member_invite", "member", "member_role", "member_remove"} <= exported or X in json.dumps(owner("GET", "/v1/audit?limit=1000")[1]):
            failures.append(f"the owner's ledger export lacks its members, or its audit log names another account: {exported}")
        erase_flows(failures, app, m, port, mailbox, O, X, owner, ann, owner_key, records)
    finally:
        server.shutdown()
        upstream.shutdown()
        mailbox.shutdown()


def erase_flows(failures, app, m, port, mailbox, O, X, owner, other, owner_key, records):
    window = "/v1/requests?since=2027-01-01T00:00:00Z"
    listed = owner("GET", window)[1]["data"]
    refused_erase = [other("DELETE", "/v1/requests")[0], raw_call(port, "DELETE", "/v1/requests", headers={"X-Api-Key": owner_key})[0]]
    status, erased = owner("DELETE", "/v1/requests")
    gone = owner("GET", window)[1]["data"]
    resent = hub(port, "/hub/requests", {"records": records})[1]
    NOW[0] += timedelta(minutes=1)
    fresh = hub(port, "/hub/requests", {"records": [{"request": "t4", "account": O, "fleet": "F3", "at": market.stamp(m.now())}]})[1]
    if len(listed) != 3 or refused_erase != [403, 403] or status != 200 or erased.get("erased") != 3 or gone or (resent["accepted"], len(resent["refused"])) != (0, 3) or fresh["accepted"] != 1:
        failures.append(f"the owner cannot delete the records of its requests at once, or deleted records come back when the hub sends them again (DPA 9.3): {listed} {refused_erase} {erased} {resent} {fresh}")
    if sp.erase_records(app, ["sparkpipe_pay.py", "erase-records", "outsider@example.com", "written request of 20 January"]) != 0 or sp.erase_records(app, ["sparkpipe_pay.py", "erase-records", "nobody@example.com", "x"]) != 2:
        failures.append("the operator cannot delete an account's request records on its written request")
    lines = [line["data"] for line in ledger_lines(app) if line["type"] == "records_erased"]
    app.mails.join()
    mailed = [rcpt for rcpt, mail in mailbox.mail if mail["Subject"] == "SparkPipe deleted the records of your requests"]
    asked = [mail.get_content() for rcpt, mail in mailbox.mail if rcpt == [app.cfg.notify] and mail["Subject"] == f"Erase the hub's request records of {O} within 30 days"]
    logged = [e["type"] for e in owner("GET", "/v1/audit?type=records.erased")[1]["data"]]
    if [(l["account"], l["records"], l["by"]) for l in lines] != [(O, 3, O), (X, 0, "operator")] or mailed != [["owner@example.com"], ["outsider@example.com"]] or logged != ["records.erased"] or not asked or f"--erase-records {O}" not in asked[0] or sp.Book(app.cfg.path("ledger.jsonl")).erased != app.book.erased:
        failures.append(f"an erasure is not recorded in the ledger and the audit log and emailed to the owner, or does not replay: {lines} {mailed} {logged}")


def view(state, active=None, target=None):
    return 200, {"state": state, "active": active, "target": target, "models": [{"id": name, "active": name == active} for name in ("glm-5.3", "kimi-k3")]}


def resident_flows(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    hub_ = tp.serve(FakeUpstream, seen=[], calls=[], probes=[])
    server, app, port = live(work, "resident", mailbox, hub_)
    saved, sp.UPTIME_GAP_SECONDS = sp.UPTIME_GAP_SECONDS, 10 ** 9
    try:
        cases = [({"glm-5.3": 200, "kimi-k3": 503}, view("idle", "glm-5.3"), True), ({"glm-5.3": 503, "kimi-k3": 200}, view("idle", "glm-5.3"), False),
                 ({"glm-5.3": 503, "kimi-k3": 503}, view("swapping", None, "kimi-k3"), True), ({"glm-5.3": 503, "kimi-k3": 503}, view("idle"), True),
                 ({"glm-5.3": 503, "kimi-k3": 503}, view("failed"), False), ({"glm-5.3": 200, "kimi-k3": 503}, view("failed", "glm-5.3"), True),
                 ({"glm-5.3": 200, "kimi-k3": 503}, None, False), ({"glm-5.3": 200, "kimi-k3": 200}, None, True),
                 ({"glm-5.3": 200, "kimi-k3": 503}, (503, {"error": {"type": "control_unavailable"}}), True), ({"glm-5.3": 503, "kimi-k3": 503}, (503, {}), False)]
        wrong = []
        for models, control, expected in cases:
            hub_.models, hub_.control = models, control
            found = sp.probe_health(app.cfg.upstream("F3"))
            if found[0] is not expected:
                wrong.append((models, control, found))
        if wrong:
            failures.append(f"a Fleet is not judged up exactly when every resident model answers its /health entry, during a swap, or with nothing resident and no failed swap: {wrong}")
        hub_.models, hub_.control, hub_.probes[:] = {"glm-5.3": 200, "kimi-k3": 503}, view("idle", "glm-5.3"), []
        for minutes in (0, 20, 40):
            probe(app, kst(2027, 1, 12, 10) + timedelta(minutes=minutes))
        hub_.control = view("swapping", "glm-5.3", "kimi-k3")
        hub_.models = {"glm-5.3": 503, "kimi-k3": 503}
        probe(app, kst(2027, 1, 12, 11))
        steady = public_status(port)[1]["F3"]
        hub_.control = view("idle", "kimi-k3")
        probe(app, kst(2027, 1, 12, 11, 10))
        probe(app, kst(2027, 1, 12, 11, 30))
        broken = public_status(port)[1]["F3"]
        if (steady["status"], steady["incidents"], broken["status"], broken["since"]) != ("up", [], "down", market.stamp(kst(2027, 1, 12, 11))) or not broken["incidents"][0]["creditable"]:
            failures.append(f"a Fleet with one model resident is shown down because another configured model does not answer, a swap is counted as an outage, or a failing resident model is not: {steady} {broken}")
        if set(hub_.probes) != {"/health", "/v1/sparkpipe/status"} or hub_.calls:
            failures.append(f"a probe asks the hub for more than /health and /v1/sparkpipe/status: {set(hub_.probes)} {hub_.calls[:2]}")
    finally:
        sp.UPTIME_GAP_SECONDS = saved
        server.shutdown()
        hub_.shutdown()
        mailbox.shutdown()


def sandbox_flows(failures, work):
    site(work / "plain-root", serving=False)
    cfg = sp.Config({"SPARKPIPE_GATEWAY_CAPACITY": "100000", "SPARKPIPE_PAY_DATA": str(work / "sandbox"), "SPARKPIPE_PAY_SITE": str(work / "plain-root" / "site"), "SPARKPIPE_PAY_SANDBOX": "1", "SPARKPIPE_PAY_PREFIX": "/pay/sandbox/api"})
    NOW[0] = datetime(2026, 11, 2, 3, 0, 0, tzinfo=timezone.utc)
    server = sp.start(cfg, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    user = tp.Client(port, origin="https://sparkpipe.ai")
    try:
        code = user("POST", "/code", {"email": "try@example.com"})[1]["code"]
        status, _ = user("POST", "/verify", {"email": "try@example.com", "code": code, "company": "Try Co", "accept": True})
        if status != 200 or "Path=/pay/sandbox/api" not in user.last_cookie:
            failures.append("sandbox sign-in")
        if user("POST", "/market/buy", {"hc": 0})[0] != 400 or user("POST", "/market/buy", {"hc": 300})[1].get("hc") != "300.000000":
            failures.append("the sandbox does not sell HC instantly by quantity")
        state = user("GET", "/market/state")[1]
        if not state["sandbox"] or not all(f["serving"] for f in state["fleets"]) or state["account"]["balance"] != "300.000000":
            failures.append(f"the sandbox state: {state['fleets']} {state['account']}")
        start = slot(datetime(2026, 11, 2, 4, tzinfo=timezone.utc))
        if user("POST", "/market/book-hours", {"fleet": "F1", "start": start, "hours": 2})[0] != 200 or user("POST", "/market/book-weeks", {"fleet": "F3", "weeks": 1})[0] != 200:
            failures.append("the sandbox does not book Hours and Weeks before the published opening dates")
        key = user("POST", "/market/keys", {"note": "try"})[1]["key"]
        if chat(port, key, {})[0] != 403:
            failures.append("a sandbox reply is given before the booked Hour starts")
        if user("POST", "/market/sandbox/clock", {"hours": 1})[1].get("now") != "2026-11-02T04:00:00Z":
            failures.append("the sandbox clock does not move")
        status, answer = chat(port, key, {"max_tokens": 6})
        if status != 200 or not answer["choices"][0]["message"]["content"].startswith("Sandbox reply from SparkPipe F1") or answer["sparkpipe"]["fleet"] != "F1":
            failures.append(f"the sandbox does not simulate replies for a holder: {status} {answer}")
        status, _, raw = raw_call(port, "POST", "/v1/messages", b"{}", {"X-Api-Key": key})
        if status != 503 or json.loads(raw)["error"]["type"] != "closed":
            failures.append(f"the sandbox claims to serve a path it does not simulate: {status} {raw!r}")
        pinned = user("GET", "/market/state")[1]["account"]["pinned"]
        user("POST", "/market/sandbox/clock", {"hours": 168})
        kept = user("GET", "/market/state")[1]["account"]["pinned"]
        off = user("PUT", "/market/auto-extend", {"fleet": "F3", "weeks_ahead": 0})
        if [p["autoExtend"] for p in pinned] != [1] or [market.slot_time(p["through"]) - market.slot_time(q["through"]) for p, q in zip(kept, pinned)] != [market.WEEK] or off[1].get("weeks_ahead") != 0 or user("GET", "/market/state")[1]["account"]["pinned"][0]["autoExtend"] != 0:
            failures.append(f"in the sandbox, a first Week does not turn auto-extend on, keep the Fleet a Week later, and turn off with one call: {pinned} {kept} {off}")
        before = Decimal(user("GET", "/market/state")[1]["account"]["balance"])
        down = user("POST", "/market/sandbox/fleet", {"fleet": "F3", "up": False})
        user("POST", "/market/sandbox/clock", {"hours": 1})
        middle = public_status(port)[1].get("F3", {})
        status, served = chat(port, key, {})
        user("POST", "/market/sandbox/fleet", {"fleet": "F3", "up": True})
        if status != 200 or served.get("sparkpipe") != {"fleet": "F1", "failover_from": "F3"} or not served["choices"][0]["message"]["content"].startswith("Sandbox reply from SparkPipe F1") or chat(port, key, {})[1].get("sparkpipe") != {"fleet": "F3"}:
            failures.append(f"the sandbox does not serve a holder on a free Fleet while its own is down, and on its own once it is back: {status} {served}")
        after = user("GET", "/market/state")[1]["account"]
        if down[0] != 200 or middle.get("status") != "down" or Decimal(after["balance"]) - before != 1 or [l["hc"] for l in after["lots"] if l["source"] == "downtime"] != ["1.000000"] or public_status(port)[1]["F3"]["status"] != "up":
            failures.append(f"the sandbox does not take a held Fleet down and credit the booked hour under clause 4.4: {down} {middle} {after['lots']}")
        invited = user("POST", "/v1/members", {"email": "mate@example.com", "role": "member"})
        mate = tp.Client(port, origin="https://sparkpipe.ai")
        joined = mate("POST", "/verify", {"email": "mate@example.com", "code": mate("POST", "/code", {"email": "mate@example.com"})[1]["code"]})
        mate_key = mate("POST", "/market/keys", {"note": "mate"})[1].get("key", "")
        booked = mate("POST", "/market/book-weeks", {"fleet": "F3", "weeks": 1})[1].get("reason")
        if invited[0] != 200 or (joined[1].get("role"), joined[1].get("company")) != ("member", "Try Co") or booked != "role" or chat(port, mate_key, {})[1].get("sparkpipe") != {"fleet": "F3"}:
            failures.append(f"in the sandbox, an invited member does not join with its email code, make its own key and send requests, without booking: {invited} {joined} {booked}")
        wiped = user("DELETE", "/v1/requests")
        if wiped != (200, {"erased": 0, "through": market.stamp(server.app.market.now())}) or mate("DELETE", "/v1/requests")[0] != 403:
            failures.append("in the sandbox, the owner cannot delete the records of its requests, or a member can")
        held = user("GET", "/market/state")[1]["account"]
    finally:
        server.shutdown()
    restarted = sp.start(cfg, port=0)
    threading.Thread(target=restarted.serve_forever, daemon=True).start()
    user.port = restarted.server_address[1]
    try:
        again = user("GET", "/market/state")[1]["account"]
        if (again["balance"], again["bookings"], again["keys"]) != (held["balance"], held["bookings"], held["keys"]):
            failures.append("a restarted sandbox does not replay its balance, bookings and keys")
        if user("POST", "/market/sandbox/reset", {})[1] != {"reset": True} or user("GET", "/market/state")[1]["account"] is not None:
            failures.append("the sandbox does not reset")
    finally:
        restarted.shutdown()


def main():
    failures = []
    work = Path(tempfile.mkdtemp(prefix="sparkpipe-market-"))
    try:
        check_units(failures)
        site(work)
        live_flows(failures, work)
        discount_flows(failures, work)
        pair_flows(failures, work)
        pair_deadline_flows(failures, work)
        allocation_flows(failures, work)
        extension_flows(failures, work)
        notice_flows(failures, work)
        uptime_flows(failures, work)
        resident_flows(failures, work)
        failover_flows(failures, work)
        team_flows(failures, work)
        sandbox_flows(failures, work)
    finally:
        shutil.rmtree(work)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("market: HC lots lapse 12 months after issue while HC spent on bookings does not; Hour Fleet hours open on hoursFrom, up to 7 days ahead and 24 a Week per account across every Hour Fleet, never allocated, and pinned only as the pair of the Fleet its holder holds, within those Weeks, at that Fleet's discount, while a quarter of the serving Fleets stays open for hours; an Hour costs less by the discount for the most Weeks held on one Fleet or the Weeks a waitlist lock books, and leaving the waitlist before a Fleet repays the waitlist part; free-Fleet Hours only to the end of the current Week and never on a pinned Fleet; Weeks from the next Monday 00:00 KST, priced with discounts at 4, 13, 26 and 52 Weeks counting the current Week, extended only contiguously and at most 52 Weeks ahead; nobody else books a pinned Fleet before its holder's Sunday 00:00 deadline; the Sunday round allocates free Fleets to the waitlist by largest lock then earliest join, booking as many Weeks as each lock pays for and unlocking the rest, and leaves a Fleet bookable by its holder or anyone when nobody is allocated; customers see only their own position; locked HC that lapses is unlocked and recorded as a refund due; the gateway passes every path of the Fleet API through to the hub only for a holder, with its own X-SparkPipe-Account, X-SparkPipe-Fleet, key, scopes and swap headers and no customer copies, by Bearer, X-Api-Key or Basic for git, with 32 MiB and chunked bodies, streams, the WebSocket of /v1/responses (closed when the hold ends) and the hub's answers unchanged, and refuses paths outside the API; keys with scopes, a fleet, allow_swap, an expiry, an IP allowlist, a default priority and rotation that keeps the old key for an hour; usage records from the hub kept a year, listed by GET /v1/requests without prompts and summed into stats; CLI bearer sessions that book, join the waitlist, sign out and end after 90 days; auto-extend on from the first booked or granted Week with one Week ahead, off with one call, keeping a Fleet for a holder who does nothing, across a year with an early warning on a short balance and a stop notice delivered like the others; booking.deadline 72, 48, 6 and 2 hours before the Sunday deadline; every event of §13 delivered signed, with retries and Slack text, and the default email notices; the pair-loss credit of 24 hours once per event with working_state.lost; each Fleet's hub probed through its own upstream, up when every resident model answers or a swap is under way, down from its last good probe until its next one, with the service's own gaps across a restart counted as down; every booked hour a period over 15 minutes touches credited once at the Fleet's Rate, as soon as it passes 15 minutes, daily and at its end, never with maintenance hours, with downtime.credited by email and our own email; the public status page with uptime and periods, without accounts; a held Fleet that is down, by its probes or a refused connection, served on the first free Fleet as that Fleet with X-SparkPipe-Failover-From, listed in its holder schedule, given up to a booking, and back on the held Fleet once it serves, with fleet.failover and the downtime credit kept; the holder feed lists Week and Hour holders without maintenance hours; maintenance credits booked hours; availability.txt; the public count of waitlist locks; an operator email for each lock created or increased that never fails the lock; the gateway refuses sparkpipe fields other than fleet; teams: members invited by email with the admin or member role who join with their own email code, keys that belong to a member, members who cannot book, pay or join the waitlist and reach only inference and Functions, admins who manage keys and webhooks but not members, usage records and an audit log per member with the hub's own events, role changes, removal that revokes keys and sessions, and invitations that lapse after 7 days or when cancelled; the records of an account's requests deleted on its owner's request or the operator's command, never taken in again for the time before (DPA 9.3); the sandbox clock, instant HC, simulated replies, a Fleet taken down and its credit, a team member, and reset; and the ledger replays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
