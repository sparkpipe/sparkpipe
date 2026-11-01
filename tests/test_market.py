#!/usr/bin/env python3

import http.server
import json
import shutil
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
ACCELERATORS = '[{"id":"A1","name":"Accelerator 1","kind":"96 Sparks","state":"serving","rate":6,"tokensPerSecond":400},{"id":"A2","name":"Accelerator 2","kind":"48 Sparks","state":"serving","rate":3,"tokensPerSecond":200}]'


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
    pricing = json.loads(path.read_text().split("=", 1)[1].strip().rstrip(";"))
    if serving:
        for fleet in pricing["fleets"]:
            fleet["state"] = "serving"
        pricing["fleets"].append(dict(pricing["fleets"][2], id="F4", name="Fleet 4"))
        pricing["accelerators"] = [{"id": "A1", "name": "Accelerator 1", "kind": "96 Sparks", "state": "serving", "rate": 6, "tokensPerSecond": 5000}]
    path.write_text("window.SP_PRICING = " + json.dumps(pricing, indent=2) + ";\n")
    (work / "pool_key").write_text(POOL_KEY + "\n")


def live(work, name, mailbox, upstream=None):
    env = {"SPARKPIPE_PAY_DATA": str(work / name), "SPARKPIPE_PAY_SITE": str(work / "site"), "COINPAYMENTS_CLIENT_ID": tp.CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(work / "secret"),
           "SPARKPIPE_PAY_SMTP": f"127.0.0.1:{mailbox.server_address[1]}", "SPARKPIPE_POOL_KEY_FILE": str(work / "pool_key")}
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

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append(({name: self.headers.get(name) for name in ("Authorization", "X-SparkPipe-Fleet", "X-SparkPipe-Account", "X-SparkPipe-Accelerator")}, self.headers.get_all("X-SparkPipe-Account"), body))
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in ({"choices": [{"index": 0, "delta": {"content": "hi"}}]}, {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 1}}):
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        data = json.dumps({"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 5, "sparkpipe_accelerator_seconds": 0.2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def check_units(failures):
    if market.text_hc("1.23456789") != "1.234567" or market.months_later(datetime(2028, 2, 29, 5, tzinfo=timezone.utc), 12) != datetime(2029, 2, 28, 5, tzinfo=timezone.utc):
        failures.append("HC rounds down to 6 places, and a lot issued on 29 February lapses on 28 February a year later")
    if market.week_start(kst(2027, 1, 10, 23, 59)) != kst(2027, 1, 4) or market.week_start(kst(2027, 1, 11)) != kst(2027, 1, 11) or market.sunday_after(kst(2027, 1, 6, 12)) != kst(2027, 1, 10) or market.sunday_after(kst(2027, 1, 10)) != kst(2027, 1, 17):
        failures.append("Weeks run from Monday 00:00 to Sunday 24:00 KST, and the round is at Sunday 00:00 KST")
    conf = (ROOT / "deployment" / "vps" / "sparkpipe-pay.conf").read_text()
    if '<Location "/availability.txt">' not in conf or 'ProxyPass "http://127.0.0.1:18090/availability.txt"' not in conf:
        failures.append("Apache does not map https://sparkpipe.ai/availability.txt to the live service")


def live_flows(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    upstream = tp.serve(FakeUpstream, seen=[])
    server, app, port = live(work, "live", mailbox, upstream)
    m = app.market
    try:
        opening(failures, app, m)
        state = weeks_and_hours(failures, app, m, port)
        waitlist(failures, app, m, port, state)
        gateway(failures, app, m, port, upstream, state)
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
        failures.append("Demo Fleet hours open on demoFrom and not before")
    NOW[0] = kst(2026, 12, 25, 12)
    if refused(lambda: m.quote_weeks(early, {"fleet": "F2", "weeks": 1})) != (409, "not_open"):
        failures.append("a Week before weeksFrom is booked")
    for moment, first in ((kst(2027, 1, 1, 12), kst(2027, 1, 4)), (kst(2027, 1, 3, 23), kst(2027, 1, 4)), (kst(2027, 1, 4, 0, 30), kst(2027, 1, 11))):
        NOW[0] = moment
        quote = m.quote_weeks(early, {"fleet": "F3", "weeks": 2})
        if (quote["first"], quote["last"], quote["end"]) != (slot(first), slot(first + market.WEEK), slot(first + 2 * market.WEEK)):
            failures.append(f"a first Week asked at {market.kst_text(moment)} does not start on the next Monday 00:00 KST and run to Sunday 24:00: {quote}")


def weeks_and_hours(failures, app, m, port):
    NOW[0] = kst(2027, 1, 1, 12)
    state = {name: person(app, name.lower(), credits) for name, credits in (("H", 10000), ("C", 5000), ("D", 100), ("G", 1000), ("K", 1000))}
    H, C, D, G, K = (state[name] for name in "HCDGK")
    first = m.book_weeks(H, {"fleet": "F2", "weeks": 1})
    if (first["first"], first["end"], first["price"], first["discount"], first["extends"]) != (slot(kst(2027, 1, 4)), slot(kst(2027, 1, 11)), "144.000000", "0", False) or balance(app, H) != 10000 - 144:
        failures.append(f"a first Week is booked from the next Monday at weekHC times Rate: {first}")
    NOW[0] = kst(2027, 1, 6, 12)
    public = m.public()
    fleets = {f["id"]: f for f in public["fleets"]}
    if [fleets[f]["status"] for f in ("F1", "F2", "F3", "F4")] != ["demo", "pinned", "free", "free"] or fleets["F3"]["firstWeek"] != slot(kst(2027, 1, 11)) or fleets["F2"]["firstWeek"] is not None or fleets["F2"]["booked"] != [[slot(kst(2027, 1, 4)), slot(kst(2027, 1, 11))]]:
        failures.append(f"the public Fleet states: {fleets}")
    if "waitlist" in json.dumps(public) or fleets["F2"]["storageTB"] != 4 or public["hc"]["prices"][1] != {"from": 144, "usd": 15, "krw": 19500} or public["bookings"]["weekHC"] != 144:
        failures.append("the public state shows the waitlist, or lacks the published settings")
    for who in (C, H):
        if refused(lambda: m.book_hours(who, {"fleet": "F2", "start": slot(kst(2027, 1, 6, 14)), "hours": 1})) != (409, "pinned"):
            failures.append("an Hour is booked on a pinned Fleet")
    if refused(lambda: m.quote_weeks(C, {"fleet": "F2", "weeks": 1})) != (409, "pinned"):
        failures.append("another customer books the next Week of a pinned Fleet before its holder's deadline")
    for weeks, off, price in ((3, "0", "432.000000"), (4, "0.05", "547.200000"), (13, "0.1", "1684.800000"), (26, "0.15", "3182.400000"), (52, "0.2", "5990.400000")):
        quote = m.quote_weeks(C, {"fleet": "F3", "weeks": weeks})
        if (quote["discount"], quote["price"]) != (off, price):
            failures.append(f"{weeks} Weeks are priced at {quote['price']} HC, {quote['discount']} off, not {price} at {off}")
    if refused(lambda: m.quote_weeks(C, {"fleet": "F3", "weeks": 53})) != (400, None) or m.quote_weeks(C, {"fleet": "F3", "weeks": 52})["last"] != slot(kst(2027, 1, 4) + 52 * market.WEEK):
        failures.append("the last Week may start more than 52 Weeks after the current Week")
    quote = m.quote_weeks(H, {"fleet": "F2", "weeks": 3})
    if (quote["first"], quote["count"], quote["discount"], quote["price"], quote["extends"]) != (slot(kst(2027, 1, 11)), 4, "0.05", "410.400000", True):
        failures.append(f"an extension does not follow the booked Week, or its discount does not count the current Week: {quote}")
    m.book_weeks(H, {"fleet": "F2", "weeks": 3})
    if refused(lambda: m.quote_weeks(H, {"fleet": "F2", "weeks": 50})) != (409, "too_far") or m.quote_weeks(H, {"fleet": "F2", "weeks": 49})["first"] != slot(kst(2027, 2, 1)):
        failures.append("an extension is not contiguous, or reaches past 52 Weeks after the current Week")
    if m.view(H)["pinned"] != [{"fleet": "F2", "weeks": 4, "through": slot(kst(2027, 2, 1)), "deadline": slot(kst(2027, 1, 31)), "canExtend": True}]:
        failures.append(f"the account's pinned Fleets: {m.view(H)['pinned']}")
    if refused(lambda: m.book_weeks(C, {"fleet": "F2", "weeks": 1}))[0] != 409 or refused(lambda: m.book_weeks(C, {"fleet": "F1", "weeks": 1})) != (409, "not_weekly"):
        failures.append("another customer books a Week of a pinned Fleet, or the Demo Fleet is booked by the Week")
    hours = m.book_hours(C, {"fleet": "F3", "start": slot(kst(2027, 1, 6, 13)), "hours": 2})
    if hours["hc"] != "2.000000" or hours["state"] != "upcoming" or refused(lambda: m.book_hours(C, {"fleet": "F3", "start": slot(kst(2027, 1, 10, 23)), "hours": 1})):
        failures.append(f"free-Fleet Hours to the end of the current Week: {hours}")
    if refused(lambda: m.book_hours(C, {"fleet": "F4", "start": slot(kst(2027, 1, 10, 23)), "hours": 2})) != (409, "too_far") or refused(lambda: m.book_hours(D, {"fleet": "F3", "start": slot(kst(2027, 1, 6, 14)), "hours": 1})) != (409, "taken"):
        failures.append("a free-Fleet Hour reaches past the current Week, or takes a booked hour")
    if refused(lambda: m.book_hours(C, {"fleet": "F4", "start": slot(kst(2027, 1, 6, 12)), "hours": 1})) != (409, "started"):
        failures.append("an hour that has started is booked")
    demo(failures, m, C, D)
    m.book_weeks(G, {"fleet": "F3", "weeks": 1})
    m.book_weeks(K, {"fleet": "F4", "weeks": 1})
    return state


def demo(failures, m, C, D):
    book = lambda who, start, hours: refused(lambda: m.book_hours(who, {"fleet": "F1", "start": slot(start), "hours": hours}))
    if book(C, kst(2027, 1, 6, 13), 24) or book(C, kst(2027, 1, 7, 13), 1) != (409, "demo_cap") or book(D, kst(2027, 1, 7, 13), 1):
        failures.append("the Demo Fleet's weekly cap is not per account")
    if book(C, kst(2027, 1, 10, 23), 2) != (409, "demo_cap") or book(C, kst(2027, 1, 11), 3):
        failures.append("Demo Fleet hours are not capped by the Week they fall in")
    if book(C, kst(2027, 1, 13, 12), 1) != (409, "too_far") or book(C, kst(2027, 1, 13, 11), 1):
        failures.append("Demo Fleet hours are not bookable exactly up to 7 days ahead")


def waitlist(failures, app, m, port, state):
    for name, credits in (("A", 1000), ("B", 1000), ("W", 1000), ("E", 50), ("X", 144)):
        state[name] = person(app, name.lower(), credits)
    A, B, W, E, X = (state[name] for name in "ABWEX")
    if refused(lambda: m.join(E, {"lock": 144})) != (402, "insufficient_hc") or refused(lambda: m.join(A, {"lock": 100})) != (409, "lock_too_small"):
        failures.append("a lock below weekHC, or above the unlocked balance, joins the waitlist")
    order = [m.join(A, {"lock": 200})["position"], m.join(B, {"lock": 300})["position"], m.position(A), m.join(W, {"lock": 200})["position"]]
    order += [m.join(A, {"lock": "50"})["position"], m.join(W, {"lock": 100})["position"], m.position(B), m.position(A)]
    if order != [1, 1, 2, 3, 2, 2, 1, 3]:
        failures.append(f"the waitlist is not ordered by largest lock, then earliest join, keeping the join time on an added lock: {order}")
    m.join(X, {"lock": 144})
    if refused(lambda: m.book_hours(X, {"fleet": "F4", "start": slot(kst(2027, 1, 6, 15)), "hours": 1})) != (402, "insufficient_hc") or m.leave(X) != {"locked": "0.000000", "position": None} or balance(app, X) != 144:
        failures.append("locked HC can be spent, or leaving does not unlock all of it")
    shown = client(app, port, A)("GET", "/market/state")[1]
    text = json.dumps(shown)
    if (shown["account"]["position"], shown["account"]["locked"], shown["account"]["balance"]) != (3, "250.000000", "750.000000") or any(other in text for other in (B, W, "b@example.com", "300.000000")):
        failures.append(f"a customer sees more of the waitlist than its own position: {shown['account']}")


def gateway(failures, app, m, port, upstream, state):
    H, C, D = state["H"], state["C"], state["D"]
    NOW[0] = kst(2027, 1, 6, 13, 30)
    keys = {who: m.create_key(who, "test")["key"] for who in (H, C, D)}
    if chat(port, keys[D], {})[0] != 403 or chat(port, keys[D], {})[1]["error"]["code"] != "no_fleet" or call(port, "GET", "/v1/models", headers={"Authorization": f"Bearer {keys[D]}"})[0] != 403:
        failures.append("the gateway serves an account that holds no Fleet")
    if chat(port, "sk-hc-wrong", {})[0] != 401 or chat(port, keys[C], {})[1].get("error", {}).get("code") != "fleet_required" or chat(port, keys[C], {"sparkpipe": {"fleet": "F2"}})[1].get("error", {}).get("code") != "fleet_not_held":
        failures.append("the gateway does not make an account holding two Fleets name one it holds")
    seen = len(upstream.seen)
    status, answer = chat(port, keys[C], {"sparkpipe": {"fleet": "F3"}}, {"X-SparkPipe-Account": "ACFAKE", "X-SparkPipe-Fleet": "F9"})
    headers, accounts, body = upstream.seen[-1] if len(upstream.seen) > seen else ({}, [], {})
    if status != 200 or answer.get("sparkpipe", {}).get("fleet") != "F3" or headers != {"Authorization": f"Bearer {POOL_KEY}", "X-SparkPipe-Fleet": "F3", "X-SparkPipe-Account": C, "X-SparkPipe-Accelerator": None} or accounts != [C] or "sparkpipe" in body:
        failures.append(f"the gateway does not forward a holder with its Fleet and account, or forwards the customer's own copies: {status} {headers} {accounts} {body}")
    status, answer = chat(port, keys[H], {})
    if status != 200 or upstream.seen[-1][0]["X-SparkPipe-Fleet"] != "F2" or upstream.seen[-1][0]["X-SparkPipe-Account"] != H:
        failures.append(f"a Week holder is not forwarded to its pinned Fleet: {status} {answer}")
    status, _, raw = call(port, "GET", "/v1/models", headers={"Authorization": f"Bearer {keys[C]}"})
    if status != 200 or "kimi-k3" not in [model["id"] for model in json.loads(raw)["data"]]:
        failures.append("a holder cannot list the models")
    before = balance(app, C)
    status, answer = chat(port, keys[C], {"sparkpipe": {"fleet": "F1", "accelerator": {"bid": 2}}})
    note = answer.get("sparkpipe", {}).get("accelerator") or {} if status == 200 else {}
    if status != 200 or upstream.seen[-1][0]["X-SparkPipe-Accelerator"] != "A1" or note.get("seconds") != "0.2" or note.get("amount") != "0.000666" or balance(app, C) != before - Decimal("0.000666"):
        failures.append(f"an accelerated prompt is not sent to its accelerator and charged the reported seconds in HC: {status} {note}")
    status, _, raw = call(port, "POST", "/v1/chat/completions", {"model": "kimi-k3", "messages": [{"role": "user", "content": "stream"}], "stream": True}, {"Authorization": f"Bearer {keys[H]}"})
    events = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
    if status != 200 or events[-1] != "[DONE]" or json.loads(events[-2])["sparkpipe"]["fleet"] != "F2":
        failures.append(f"a streamed request is not passed through: {events}")
    if chat(port, keys[C], {"model": "gpt-9", "sparkpipe": {"fleet": "F3"}})[0] != 404:
        failures.append("a model that is not listed is forwarded")


def holders_and_maintenance(failures, app, m, port, state):
    H, C = state["H"], state["C"]
    status, found = feed(port, "F2", slot(kst(2027, 1, 6)), slot(kst(2027, 1, 7)))
    if status != 200 or found["fleet"] != "F2" or [h["start"] for h in found["hours"]] != [slot(kst(2027, 1, 6) + market.HOUR * n) for n in range(24)] or {h["account"] for h in found["hours"]} != {H}:
        failures.append(f"the holder feed does not list every hour of a booked Week: {status} {found}")
    status, found = feed(port, "F3", slot(kst(2027, 1, 6, 12)), slot(kst(2027, 1, 6, 16)))
    if found.get("hours") != [{"start": slot(kst(2027, 1, 6, 13)), "account": C}, {"start": slot(kst(2027, 1, 6, 14)), "account": C}]:
        failures.append(f"the holder feed does not list HC-booked Hours: {found}")
    refusals = [feed(port, "F2", slot(kst(2027, 1, 6)), slot(kst(2027, 1, 7)), "wrong")[0], feed(port, "F9", slot(kst(2027, 1, 6)), slot(kst(2027, 1, 7)))[0],
                feed(port, "F2", slot(kst(2027, 1, 6)), slot(kst(2027, 1, 14, 1)))[0], feed(port, "F2", "", slot(kst(2027, 1, 7)))[0]]
    if refusals != [401, 404, 400, 400]:
        failures.append(f"the holder feed does not refuse a bad key, an unknown Fleet or a bad window: {refusals}")
    planned = m.schedule_maintenance("F2", "2027-01-18")
    if planned["start"] != slot(kst(2027, 1, 18)) or planned["hours"] != 3 or refused(lambda: m.schedule_maintenance("F2", "2027-01-18", 1)) != (400, None) or refused(lambda: m.schedule_maintenance("F2", "2027-01-19")) != (400, None):
        failures.append(f"maintenance is not limited to the weekly allowance in a Week starting on Monday: {planned}")
    state["maintenance"] = planned["maintenance"]
    status, found = feed(port, "F2", slot(kst(2027, 1, 17, 21)), slot(kst(2027, 1, 18, 5)))
    if [h["start"] for h in found["hours"]] != [slot(kst(2027, 1, 17, h)) for h in (21, 22, 23)] + [slot(kst(2027, 1, 18, h)) for h in (3, 4)]:
        failures.append(f"the holder feed lists maintenance hours as held: {found}")
    status, kind, raw = call(port, "GET", "/availability.txt")
    if status != 200 or not kind.startswith("text/plain") or not raw.decode().endswith("F1 serving demo\nF2 serving pinned\nF3 serving free\nF4 serving free\n"):
        failures.append(f"availability.txt: {status} {kind} {raw[-120:]!r}")
    if client(app, port, C)("POST", "/market/sandbox/clock", {"hours": 1})[0] != 404:
        failures.append("the live service moves its clock")


def deadline(failures, app, m, mailbox, state):
    H, C, G, K, A, B, W = (state[name] for name in "HCGKABW")
    NOW[0] = kst(2027, 1, 10)
    m.tick()
    if [m.position(x) for x in (B, W, A)] != [1, 2, 3] or app.book.market.round != market.stamp(kst(2027, 1, 10)):
        failures.append("a round with no free Fleet changes the waitlist")
    NOW[0] = kst(2027, 1, 14, 12)
    m.leave(W)
    if balance(app, W) != 1000 or [m.position(x) for x in (B, A)] != [1, 2] or refused(lambda: m.quote_weeks(C, {"fleet": "F3", "weeks": 1})) != (409, "pinned"):
        failures.append("leaving the waitlist, or the holder's deadline before Sunday 00:00")
    NOW[0] = kst(2027, 1, 16, 23, 59)
    if refused(lambda: m.quote_weeks(G, {"fleet": "F3", "weeks": 1})):
        failures.append("the holder cannot extend before Sunday 00:00")
    NOW[0] = kst(2027, 1, 17)
    m.tick()
    app.mails.join()
    granted = {b["account"]: b for b in app.book.market.bookings.values() if b["source"] == "waitlist"}
    expect = {B: ("F3", slot(kst(2027, 1, 18)), slot(kst(2027, 2, 1)), "288.000000", "12.000000"), A: ("F4", slot(kst(2027, 1, 18)), slot(kst(2027, 1, 25)), "144.000000", "106.000000")}
    if {who: (b["fleet"], b["start"], b["end"], b["hc"], b["unlocked"]) for who, b in granted.items()} != expect or balance(app, B) != 712 or balance(app, A) != 856 or app.book.market.locks:
        failures.append(f"Sunday 00:00 does not allocate each free Fleet to the waitlist in order, as many Weeks as each lock pays for with the rest unlocked: {granted}")
    if refused(lambda: m.book_weeks(G, {"fleet": "F3", "weeks": 1})) != (409, "taken") or refused(lambda: m.book_weeks(K, {"fleet": "F4", "weeks": 1})) != (409, "taken"):
        failures.append("a holder past its Sunday deadline keeps a Fleet the waitlist was allocated")
    if not any("b@example.com" in rcpt and m_["Subject"].startswith("SparkPipe Fleet 3 is yours") for rcpt, m_ in mailbox.mail):
        failures.append("the allocated customer is not emailed")
    NOW[0] = kst(2027, 1, 30, 23)
    if refused(lambda: m.book_weeks(C, {"fleet": "F3", "weeks": 1})) != (409, "pinned"):
        failures.append("another customer books before the holder's Sunday deadline")
    NOW[0] = kst(2027, 1, 31, 0, 30)
    if refused(lambda: m.book_weeks(H, {"fleet": "F2", "weeks": 1})) or refused(lambda: m.book_weeks(C, {"fleet": "F3", "weeks": 1})):
        failures.append("with nobody allocated at Sunday 00:00, the holder or another customer cannot book the next Week before it starts")
    credited = [lot for lot in app.book.market.lots[H] if lot["source"] == "maintenance"]
    if [(lot["first"], lot["ref"]) for lot in credited] != [("3.000000", state["maintenance"])]:
        failures.append(f"booked hours under maintenance are not credited at the Fleet's Rate: {credited}")


def order_lot(app, account, count, price, total):
    ident = sp.new_id("SP")
    app.book.append("order", {"order": ident, "account": account, "email": "x@example.com", "company": "X", "product": "credits", "processor": "toss", "currency": "KRW", "hc": str(count), "price": price, "net": "0", "vat": "0", "total": total})
    app.market.issue(account, Decimal(count), ident)
    return ident


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
        key = m.create_key(S, "s")["key"]
        m.book_weeks(S, {"fleet": "F2", "weeks": 52})
        for who, lock in ((X, 3200), (Y, 600), (Z, 144)):
            m.join(who, {"lock": lock})
        NOW[0] = kst(2027, 1, 7, 12)
        app.market.credit(L2, Decimal(150), "b")
        app.market.credit(L3, Decimal(100), "d")
        NOW[0] = kst(2027, 1, 10, 0, 1)
        m.tick()
        granted = {b["account"]: (b["fleet"], b["weeks"], b["hc"], b["unlocked"], b["discount"]) for b in app.book.market.bookings.values() if b["source"] == "waitlist"}
        if granted != {X: ("F3", 26, "3182.400000", "17.600000", "0.15"), Y: ("F4", 4, "547.200000", "52.800000", "0.05")} or m.position(Z) != 1:
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
        if balance(app, Q) != 0 or balance(app, S) != 0 or not view["lots"][0]["expired"] or m.held(S, NOW[0]) != ["F2"] or m.gateway(key, None)[1] != "F2" or view["bookings"][0]["state"] != "live":
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


def sandbox_flows(failures, work):
    site(work / "plain-root", serving=False)
    cfg = sp.Config({"SPARKPIPE_PAY_DATA": str(work / "sandbox"), "SPARKPIPE_PAY_SITE": str(work / "plain-root" / "site"), "SPARKPIPE_PAY_SANDBOX": "1", "SPARKPIPE_PAY_PREFIX": "/pay/sandbox/api", "SPARKPIPE_PAY_SANDBOX_ACCELERATORS": ACCELERATORS})
    NOW[0] = datetime(2026, 11, 2, 3, 0, 0, tzinfo=timezone.utc)
    server = sp.start(cfg, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, app = server.server_address[1], server.app
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
        if user("POST", "/market/book-hours", {"fleet": "F1", "start": start, "hours": 2})[0] != 200 or user("POST", "/market/book-weeks", {"fleet": "F2", "weeks": 1})[0] != 200:
            failures.append("the sandbox does not book Hours and Weeks before the published opening dates")
        key = user("POST", "/market/keys", {"note": "try"})[1]["key"]
        if chat(port, key, {})[0] != 403:
            failures.append("a sandbox reply is given before the booked Hour starts")
        if user("POST", "/market/sandbox/clock", {"hours": 1})[1].get("now") != "2026-11-02T04:00:00Z":
            failures.append("the sandbox clock does not move")
        status, answer = chat(port, key, {"max_tokens": 6})
        if status != 200 or not answer["choices"][0]["message"]["content"].startswith("Sandbox reply from SparkPipe F1") or answer["sparkpipe"]["fleet"] != "F1":
            failures.append(f"the sandbox does not simulate replies for a holder: {status} {answer}")
        orphan = accelerators(failures, app, port, key, app.book.emails["try@example.com"])
    finally:
        server.shutdown()
    restarted = sp.start(cfg, port=0)
    threading.Thread(target=restarted.serve_forever, daemon=True).start()
    user.port = restarted.server_address[1]
    try:
        for _ in range(200):
            if restarted.app.book.market.held(orphan) == 0:
                break
            time.sleep(0.01)
        if restarted.app.book.market.held(orphan) != 0:
            failures.append("a restart does not return HC held for prompts that died with the old process")
        if user("POST", "/market/sandbox/reset", {})[1] != {"reset": True} or user("GET", "/market/state")[1]["account"] is not None:
            failures.append("the sandbox does not reset")
    finally:
        restarted.shutdown()


def joined(app, ticket, options, tokens, out):
    try:
        out.append(app.market.accelerator_join(ticket, options, tokens))
    except market.Refusal as error:
        out.append(error)


def waiting_for(app, count):
    for _ in range(500):
        with app.market.queue:
            if len(app.market.waiting) >= count:
                return
        time.sleep(0.01)


def accelerators(failures, app, port, key, account):
    m, limits = app.market, app.cfg.pricing()["accelerator"]
    options = lambda **given: market.accelerator_field(dict({"bid": 1, "maxWait": 30}, **given), limits)
    for bad in ({"bid": "0.5"}, {"bid": 11}, {"bid": "1.234"}, {"bid": True}, {"maxWait": 601}, {"maxWait": "60"}, {"fallback": "yes"}):
        status, answer = chat(port, key, {"sparkpipe": {"accelerator": bad}})
        if status != 400 or answer["error"]["code"] != "invalid_accelerator":
            failures.append(f"an invalid accelerator request is accepted: {bad}")
    listed = [(a["id"], a["rate"], a["hcPerMinute"], a["tokensPerSecond"]) for a in m.public()["accelerators"]]
    if listed != [("A1", "6", "0.100000", "400"), ("A2", "3", "0.050000", "200")]:
        failures.append(f"the accelerators are listed with their Rate and speed: {listed}")
    base = balance(app, account)
    ticket = {"account": account, "key": "KTEST", "model": "kimi-k3", "fleet": "F1"}
    first, second = m.accelerator_join(ticket, options(), 400), m.accelerator_join(ticket, options(), 400)
    if first["accelerator"]["id"] != "A1" or second["accelerator"]["id"] != "A2":
        failures.append("free accelerators take prompts at once, fastest first")
    results, threads = {"low": [], "high": [], "late": []}, []
    for name, bid in (("low", 1), ("high", 2), ("late", 1)):
        threads.append(threading.Thread(target=joined, args=(app, ticket, options(bid=bid), 400, results[name])))
        threads[-1].start()
        waiting_for(app, len(threads))
    if [q["bid"] for q in m.public()["acceleratorQueue"]] != ["2", "1", "1"]:
        failures.append("the published queue does not order prompts by bid, then arrival")
    if m.accelerator_join(ticket, options(maxWait=0), 400) != {"used": False, "reason": "waited"} or refused(lambda: m.accelerator_join(ticket, options(maxWait=0, fallback=False), 400)) != (503, "accelerator_busy"):
        failures.append("a prompt no accelerator starts within its wait does not fall back, or is not refused when asked")
    held = sum((j["estimate"] for j in [first, second] + m.waiting), Decimal(0))
    if Decimal(m.view(account)["held"]) != held or balance(app, account) != base - held:
        failures.append("waiting and running prompts do not hold their estimates out of the balance")
    charged = m.accelerator_finish(first, 60)
    threads[1].join(5)
    if charged.get("amount") != "0.100000" or results["high"][0]["accelerator"]["id"] != "A1" or results["low"] or m.accelerator_finish(first, 60) != charged:
        failures.append(f"the highest bid takes the next free accelerator, and a prompt is charged once: {charged}")
    if m.accelerator_finish(second, None) != {"used": False, "reason": "unfinished", "accelerator": "A2"}:
        failures.append("an unfinished prompt is charged")
    threads[0].join(5)
    amounts = [charged["amount"]] + [m.accelerator_finish(job, 1)["amount"] for job in (results["high"][0], results["low"][0])]
    threads[2].join(5)
    amounts.append(m.accelerator_finish(results["late"][0], 1)["amount"])
    if Decimal(m.view(account)["held"]) != 0 or m.busy_accelerators or m.waiting or balance(app, account) != base - sum(Decimal(a) for a in amounts):
        failures.append(f"finished prompts leave HC held, or the balance is not charged exactly what they cost: {amounts} {balance(app, account)}")
    if refused(lambda: m.accelerator_join(ticket, options(), 10 ** 9)) != (402, "insufficient_hc"):
        failures.append("a prompt whose estimate the balance does not cover joins the queue")
    return m.accelerator_join(ticket, options(), 400)["job"]


def main():
    failures = []
    work = Path(tempfile.mkdtemp(prefix="sparkpipe-market-"))
    try:
        check_units(failures)
        site(work)
        live_flows(failures, work)
        allocation_flows(failures, work)
        sandbox_flows(failures, work)
    finally:
        shutil.rmtree(work)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("market: HC lots lapse 12 months after issue while HC spent on bookings does not; Demo Fleet hours open on demoFrom, up to 7 days ahead and 24 a Week per account; free-Fleet Hours only to the end of the current Week and never on a pinned Fleet; Weeks from the next Monday 00:00 KST, priced with discounts at 4, 13, 26 and 52 Weeks counting the current Week, extended only contiguously and at most 52 Weeks ahead; nobody else books a pinned Fleet before its holder's Sunday 00:00 deadline; the Sunday round allocates free Fleets to the waitlist by largest lock then earliest join, booking as many Weeks as each lock pays for and unlocking the rest, and leaves a Fleet bookable by its holder or anyone when nobody is allocated; customers see only their own position; locked HC that lapses is unlocked and recorded as a refund due; the gateway forwards only holders, with X-SparkPipe-Fleet and X-SparkPipe-Account and no customer copies; the holder feed lists Week and Hour holders without maintenance hours; maintenance credits booked hours; availability.txt; prefill accelerators in HC; the sandbox clock, instant HC, simulated replies and reset; and the ledger replays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
