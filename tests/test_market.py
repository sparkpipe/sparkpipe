#!/usr/bin/env python3

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
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import sparkpipe_market as market
import sparkpipe_pay as sp
import test_pay as tp

START = datetime(2026, 11, 2, 3, 0, 0, tzinfo=timezone.utc)


def at(hours):
    return market.slot_text(START + timedelta(hours=hours))


def site(work):
    (work / "site" / "terms").mkdir(parents=True)
    shutil.copy(ROOT / "site" / "pricing.js", work / "site" / "pricing.js")
    for name in sp.TERMS.values():
        shutil.copy(ROOT / "site" / "terms" / name, work / "site" / "terms" / name)


def sandbox_in(client, address, company):
    status, sent = client("POST", "/code", {"email": address})
    if status != 200 or not sent.get("code"):
        return status, sent
    return client("POST", "/verify", {"email": address, "code": sent["code"], "company": company, "accept": True})


def balance(client):
    return Decimal(client("GET", "/market/state")[1]["account"]["balance"])


def text_get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
        return response.status, response.headers.get("Content-Type"), response.read().decode()


def pool_call(port, key, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/pool/v1/chat/completions", data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def check_units(failures):
    if market.text_hc("1.23456789") != "1.234567" or market.slot_text(START) != "2026-11-02T03:00Z":
        failures.append("HC rounds down to 6 places and hours print as UTC slots")
    if market.year_later(datetime(2028, 2, 29, tzinfo=timezone.utc)) != datetime(2029, 2, 28, tzinfo=timezone.utc):
        failures.append("a lot bought on 29 February lapses on 28 February a year later")
    offers = [{"fleet": "F1", "start": at(h), "price": Decimal(p), "kind": "hyor", "seq": 0, "seller": None} for h, p in ((1, "1"), (2, "0.5"), (3, "0.5"), (4, "1"))]
    run = market.Market.choose(None, offers, 2, True, Decimal("1"), "AC")
    if [o["start"] for o in run] != [at(2), at(3)]:
        failures.append(f"a consecutive bid takes the cheapest run: {run}")
    released = dict(offers[0], kind="release", seq=7, seller="AS")
    if market.Market.choose(None, [offers[0], released], 1, False, Decimal("1"), "AC")[0]["kind"] != "release":
        failures.append("at equal prices a released hour fills before ours")
    if market.Market.choose(None, [released], 1, False, Decimal("1"), "AS"):
        failures.append("a bid fills from its own released hour")


def sandbox_flows(failures, work):
    cfg = sp.Config({"SPARKPIPE_PAY_DATA": str(work / "sandbox"), "SPARKPIPE_PAY_SITE": str(work / "site"), "SPARKPIPE_PAY_SANDBOX": "1", "SPARKPIPE_PAY_PREFIX": "/pay/sandbox/api"})
    server = sp.start(cfg, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, app = server.server_address[1], server.app
    seller, buyer, anyone = tp.Client(port, origin="https://sparkpipe.ai"), tp.Client(port, origin="https://sparkpipe.ai"), tp.Client(port)
    try:
        state = anyone("GET", "/market/state")[1]
        if not state["sandbox"] or state["account"] is not None or [f["id"] for f in state["fleets"]] != ["F1", "F2", "F3"] or state["market"]["hcPrice"] != 20:
            failures.append(f"the public market state: {state}")
        status, data = sandbox_in(seller, "seller@example.com", "Seller Co")
        if status != 200 or "Path=/pay/sandbox/api" not in seller.last_cookie:
            failures.append(f"sandbox sign-in: {status} {data}")
        sandbox_in(buyer, "buyer@example.com", "Buyer Co")
        if seller("POST", "/market/buy", {"amount": "5"})[0] != 400:
            failures.append("a purchase under the minimum is accepted")
        bought = seller("POST", "/market/buy", {"amount": "100"})[1]
        buyer("POST", "/market/buy", {"amount": "200"})
        lots = seller("GET", "/market/state")[1]["account"]["lots"]
        if bought.get("hc") != "5.000000" or balance(seller) != 5 or balance(buyer) != 10 or lots[0]["expires"] != "2027-11-02T03:00:00Z":
            failures.append(f"HC bought by amount, lasting 12 months: {bought} {lots}")
        cal = buyer("GET", "/market/calendar?days=10")[1]
        cells = cal["fleets"][0]["cells"]
        if cal["from"] != at(1) or len(cells) != 240 or cells[0] != ["hyor", "1.000000"] or cells[167] != ["hyor", "1.000000"] or cells[168] != ["hyor", "1.250000"]:
            failures.append(f"the calendar sells our hours at Rate 7 days ahead and at the advance price after: {cal['from']} {cells[0]} {cells[167]} {cells[168]}")
        reserve_and_release(failures, seller, buyer)
        bids(failures, seller, buyer)
        pool(failures, app, port, seller, buyer)
        fee_locked(failures, work, seller, buyer)
        replay = sp.Book(cfg.path("ledger.jsonl"))
        moment = market.stamp(app.market.now())
        accounts = list(app.book.accounts)
        if [replay.market.balance(a, moment) for a in accounts] != [app.book.market.balance(a, moment) for a in accounts]:
            failures.append("a replayed ledger gives different HC balances")
        lines = buyer("GET", "/market/ledger")[1]["lines"]
        mine = app.book.emails["buyer@example.com"]
        others = [line for line in lines if any(line["data"].get(name) not in (None, mine) for name in ("account", "seller"))]
        kinds = {line["type"] for line in lines}
        if not lines or others or not all(line.get("hash") for line in lines) or not {"bid_cancel", "hc_unhold", "hc_spend_hold"} <= kinds:
            failures.append(f"the ledger export holds every line of the account and no other account's: {sorted(kinds)} {others[:2]}")
        seller_lines = seller("GET", "/market/ledger")[1]["lines"]
        if not {"ask_cancel", "pool_join", "pool_settle"} <= {line["type"] for line in seller_lines} or any(line["data"].get("account") not in (None, app.book.emails["seller@example.com"]) for line in seller_lines):
            failures.append(f"the seller's ledger export misses its releases or shows the buyer's lines: {sorted({line['type'] for line in seller_lines})}")

        status, kind, text = text_get(port, "/market/market.txt")
        if status != 200 or not kind.startswith("text/plain") or not re.search(r"\n\d{4}-\d\d-\d\dT\d\d:00Z F1 1\.000000 hyor -\n", text) or "# last trades" not in text:
            failures.append(f"market.txt lists the best ask per fleet-hour: {text[:300]}")
        seller("POST", "/market/sandbox/clock", {"hours": 24 * 367})
        if balance(seller) != 0 or balance(buyer) != 0:
            failures.append(f"HC does not lapse 12 months after issue: {seller('GET', '/market/state')[1]['account']['lots']} {buyer('GET', '/market/state')[1]['account']}")
        if seller("POST", "/market/sandbox/reset", {})[1] != {"reset": True} or seller("GET", "/market/state")[1]["account"] is not None or app.book.count != 0:
            failures.append("the sandbox reset leaves state behind")
    finally:
        server.shutdown()


def fee_locked(failures, work, seller, buyer):
    held = seller("POST", "/market/reserve", {"fleet": "F1", "start": at(60), "model": "kimi-k3", "max": "1"})[1]
    seller("POST", "/market/release", {"reservation": held.get("reservation"), "min": "1"})
    prices = work / "site" / "pricing.js"
    text = prices.read_text()
    prices.write_text(text.replace('"releaseFee": 0.10', '"releaseFee": 0.5'))
    before = balance(seller)
    status, bought = buyer("POST", "/market/reserve", {"fleet": "F1", "start": at(60), "model": "kimi-k3", "max": "1"})
    prices.write_text(text)
    if status != 200 or balance(seller) != before + Decimal("0.9"):
        failures.append(f"a fee change applies to a release already on offer: {status} {bought} {before} {balance(seller)}")


def reserve_and_release(failures, seller, buyer):

    status, held = seller("POST", "/market/reserve", {"fleet": "F1", "start": at(5), "model": "kimi-k3", "max": "1"})
    if status != 200 or held["price"] != "1.000000" or balance(seller) != 4:
        failures.append(f"reserving our hour at Rate: {status} {held}")
    if buyer("POST", "/market/reserve", {"fleet": "F1", "start": at(5), "model": "kimi-k3"})[0] != 409:
        failures.append("an hour is sold twice")
    if buyer("POST", "/market/reserve", {"fleet": "F1", "start": at(6), "model": "kimi-k3", "max": "0.5"})[0] != 409:
        failures.append("a reservation goes through above the price the buyer saw")
    status, ask = seller("POST", "/market/release", {"reservation": held["reservation"], "min": "0.8"})
    if status != 200 or ask["state"] != "open":
        failures.append(f"releasing a reserved hour: {status} {ask}")
    if seller("POST", "/market/reserve", {"fleet": "F1", "start": at(5), "model": "kimi-k3"})[0] != 409:
        failures.append("a seller buys back its own release")
    status, bought = buyer("POST", "/market/reserve", {"fleet": "F1", "start": at(5), "model": "glm-5.3", "max": "0.8"})
    if status != 200 or bought["price"] != "0.800000" or balance(buyer) != Decimal("9.2") or balance(seller) != Decimal("4.72"):
        failures.append(f"a released hour sells at its ask and pays the seller less the fee: {bought} {balance(buyer)} {balance(seller)}")
    mine = {r["reservation"]: r for r in seller("GET", "/market/state")[1]["account"]["reservations"]}
    if mine[held["reservation"]]["state"] != "sold" or mine[held["reservation"]]["release"]["price"] != "0.800000":
        failures.append(f"the seller sees its hour sold: {mine}")
    trades = buyer("GET", "/market/book")[1]["trades"]
    if trades[0]["kind"] != "release" or trades[0]["price"] != "0.800000":
        failures.append(f"the trade is published: {trades[:1]}")
    status, again = seller("POST", "/market/reserve", {"fleet": "F2", "start": at(8), "model": "kimi-k3"})
    seller("POST", "/market/release", {"reservation": again["reservation"]})
    asks = seller("GET", "/market/state")[1]["account"]["reservations"]
    open_ask = next(r["release"]["ask"] for r in asks if r["reservation"] == again["reservation"])
    if seller("POST", "/market/cancel", {"ask": open_ask})[1].get("state") != "cancelled" or next(r for r in seller("GET", "/market/state")[1]["account"]["reservations"] if r["reservation"] == again["reservation"])["state"] != "held":
        failures.append("cancelling a release keeps the hour")


def bids(failures, seller, buyer):
    before = balance(buyer)
    status, bid = buyer("POST", "/market/bid", {"fleet": "F2", "model": "kimi-k3", "from": at(20), "to": at(30), "hours": 2, "consecutive": True, "max": "1"})
    if status != 200 or bid["state"] != "filled" or balance(buyer) != before - 2:
        failures.append(f"a consecutive bid fills from our capacity at Rate: {status} {bid} {balance(buyer)}")
    status, resting = buyer("POST", "/market/bid", {"fleet": None, "model": "kimi-k3", "from": at(24 * 8), "to": at(24 * 9), "hours": 1, "max": "0.5"})
    account = buyer("GET", "/market/state")[1]["account"]
    if status != 200 or resting["state"] != "open" or account["held"] != "0.500000":
        failures.append(f"a bid beyond our horizon rests and holds its HC: {resting} {account['held']}")
    granted = seller("POST", "/market/sandbox/tranche", {})[1]
    status, tranche = seller("POST", "/market/reserve", {"fleet": "F3", "start": at(24 * 8 + 2), "model": "kimi-k3", "pay": "tranche"})
    left = seller("GET", "/market/state")[1]["account"]["tranche"][0]
    if status != 200 or tranche["price"] != "0.000000" or left["left"] != granted["hours"] - 1:
        failures.append(f"Tranche hours reserve beyond the horizon: {status} {tranche} {left}")
    seller_before, buyer_before = balance(seller), balance(buyer)
    seller("POST", "/market/release", {"reservation": tranche["reservation"], "min": "0.4"})
    filled = next(b for b in buyer("GET", "/market/state")[1]["account"]["bids"] if b["bid"] == resting["bid"])
    if filled["state"] != "filled" or balance(seller) != seller_before + Decimal("0.45") or balance(buyer) != buyer_before:
        failures.append(f"a release fills the older resting bid at the bid's price: {filled} {balance(seller)} {balance(buyer)}")
    status, standing = buyer("POST", "/market/bid", {"fleet": "F1", "model": "kimi-k3", "from": at(40), "to": at(42), "hours": 1, "max": "1", "standing": True})
    if status != 200 or standing["state"] != "filled":
        failures.append(f"a standing bid fills this week: {standing}")
    buyer("POST", "/market/bid", {"fleet": "F2", "model": "kimi-k3", "from": at(24 * 10), "to": at(24 * 10 + 3), "hours": 1, "max": "0.3"})
    if buyer("POST", "/market/bid", {"fleet": "F1", "model": "kimi-k3", "from": at(50), "to": at(50 + 24 * 8), "hours": 1, "max": "1", "standing": True})[0] != 400:
        failures.append("a standing bid accepts a window longer than a week")
    if buyer("POST", "/market/bid", {"model": "kimi-k3", "from": at(50), "to": at(52), "hours": 1, "max": "900"})[0] != 402:
        failures.append("a bid holds more HC than the account has")
    before = balance(buyer)
    seller("POST", "/market/sandbox/clock", {"hours": 43})
    shown = {b["bid"]: b for b in buyer("GET", "/market/state")[1]["account"]["bids"]}
    rolled = shown.get(standing["bid"])
    if not rolled or rolled["state"] != "filled" or rolled["from"] != at(40 + 168) or balance(buyer) != before - 1:
        failures.append(f"a standing bid rolls to next week and fills again: {rolled} {balance(buyer)} vs {before}")
    status, waiting = buyer("POST", "/market/bid", {"fleet": "F2", "model": "kimi-k3", "from": at(24 * 12), "to": at(24 * 12 + 3), "hours": 1, "max": "0.7"})
    before = balance(buyer)
    if buyer("POST", "/market/cancel", {"bid": waiting["bid"]})[1].get("state") != "cancelled" or balance(buyer) != before + Decimal("0.7"):
        failures.append("cancelling a bid returns its held HC")


def pool(failures, app, port, seller, buyer):
    key = buyer("POST", "/market/keys", {"note": "agents"})[1]["key"]
    models = urllib.request.Request(f"http://127.0.0.1:{port}/pool/v1/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(models, timeout=10) as response:
        if [m["id"] for m in json.loads(response.read())["data"]] != ["kimi-k3", "glm-5.3"]:
            failures.append("the pool lists its models")
    if pool_call(port, "sk-hc-wrong", {"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]})[0] != 401:
        failures.append("the pool takes a wrong key")
    status, held = seller("POST", "/market/reserve", {"fleet": "F3", "start": market.slot_text(sp.clock() + timedelta(seconds=app.book.market.offset) + timedelta(hours=2)), "model": "kimi-k3"})
    seller("POST", "/market/release", {"reservation": held["reservation"]})
    seller("POST", "/market/sandbox/clock", {"hours": 2})
    reservation = next(r for r in seller("GET", "/market/state")[1]["account"]["reservations"] if r["reservation"] == held["reservation"])
    if reservation["state"] != "pooled":
        failures.append(f"an unsold release joins the pool at its start: {reservation}")
    before = balance(buyer)
    status, raw = pool_call(port, key, {"model": "kimi-k3", "messages": [{"role": "user", "content": "Summarize the Hour Market."}], "max_tokens": 20})
    answer = json.loads(raw) if status == 200 else {}
    spent = Decimal(answer.get("sparkpipe", {}).get("hc", "0"))
    if status != 200 or not answer["choices"][0]["message"]["content"].startswith("Sandbox reply") or spent <= 0 or balance(buyer) != before - spent:
        failures.append(f"a pool request is answered and charged by seat-seconds: {status} {raw[:200]}")
    status, raw = pool_call(port, key, {"model": "kimi-k3", "messages": [{"role": "user", "content": "Stream it."}], "max_tokens": 8, "stream": True})
    events = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
    streamed = Decimal(json.loads(events[-2])["sparkpipe"]["hc"]) if len(events) > 2 else Decimal(0)
    if status != 200 or events[-1] != "[DONE]" or streamed <= 0:
        failures.append(f"a streamed pool request ends with its charge: {status} {events[-2:]}")
    charges = buyer("GET", "/market/state")[1]["account"]["charges"]
    if len(charges) != 2 or Decimal(charges[0]["hc"]) != streamed:
        failures.append(f"the account lists its pool charges: {charges}")
    seller_before = balance(seller)
    seller("POST", "/market/sandbox/clock", {"hours": 1})
    expect = market.hc((spent + streamed) * Decimal("0.9"))
    reservation = next(r for r in seller("GET", "/market/state")[1]["account"]["reservations"] if r["reservation"] == held["reservation"])
    if balance(seller) != seller_before + expect or reservation["release"]["earned"] != market.text_hc(expect):
        failures.append(f"a pooled release earns its seat-seconds less the fee: {balance(seller)} vs {seller_before} + {expect}, {reservation['release']}")
    status, _, text = text_get(port, "/market/pool.txt")
    if status != 200 or "\nkimi-k3 " not in text or "\nglm-5.3 " not in text:
        failures.append(f"pool.txt lists the price per model: {text}")
    revoke = buyer("GET", "/market/state")[1]["account"]["keys"][0]["key"]
    buyer("POST", "/market/keys/revoke", {"id": revoke})
    if pool_call(port, key, {"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]})[0] != 401:
        failures.append("a revoked key still uses the pool")


def plan(number, email, windows, fleets=1, days="every", lock=None, fee="0.10"):
    found = {"id": f"SP-COVER-{number}", "email": email, "company": "Plan Co", "start": "2026-11-02", "months": 24, "windows": [tp.WINDOWS[w] for w in windows], "days": days, "fleets": fleets,
             "hourPrice": "15.00", "payment": "gates", "releaseFee": fee, "maintenanceHoursPerWeek": 3}
    if lock:
        found["lock"] = lock
    return found


def cover_flows(failures, work):
    cfg = sp.Config({"SPARKPIPE_PAY_DATA": str(work / "cover"), "SPARKPIPE_PAY_SITE": str(work / "site"), "SPARKPIPE_PAY_SANDBOX": "1", "SPARKPIPE_PAY_PREFIX": "/pay/sandbox/api"})
    server = sp.start(cfg, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, app = server.server_address[1], server.app
    holder, early, late = tp.Client(port, origin="https://sparkpipe.ai"), tp.Client(port, origin="https://sparkpipe.ai"), tp.Client(port, origin="https://sparkpipe.ai")
    try:
        sandbox_in(holder, "holder@example.com", "Plan Co")
        sandbox_in(early, "early@example.com", "Early Co")
        sandbox_in(late, "late@example.com", "Late Co")
        asia, europe, friday, saturday, sunday = 22, 30, 94, 118, 132
        three = [plan(1, "holder@example.com", ["asia"], 3), plan(2, "b@example.com", ["europe"], 3), plan(3, "c@example.com", ["americas"], 3)]
        tp.set_plans(app, three)
        cells = [f["cells"] for f in late("GET", "/market/calendar?days=14")[1]["fleets"]]
        if any(c[4] != ["hyor", "1.000000"] or c[5] != ["closed", None] or c[asia - 1] != ["closed", None] or c[sunday - 1] != ["closed", None] for c in cells):
            failures.append(f"three-fleet plans in every window keep every fleet off sale every day until 6 hours before: {[c[4:6] + [c[asia - 1], c[sunday - 1]] for c in cells]}")
        if late("POST", "/plans/quote", {"start": "2026-12-15", "months": 3})[0] != 404:
            failures.append("the sandbox takes Plan Orders")
        status, refused = late("POST", "/market/buy", {"amount": "100"})
        if status != 409 or refused.get("reason") != "cover" or late("GET", "/market/state")[1]["onSale"] != "0.000000":
            failures.append(f"HC is sold although every hour is kept for plans: {status} {refused}")
        holder("POST", "/market/sandbox/tranche", {})
        if holder("POST", "/market/reserve", {"fleet": "F1", "start": at(asia), "model": "kimi-k3", "pay": "tranche"})[0] != 200:
            failures.append("a plan holder cannot reserve its kept capacity in its window with Tranche hours")
        if holder("POST", "/market/reserve", {"fleet": "F1", "start": at(europe), "model": "kimi-k3", "pay": "tranche"})[0] != 409:
            failures.append("a plan holder takes capacity kept for another customer's window")
        if app.market.report(app.commitments(app.market.now(), [plan(4, "d@example.com", ["asia"])]))["clashes"] == 0 or app.market.report()["clashes"] != 0:
            failures.append("the cover report misses a plan that the fleets cannot keep, or flags one they can")

        tp.set_plans(app, [plan(1, "holder@example.com", ["asia"], 3)])
        before = [f["cells"] for f in late("GET", "/market/calendar?days=3")[1]["fleets"]]
        if [c[asia + 23] for c in before] != [["closed", None]] * 3 or [c[europe + 23] for c in before] != [["hyor", "1.000000"]] * 3:
            failures.append(f"an Asia plan for three fleets does not keep exactly the Asia Window: {[c[asia + 23] for c in before]} {[c[europe + 23] for c in before]}")
        tp.set_plans(app, [plan(1, "holder@example.com", ["asia"], 3, "weekdays")])
        week = [f["cells"] for f in late("GET", "/market/calendar?days=7")[1]["fleets"]]
        if [c[friday - 1] for c in week] != [["closed", None]] * 3 or [c[saturday - 1] for c in week] != [["hyor", "1.000000"]] * 3:
            failures.append(f"a weekday Asia plan keeps Saturday or frees Friday: {[c[friday - 1] for c in week]} {[c[saturday - 1] for c in week]}")
        tp.set_plans(app, [plan(1, "holder@example.com", ["asia"])])
        late("POST", "/market/buy", {"amount": "100"})
        for fleet in ("F1", "F2"):
            late("POST", "/market/reserve", {"fleet": fleet, "start": at(asia + 24), "model": "kimi-k3", "max": "1"})
        if late("POST", "/market/reserve", {"fleet": "F3", "start": at(asia + 24), "model": "kimi-k3", "max": "1"})[0] != 409:
            failures.append("the last fleet of an hour that a plan still needs is sold to HC")

        tp.set_plans(app, [plan(1, "holder@example.com", ["americas", "asia", "europe"], lock=["F2"])])
        locked = [f["cells"] for f in late("GET", "/market/calendar?days=2")[1]["fleets"]]
        pool = late("GET", "/market/state")[1]["pool"]["fleets"]
        if locked[1][0] != ["closed", None] or locked[1][40] != ["closed", None] or locked[0][0] != ["hyor", "1.000000"] or "F2" in pool or "F1" not in pool:
            failures.append(f"a locked fleet is offered to others, even within 6 hours, or joins the shared pool: {locked[1][0]} {locked[1][40]} {locked[0][0]} {pool}")
        if late("POST", "/market/reserve", {"fleet": "F2", "start": at(2), "model": "kimi-k3", "max": "1"})[0] != 409:
            failures.append("HC reserves a locked fleet within 6 hours of the hour")
        if holder("POST", "/market/reserve", {"fleet": "F2", "start": at(40), "model": "kimi-k3", "pay": "tranche"})[0] != 200:
            failures.append("the holder cannot reserve its own locked fleet")

        tp.set_plans(app, [plan(1, "holder@example.com", ["asia"], 2), plan(2, "b@example.com", ["europe"], 2), plan(3, "c@example.com", ["americas"], 2)])
        pricing = cfg.pricing()
        sale = Decimal(late("GET", "/market/state")[1]["onSale"])
        if not 8000 < sale < 8700:
            failures.append(f"with one fleet free every day, HC on sale is not about a year of its hours less 3 maintenance hours a week: {sale}")
        bought = late("POST", "/market/buy", {"amount": "100000"})[1]
        left = Decimal(late("GET", "/market/state")[1]["onSale"])
        if bought.get("hc") != "5000.000000" or left != sale - 5000:
            failures.append(f"buying HC does not take it off sale: {bought} {sale} {left}")
        if late("POST", "/market/buy", {"amount": str((left * 20 + 100).quantize(Decimal(1)))})[0] != 409:
            failures.append("a purchase above the HC on sale goes through")
        with app.book.lock:
            now = app.market.now()
            lapse = START + timedelta(days=3)
            due = app.market.cover(now, pricing).supply(lapse)
            app.book.append("hc_lot", {"lot": "LTEARLY", "account": app.book.emails["early@example.com"], "hc": market.text_hc(due), "expires": market.stamp(lapse), "source": "purchase", "ref": "test", "now": market.stamp(now), "terms": market.published_terms(pricing["market"])})
        status, kept = late("POST", "/market/reserve", {"fleet": "F1", "start": at(48), "model": "kimi-k3", "max": "1"})
        if status != 409 or kept.get("reason") != "kept":
            failures.append(f"HC lapsing in a year takes an hour that HC lapsing sooner needs: {status} {kept}")
        if late("POST", "/market/reserve", {"fleet": "F1", "start": at(3), "model": "kimi-k3", "max": "1"})[0] != 200:
            failures.append("a kept hour does not open to any HC 6 hours before it starts")
        if late("POST", "/market/reserve", {"fleet": "F1", "start": at(24 * 4), "model": "kimi-k3", "max": "1"})[0] != 200:
            failures.append("an hour after the earlier HC lapses is kept from later HC")
        if early("POST", "/market/reserve", {"fleet": "F1", "start": at(48), "model": "kimi-k3", "max": "1"})[0] != 200:
            failures.append("HC lapsing sooner cannot reserve the hour kept for it")
        status, advance = late("POST", "/market/reserve", {"fleet": "F1", "start": at(24 * 9), "model": "kimi-k3", "max": "1.25"})
        if status != 200 or advance.get("price") != "1.250000" or late("POST", "/market/reserve", {"fleet": "F1", "start": at(24 * 31 + 2), "model": "kimi-k3", "max": "2"})[0] != 409:
            failures.append(f"advance hours sell at 1.25 up to 31 days ahead and not beyond: {status} {advance}")
        credited = app.market.credit(app.book.emails["late@example.com"], Decimal(late("GET", "/market/state")[1]["onSale"]) + 100, "outage test")
        if market.stamp_time(credited["expires"]) <= market.year_later(app.market.now()):
            failures.append(f"a credit beyond the cover lapses after 12 months instead of later: {credited}")
        status, kind, text = text_get(port, "/market/market.txt")
        published = re.search(r"# hour-credits on sale now: ([0-9.]+) HC", text)
        if not published or not Decimal(published.group(1)) < 1:
            failures.append(f"market.txt does not publish the HC on sale, or a credit beyond the cover leaves HC on sale: {text[:200]}")
    finally:
        server.shutdown()


def locked_terms(failures, work):
    cfg = sp.Config({"SPARKPIPE_PAY_DATA": str(work / "terms"), "SPARKPIPE_PAY_SITE": str(work / "site"), "SPARKPIPE_PAY_SANDBOX": "1", "SPARKPIPE_PAY_PREFIX": "/pay/sandbox/api"})
    server = sp.start(cfg, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, app = server.server_address[1], server.app
    early, late, buyer, holder = (tp.Client(port, origin="https://sparkpipe.ai") for _ in range(4))
    prices = work / "site" / "pricing.js"
    text = prices.read_text()
    try:
        for client, name in ((early, "early"), (late, "late"), (buyer, "buyer"), (holder, "holder")):
            sandbox_in(client, f"{name}@example.com", f"{name} Co")
        early("POST", "/market/buy", {"amount": "100"})
        buyer("POST", "/market/buy", {"amount": "100"})
        prices.write_text(text.replace('"releaseFee": 0.10', '"releaseFee": 0.5').replace('"advanceMultiplier": 1.25', '"advanceMultiplier": 2'))
        late("POST", "/market/buy", {"amount": "100"})
        first = early("POST", "/market/reserve", {"fleet": "F1", "start": at(24 * 9), "model": "kimi-k3", "max": "3"})[1]
        second = late("POST", "/market/reserve", {"fleet": "F1", "start": at(24 * 9 + 1), "model": "kimi-k3", "max": "3"})[1]
        if first.get("price") != "1.250000" or second.get("price") != "2.000000":
            failures.append(f"an advance hour is not priced by the multiplier in force when the buyer's HC was bought: {first} {second}")
        for client, held, credit in ((early, first, Decimal("0.9")), (late, second, Decimal("0.5"))):
            before = balance(client)
            client("POST", "/market/release", {"reservation": held["reservation"], "min": "1"})
            bought = buyer("POST", "/market/reserve", {"fleet": "F1", "start": at(24 * 9) if held is first else at(24 * 9 + 1), "model": "kimi-k3", "max": "1"})
            if bought[0] != 200 or balance(client) != before + credit:
                failures.append(f"a release is not charged the fee in force when the hour's HC was bought: {bought} {before} {balance(client)}")
        prices.write_text(text)
        status, filled = late("POST", "/market/bid", {"fleet": "F3", "model": "kimi-k3", "from": at(24 * 10), "to": at(24 * 10 + 1), "hours": 1, "max": "3"})
        bought = [r for r in late("GET", "/market/state")[1]["account"]["reservations"] if r["fleet"] == "F3"]
        if status != 200 or filled.get("state") != "filled" or not bought or bought[0]["price"] != "2.000000":
            failures.append(f"a bid fills an advance hour at today's multiplier instead of the one its HC was bought under: {filled} {bought}")
        tp.set_plans(app, [plan(1, "holder@example.com", ["asia"], fee="0.2")])
        holder("POST", "/market/sandbox/tranche", {})
        held = holder("POST", "/market/reserve", {"fleet": "F2", "start": at(22), "model": "kimi-k3", "pay": "tranche"})[1]
        holder("POST", "/market/release", {"reservation": held.get("reservation"), "min": "1"})
        before = balance(holder)
        buyer("POST", "/market/reserve", {"fleet": "F2", "start": at(22), "model": "kimi-k3", "max": "1"})
        if balance(holder) != before + Decimal("0.8"):
            failures.append(f"a released plan hour is not charged its Plan Order's release fee: {before} {balance(holder)}")
    finally:
        prices.write_text(text)
        server.shutdown()


def maintenance_flows(failures, work):
    cfg = sp.Config({"SPARKPIPE_PAY_DATA": str(work / "maint"), "SPARKPIPE_PAY_SITE": str(work / "site"), "SPARKPIPE_PAY_SANDBOX": "1", "SPARKPIPE_PAY_PREFIX": "/pay/sandbox/api"})
    server = sp.start(cfg, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, app = server.server_address[1], server.app
    buyer, holder = tp.Client(port, origin="https://sparkpipe.ai"), tp.Client(port, origin="https://sparkpipe.ai")
    try:
        sandbox_in(buyer, "buyer@example.com", "Buyer Co")
        sandbox_in(holder, "holder@example.com", "Holder Co")
        buyer("POST", "/market/buy", {"amount": "2000"})
        for n in range(169, 325, 3):
            buyer("POST", "/market/reserve", {"fleet": "F1", "start": at(n), "model": "kimi-k3", "max": "1.25"})
        before = balance(buyer)
        for early, label in (("2026-11-02", "with less than 7 days' notice"), ("2026-11-10", "for a week that does not start on a Monday")):
            try:
                app.market.schedule_maintenance("F1", early, 3)
                failures.append(f"maintenance is scheduled {label}")
            except market.Refusal:
                pass
        done = app.market.schedule_maintenance("F1", "2026-11-09", 3)
        start = market.slot_time(done["start"])
        cells = buyer("GET", "/market/calendar?days=10")[1]["fleets"][0]["cells"]
        offset = int((start - market.slot_time(at(1))).total_seconds() // 3600)
        states = {r["reservation"]: r["state"] for r in buyer("GET", "/market/state")[1]["account"]["reservations"]}
        if len(done["credited"]) != 1 or states.get(done["credited"][0]) != "credited" or balance(buyer) != before + Decimal("1.25") or cells[offset:offset + 3] != [["maintenance", None]] * 3:
            failures.append(f"maintenance does not take the least-reserved run, credit the reservation it touches and show on the calendar: {done} {balance(buyer) - before} {cells[offset:offset + 3]}")
        try:
            app.market.schedule_maintenance("F1", "2026-11-09", 1)
            failures.append("maintenance goes beyond the weekly allowance")
        except market.Refusal:
            pass
        later = app.market.schedule_maintenance("F1", "2026-11-16", 3)
        if market.window_of(market.slot_time(later["start"]), cfg.pricing()["windows"]) == market.window_of(start, cfg.pricing()["windows"]):
            failures.append(f"maintenance does not rotate between windows: {done['start']} then {later['start']}")
        tp.set_plans(app, [plan(1, "holder@example.com", ["americas", "asia", "europe"], lock=["F2"])])
        locked = app.market.schedule_maintenance("F2", "2026-11-09", 3)
        granted = holder("GET", "/market/state")[1]["account"]["tranche"]
        if locked["credited"] != ["SP-COVER-1"] * 3 or not granted or granted[0]["granted"] != 3:
            failures.append(f"maintenance on a locked fleet does not credit the plan's hours: {locked} {granted}")
    finally:
        server.shutdown()


def live_flows(failures, work):
    (work / "secret").write_text("test-client-secret\n")
    fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), tp.FakeCoinPayments)
    fake.invoices, fake.down = {}, False
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    mailbox = socketserver.ThreadingTCPServer(("127.0.0.1", 0), tp.FakeSMTP)
    mailbox.daemon_threads, mailbox.mail = True, []
    threading.Thread(target=mailbox.serve_forever, daemon=True).start()
    cfg = sp.Config({
        "SPARKPIPE_PAY_DATA": str(work / "live"), "SPARKPIPE_PAY_SITE": str(work / "site"),
        "COINPAYMENTS_API": f"http://127.0.0.1:{fake.server_address[1]}/api/v2",
        "COINPAYMENTS_CLIENT_ID": tp.CLIENT, "COINPAYMENTS_CLIENT_SECRET_FILE": str(work / "secret"),
        "SPARKPIPE_PAY_SMTP": f"127.0.0.1:{mailbox.server_address[1]}",
    })
    server = sp.start(cfg, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, app = server.server_address[1], server.app
    buyer = tp.Client(port, origin="https://sparkpipe.ai")
    try:
        if buyer("POST", "/code", {"email": "live@example.com"})[1] != {"sent": True}:
            failures.append("the live service shows the sign-in code")
        buyer("POST", "/verify", {"email": "live@example.com", "code": tp.last_code(mailbox, "live@example.com"), "company": "Live Co", "accept": True})
        status, placed = buyer("POST", "/market/buy", {"amount": "50", "coin": "USDC.BASE"})
        invoice = fake.invoices.get("inv1", {}).get("body", {})
        if status != 200 or invoice.get("amount", {}).get("total") != "55.00" or "2.500000 SparkPipe hour-credits" not in invoice.get("items", [{}])[0].get("name", ""):
            failures.append(f"HC is invoiced through CoinPayments: {status} {placed} {invoice.get('items')}")
        prices = work / "site" / "pricing.js"
        text = prices.read_text()
        prices.write_text(text.replace('"advanceMultiplier": 1.25', '"advanceMultiplier": 3'))
        fake.invoices["inv1"]["status"] = "completed"
        tp.webhook(buyer, cfg, "inv1")
        app.jobs.join()
        app.mails.join()
        prices.write_text(text)
        if app.book.market.lots[app.book.emails["live@example.com"]][0]["terms"]["advanceMultiplier"] != "1.25":
            failures.append("HC paid for after a settings change does not keep the settings in force when it was ordered")
        if balance(buyer) != Decimal("2.5"):
            failures.append(f"a completed HC invoice issues its HC: {balance(buyer)}")
        buyer("POST", "/market/buy", {"amount": "40", "coin": "USDC.BASE"})
        fake.invoices["inv2"].update(status="completed", received="22.00")
        tp.webhook(buyer, cfg, "inv2")
        app.jobs.join()
        app.mails.join()
        if balance(buyer) != Decimal("3.5"):
            failures.append(f"a part payment buys HC in proportion: {balance(buyer)}")
        receipt = [m for rcpt, m in mailbox.mail if "live@example.com" in rcpt and "Payment received" in m["Subject"]]
        if not receipt or "hour-credits, usable until 2027-11-02" not in receipt[0].get_content():
            failures.append("the HC receipt names hour-credits and the date they lapse")
        if buyer("POST", "/market/buy", {"amount": "100000", "coin": "USDC.BASE"})[0] != 200:
            failures.append("a large HC order within the cover is refused")
        status, refused = buyer("POST", "/market/buy", {"amount": "100000", "coin": "USDC.BASE"})
        sale = Decimal(buyer("GET", "/market/state")[1]["onSale"])
        if status != 409 or refused.get("reason") != "cover" or not 0 < sale < 5000:
            failures.append(f"HC beyond what Fleet 1's free hours can deliver, counting unpaid orders, is sold: {status} {refused} {sale}")
        cells = buyer("GET", "/market/calendar")[1]["fleets"]
        if cells[0]["cells"][0] != ["closed", None] or cells[1]["cells"][0] != ["closed", None]:
            failures.append("the live calendar sells hours before 1 December or on a fleet still being built")
        key = buyer("POST", "/market/keys", {})[1]["key"]
        if pool_call(port, key, {"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]})[0] != 503:
            failures.append("the live pool answers before it is connected to the fleets")
        for path in ("/market/sandbox/clock", "/market/sandbox/reset"):
            if buyer("POST", path, {"hours": 1})[0] != 404:
                failures.append(f"{path} exists outside the sandbox")
    finally:
        server.shutdown()
        fake.shutdown()
        mailbox.shutdown()


def main():
    failures = []
    check_units(failures)
    work = Path(tempfile.mkdtemp(prefix="sparkpipe-market-"))
    try:
        site(work)
        sp.clock = lambda: START
        sandbox_flows(failures, work)
        cover_flows(failures, work)
        locked_terms(failures, work)
        maintenance_flows(failures, work)
        live_flows(failures, work)
    finally:
        shutil.rmtree(work)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("market: HC bought by amount and lapsing after 12 months, sold only while free hours after plans and maintenance cover it, later lapse dates instead of overselling, plan set-asides by window, day choice and fleet lock that open 6 hours before except locked fleets, earlier-lapsing HC first, advance hours at 1.25 up to 31 days, settings locked per HC lot, reservation and Plan Order, maintenance with 7 days' notice inside the weekly allowance, rotated between windows and credited back, reservations at Rate, releases sold before our hours and credited less the fee, one-off, resting, standing and consecutive bids, Tranche hours, the shared pool's metering and its hourly settlement to released hours, API keys, the ledger export, feeds, the sandbox clock and reset, and live HC invoices behave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
