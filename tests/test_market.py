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
        if cal["from"] != at(1) or len(cells) != 240 or cells[0] != ["hyor", "1.000000"] or cells[167] != ["hyor", "1.000000"] or cells[168] != ["closed", None]:
            failures.append(f"the calendar sells our hours 7 days ahead: {cal['from']} {cells[0]} {cells[167]} {cells[168]}")
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
        fake.invoices["inv1"]["status"] = "completed"
        tp.webhook(buyer, cfg, "inv1")
        app.jobs.join()
        app.mails.join()
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
        if not receipt or "hour-credits, usable for 12 months" not in receipt[0].get_content():
            failures.append("the HC receipt names hour-credits")
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
        live_flows(failures, work)
    finally:
        shutil.rmtree(work)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("market: HC bought by amount and lapsing after 12 months, reservations at Rate, releases sold before our hours and credited less the fee, one-off, resting, standing and consecutive bids, Tranche hours, the shared pool's metering and its hourly settlement to released hours, API keys, the ledger export, feeds, the sandbox clock and reset, and live HC invoices behave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
