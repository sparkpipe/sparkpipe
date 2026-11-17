#!/usr/bin/env python3
import json
import re
import shutil
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import sparkpipe_market as market
import sparkpipe_pay as sp
import test_market as tm
import test_pay as tp


def refusal(call):
    try:
        call()
    except sp.Refusal as error:
        return error.status, getattr(error, "reason", None)
    return None


def flapping(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    upstream = tp.serve(tm.FakeUpstream, seen=[], calls=[], healthy=True)
    server, app, port = tm.live(work, "flap", mailbox, upstream)
    saved, sp.UPTIME_GAP_SECONDS = sp.UPTIME_GAP_SECONDS, 10 ** 9
    try:
        tm.NOW[0] = tm.kst(2027, 1, 8, 12)
        holder = tm.person(app, "flap", 1000)
        app.market.book_weeks(holder, {"fleet": "F3", "weeks": 1})
        tm.probe(app, tm.kst(2027, 1, 12, 10))
        tm.probe(app, tm.kst(2027, 1, 12, 10, 1), (upstream, False))
        tm.probe(app, tm.kst(2027, 1, 12, 10, 10), (upstream, True))
        tm.probe(app, tm.kst(2027, 1, 12, 10, 11), (upstream, False))
        tm.probe(app, tm.kst(2027, 1, 12, 10, 20), (upstream, True))
        lots = tm.downtime_lots(app, holder)
        shown = tm.public_status(port)[1]["F3"]["incidents"]
        if lots != [("1.000000", ["01-12 10:00"])]:
            failures.append(f"two touching 10-minute down periods are not credited as one period over 15 minutes: {lots}")
        if [(i["minutes"], i["creditable"]) for i in shown] != [(10, True), (10, True)]:
            failures.append(f"the status page does not mark periods one good probe apart creditable when together they pass 15 minutes: {shown}")
        tm.probe(app, tm.kst(2027, 1, 12, 10, 21), (upstream, False))
        tm.probe(app, tm.kst(2027, 1, 12, 11, 5), (upstream, True))
        grown = tm.downtime_lots(app, holder)
        if grown != [("1.000000", ["01-12 10:00"]), ("1.000000", ["01-12 11:00"])]:
            failures.append(f"a joined period that grows after its credit is not credited for the hours it adds: {grown}")
    finally:
        sp.UPTIME_GAP_SECONDS = saved
        server.shutdown()


def codes(failures, work):
    mailbox = tp.serve(tp.FakeSMTP, mail=[])
    server, app, port = tm.live(work, "codes", mailbox)
    try:
        victim = "victim@example.com"
        app.send_code({"email": victim}, "198.51.100.1")
        first = tp.last_code(mailbox, victim)
        for n in range(5):
            app.send_code({"email": victim}, "203.0.113.9")
        blocked = refusal(lambda: app.send_code({"email": victim}, "203.0.113.9"))
        own = refusal(lambda: app.send_code({"email": victim}, "198.51.100.1"))
        if blocked != (429, "busy") or own is not None or tp.last_code(mailbox, victim) != first:
            failures.append(f"a third party requesting codes invalidates the customer's code or exhausts the customer's own requests: {blocked} {own}")
        new = refusal(lambda: app.verify({"email": victim, "code": first}, "198.51.100.1", "test"))
        if new != (404, "new"):
            failures.append(f"a new email signing in without a company is not refused with new: {new}")
        terms = refusal(lambda: app.verify({"email": victim, "code": first, "company": "Victim"}, "198.51.100.1", "test"))
        opened = app.verify({"email": victim, "code": first, "company": "Victim", "accept": True}, "198.51.100.1", "test")
        again = refusal(lambda: app.verify({"email": victim, "code": first}, "198.51.100.1", "test"))
        if terms != (409, "terms") or opened[0]["role"] != "owner" or again != (403, None):
            failures.append(f"a refused sign-in burns the code, or a used code works again: {terms} {opened[0]} {again}")
        owner = app.book.emails[victim]
        other = tm.person(app, "other")
        order = tm.order_lot(app, owner, 2, "27200", "59840")
        mine = {"Cookie": f"{sp.COOKIE}={sp.session_token(app.key, owner, 10 ** 10)}"}
        theirs = {"Cookie": f"{sp.COOKIE}={sp.session_token(app.key, other, 10 ** 10)}"}
        anonymous = refusal(lambda: app.status(order, {}))
        foreign = refusal(lambda: app.status(order, theirs))
        if anonymous != (401, None) or foreign != (404, None) or app.status(order, mine)["order"] != order:
            failures.append(f"an order's payment details are shown to someone other than its account's owner: {anonymous} {foreign}")
    finally:
        server.shutdown()


def quantities(failures):
    big = refusal(lambda: sp.quantity_field("999999999.5", Decimal(1)))
    huge = sp.Refusal("")
    try:
        sp.quantity_field("1" + "0" * 9, Decimal(1))
    except sp.Refusal as error:
        huge = error
    if big is not None or "place another order" not in str(huge):
        failures.append(f"a large HC quantity is refused, or the refusal of one over the per-order size says to buy at least 1 HC: {big} {huge}")


def quoted_interpolations(source):
    found, stack, i = [], ["code"], 0
    while i < len(source):
        state, char = stack[-1], source[i]
        if state == "template":
            if char == "\\":
                i += 2
                continue
            if char == "`":
                stack.pop()
            elif source.startswith("${", i):
                stack.append(0)
                i += 2
                continue
            i += 1
            continue
        if source.startswith("//", i):
            i = source.find("\n", i) if "\n" in source[i:] else len(source)
            continue
        if source.startswith("/*", i):
            i = source.index("*/", i) + 2
            continue
        if char in "'\"":
            end = i + 1
            while source[end] != char:
                end += 2 if source[end] == "\\" else 1
            if "${" in source[i:end]:
                found.append(source[i:end + 1])
            i = end + 1
            continue
        if char == "`":
            stack.append("template")
        elif isinstance(state, int) and char == "{":
            stack[-1] += 1
        elif isinstance(state, int) and char == "}":
            if state == 0:
                stack.pop()
            else:
                stack[-1] -= 1
        i += 1
    return found


def acceptance_box(account):
    engine = next(item for item in sp.published(str(ROOT / "site"), "catalogue.js")["items"] if item["acceptance"])
    values = {"me.company": "Acme", "item.name": engine["name"], "fleetName(entry)": "Fleet 1B", "item.license.name": engine["license"]["name"]}
    found = re.search(r"make\('span', ('For ' \+ me\.company \+ [^;]*)\);", account)
    parts = found.group(1).split(" + ") if found else []
    shown = "".join(values.get(part, part[1:-1] if len(part) > 1 and part[0] == part[-1] == "'" else "\0") for part in parts)
    hashes = {path: "0" * 64 for path in engine["license"]["texts"]}
    recorded = sp.install_statement(SimpleNamespace(public="https://sparkpipe.ai"), "Acme", engine, "Fleet 1B", hashes)
    tail = " License texts: "
    return shown, recorded, bool(parts) and shown.endswith(tail) and recorded.startswith(shown[:-len(tail)] + " ")


def pages(failures):
    site = ROOT / "site"
    for name in ("status.html", "status-ko.html"):
        if re.search(r"hub|허브", (site / name).read_text()):
            failures.append(f"{name} describes the fleet's hub")
    script = (site / "availability.js").read_text()
    if re.search(r"7 days|7일|24 hours|24시간|4 TB", script):
        failures.append("availability.js states published settings instead of reading them")
    if quoted_interpolations("const a = 'x ${b} y';") != ["'x ${b} y'"] or quoted_interpolations("const a = `it's ${b ? ` ${c} ` : ''} ok's`;"):
        failures.append("the quoted-interpolation scanner misreads strings and template literals")
    for leftover in quoted_interpolations(script):
        failures.append(f"availability.js has ${{...}} inside a quoted string, which shows it literally: {leftover}")
    for name in ("availability.html", "availability-ko.html"):
        if "availability.js?v=6" not in (site / name).read_text():
            failures.append(f"{name} loads a cached availability.js from before the settings fix")
    account = (site / "account.html").read_text()
    if "12 months" in account or "me.role !== 'owner'" not in account or "not-owner" not in account:
        failures.append("account.html hard-codes the HC validity or invites a non-owner to buy")
    buy = (site / "buy.html").read_text()
    if "String(S.hc)" in buy or "S.me.role !== 'owner'" not in buy:
        failures.append("buy.html sends HC in exponent form or lets a non-owner reach payment")
    for name, label in (("buy.html", 'id="accept"'), ("buy.html", 'id="accept-again"'), ("market.html", 'id="accept"'), ("market.html", 'id="accept-again"')):
        text = (site / name).read_text()
        line = next((line for line in text.splitlines() if label in line), "")
        if 'href="agreement.html"' not in line or 'href="dpa.html"' not in line:
            failures.append(f"{name}: the acceptance box at {label} does not name the Data Processing Agreement next to the Customer Agreement")
    if 'href="dpa.html"' not in account:
        failures.append("account.html does not link the Data Processing Agreement next to the Customer Agreement")
    if 'src="catalogue.js' not in account or "SP.call('/v1/installs')" not in account or "{ item: item.id, accept }, 'PUT')" not in account:
        failures.append("account.html does not set up fleets from the published Catalogue through GET and PUT /v1/installs")
    shown, recorded, same = acceptance_box(account)
    if not same:
        failures.append(f"account.html: the license box a customer ticks does not read as the statement the service records: box {shown!r}, record {recorded!r}")
    if "if (box && !accept)" not in account or "box.type = 'checkbox';" not in account or "box.checked" in account:
        failures.append("account.html lets an item that asks for acceptance be chosen without the company's own tick of its license box")
    checkout = (site / "checkout.js").read_text()
    if "method: method || 'POST'" not in checkout or "typeof data.error === 'object'" not in checkout:
        failures.append("checkout.js cannot send the account API's PUT or read its error objects")
    if "data.complete === false" not in (site / "market.js").read_text() or 'id="ledger-partial"' not in (site / "market.html").read_text():
        failures.append("the ledger panel does not say when an admin's export leaves lines and fields out")
    if "written order" in (site / "privacy.html").read_text() or "서면 주문서에서 제공" in (site / "privacy-ko.html").read_text():
        failures.append("the privacy pages say a USD transfer needs a written order though checkout offers SWIFT")


def main():
    failures = []
    work = Path(tempfile.mkdtemp(prefix="sparkpipe-site-app-"))
    try:
        tm.site(work)
        flapping(failures, work)
        codes(failures, work)
        quantities(failures)
        pages(failures)
    finally:
        shutil.rmtree(work)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print("site app: periods one good probe apart are one creditable period, marked so on the status page, and credited again as they grow; a third party cannot invalidate or exhaust a customer's codes; a refused sign-in keeps the code; order status is the owner's; an order over the per-order size says to place another; pages read published settings, stop non-owners before payment, name the Data Processing Agreement where the Agreement is accepted, set up fleets from the published Catalogue only with the company's acceptance of a license that asks for it, mark a partial ledger export and do not name hubs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
