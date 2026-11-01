import base64
import calendar
import collections
import hashlib
import json
import math
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal, InvalidOperation

HC_STEP = Decimal("0.000001")
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
WEEK = timedelta(days=7)
KST = timezone(timedelta(hours=9))
STAMP = "%Y-%m-%dT%H:%M:%SZ"
SLOT = "%Y-%m-%dT%H:00Z"
EMPTY = Decimal("0.0001")
KINDS = ("hc_lot", "hc_debit", "hc_hold", "hc_unhold", "accelerator_charge", "booking", "lock", "unlock", "lock_lapse", "round", "api_key", "api_key_revoke", "maintenance", "maintenance_done", "clock")


class Refusal(Exception):
    def __init__(self, text, status=400, reason=None):
        super().__init__(text)
        self.status = status
        self.reason = reason


def hc(value):
    return Decimal(str(value)).quantize(HC_STEP, rounding=ROUND_DOWN)


def text_hc(value):
    return format(hc(value), "f")


def stamp(moment):
    return moment.astimezone(timezone.utc).strftime(STAMP)


def stamp_time(text):
    return datetime.strptime(text, STAMP).replace(tzinfo=timezone.utc)


def slot_text(moment):
    return moment.astimezone(timezone.utc).strftime(SLOT)


def slot_time(text):
    return datetime.strptime(text, SLOT).replace(tzinfo=timezone.utc)


def kst_text(moment):
    return moment.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")


def kst_day(text):
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=KST).astimezone(timezone.utc)


def months_later(moment, months):
    total = moment.month - 1 + months
    year, month = moment.year + total // 12, total % 12 + 1
    return moment.replace(year=year, month=month, day=min(moment.day, calendar.monthrange(year, month)[1]))


def week_start(moment):
    local = moment.astimezone(KST)
    day = local.date() - timedelta(days=local.weekday())
    return datetime(day.year, day.month, day.day, tzinfo=KST).astimezone(timezone.utc)


def sunday_after(moment):
    found = week_start(moment) + 6 * DAY
    return found if found > moment else found + WEEK


def overlap(begins, ends, start, end):
    return max(timedelta(0), min(ends, end) - max(begins, start))


def merged(spans):
    found = []
    for begins, ends in sorted(spans):
        if found and begins <= found[-1][1]:
            found[-1][1] = max(found[-1][1], ends)
        else:
            found.append([begins, ends])
    return [[slot_text(a), slot_text(b)] for a, b in found]


def new_id(prefix):
    return prefix + base64.b32encode(secrets.token_bytes(10)).decode()


def key_digest(key):
    return hashlib.sha256(key.encode()).hexdigest()


def estimate_tokens(messages):
    return max(1, math.ceil(len(json.dumps(messages, ensure_ascii=False).encode()) / 3))


def hc_field(value, label):
    try:
        amount = Decimal(str(value)) if isinstance(value, (int, float, str)) and not isinstance(value, bool) else None
    except InvalidOperation:
        amount = None
    if amount is None or not amount.is_finite() or amount <= 0 or amount >= 10 ** 9 or amount != hc(amount):
        raise Refusal(f"Give the {label} in HC, more than 0 and to at most six decimal places.")
    return amount


def slot_field(value, label):
    try:
        return slot_time(value)
    except (TypeError, ValueError):
        raise Refusal(f"Give the {label} as an hour such as 2026-12-01T01:00Z.")


def count_field(value, label, most):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > most:
        raise Refusal(f"The {label} must be a whole number from 1 to {most}.")
    return value


def accelerator_field(value, limits):
    if value is None or value is False:
        return None
    options = {} if value is True else value
    if not isinstance(options, dict):
        raise Refusal("Give sparkpipe.accelerator as true or as an object with bid, maxWait and fallback.", 400, "invalid_accelerator")
    try:
        bid = Decimal(str(options.get("bid", 1)))
    except InvalidOperation:
        bid = Decimal(-1)
    most = Decimal(str(limits["maxBid"]))
    if isinstance(options.get("bid", 1), bool) or not bid.is_finite() or not 1 <= bid <= most or bid != bid.quantize(Decimal("0.01")):
        raise Refusal(f"The accelerator bid must be from 1 to {most}, in steps of 0.01.", 400, "invalid_accelerator")
    wait = options.get("maxWait", limits["defaultWait"])
    if isinstance(wait, bool) or not isinstance(wait, int) or not 0 <= wait <= limits["maxWait"]:
        raise Refusal(f"The accelerator maxWait must be a whole number of seconds from 0 to {limits['maxWait']}.", 400, "invalid_accelerator")
    fallback = options.get("fallback", True)
    if not isinstance(fallback, bool):
        raise Refusal("Give the accelerator fallback as true or false.", 400, "invalid_accelerator")
    return {"bid": bid, "maxWait": wait, "fallback": fallback}


def simulate(model, fleet, messages, limit):
    words = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    tokens_in = max(1, len(words) // 4)
    text = (f"Sandbox reply from SparkPipe {fleet} ({model}). Your messages held about {tokens_in} tokens. "
            "On the live service this request runs on the whole Fleet you hold for your booked Hour or Week, "
            "with your working state kept on it from one booking to the next.")
    words_out = text.split()[:max(1, limit)]
    return " ".join(words_out), tokens_in, len(words_out)


def spend(parts, amount):
    left = Decimal(amount)
    while left > 0 and parts:
        used = min(parts[0][1], left)
        left -= used
        if used == parts[0][1]:
            parts.pop(0)
        else:
            parts[0][1] -= used


def give_back(parts):
    for lot, part in parts:
        lot["hc"] += part
    parts.clear()


class State:
    def __init__(self):
        self.lots = collections.defaultdict(list)
        self.locks = {}
        self.holds = {}
        self.hold_owner = {}
        self.bookings = {}
        self.keys = {}
        self.key_hashes = {}
        self.charges = collections.defaultdict(lambda: collections.deque(maxlen=50))
        self.refunds = collections.defaultdict(list)
        self.maintenance = {}
        self.round = None
        self.offset = 0
        self.seq = 0

    def apply(self, line):
        self.seq = line["seq"]
        getattr(self, "on_" + line["type"])(line["data"])

    def take(self, account, amount, now):
        taken, left = [], Decimal(amount)
        for lot in self.lots[account]:
            if left <= 0:
                break
            if lot["expires"] > now and lot["hc"] > 0:
                part = min(lot["hc"], left)
                lot["hc"] -= part
                left -= part
                taken.append([lot, part])
        return taken

    def balance(self, account, now):
        return sum((lot["hc"] for lot in self.lots[account] if lot["expires"] > now), Decimal(0))

    def locked(self, account):
        return sum((part for _, part in self.locks.get(account, {}).get("parts", [])), Decimal(0))

    def held(self, job):
        return sum((part for _, part in self.holds.get(job, [])), Decimal(0))

    def on_hc_lot(self, data):
        self.lots[data["account"]].append({"lot": data["lot"], "hc": Decimal(data["hc"]), "first": data["hc"], "expires": data["expires"], "source": data["source"], "ref": data["ref"], "issued": data["now"]})
        self.lots[data["account"]].sort(key=lambda lot: (lot["expires"], lot["issued"]))

    def on_hc_debit(self, data):
        if data.get("lot"):
            next(lot for lot in self.lots[data["account"]] if lot["lot"] == data["lot"])["hc"] -= Decimal(data["hc"])
        else:
            self.take(data["account"], data["hc"], data["now"])

    def on_hc_hold(self, data):
        self.hold_owner[data["job"]] = data["account"]
        self.holds.setdefault(data["job"], []).extend(self.take(data["account"], data["hc"], data["now"]))

    def on_hc_unhold(self, data):
        give_back(self.holds.get(data["job"], []))

    def on_accelerator_charge(self, data):
        amount, held = Decimal(data["amount"]), self.held(data["job"])
        spend(self.holds.get(data["job"], []), min(amount, held))
        self.take(data["account"], amount - min(amount, held), data["now"])
        give_back(self.holds.get(data["job"], []))
        self.charges[data["account"]].appendleft({key: data[key] for key in ("charge", "key", "model", "fleet", "accelerator", "rate", "bid", "seconds", "waited", "tokens", "amount", "now")})

    def on_booking(self, data):
        self.bookings[data["booking"]] = dict(data, begins=slot_time(data["start"]), ends=slot_time(data["end"]))
        if data["source"] == "waitlist":
            parts = self.locks.pop(data["account"])["parts"]
            spend(parts, data["hc"])
            give_back(parts)
        else:
            self.take(data["account"], data["hc"], data["now"])

    def on_lock(self, data):
        lock = self.locks.setdefault(data["account"], {"parts": [], "joined": data["now"], "seq": self.seq})
        lock["parts"].extend(self.take(data["account"], data["hc"], data["now"]))

    def on_unlock(self, data):
        give_back(self.locks.pop(data["account"])["parts"])

    def on_lock_lapse(self, data):
        lock = self.locks[data["account"]]
        lock["parts"] = [[lot, part] for lot, part in lock["parts"] if lot["lot"] != data["lot"]]
        if not lock["parts"]:
            del self.locks[data["account"]]
        self.refunds[data["account"]].append({key: data[key] for key in ("lot", "hc", "ref", "refund", "now")})

    def on_round(self, data):
        self.round = data["sunday"]

    def on_api_key(self, data):
        self.keys[data["key"]] = {"key": data["key"], "account": data["account"], "note": data["note"], "hint": data["hint"], "created": data["now"], "revoked": False}
        self.key_hashes[data["hash"]] = data["key"]

    def on_api_key_revoke(self, data):
        self.keys[data["key"]]["revoked"] = True

    def on_maintenance(self, data):
        self.maintenance[data["maintenance"]] = dict(data, begins=slot_time(data["start"]), ends=slot_time(data["start"]) + HOUR * data["hours"], done=False)

    def on_maintenance_done(self, data):
        self.maintenance[data["maintenance"]]["done"] = True

    def on_clock(self, data):
        self.offset = data["offset"]


class Market:
    def __init__(self, app, sandbox, clock):
        self.app = app
        self.sandbox = sandbox
        self.clock = clock
        self.queue = threading.Condition()
        self.waiting = []
        self.busy_accelerators = {}
        self.joined = 0

    @property
    def book(self):
        return self.app.book

    @property
    def st(self):
        return self.app.book.market

    def settings(self):
        return self.app.cfg.pricing()

    def now(self):
        return self.clock() + timedelta(seconds=self.st.offset)

    def fleets(self, pricing):
        return [dict(f, rate=Decimal(str(f["rate"])), serving=self.sandbox or f["state"] == "serving") for f in pricing["fleets"]]

    def fleet_field(self, pricing, value):
        found = next((f for f in self.fleets(pricing) if f["id"] == value), None)
        if not found:
            raise Refusal("Choose one of the Fleets: " + ", ".join(f["id"] for f in pricing["fleets"]) + ".")
        return found

    def accelerators(self, pricing):
        listed = pricing.get("accelerators", []) + (self.app.cfg.sandbox_accelerators if self.sandbox else [])
        found = []
        for a in listed:
            speed = Decimal(str(a["tokensPerSecond"])) if a.get("tokensPerSecond") else None
            found.append({"id": a["id"], "name": a["name"], "kind": a["kind"], "rate": Decimal(str(a["rate"])), "speed": speed, "serving": bool(speed) and (self.sandbox or a["state"] == "serving")})
        return found

    def gate(self, pricing, key):
        return None if self.sandbox else kst_day(pricing[key])

    def on_fleet(self, fleet_id):
        return [b for b in self.st.bookings.values() if b["fleet"] == fleet_id]

    def busy(self, fleet_id, start, end):
        return sorted((b for b in self.on_fleet(fleet_id) if b["begins"] < end and b["ends"] > start), key=lambda b: b["begins"])

    def maintained(self, fleet_id, start, end):
        return [m for m in self.st.maintenance.values() if m["fleet"] == fleet_id and m["begins"] < end and m["ends"] > start]

    def weeks_left(self, account, fleet_id, moment):
        current = week_start(moment)
        mine = [b for b in self.on_fleet(fleet_id) if b["account"] == account and b["kind"] == "week" and b["ends"] > moment]
        return sum((b["ends"] - max(b["begins"], current)) // WEEK for b in mine), max((b["ends"] for b in mine), default=None)

    def discount(self, rules, count):
        return max((Decimal(str(d["off"])) for d in rules["discounts"] if d["weeks"] <= count), default=Decimal(0))

    def week_price(self, rules, fleet, weeks, off):
        return hc(Decimal(str(rules["weekHC"])) * fleet["rate"] * weeks * (1 - off))

    def check_demo(self, pricing, account, fleet, start, end, now):
        rules = pricing["bookings"]
        gate = self.gate(pricing, "demoFrom")
        if gate and start < gate:
            raise Refusal(f"The Demo Fleet opens on {pricing['demoFrom']}.", 409, "not_open")
        if end > now + DAY * rules["demoDaysAhead"]:
            raise Refusal(f"Demo Fleet hours can be booked at most {rules['demoDaysAhead']} days ahead.", 409, "too_far")
        week = week_start(start)
        while week < end:
            used = sum((overlap(b["begins"], b["ends"], week, week + WEEK) for b in self.on_fleet(fleet["id"]) if b["account"] == account), timedelta(0))
            if used + overlap(start, end, week, week + WEEK) > HOUR * rules["demoHoursPerWeek"]:
                raise Refusal(f"An account may hold at most {rules['demoHoursPerWeek']} Demo Fleet hours in a Week; it holds {used // HOUR} in the Week of {kst_text(week)[:10]}.", 409, "demo_cap")
            week += WEEK

    def hour_price(self, pricing, account, fleet, start, end, now):
        if not fleet["serving"]:
            raise Refusal(f"{fleet['name']} is not serving.", 409, "not_serving")
        if start <= now:
            raise Refusal("Choose an hour that has not started.", 409, "started")
        if fleet["demo"]:
            self.check_demo(pricing, account, fleet, start, end, now)
        elif end > week_start(now) + WEEK:
            raise Refusal(f"Hours on {fleet['name']} can be booked only to the end of the current Week, {kst_text(week_start(now) + WEEK)}.", 409, "too_far")
        taken = self.busy(fleet["id"], start, end)
        if any(b["kind"] == "week" for b in taken):
            raise Refusal(f"{fleet['name']} is pinned to a customer this Week.", 409, "pinned")
        if taken:
            raise Refusal(f"{fleet['name']} is already booked at {kst_text(max(taken[0]['begins'], start))}.", 409, "taken")
        if self.maintained(fleet["id"], start, end):
            raise Refusal(f"{fleet['name']} is under maintenance in those hours.", 409, "maintenance")
        return hc(Decimal((end - start) // HOUR) * Decimal(str(pricing["bookings"]["hourHC"])) * fleet["rate"])

    def check_first(self, pricing, fleet, first, now):
        gate = self.gate(pricing, "weeksFrom")
        if gate and first < gate:
            raise Refusal(f"Weeks can be booked from the Week of {pricing['weeksFrom']}.", 409, "not_open")
        held = [b for b in self.busy(fleet["id"], first - WEEK, first) if b["kind"] == "week"]
        if held and now < first - DAY:
            raise Refusal(f"{fleet['name']} is pinned to a customer, who may book its next Week until {kst_text(first - DAY)}.", 409, "pinned")

    def week_plan(self, pricing, account, fleet, weeks, now):
        rules = pricing["bookings"]
        if fleet["demo"] or not fleet["serving"]:
            raise Refusal(f"{fleet['name']} is not booked by the Week: choose a serving Fleet other than the Demo Fleet.", 409, "not_weekly")
        current = week_start(now)
        count, through = self.weeks_left(account, fleet["id"], now)
        first = through or current + WEEK
        if not through:
            self.check_first(pricing, fleet, first, now)
        end = first + WEEK * weeks
        if end - WEEK > current + WEEK * rules["maxWeeksAhead"]:
            raise Refusal(f"The last Week may start at most {rules['maxWeeksAhead']} Weeks after the current Week.", 409, "too_far")
        taken = self.busy(fleet["id"], first, end)
        if taken:
            raise Refusal(f"{fleet['name']} is already booked for the Week of {kst_text(max(taken[0]['begins'], first))[:10]}.", 409, "taken")
        off = self.discount(rules, count + weeks)
        return {"fleet": fleet["id"], "weeks": weeks, "first": slot_text(first), "last": slot_text(end - WEEK), "end": slot_text(end),
                "count": count + weeks, "discount": format(off, "f"), "price": text_hc(self.week_price(rules, fleet, weeks, off)), "extends": bool(through)}

    def pay(self, account, fleet, kind, start, end, price, now, extra):
        if self.st.balance(account, stamp(now)) < price:
            raise Refusal(f"This booking costs {text_hc(price)} HC, more than the account's unlocked HC. Buy HC first.", 402, "insufficient_hc")
        data = dict(extra, booking=new_id("BK"), account=account, fleet=fleet["id"], kind=kind, start=slot_text(start), end=slot_text(end), hc=text_hc(price), rate=text_hc(fleet["rate"]), source="hc", now=stamp(now))
        self.book.append("booking", data)
        return data["booking"]

    def book_hours(self, account, body):
        pricing = self.settings()
        fleet = self.fleet_field(pricing, body.get("fleet"))
        start = slot_field(body.get("start"), "start")
        hours = count_field(body.get("hours"), "number of hours", 168)
        with self.book.lock:
            now = self.now()
            self.advance(now)
            price = self.hour_price(pricing, account, fleet, start, start + HOUR * hours, now)
            ident = self.pay(account, fleet, "hour", start, start + HOUR * hours, price, now, {"hours": hours})
            return self.booking_view(self.st.bookings[ident], now)

    def weeks_request(self, body):
        pricing = self.settings()
        return pricing, self.fleet_field(pricing, body.get("fleet")), count_field(body.get("weeks"), "number of Weeks", pricing["bookings"]["maxWeeksAhead"])

    def quote_weeks(self, account, body):
        pricing, fleet, weeks = self.weeks_request(body)
        with self.book.lock:
            now = self.now()
            self.advance(now)
            return self.week_plan(pricing, account, fleet, weeks, now)

    def book_weeks(self, account, body):
        pricing, fleet, weeks = self.weeks_request(body)
        with self.book.lock:
            now = self.now()
            self.advance(now)
            plan = self.week_plan(pricing, account, fleet, weeks, now)
            ident = self.pay(account, fleet, "week", slot_time(plan["first"]), slot_time(plan["end"]), Decimal(plan["price"]), now, {"weeks": weeks, "discount": plan["discount"]})
            return dict(plan, booking=ident)

    def queue_order(self):
        return sorted(self.st.locks, key=lambda account: (-self.st.locked(account), self.st.locks[account]["seq"]))

    def position(self, account):
        order = self.queue_order()
        return order.index(account) + 1 if account in order else None

    def join(self, account, body):
        rules = self.settings()["bookings"]
        amount = hc_field(body.get("lock"), "lock")
        with self.book.lock:
            now = self.now()
            self.advance(now)
            if account not in self.st.locks and amount < rules["weekHC"]:
                raise Refusal(f"Join the waitlist by locking at least {rules['weekHC']} HC.", 409, "lock_too_small")
            if self.st.balance(account, stamp(now)) < amount:
                raise Refusal(f"The account has less than {text_hc(amount)} unlocked HC. Buy HC first.", 402, "insufficient_hc")
            self.book.append("lock", {"account": account, "hc": text_hc(amount), "now": stamp(now)})
            return {"locked": text_hc(self.st.locked(account)), "position": self.position(account)}

    def leave(self, account):
        with self.book.lock:
            self.advance(self.now())
            if account not in self.st.locks:
                raise Refusal("This account is not on the waitlist.", 404)
            self.book.append("unlock", {"account": account, "hc": text_hc(self.st.locked(account)), "reason": "leave", "now": stamp(self.now())})
        return {"locked": text_hc(0), "position": None}

    def lapse_at(self, now):
        moment = stamp(now)
        return min((lot["expires"] for lock in self.st.locks.values() for lot, _ in lock["parts"] if lot["expires"] <= moment), default=None)

    def refund_due(self, lot, part):
        order = self.book.orders.get(lot["ref"]) if lot["source"] == "purchase" else None
        if not order:
            return None
        unit = Decimal(1) if order.get("currency") == "KRW" else Decimal("0.01")
        amount = (part * Decimal(order["total"]) / Decimal(order["hc"])).quantize(unit, rounding=ROUND_DOWN)
        return {"order": order["order"], "processor": order.get("processor", "coinpayments"), "currency": order.get("currency", "USD"), "amount": format(amount, "f")}

    def lapse(self, moment, pricing):
        notes = []
        for account in list(self.st.locks):
            lapsed = {}
            for lot, part in self.st.locks[account]["parts"]:
                if lot["expires"] <= moment:
                    lapsed[lot["lot"]] = (lot, lapsed.get(lot["lot"], (lot, Decimal(0)))[1] + part)
            for lot, part in lapsed.values():
                due = self.refund_due(lot, part)
                self.book.append("lock_lapse", {"account": account, "lot": lot["lot"], "hc": text_hc(part), "ref": lot["ref"], "refund": due, "now": moment})
                notes.append(f"{self.app.who(account)}: {text_hc(part)} locked HC of lot {lot['lot']} lapsed on {moment}; " + (f"refund {due['amount']} {due['currency']} of order {due['order']} through {due['processor']} under clause 16.4" if due else "nothing was paid for it"))
            if account in self.st.locks and self.st.locked(account) < pricing["bookings"]["weekHC"]:
                self.book.append("unlock", {"account": account, "hc": text_hc(self.st.locked(account)), "reason": "below", "now": moment})
        if notes:
            self.app.mail(self.app.cfg.notify, "Refund due: locked HC lapsed", "\n".join(notes + ["", "Toss Payments: sparkpipe_pay.py refund ORDER 16.4 AMOUNT REASON. CoinPayments: refund it in CoinPayments."]))

    def round_at(self, now):
        if not self.st.locks:
            return None
        since = min(stamp_time(lock["joined"]) for lock in self.st.locks.values())
        if self.st.round:
            since = max(since, stamp_time(self.st.round))
        found = sunday_after(since)
        return found if found <= now else None

    def affordable(self, rules, account, fleet, sunday):
        lock = self.st.locked(account)
        count, _ = self.weeks_left(account, fleet["id"], sunday)
        for weeks in range(rules["maxWeeksAhead"], 0, -1):
            off = self.discount(rules, count + weeks)
            if self.week_price(rules, fleet, weeks, off) <= lock:
                return weeks, off, self.week_price(rules, fleet, weeks, off)
        return None

    def allocate(self, sunday, pricing):
        monday, waiting = sunday + DAY, self.queue_order()
        for fleet in self.fleets(pricing):
            if fleet["demo"] or not fleet["serving"] or self.busy(fleet["id"], monday, monday + WEEK):
                continue
            for account in waiting:
                found = self.affordable(pricing["bookings"], account, fleet, sunday)
                if found:
                    self.grant(account, fleet, found, sunday)
                    waiting.remove(account)
                    break
        self.book.append("round", {"sunday": stamp(sunday)})

    def grant(self, account, fleet, found, sunday):
        weeks, off, price = found
        monday, left = sunday + DAY, self.st.locked(account) - price
        self.book.append("booking", {"booking": new_id("BK"), "account": account, "fleet": fleet["id"], "kind": "week", "start": slot_text(monday), "end": slot_text(monday + WEEK * weeks),
                                     "hc": text_hc(price), "rate": text_hc(fleet["rate"]), "weeks": weeks, "discount": format(off, "f"), "source": "waitlist", "unlocked": text_hc(left), "now": stamp(sunday)})
        if account in self.book.accounts:
            self.app.mail(self.book.accounts[account]["email"], f"SparkPipe {fleet['name']} is yours from {kst_text(monday)[:10]}", "\n".join([
                f"The waitlist allocated {fleet['name']} to you: {weeks} Weeks from {kst_text(monday)} to {kst_text(monday + WEEK * weeks)}, for {text_hc(price)} HC of your lock" + (f" at {off * 100:.0f}% off." if off else "."),
                f"{text_hc(left)} HC of your lock is unlocked and back in your balance." if left > 0 else "Your whole lock paid for those Weeks.",
                f"To keep the Fleet, book its next Week by {kst_text(monday + WEEK * weeks - DAY)}: {self.app.cfg.public}/market.html",
            ]))

    def maintenance_at(self, now):
        return min(((m["ends"], m["maintenance"]) for m in self.st.maintenance.values() if not m["done"] and m["ends"] <= now), default=None)

    def credit_maintenance(self, ident):
        found = self.st.maintenance[ident]
        owed = collections.defaultdict(Decimal)
        for b in self.busy(found["fleet"], found["begins"], found["ends"]):
            owed[b["account"]] += Decimal(b["rate"]) * (overlap(b["begins"], b["ends"], found["begins"], found["ends"]) // HOUR)
        for account, credits in sorted(owed.items()):
            self.lot(account, credits, "maintenance", ident, found["ends"])
        self.book.append("maintenance_done", {"maintenance": ident, "credited": [[account, text_hc(credits)] for account, credits in sorted(owed.items())]})

    def advance(self, now):
        pricing = self.settings()
        while True:
            lapse, sunday, done = self.lapse_at(now), self.round_at(now), self.maintenance_at(now)
            events = []
            if lapse:
                events.append((stamp_time(lapse), 0, lambda: self.lapse(lapse, pricing)))
            if sunday:
                events.append((sunday, 1, lambda: self.allocate(sunday, pricing)))
            if done:
                events.append((done[0], 2, lambda: self.credit_maintenance(done[1])))
            if not events:
                return
            min(events, key=lambda event: event[:2])[2]()

    def tick(self):
        with self.book.lock:
            self.advance(self.now())

    def run(self, log):
        self.release_orphans()
        while True:
            time.sleep(30)
            try:
                self.tick()
            except Exception as error:
                log(f"market tick: {error}")

    def lot(self, account, credits, source, ref, now, note=None):
        data = {"lot": new_id("LT"), "account": account, "hc": text_hc(credits), "expires": stamp(months_later(now, self.settings()["hc"]["validityMonths"])), "source": source, "ref": ref, "now": stamp(now)}
        if note:
            data["note"] = note
        self.book.append("hc_lot", data)
        return data

    def buy(self, account, credits):
        with self.book.lock:
            data = self.lot(account, credits, "purchase", "sandbox", self.now())
        return {key: data[key] for key in ("lot", "hc", "expires")}

    def issue(self, account, credits, order):
        with self.book.lock:
            return self.lot(account, credits, "purchase", order, self.now())

    def credit(self, account, credits, note):
        with self.book.lock:
            return self.lot(account, credits, "credit", new_id("CR"), self.now(), note)

    def create_key(self, account, note):
        key = "sk-hc-" + secrets.token_urlsafe(24)
        found = new_id("KY")
        with self.book.lock:
            self.book.append("api_key", {"key": found, "account": account, "hash": key_digest(key), "note": note, "hint": key[-4:], "now": stamp(self.now())})
        return {"id": found, "key": key}

    def revoke_key(self, account, found):
        with self.book.lock:
            record = self.st.keys.get(found)
            if not record or record["account"] != account or record["revoked"]:
                raise Refusal("No active key with that id on this account.", 404)
            self.book.append("api_key_revoke", {"key": found})
        return {"id": found, "revoked": True}

    def key_account(self, key):
        found = self.st.key_hashes.get(key_digest(key)) if key else None
        record = self.st.keys.get(found) if found else None
        if not record or record["revoked"]:
            raise Refusal("Send a valid SparkPipe API key as Authorization: Bearer sk-hc-...", 401, "invalid_api_key")
        return record

    def held(self, account, now):
        serving = {f["id"] for f in self.fleets(self.settings()) if f["serving"]}
        return sorted({b["fleet"] for b in self.st.bookings.values() if b["account"] == account and b["begins"] <= now < b["ends"] and b["fleet"] in serving})

    def gateway(self, key, requested, choose=True):
        record = self.key_account(key)
        with self.book.lock:
            now = self.now()
            self.advance(now)
            held = self.held(record["account"], now)
        if not held:
            raise Refusal("This account holds no Fleet now. Book an Hour or a Week to send requests.", 403, "no_fleet")
        if not choose:
            return record, held[0]
        if requested is not None and requested not in held:
            raise Refusal(f"This account does not hold {requested} now; it holds {', '.join(held)}.", 403, "fleet_not_held")
        if requested is None and len(held) > 1:
            raise Refusal(f"This account holds {', '.join(held)} now: name one in sparkpipe.fleet.", 400, "fleet_required")
        return record, requested or held[0]

    def schedule_maintenance(self, fleet_id, day, hours=None):
        pricing = self.settings()
        fleet = self.fleet_field(pricing, fleet_id)
        try:
            monday = kst_day(day)
        except (TypeError, ValueError):
            raise Refusal("Give the Monday of the Week as YYYY-MM-DD, in KST.")
        if monday.astimezone(KST).weekday() != 0:
            raise Refusal("Maintenance allowances run by Week, Monday to Sunday: give the Monday of the Week.")
        allowance = pricing["bookings"]["maintenanceHoursPerWeek"]
        hours = allowance if hours is None else hours
        with self.book.lock:
            now = self.now()
            self.advance(now)
            done = sum(m["hours"] for m in self.st.maintenance.values() if m["fleet"] == fleet["id"] and monday <= m["begins"] < monday + WEEK)
            if not fleet["serving"] or hours < 1 or done + hours > allowance:
                raise Refusal(f"{fleet['name']} has {max(0, allowance - done)} of its {allowance} maintenance hours left in that Week, and must be serving.")
            start = self.quiet_run(fleet, monday, hours, now)
            ident = new_id("MT")
            self.book.append("maintenance", {"maintenance": ident, "fleet": fleet["id"], "start": slot_text(start), "hours": hours, "now": stamp(now)})
            for account in sorted({b["account"] for b in self.busy(fleet["id"], start, start + HOUR * hours)}):
                self.app.mail(self.book.accounts[account]["email"], f"SparkPipe maintenance on {fleet['name']}, {kst_text(start)}",
                              f"We will maintain {fleet['name']} for {hours} hours from {kst_text(start)}. Every hour of your booking it overlaps is credited to your account as HC equal to the Fleet's Rate once it ends (clauses 4.3 and 4.4).")
        return {"maintenance": ident, "fleet": fleet["id"], "start": slot_text(start), "hours": hours}

    def quiet_run(self, fleet, monday, hours, now):
        earliest = now + WEEK
        used = collections.Counter(m["begins"].astimezone(KST).hour for m in self.st.maintenance.values() if m["fleet"] == fleet["id"])
        best = None
        for n in range(7 * 24 - hours + 1):
            start, end = monday + HOUR * n, monday + HOUR * (n + hours)
            if start < earliest or self.maintained(fleet["id"], start, end):
                continue
            held = sum((overlap(b["begins"], b["ends"], start, end) for b in self.busy(fleet["id"], start, end)), timedelta(0))
            score = (held, used[start.astimezone(KST).hour], start)
            best = score if best is None or score < best else best
        if best is None:
            raise Refusal(f"No run of {hours} hours on {fleet['name']} that Week starts at least 7 days from now.")
        return best[2]

    def move_clock(self, hours):
        with self.book.lock:
            self.book.append("clock", {"offset": self.st.offset + hours * 3600})
            self.advance(self.now())
        return {"now": stamp(self.now())}

    def accelerator_join(self, ticket, options, tokens):
        pricing = self.settings()
        serving = [a for a in self.accelerators(pricing) if a["serving"]]
        if not serving:
            return self.unaccelerated(options, "none_serving")
        account, bid = ticket["account"], options["bid"]
        estimate = max(max(hc(Decimal(tokens) / a["speed"] * a["rate"] * bid / 3600) for a in serving), EMPTY)
        job = {"job": new_id("AJ"), "account": account, "key": ticket["key"], "model": ticket["model"], "fleet": ticket.get("fleet"), "bid": bid, "tokens": tokens, "estimate": estimate, "accelerator": None}
        with self.book.lock:
            now = self.now()
            if self.st.balance(account, stamp(now)) < estimate:
                raise Refusal("This account's unlocked HC does not cover this prompt's estimated accelerator time.", 402, "insufficient_hc")
            self.book.append("hc_hold", {"job": job["job"], "account": account, "hc": text_hc(estimate), "now": stamp(now)})
        with self.queue:
            self.joined += 1
            job.update(seq=self.joined, joined=time.monotonic())
            self.waiting.append(job)
            self.dispatch(serving)
            deadline = job["joined"] + options["maxWait"]
            while job["accelerator"] is None and time.monotonic() < deadline:
                self.queue.wait(deadline - time.monotonic())
            if job["accelerator"] is None:
                self.waiting.remove(job)
        if job["accelerator"] is None:
            self.unhold(job)
            return self.unaccelerated(options, "waited")
        job["started"] = time.monotonic()
        return job

    def unhold(self, job):
        with self.book.lock:
            if self.st.held(job["job"]) > 0:
                self.book.append("hc_unhold", {"job": job["job"]})

    def release_orphans(self):
        with self.book.lock:
            for job in list(self.st.holds):
                if self.st.held(job) > 0:
                    self.book.append("hc_unhold", {"job": job})

    def unaccelerated(self, options, reason):
        if not options["fallback"]:
            raise Refusal("No prefill accelerator started this prompt in time, and the request asked not to fall back.", 503, "accelerator_busy")
        return {"used": False, "reason": reason}

    def dispatch(self, serving):
        free = sorted((a for a in serving if a["id"] not in self.busy_accelerators), key=lambda a: (-a["speed"], a["id"]))
        while free and self.waiting:
            job = min(self.waiting, key=lambda j: (-j["bid"], j["seq"]))
            self.waiting.remove(job)
            job["accelerator"] = free.pop(0)
            self.busy_accelerators[job["accelerator"]["id"]] = job["job"]
        self.queue.notify_all()

    def accelerator_finish(self, job, seconds):
        pricing = self.settings()
        accel = job["accelerator"]
        with self.queue:
            if self.busy_accelerators.get(accel["id"]) != job["job"]:
                return job.get("result")
            del self.busy_accelerators[accel["id"]]
            self.dispatch([a for a in self.accelerators(pricing) if a["serving"]])
        waited = format(Decimal(str(round(job["started"] - job["joined"], 3))), "f")
        if seconds is None:
            self.unhold(job)
            job["result"] = {"used": False, "reason": "unfinished", "accelerator": accel["id"]}
            return job["result"]
        seconds = Decimal(str(round(seconds, 3)))
        with self.book.lock:
            now = self.now()
            amount = min(hc(seconds * accel["rate"] * job["bid"] / 3600), hc(self.st.held(job["job"]) + self.st.balance(job["account"], stamp(now))))
            data = {"charge": new_id("XC"), "job": job["job"], "account": job["account"], "key": job["key"], "model": job["model"], "fleet": job["fleet"], "accelerator": accel["id"],
                    "rate": format(accel["rate"].normalize(), "f"), "bid": format(job["bid"], "f"), "seconds": format(seconds, "f"), "waited": waited, "tokens": job["tokens"], "amount": text_hc(amount), "now": stamp(now)}
            self.book.append("accelerator_charge", data)
        job["result"] = dict({key: data[key] for key in ("charge", "accelerator", "rate", "bid", "seconds", "waited", "amount")}, used=True)
        return job["result"]

    def accelerator_view(self, pricing):
        listed = self.accelerators(pricing)
        speeds = [a["speed"] for a in listed if a["serving"]]
        with self.queue:
            queue = sorted(self.waiting, key=lambda j: (-j["bid"], j["seq"]))
            busy = set(self.busy_accelerators)
        number = lambda value: None if value is None else format(value.normalize(), "f")
        return {
            "accelerators": [{"id": a["id"], "name": a["name"], "kind": a["kind"], "rate": number(a["rate"]), "tokensPerSecond": number(a["speed"]), "serving": a["serving"], "busy": a["id"] in busy, "hcPerMinute": text_hc(a["rate"] / 60)} for a in listed],
            "queue": [{"bid": format(j["bid"], "f"), "seconds": number(hc(Decimal(j["tokens"]) / max(speeds))) if speeds else None} for j in queue],
        }

    def first_open(self, pricing, fleet, now):
        first = week_start(now) + WEEK
        if fleet["demo"] or not fleet["serving"] or self.busy(fleet["id"], first, first + WEEK):
            return None
        try:
            self.check_first(pricing, fleet, first, now)
        except Refusal:
            return None
        return slot_text(first)

    def status(self, fleet, now):
        if fleet["demo"]:
            return "demo"
        return "pinned" if any(b["kind"] == "week" for b in self.busy(fleet["id"], now, now + timedelta(seconds=1))) else "free"

    def fleet_view(self, pricing, listed, fleet, now):
        return dict(listed, serving=fleet["serving"], status=self.status(fleet, now), firstWeek=self.first_open(pricing, fleet, now),
                    booked=merged((b["begins"], b["ends"]) for b in self.on_fleet(fleet["id"]) if b["ends"] > now),
                    maintenance=merged((m["begins"], m["ends"]) for m in self.maintained(fleet["id"], now, now + 60 * WEEK)))

    def public(self):
        pricing = self.settings()
        with self.book.lock:
            now = self.now()
            self.advance(now)
            fleets = [self.fleet_view(pricing, listed, fleet, now) for listed, fleet in zip(pricing["fleets"], self.fleets(pricing))]
        accelerated = self.accelerator_view(pricing)
        return {
            "now": stamp(now), "sandbox": self.sandbox, "vatRate": pricing["vatRate"], "demoFrom": pricing["demoFrom"], "weeksFrom": pricing["weeksFrom"],
            "hc": pricing["hc"], "bookings": pricing["bookings"], "fleets": fleets, "models": pricing["models"],
            "accelerators": accelerated["accelerators"], "acceleratorQueue": accelerated["queue"], "acceleratorLimits": pricing["accelerator"],
        }

    def booking_view(self, b, now):
        state = "ended" if b["ends"] <= now else "live" if b["begins"] <= now else "upcoming"
        shown = {key: b.get(key) for key in ("booking", "fleet", "kind", "start", "end", "hours", "weeks", "hc", "discount", "source", "unlocked")}
        return dict(shown, created=b["now"], state=state)

    def pinned_view(self, account, pricing, now):
        found = []
        for fleet in self.fleets(pricing):
            count, through = self.weeks_left(account, fleet["id"], now)
            if through:
                room = through - WEEK < week_start(now) + WEEK * pricing["bookings"]["maxWeeksAhead"] and not self.busy(fleet["id"], through, through + WEEK)
                found.append({"fleet": fleet["id"], "weeks": count, "through": slot_text(through), "deadline": slot_text(through - DAY), "canExtend": room})
        return found

    def view(self, account):
        pricing = self.settings()
        with self.book.lock:
            now = self.now()
            self.advance(now)
            st, moment = self.st, stamp(now)
            mine = sorted((b for b in st.bookings.values() if b["account"] == account and b["ends"] > now - WEEK), key=lambda b: (b["begins"], b["fleet"]))
            return {
                "now": moment,
                "balance": text_hc(st.balance(account, moment)),
                "locked": text_hc(st.locked(account)),
                "position": self.position(account),
                "joined": st.locks[account]["joined"] if account in st.locks else None,
                "held": text_hc(sum((st.held(job) for job, owner in st.hold_owner.items() if owner == account), Decimal(0))),
                "lots": [{"lot": l["lot"], "hc": text_hc(l["hc"]), "first": l["first"], "expires": l["expires"], "source": l["source"], "ref": l["ref"], "issued": l["issued"], "expired": l["expires"] <= moment} for l in st.lots[account]],
                "bookings": [self.booking_view(b, now) for b in mine],
                "pinned": self.pinned_view(account, pricing, now),
                "refunds": list(st.refunds[account]),
                "keys": [{k: r[k] for k in ("key", "note", "hint", "created", "revoked")} for r in st.keys.values() if r["account"] == account],
                "charges": list(st.charges[account])[:20],
            }

    def availability_text(self):
        pricing = self.settings()
        with self.book.lock:
            now = self.now()
            self.advance(now)
            rows = [f"{f['id']} {'serving' if f['serving'] else 'not-serving'} {self.status(f, now)}" for f in self.fleets(pricing)]
        return "\n".join([f"# SparkPipe availability, generated {stamp(now)}, Customer Agreement clause 4.2", "# fleet serving status (demo, pinned or free)"] + rows) + "\n"

    def holder_hours(self, fleet_id, start, end):
        if fleet_id not in [f["id"] for f in self.settings()["fleets"]]:
            raise Refusal(f"No Fleet {fleet_id}.", 404, "unknown_fleet")
        if not start < end <= start + WEEK:
            raise Refusal("Give from and to as hours such as 2026-12-01T01:00Z, to after from and at most 7 days later.", 400, "invalid_window")
        with self.book.lock:
            now = self.now()
            self.advance(now)
            closed = self.maintained(fleet_id, start, end)
            hours = []
            for b in self.busy(fleet_id, start, end):
                moment = max(b["begins"], start)
                while moment < min(b["ends"], end):
                    if not any(m["begins"] <= moment < m["ends"] for m in closed):
                        hours.append({"start": slot_text(moment), "account": b["account"]})
                    moment += HOUR
        return {"fleet": fleet_id, "now": stamp(now), "hours": sorted(hours, key=lambda h: h["start"])}
