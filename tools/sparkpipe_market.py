import base64
import bisect
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
PRICE_STEP = Decimal("0.0000000001")
HOUR = timedelta(hours=1)
WEEK = timedelta(days=7)
KST = timezone(timedelta(hours=9))
LONGEST = timedelta(days=3653)
STAMP = "%Y-%m-%dT%H:%M:%SZ"
SLOT = "%Y-%m-%dT%H:00Z"
EMPTY = Decimal("0.0001")
Commitment = collections.namedtuple("Commitment", "account begins ends hours fleets lock allowance plan")
KINDS = ("hc_lot", "hc_debit", "hc_hold", "hc_unhold", "hc_spend_hold", "reserve", "ask", "ask_cancel", "bid", "bid_cancel", "bid_close", "bid_roll", "pool_join", "pool_settle", "pool_charge", "api_key", "api_key_revoke", "tranche_grant", "clock", "maintenance", "reservation_credit", "accelerator_charge")


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


def slot_text(moment):
    return moment.astimezone(timezone.utc).strftime(SLOT)


def slot_time(text):
    return datetime.strptime(text, SLOT).replace(tzinfo=timezone.utc)


def stamp_time(text):
    return datetime.strptime(text, STAMP).replace(tzinfo=timezone.utc)


def kst_text(moment):
    return moment.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")


def text_hours(value):
    return format(Decimal(value).quantize(HC_STEP, rounding=ROUND_DOWN).normalize(), "f")


def kst_month(moment):
    return (moment + timedelta(hours=9)).strftime("%Y-%m")


def estimate_tokens(messages):
    return max(1, math.ceil(len(json.dumps(messages, ensure_ascii=False).encode()) / 3))


def accelerator_field(value, market):
    if value is None or value is False:
        return None
    limits = market["accelerator"]
    options = {} if value is True else value
    if not isinstance(options, dict):
        raise Refusal("Give sparkpipe.accelerator as true or as an object with bid, maxWait, pay and fallback.", 400, "invalid_accelerator")
    try:
        bid = Decimal(str(options.get("bid", 1)))
    except InvalidOperation:
        bid = Decimal(-1)
    most = Decimal(str(limits["maxBid"]))
    if isinstance(options.get("bid", 1), bool) or not bid.is_finite() or bid < 1 or bid > most or bid != bid.quantize(Decimal("0.01")):
        raise Refusal(f"The accelerator bid must be from 1 to {most}, in steps of 0.01.", 400, "invalid_accelerator")
    wait = options.get("maxWait", limits["defaultWait"])
    if isinstance(wait, bool) or not isinstance(wait, int) or wait < 0 or wait > limits["maxWait"]:
        raise Refusal(f"The accelerator maxWait must be a whole number of seconds from 0 to {limits['maxWait']}.", 400, "invalid_accelerator")
    pay = options.get("pay", "hc")
    if pay not in ("hc", "plan"):
        raise Refusal("Pay for accelerator time with hc or plan.", 400, "invalid_accelerator")
    fallback = options.get("fallback", True)
    if not isinstance(fallback, bool):
        raise Refusal("Give the accelerator fallback as true or false.", 400, "invalid_accelerator")
    return {"bid": bid, "maxWait": wait, "pay": pay, "fallback": fallback}


def floor_hour(moment):
    return moment.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def year_later(moment):
    try:
        return moment.replace(year=moment.year + 1)
    except ValueError:
        return moment.replace(year=moment.year + 1, day=28)


def month_of(start):
    return (slot_time(start) + timedelta(hours=9)).strftime("%Y-%m")


def new_id(prefix):
    return prefix + base64.b32encode(secrets.token_bytes(10)).decode()


def key_digest(key):
    return hashlib.sha256(key.encode()).hexdigest()


def rank(offer):
    return (offer["price"], offer["kind"] != "release", offer["seq"], offer["start"], offer["fleet"])


def per_million(usd_per_second, tokens_per_second):
    return format((usd_per_second * Decimal(1000000) / Decimal(str(tokens_per_second))).quantize(Decimal("0.01")), "f")


def hour_key(moment):
    return int(moment.timestamp()) // 3600


def window_of(moment, windows):
    hour = moment.astimezone(KST).hour
    return next((w["id"] for w in windows if w["from"] <= hour < w["to"]), None)


def hc_field(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise Refusal(f"Give the {label} in HC.")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise Refusal(f"Give the {label} in HC.")
    if not amount.is_finite() or amount <= 0 or amount > 1000:
        raise Refusal(f"The {label} must be more than 0 and at most 1000 HC.")
    return hc(amount)


def slot_field(value, label):
    try:
        return slot_time(value)
    except (TypeError, ValueError):
        raise Refusal(f"Give the {label} as an hour such as 2026-12-01T01:00Z.")


def count_field(value, label, most):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > most:
        raise Refusal(f"The {label} must be a whole number from 1 to {most}.")
    return value


def simulate(model, messages, limit):
    words = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    tokens_in = max(1, len(words) // 4)
    text = (f"Sandbox reply from the SparkPipe shared pool ({model}). Your messages held about {tokens_in} tokens. "
            "On the live service this request runs on a SparkPipe fleet and is billed by the seat-seconds it holds, "
            "so a long conversation that stays cached costs little to continue.")
    words_out = text.split()[:max(1, limit)]
    return " ".join(words_out), tokens_in, len(words_out)


class State:
    def __init__(self):
        self.lots = {}
        self.holds = {}
        self.hold_owner = {}
        self.reservations = {}
        self.slots = {}
        self.asks = {}
        self.bids = {}
        self.trades = collections.deque(maxlen=200)
        self.pool_hc = collections.defaultdict(Decimal)
        self.pool_seconds = collections.defaultdict(Decimal)
        self.charges = collections.defaultdict(lambda: collections.deque(maxlen=50))
        self.keys = {}
        self.key_hashes = {}
        self.tranches = collections.defaultdict(int)
        self.tranche_used = collections.defaultdict(Decimal)
        self.settled = set()
        self.maintenance = {}
        self.maintained = []
        self.offset = 0
        self.seq = 0
        self.handlers = dict(zip(KINDS, (
            self.on_lot, self.on_debit, self.on_hold, self.on_unhold, self.on_spend_hold, self.on_reserve,
            self.on_ask, self.on_ask_cancel, self.on_bid, self.on_bid_cancel, self.on_bid_close, self.on_bid_roll,
            self.on_pool_join, self.on_pool_settle, self.on_charge, self.on_key, self.on_key_revoke,
            self.on_tranche, self.on_clock, self.on_maintenance, self.on_reservation_credit, self.on_accelerator)))

    def apply(self, line):
        self.seq = line["seq"]
        self.handlers[line["type"]](line["data"])

    def take(self, account, amount, now):
        taken, left = [], Decimal(amount)
        for lot in self.lots.get(account, []):
            if left <= 0:
                break
            if lot["expires"] > now and lot["hc"] > 0:
                part = min(lot["hc"], left)
                lot["hc"] -= part
                left -= part
                taken.append([lot, part])
        return taken

    def balance(self, account, now):
        return sum((lot["hc"] for lot in self.lots.get(account, []) if lot["expires"] > now), Decimal(0))

    def held(self, bid):
        return sum((part for _, part in self.holds.get(bid, [])), Decimal(0))

    def on_lot(self, data):
        lots = self.lots.setdefault(data["account"], [])
        lots.append({"lot": data["lot"], "hc": Decimal(data["hc"]), "first": data["hc"], "expires": data["expires"], "source": data["source"], "ref": data["ref"], "issued": data["now"]})
        lots.sort(key=lambda lot: (lot["expires"], lot["issued"]))

    def on_debit(self, data):
        if data.get("lot"):
            next(lot for lot in self.lots[data["account"]] if lot["lot"] == data["lot"])["hc"] -= Decimal(data["hc"])
            return
        self.take(data["account"], data["hc"], data["now"])

    def on_hold(self, data):
        self.hold_owner[data["bid"]] = data["account"]
        self.holds.setdefault(data["bid"], []).extend(self.take(data["account"], data["hc"], data["now"]))

    def on_unhold(self, data):
        left, hold = Decimal(data["hc"]), self.holds.get(data["bid"], [])
        while left > 0 and hold:
            lot, part = hold[-1]
            back = min(part, left)
            lot["hc"] += back
            left -= back
            if back == part:
                hold.pop()
            else:
                hold[-1][1] = part - back

    def on_spend_hold(self, data):
        left, hold = Decimal(data["hc"]), self.holds.get(data["bid"], [])
        while left > 0 and hold:
            part = hold[0][1]
            used = min(part, left)
            left -= used
            if used == part:
                hold.pop(0)
            else:
                hold[0][1] = part - used

    def on_reserve(self, data):
        self.reservations[data["reservation"]] = {
            "reservation": data["reservation"], "account": data["account"], "fleet": data["fleet"], "start": data["start"],
            "model": data["model"], "price": data["price"], "source": data["source"], "state": "held", "ask": None, "created": data["now"],
        }
        if data["ask"]:
            ask = self.asks[data["ask"]]
            ask.update(state="sold", price=data["price"])
            self.reservations[ask["reservation"]]["state"] = "sold"
        if data["bid"]:
            bid = self.bids[data["bid"]]
            bid["filled"] += 1
            if bid["filled"] >= bid["hours"]:
                bid["state"] = "filled"
        if data["source"] == "tranche":
            self.tranche_used[(data["account"], month_of(data["start"]))] += 1
        self.slots[(data["fleet"], data["start"])] = data["reservation"]
        self.trades.appendleft({"fleet": data["fleet"], "start": data["start"], "price": data["price"], "kind": data["source"], "at": data["now"]})

    def on_ask(self, data):
        self.asks[data["ask"]] = {"ask": data["ask"], "account": data["account"], "reservation": data["reservation"], "min": data["min"], "fee": Decimal(data["fee"]), "state": "open", "at": data["now"], "seq": self.seq, "price": None, "earned": None}
        self.reservations[data["reservation"]].update(state="released", ask=data["ask"])

    def on_ask_cancel(self, data):
        ask = self.asks[data["ask"]]
        ask["state"] = "cancelled"
        self.reservations[ask["reservation"]].update(state="held", ask=None)

    def on_bid(self, data):
        self.bids[data["bid"]] = dict(data, filled=0, state="open", seq=self.seq)

    def on_bid_cancel(self, data):
        self.bids[data["bid"]]["state"] = "cancelled"

    def on_bid_close(self, data):
        self.bids[data["bid"]]["state"] = data["state"]

    def on_bid_roll(self, data):
        self.bids[data["bid"]].update({"from": data["from"], "to": data["to"], "filled": 0, "state": "open"})

    def on_pool_join(self, data):
        ask = self.asks[data["ask"]]
        ask["state"] = "pooled"
        self.reservations[ask["reservation"]]["state"] = "pooled"

    def on_pool_settle(self, data):
        self.settled.add((data["fleet"], data["start"]))
        self.asks[data["ask"]]["earned"] = data["credit"]

    def on_charge(self, data):
        self.take(data["account"], data["hc"], data["now"])
        for fleet, start, part_hc, part_seconds in data["split"]:
            self.pool_hc[(fleet, start)] += Decimal(part_hc)
            self.pool_seconds[(fleet, start)] += Decimal(part_seconds)
        self.charges[data["account"]].appendleft(dict({key: data[key] for key in ("charge", "key", "model", "seconds", "seats", "hc", "tokensIn", "tokensOut", "now")}, kind="pool"))

    def on_accelerator(self, data):
        if data["pay"] == "plan":
            self.tranche_used[(data["account"], data["month"])] += Decimal(data["amount"])
        else:
            amount, held = Decimal(data["amount"]), self.held(data["job"])
            spent = min(amount, held)
            self.on_spend_hold({"bid": data["job"], "hc": spent})
            self.take(data["account"], amount - spent, data["now"])
            self.on_unhold({"bid": data["job"], "hc": held - spent})
        self.charges[data["account"]].appendleft({
            "kind": "accelerator", "charge": data["charge"], "key": data["key"], "model": data["model"], "seconds": data["seconds"], "seats": None,
            "hc": data["amount"] if data["pay"] == "hc" else "0.000000", "tokensIn": data["tokens"], "tokensOut": 0, "now": data["now"],
            "accelerator": data["accelerator"], "rate": data["rate"], "bid": data["bid"], "pay": data["pay"], "amount": data["amount"], "waited": data["waited"],
        })

    def on_key(self, data):
        self.keys[data["key"]] = {"key": data["key"], "account": data["account"], "note": data["note"], "hint": data["hint"], "created": data["now"], "revoked": False}
        self.key_hashes[data["hash"]] = data["key"]

    def on_key_revoke(self, data):
        self.keys[data["key"]]["revoked"] = True

    def on_tranche(self, data):
        self.tranches[(data["account"], data["month"])] += data["hours"]

    def on_clock(self, data):
        self.offset = data["offset"]

    def on_maintenance(self, data):
        first = slot_time(data["start"])
        for n in range(data["hours"]):
            self.maintenance[(data["fleet"], slot_text(first + HOUR * n))] = data["maintenance"]
        self.maintained.append(data)

    def on_reservation_credit(self, data):
        res = self.reservations[data["reservation"]]
        res["state"] = "credited"
        if res["ask"] and self.asks[res["ask"]]["state"] == "open":
            self.asks[res["ask"]]["state"] = "cancelled"
        if res["source"] == "tranche":
            self.tranche_used[(res["account"], month_of(res["start"]))] -= 1
        if self.slots.get((res["fleet"], res["start"])) == res["reservation"]:
            del self.slots[(res["fleet"], res["start"])]


def take_parts(parts, amount, front):
    used, left = [], Decimal(amount)
    while left > 0 and parts:
        index = 0 if front else -1
        expires, have = parts[index]
        part = min(have, left)
        used.append((expires, part))
        left -= part
        if part == have:
            parts.pop(index)
        else:
            parts[index][1] = have - part
    return used


class Cover:
    def __init__(self, st, now, first, fleets, commitments, maintenance):
        self.first = first
        self.rates = {f["id"]: f["rate"] for f in fleets if f["serving"]}
        self.commitments = commitments
        self.share = 1 - Decimal(str(maintenance)) / 168
        self.taken, self.kept, self.owners = {}, {}, {}
        for (fleet, start), held in st.slots.items():
            if fleet in self.rates and slot_time(start) >= first:
                self.taken.setdefault(start, set()).add(fleet)
                res = st.reservations[held]
                self.owners.setdefault(start, {})[fleet] = res["account"]
                if res["source"] == "tranche":
                    self.kept.setdefault(start, collections.Counter())[res["account"]] += 1
        for fleet, start in st.maintenance:
            if fleet in self.rates and slot_time(start) >= first:
                self.taken.setdefault(start, set()).add(fleet)
        moment, due = stamp(now), collections.defaultdict(Decimal)
        for lots in st.lots.values():
            for lot in lots:
                if lot["expires"] > moment and lot["hc"] > 0:
                    due[lot["expires"]] += lot["hc"]
        for parts in st.holds.values():
            for lot, part in parts:
                if lot["expires"] > moment:
                    due[lot["expires"]] += part
        self.lapses = sorted(due)
        self.moments = [stamp_time(e) for e in self.lapses]
        self.owed, total = [], Decimal(0)
        for expires in self.lapses:
            total += due[expires]
            self.owed.append(total)
        self.curve = [Decimal(0)]
        self.lowest = None

    def active(self, moment):
        key = hour_key(moment)
        return [(c.account, c.fleets, c.lock) for c in self.commitments if c.begins <= moment < c.ends and key in c.hours]

    def locked(self, moment, holder=None):
        return {fleet for account, need, lock in self.active(moment) if lock and (holder is None or account != holder) for fleet in lock}

    def aside(self, moment, holder=None):
        used = self.kept.get(slot_text(moment), {})
        return sum(max(0, need - used.get(account, 0)) for account, need, lock in self.active(moment) if not lock and (holder is None or account != holder))

    def free(self, moment, without=None, holder=None):
        closed = self.taken.get(slot_text(moment), set()) | self.locked(moment, holder)
        return sorted(rate for fleet, rate in self.rates.items() if fleet not in closed and fleet != without)

    def open(self, moment, now, last_call, holder=None, fleet=None):
        if fleet in self.locked(moment, holder):
            return False
        return moment - now < last_call or len(self.free(moment, holder=holder)) > self.aside(moment, holder)

    def clash(self, moment):
        taken, held = self.taken.get(slot_text(moment), set()), collections.Counter()
        for account, need, lock in self.active(moment):
            held.update(lock)
        owners = {fleet: res for fleet, res in self.owners.get(slot_text(moment), {}).items()}
        for account, need, lock in self.active(moment):
            if any(held[fleet] > 1 or fleet not in self.rates or (fleet in taken and owners.get(fleet) != account) for fleet in lock):
                return True
        return len(self.free(moment)) < self.aside(moment)

    def value(self, moment, without=None):
        free = self.free(moment, without)
        keep = len(free) - self.aside(moment)
        return sum(free[:keep], Decimal(0)) * self.share if keep > 0 else Decimal(0)

    def supply(self, until):
        hours = max(0, math.ceil((until - self.first) / HOUR))
        while len(self.curve) <= hours:
            self.curve.append(self.curve[-1] + self.value(self.first + HOUR * (len(self.curve) - 1)))
        return self.curve[hours]

    def due(self, until):
        found = bisect.bisect_right(self.lapses, stamp(until))
        return self.owed[found - 1] if found else Decimal(0)

    def unspent(self):
        return self.owed[-1] if self.owed else Decimal(0)

    def slack(self, until):
        return self.supply(until) - self.due(until)

    def low(self):
        if self.lowest is None and self.moments:
            self.lowest = min(self.slack(moment) for moment in self.moments)
        return self.lowest

    def tightest(self):
        return min(((self.slack(moment), moment) for moment in self.moments), default=None)

    def headroom(self, life):
        return max(Decimal(0), min(self.slack(moment) for moment in [life] + [m for m in self.moments if m > life]))

    def lapse(self, amount, life):
        moment = max([life] + [m for m in self.moments if m > life and self.slack(m) < amount])
        while moment < life + LONGEST and self.slack(moment) < amount:
            moment += HOUR
        return moment

    def blocked(self, hours, spent, now, last_call):
        hours = [(start, fleet) for start, fleet in hours if start - now >= last_call]
        if not hours:
            return None
        losses = [(start, self.value(start) - self.value(start, fleet)) for start, fleet in hours]
        low = self.low()
        if low is None or low >= sum(loss for _, loss in losses):
            return None
        first = min(start for start, _ in hours)
        for until in sorted({start + HOUR for start, _ in hours} | {m for m in self.moments if m > first}):
            before = self.slack(until)
            lost = sum((loss for start, loss in losses if start < until), Decimal(0))
            freed = sum((part for expires, part in spent if expires <= stamp(until)), Decimal(0))
            if before - lost + freed < min(Decimal(0), before):
                return until
        return None


class Market:
    def __init__(self, app, sandbox, clock):
        self.app = app
        self.sandbox = sandbox
        self.clock = clock
        self.lock = threading.Lock()
        self.active = 0
        self.recent = collections.deque()
        self.speed = collections.deque(maxlen=50)
        self.multiplier = Decimal("0.5")
        self.priced = 0.0
        self.covered = None
        self.queue = threading.Condition()
        self.waiting = []
        self.busy = {}
        self.plan_held = collections.defaultdict(Decimal)
        self.joined = 0

    @property
    def book(self):
        return self.app.book

    @property
    def st(self):
        return self.app.book.market

    def settings(self):
        pricing = self.app.cfg.pricing()
        return pricing, pricing["market"]

    def now(self):
        return self.clock() + timedelta(seconds=self.st.offset)

    def fleets(self, pricing):
        return [{"id": "F" + f["name"].split()[-1], "name": f["name"], "kind": f["kind"], "rate": Decimal(str(f["rate"])), "serving": self.sandbox or f["state"] == "serving"} for f in pricing["fleets"]]

    def accelerators(self, pricing):
        listed = pricing.get("accelerators", []) + (self.app.cfg.sandbox_accelerators if self.sandbox else [])
        found = []
        for a in listed:
            speed = Decimal(str(a["tokensPerSecond"])) if a.get("tokensPerSecond") else None
            found.append({"id": "A" + a["name"].split()[-1], "name": a["name"], "kind": a["kind"], "rate": Decimal(str(a["rate"])), "speed": speed, "serving": bool(speed) and (self.sandbox or a["state"] == "serving")})
        return found

    def fleet_field(self, pricing, value):
        found = next((f for f in self.fleets(pricing) if f["id"] == value), None)
        if not found:
            raise Refusal("Choose one of the fleets.")
        return found

    def model_field(self, pricing, value):
        if value not in [m["id"] for m in pricing["models"]]:
            raise Refusal("Choose one of the models: " + ", ".join(m["id"] for m in pricing["models"]) + ".", 404)
        return value

    def earliest(self, pricing):
        if self.sandbox:
            return None
        return datetime.fromisoformat(pricing["hourlyEarliestStart"]).replace(tzinfo=timezone(timedelta(hours=9)))

    def last_call(self, market):
        return timedelta(hours=market["lastCallHours"])

    def cover(self, now, pricing, commitments=None):
        found = self.app.commitments(now) if commitments is None else commitments
        first = floor_hour(now) + HOUR
        earliest = self.earliest(pricing)
        if earliest and earliest > first:
            first = earliest.astimezone(timezone.utc)
        key = (id(self.st), self.st.seq, first, floor_hour(now), found)
        if commitments is None and self.covered and self.covered[0] == key:
            return self.covered[1]
        cover = Cover(self.st, now, first, self.fleets(pricing), found, pricing["plans"]["maintenanceHoursPerWeek"])
        if commitments is None:
            self.covered = (key, cover)
        return cover

    def pending(self):
        now = self.clock()
        return sum((Decimal(o["hours"]) for o in self.book.orders.values() if o["product"] == "credits" and self.book.waiting(o, now)), Decimal(0))

    def on_sale(self, now, pricing):
        return max(Decimal(0), self.cover(now, pricing).headroom(year_later(now)) - self.pending())

    def check_cover(self, credits, now, pricing):
        sale = self.on_sale(now, pricing)
        if credits > sale:
            if sale <= 0:
                raise Refusal("Hour-credits are sold out for now: every free fleet-hour in the next 12 months already backs HC that has been sold, and we never sell HC that cannot be spent. Reserve a released hour, or try again later.", 409, "cover")
            raise Refusal(f"We can sell at most {text_hc(sale)} HC now: that is what our free fleet-hours in the next 12 months can still deliver, and we never sell HC that cannot be spent. Choose a smaller amount.", 409, "cover")

    def lot_parts(self, account, now):
        moment = stamp(now)
        return [[lot["expires"], lot["hc"]] for lot in self.st.lots.get(account, []) if lot["expires"] > moment and lot["hc"] > 0]

    def held_parts(self, bid):
        return [[lot["expires"], part] for lot, part in self.st.holds.get(bid["bid"], [])]

    def kept(self, moment):
        return Refusal(f"Our hours before {kst_text(moment)} are kept for hour-credits that lapse sooner. HC that lapses later can reserve them from {self.app.cfg.pricing()['market']['lastCallHours']} hours before they start. Choose a later hour, a released hour or the shared pool.", 409, "kept")

    def offer(self, fleet, start, now, market, earliest, cover, tranche=False, account=None):
        if start <= now:
            return None
        key = (fleet["id"], slot_text(start))
        if key in self.st.maintenance:
            return None
        holder = self.st.slots.get(key)
        if holder:
            res = self.st.reservations[holder]
            if res["state"] != "released" or tranche:
                return None
            ask = self.st.asks[res["ask"]]
            return {"fleet": key[0], "start": key[1], "price": Decimal(ask["min"]), "kind": "release", "ask": ask["ask"], "seq": ask["seq"], "seller": ask["account"]}
        if not fleet["serving"] or (earliest and start < earliest):
            return None
        price = fleet["rate"]
        if not cover.open(start, now, self.last_call(market), account if tranche else None, fleet["id"]):
            return None
        if not tranche and start - now > timedelta(days=market["advanceDays"]):
            return None
        return {"fleet": key[0], "start": key[1], "price": price, "kind": "hyor", "ask": None, "seq": 0, "seller": None}

    def window(self, bid, now, pricing, market, cover):
        earliest, found = self.earliest(pricing), []
        first, last = max(slot_time(bid["from"]), floor_hour(now) + HOUR), slot_time(bid["to"])
        for fleet in self.fleets(pricing):
            if bid["fleet"] and fleet["id"] != bid["fleet"]:
                continue
            moment = first
            while moment < last:
                offer = self.offer(fleet, moment, now, market, earliest, cover)
                if offer:
                    found.append(offer)
                moment += HOUR
        return found

    def choose(self, offers, hours, consecutive, cap, account):
        good = [o for o in offers if o["price"] <= cap and o["seller"] != account]
        if not consecutive:
            return sorted(good, key=rank)[:hours]
        rows, best = collections.defaultdict(dict), None
        for offer in good:
            rows[offer["fleet"]][offer["start"]] = offer
        for fleet in sorted(rows):
            row = rows[fleet]
            for start in sorted(row):
                run = [row.get(slot_text(slot_time(start) + HOUR * n)) for n in range(hours)]
                if all(run):
                    total = sum((o["price"] for o in run), Decimal(0))
                    if best is None or (total, start) < (best[0], best[1]):
                        best = (total, start, run)
        return best[2] if best else []

    def trade(self, offer, account, model, now, price, bid=None, pay="hc"):
        moment, reservation = stamp(now), new_id("RS")
        if bid:
            self.book.append("hc_spend_hold", {"bid": bid["bid"], "hc": text_hc(price)})
            spare = Decimal(bid["max"]) - price
            if spare > 0:
                self.book.append("hc_unhold", {"bid": bid["bid"], "hc": text_hc(spare)})
        elif pay == "hc":
            self.book.append("hc_debit", {"account": account, "hc": text_hc(price), "reason": "reserve", "ref": reservation, "now": moment})
        source = "tranche" if pay == "tranche" else offer["kind"]
        self.book.append("reserve", {"reservation": reservation, "account": account, "fleet": offer["fleet"], "start": offer["start"], "model": model, "price": text_hc(price), "source": source, "ask": offer["ask"], "bid": bid["bid"] if bid else None, "now": moment})
        if offer["kind"] == "release":
            credit = hc(price * (1 - self.st.asks[offer["ask"]]["fee"]))
            if credit > 0:
                self.book.append("hc_lot", {"lot": new_id("LT"), "account": offer["seller"], "hc": text_hc(credit), "expires": stamp(year_later(now)), "source": "sale", "ref": reservation, "now": moment})
        return reservation

    def bid_price(self, offer, bid):
        return offer["price"] if offer["kind"] == "hyor" or offer["seq"] < bid["seq"] else Decimal(bid["max"])

    def bid_blocked(self, chosen, bid, cover, now, market):
        parts, most, ours, spent = self.held_parts(bid), Decimal(bid["max"]), [], []
        for offer in chosen:
            price = self.bid_price(offer, bid)
            used = take_parts(parts, price, True)
            take_parts(parts, most - price, False)
            if offer["kind"] == "hyor":
                ours.append((slot_time(offer["start"]), offer["fleet"]))
                spent.extend(used)
        return cover.blocked(ours, spent, now, self.last_call(market)) if ours else None

    def match_resting(self, now, pricing, market):
        for bid in sorted((b for b in self.st.bids.values() if b["state"] == "open"), key=lambda b: b["seq"]):
            left = bid["hours"] - bid["filled"]
            if left <= 0 or (bid["consecutive"] and bid["filled"]):
                continue
            cover = self.cover(now, pricing)
            chosen = self.choose(self.window(bid, now, pricing, market, cover), left, bid["consecutive"], Decimal(bid["max"]), bid["account"])
            if bid["consecutive"] and len(chosen) < left:
                continue
            while chosen and self.bid_blocked(chosen, bid, cover, now, market):
                chosen = [] if bid["consecutive"] else chosen[:-1]
            for offer in chosen:
                self.trade(offer, bid["account"], bid["model"], now, self.bid_price(offer, bid), bid=bid)

    def advance(self, now, pricing, market):
        st, moment = self.st, stamp(now)
        for res in list(st.reservations.values()):
            if res["state"] == "released" and slot_time(res["start"]) <= now:
                self.book.append("pool_join", {"ask": res["ask"], "fleet": res["fleet"], "start": res["start"], "now": moment})
        for res in list(st.reservations.values()):
            key = (res["fleet"], res["start"])
            if res["state"] == "pooled" and slot_time(res["start"]) + HOUR <= now and key not in st.settled:
                earned = hc(st.pool_hc.get(key, Decimal(0)))
                credit = hc(earned * (1 - st.asks[res["ask"]]["fee"]))
                self.book.append("pool_settle", {"fleet": res["fleet"], "start": res["start"], "ask": res["ask"], "seller": res["account"], "earned": text_hc(earned), "credit": text_hc(credit), "seconds": text_hc(st.pool_seconds.get(key, Decimal(0))), "now": moment})
                if credit > 0:
                    self.book.append("hc_lot", {"lot": new_id("LT"), "account": res["account"], "hc": text_hc(credit), "expires": stamp(year_later(now)), "source": "pool", "ref": res["ask"], "now": moment})
        for bid in list(st.bids.values()):
            if bid["state"] not in ("open", "filled") or slot_time(bid["to"]) > now:
                continue
            held = st.held(bid["bid"])
            if held > 0:
                self.book.append("hc_unhold", {"bid": bid["bid"], "hc": text_hc(held)})
            if not bid["standing"]:
                self.book.append("bid_close", {"bid": bid["bid"], "state": "done" if bid["state"] == "filled" else "expired"})
                continue
            first, last = slot_time(bid["from"]), slot_time(bid["to"])
            while last <= now:
                first, last = first + WEEK, last + WEEK
            need = Decimal(bid["max"]) * bid["hours"]
            if st.balance(bid["account"], moment) < need:
                self.book.append("bid_close", {"bid": bid["bid"], "state": "paused"})
                continue
            self.book.append("bid_roll", {"bid": bid["bid"], "from": slot_text(first), "to": slot_text(last), "now": moment})
            self.book.append("hc_hold", {"bid": bid["bid"], "account": bid["account"], "hc": text_hc(need), "now": moment})
        self.match_resting(now, pricing, market)

    def tick(self):
        pricing, market = self.settings()
        with self.book.lock:
            self.advance(self.now(), pricing, market)

    def run(self, log):
        self.release_orphans()
        while True:
            time.sleep(30)
            try:
                self.tick()
            except Exception as error:
                log(f"market tick: {error}")

    def lot(self, account, credits, source, ref, now, pricing, note=None):
        data = {"lot": new_id("LT"), "account": account, "hc": text_hc(credits), "expires": stamp(self.cover(now, pricing).lapse(credits, year_later(now))), "source": source, "ref": ref, "now": stamp(now)}
        if note:
            data["note"] = note
        self.book.append("hc_lot", data)
        return data

    def buy(self, account, amount, pricing):
        with self.book.lock:
            now = self.now()
            credits = hc(amount / Decimal(str(pricing["market"]["hcPrice"])))
            self.check_cover(credits, now, pricing)
            data = self.lot(account, credits, "purchase", "sandbox", now, pricing)
        return {"lot": data["lot"], "hc": data["hc"], "expires": data["expires"]}

    def issue(self, account, credits, order):
        pricing, _ = self.settings()
        return self.lot(account, credits, "purchase", order, self.now(), pricing)

    def credit(self, account, credits, note):
        pricing, _ = self.settings()
        with self.book.lock:
            return self.lot(account, credits, "credit", new_id("CR"), self.now(), pricing, note)

    def tranche_left(self, account, month):
        return self.st.tranches[(account, month)] - self.st.tranche_used[(account, month)]

    def reserve(self, account, body):
        pricing, market = self.settings()
        fleet = self.fleet_field(pricing, body.get("fleet"))
        start = slot_field(body.get("start"), "hour")
        model = self.model_field(pricing, body.get("model"))
        pay = body.get("pay", "hc")
        if pay not in ("hc", "tranche"):
            raise Refusal("Pay with HC or with Tranche hours.")
        with self.book.lock:
            now = self.now()
            self.advance(now, pricing, market)
            cover = self.cover(now, pricing)
            offer = self.offer(fleet, start, now, market, self.earliest(pricing), cover, tranche=pay == "tranche", account=account)
            if not offer:
                raise Refusal("That hour is not for sale.", 409)
            if offer["seller"] == account:
                raise Refusal("That is your own released hour. Cancel the release to keep it.", 409)
            if pay == "tranche":
                month = month_of(offer["start"])
                with self.queue:
                    held = self.plan_held[(account, month)]
                if self.tranche_left(account, month) - held < 1:
                    raise Refusal(f"This account has no Tranche hours left for {month}{', apart from those held for prefill accelerator prompts' if held else ''}.", 409)
                price = Decimal(0)
            else:
                price = offer["price"]
                if body.get("max") is not None and price > hc_field(body.get("max"), "price"):
                    raise Refusal(f"The price is now {text_hc(price)} HC. Reserve again to accept it.", 409)
                if self.st.balance(account, stamp(now)) < price:
                    raise Refusal(f"This hour costs {text_hc(price)} HC, more than the account holds. Buy HC first.", 402)
                if offer["kind"] == "hyor":
                    stop = cover.blocked([(start, fleet["id"])], take_parts(self.lot_parts(account, now), price, True), now, self.last_call(market))
                    if stop:
                        raise self.kept(stop)
            reservation = self.trade(offer, account, model, now, price, pay=pay)
        return {"reservation": reservation, "price": text_hc(price)}

    def release(self, account, body):
        pricing, market = self.settings()
        with self.book.lock:
            now = self.now()
            self.advance(now, pricing, market)
            res = self.st.reservations.get(body.get("reservation"))
            if not res or res["account"] != account:
                raise Refusal("No such reservation on this account.", 404)
            if res["state"] != "held" or slot_time(res["start"]) <= now:
                raise Refusal("Only a reserved hour that has not started can be released.", 409)
            rate = self.fleet_field(pricing, res["fleet"])["rate"]
            least = rate if body.get("min") in (None, "") else hc_field(body.get("min"), "minimum price")
            ask = new_id("AK")
            self.book.append("ask", {"ask": ask, "account": account, "reservation": res["reservation"], "min": text_hc(least), "fee": format(Decimal(str(market["releaseFee"])), "f"), "now": stamp(now)})
            self.match_resting(now, pricing, market)
            return {"ask": ask, "state": self.st.asks[ask]["state"]}

    def cancel(self, account, body):
        pricing, market = self.settings()
        with self.book.lock:
            self.advance(self.now(), pricing, market)
            if body.get("ask"):

                ask = self.st.asks.get(body["ask"])
                if not ask or ask["account"] != account or ask["state"] != "open":
                    raise Refusal("No open release with that number on this account.", 404)
                self.book.append("ask_cancel", {"ask": ask["ask"]})
                return {"ask": ask["ask"], "state": "cancelled"}
            bid = self.st.bids.get(body.get("bid"))
            if not bid or bid["account"] != account or bid["state"] not in ("open", "filled", "paused"):
                raise Refusal("No open bid with that number on this account.", 404)
            held = self.st.held(bid["bid"])
            if held > 0:
                self.book.append("hc_unhold", {"bid": bid["bid"], "hc": text_hc(held)})
            self.book.append("bid_cancel", {"bid": bid["bid"]})
            return {"bid": bid["bid"], "state": "cancelled"}

    def bid(self, account, body):
        pricing, market = self.settings()
        fleet = None if body.get("fleet") in (None, "", "any") else self.fleet_field(pricing, body.get("fleet"))["id"]
        model = self.model_field(pricing, body.get("model"))
        first, last = slot_field(body.get("from"), "window start"), slot_field(body.get("to"), "window end")
        cap = hc_field(body.get("max"), "maximum price")
        consecutive, standing = body.get("consecutive") is True, body.get("standing") is True
        with self.book.lock:
            now = self.now()
            self.advance(now, pricing, market)
            if last <= first or last <= now:
                raise Refusal("The window must end after it starts, and in the future.")
            if last - first > timedelta(days=market["maxBidDays"]):
                raise Refusal(f"A window can span at most {market['maxBidDays']} days.")
            if standing and last - first > WEEK:
                raise Refusal("A standing bid repeats every week, so its window can span at most 7 days.")
            hours = count_field(body.get("hours"), "number of hours", int((last - first) / HOUR))
            need = cap * hours
            if self.st.balance(account, stamp(now)) < need:
                raise Refusal(f"This bid holds {text_hc(need)} HC until it fills, more than the account holds. Buy HC first.", 402)
            bid, moment = new_id("BD"), stamp(now)
            self.book.append("bid", {"bid": bid, "account": account, "fleet": fleet, "model": model, "from": slot_text(first), "to": slot_text(last), "hours": hours, "consecutive": consecutive, "max": text_hc(cap), "standing": standing, "now": moment})
            self.book.append("hc_hold", {"bid": bid, "account": account, "hc": text_hc(need), "now": moment})
            self.match_resting(now, pricing, market)
            found = self.st.bids[bid]
            return {"bid": bid, "filled": found["filled"], "state": found["state"]}

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
            raise Refusal("Send a valid SparkPipe pool key as Authorization: Bearer sk-hc-...", 401, "invalid_api_key")
        return record

    def grant_tranche(self, account, month, hours):
        with self.book.lock:
            self.book.append("tranche_grant", {"account": account, "month": month, "hours": hours})
        return {"month": month, "hours": hours}

    def demo_tranche(self, account):
        _, market = self.settings()
        local = self.now() + timedelta(hours=9)
        months = [local.strftime("%Y-%m"), (local.replace(day=1) + timedelta(days=32)).strftime("%Y-%m")]
        for month in months:
            self.grant_tranche(account, month, market["trancheDemoHours"])
        return {"months": months, "hours": market["trancheDemoHours"]}

    def schedule_maintenance(self, fleet_id, day, hours=None):
        pricing, market = self.settings()
        fleet = self.fleet_field(pricing, fleet_id)
        try:
            local = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=KST)
        except (TypeError, ValueError):
            raise Refusal("Give the Monday of the week as YYYY-MM-DD, in KST.")
        if local.weekday() != 0:
            raise Refusal("Maintenance allowances run by calendar week, Monday to Sunday: give the Monday of the week.")
        with self.book.lock:
            now = self.now()
            self.advance(now, pricing, market)
            begin, end = local.astimezone(timezone.utc), (local + WEEK).astimezone(timezone.utc)
            if not fleet["serving"]:
                raise Refusal(f"{fleet['name']} is not serving.")
            earliest = max(begin, floor_hour(now + WEEK) + (HOUR if now + WEEK > floor_hour(now + WEEK) else timedelta(0)))
            commitments = self.app.commitments(now)
            allowance = min([int(pricing["plans"]["maintenanceHoursPerWeek"])] + [c.allowance for c in commitments if c.begins < end and c.ends > begin])
            hours = allowance if hours is None else hours
            done = sum(m["hours"] for m in self.st.maintained if m["fleet"] == fleet["id"] and begin <= slot_time(m["start"]) < end)
            if hours < 1 or done + hours > allowance:
                raise Refusal(f"{fleet['name']} has {max(0, allowance - done)} of its {allowance} maintenance hours left in that week.")
            last = {}
            for m in self.st.maintained:
                if m["fleet"] == fleet["id"]:
                    last[window_of(slot_time(m["start"]), pricing["windows"])] = m["start"]
            best = None
            for n in range(7 * 24 - hours + 1):
                start = begin + HOUR * n
                if start < earliest:
                    continue
                run = [slot_text(start + HOUR * k) for k in range(hours)]
                if any((fleet["id"], slot) in self.st.maintenance for slot in run):
                    continue
                held = sum(1 for slot in run if self.st.slots.get((fleet["id"], slot)))
                score = (held, last.get(window_of(start, pricing["windows"]), ""), start)
                if best is None or score < best[0]:
                    best = (score, start, run)
            if best is None:
                raise Refusal(f"No run of {hours} free hours on {fleet['name']} that week starts at least 7 days from now.")
            start, run = best[1], best[2]
            moment, ident = stamp(now), new_id("MT")
            touched = [self.st.reservations[self.st.slots[(fleet["id"], slot)]] for slot in run if self.st.slots.get((fleet["id"], slot))]
            self.book.append("maintenance", {"maintenance": ident, "fleet": fleet["id"], "start": slot_text(start), "hours": hours, "now": moment})
            credits, notices = [], collections.defaultdict(list)
            for res in touched:
                self.book.append("reservation_credit", {"reservation": res["reservation"], "reason": "maintenance", "ref": ident, "now": moment})
                if res["source"] != "tranche" and Decimal(res["price"]) > 0:
                    self.lot(res["account"], Decimal(res["price"]), "maintenance", res["reservation"], now, pricing)
                credits.append(res["reservation"])
                notices[res["account"]].append(f"{kst_text(slot_time(res['start']))} on {fleet['name']}: your reservation {res['reservation']} is credited back in full")
            cover = self.cover(now, pricing)
            for slot in run:
                moment_slot = slot_time(slot)
                key = hour_key(moment_slot)
                live = [c for c in commitments if c.begins <= moment_slot < c.ends and key in c.hours]
                hit = [c for c in live if fleet["id"] in c.lock]
                short = max(0, cover.aside(moment_slot) - len(cover.free(moment_slot)))
                hit += sorted((c for c in live if not c.lock), key=lambda c: c.begins, reverse=True)[:short]
                for found in hit:
                    if any(res["account"] == found.account and res["start"] == slot for res in touched):
                        continue
                    self.book.append("tranche_grant", {"account": found.account, "month": month_of(slot), "hours": 1, "credit": "maintenance", "plan": found.plan, "ref": ident})
                    credits.append(found.plan)
                    if found.account:
                        notices[found.account].append(f"{kst_text(moment_slot)}: one plan hour of {found.plan} is credited")
            for account, lines in notices.items():
                if account in self.app.book.accounts:
                    self.app.mail(self.app.book.accounts[account]["email"], f"SparkPipe maintenance on {fleet['name']}, {kst_text(start)}",
                                  "\n".join([f"We will maintain {fleet['name']} for {hours} hours from {kst_text(start)}."] + lines + ["", "Nothing is counted against you for these hours."]))
        return {"maintenance": ident, "fleet": fleet["id"], "start": slot_text(start), "hours": hours, "credited": credits}

    def move_clock(self, hours):
        pricing, market = self.settings()
        with self.book.lock:
            self.book.append("clock", {"offset": self.st.offset + hours * 3600})
            self.advance(self.now(), pricing, market)
        return {"now": stamp(self.now())}

    def pooled(self, now, pricing):
        start, found = slot_text(floor_hour(now)), []
        locked = self.cover(now, pricing).locked(floor_hour(now))
        for fleet in self.fleets(pricing):
            if not fleet["serving"]:
                continue
            holder = self.st.slots.get((fleet["id"], start))
            if (holder is None and fleet["id"] in locked) or (fleet["id"], start) in self.st.maintenance:
                continue
            if holder is None:
                found.append({"fleet": fleet["id"], "start": start, "order": (1, 0), "seller": None})
            elif self.st.reservations[holder]["state"] == "pooled":
                res = self.st.reservations[holder]
                found.append({"fleet": fleet["id"], "start": start, "order": (0, self.st.asks[res["ask"]]["seq"]), "seller": res["account"]})
        return sorted(found, key=lambda s: (s["order"], s["fleet"]))

    def dedicated(self, account, now):
        start = slot_text(floor_hour(now))
        return next((r for r in self.st.reservations.values() if r["account"] == account and r["start"] == start and r["state"] == "held"), None)

    def reprice(self, capacity, market):
        moment = time.monotonic()
        while self.recent and self.recent[0][0] < moment - 60:
            self.recent.popleft()
        if moment - self.priced < 60 and self.priced:
            return
        used = sum((seconds for _, seconds in self.recent), Decimal(0)) + Decimal(self.active * 60)
        load = min(Decimal(1), used / Decimal(capacity * 60)) if capacity else Decimal(0)
        pool = market["pool"]
        self.multiplier = min(Decimal(str(pool["maxMultiplier"])), max(Decimal(str(pool["minMultiplier"])), Decimal("0.5") + Decimal("2.5") * load))
        self.priced = moment

    def price(self, market):
        base = Decimal(1) / (Decimal(market["pool"]["seats"]) * 3600)
        return (base * self.multiplier).quantize(PRICE_STEP)

    def accelerator_join(self, ticket, options, tokens):
        pricing, market = self.settings()
        serving = [a for a in self.accelerators(pricing) if a["serving"]]
        if not serving:
            return self.unaccelerated(options, "none_serving")
        account, pay, bid = ticket["account"], options["pay"], options["bid"]
        estimate = max(max(hc(Decimal(tokens) / a["speed"] * a["rate"] * bid / 3600) for a in serving), EMPTY)
        job = {"job": new_id("AJ"), "account": account, "key": ticket["key"], "model": ticket["model"], "pay": pay, "bid": bid,
               "tokens": tokens, "estimate": estimate, "accelerator": None}
        with self.book.lock:
            now = self.now()
            job["month"] = kst_month(now)
            if pay == "hc":
                if self.st.balance(account, stamp(now)) < estimate:
                    raise Refusal("This account's HC does not cover this prompt's estimated accelerator time.", 402, "insufficient_hc")
                self.book.append("hc_hold", {"bid": job["job"], "account": account, "hc": text_hc(estimate), "now": stamp(now)})
            else:
                with self.queue:
                    if self.tranche_left(account, job["month"]) - self.plan_held[(account, job["month"])] < estimate:
                        raise Refusal("This month's Tranche has too few unused Plan hours for this prompt's estimated accelerator time.", 402, "insufficient_plan_hours")
                    self.plan_held[(account, job["month"])] += estimate
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
        if job["pay"] == "plan":
            with self.queue:
                self.plan_held[(job["account"], job["month"])] -= job["estimate"]
            return
        with self.book.lock:
            held = self.st.held(job["job"])
            if held > 0:
                self.book.append("hc_unhold", {"bid": job["job"], "hc": text_hc(held)})

    def release_orphans(self):
        with self.book.lock:
            for job, parts in list(self.st.holds.items()):
                if job.startswith("AJ") and self.st.held(job) > 0:
                    self.book.append("hc_unhold", {"bid": job, "hc": text_hc(self.st.held(job))})

    def unaccelerated(self, options, reason):
        if not options["fallback"]:
            raise Refusal("No prefill accelerator started this prompt in time, and the request asked not to fall back.", 503, "accelerator_busy")
        return {"used": False, "reason": reason}

    def dispatch(self, serving):
        free = sorted((a for a in serving if a["id"] not in self.busy), key=lambda a: (-a["speed"], a["id"]))
        while free and self.waiting:
            job = min(self.waiting, key=lambda j: (-j["bid"], j["seq"]))
            self.waiting.remove(job)
            job["accelerator"] = free.pop(0)
            self.busy[job["accelerator"]["id"]] = job["job"]
        self.queue.notify_all()

    def accelerator_finish(self, job, seconds):
        pricing, _ = self.settings()
        accel = job["accelerator"]
        with self.queue:
            if self.busy.get(accel["id"]) != job["job"]:
                return job.get("result")
            del self.busy[accel["id"]]
            self.dispatch([a for a in self.accelerators(pricing) if a["serving"]])
        waited = Decimal(str(round(job["started"] - job["joined"], 3)))
        if seconds is None:
            self.unhold(job)
            job["result"] = {"used": False, "reason": "unfinished", "accelerator": accel["id"]}
            return job["result"]
        if job["pay"] == "plan":
            self.unhold(job)
        seconds = Decimal(str(round(seconds, 3)))
        with self.book.lock:
            now = self.now()
            month = kst_month(now)
            have = max(self.tranche_left(job["account"], month), Decimal(0)) if job["pay"] == "plan" else self.st.held(job["job"]) + self.st.balance(job["account"], stamp(now))
            amount = min(hc(seconds * accel["rate"] * job["bid"] / 3600), hc(have))
            charge = new_id("XC")
            self.book.append("accelerator_charge", {
                "charge": charge, "job": job["job"], "account": job["account"], "key": job["key"], "model": job["model"], "accelerator": accel["id"], "rate": text_hours(accel["rate"]),
                "bid": format(job["bid"], "f"), "pay": job["pay"], "month": month if job["pay"] == "plan" else None, "seconds": format(seconds, "f"), "waited": format(waited, "f"),
                "tokens": job["tokens"], "amount": text_hc(amount), "now": stamp(now),
            })
        job["result"] = {"used": True, "charge": charge, "accelerator": accel["id"], "rate": text_hours(accel["rate"]), "bid": format(job["bid"], "f"), "seconds": format(seconds, "f"), "waited": format(waited, "f"), "pay": job["pay"], "amount": text_hc(amount)}
        return job["result"]

    def accelerator_view(self, pricing):
        listed = self.accelerators(pricing)
        speeds = [a["speed"] for a in listed if a["serving"]]
        with self.queue:
            queue = sorted(self.waiting, key=lambda j: (-j["bid"], j["seq"]))
            busy = set(self.busy)
        return {
            "accelerators": [{"id": a["id"], "name": a["name"], "kind": a["kind"], "rate": text_hours(a["rate"]), "tokensPerSecond": None if a["speed"] is None else text_hours(a["speed"]),
                              "serving": a["serving"], "busy": a["id"] in busy, "hcPerMinute": text_hc(a["rate"] / 60)} for a in listed],
            "queue": [{"bid": format(j["bid"], "f"), "seconds": text_hours(Decimal(j["tokens"]) / max(speeds)) if speeds else None} for j in queue],
        }

    def pool_open(self, key, model, deferred=False):
        pricing, market = self.settings()
        record = self.key_account(key)
        model = self.model_field(pricing, model)
        account = record["account"]
        with self.book.lock:
            now = self.now()
            self.advance(now, pricing, market)
            mine = self.dedicated(account, now)
            slots = [] if mine else self.pooled(now, pricing)
            if not mine and not slots:
                raise Refusal("No fleet is in the shared pool this hour. Try again next hour, or reserve an hour.", 503, "pool_closed")
            if not mine and self.st.balance(account, stamp(now)) < EMPTY:
                raise Refusal("This account has no HC left. Buy HC to use the shared pool.", 402, "insufficient_hc")
        seats = 1
        with self.lock:
            capacity = len(slots) * market["pool"]["seats"]
            self.reprice(capacity, market)
            if not mine and not deferred:
                if self.active + seats > capacity:
                    raise Refusal("The shared pool is full right now. Try again in a moment.", 429, "pool_full")
                self.active += seats
            price = self.price(market)
        return {"account": account, "key": record["key"], "model": model, "dedicated": mine["reservation"] if mine else None, "seats": seats, "price": price, "slots": slots, "opened": time.monotonic(), "began": time.monotonic(), "seated": not deferred}

    def pool_seat(self, ticket):
        _, market = self.settings()
        with self.lock:
            if ticket["seated"]:
                return
            if not ticket["dedicated"]:
                self.active += ticket["seats"]
            ticket.update(seated=True, price=self.price(market), began=time.monotonic())

    def split(self, slots, seat_seconds, cost, market):
        if seat_seconds <= 0 or not slots:
            return []
        room_each, parts, left = Decimal(market["pool"]["seats"]) * 3600, [], seat_seconds
        for index, slot in enumerate(slots):
            room = room_each - self.st.pool_seconds.get((slot["fleet"], slot["start"]), Decimal(0))
            share = left if index == len(slots) - 1 else min(left, max(room, Decimal(0)))
            if share > 0:
                parts.append([slot["fleet"], slot["start"], share])
                left -= share
            if left <= 0:
                break
        result, spent = [], Decimal(0)
        for index, (fleet, start, share) in enumerate(parts):
            part = cost - spent if index == len(parts) - 1 else hc(cost * share / seat_seconds)
            spent += part
            result.append([fleet, start, text_hc(part), format(share, "f")])
        return result

    def pool_close(self, ticket, tokens_in, tokens_out):
        self.pool_seat(ticket)
        seconds = Decimal(str(round(time.monotonic() - ticket["began"], 3)))
        seat_seconds = seconds * ticket["seats"]
        _, market = self.settings()
        with self.lock:
            if not ticket["dedicated"]:
                self.active -= ticket["seats"]
                self.recent.append((time.monotonic(), seat_seconds))
            self.speed.append((tokens_out, seat_seconds))
        if ticket["dedicated"]:
            return {"hc": "0.000000", "seconds": format(seconds, "f"), "dedicated": ticket["dedicated"]}
        with self.book.lock:
            now = self.now()
            cost = min(hc(seat_seconds * ticket["price"]), hc(self.st.balance(ticket["account"], stamp(now))))
            charge = new_id("CH")
            self.book.append("pool_charge", {"charge": charge, "account": ticket["account"], "key": ticket["key"], "model": ticket["model"], "seconds": format(seconds, "f"), "seats": ticket["seats"], "hc": text_hc(cost), "tokensIn": tokens_in, "tokensOut": tokens_out, "split": self.split(ticket["slots"], seat_seconds, cost, market), "now": stamp(now)})
        return {"charge": charge, "hc": text_hc(cost), "seconds": format(seconds, "f"), "dedicated": None}

    def tokens_per_seat_second(self):
        tokens = sum(t for t, _ in self.speed)
        seconds = sum((s for _, s in self.speed), Decimal(0))
        return Decimal(tokens) / seconds if seconds > 0 else None

    def pool_view(self, pricing, market, now):
        slots = self.pooled(now, pricing)
        capacity = len(slots) * market["pool"]["seats"]
        with self.lock:
            self.reprice(capacity, market)
            price, active = self.price(market), self.active
        dollars = Decimal(str(market["hcPrice"]))
        speed = self.tokens_per_seat_second()
        return {
            "fleets": [s["fleet"] for s in slots], "seats": capacity, "active": active,
            "hcPerSeatSecond": format(price, "f"), "usdPerSeatHour": format((price * 3600 * dollars).quantize(Decimal("0.01")), "f"),
            "usdPerMillionTokens": None if not speed else format((price * dollars * Decimal(1000000) / speed).quantize(Decimal("0.01")), "f"),
            "models": [dict(m, usdPerMillionOutput=per_million(price * dollars, m["decode"]), usdPerMillionPrompt=per_million(price * dollars, m["prefill"])) for m in pricing["models"]],
        }

    def public(self):
        pricing, market = self.settings()
        with self.book.lock:
            now = self.now()
            self.advance(now, pricing, market)
            pool = self.pool_view(pricing, market, now)
            sale = self.on_sale(now, pricing)
        accelerated = self.accelerator_view(pricing)
        return {
            "now": stamp(now), "sandbox": self.sandbox,
            "market": {k: market[k] for k in ("hcPrice", "hcPriceKRW", "minPurchase", "validityMonths", "releaseFee", "advanceDays", "lastCallHours", "maxBidDays")},
            "onSale": text_hc(sale),
            "vatRate": pricing["vatRate"],
            "fleets": [{"id": f["id"], "name": f["name"], "kind": f["kind"], "rate": text_hc(f["rate"]), "serving": f["serving"]} for f in self.fleets(pricing)],
            "pool": pool,
            "accelerators": accelerated["accelerators"],
            "acceleratorQueue": accelerated["queue"],
            "acceleratorLimits": market["accelerator"],
        }

    def reservation_view(self, res, now):
        begins = slot_time(res["start"])
        state = res["state"]
        if state == "held" and begins + HOUR <= now:
            state = "used"
        elif state == "held" and begins <= now:
            state = "live"
        shown = {key: res[key] for key in ("reservation", "fleet", "start", "model", "price", "source", "created")}
        shown["state"] = state
        ask = self.st.asks.get(res["ask"]) if res["ask"] else None
        shown["release"] = {key: ask[key] for key in ("ask", "min", "state", "price", "earned")} if ask else None
        return shown

    def view(self, account):
        pricing, market = self.settings()
        with self.book.lock:
            now = self.now()
            self.advance(now, pricing, market)
            st, moment = self.st, stamp(now)
            month = (now + timedelta(hours=9)).strftime("%Y-%m")
            following = (now.replace(day=1) + timedelta(days=40) + timedelta(hours=9)).strftime("%Y-%m")
            reservations = sorted((r for r in st.reservations.values() if r["account"] == account and slot_time(r["start"]) + timedelta(hours=25) > now), key=lambda r: (r["start"], r["fleet"]))
            return {
                "now": moment,
                "balance": text_hc(st.balance(account, moment)),
                "held": text_hc(sum((st.held(b["bid"]) for b in st.bids.values() if b["account"] == account), Decimal(0)) + sum((st.held(j) for j, owner in st.hold_owner.items() if owner == account and j.startswith("AJ")), Decimal(0))),
                "lots": [{"lot": l["lot"], "hc": text_hc(l["hc"]), "first": l["first"], "expires": l["expires"], "source": l["source"], "ref": l["ref"], "issued": l["issued"], "expired": l["expires"] <= moment} for l in st.lots.get(account, [])],
                "reservations": [self.reservation_view(r, now) for r in reservations],
                "bids": [dict({k: b[k] for k in ("bid", "fleet", "model", "from", "to", "hours", "filled", "consecutive", "max", "standing", "state", "now")}, held=text_hc(st.held(b["bid"]))) for b in st.bids.values() if b["account"] == account and b["state"] in ("open", "filled", "paused")],
                "tranche": [{"month": m, "granted": st.tranches[(account, m)], "used": text_hours(st.tranche_used[(account, m)]), "left": text_hours(self.tranche_left(account, m))} for m in (month, following) if st.tranches[(account, m)]],
                "keys": [{k: r[k] for k in ("key", "note", "hint", "created", "revoked")} for r in st.keys.values() if r["account"] == account],
                "charges": list(st.charges[account])[:20],
                "dedicated": (self.dedicated(account, now) or {}).get("reservation"),
            }

    def calendar(self, account, days):
        pricing, market = self.settings()
        days = max(1, min(days, market["maxBidDays"]))
        with self.book.lock:
            now = self.now()
            self.advance(now, pricing, market)
            first, earliest, fleets = floor_hour(now) + HOUR, self.earliest(pricing), []
            cover = self.cover(now, pricing)
            for fleet in self.fleets(pricing):
                cells = []
                for n in range(days * 24):
                    start = first + HOUR * n
                    holder = self.st.slots.get((fleet["id"], slot_text(start)))
                    if (fleet["id"], slot_text(start)) in self.st.maintenance:
                        cells.append(["maintenance", None])
                    elif holder:
                        res = self.st.reservations[holder]
                        mine = res["account"] == account
                        if res["state"] == "released":
                            cells.append(["mine-release" if mine else "release", self.st.asks[res["ask"]]["min"]])
                        else:
                            cells.append(["mine" if mine else "taken", None])
                    else:
                        offer = self.offer(fleet, start, now, market, earliest, cover)
                        cells.append(["hyor", text_hc(offer["price"])] if offer else ["closed", None])
                fleets.append({"id": fleet["id"], "name": fleet["name"], "cells": cells})
        return {"from": slot_text(first), "hours": days * 24, "fleets": fleets}

    def book_view(self):
        with self.book.lock:
            st = self.st
            return {
                "asks": sorted(({"fleet": st.reservations[a["reservation"]]["fleet"], "start": st.reservations[a["reservation"]]["start"], "min": a["min"], "at": a["at"]} for a in st.asks.values() if a["state"] == "open"), key=lambda a: (a["start"], a["fleet"])),
                "bids": sorted(({"fleet": b["fleet"], "model": b["model"], "from": b["from"], "to": b["to"], "left": b["hours"] - b["filled"], "max": b["max"], "standing": b["standing"], "consecutive": b["consecutive"]} for b in st.bids.values() if b["state"] == "open"), key=lambda b: (b["from"], b["max"])),
                "trades": list(st.trades)[:30],
            }

    def market_text(self):
        pricing, market = self.settings()
        cal = self.calendar(None, market["advanceDays"])
        bids = self.book_view()["bids"]
        sale = self.public()["onSale"]
        lines = [f"# SparkPipe Hour Market, generated {stamp(self.now())}", f"# hour-credits on sale now: {sale} HC", "# prices in HC per fleet-hour; times in UTC; ask kinds: hyor = our capacity, release = a customer's released hour", "# start fleet best_ask kind best_bid"]
        start = slot_time(cal["from"])
        for fleet in cal["fleets"]:
            for n, (kind, price) in enumerate(fleet["cells"]):
                moment = slot_text(start + HOUR * n)
                best = max((Decimal(b["max"]) for b in bids if b["from"] <= moment < b["to"] and b["fleet"] in (None, fleet["id"])), default=None)
                ask = price if kind in ("hyor", "release", "mine-release") else "-"
                lines.append(f"{moment} {fleet['id']} {ask} {kind.replace('mine-', '')} {text_hc(best) if best is not None else '-'}")
        lines.append("# last trades: at fleet start price kind")
        lines.extend(f"{t['at']} {t['fleet']} {t['start']} {t['price']} {t['kind']}" for t in self.book_view()["trades"])
        return "\n".join(lines) + "\n"

    def report(self, commitments=None):
        pricing, market = self.settings()
        with self.book.lock:
            now = self.now()
            found = self.app.commitments(now) if commitments is None else commitments
            cover, life, pending = self.cover(now, pricing, found), year_later(now), self.pending()
            clashes, moment = [], cover.first
            while moment < max([cover.first] + [c.ends for c in found]):
                if cover.clash(moment):
                    clashes.append(moment)
                moment += HOUR
            tight = cover.tightest()
            return {
                "now": stamp(now), "first": stamp(cover.first), "fleets": sorted(cover.rates), "plans": len(found),
                "free": text_hc(cover.supply(life)), "unspent": text_hc(cover.unspent()), "pending": text_hc(pending),
                "onSale": text_hc(max(Decimal(0), cover.headroom(life) - pending)),
                "tightest": None if tight is None else {"slack": text_hc(tight[0]), "lapse": stamp(tight[1])},
                "clashes": len(clashes), "firstClash": stamp(clashes[0]) if clashes else None,
            }

    def pool_text(self):
        state = self.public()
        pool = state["pool"]
        lines = [f"# SparkPipe shared pool, generated {state['now']}", "# model hc_per_seat_second usd_per_seat_hour usd_per_million_tokens active_seats seats pooled_fleets est_usd_per_million_prompt est_usd_per_million_output"]
        for model in pool["models"]:
            lines.append(f"{model['id']} {pool['hcPerSeatSecond']} {pool['usdPerSeatHour']} {pool['usdPerMillionTokens'] or '-'} {pool['active']} {pool['seats']} {','.join(pool['fleets']) or '-'} {model['usdPerMillionPrompt']} {model['usdPerMillionOutput']}")
        lines.append("# prefill accelerators: id rate hc_per_minute_at_bid_1 tokens_per_second state")
        for a in state["accelerators"]:
            lines.append(f"{a['id']} {a['rate']} {a['hcPerMinute']} {a['tokensPerSecond'] or '-'} {'busy' if a['busy'] else 'free' if a['serving'] else 'not-serving'}")
        lines.append("# accelerator queue, first to be computed first: bid estimated_seconds")
        for job in state["acceleratorQueue"]:
            lines.append(f"{job['bid']} {job['seconds'] or '-'}")
        return "\n".join(lines) + "\n"
