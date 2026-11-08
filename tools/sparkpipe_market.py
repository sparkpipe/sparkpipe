import base64
import calendar
import collections
import hashlib
import ipaddress
import json
import math
import secrets
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
KINDS = ("hc_lot", "hc_debit", "booking", "lock", "unlock", "lock_lapse", "round", "api_key", "api_key_revoke", "api_key_rotate", "maintenance", "maintenance_done", "clock",
         "auto_extend", "auto_extend_wait", "downtime", "failover", "failover_end")
AVAILABILITY = {"free": "available"}
SCOPES = ("inference", "state", "storage", "functions", "models", "account:read")
ROLES = ("owner", "admin", "member")
ROLE_SCOPES = {"owner": SCOPES, "admin": SCOPES, "member": ("inference", "functions", "account:read")}
PRIORITIES = ("interactive", "background")
EVENTS = ("booking.deadline", "booking.confirmed", "fleet.allocated", "waitlist.position", "run.finished", "run.failed", "report.ready", "fetch.finished", "fetch.failed", "fetch.needs_token",
          "pack.warm", "cache.restored", "models.changed", "downtime.credited", "working_state.lost", "fleet.failover")
HUB_EVENTS = ("run.finished", "run.failed", "report.ready", "fetch.finished", "fetch.failed", "fetch.needs_token", "pack.warm", "cache.restored", "models.changed")
EMAIL_EVENTS = ("booking.deadline", "fleet.allocated", "downtime.credited", "working_state.lost", "fleet.failover")
REQUIRED_EMAIL = ("fleet.allocated", "downtime.credited", "working_state.lost")
DEADLINE_HOURS = (72, 48, 6, 2)
DEFAULT_WEEKS_AHEAD = 1
REBUILD_HOURS = 24
OUTAGE = timedelta(minutes=15)
OUTAGE_PASS = timedelta(days=1)
UPTIME_WINDOWS = (("24h", 1), ("7d", 7), ("30d", 30), ("90d", 90))
KEY_IPS = 32
GRACE_SECONDS = 30


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


def slot_floor(moment):
    return slot_time(slot_text(moment))


def slot_ceil(moment):
    return slot_floor(moment) + HOUR if slot_floor(moment) < moment else moment


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


def count_field(value, label, most, least=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < least or value > most:
        raise Refusal(f"The {label} must be a whole number from {least} to {most}.")
    return value


def stamp_field(value, label):
    try:
        return stamp_time(value)
    except (TypeError, ValueError):
        raise Refusal(f"Give the {label} as a UTC time such as 2027-01-05T09:00:00Z.", 400, "invalid_request")


def short_text(value, label, size):
    if not isinstance(value, str) or not 0 < len(value) <= size or not value.isprintable():
        raise Refusal(f"Give the {label} as text of at most {size} characters.", 400, "invalid_request")
    return value


def address_in(ip, networks):
    try:
        found = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(found in ipaddress.ip_network(network) for network in networks)


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
        self.bookings = {}
        self.keys = {}
        self.key_hashes = {}
        self.refunds = collections.defaultdict(list)
        self.maintenance = {}
        self.round = None
        self.offset = 0
        self.seq = 0
        self.extend = {}
        self.rebuilds = {}
        self.downtime = {}
        self.downtime_hours = set()
        self.failovers = {}

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

    def on_hc_lot(self, data):
        self.lots[data["account"]].append(dict({name: data[name] for name in ("fleet", "hours") if name in data}, lot=data["lot"], hc=Decimal(data["hc"]), first=data["hc"], expires=data["expires"], source=data["source"], ref=data["ref"], issued=data["now"]))
        self.lots[data["account"]].sort(key=lambda lot: (lot["expires"], lot["issued"]))
        if data["source"] == "rebuild":
            self.rebuilds[data["ref"]] = data
        if data["source"] == "downtime":
            self.downtime_hours.update((data["fleet"], hour) for hour in data["hours"])

    def on_hc_debit(self, data):
        if data.get("lot"):
            next(lot for lot in self.lots[data["account"]] if lot["lot"] == data["lot"])["hc"] -= Decimal(data["hc"])
        else:
            self.take(data["account"], data["hc"], data["now"])

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
        self.keys[data["key"]] = {"key": data["key"], "account": data["account"], "member": data.get("member", data["account"]), "note": data["note"], "hint": data["hint"], "created": data["now"], "revoked": False,
                                  "scopes": data.get("scopes", list(SCOPES)), "fleet": data.get("fleet"), "allow_swap": data.get("allow_swap", False), "expires": data.get("expires"),
                                  "ips": data.get("ips", []), "priority": data.get("priority"), "rotated_from": data.get("rotated_from"), "retires": None}
        self.key_hashes[data["hash"]] = data["key"]

    def on_api_key_revoke(self, data):
        self.keys[data["key"]]["revoked"] = True

    def on_api_key_rotate(self, data):
        self.keys[data["key"]].update(retires=data["retires"], rotated_to=data["new"])

    def on_auto_extend(self, data):
        if data["weeks"]:
            self.extend[(data["account"], data["fleet"])] = {"weeks": data["weeks"], "now": data["now"]}
        else:
            self.extend.pop((data["account"], data["fleet"]), None)

    def on_auto_extend_wait(self, data):
        self.extend[(data["account"], data["fleet"])].update(through=data["through"], waiting=data["reason"])

    def on_downtime(self, data):
        self.downtime[data["downtime"]] = data

    def on_failover(self, data):
        self.failovers[data["failover"]] = dict(data, begins=stamp_time(data["start"]), ends=None, end=None)

    def on_failover_end(self, data):
        self.failovers[data["failover"]].update(end=data["end"], ends=stamp_time(data["end"]), reason=data["reason"])

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
            raise Refusal("Choose a Fleet on the Fleet List: " + ", ".join(f["id"] for f in pricing["fleets"]) + ".")
        return found

    def fleet_name(self, fleet_id):
        return next((f["name"] for f in self.settings()["fleets"] if f["id"] == fleet_id), fleet_id)

    def notify(self, account, kind, data, now, summary, text, once=None, mail=False):
        targets = self.targets(account, kind)
        if mail and "mail" not in targets:
            targets.append("mail")
        found = {"notice": new_id("NT"), "account": account, "type": kind, "data": data, "summary": summary, "text": text, "targets": targets, "now": stamp(now)}
        if once:
            found["once"] = once
        self.app.notices.add(found)
        self.app.notified(found)
        return found

    def targets(self, account, kind):
        mine = self.app.hooks_of(account)
        found = [w["webhook"] for w in mine if w["kind"] == "url" and (kind in w["events"] or "*" in w["events"])]
        email = next((w for w in mine if w["kind"] == "email"), None)
        if kind in REQUIRED_EMAIL or (kind in email["events"] or "*" in email["events"] if email else kind in EMAIL_EVENTS):
            found.append("mail")
        return found

    def confirmed(self, booking, now):
        what = f"{booking['weeks']} Weeks" if booking["kind"] == "week" else f"{booking['hours']} hours"
        data = {key: booking[key] for key in ("booking", "fleet", "kind", "start", "end", "hours", "weeks", "hc", "discount", "source", "auto") if key in booking}
        keep = " " + self.extend_line(booking["account"], booking["fleet"]) if booking["kind"] == "week" else ""
        self.notify(booking["account"], "booking.confirmed", data, now, f"Booked {self.fleet_name(booking['fleet'])}: {what} from {kst_text(booking['begins'])}",
                    f"Booking {booking['booking']}: {self.fleet_name(booking['fleet'])}, {what} from {kst_text(booking['begins'])} to {kst_text(booking['ends'])}, for {booking['hc']} HC.{keep}")

    def positions(self):
        return {account: n for n, account in enumerate(self.queue_order(), 1)}

    def moved(self, before, now, actors):
        after = self.positions()
        for account in sorted(set(before) | set(after)):
            if account not in actors and before.get(account) != after.get(account):
                where = f"position {after[account]} of {len(after)}" if account in after else "off the waitlist"
                self.notify(account, "waitlist.position", {"position": after.get(account), "locked": text_hc(self.st.locked(account)), "waiting": len(after)}, now,
                            f"Your SparkPipe waitlist place: {where}", f"Your account is now {where}, with {text_hc(self.st.locked(account))} HC locked.")

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

    def week_price(self, rules, rate, weeks, off):
        return hc(Decimal(str(rules["weekHC"])) * rate * weeks * (1 - off))

    def lock_plan(self, rules, lock, rate, count):
        for weeks in range(rules["maxWeeksAhead"], 0, -1):
            off = self.discount(rules, count + weeks)
            price = self.week_price(rules, rate, weeks, off)
            if price <= lock:
                return weeks, off, price
        return None

    def held_discount(self, rules, account, now):
        pinned = {b["fleet"] for b in self.st.bookings.values() if b["account"] == account and b["kind"] == "week" and b["ends"] > now}
        weeks, fleet = max(((self.weeks_left(account, fleet_id, now)[0], fleet_id) for fleet_id in pinned), default=(0, None))
        return weeks, fleet, self.discount(rules, weeks)

    def hour_discount(self, rules, account, now):
        weeks, fleet, off = self.held_discount(rules, account, now)
        found = {"discount": off, "weeks": weeks, "basis": "held" if weeks else None, "fleet": fleet}
        waiting = self.lock_plan(rules, self.st.locked(account), Decimal(1), 0) if account in self.st.locks else None
        if waiting and waiting[1] > off:
            found = {"discount": waiting[1], "weeks": waiting[0], "basis": "waitlist", "fleet": None}
        return found

    def waitlist_repay(self, account):
        lock = self.st.locks.get(account)
        if not lock:
            return Decimal(0), []
        joined = stamp_time(lock["joined"])
        mine = sorted((b for b in self.st.bookings.values() if b["account"] == account and b.get("waitlist_saving") and stamp_time(b["now"]) >= joined), key=lambda b: b["booking"])
        return sum((Decimal(b["waitlist_saving"]) for b in mine), Decimal(0)), [b["booking"] for b in mine]

    def discount_view(self, found):
        return dict(found, discount=format(found["discount"], "f"))

    def check_hour_fleet(self, pricing, account, start, end, now):
        rules = pricing["bookings"]
        gate = self.gate(pricing, "hoursFrom")
        if gate and start < gate:
            raise Refusal(f"Hour Fleets open on {pricing['hoursFrom']}.", 409, "not_open")
        if end > now + DAY * rules["hourFleetDaysAhead"]:
            raise Refusal(f"Hours on Hour Fleets can be booked at most {rules['hourFleetDaysAhead']} days ahead.", 409, "too_far")
        listed = {f["id"] for f in pricing["fleets"] if f["hourFleet"]}
        mine = [b for b in self.st.bookings.values() if b["account"] == account and b["fleet"] in listed]
        week = week_start(start)
        while week < end:
            used = sum((overlap(b["begins"], b["ends"], week, week + WEEK) for b in mine), timedelta(0))
            if used + overlap(start, end, week, week + WEEK) > HOUR * rules["hourFleetHoursPerWeek"]:
                raise Refusal(f"An account may hold at most {rules['hourFleetHoursPerWeek']} hours on Hour Fleets in a Week; it holds {used // HOUR} in the Week of {kst_text(week)[:10]}.", 409, "hour_cap")
            week += WEEK

    def check_hours(self, pricing, account, fleet, start, end, now):
        if not fleet["serving"]:
            raise Refusal(f"{fleet['name']} is not serving.", 409, "not_serving")
        if start <= now:
            raise Refusal("Choose an hour that has not started.", 409, "started")
        if fleet["hourFleet"]:
            self.check_hour_fleet(pricing, account, start, end, now)
        elif end > week_start(now) + WEEK:
            raise Refusal(f"Hours on {fleet['name']} can be booked only to the end of the current Week, {kst_text(week_start(now) + WEEK)}.", 409, "too_far")
        taken = self.busy(fleet["id"], start, end)
        if any(b["kind"] == "week" for b in taken):
            raise Refusal(f"{fleet['name']} is pinned to a customer this Week.", 409, "pinned")
        if taken:
            raise Refusal(f"{fleet['name']} is already booked at {kst_text(max(taken[0]['begins'], start))}.", 409, "taken")
        if self.maintained(fleet["id"], start, end):
            raise Refusal(f"{fleet['name']} is under maintenance in those hours.", 409, "maintenance")

    def hour_plan(self, pricing, account, fleet, start, hours, now):
        end = start + HOUR * hours
        self.check_hours(pricing, account, fleet, start, end, now)
        rules = pricing["bookings"]
        found = self.hour_discount(rules, account, now)
        base = Decimal(hours) * Decimal(str(rules["hourHC"])) * fleet["rate"]
        price = hc(base * (1 - found["discount"]))
        plan = {"fleet": fleet["id"], "start": slot_text(start), "end": slot_text(end), "hours": hours, "discount": format(found["discount"], "f"), "weeks": found["weeks"],
                "basis": found["basis"], "price": text_hc(price)}
        if found["basis"] == "waitlist":
            plan["waitlist_saving"] = text_hc(hc(base * (1 - self.held_discount(rules, account, now)[2])) - price)
        return plan

    def check_first(self, pricing, fleet, first, now):
        gate = self.gate(pricing, "weeksFrom")
        if gate and first < gate:
            raise Refusal(f"Weeks can be booked from the Week of {pricing['weeksFrom']}.", 409, "not_open")
        held = [b for b in self.busy(fleet["id"], first - WEEK, first) if b["kind"] == "week"]
        if held and now < first - DAY:
            raise Refusal(f"{fleet['name']} is pinned to a customer, who may book its next Week until {kst_text(first - DAY)}.", 409, "pinned")
        week = Decimal(str(pricing["bookings"]["weekHC"])) * fleet["rate"]
        if any(self.st.locked(account) >= week for account in self.st.locks):
            raise Refusal("Customers on the waitlist come first: join the waitlist, which allocates each available Fleet every Sunday at 00:00 KST.", 409, "waitlist")

    def pair_of(self, pricing, fleet):
        return next((f for f in self.fleets(pricing) if f["id"] == fleet.get("pair")), None) if fleet.get("pair") else None

    def pair_room(self, pricing, monday, fleet_id=None):
        rules = pricing["bookings"]
        serving = [f for f in self.fleets(pricing) if f["serving"]]
        listed = [f for f in serving if f["hourFleet"]]
        pinned = sum(1 for f in listed if f["id"] != fleet_id and any(b["kind"] == "week" for b in self.busy(f["id"], monday, monday + WEEK)))
        limit = max(0, len(listed) - math.ceil(Decimal(str(rules["openHourShare"])) * len(serving)))
        return max(0, limit - pinned), limit

    def pair_plan(self, pricing, account, fleet, weeks, now):
        rules = pricing["bookings"]
        partner = self.pair_of(pricing, fleet)
        if not partner or not fleet["serving"]:
            raise Refusal(f"{fleet['name']} is booked by the hour" + (f"; only the holder of {partner['name']} may pin it, as its pair." if partner else "."), 409, "not_weekly")
        current = week_start(now)
        held, held_through = self.weeks_left(account, partner["id"], now)
        count, through = self.weeks_left(account, fleet["id"], now)
        if not held_through:
            raise Refusal(f"Only the holder of {partner['name']} may pin {fleet['name']}, as its pair.", 409, "pair_not_held")
        first = through or current + WEEK
        if not through:
            gate = self.gate(pricing, "weeksFrom")
            if gate and first < gate:
                raise Refusal(f"Weeks can be booked from the Week of {pricing['weeksFrom']}.", 409, "not_open")
            while first < held_through and self.busy(fleet["id"], first, first + WEEK):
                first += WEEK
        end = first + WEEK * weeks
        if end > held_through:
            raise Refusal(f"{fleet['name']} is pinned only in Weeks you hold {partner['name']}, now through {kst_text(held_through)}.", 409, "pair_not_held")
        taken = self.busy(fleet["id"], first, end)
        if taken:
            raise Refusal(f"{fleet['name']} is already booked for the Week of {kst_text(max(taken[0]['begins'], first))[:10]}.", 409, "taken")
        week = first
        while week < end:
            if not self.pair_room(pricing, week, fleet["id"])[0]:
                raise Refusal(f"At least {Decimal(str(rules['openHourShare'])) * 100:.0f}% of the serving Fleets stay open for hours in every Week, so no other pair can be pinned in the Week of {kst_text(week)[:10]}.", 409, "pair_limit")
            week += WEEK
        off = self.discount(rules, held)
        return {"fleet": fleet["id"], "weeks": weeks, "first": slot_text(first), "last": slot_text(end - WEEK), "end": slot_text(end),
                "count": held, "discount": format(off, "f"), "price": text_hc(self.week_price(rules, fleet["rate"], weeks, off)), "extends": bool(through), "pair": partner["id"]}

    def week_plan(self, pricing, account, fleet, weeks, now):
        rules = pricing["bookings"]
        if fleet["hourFleet"]:
            return self.pair_plan(pricing, account, fleet, weeks, now)
        if not fleet["serving"]:
            raise Refusal(f"{fleet['name']} is not booked by the Week: choose a serving Fleet that is not an Hour Fleet.", 409, "not_weekly")
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
                "count": count + weeks, "discount": format(off, "f"), "price": text_hc(self.week_price(rules, fleet["rate"], weeks, off)), "extends": bool(through)}

    def pay(self, account, fleet, kind, start, end, price, now, extra):
        if self.st.balance(account, stamp(now)) < price:
            raise Refusal(f"This booking costs {text_hc(price)} HC, more than the account's unlocked HC. Buy HC first.", 402, "insufficient_hc")
        first = kind == "week" and not self.weeks_left(account, fleet["id"], now)[1]
        data = dict(extra, booking=new_id("BK"), account=account, fleet=fleet["id"], kind=kind, start=slot_text(start), end=slot_text(end), hc=text_hc(price), rate=text_hc(fleet["rate"]), source="hc", now=stamp(now))
        self.book.append("booking", data)
        if first:
            self.default_extend(account, fleet["id"], now)
        return data["booking"]

    def default_extend(self, account, fleet_id, now):
        if (account, fleet_id) not in self.st.extend:
            self.book.append("auto_extend", {"account": account, "fleet": fleet_id, "weeks": DEFAULT_WEEKS_AHEAD, "default": True, "now": stamp(now)})

    def hours_request(self, body):
        pricing = self.settings()
        return pricing, self.fleet_field(pricing, body.get("fleet")), slot_field(body.get("start"), "start"), count_field(body.get("hours"), "number of hours", 168)

    def quote_hours(self, account, body):
        pricing, fleet, start, hours = self.hours_request(body)
        with self.book.lock:
            now = self.now()
            self.advance(now)
            return self.hour_plan(pricing, account, fleet, start, hours, now)

    def book_hours(self, account, body):
        pricing, fleet, start, hours = self.hours_request(body)
        with self.book.lock:
            now = self.now()
            self.advance(now)
            plan = self.hour_plan(pricing, account, fleet, start, hours, now)
            ident = self.pay(account, fleet, "hour", start, slot_time(plan["end"]), Decimal(plan["price"]), now, {key: plan[key] for key in ("hours", "discount", "waitlist_saving") if key in plan})
            self.confirmed(self.st.bookings[ident], now)
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
            self.confirmed(self.st.bookings[ident], now)
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
            before, joining, places = self.st.locked(account), account not in self.st.locks, self.positions()
            self.book.append("lock", {"account": account, "hc": text_hc(amount), "now": stamp(now)})
            found = {"locked": text_hc(self.st.locked(account)), "position": self.position(account)}
            waiting = len(self.st.locks)
            self.moved(places, now, {account})
        what = f"joined the waitlist with {text_hc(amount)} HC" if joining else f"added {text_hc(amount)} HC to its lock of {text_hc(before)} HC"
        company = self.book.accounts.get(account, {}).get("company", account)
        self.app.mail(self.app.cfg.notify, f"Waitlist lock: {company}, {found['locked']} HC (+{text_hc(amount)})",
                      f"{self.app.who(account)} {what}: the lock is now {found['locked']} HC (+{text_hc(amount)} HC), position {found['position']} of {waiting} on the waitlist, at {kst_text(now)}.")
        return found

    def leave(self, account):
        with self.book.lock:
            now = self.now()
            self.advance(now)
            if account not in self.st.locks:
                raise Refusal("This account is not on the waitlist.", 404)
            places = self.positions()
            owed, refs = self.waitlist_repay(account)
            self.book.append("unlock", {"account": account, "hc": text_hc(self.st.locked(account)), "reason": "leave", "now": stamp(now)})
            repaid = hc(min(owed, self.st.balance(account, stamp(now))))
            if repaid > 0:
                self.book.append("hc_debit", {"account": account, "hc": text_hc(repaid), "reason": "waitlist_discount", "ref": ",".join(refs), "now": stamp(now)})
            self.moved(places, now, {account})
        return {"locked": text_hc(0), "position": None, "repaid": text_hc(repaid)}

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
        notes, places = [], self.positions()
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
        self.moved(places, stamp_time(moment), set())
        if notes:
            self.app.mail(self.app.cfg.notify, "Refund due: locked HC lapsed", "\n".join(notes + ["", "Toss Payments: sparkpipe_pay.py refund ORDER 16.4 AMOUNT REASON. SWIFT: the same command records it; then send it back by SWIFT to the account it came from. CoinPayments: refund it in CoinPayments."]))

    def round_at(self, now):
        if not self.st.locks:
            return None
        since = min(stamp_time(lock["joined"]) for lock in self.st.locks.values())
        if self.st.round:
            since = max(since, stamp_time(self.st.round))
        found = sunday_after(since)
        return found if found <= now else None

    def affordable(self, rules, account, fleet, sunday):
        return self.lock_plan(rules, self.st.locked(account), fleet["rate"], self.weeks_left(account, fleet["id"], sunday)[0])

    def allocate(self, sunday, pricing):
        monday, waiting, places, granted = sunday + DAY, self.queue_order(), self.positions(), set()
        for fleet in self.fleets(pricing):
            if fleet["hourFleet"] or not fleet["serving"] or self.busy(fleet["id"], monday, monday + WEEK):
                continue
            for account in waiting:
                found = self.affordable(pricing["bookings"], account, fleet, sunday)
                if found:
                    self.grant(account, fleet, found, sunday)
                    waiting.remove(account)
                    granted.add(account)
                    break
        self.book.append("round", {"sunday": stamp(sunday)})
        self.moved(places, sunday, granted)

    def grant(self, account, fleet, found, sunday):
        weeks, off, price = found
        monday, left = sunday + DAY, self.st.locked(account) - price
        data = {"booking": new_id("BK"), "account": account, "fleet": fleet["id"], "kind": "week", "start": slot_text(monday), "end": slot_text(monday + WEEK * weeks),
                "hc": text_hc(price), "rate": text_hc(fleet["rate"]), "weeks": weeks, "discount": format(off, "f"), "source": "waitlist", "unlocked": text_hc(left), "now": stamp(sunday)}
        self.book.append("booking", data)
        self.default_extend(account, fleet["id"], sunday)
        self.app.audit(account, "system", "fleet.allocated", {"booking": data["booking"], "fleet": fleet["id"], "first": data["start"], "weeks": weeks, "hc": data["hc"]}, sunday)
        self.notify(account, "fleet.allocated", {key: data[key] for key in ("booking", "fleet", "start", "end", "weeks", "hc", "discount", "unlocked")}, sunday,
                    f"SparkPipe {fleet['name']} is yours from {kst_text(monday)[:10]}", "\n".join([
                        f"The waitlist allocated {fleet['name']} to you: {weeks} Weeks from {kst_text(monday)} to {kst_text(monday + WEEK * weeks)}, for {text_hc(price)} HC of your lock" + (f" at {off * 100:.0f}% off." if off else "."),
                        f"{text_hc(left)} HC of your lock is unlocked and back in your balance." if left > 0 else "Your whole lock paid for those Weeks.",
                        self.extend_line(account, fleet["id"]),
                        f"Without it, book the next Week by {kst_text(monday + WEEK * weeks - DAY)} to keep the Fleet.",
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
        name = self.fleet_name(found["fleet"])
        for account, credits in sorted(owed.items()):
            self.notify(account, "downtime.credited", {"maintenance": ident, "fleet": found["fleet"], "start": found["start"], "hours": found["hours"], "hc": text_hc(credits)}, found["ends"],
                        f"SparkPipe credited {text_hc(credits)} HC for maintenance on {name}",
                        f"Maintenance on {name} from {kst_text(found['begins'])} to {kst_text(found['ends'])} overlapped your booking. We credited {text_hc(credits)} HC to your account (clause 4.4).")

    def outage_at(self, now):
        found = []
        for span in self.app.uptime.spans(now - 30 * DAY):
            begins, ends = stamp_time(span["start"]), stamp_time(span["end"]) if span["end"] else None
            ends = ends if ends and ends <= now else None
            if (ends or now) - begins <= OUTAGE:
                continue
            done = self.st.downtime.get(span["span"])
            if not done:
                at = ends or begins + OUTAGE + timedelta(seconds=1)
            elif ends and not done["end"]:
                at = ends
            elif not ends and stamp_time(done["now"]) + OUTAGE_PASS <= now:
                at = stamp_time(done["now"]) + OUTAGE_PASS
            else:
                continue
            if at <= now:
                found.append((at, span["span"], span, ends))
        return min(found, key=lambda item: item[:2]) if found else None

    def credit_outage(self, span, at, ends):
        fleet_id, begins, first = span["fleet"], stamp_time(span["start"]), span["span"] not in self.st.downtime
        through, moment, owed = slot_ceil(ends) if ends else slot_floor(at) + HOUR, slot_floor(begins), {}
        while moment < through:
            if (fleet_id, slot_text(moment)) not in self.st.downtime_hours and not self.maintained(fleet_id, moment, moment + HOUR):
                for b in self.busy(fleet_id, moment, moment + HOUR):
                    found = owed.setdefault(b["account"], [Decimal(0), []])
                    found[0] += Decimal(b["rate"])
                    found[1].append(slot_text(moment))
            moment += HOUR
        lots = {account: self.lot(account, credits, "downtime", span["span"], at, extra={"fleet": fleet_id, "hours": hours}) for account, (credits, hours) in sorted(owed.items())}
        name, period = self.fleet_name(fleet_id), f"from {kst_text(begins)} to {kst_text(ends)}" if ends else f"since {kst_text(begins)}"
        for account, lot in lots.items():
            listed = ", ".join(kst_text(slot_time(hour))[11:16] for hour in lot["hours"])
            self.notify(account, "downtime.credited", {"downtime": span["span"], "fleet": fleet_id, "start": span["start"], "end": stamp(ends) if ends else None, "hours": lot["hours"], "hc": lot["hc"], "lot": lot["lot"], "clause": "4.4"}, at,
                        f"SparkPipe credited {lot['hc']} HC for downtime on {name}",
                        f"{name} was unavailable or unreachable {period}{'' if ends else ', and we are still on it'}. We credited {lot['hc']} HC for {len(lot['hours'])} booked hour{'s' if len(lot['hours']) != 1 else ''} it touched ({listed} KST), at the Fleet's Rate (clause 4.4). "
                        f"The HC lapses on {lot['expires'][:10]}. Our uptime records: {self.app.cfg.public}/status.html", once=f"downtime:{span['span']}:{account}:{slot_text(through)}")
        self.book.append("downtime", {"downtime": span["span"], "fleet": fleet_id, "start": span["start"], "end": stamp(ends) if ends else None, "cause": span["cause"], "through": slot_text(through),
                                      "credited": [[account, lot["hc"]] for account, lot in lots.items()], "now": stamp(at)})
        if first or ends:
            credited = "; ".join(f"{self.app.who(account)} {lot['hc']} HC" for account, lot in lots.items()) or "no booked hours"
            self.app.mail(self.app.cfg.notify, f"{'Fleet down' if not ends else 'Fleet back'}: {fleet_id} {period}", f"{name} was unavailable or unreachable {period} ({span['cause']}). Credited under clause 4.4: {credited}.")

    def fleet_down(self, fleet_id):
        entry = self.app.uptime.entry(fleet_id)
        return bool(entry and entry.get("open"))

    def fleet_up(self, fleet_id):
        entry = self.app.uptime.entry(fleet_id)
        return not (entry and entry.get("open")) if self.sandbox else bool(entry and not entry.get("open"))

    def active_failovers(self):
        return [f for f in self.st.failovers.values() if f["ends"] is None]

    def free_now(self, fleet_id, now):
        return not self.busy(fleet_id, now, now + timedelta(seconds=1)) and not self.maintained(fleet_id, now, now + timedelta(seconds=1))

    def route(self, account, fleet_id):
        with self.book.lock:
            now = self.now()
            self.advance(now)
            if not self.fleet_down(fleet_id):
                return fleet_id, None
            current = next((f for f in self.active_failovers() if f["account"] == account and f["from"] == fleet_id), None)
            if current:
                return current["to"], fleet_id
            taken = {f["to"] for f in self.st.failovers.values() if f["account"] != account and (f["ends"] is None or f["ends"] > slot_floor(now))}
            choices = sorted((f for f in self.fleets(self.settings()) if f["id"] != fleet_id and f["serving"] and f["id"] not in taken and self.fleet_up(f["id"]) and self.free_now(f["id"], now)), key=lambda f: f["hourFleet"])
            if not choices:
                raise Refusal(f"{self.fleet_name(fleet_id)} is down and no free Fleet can serve you now; try again in a minute. Every booked hour a period over 15 minutes touches is credited (clause 4.4).", 503, "fleet_down")
            target, ident = choices[0], new_id("FO")
            self.book.append("failover", {"failover": ident, "account": account, "from": fleet_id, "to": target["id"], "start": stamp(now), "now": stamp(now)})
            held, name = self.fleet_name(fleet_id), self.fleet_name(target["id"])
            self.notify(account, "fleet.failover", {"failover": ident, "state": "started", "from": fleet_id, "to": target["id"], "start": stamp(now)}, now, f"SparkPipe serves you on {name} while {held} is down",
                        f"{held}, which you hold, is down. From {kst_text(now)} we serve your requests on {name}, which is free, without your working state, so the first requests there prefill again (clause 15.7). "
                        f"Every booked hour of {held} that the period touches is still credited (clause 4.4). Your requests go back to {held} as soon as it serves again. {self.app.cfg.public}/status.html")
            return target["id"], fleet_id

    def failover_left(self, target, now):
        found = [b["begins"] for b in self.on_fleet(target) if b["begins"] > now] + [m["begins"] for m in self.st.maintenance.values() if m["fleet"] == target and m["begins"] > now]
        return (min(found) - now).total_seconds() + GRACE_SECONDS if found else 7 * 86400

    def hold_end(self, account, fleet_id, moment):
        mine, end = [b for b in self.on_fleet(fleet_id) if b["account"] == account], moment
        while any(b["begins"] <= end < b["ends"] for b in mine):
            end = max(b["ends"] for b in mine if b["begins"] <= end < b["ends"])
        return end

    def failover_end_at(self, now):
        found = []
        for f in self.active_failovers():
            later = [b["begins"] for b in self.on_fleet(f["to"]) if f["begins"] < b["ends"] and b["begins"] <= now] + [m["begins"] for m in self.st.maintenance.values() if m["fleet"] == f["to"] and f["begins"] < m["ends"] and m["begins"] <= now]
            held = self.hold_end(f["account"], f["from"], f["begins"])
            if not self.fleet_down(f["from"]):
                found.append((now, f["failover"], "recovered"))
            elif later:
                found.append((max(min(later), f["begins"]), f["failover"], "taken"))
            elif held <= now:
                found.append((max(held, f["begins"]), f["failover"], "hold_ended"))
            elif not self.fleet_up(f["to"]):
                found.append((now, f["failover"], "down"))
        return min(found) if found else None

    def end_failover(self, at, ident, reason):
        f = self.st.failovers[ident]
        self.book.append("failover_end", {"failover": ident, "end": stamp(at), "reason": reason, "now": stamp(at)})
        held, name = self.fleet_name(f["from"]), self.fleet_name(f["to"])
        why = {"recovered": f"{held} serves again, and your requests go back to it", "taken": f"{name} is no longer free; while {held} is down your next request goes to another free Fleet if there is one",
               "hold_ended": f"your hold of {held} ended", "down": f"{name} is down too; while {held} is down your next request goes to another free Fleet if there is one"}[reason]
        self.notify(f["account"], "fleet.failover", {"failover": ident, "state": "ended", "from": f["from"], "to": f["to"], "start": f["start"], "end": stamp(at), "reason": reason}, at,
                    f"SparkPipe stopped serving you on {name}", f"We stopped serving your requests on {name} at {kst_text(at)}: {why}.")

    def status_page(self):
        pricing = self.settings()
        with self.book.lock:
            now = self.now()
            self.advance(now)
            spans = self.app.uptime.spans(now - (UPTIME_WINDOWS[-1][1] + 1) * DAY)
            fleets = [self.fleet_status(listed, fleet, [s for s in spans if s["fleet"] == fleet["id"]], now) for listed, fleet in zip(pricing["fleets"], self.fleets(pricing))]
        return {"now": stamp(now), "outageMinutes": OUTAGE // timedelta(minutes=1), "fleets": fleets}

    def fleet_status(self, listed, fleet, spans, now):
        entry = self.app.uptime.entry(fleet["id"])
        closing = [m for m in self.maintained(fleet["id"], now, now + timedelta(seconds=1))]
        opened = next((s for s in spans if not s["end"]), None)
        if not fleet["serving"]:
            status, since = "not_serving", None
        elif closing:
            status, since = "maintenance", slot_text(closing[0]["begins"])
        elif opened:
            status, since = "down", opened["start"]
        elif entry:
            status, since = "up", max([s["end"] for s in spans] + [entry["first"]])
        else:
            status, since = "unknown", None
        first = stamp_time(entry["first"]) if entry else None
        uptime = {}
        for label, days in UPTIME_WINDOWS:
            start = max(now - DAY * days, first) if first else None
            if not start or start >= now:
                uptime[label] = None
                continue
            down = timedelta(0)
            for s in spans:
                begins, ends = max(stamp_time(s["start"]), start), stamp_time(s["end"]) if s["end"] else now
                down += overlap(begins, ends, start, now) - sum((overlap(max(begins, m["begins"]), min(ends, m["ends"]), start, now) for m in self.maintained(fleet["id"], begins, ends)), timedelta(0))
            uptime[label] = round(100 * (1 - down / (now - start)), 3)
        incidents = [{"start": s["start"], "end": s["end"], "minutes": int(((stamp_time(s["end"]) if s["end"] else now) - stamp_time(s["start"])) // timedelta(minutes=1)), "cause": s["cause"],
                      "creditable": ((stamp_time(s["end"]) if s["end"] else now) - stamp_time(s["start"])) > OUTAGE} for s in sorted(spans, key=lambda s: s["start"], reverse=True)]
        return {"id": fleet["id"], "name": listed["name"], "serving": fleet["serving"], "hourFleet": listed["hourFleet"], "status": status, "since": since, "recordsSince": entry["first"] if entry else None,
                "uptime": uptime, "incidents": incidents, "maintenance": merged((m["begins"], m["ends"]) for m in self.maintained(fleet["id"], now, now + 60 * WEEK))}

    def pins(self, now):
        found = {}
        for b in self.st.bookings.values():
            if b["kind"] == "week" and b["ends"] > now:
                key = (b["account"], b["fleet"])
                found[key] = max(found.get(key, b["ends"]), b["ends"])
        return found

    def deadline_at(self, now):
        found = []
        for (account, fleet_id), through in sorted(self.pins(now).items()):
            deadline = through - DAY
            marks = [hours for hours in DEADLINE_HOURS if deadline - HOUR * hours <= now < deadline]
            if marks:
                once = f"deadline:{account}:{fleet_id}:{slot_text(deadline)}:{min(marks)}"
                if not self.app.notices.sent(once):
                    found.append((deadline - HOUR * min(marks), account, fleet_id, through, min(marks), once))
        return min(found) if found else None

    def deadline(self, pricing, at, account, fleet_id, through, hours, once):
        fleet, deadline = self.fleet_field(pricing, fleet_id), through - DAY
        data = {"fleet": fleet_id, "deadline": slot_text(deadline), "through": slot_text(through), "hours": hours, "balance": text_hc(self.st.balance(account, stamp(at)))}
        try:
            data["next_week_hc"] = self.week_plan(pricing, account, fleet, 1, at)["price"]
        except Refusal:
            pass
        setting = self.st.extend.get((account, fleet_id))
        data["auto_extend"] = setting["weeks"] if setting else 0
        if self.extend_waiting(account, fleet_id):
            data["waiting"] = self.extend_waiting(account, fleet_id)
        cost = f" It costs {data['next_week_hc']} HC; your unlocked balance is {data['balance']} HC." if "next_week_hc" in data else ""
        self.notify(account, "booking.deadline", data, at, f"Book the next Week of {fleet['name']} by {kst_text(deadline)}",
                    f"Your last booked Week of {fleet['name']} ends at {kst_text(through)}. Book the next Week by {kst_text(deadline)} to keep the Fleet; after that the waitlist may be allocated it (clause 15).{cost} {self.extend_line(account, fleet_id)}", once)

    def extend_line(self, account, fleet_id):
        setting, page = self.st.extend.get((account, fleet_id)), f"{self.app.cfg.public}/market.html"
        if not setting:
            return f"Auto-extend is off for this Fleet: turn it on on your account page to book the next Week from your HC without anyone having to remember: {page}"
        waits = {"terms": "until you accept the current Customer Agreement", "insufficient_hc": "until your unlocked HC covers the next Week"}
        ahead = f"{setting['weeks']} Week{'s' if setting['weeks'] != 1 else ''}"
        if self.extend_waiting(account, fleet_id) in waits:
            return f"Auto-extend is on, keeping {ahead} booked ahead, but it waits {waits[self.extend_waiting(account, fleet_id)]}: {page}"
        return f"Auto-extend is on: it keeps {ahead} booked ahead from your unlocked HC, so you keep the Fleet. Turn it off or change it on your account page: {page}"

    def extend_waiting(self, account, fleet_id):
        setting = self.st.extend.get((account, fleet_id))
        through = max((b["ends"] for b in self.on_fleet(fleet_id) if b["account"] == account and b["kind"] == "week"), default=None)
        return setting.get("waiting") if setting and through and setting.get("through") == slot_text(through) else None

    def extend_at(self, now, pricing):
        found = []
        for account, fleet_id in sorted(self.st.extend):
            step = self.extend_step(now, pricing, account, fleet_id)
            if step:
                found.append((step[0], account, fleet_id, step[1]))
        return min(found, key=lambda item: item[:3])[::3] if found else None

    def extend_step(self, now, pricing, account, fleet_id):
        setting = self.st.extend[(account, fleet_id)]
        through = max((b["ends"] for b in self.on_fleet(fleet_id) if b["account"] == account and b["kind"] == "week"), default=None)
        if not through or through <= stamp_time(setting["now"]):
            return now, lambda: self.stop_extend(account, fleet_id, "not_pinned", now)
        due = max(through - WEEK * setting["weeks"], stamp_time(setting["now"]))
        if due > now:
            return None
        waiting = setting.get("waiting") if setting.get("through") == slot_text(through) else None
        if waiting and through <= now:
            return through, lambda: self.stop_extend(account, fleet_id, waiting, through)
        at = now if waiting else due
        fleet = self.fleet_field(pricing, fleet_id)
        weeks = max(1, (week_start(at) + WEEK * (1 + setting["weeks"]) - through) // WEEK)
        try:
            plan = self.week_plan(pricing, account, fleet, weeks, at)
        except Refusal as error:
            if error.reason == "pair_not_held":
                return (through, lambda: self.stop_extend(account, fleet_id, "pair_not_held", through)) if through <= now else None
            reason = error.reason or "refused"
            return at, lambda: self.stop_extend(account, fleet_id, reason, at)
        blocked = None if self.app.terms_current(account) else "terms"
        if not blocked and self.st.balance(account, stamp(at)) < Decimal(plan["price"]):
            blocked = "insufficient_hc"
        if blocked and blocked == waiting:
            return None
        return at, lambda: self.extend(account, fleet, plan, at, blocked)

    def extend(self, account, fleet, plan, at, blocked):
        if blocked:
            self.book.append("auto_extend_wait", {"account": account, "fleet": fleet["id"], "through": plan["first"], "reason": blocked, "now": stamp(at)})
            balance, first = self.st.balance(account, stamp(at)), slot_time(plan["first"])
            data = {"fleet": fleet["id"], "deadline": slot_text(first - DAY), "through": plan["first"], "early": True, "reason": blocked, "weeks": plan["weeks"], "price": plan["price"], "balance": text_hc(balance)}
            if blocked == "terms":
                return self.notify(account, "booking.deadline", data, at, f"Auto-extend on {fleet['name']} waits for you to accept the new Customer Agreement",
                                   f"Auto-extend could not book {plan['weeks']} more Weeks of {fleet['name']} from {kst_text(first)}: the Customer Agreement has changed, and we accept no Booking until you accept the new version (clause 9.1). "
                                   f"Accept it on your account page before {kst_text(first - DAY)} to keep the Fleet: {self.app.cfg.public}/market.html")
            need = Decimal(plan["price"]) - balance
            return self.notify(account, "booking.deadline", dict(data, hc_needed=text_hc(need)), at, f"Auto-extend on {fleet['name']} needs {text_hc(need)} more HC",
                               f"Auto-extend could not book {plan['weeks']} more Weeks of {fleet['name']} from {kst_text(first)}: they cost {plan['price']} HC and your unlocked balance is {text_hc(balance)} HC. "
                               f"Buy {text_hc(need)} HC before {kst_text(first - DAY)} to keep the Fleet: {self.app.cfg.public}/market.html")
        ident = self.pay(account, fleet, "week", slot_time(plan["first"]), slot_time(plan["end"]), Decimal(plan["price"]), at, {"weeks": plan["weeks"], "discount": plan["discount"], "auto": True})
        self.confirmed(self.st.bookings[ident], at)
        self.app.audit(account, "system", "booking.auto", {"booking": ident, "fleet": fleet["id"], "first": plan["first"], "weeks": plan["weeks"], "price": plan["price"]}, at)

    def stop_extend(self, account, fleet_id, reason, at):
        through = max((b["ends"] for b in self.on_fleet(fleet_id) if b["account"] == account and b["kind"] == "week"), default=None)
        self.book.append("auto_extend", {"account": account, "fleet": fleet_id, "weeks": 0, "reason": reason, "now": stamp(at)})
        why = {"not_pinned": "the Weeks it kept have ended", "taken": "someone else booked the next Week after your deadline passed", "terms": "your last booked Week ended while it waited for you to accept the current Customer Agreement",
               "insufficient_hc": "your last booked Week ended while your unlocked HC did not cover the next one", "not_weekly": "the Fleet stopped serving by the Week", "too_far": "the next Week is more than 52 Weeks ahead",
               "pair_not_held": "you no longer hold the next Week of its pair", "pair_limit": "the Hour Fleets that stay open for hours left no pair free"}.get(reason, reason)
        data = {"fleet": fleet_id, "stopped": True, "reason": reason, "through": slot_text(through) if through else None, "deadline": slot_text(through - DAY) if through else None, "auto_extend": 0}
        name, page = self.fleet_name(fleet_id), f"{self.app.cfg.public}/market.html"
        keep = f" Book the next Week by {kst_text(through - DAY)} to keep the Fleet, or turn auto-extend on again." if through and through - DAY > at else " Book Weeks yourself, or turn auto-extend on again while you hold the Fleet."
        self.notify(account, "booking.deadline", data, at, f"SparkPipe auto-extend stopped on {name}", f"Auto-extend on {name} stopped at {kst_text(at)}: {why}.{keep} {page}", mail=True)

    def auto_extend(self, account, body):
        pricing = self.settings()
        fleet = self.fleet_field(pricing, body.get("fleet"))
        weeks = count_field(body.get("weeks_ahead"), "number of Weeks ahead", pricing["bookings"]["maxWeeksAhead"], 0)
        with self.book.lock:
            now = self.now()
            self.advance(now)
            if weeks and not self.weeks_left(account, fleet["id"], now)[1]:
                raise Refusal(f"Auto-extend keeps Weeks you hold: book a Week of {fleet['name']} first.", 409, "not_pinned")
            self.book.append("auto_extend", {"account": account, "fleet": fleet["id"], "weeks": weeks, "now": stamp(now)})
            self.advance(now)
            through = self.weeks_left(account, fleet["id"], now)[1]
        return {"fleet": fleet["id"], "weeks_ahead": weeks, "through": slot_text(through) if through else None}

    def advance(self, now):
        pricing = self.settings()
        while True:
            lapse, sunday, done = self.lapse_at(now), self.round_at(now), self.maintenance_at(now)
            due, extension, outage, moved = self.deadline_at(now), self.extend_at(now, pricing), self.outage_at(now), self.failover_end_at(now)
            events = []
            if lapse:
                events.append((stamp_time(lapse), 0, lambda: self.lapse(lapse, pricing)))
            if sunday:
                events.append((sunday, 1, lambda: self.allocate(sunday, pricing)))
            if done:
                events.append((done[0], 2, lambda: self.credit_maintenance(done[1])))
            if outage:
                events.append((outage[0], 3, lambda: self.credit_outage(outage[2], outage[0], outage[3])))
            if moved:
                events.append((moved[0], 4, lambda: self.end_failover(*moved)))
            if extension:
                events.append((extension[0], 5, extension[1]))
            if due:
                events.append((due[0], 6, lambda: self.deadline(pricing, *due)))
            if not events:
                return
            min(events, key=lambda event: event[:2])[2]()

    def tick(self):
        with self.book.lock:
            self.advance(self.now())

    def run(self, log):
        while True:
            time.sleep(30)
            try:
                self.tick()
            except Exception as error:
                log(f"market tick: {error}")

    def lot(self, account, credits, source, ref, now, note=None, extra=None):
        data = dict(extra or {}, lot=new_id("LT"), account=account, hc=text_hc(credits), expires=stamp(months_later(now, self.settings()["hc"]["validityMonths"])), source=source, ref=ref, now=stamp(now))
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

    def downtime_credit(self, account, credits, note):
        with self.book.lock:
            now = self.now()
            lot = self.lot(account, credits, "credit", new_id("CR"), now, note)
            self.notify(account, "downtime.credited", {"lot": lot["lot"], "hc": lot["hc"], "clause": "4.4", "note": note}, now, f"SparkPipe credited {lot['hc']} HC for downtime",
                        f"We credited {lot['hc']} HC to your account for hours a Fleet you held was unavailable or unreachable (clause 4.4): {note}. The HC lapses on {lot['expires'][:10]}.")
        return lot

    def key_options(self, pricing, body, now):
        bad = lambda text: Refusal(text, 400, "invalid_request")
        scopes = body.get("scopes", list(SCOPES))
        if not isinstance(scopes, list) or not scopes or any(scope not in SCOPES for scope in scopes):
            raise bad("Give scopes as a non-empty list of " + ", ".join(SCOPES) + ".")
        fleet = body.get("fleet")
        if fleet is not None and fleet not in [f["id"] for f in pricing["fleets"]]:
            raise bad("Give fleet as one of the Fleets, " + ", ".join(f["id"] for f in pricing["fleets"]) + ", or leave it out.")
        if not isinstance(body.get("allow_swap", False), bool):
            raise bad("Give allow_swap as true or false.")
        expires = body.get("expires")
        if expires is not None and stamp_field(expires, "expiry") <= now:
            raise bad("Give an expiry in the future, or leave it out.")
        ips = body.get("ips", [])
        try:
            if not isinstance(ips, list) or len(ips) > KEY_IPS or not all(isinstance(ip, str) for ip in ips):
                raise ValueError
            ips = [str(ipaddress.ip_network(ip.strip(), strict=False)) for ip in ips]
        except ValueError:
            raise bad(f"Give ips as a list of at most {KEY_IPS} addresses or networks, such as 203.0.113.7 or 2001:db8::/48.")
        priority = body.get("priority")
        if priority is not None and priority not in PRIORITIES:
            raise bad("Give priority as interactive or background, or leave it out.")
        note = body.get("note") or ""
        if not isinstance(note, str) or len(note) > 100 or not note.isprintable():
            raise bad("Give the note as one line of at most 100 characters.")
        return {"note": note.strip(), "scopes": [scope for scope in SCOPES if scope in scopes], "fleet": fleet, "allow_swap": body.get("allow_swap", False), "expires": expires, "ips": ips, "priority": priority}

    def key_view(self, record):
        found = {name: record.get(name) for name in ("key", "member", "note", "hint", "created", "revoked", "scopes", "fleet", "allow_swap", "expires", "ips", "priority", "retires", "rotated_from")}
        return dict(found, id=record["key"])

    def new_key(self, account, options, now, rotated_from=None, member=None):
        key = "sk-hc-" + secrets.token_urlsafe(24)
        found = new_id("KY")
        data = dict(options, key=found, account=account, member=member or account, hash=key_digest(key), hint=key[-4:], now=stamp(now))
        if rotated_from:
            data["rotated_from"] = rotated_from
        self.book.append("api_key", data)
        return dict(self.key_view(self.st.keys[found]), key=key)

    def create_key(self, account, body, member=None, role="owner"):
        pricing = self.settings()
        allowed = ROLE_SCOPES[role]
        body = body if "scopes" in body else dict(body, scopes=list(allowed))
        with self.book.lock:
            now = self.now()
            options = self.key_options(pricing, body, now)
            if any(scope not in allowed for scope in options["scopes"]) or (options["allow_swap"] and role == "member"):
                raise Refusal(f"A {role} of the account makes keys with the scopes {', '.join(allowed)} only, without allow_swap.", 403, "role")
            return self.new_key(account, options, now, member=member)

    def own_key(self, account, found, now, member=None, role="owner"):
        record = self.st.keys.get(found) if isinstance(found, str) else None
        if not record or record["account"] != account or record["revoked"] or (record["retires"] and stamp_time(record["retires"]) <= now) or (role == "member" and record["member"] != member):
            raise Refusal("No active key with that id on this account.", 404)
        return record

    def revoke_key(self, account, found, member=None, role="owner"):
        with self.book.lock:
            self.own_key(account, found, self.now(), member, role)
            self.book.append("api_key_revoke", {"key": found})
        return {"id": found, "revoked": True}

    def rotate_key(self, account, found, member=None, role="owner"):
        with self.book.lock:
            now = self.now()
            record = self.own_key(account, found, now, member, role)
            if record["retires"]:
                raise Refusal("This key was already rotated.", 409, "rotated")
            made = self.new_key(account, {name: record[name] for name in ("note", "scopes", "fleet", "allow_swap", "expires", "ips", "priority")}, now, found, record["member"])
            self.book.append("api_key_rotate", {"key": found, "account": account, "new": made["id"], "retires": stamp(now + HOUR)})
        return dict(made, replaces=found, retires=stamp(now + HOUR))

    def key_account(self, key):
        found = self.st.key_hashes.get(key_digest(key)) if key else None
        record = self.st.keys.get(found) if found else None
        if not record or record["revoked"]:
            raise Refusal("Send a valid SparkPipe API key as Authorization: Bearer sk-hc-... or X-Api-Key.", 401, "invalid_key")
        now = self.now()
        if record["retires"] and stamp_time(record["retires"]) <= now:
            raise Refusal("This key was rotated more than an hour ago: use the key that replaced it.", 401, "invalid_key")
        if record["expires"] and stamp_time(record["expires"]) <= now:
            raise Refusal("This key has expired.", 401, "invalid_key")
        return record

    def authorize(self, key, scope, ip):
        record = self.key_account(key)
        if record["ips"] and not address_in(ip, record["ips"]):
            raise Refusal("This key is not allowed from this address.", 403, "invalid_key")
        if scope not in record["scopes"]:
            raise Refusal(f"This key does not have the {scope} scope.", 403, "invalid_key")
        return record

    def held(self, account, now):
        serving = {f["id"] for f in self.fleets(self.settings()) if f["serving"]}
        return sorted({b["fleet"] for b in self.st.bookings.values() if b["account"] == account and b["begins"] <= now < b["ends"] and b["fleet"] in serving})

    def gateway_share(self, account):
        serving = {f["id"] for f in self.fleets(self.settings()) if f["serving"]}
        with self.book.lock:
            now = self.now()
            self.advance(now)
            active = [b for b in self.st.bookings.values() if b["begins"] <= now < b["ends"] and b["fleet"] in serving]
        return any(b["account"] == account for b in active), len({b["fleet"] for b in active})

    def holds(self, account, fleet):
        with self.book.lock:
            now = self.now()
            self.advance(now)
            return fleet in self.held(account, now)

    def hold_left(self, account, fleet_id):
        with self.book.lock:
            now = self.now()
            self.advance(now)
            mine = [b for b in self.on_fleet(fleet_id) if b["account"] == account]
            end = now
            while any(b["begins"] <= end < b["ends"] for b in mine):
                end = max(b["ends"] for b in mine if b["begins"] <= end < b["ends"])
            if end == now:
                end = max((b["ends"] for b in mine if b["ends"] <= now), default=now - timedelta(seconds=GRACE_SECONDS))
        return (end - now).total_seconds() + GRACE_SECONDS

    def admit_first(self, account, record, requested):
        if requested or (record and record["fleet"]):
            return self.admit(account, record, requested)
        with self.book.lock:
            now = self.now()
            self.advance(now)
            if not self.held(account, now):
                raise Refusal("This account holds no Fleet now. Book an Hour or a Week to send requests.", 403, "fleet_not_held")
        return None

    def admit(self, account, record, requested, also=()):
        if record and record["fleet"]:
            requested = record["fleet"]
        with self.book.lock:
            now = self.now()
            self.advance(now)
            held = self.held(account, now)
        for fleet in [found for found in (requested, *also) if found]:
            if fleet not in held:
                raise Refusal(f"This account does not hold {fleet} now" + (f"; it holds {', '.join(held)}." if held else "."), 403, "fleet_not_held")
        if requested:
            return requested
        if not held:
            raise Refusal("This account holds no Fleet now. Book an Hour or a Week to send requests.", 403, "fleet_not_held")
        if len(held) > 1:
            raise Refusal(f"This account holds {', '.join(held)} now: name one in the X-SparkPipe-Fleet header.", 409, "fleet_required")
        return held[0]

    def pair_loss(self, body):
        pricing = self.settings()
        fleet_id = body.get("fleet")
        if fleet_id not in [f["id"] for f in pricing["fleets"]]:
            raise Refusal(f"No Fleet {fleet_id}.", 404, "unknown_fleet")
        event, account = short_text(body.get("event"), "event", 100), body.get("account")
        if account not in self.book.accounts:
            raise Refusal("Give the account whose working state was lost.", 400, "invalid_request")
        lost = {}
        for name in ("contexts", "storage", "pair"):
            items = body.get(name, [])
            if not isinstance(items, list) or len(items) > 1000:
                raise Refusal(f"Give {name} as a list of at most 1000 entries.", 400, "invalid_request")
            lost[name] = [short_text(item, name, 300) for item in items]
        run = short_text(body["run"], "run", 100) if body.get("run") is not None else None
        ref = f"{fleet_id}:{event}"
        with self.book.lock:
            now = self.now()
            self.advance(now)
            at = stamp_field(body["at"], "time of the loss") if body.get("at") is not None else now
            if at > now + timedelta(minutes=5):
                raise Refusal("The loss cannot be in the future.", 400, "invalid_request")
            lot, duplicate = self.st.rebuilds.get(ref), ref in self.st.rebuilds
            if not lot:
                covering = [b for b in self.on_fleet(fleet_id) if b["account"] == account and b["begins"] <= at < b["ends"]]
                if not covering:
                    raise Refusal(f"{account} did not hold {fleet_id} at {stamp(at)}.", 409, "not_held")
                lot = self.lot(account, Decimal(covering[0]["rate"]) * REBUILD_HOURS, "rebuild", ref, now)
            found = {"fleet": fleet_id, "event": event, "account": lot["account"], "hc": lot["hc"], "lot": lot["lot"], "credited": lot["now"], "duplicate": duplicate}
            once = f"loss:{ref}"
            if self.app.notices.sent(once):
                return found
            name = self.fleet_name(fleet_id)
            listed = [f"Lost {label}: {item}" for label, items in (("context", lost["contexts"]), ("storage path", lost["storage"])) for item in items[:50]]
            more = len(lost["contexts"]) + len(lost["storage"]) - len(listed)
            self.notify(lot["account"], "working_state.lost", dict(lost, fleet=fleet_id, event=event, at=stamp(at), run=run, hc=lot["hc"], lot=lot["lot"]), now,
                        f"SparkPipe {name} lost part of your working state: {lot['hc']} HC credited",
                        "\n".join([f"A node pair of {name} lost part of your working state at {kst_text(at)}. We credited {lot['hc']} HC, 24 hours of the Fleet at its Rate (clause 6.6)." + (f" Rebuild run {run} started." if run else ""), ""]
                                  + listed + ([f"and {more} more, listed in the webhook and in GET /v1/runs."] if more > 0 else [])), once)
        self.app.mail(self.app.cfg.notify, f"Pair loss on {fleet_id}: {lot['hc']} HC credited", f"{self.app.who(lot['account'])}: event {event} on {fleet_id} at {stamp(at)}, {len(lost['contexts'])} contexts and {len(lost['storage'])} storage paths lost; lot {lot['lot']} credited {lot['hc']} HC under clause 6.6.")
        return found

    def hub_event(self, item):
        if not isinstance(item, dict) or item.get("type") not in HUB_EVENTS:
            raise Refusal("Give each event a type among " + ", ".join(HUB_EVENTS) + ".", 400, "invalid_request")
        fleet_id, account = item.get("fleet"), item.get("account")
        if fleet_id not in [f["id"] for f in self.settings()["fleets"]] or account not in self.book.accounts:
            raise Refusal("Give a known fleet and account.", 404, "unknown")
        ident, data = short_text(item.get("id"), "id", 100), item.get("data", {})
        if not isinstance(data, dict) or len(json.dumps(data)) > 16384:
            raise Refusal("Give data as an object of at most 16 KiB.", 400, "invalid_request")
        name = self.fleet_name(fleet_id)
        summary = short_text(item["summary"], "summary", 300) if item.get("summary") is not None else f"SparkPipe {name}: {item['type']}"
        lines = [f"{key}: {value}" for key, value in sorted(data.items()) if isinstance(value, (str, int, float)) and not isinstance(value, bool)]
        once = f"hub:{fleet_id}:{ident}"
        with self.book.lock:
            if self.app.notices.sent(once):
                return {"id": ident, "duplicate": True}
            found = self.notify(account, item["type"], dict(data, fleet=fleet_id), self.now(), summary, "\n".join([summary, ""] + lines), once)
        return {"id": ident, "notice": found["notice"], "duplicate": False}

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

    def sandbox_fleet(self, body):
        fleet = self.fleet_field(self.settings(), body.get("fleet"))
        if not isinstance(body.get("up"), bool):
            raise Refusal("Give up as true or false.", 400, "invalid_request")
        with self.book.lock:
            now = self.now()
            self.advance(now)
            self.app.uptime.observe(fleet["id"], body["up"], now, "sandbox", watch=False)
            self.advance(now)
        return {"fleet": fleet["id"], "up": body["up"], "now": stamp(now)}

    def move_clock(self, hours):
        with self.book.lock:
            self.book.append("clock", {"offset": self.st.offset + hours * 3600})
            self.advance(self.now())
        return {"now": stamp(self.now())}

    def first_open(self, pricing, fleet, now):
        first = week_start(now) + WEEK
        if fleet["hourFleet"] or not fleet["serving"] or self.busy(fleet["id"], first, first + WEEK):
            return None
        try:
            self.check_first(pricing, fleet, first, now)
        except Refusal:
            return None
        return slot_text(first)

    def status(self, fleet, now):
        if any(b["kind"] == "week" for b in self.busy(fleet["id"], now, now + timedelta(seconds=1))):
            return "pinned"
        return "hour-fleet" if fleet["hourFleet"] else "free"

    def fleet_view(self, pricing, listed, fleet, now):
        return dict(listed, serving=fleet["serving"], status=self.status(fleet, now), firstWeek=self.first_open(pricing, fleet, now),
                    booked=merged((b["begins"], b["ends"]) for b in self.on_fleet(fleet["id"]) if b["ends"] > now),
                    pinned=merged((b["begins"], b["ends"]) for b in self.on_fleet(fleet["id"]) if b["ends"] > now and b["kind"] == "week"),
                    maintenance=merged((m["begins"], m["ends"]) for m in self.maintained(fleet["id"], now, now + 60 * WEEK)))

    def public(self):
        pricing = self.settings()
        with self.book.lock:
            now = self.now()
            self.advance(now)
            fleets = [self.fleet_view(pricing, listed, fleet, now) for listed, fleet in zip(pricing["fleets"], self.fleets(pricing))]
            waiting = len(self.st.locks)
            room, limit = self.pair_room(pricing, week_start(now) + WEEK)
        return {
            "now": stamp(now), "sandbox": self.sandbox, "vatRate": pricing["vatRate"], "hoursFrom": pricing["hoursFrom"], "weeksFrom": pricing["weeksFrom"],
            "hc": pricing["hc"], "bookings": pricing["bookings"], "fleets": fleets, "models": pricing["models"], "waitlist": {"locks": waiting}, "pairs": {"open": room, "limit": limit},
        }

    def booking_view(self, b, now):
        state = "ended" if b["ends"] <= now else "live" if b["begins"] <= now else "upcoming"
        shown = {key: b.get(key) for key in ("booking", "fleet", "kind", "start", "end", "hours", "weeks", "hc", "discount", "waitlist_saving", "source", "unlocked")}
        return dict(shown, created=b["now"], state=state)

    def pinned_view(self, account, pricing, now):
        found = []
        for fleet in self.fleets(pricing):
            count, through = self.weeks_left(account, fleet["id"], now)
            if through:
                room = through - WEEK < week_start(now) + WEEK * pricing["bookings"]["maxWeeksAhead"] and not self.busy(fleet["id"], through, through + WEEK)
                if fleet["hourFleet"]:
                    held = self.weeks_left(account, fleet.get("pair"), now)[1]
                    room = room and bool(held) and held > through and bool(self.pair_room(pricing, through, fleet["id"])[0])
                found.append({"fleet": fleet["id"], "weeks": count, "through": slot_text(through), "deadline": slot_text(through - DAY), "canExtend": room, "autoExtend": self.st.extend.get((account, fleet["id"]), {}).get("weeks", 0),
                              "autoExtendWaiting": self.extend_waiting(account, fleet["id"])})
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
                "discount": self.discount_view(self.hour_discount(pricing["bookings"], account, now)),
                "waitlistRepay": text_hc(self.waitlist_repay(account)[0]) if account in st.locks else None,
                "lots": [{"lot": l["lot"], "hc": text_hc(l["hc"]), "first": l["first"], "expires": l["expires"], "source": l["source"], "ref": l["ref"], "issued": l["issued"], "expired": l["expires"] <= moment} for l in st.lots[account]],
                "bookings": [self.booking_view(b, now) for b in mine],
                "pinned": self.pinned_view(account, pricing, now),
                "refunds": list(st.refunds[account]),
                "keys": [self.key_view(r) for r in st.keys.values() if r["account"] == account],
            }

    def availability_text(self):
        pricing = self.settings()
        with self.book.lock:
            now = self.now()
            self.advance(now)
            rows = [f"{f['id']} {'serving' if f['serving'] else 'not-serving'} {AVAILABILITY.get(self.status(f, now), self.status(f, now))}" for f in self.fleets(pricing)]
            rows.append(f"waitlist locks {len(self.st.locks)}")
        return "\n".join([f"# SparkPipe availability, generated {stamp(now)}, Customer Agreement clause 4.2", "# fleet serving status (hour-fleet, pinned or available)"] + rows) + "\n"

    def holder_hours(self, fleet_id, start, end):
        if fleet_id not in [f["id"] for f in self.settings()["fleets"]]:
            raise Refusal(f"No Fleet {fleet_id}.", 404, "unknown_fleet")
        if not start < end <= start + WEEK:
            raise Refusal("Give from and to as hours such as 2026-12-01T01:00Z, to after from and at most 7 days later.", 400, "invalid_window")
        with self.book.lock:
            now = self.now()
            self.advance(now)
            closed = self.maintained(fleet_id, start, end)
            spans = [(b["begins"], b["ends"], b["account"]) for b in self.busy(fleet_id, start, end)]
            spans += [(slot_floor(f["begins"]), slot_ceil(f["ends"]) if f["ends"] else slot_floor(now) + HOUR, f["account"]) for f in self.st.failovers.values() if f["to"] == fleet_id]
            hours = set()
            for begins, ends, account in spans:
                moment = max(begins, start)
                while moment < min(ends, end):
                    if not any(m["begins"] <= moment < m["ends"] for m in closed):
                        hours.add((slot_text(moment), account))
                    moment += HOUR
        return {"fleet": fleet_id, "now": stamp(now), "hours": [{"start": hour, "account": account} for hour, account in sorted(hours)]}
