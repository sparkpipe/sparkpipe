import asyncio
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from aiohttp import web

RETENTION_SECONDS = 86400
HOUR_SECONDS = 3600
WINDOW_HOURS = 25
OWN_ACCOUNT = "sparkpipe"
ACCOUNT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
DIRECTIVE = re.compile(r"sparkpipe-working-state 1\nwiped ([0-9]{1,19})\nheld_until ([0-9]{1,19})\n")
SLOT = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):00Z")
REQUIRED = ("fleet", "holders_url", "holders_token_file", "state_file", "hub_directive")
NODE_DIRECTIVE = ("mkdir -p \"$HOME/sparkdata/working-state\" && "
                  "printf 'sparkpipe-working-state 1\\nwiped {wiped}\\nheld_until {held_until}\\n' > \"$HOME/sparkdata/working-state/directive.tmp.$$\" && "
                  "mv -f \"$HOME/sparkdata/working-state/directive.tmp.$$\" \"$HOME/sparkdata/working-state/directive\"")


def expired(used, wiped, held_until, now):
    return used < wiped or now >= max(used, held_until) + RETENTION_SECONDS


def directive_text(wiped, held_until):
    return f"sparkpipe-working-state 1\nwiped {int(wiped)}\nheld_until {int(held_until)}\n"


def parse_directive(text):
    match = DIRECTIVE.fullmatch(text)
    if match is None:
        raise ValueError("not three lines: sparkpipe-working-state 1, wiped <unix seconds>, held_until <unix seconds>")
    return int(match.group(1)), int(match.group(2))


def read_directive(path):
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return 0, 0
    except (OSError, UnicodeDecodeError) as error:
        print(json.dumps({"event": "working_state_directive_malformed", "path": str(path), "error": str(error)}), file=sys.stderr, flush=True)
        return 0, 0
    try:
        return parse_directive(text)
    except ValueError as error:
        print(json.dumps({"event": "working_state_directive_malformed", "path": str(path), "error": str(error)}), file=sys.stderr, flush=True)
        return 0, 0


def write_directive(path, wiped, held_until):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(directive_text(wiped, held_until))
    os.replace(temporary, path)


def hour_floor(moment):
    return int(moment) // HOUR_SECONDS * HOUR_SECONDS


def slot_text(moment):
    return time.strftime("%Y-%m-%dT%H:00Z", time.gmtime(moment))


def slot_time(text):
    match = SLOT.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(f"hour start {text!r} is not YYYY-MM-DDTHH:00Z")
    year, month, day, hour = (int(group) for group in match.groups())
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp())


def valid_account(value):
    return isinstance(value, str) and ACCOUNT.fullmatch(value) is not None


def load_state(path):
    try:
        mode = os.stat(path).st_mode & 0o777
    except FileNotFoundError:
        return None, 0, 0
    if mode & 0o077:
        raise SystemExit(f"working_state: {path} has mode {mode:o}; the working-state file must be readable by its owner only")
    try:
        state = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"working_state: {path} is not a working-state file: {error}")
    if (not isinstance(state, dict) or set(state) != {"user", "wiped", "held_until"} or not (state["user"] is None or valid_account(state["user"])) or
            not all(type(state[key]) is int and state[key] >= 0 for key in ("wiped", "held_until"))):
        raise SystemExit(f"working_state: {path} needs exactly user (an account or null), wiped and held_until (unix seconds)")
    return state["user"], state["wiped"], state["held_until"]


def refusal(status, code, message, retry_after):
    return web.json_response({"error": {"code": code, "message": message}}, status=status, headers={"Retry-After": str(int(retry_after))})


class HandoverError(Exception):
    pass


class Controller:
    POLL_SECONDS = 30.0
    STALE_SECONDS = 3600.0
    WATCH_SECONDS = 1.0
    CHECK_SECONDS = 0.5
    PUSH_SECONDS = 20.0
    RETRY_SECONDS = 60.0
    REPUSH_SECONDS = 300.0

    def __init__(self, spec, swapper, token):
        if not isinstance(spec.get("fleet"), str) or not spec["fleet"] or not isinstance(spec.get("holders_url"), str) or not spec["holders_url"].startswith(("http://", "https://")):
            raise SystemExit("working_state: fleet must name the fleet and holders_url must be the market's http(s) holder schedule")
        for key in ("state_file", "hub_directive"):
            if not isinstance(spec.get(key), str) or not Path(spec[key]).is_absolute():
                raise SystemExit(f"working_state: {key} must be an absolute path")
        self.swapper = swapper
        self.fleet = spec["fleet"]
        self.holders_url = spec["holders_url"]
        self.headers = {"Authorization": "Bearer " + token}
        self.state_file = Path(spec["state_file"])
        self.hub_directive = Path(spec["hub_directive"])
        self.drain_seconds = float(spec.get("drain_seconds", 60))
        self.confirm_seconds = float(spec.get("confirm_seconds", 30))
        if self.drain_seconds <= 0 or self.confirm_seconds <= 0:
            raise SystemExit("working_state: drain_seconds and confirm_seconds must be positive")
        self.user, self.wiped, self.held_until = load_state(self.state_file)
        self.confirmed = None
        self.hours = {}
        self.fetched = None
        self.poll_error = None
        self.handover = None
        self.handing = None
        self.last = None
        self.push_lock = asyncio.Lock()
        self.pushing = None
        self.attempted = None
        self.pushed = None
        self.pending = []
        self.tasks = []

    def fresh(self, now):
        return self.fetched is not None and now - self.fetched < self.STALE_SECONDS

    def holder_at(self, now):
        return self.hours.get(hour_floor(now)) if self.fresh(now) else None

    def held_end(self, holder, now):
        end = hour_floor(now)
        while self.hours.get(end) == holder:
            end += HOUR_SECONDS
        return end

    def held_until_for(self, account, now):
        return max((start + HOUR_SECONDS for start, holder in self.hours.items() if holder == account and start <= now), default=0)

    def view(self):
        now = time.time()
        return {"fleet": self.fleet, "user": self.user, "holder": self.holder_at(now), "wiped": self.wiped, "held_until": self.held_until,
                "confirmed": self.confirmed == self.wiped, "handover": dict(self.handover) if self.handover else None, "last_handover": self.last,
                "schedule": {"fresh": self.fresh(now), "age_s": round(now - self.fetched, 1) if self.fetched is not None else None,
                             "held_hours": len(self.hours), "error": self.poll_error},
                "nodes": {"pushed": self.pushed, "pending": list(self.pending)}}

    def persist(self):
        temporary = self.state_file.with_name(self.state_file.name + ".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps({"user": self.user, "wiped": self.wiped, "held_until": self.held_until}) + "\n")
        os.replace(temporary, self.state_file)

    def accept(self, answer, now):
        if not isinstance(answer, dict) or answer.get("fleet") != self.fleet or not isinstance(answer.get("hours"), list):
            raise ValueError(f"the answer is not a holder schedule for fleet {self.fleet}")
        hours = {}
        for entry in answer["hours"]:
            if not isinstance(entry, dict) or not valid_account(entry.get("account")):
                raise ValueError(f"held hour {entry!r} has no valid account")
            start = slot_time(entry.get("start"))
            if hours.setdefault(start, entry["account"]) != entry["account"]:
                raise ValueError(f"hour {entry['start']} has two holders")
        self.hours, self.fetched = hours, now

    async def poll_once(self):
        now = time.time()
        params = {"from": slot_text(hour_floor(now) - WINDOW_HOURS * HOUR_SECONDS), "to": slot_text(hour_floor(now) + WINDOW_HOURS * HOUR_SECONDS)}
        async with self.swapper.session.get(self.holders_url, params=params, headers=self.headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status != 200:
                raise ValueError(f"the market answered {response.status}")
            answer = await response.json(content_type=None)
        self.accept(answer, now)

    async def poll_loop(self):
        while True:
            try:
                await self.poll_once()
                if self.poll_error is not None:
                    print(json.dumps({"event": "working_state_schedule_restored", "held_hours": len(self.hours)}), file=sys.stderr, flush=True)
                self.poll_error = None
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, ValueError) as error:
                text = f"{type(error).__name__}: {error}"
                if text != self.poll_error:
                    print(json.dumps({"event": "working_state_schedule_failed", "error": text, "fresh": self.fresh(time.time())}), file=sys.stderr, flush=True)
                self.poll_error = text
            await asyncio.sleep(self.POLL_SECONDS)

    async def engines(self):
        models = list({model["health"]: model for model in self.swapper.models.values()}.values())
        return [(status, payload if isinstance(payload, dict) else {}) for status, payload in await asyncio.gather(*(self.swapper.health(model) for model in models))]

    async def drain(self):
        deadline = time.monotonic() + self.drain_seconds
        while True:
            live = sum(int(payload.get("live_requests") or 0) for status, payload in await self.engines() if status != 0)
            if not live:
                return
            if time.monotonic() >= deadline:
                raise HandoverError(f"{live} request(s) still live after {self.drain_seconds:g} s of drain")
            self.handover["live_requests"] = live
            await asyncio.sleep(self.CHECK_SECONDS)

    async def confirm(self):
        deadline = time.monotonic() + self.confirm_seconds
        while True:
            behind = [payload.get("working_state_wiped") for status, payload in await self.engines()
                      if (status == 200 or "working_state_wiped" in payload) and
                      not (type(payload.get("working_state_wiped")) is int and payload["working_state_wiped"] >= self.wiped)]
            if not behind:
                return None
            if time.monotonic() >= deadline:
                return f"the engine reports working_state_wiped {behind[0]!r}, not {self.wiped}, after {self.confirm_seconds:g} s"
            await asyncio.sleep(self.CHECK_SECONDS)

    async def push(self):
        async with self.push_lock:
            wiped, held_until = self.wiped, self.held_until
            results = await self.swapper.fan(NODE_DIRECTIVE, self.PUSH_SECONDS, {"wiped": wiped, "held_until": held_until})
            self.attempted = time.time()
            failed = [(self.swapper.nodes[rank], code, text[-200:]) for rank, (code, text) in enumerate(results) if code != 0]
            self.pending = [host for host, _, _ in failed]
            self.pushed = {"wiped": wiped, "held_until": held_until, "at": int(self.attempted), "failed": len(failed)}
            if failed:
                print(json.dumps({"event": "working_state_push_failed", "wiped": wiped, "held_until": held_until, "nodes": self.pending,
                                  "first": {"host": failed[0][0], "exit": failed[0][1], "output": failed[0][2]}}), file=sys.stderr, flush=True)

    def push_due(self, now):
        if self.pushing is not None and not self.pushing.done():
            return False
        if self.pushed is None or (self.pushed["wiped"], self.pushed["held_until"]) != (self.wiped, self.held_until):
            return True
        return now - self.attempted >= (self.RETRY_SECONDS if self.pending else self.REPUSH_SECONDS)

    async def hand_over(self, account):
        started = time.time()
        previous = self.user
        self.handover = {"from": previous, "to": account, "phase": "draining" if account != previous else "confirming", "started": int(started)}
        print(json.dumps({"event": "working_state_handover", "from": previous, "to": account}), file=sys.stderr, flush=True)
        record = {"from": previous, "to": account, "started": int(started)}
        try:
            if account != previous:
                await self.drain()
                self.handover["phase"] = "wiping"
                now = time.time()
                self.user, self.wiped, self.held_until = account, max(int(now) + 1, self.wiped + 1), self.held_until_for(account, now)
                self.persist()
            write_directive(self.hub_directive, self.wiped, self.held_until)
            self.handover["phase"] = "confirming"
            _, behind = await asyncio.gather(self.push(), self.confirm())
            if behind is not None:
                raise HandoverError(behind)
            await asyncio.sleep(max(0.0, self.wiped - time.time()))
            self.confirmed = self.wiped
            record.update(ok=True, error=None)
        except HandoverError as error:
            record.update(ok=False, error=str(error))
        except Exception as error:
            record.update(ok=False, error=f"unexpected {type(error).__name__} while {self.handover['phase']}: {error}")
        finally:
            self.handover = None
            self.handing = None
        record.update(wiped=self.wiped, held_until=self.held_until, seconds=round(time.time() - started, 1))
        print(json.dumps(dict(record, event="working_state_handover_done")), file=sys.stderr, flush=True)
        self.last = record
        return record

    def begin(self, account):
        self.handing = asyncio.ensure_future(self.hand_over(account))
        return self.handing

    def tick(self, now):
        holder = self.holder_at(now)
        if self.handing is None and holder is not None and holder != self.user:
            self.begin(holder)
        if self.handing is None and self.user is not None and self.fresh(now):
            held_until = max(self.held_until, self.held_until_for(self.user, now))
            if held_until != self.held_until:
                self.held_until = held_until
                self.persist()
                write_directive(self.hub_directive, self.wiped, self.held_until)
        if self.push_due(now):
            self.pushing = asyncio.ensure_future(self.push())

    async def watch_loop(self):
        while True:
            try:
                self.tick(time.time())
            except OSError as error:
                print(json.dumps({"event": "working_state_write_failed", "error": str(error)}), file=sys.stderr, flush=True)
            await asyncio.sleep(self.WATCH_SECONDS)

    async def admit(self, account):
        deadline = time.monotonic() + self.drain_seconds + self.confirm_seconds + 2 * self.PUSH_SECONDS + 10
        while True:
            now = time.time()
            if not self.fresh(now):
                return refusal(503, "holder_schedule_unknown", "the fleet's holder schedule is unknown or more than an hour old", self.POLL_SECONDS)
            holder = self.holder_at(now)
            if holder is not None and account != holder:
                end = self.held_end(holder, now)
                return refusal(503, "fleet_held", f"this fleet is held by another account until {slot_text(end)}", max(1, math.ceil(end - now)))
            task = self.handing
            if task is None:
                if account == self.user and self.confirmed == self.wiped:
                    return None
                task = self.begin(account)
            try:
                record = await asyncio.wait_for(asyncio.shield(task), max(0.0, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                return refusal(503, "handover_pending", "the fleet is still passing to another account", 5)
            if record["to"] == account and not record["ok"]:
                return refusal(503, "handover_failed", f"the fleet could not pass to this account: {record['error']}", 5)

    async def admit_route(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": {"code": "invalid_json", "message": "the body must be JSON with an account"}}, status=400)
        account = body.get("account") if isinstance(body, dict) else None
        if not valid_account(account):
            return web.json_response({"error": {"code": "invalid_account", "message": "account must be 1 to 64 letters, digits, - or _"}}, status=400)
        refused = await self.admit(account)
        return refused if refused is not None else web.json_response(self.view())

    async def view_route(self, request):
        return web.json_response(self.view())

    def refuse_swap(self, body):
        now = time.time()
        if not self.fresh(now):
            return refusal(503, "holder_schedule_unknown", "the fleet's holder schedule is unknown or more than an hour old", self.POLL_SECONDS)
        holder = self.holder_at(now)
        if holder is not None and (body.get("account") if isinstance(body, dict) else None) != holder:
            end = self.held_end(holder, now)
            return refusal(409, "fleet_held", f"this fleet is held until {slot_text(end)}; only its holder's requests change the model", max(1, math.ceil(end - now)))
        return None

    async def startup(self):
        write_directive(self.hub_directive, self.wiped, self.held_until)
        self.tasks = [asyncio.ensure_future(self.poll_loop()), asyncio.ensure_future(self.watch_loop())]

    async def cleanup(self):
        tasks = self.tasks + [task for task in (self.handing, self.pushing) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
