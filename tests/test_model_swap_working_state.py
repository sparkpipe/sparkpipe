#!/usr/bin/env python3
import asyncio
import json
import os
import pathlib
import sys
import tempfile
import time

import aiohttp
from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "serving"))
import model_swap  # noqa: E402
import working_state_fixture  # noqa: E402

working_state = model_swap.working_state
TOKEN = "t" * 40
FAKE_SSH = """import os, subprocess, sys
host, command = sys.argv[1], sys.argv[2]
state = STATE
if os.path.exists(os.path.join(state, "down-" + host)):
    print("ssh: connect to host " + host + ": No route to host")
    sys.exit(255)
home = os.path.join(state, "nodes", host)
os.makedirs(home, exist_ok=True)
sys.exit(subprocess.run(["bash", "-c", command], env=dict(os.environ, HOME=home)).returncode)
"""


class Market:
    def __init__(self):
        self.hours = []
        self.failing = True
        self.asked = []

    async def holders(self, request):
        self.asked.append((request.headers.get("Authorization"), dict(request.query)))
        if self.failing:
            return web.json_response({"error": "down"}, status=500)
        return web.json_response({"fleet": working_state_fixture.FLEET, "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hours": self.hours})


class Engine:
    def __init__(self, directive):
        self.directive = directive
        self.live = 0
        self.follow = True
        self.applied = 0
        self.status = 200
        self.reached = {}

    async def health(self, request):
        if self.follow:
            self.applied = working_state.read_directive(self.directive)[0]
            self.reached.setdefault(self.applied, time.monotonic())
        return web.json_response({"connected_ranks": 2, "live_requests": self.live, "working_state_wiped": self.applied}, status=self.status)


async def serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


async def until(condition, seconds):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    return condition()


def slot(moment):
    return working_state.slot_text(moment)


def config(state, market_base, engine_base):
    block = dict(working_state_fixture.block(state, market_base + "/market/fleets/F1/holders"), drain_seconds=3, confirm_seconds=1.5)
    return {"working_state": block, "nodes": ["n0", "n1"], "ssh": [sys.executable, os.path.join(state, "fake_ssh.py")], "drop_kv": "DROP {label}",
            "memory_probe": "PROBE", "memory_margin_bytes": 2 ** 30, "ready_seconds": 5, "api_seconds": 5, "drain_seconds": 5, "node_seconds": 10,
            "models": [{"id": "m", "kv_label": "m_stage", "node_memory_bytes": 2 ** 30, "node_reclaim": "true", "node_start": "true",
                        "node_ready": "echo READY", "node_stop": "true", "api_start": "true", "api_stop": "true", "health": engine_base + "/health"}]}


def node_directive(state, host):
    path = pathlib.Path(state, "nodes", host, "sparkdata", "working-state", "directive")
    return path.read_text() if path.exists() else None


async def scenario(state, failures):
    def check(condition, message):
        if not condition:
            failures.append(message)

    controller = working_state.Controller
    controller.POLL_SECONDS, controller.WATCH_SECONDS, controller.CHECK_SECONDS = 0.1, 0.05, 0.05
    controller.RETRY_SECONDS, controller.REPUSH_SECONDS, controller.PUSH_SECONDS = 0.3, 3600.0, 10.0
    pathlib.Path(state, "fake_ssh.py").write_text(FAKE_SSH.replace("STATE", repr(state)))
    working_state_fixture.write_token(state)
    hub_directive = pathlib.Path(working_state_fixture.block(state)["hub_directive"])
    market, engine = Market(), Engine(hub_directive)
    market_app = web.Application()
    market_app.router.add_get("/market/fleets/F1/holders", market.holders)
    engine_app = web.Application()
    engine_app.router.add_get("/health", engine.health)
    market_runner, market_base = await serve(market_app)
    engine_runner, engine_base = await serve(engine_app)
    swapper = model_swap.Swapper(config(state, market_base, engine_base))
    working = swapper.working
    control_runner, base = await serve(model_swap.build(swapper, TOKEN))
    headers = {"Authorization": "Bearer " + TOKEN}
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async def admit(account):
                async with session.post(base + "/working-state/admit", json={"account": account}) as response:
                    body = await response.json()
                    return response.status, (body.get("error") or {}).get("code"), response.headers.get("Retry-After"), body

            async def view():
                async with session.get(base + "/working-state") as response:
                    return await response.json()

            async def activate(body):
                async with session.post(base + "/activate", json=body) as response:
                    return response.status, ((await response.json()).get("error") or {})

            check(hub_directive.read_text() == working_state.directive_text(0, 0), "the controller writes the hub directive at start; with no state yet it is the plain 24-hour rule")
            status, code, retry, _ = await admit("sparkpipe")
            check((status, code) == (503, "holder_schedule_unknown") and retry is not None, f"with no holder schedule every admit fails closed: {status} {code}")
            check(await until(lambda: len(market.asked) >= 2, 5), "the controller polls the market's holder schedule")
            authorization, query = market.asked[0]
            hour = working_state.hour_floor(time.time())
            check(authorization == "Bearer " + working_state_fixture.HOLDERS_TOKEN, f"the holder schedule is asked with the hub's market token: {authorization}")
            check(query.get("from") in (slot(hour - 25 * 3600), slot(hour - 24 * 3600)) and query.get("to") in (slot(hour + 25 * 3600), slot(hour + 26 * 3600)),
                  f"the controller asks for the last 25 hours through the next 25: {query}")
            check(await until(lambda: node_directive(state, "n0") is not None and node_directive(state, "n1") is not None, 10),
                  "the controller pushes the directive to every node at start")

            market.failing = False
            check(await until(lambda: working.fresh(time.time()), 5), "a good answer makes the schedule known")
            status, code, _, body = await admit("sparkpipe")
            first = working.wiped
            check(status == 200 and body["user"] == "sparkpipe" and body["confirmed"] is True and first > 0, f"the first account is admitted after a handover from nobody: {status} {code} {body}")
            check(hub_directive.read_text() == working_state.directive_text(first, 0), f"the handover writes wiped and held_until 0 for an account with no held hours on the hub: {hub_directive.read_text()!r}")
            check(node_directive(state, "n0") == node_directive(state, "n1") == working_state.directive_text(first, 0), "and on every node, written whole")
            check(not list(pathlib.Path(state, "nodes", "n0", "sparkdata", "working-state").glob("*.tmp*")), "the node write leaves no temporary file")
            stored = pathlib.Path(working_state_fixture.block(state)["state_file"])
            check(stored.stat().st_mode & 0o777 == 0o600 and json.loads(stored.read_text()) == {"user": "sparkpipe", "wiped": first, "held_until": 0},
                  f"the hub keeps the current user, wiped and held_until in a private file: {oct(stored.stat().st_mode)} {stored.read_text()}")
            check(engine.applied >= first, "the engine confirmed the wipe before the admit returned")
            status, _, _, _ = await admit("sparkpipe")
            check(status == 200 and working.wiped == first, "the current user is admitted again without another wipe")

            hour = working_state.hour_floor(time.time())
            engine.live, engine.follow = 1, False
            market.hours = [{"start": slot(hour), "account": "ACHOLDER"}, {"start": slot(hour + 3600), "account": "ACHOLDER"},
                            {"start": slot(hour + 7200), "account": "ACOTHER"}]
            check(await until(lambda: working.handover is not None and working.handover["to"] == "ACHOLDER", 5),
                  "the start of another account's held hour starts a handover without waiting for its first request")
            holder_admit = asyncio.ensure_future(admit("ACHOLDER"))
            state_view = await view()
            check(state_view["handover"] and state_view["handover"]["phase"] == "draining" and state_view["holder"] == "ACHOLDER",
                  f"while the old user's requests are live the handover drains, and the view names it: {state_view}")
            status, code, retry, body = await admit("sparkpipe")
            expected = hour + 7200 - time.time()
            check((status, code) == (503, "fleet_held") and retry is not None and abs(int(retry) - expected) <= 2 and "ACHOLDER" not in json.dumps(body),
                  f"during a held hour another account gets 503 fleet_held with Retry-After at the end of the holder's contiguous hours, without the holder's id: {status} {code} {retry} vs {expected:.0f} {body}")
            status, error = await activate({"model": "m"})
            check((status, error.get("code")) == (409, "fleet_held"), f"a model change during a held hour without the holder's account is refused: {status} {error}")
            status, error = await activate({"model": "m", "account": "sparkpipe"})
            check((status, error.get("code")) == (409, "fleet_held"), f"a model change for another account during a held hour is refused: {status} {error}")
            status, error = await activate({"model": "m", "account": "ACHOLDER"})
            check(status in (200, 202) and not error, f"the holder's model change goes ahead: {status} {error}")
            async with session.post(base + "/force", json={"model": "m"}) as response:
                check(response.status == 409 and (await response.json())["error"]["code"] == "fleet_held", f"Switch now during a held hour is refused: {response.status}")
            await asyncio.sleep(0.5)
            check(working.wiped == first and not holder_admit.done() and hub_directive.read_text() == working_state.directive_text(first, 0),
                  "nothing is wiped and the holder waits while the old user still has a live request")
            engine.live = 0
            check(await until(lambda: working.wiped > first, 5), "the handover goes on once the engine has no live request")
            second = working.wiped
            check(hub_directive.read_text() == working_state.directive_text(second, hour + 3600), f"the hub directive takes the new wiped and the end of the current held hour: {hub_directive.read_text()!r}")
            check(await until(lambda: node_directive(state, "n1") == working_state.directive_text(second, hour + 3600), 5), "every node gets the new directive")
            await asyncio.sleep(0.4)
            check(not holder_admit.done(), "the holder is not admitted while the engine has not applied the wipe")
            engine.follow = True
            status, code, _, body = await holder_admit
            admitted = time.monotonic()
            check(status == 200 and body["user"] == "ACHOLDER" and body["held_until"] == hour + 3600 and second > first,
                  f"the holder is admitted after the wipe: {status} {code} {body}")
            check(second in engine.reached and engine.reached[second] <= admitted, "the admit returned only after the engine reported the new wiped")

            pathlib.Path(state, "down-n1").touch()
            market.hours = [{"start": slot(hour), "account": "ACNEXT"}, {"start": slot(hour + 3600), "account": "ACNEXT"}]
            check(await until(lambda: working.hours.get(hour) == "ACNEXT", 5), "the controller follows a changed holder schedule")
            status, code, _, body = await admit("ACNEXT")
            third = working.wiped
            check(status == 200 and body["user"] == "ACNEXT" and third > second, f"a held hour that changes hands passes the fleet to the new holder: {status} {code} {body}")
            check(node_directive(state, "n0") == working_state.directive_text(third, hour + 3600) and node_directive(state, "n1") == working_state.directive_text(second, hour + 3600),
                  "a node that cannot be reached keeps its old directive while the reachable ones take the new one")
            check((await view())["nodes"]["pending"] == ["n1"], "the failed node is listed as pending")
            pathlib.Path(state, "down-n1").unlink()
            check(await until(lambda: node_directive(state, "n1") == working_state.directive_text(third, hour + 3600), 5), "the push to a failed node is retried until it lands")
            check(await until(lambda: working.pending == [], 2), "and the node leaves the pending list")

            engine.follow = False
            market.hours = []
            check(await until(lambda: working.hours == {}, 5), "the controller follows a schedule with no held hours")
            status, code, _, _ = await admit("sparkpipe")
            fourth = working.wiped
            check((status, code) == (503, "handover_failed") and fourth > third and working.user == "sparkpipe",
                  f"an engine that never applies the wipe fails the admit closed: {status} {code}")
            status, code, _, _ = await admit("sparkpipe")
            check((status, code) == (503, "handover_failed") and working.wiped == fourth, f"and keeps failing, without another wipe, until the engine confirms: {status} {code}")
            engine.status = 503
            status, code, _, _ = await admit("sparkpipe")
            check((status, code) == (503, "handover_failed"), f"a degraded engine (503) that reports an older wipe does not count as confirmed: {status} {code}")
            engine.status = 200
            engine.follow = True
            status, _, _, _ = await admit("sparkpipe")
            check(status == 200 and working.wiped == fourth, "once the engine confirms the same wipe, the account is admitted")
            check(working.held_until == 0, "an account with no held hours has held_until 0")

            market.failing = True
            working.fetched = time.time() - working_state.Controller.STALE_SECONDS - 1
            status, code, _, _ = await admit("sparkpipe")
            check((status, code) == (503, "holder_schedule_unknown"), f"a holder schedule older than an hour fails every admit closed: {status} {code}")
            status, error = await activate({"model": "m", "account": "sparkpipe"})
            check((status, error.get("code")) == (503, "holder_schedule_unknown"), f"and every model change: {status} {error}")
            async with session.post(base + "/working-state/admit", json={"account": "bad account"}) as response:
                check(response.status == 400, "an account that is not an id is refused")
    finally:
        await control_runner.cleanup()
        await engine_runner.cleanup()
        await market_runner.cleanup()

    restarted = model_swap.Swapper(config(state, market_base, engine_base))
    check((restarted.working.user, restarted.working.wiped, restarted.working.held_until) == ("sparkpipe", fourth, 0),
          "a restarted controller keeps the current user, wiped and held_until")
    check(restarted.working.confirmed is None, "a restarted controller confirms the engine again before it admits")
    late = restarted.working
    late.user, late.held_until = "ACLATE", 0
    late.hours = {hour: "ACLATE", hour + 3600: "ACLATE"}
    late.fetched = hour + 3700
    late.tick(hour + 3700)
    check(late.held_until == hour + 7200 and hub_directive.read_text() == working_state.directive_text(fourth, hour + 7200)
          and json.loads(stored.read_text())["held_until"] == hour + 7200, "held_until follows the current user's held hours as they pass, on the hub and in the state file")
    late.tick(hour + 7300)
    check(late.held_until == hour + 7200, "after the last held hour held_until stays at its end")
    if late.pushing is not None:
        await late.pushing
    check(node_directive(state, "n0") == working_state.directive_text(fourth, hour + 7200), "a changed held_until is pushed to the nodes")
    stored.chmod(0o644)
    try:
        model_swap.Swapper(config(state, market_base, engine_base))
        check(False, "a working-state file others can read stops the controller")
    except SystemExit as error:
        check("owner only" in str(error), f"the refusal names the mode problem: {error}")
    stored.chmod(0o600)
    stored.write_text('{"user": "sparkpipe"}')
    try:
        model_swap.Swapper(config(state, market_base, engine_base))
        check(False, "a malformed working-state file stops the controller")
    except SystemExit:
        pass
    broken = config(state, market_base, engine_base)
    del broken["working_state"]["hub_directive"]
    try:
        model_swap.Swapper(broken)
        check(False, "a working_state block without hub_directive stops the controller")
    except SystemExit as error:
        check("hub_directive" in str(error), f"the refusal names the missing key: {error}")
    del broken["working_state"]
    try:
        model_swap.Swapper(broken)
        check(False, "a config without working_state stops the controller")
    except SystemExit as error:
        check("working_state" in str(error), f"the refusal names working_state: {error}")


def rule(failures):
    expired = working_state.expired
    day = working_state.RETENTION_SECONDS
    cases = [((100, 101, 0, 102), True, "state last used before the wipe is expired"),
             ((101, 101, 0, 102), False, "state used at the wipe belongs to the new user"),
             ((1000, 0, 0, 1000 + day - 1), False, "state stays for 24 hours after its last use"),
             ((1000, 0, 0, 1000 + day), True, "and goes at 24 hours"),
             ((1000, 0, 5000, 5000 + day - 1), False, "held hours keep state 24 hours past their end"),
             ((1000, 0, 5000, 5000 + day), True, "and no longer"),
             ((9000, 0, 5000, 9000 + day - 1), False, "a use after the held hours counts from the use")]
    for arguments, outcome, message in cases:
        if expired(*arguments) is not outcome:
            failures.append(f"{message}: expired{arguments} is not {outcome}")
    for text in ("sparkpipe-working-state 1\nwiped 5\nheld_until 6", "sparkpipe-working-state 2\nwiped 5\nheld_until 6\n",
                 "sparkpipe-working-state 1\nwiped -5\nheld_until 6\n", "sparkpipe-working-state 1\nheld_until 6\nwiped 5\n"):
        try:
            working_state.parse_directive(text)
            failures.append(f"a malformed directive is refused: {text!r}")
        except ValueError:
            pass
    if working_state.parse_directive(working_state.directive_text(1764550800, 1764554400)) != (1764550800, 1764554400):
        failures.append("the directive is three newline-terminated lines that read back as written")


def main():
    failures = []
    rule(failures)
    with tempfile.TemporaryDirectory() as state:
        asyncio.run(scenario(state, failures))
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS test_model_swap_working_state: the rule wipes at a change of user and keeps state 24 hours past the later of last use and held hours; "
          "an unknown or stale holder schedule fails every admit closed; a held hour admits only its holder and refuses others with Retry-After "
          "at the end of the holder's hours and refuses their model changes; a change of user drains live requests, writes wiped and held_until "
          "on the hub and every node, waits for the engine to apply it and only then admits; a failed node push is retried; an engine that never "
          "applies the wipe fails closed; a restart keeps the user, wiped and held_until")


if __name__ == "__main__":
    main()
