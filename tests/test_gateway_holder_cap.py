#!/usr/bin/env python3
import json
import re
import sys
import tempfile
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import sparkpipe_market as market
import sparkpipe_pay as sp
import test_market as tm

CAPACITY = 12
VPS = ROOT / "deployment" / "vps"
SITE = VPS / "sparkpipe-ai-ssl.conf"
DOOR = VPS / "sparkpipe-public-api.conf"
PAY_ENV = VPS / "sparkpipe-pay.env.example"


def tunnel_ports():
    ports = set()
    for line in PAY_ENV.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() in ("SPARKPIPE_POOL_UPSTREAM", "SPARKPIPE_FLEET_UPSTREAMS"):
            for url in re.findall(r"https?://[^\s\"']+", value):
                ports.add(urllib.parse.urlsplit(url).port)
    return ports


BASE = {"SPARKPIPE_PAY_SANDBOX": "1", "SPARKPIPE_PAY_PREFIX": "/pay/sandbox/api"}


def environment(work, capacity):
    env = dict(BASE, SPARKPIPE_PAY_DATA=str(work / "data"), SPARKPIPE_PAY_SITE=str(work / "site"))
    if capacity is not None:
        env["SPARKPIPE_GATEWAY_CAPACITY"] = capacity
    return env


def startup_rules(failures, work):
    for capacity in (None, "", "0", "-5", "lots", "12.5", "²"):
        try:
            sp.Config(environment(work, capacity))
            failures.append(f"the gateway starts with SPARKPIPE_GATEWAY_CAPACITY={capacity!r}; it must refuse without a measured capacity")
        except ValueError as error:
            if "SPARKPIPE_GATEWAY_CAPACITY" not in str(error):
                failures.append(f"the startup refusal names the missing setting: {error}")
    if getattr(sp.Config(environment(work, str(CAPACITY))), "gateway_capacity", None) != CAPACITY:
        failures.append("a measured capacity is taken as configured")


def at(*parts):
    return datetime(*parts, tzinfo=timezone.utc)


def burst(port, key, count):
    return [tm.chat(port, key, {"max_tokens": 2}) for _ in range(count)]


def refused_busy(port, key):
    status, _, raw = tm.raw_call(port, "POST", "/v1/chat/completions", json.dumps({"model": "kimi-k3", "max_tokens": 2, "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return status, raw


def gateway_rules(failures, work):
    tm.site(work, serving=True)
    tm.NOW[0] = at(2026, 11, 2, 3, 0)
    sp.clock = lambda: tm.NOW[0]
    saved = sp.LIMITS["gateway"]
    sp.LIMITS["gateway"] = (5, 600)
    server = sp.start(sp.Config(environment(work, str(CAPACITY))), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    app, port = server.app, server.server_address[1]
    m = app.market
    try:
        a, b, c, d, outsider, spare = (tm.person(app, name, 100000) for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot"))
        m.book_hours(a, {"fleet": "F1", "start": tm.slot(at(2026, 11, 2, 4)), "hours": 1})
        for who, fleet in ((b, "F3"), (c, "F4"), (d, "F5")):
            m.book_hours(who, {"fleet": fleet, "start": tm.slot(at(2026, 11, 2, 5)), "hours": 1})
        keys = {who: m.create_key(who, {"note": "cap"})["key"] for who in (a, b, outsider)}

        tm.NOW[0] = at(2026, 11, 2, 4, 30)
        answers = burst(port, keys[a], CAPACITY)
        if [status for status, _ in answers] != [200] * CAPACITY:
            failures.append(f"a holder above the non-holder limit of 5 per window is served up to capacity/N = {CAPACITY}/1: {[status for status, _ in answers]}")
        status, raw = refused_busy(port, keys[a])
        body = json.loads(raw)
        if status != 429 or body["error"]["code"] != "busy" or "12 requests per 10 minutes while 1 Fleet is rented" not in body["error"]["message"]:
            failures.append(f"with one Fleet rented the holder's next request is the busy 429 of its share: {status} {raw[:300]}")
        answers = burst(port, keys[outsider], 5)
        if [status for status, _ in answers] != [403] * 5 or any(answer["error"]["code"] != "fleet_not_held" for _, answer in answers):
            failures.append(f"an account that holds no Fleet is refused fleet_not_held within the old limit: {answers}")
        status, raw = refused_busy(port, keys[outsider])
        if status != 429 or json.loads(raw)["error"]["code"] != "busy" or "Too many requests" not in json.loads(raw)["error"]["message"]:
            failures.append(f"a non-holder still hits LIMITS['gateway'] (5 here) and gets the old busy 429: {status} {raw[:200]}")

        tm.NOW[0] = at(2026, 11, 2, 5, 30)
        share = CAPACITY // 3
        answers = burst(port, keys[b], share)
        status, raw = refused_busy(port, keys[b])
        if [code for code, _ in answers] != [200] * share or status != 429 or f"{share} requests per 10 minutes while 3 Fleets are rented" not in json.loads(raw)["error"]["message"]:
            failures.append(f"with three Fleets rented a holder's share is capacity/3 = {share}: {[code for code, _ in answers]} then {status} {raw[:200]}")

        sp.LIMITS["gateway"] = saved
        app.cfg.gateway_capacity = 30000
        served = 0
        gate = getattr(app, "throttle_gateway", lambda account: app.throttle("gateway", account))
        try:
            for _ in range(saved[0] + 1):
                gate(c)
                served += 1
        except market.Refusal:
            pass
        if served != saved[0] + 1:
            failures.append(f"a holder sends more than the {saved[0]} per {saved[1]} s of LIMITS['gateway'] and is served (share 30000/3): {served}")
        allowed = 0
        try:
            for _ in range(saved[0] + 1):
                gate(spare)
                allowed += 1
        except market.Refusal as error:
            if error.status != 429 or error.reason != "busy":
                failures.append(f"the non-holder refusal is the busy 429: {error.status} {error.reason}")
        if allowed != saved[0]:
            failures.append(f"an account without an active rental keeps LIMITS['gateway'] = {saved}: {allowed} served")
    finally:
        sp.LIMITS["gateway"] = saved
        server.shutdown()


def retry_header_rules(failures, work):
    tm.site(work, serving=True)
    tm.NOW[0] = at(2026, 11, 2, 3, 0)
    sp.clock = lambda: tm.NOW[0]
    server = sp.start(sp.Config(environment(work, "1")), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    app, port = server.app, server.server_address[1]
    try:
        holder = tm.person(app, "golf", 100000)
        app.market.book_hours(holder, {"fleet": "F1", "start": tm.slot(at(2026, 11, 2, 4)), "hours": 1})
        key = app.market.create_key(holder, {"note": "retry"})["key"]
        tm.NOW[0] = at(2026, 11, 2, 4, 10)
        first = tm.chat(port, key, {"max_tokens": 2})[0]
        request = tm.urllib.request.Request(f"http://127.0.0.1:{port}/v1/messages", data=json.dumps({"model": "kimi-k3", "max_tokens": 2, "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                            method="POST", headers={"X-Api-Key": key, "Content-Type": "application/json"})
        try:
            tm.urllib.request.urlopen(request, timeout=30)
            status, retry, body = 200, None, {}
        except tm.urllib.error.HTTPError as error:
            status, retry, body = error.code, error.headers.get("Retry-After"), json.loads(error.read())
        if first != 200 or status != 429 or retry != "60" or body.get("type") != "error" or body["error"]["type"] != "busy":
            failures.append(f"the holder cap's 429 carries retry-after like the other gateway 429s, in the API shape's body: {first} {status} {retry} {body}")
    finally:
        server.shutdown()


def public_door_rules(failures):
    ports = tunnel_ports()
    if not ports:
        failures.append(f"{PAY_ENV.name} names no hub tunnel upstream, so this test checks nothing")
    for conf in sorted(VPS.glob("*.conf")):
        for number, line in enumerate(conf.read_text().splitlines(), 1):
            words = line.split()
            if words and words[0] in ("ProxyPass", "ProxyPassMatch", "ProxyPassReverse"):
                for url in re.findall(r"https?://[^\s\"']+", line):
                    if urllib.parse.urlsplit(url).port in ports:
                        failures.append(f"{conf.name}:{number} proxies to the hub tunnel {url}; only the pay gateway may reach a hub")
    site = SITE.read_text()
    if "Include /etc/apache2/sparkpipe/public-api.conf" not in site:
        failures.append(f"{SITE.name} does not include public-api.conf, so the public header is not stripped and /v1/ is not closed")
    door = [line.strip() for line in DOOR.read_text().splitlines() if line.strip()]
    if "RequestHeader unset X-SparkPipe-Public" not in door:
        failures.append(f"{DOOR.name} does not strip X-SparkPipe-Public from incoming requests")
    if any("RequestHeader set" in line for line in door):
        failures.append(f"{DOOR.name} sets the public header itself")
    closed = [re.fullmatch(r'RedirectMatch 404 "?(.+?)"?', line) for line in door]
    patterns = [re.compile(match.group(1)) for match in closed if match]
    for path in ("/v1/models", "/v1/messages", "/v1", "/v1/chat/completions"):
        if not any(pattern.search(path) for pattern in patterns):
            failures.append(f"{DOOR.name} does not answer 404 on {path}")
    for path in ("/pay/api/v1/models", "/index.html", "/v1x"):
        if any(pattern.search(path) for pattern in patterns):
            failures.append(f"{DOOR.name} closes {path}, which is not the old hub door")


def main():
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        tm.tp.site_copy(work / "startup")
        startup_rules(failures, work / "startup")
        gateway_rules(failures, work / "gateway")
        retry_header_rules(failures, work / "retry")
    public_door_rules(failures)
    for failure in failures:
        print("FAIL test_gateway_holder_cap:", failure)
    if failures:
        raise SystemExit(1)
    print("PASS test_gateway_holder_cap: the gateway refuses to start without SPARKPIPE_GATEWAY_CAPACITY; an account with an active rental is not held to "
          "LIMITS['gateway'] but to capacity/N per window with N the Fleets rented now (N=1 and N=3), with the busy 429 and retry-after; "
          "an account without one keeps LIMITS['gateway']; the gateway is the only public inference door: no Apache rule proxies to a hub tunnel, "
          "the public header is stripped and /v1/ answers 404")


if __name__ == "__main__":
    main()
