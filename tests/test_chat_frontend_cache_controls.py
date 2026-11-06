#!/usr/bin/env python3
import asyncio
import contextlib
import importlib.util
import io
import json
import sqlite3
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
chat_frontend = fixture.chat_frontend
chat_cache = chat_frontend.chat_cache
BLOCK = chat_cache.BLOCK_TOKENS
WORDS = ["hello world, the weather in Paris is sunny today", "read the file a.txt and list the directory", "call the tool with path and pattern",
         "user assistant system tool think reply answer question", "The answer is 42. Let me check the file first."]
ACCOUNT = "acct-a"
OTHER = "acct-b"
PINNER = "acct-p"


def text(seed, repeats):
    return " ".join(WORDS[(seed + index) % len(WORDS)] + f" {seed}" for index in range(repeats))


def chat_body(content, **extra):
    body = {"model": "alpha", "messages": [{"role": "user", "content": content}], "max_tokens": 16}
    body.update(extra)
    return body


def messages_body(content, **extra):
    body = {"model": "alpha", "max_tokens": 16, "system": "You are a coding agent.", "messages": [{"role": "user", "content": content}]}
    body.update(extra)
    return body


def responses_body(content, **extra):
    body = {"model": "alpha", "instructions": "You are a coding agent.", "input": content, "max_output_tokens": 16}
    body.update(extra)
    return body


SHAPES = (("/v1/chat/completions", "openai_chat", chat_body), ("/v1/messages", "anthropic", messages_body), ("/v1/responses", "openai_responses", responses_body))


def scan_replay(path):
    items, size, newest = 0, 0, None
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for table in ("turns", "responses", "aliases"):
            for used, held in connection.execute(f"SELECT used, bytes FROM {table}"):
                items, size = items + 1, size + int(held or 0)
                newest = float(used) if newest is None else max(newest, float(used))
    finally:
        connection.close()
    return items, size, newest


async def commit_only(harness, check):
    engine = harness.engine
    for index, (path, _, build) in enumerate(SHAPES):
        before = len(engine.requests)
        status, headers, reply = await harness.post(path, build(text(index, 12), commit_only=True, stream=True), account=ACCOUNT,
                                                    **{"X-SparkPipe-Priority": "interactive"})
        body = json.loads(reply)
        sent = engine.requests[before:]
        prompt = len(sent[0]["prompt_token_ids"]) if sent else -1
        check(status == 200 and body.get("object") == "cache.commit" and "choices" not in body and "content" not in body and "output" not in body,
              f"{path} commit_only answers the commit object, not a reply: {status} {reply[:300]}")
        check(len(sent) == 1 and sent[0].get("commit_only") is True and "max_tokens" not in sent[0] and sent[0]["priority"] == chat_frontend.BULK_PRIORITY,
              f"{path} commit_only sends one commit-only engine request at background priority even with an interactive hint: {sent}")
        check(body.get("prompt_tokens") == prompt and body.get("cached_tokens") == 0 and body.get("committed_tokens") == prompt // BLOCK * BLOCK and
              body.get("prefill_ms") == 12, f"{path} commit_only returns prompt, cached and committed tokens and the engine's prefill time: {body}")
        record = body.get("sparkpipe") or {}
        check(record.get("output_tokens") == 0 and record.get("commit_only") is True and record.get("prompt_tokens") == prompt and
              record.get("priority") == "background" and headers.get("X-SparkPipe-Request") == record.get("request"),
              f"{path} commit_only carries the usage record a normal request has, with no output: {record}")
    status, _, reply = await harness.post("/v1/chat/completions", chat_body("hi", commit_only="yes"), account=ACCOUNT)
    check(status == 400 and json.loads(reply)["error"]["code"] == "invalid_request", f"commit_only that is not a boolean is a named 400: {reply}")
    status, _, reply = await harness.post("/v1/messages", messages_body("hi", commit_only=1), account=ACCOUNT)
    check(status == 400 and json.loads(reply)["type"] == "error", f"Messages refuses a non-boolean commit_only in its own error shape: {reply}")
    engine.commits = False
    status, _, reply = await harness.post("/v1/chat/completions", chat_body(text(7, 3), commit_only=True), account=ACCOUNT)
    engine.commits = True
    check(status == 502 and json.loads(reply)["error"]["code"] == "commit_only_unsupported",
          f"an engine that generates for a commit-only request is refused, never passed off as a commit: {status} {reply}")


async def warm_then_match(harness, check):
    engine = harness.engine
    for index, (path, fmt, build) in enumerate(SHAPES):
        content = text(20 + index, 18)
        before = len(engine.requests)
        status, _, reply = await harness.post("/v1/cache/warm", dict(build(content), format=fmt), account=ACCOUNT)
        warm = json.loads(reply)
        check(status == 200 and warm.get("object") == "cache.commit" and engine.requests[before].get("commit_only") is True and
              engine.requests[before]["priority"] == chat_frontend.BULK_PRIORITY, f"POST /v1/cache/warm with format {fmt} commits at background priority: {reply[:300]}")
        status, _, reply = await harness.post(path, build(content), account=ACCOUNT)
        real = engine.requests[-1]
        record = json.loads(reply).get("sparkpipe") or {}
        check(status == 200 and real["prompt_token_ids"] == engine.requests[before]["prompt_token_ids"],
              f"{fmt}: the warm-up renders exactly the tokens the next real request sends")
        check(warm["committed_tokens"] >= 3 * BLOCK and record.get("cached_tokens") == warm["committed_tokens"],
              f"{fmt}: the next real request matches every warmed block ({warm['committed_tokens']} committed, {record.get('cached_tokens')} cached)")
    status, _, reply = await harness.post("/v1/cache/warm", dict(chat_body("hi"), format="openai_completions"), account=ACCOUNT)
    check(status == 400 and json.loads(reply)["error"]["code"] == "invalid_request", f"a warm-up names one of the three formats: {reply}")
    status, _, reply = await harness.post("/v1/cache/warm", dict(chat_body("hi"), format="openai_chat", commit_only=False), account=ACCOUNT)
    check(status == 400, f"a warm-up cannot turn commit_only off: {reply}")


async def warm_yields(tmp, check):
    async with fixture.Harness(tmp / "yield", sequences=3) as harness:
        engine = harness.engine
        engine.hold = asyncio.Event()
        first = asyncio.ensure_future(harness.post("/v1/cache/warm", dict(chat_body(text(40, 6)), format="openai_chat"), account=ACCOUNT))
        while len(engine.requests) < 1:
            await asyncio.sleep(0.01)
        second = asyncio.ensure_future(harness.post("/v1/cache/warm", dict(chat_body(text(41, 6)), format="openai_chat"), account=ACCOUNT))
        await asyncio.sleep(0.2)
        waiting = harness.frontend.models["alpha"].sequences.state()["waiting"]
        interactive = asyncio.ensure_future(harness.post("/v1/chat/completions", chat_body(text(42, 6)), account=ACCOUNT))
        while len(engine.requests) < 2:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        order = [bool(request.get("commit_only")) for request in engine.requests]
        engine.hold.set()
        results = await asyncio.gather(first, second, interactive)
        check(waiting == 1 and order == [True, False] and all(result[0] == 200 for result in results) and
              [bool(request.get("commit_only")) for request in engine.requests] == [True, False, True],
              f"a warm-up holds at most the background share of sequences, and an interactive request goes ahead of a queued warm-up: {order} waiting={waiting}")


async def pins(harness, check):
    engine = harness.engine
    status, _, reply = await harness.post("/v1/chat/completions", chat_body(text(60, 16)), account=PINNER)
    first = json.loads(reply)["sparkpipe"]["request"]
    first_ids = list(engine.requests[-1]["prompt_token_ids"])
    status, _, reply = await harness.post("/v1/chat/completions", chat_body(text(61, 20)), account=PINNER)
    second_ids = list(engine.requests[-1]["prompt_token_ids"])
    status, state = await harness.call("GET", "/v1/state", account=PINNER)
    chains = {chain["tokens"]: chain for chain in state["chains"]}
    first_tokens, second_tokens = len(first_ids) // BLOCK * BLOCK, len(second_ids) // BLOCK * BLOCK
    check(status == 200 and first_tokens in chains and second_tokens in chains and not any(chain["pinned"] for chain in state["chains"]) and
          all(chain["model"] == "alpha" and chain["last_used"] > 0 for chain in state["chains"]),
          f"GET /v1/state lists the account's chains with tokens, last use and no pin yet: {state.get('chains')}")
    check(state["cache"]["alpha"]["capacity_tokens"] is None and state["cache"]["alpha"]["pin_limit_tokens"] is None,
          f"an engine that reports no capacity shows none: {state['cache']['alpha']}")
    status, body = await harness.call("POST", "/v1/cache/pins", {"request": first}, account=PINNER)
    check(status == 503 and body["error"]["code"] == "cache_capacity_unknown" and not engine.pin_sets,
          f"a pin is refused, by name, while the engine does not report the capacity it is checked against: {body}")
    engine.capacity = 2 * (first_tokens + second_tokens) - 2 * BLOCK
    limit = engine.capacity // 2
    status, body = await harness.call("POST", "/v1/cache/pins", {"request": first}, account=PINNER)
    pin = (body or {}).get("pin") or {}
    check(status == 201 and body.get("created") is True and pin.get("tokens") == first_tokens and pin.get("source") == {"request": first} and
          body.get("limit_tokens") == limit and engine.pin_sets == [[first_ids[:first_tokens]]],
          f"POST /v1/cache/pins by request id pins that request's prompt chain and hands the engine the pin set: {status} {body} {len(engine.pin_sets)}")
    status, again = await harness.call("POST", "/v1/cache/pins", {"request": first}, account=PINNER)
    check(status == 200 and again.get("created") is False and again["pin"]["id"] == pin["id"] and len(engine.pin_sets) == 1,
          f"pinning the same chain again returns the pin it has: {again}")
    status, state = await harness.call("GET", "/v1/state", account=PINNER)
    by_tokens = {chain["tokens"]: chain for chain in state["chains"]}
    check(by_tokens[first_tokens]["pinned"] is True and by_tokens[second_tokens]["pinned"] is False and [entry["id"] for entry in state["pins"]] == [pin["id"]] and
          state["cache"]["alpha"] == {"capacity_tokens": engine.capacity, "pin_limit_tokens": limit, "pinned_tokens": first_tokens,
                                      "used_tokens": first_tokens + second_tokens},
          f"GET /v1/state shows the pinned flag, the pins and the cache capacity and use: {state['cache']} {state['pins']}")
    second_chain = by_tokens[second_tokens]["id"]
    status, body = await harness.call("POST", "/v1/cache/pins", {"prefix": second_chain}, account=PINNER)
    check(status == 409 and body["error"]["code"] == "pin_limit_exceeded" and body["error"]["limit_tokens"] == limit and
          body["error"]["pinned_tokens"] == first_tokens + second_tokens and len(engine.pin_sets) == 1,
          f"a pin that would take pins past half the cache is refused by name and nothing reaches the engine: {body}")
    for request_body, code in (({"request": "r_missing"}, "chain_not_found"), ({"pack": "core"}, "pack_not_found"), ({}, "invalid_request"),
                               ({"request": first, "prefix": second_chain}, "invalid_request")):
        status, body = await harness.call("POST", "/v1/cache/pins", request_body, account=PINNER)
        check(body["error"]["code"] == code, f"POST /v1/cache/pins {request_body} is refused with {code}: {body}")
    status, other = await harness.call("GET", "/v1/state", account=OTHER)
    check(status == 200 and other["chains"] == [] and other["pins"] == [] and other["cache"]["alpha"]["used_tokens"] == 0,
          f"another account sees none of this account's chains or pins: {other}")
    status, body = await harness.call("POST", "/v1/cache/pins", {"request": first}, account=OTHER)
    check(status == 404 and body["error"]["code"] == "chain_not_found", f"another account cannot pin this account's request: {body}")
    status, body = await harness.call("DELETE", "/v1/cache/pins/" + pin["id"], account=OTHER)
    check(status == 404 and body["error"]["code"] == "pin_not_found", f"another account cannot remove this account's pin: {body}")
    status, listed = await harness.call("GET", "/v1/cache/pins", account=PINNER)
    check(status == 200 and [entry["id"] for entry in listed["data"]] == [pin["id"]], f"GET /v1/cache/pins lists the account's pins: {listed}")
    status, body = await harness.call("DELETE", "/v1/cache/pins/" + pin["id"], account=PINNER)
    check(status == 200 and body == {"deleted": pin["id"], "engine_synced": True} and engine.pin_sets[-1] == [],
          f"DELETE /v1/cache/pins/{{id}} unpins and hands the engine the smaller set: {body}")
    engine.pin_reply = (501, {"error": {"message": "this engine cannot carry pins to its ranks' page caches yet", "type": "server_error", "code": "pins_unsupported"}})
    status, body = await harness.call("POST", "/v1/cache/pins", {"prefix": by_tokens[first_tokens]["id"]}, account=PINNER)
    status_state, state = await harness.call("GET", "/v1/state", account=PINNER)
    check(status == 501 and body["error"]["code"] == "pins_unsupported" and state["pins"] == [],
          f"an engine that cannot hold pins refuses them by name and the chat layer records none: {status} {body}")
    engine.pin_reply = (200, {"pinned_pages": 1})
    status, body = await harness.call("POST", "/v1/cache/pins", {"prefix": by_tokens[first_tokens]["id"]}, account=PINNER)
    check(status == 201, f"pinning by prefix id works: {body}")
    harness.directive(time.time() + 1, 0)
    harness.frontend.expire_working_state(time.time(), False)
    await harness.frontend.sync_dirty_pins()
    status, state = await harness.call("GET", "/v1/state", account=PINNER)
    check(state["pins"] == [] and state["chains"] == [] and engine.pin_sets[-1] == [],
          f"a change of user wipes the pins and chains with the working state and clears the engine's pin set: {state['pins']} {state['chains']}")
    await asyncio.sleep(1.1)
    status, _, reply = await harness.post("/v1/chat/completions", chat_body(text(62, 16)), account=PINNER)
    status, body = await harness.call("POST", "/v1/cache/pins", {"request": json.loads(reply)["sparkpipe"]["request"]}, account=PINNER)
    check(status == 201, f"the new user's own chains pin after the wipe: {body}")
    status, deleted = await harness.call("DELETE", "/v1/cache", account=PINNER)
    status, state = await harness.call("GET", "/v1/state", account=PINNER)
    check("pins" in deleted["deleted"] and "chain_ledger" in deleted["deleted"] and state["pins"] == [] and state["chains"] == [] and engine.pin_sets[-1] == [],
          f"DELETE /v1/cache deletes the account's pins and chains: {deleted} {state['pins']}")


async def stored_state(harness, tmp, check):
    status, _, reply = await harness.post("/v1/messages", messages_body(text(80, 4)), account=ACCOUNT)
    async with harness.session.put(harness.base + "/v1/models/aliases", json={"claude-*": "alpha"}, headers=harness.headers(account=ACCOUNT)) as response:
        await response.read()
    status, state = await harness.call("GET", "/v1/state", account=ACCOUNT)
    items, size, newest = scan_replay(tmp / "state" / "replay" / f"account-{ACCOUNT}.sqlite")
    stored = state["stored"]
    check(status == 200 and items >= 2 and stored["items"] == items and stored["bytes"] == size and stored["newest_used"] == newest,
          f"GET /v1/state reports the replay store as a scan of its file finds it ({items} items, {size} bytes): {stored}")
    check(state["working_state"]["deleted_after"] == round(max(newest, state["working_state"]["held_until"], *(entry["last_used"] for entry in state["chains"])) + 86400),
          f"GET /v1/state gives when the working state would be deleted: {state['working_state']}")


async def telemetry(harness, check):
    engine = harness.engine
    model = harness.frontend.models["alpha"]
    model.cold_prefill = chat_cache.PrefillSeed("alpha", {"points": [[1000, 2000], [100000, 400000]], "source": "test receipts"})
    content = text(90, 20)
    engine.drafts = (10, 7)
    status, _, reply = await harness.post("/v1/chat/completions", chat_body(content), account=ACCOUNT)
    cold = json.loads(reply)["sparkpipe"]
    status, _, reply = await harness.post("/v1/chat/completions", chat_body(content), account=ACCOUNT)
    warm = json.loads(reply)["sparkpipe"]
    engine.drafts = None
    expected = warm["cached_tokens"] * model.cold_prefill.rate(warm["prompt_tokens"])
    check(cold["pack"] is None and cold["draft_accept"] == 0.7 and cold["draft_tokens"] == 10 and cold["accepted_draft_tokens"] == 7,
          f"the sparkpipe object carries pack null until packs exist and the engine's own draft counts: {cold}")
    check(cold["cold_prefill_ms_estimate"] == 0 and warm["cached_tokens"] > 0 and warm["cold_prefill_ms_estimate"] == round(expected),
          f"with no observed rate the estimate comes from the model's measured seed, never null: {cold['cold_prefill_ms_estimate']} {warm}")
    seed = model.cold_prefill
    check(seed.rate(1000) == 2.0 and seed.rate(100000) == 4.0 and seed.rate(10) == 2.0 and seed.rate(10 ** 6) == 4.0 and abs(seed.rate(10000) - 3.0) < 1e-9,
          "the seed interpolates the measured ms per token over the log of the prompt length and holds the end rates outside them")
    for bad in ({"points": [], "source": "x"}, {"points": [[1, 2]], "source": ""}, {"points": [[1, 2], [1, 3]], "source": "x"}, {"points": [[0, 2]], "source": "x"},
                {"points": [[1, 2]]}):
        try:
            chat_cache.PrefillSeed("alpha", bad)
            check(False, f"a malformed cold_prefill seed is refused: {bad}")
        except ValueError:
            pass


def deployment_seeds(check):
    config = json.loads((ROOT / "deployment" / "hub" / "chat_frontend.json").read_text())
    seeded = 0
    for spec in config["models"]:
        if spec.get("cold_prefill") is None:
            continue
        seeded += 1
        seed = chat_cache.PrefillSeed(spec["id"], spec["cold_prefill"])
        receipts = [ROOT / word for word in seed.source.split() if word.endswith(".jsonl")]
        check(receipts and all(receipt.is_file() for receipt in receipts), f"{spec['id']} cold_prefill names receipts in the repository: {seed.source}")
        expected = []
        for receipt in (receipt for receipt in receipts if receipt.is_file()):
            for row in (json.loads(line) for line in receipt.read_text().splitlines() if line.strip()):
                if row.get("case") == "ttft6000":
                    expected.append([row["prompt_tokens"], round(1000 * row["prompt_tokens"] / row["prefill_tok_s"])])
                if row.get("case") == "warm" and row["cached_tokens"] < row["prompt_tokens"] // 2:
                    expected.append([row["prompt_tokens"], round(1000 * row["seconds"] * row["prompt_tokens"] / (row["prompt_tokens"] - row["cached_tokens"]))])
        check(sorted(expected) == sorted(spec["cold_prefill"]["points"]),
              f"{spec['id']} cold_prefill points are exactly the cold prefills its receipts measured: {spec['cold_prefill']['points']} against {expected}")
    check(seeded >= 14, f"every model with a release receipt has a cold prefill seed: {seeded}")


async def run(tmp, check):
    async with fixture.Harness(tmp) as harness:
        await commit_only(harness, check)
        await warm_then_match(harness, check)
        await pins(harness, check)
        await stored_state(harness, tmp, check)
        await telemetry(harness, check)
    await warm_yields(tmp, check)


def main():
    check = fixture.Checks("test_chat_frontend_cache_controls")
    captured = io.StringIO()
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(captured):
        asyncio.run(run(Path(directory), check))
    deployment_seeds(check)
    if check.failures:
        print(captured.getvalue()[-4000:])
    check.finish("commit_only on chat completions, Messages and Responses, and POST /v1/cache/warm in all three formats, commit the prompt at background "
                 "priority and return prompt, cached and committed tokens and the engine's prefill time, with the usual record and no output; the next "
                 "real request renders the same tokens and matches every warmed block; a warm-up never takes an interactive request's sequence; pins by "
                 "request or prefix id reach the engine as one set, refuse past half the cache, are named errors otherwise, stay within the account, and "
                 "go with the working state at a change of user and with DELETE /v1/cache; GET /v1/state lists chains, pins, capacity and use, and the "
                 "replay store as a scan finds it; the sparkpipe object carries the engine's draft counts and a cold prefill estimate seeded from receipts")


if __name__ == "__main__":
    main()
