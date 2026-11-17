#!/usr/bin/env python3
import asyncio
import importlib.util
import json
import random
import tempfile
import time
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
chat_frontend = fixture.chat_frontend
ANSWER = "Done.</think>It says hello."
HOLDER_REQUESTS = 64
SEQUENCES = 8


class TrackingEngine(fixture.Engine):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.open = 0
        self.peak = 0
        self.transports = []

    async def completions(self, request):
        self.open += 1
        self.peak = max(self.peak, self.open)
        self.transports.append(request.transport)
        try:
            return await super().completions(request)
        finally:
            self.open -= 1


fixture.Engine = TrackingEngine


def chat(content, **extra):
    return dict({"model": "alpha", "messages": [{"role": "user", "content": content}]}, **extra)


def messages(content, **extra):
    return dict({"model": "alpha", "max_tokens": 64, "messages": [{"role": "user", "content": content}]}, **extra)


async def until(condition, seconds):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    return condition()


async def priority_rules(harness, check):
    engine = harness.engine
    cases = [("/v1/messages", messages("hi", metadata={"x-sparkpipe-priority": "background"}), {}, chat_frontend.BULK_PRIORITY,
              "a Messages request whose metadata says background runs below interactive requests"),
             ("/v1/responses", {"model": "alpha", "input": "hi", "metadata": {"x-sparkpipe-priority": "background"}}, {}, chat_frontend.BULK_PRIORITY,
              "a Responses request whose metadata says background runs below interactive requests"),
             ("/v1/chat/completions", chat("hi", metadata={"x-sparkpipe-priority": "background"}), {}, chat_frontend.BULK_PRIORITY,
              "a chat completions request whose metadata says background runs below interactive requests"),
             ("/v1/messages", messages("hi", metadata={"X-SparkPipe-Priority": "background"}), {}, chat_frontend.BULK_PRIORITY,
              "the metadata key matches the header's name in any case"),
             ("/v1/messages", messages("hi", metadata={"x-sparkpipe-priority": "background"}), {"X-SparkPipe-Priority": "interactive"},
              chat_frontend.INTERACTIVE_PRIORITY, "the x-sparkpipe-priority header wins over the same field in metadata"),
             ("/v1/messages", messages("hi", metadata={"user_id": "user_x"}), {}, chat_frontend.INTERACTIVE_PRIORITY,
              "a request that names no class is interactive")]
    for path, body, headers, expected, message in cases:
        engine.reply(ANSWER)
        status, _, text = await harness.post(path, body, account="acme", **headers)
        check(status == 200 and engine.requests[-1]["priority"] == expected, f"{message}: {status} {engine.requests[-1].get('priority')} {text[:160]}")
    status, _, text = await harness.post("/v1/messages", messages("hi", metadata={"x-sparkpipe-priority": "later"}), account="acme")
    check(status == 400 and "x-sparkpipe-priority" in text, f"a metadata class that is neither interactive nor background is a named 400: {status} {text[:200]}")


async def cancel_rules(harness, check):
    engine, frontend = harness.engine, harness.frontend
    gate = frontend.models["alpha"].sequences
    engine.hold = asyncio.Event()
    engine.reply(ANSWER)
    async with harness.session.post(harness.base + "/v1/messages", json=messages("hold", stream=True), headers=harness.headers(account="acme")) as response:
        identifier = response.headers.get("X-SparkPipe-Request")
        check(response.status == 200 and identifier is not None and identifier in frontend.live,
              f"a streamed reply carries its request id in X-SparkPipe-Request once the stream opens: {response.status} {dict(response.headers)}")
        await until(lambda: engine.open == 1, 2)
        for headers, label in ((dict(account="bob"), "another account"), (dict(Authorization="Bearer " + harness.key), "a key of another scope"),
                               ({"account": "acme", "X-SparkPipe-Member": "MBOTHER"}, "another member of the account")):
            async with harness.session.delete(f"{harness.base}/v1/requests/{identifier}", headers=harness.headers(**headers)) as refused:
                body = await refused.json()
            check(refused.status == 404 and body["error"]["code"] == "request_not_found" and identifier in frontend.live,
                  f"{label} cannot see or cancel the request: {refused.status} {body}")
        async with harness.session.delete(f"{harness.base}/v1/requests/r_{'0' * 24}", headers=harness.headers(account="acme")) as missing:
            check(missing.status == 404, "an id that is not running answers 404")
        started = time.monotonic()
        async with harness.session.delete(f"{harness.base}/v1/requests/{identifier}", headers=harness.headers(account="acme")) as cancelled:
            reply = await cancelled.json()
        elapsed = time.monotonic() - started
        check(cancelled.status == 200 and reply == {"id": identifier, "object": "request", "cancelled": True, "released": True} and elapsed < 1.0,
              f"the account's own DELETE cancels its running request and answers once its sequence is free: {cancelled.status} {reply} {elapsed:.2f} s")
        text = (await response.read()).decode()
    events = fixture.sse_events(text)
    check(any(name == "error" and isinstance(data, dict) and "cancelled with DELETE" in data["error"]["message"] for name, data in events),
          f"the cancelled stream ends with a named error event, never a silent cut: {events[-3:]}")
    check(gate.state()["in_use"] == 0 and identifier not in frontend.live, f"the cancelled request holds no sequence: {gate.state()}")
    check(await until(lambda: all(transport is None or transport.is_closing() for transport in engine.transports), 2),
          "the chat layer closes the engine stream, so the engine API sees the disconnect and frees the engine sequence")
    engine.hold.set()
    engine.hold = None

    engine.hold = asyncio.Event()
    held = [asyncio.ensure_future(harness.post("/v1/chat/completions", chat(f"fill {index}"), account="acme")) for index in range(SEQUENCES)]
    await until(lambda: gate.state()["in_use"] == SEQUENCES, 2)
    target = "r_" + "c0ffee" * 4
    queued = asyncio.ensure_future(harness.post("/v1/chat/completions", chat("queued"), account="acme", **{"X-SparkPipe-Request": target}))
    await until(lambda: gate.state()["waiting"] == 1, 2)
    status, _, text = await harness.post("/v1/chat/completions", chat("again"), account="acme", **{"X-SparkPipe-Request": target})
    check(status == 409 and json.loads(text)["error"]["code"] == "request_id_in_use", f"an id already queued or running is refused: {status} {text[:200]}")
    status, _, text = await harness.post("/v1/chat/completions", chat("bad"), account="acme", **{"X-SparkPipe-Request": "nope"})
    check(status == 400 and json.loads(text)["error"]["code"] == "invalid_request", f"a malformed chosen id is a 400: {status} {text[:200]}")
    async with harness.session.delete(f"{harness.base}/v1/requests/{target}", headers=harness.headers(account="acme")) as cancelled:
        check(cancelled.status == 200, f"a request still waiting for a sequence is cancelled by the id its client chose in X-SparkPipe-Request: {cancelled.status}")
    status, _, text = await queued
    check(status == 409 and json.loads(text)["error"]["code"] == "request_cancelled" and gate.state()["waiting"] == 0 and gate.state()["in_use"] == SEQUENCES,
          f"the waiting request gets the named error and leaves the queue, and no running request is touched: {status} {text[:200]} {gate.state()}")
    engine.hold.set()
    results = await asyncio.gather(*held)
    check(all(status == 200 for status, _, _ in results), "the running requests finish normally")
    engine.hold = None


async def holder_rules(harness, check):
    engine, frontend = harness.engine, harness.frontend
    gate = frontend.models["alpha"].sequences
    check(chat_frontend.BUSY_WAIT_SECONDS == 60.0 and chat_frontend.BUSY_RETRY_AFTER_SECONDS > 0,
          "the contract: a request waits 60 s for a sequence before 429 busy with a retry-after")
    chat_frontend.BUSY_WAIT_SECONDS = 1.0
    engine.hold = asyncio.Event()
    engine.peak = 0
    before = len(engine.requests)
    started = time.monotonic()
    tasks = [asyncio.ensure_future(harness.post("/v1/chat/completions", chat(f"agent {index}"), account="acme", **{"X-SparkPipe-Key": "k_holder"}))
             for index in range(HOLDER_REQUESTS)]
    await asyncio.sleep(0.5)
    check(not any(task.done() for task in tasks) and len(engine.requests) - before == SEQUENCES and gate.state()["waiting"] == HOLDER_REQUESTS - SEQUENCES,
          f"a holder key sending {HOLDER_REQUESTS} requests is never refused by a per-key limit: {SEQUENCES} run and the rest wait: "
          f"{sum(task.done() for task in tasks)} answered, {len(engine.requests) - before} at the engine, {gate.state()}")
    await asyncio.wait(tasks, timeout=5.0, return_when=asyncio.FIRST_COMPLETED)
    first_busy = time.monotonic() - started
    await until(lambda: sum(task.done() for task in tasks) >= HOLDER_REQUESTS - SEQUENCES, 5.0)
    busy = [task.result() for task in tasks if task.done() and task.result()[0] == 429]
    check(len(busy) == HOLDER_REQUESTS - SEQUENCES and all(headers.get("Retry-After") == str(chat_frontend.BUSY_RETRY_AFTER_SECONDS) and headers.get("x-should-retry") == "true"
                                                         and json.loads(text)["error"]["code"] == "busy" for _, headers, text in busy),
          f"only after the wait do the requests beyond the fleet's sequences get 429 busy with retry-after: {len(busy)} of {HOLDER_REQUESTS - SEQUENCES}")
    check(first_busy >= chat_frontend.BUSY_WAIT_SECONDS, f"no 429 comes before the full wait: {first_busy:.2f} s")
    engine.hold.set()
    results = await asyncio.gather(*tasks)
    check(sum(status == 200 for status, _, _ in results) == SEQUENCES, "the requests that held sequences complete")
    engine.hold = None

    chat_frontend.BUSY_WAIT_SECONDS = 30.0
    engine.peak = 0
    for _ in range(HOLDER_REQUESTS):
        engine.reply(ANSWER)
    results = await asyncio.gather(*(harness.post("/v1/chat/completions", chat(f"burst {index}"), account="acme", **{"X-SparkPipe-Key": "k_holder"})
                                     for index in range(HOLDER_REQUESTS)))
    check(all(status == 200 for status, _, _ in results) and engine.peak <= SEQUENCES,
          f"{HOLDER_REQUESTS} concurrent holder requests all complete, at most {SEQUENCES} at the engine at once: "
          f"{[status for status, _, _ in results if status != 200][:5]} peak {engine.peak}")
    chat_frontend.BUSY_WAIT_SECONDS = 60.0


async def memory_rules(harness, check):
    engine, frontend = harness.engine, harness.frontend
    model = frontend.models["alpha"]
    gate = model.sequences
    limit = SEQUENCES - chat_frontend.INTERACTIVE_RESERVE
    for _ in range(limit):
        await gate.acquire(chat_frontend.INTERACTIVE_PRIORITY, None)
    before = len(engine.requests)
    engine.reply(ANSWER)
    job = asyncio.ensure_future(frontend.memory_complete(model, engine.tokens("summarize"), 8))
    await asyncio.sleep(0.3)
    check(not job.done() and len(engine.requests) == before and gate.state()["waiting"] == 1,
          f"a chat-memory job waits at the gate behind interactive work and never takes the two kept sequences: {gate.state()} {len(engine.requests) - before}")
    await gate.release()
    await asyncio.wait_for(job, 5)
    check(len(engine.requests) == before + 1 and engine.requests[-1]["priority"] == chat_frontend.BACKGROUND_PRIORITY and gate.state()["in_use"] == limit - 1,
          f"once a sequence is free within background's share the memory job runs at background priority and gives its sequence back: {gate.state()}")
    for _ in range(limit - 1):
        await gate.release()


async def fair_share_rules(check):
    rng = random.Random(1794)
    gate = chat_frontend.SequenceGate(SEQUENCES)
    waiting = {"m1": [], "m2": []}

    def flood(member):
        waiting[member].append(asyncio.ensure_future(gate.acquire(chat_frontend.INTERACTIVE_PRIORITY, None, "account:acme", member)))

    for _ in range(12):
        flood("m1")
    await asyncio.sleep(0)
    for _ in range(12):
        flood("m2")
    await asyncio.sleep(0)
    violations, settled = [], False
    for step in range(400):
        held = {member: gate.held_members[("account:acme", member)] for member in waiting}
        choices = [member for member in waiting if held[member] > 0]
        member = rng.choice(choices)
        await gate.release("account:acme", member)
        flood(member)
        await asyncio.sleep(0)
        held = {member: gate.held_members[("account:acme", member)] for member in waiting}
        both_wait = all(any(not task.done() for task in waiting[member]) for member in waiting)
        settled = settled or max(held.values()) <= gate.share()
        if settled and both_wait and max(held.values()) > gate.share():
            violations.append((step, held))
    check(settled and not violations, f"two members flooding one account: once settled, neither holds more than three quarters of the sequences while the other waits: {violations[:3]}")
    for tasks in waiting.values():
        for task in tasks:
            task.cancel()
    await asyncio.gather(*(task for tasks in waiting.values() for task in tasks), return_exceptions=True)


async def run(tmp, check):
    await fair_share_rules(check)
    chat_frontend.KEEPALIVE_SECONDS = 0.1
    async with fixture.Harness(tmp, sequences=SEQUENCES) as harness:
        await priority_rules(harness, check)
        await cancel_rules(harness, check)
        await holder_rules(harness, check)
        await memory_rules(harness, check)


def main():
    check = fixture.Checks("test_chat_frontend_interactive_first")
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(run(Path(directory), check))
    check.finish("interactive and background classes from header or metadata; DELETE /v1/requests/{id} for the own account only, with a named error and "
                 "the sequence freed; a holder's 64 requests wait, then 429 busy; memory jobs at the gate; fair share between flooding members")


if __name__ == "__main__":
    main()
