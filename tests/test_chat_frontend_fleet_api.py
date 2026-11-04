#!/usr/bin/env python3
import asyncio
import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
chat_frontend = fixture.chat_frontend
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
CALL_REPLY = "Read it.</think>Checking.<tool_call>read<arg_key>path</arg_key><arg_value>a.txt</arg_value></tool_call>"
ANSWER = "Done.</think>It says hello."
LONG = " ".join(f"line {index} of the repository notes, kept in context." for index in range(160))


def chat(messages, **extra):
    body = {"model": "alpha", "messages": messages, "tools": TOOLS}
    body.update(extra)
    return body


async def gate_rules(check):
    gate = chat_frontend.SequenceGate(4)
    await gate.acquire(chat_frontend.BULK_PRIORITY, 1)
    await gate.acquire(chat_frontend.BULK_PRIORITY, 1)
    try:
        await gate.acquire(chat_frontend.BULK_PRIORITY, 0.05)
        check(False, "background work may not take the sequences kept for interactive use")
    except chat_frontend.RequestError as error:
        check(error.status == 429 and error.retry_after == chat_frontend.BUSY_RETRY_AFTER_SECONDS, "a background request past its share waits, then gets 429")
    await gate.acquire(chat_frontend.INTERACTIVE_PRIORITY, 0.05)
    await gate.acquire(chat_frontend.INTERACTIVE_PRIORITY, 0.05)
    check(gate.state()["in_use"] == 4, "interactive requests take the two kept sequences")
    waiting = asyncio.ensure_future(gate.acquire(chat_frontend.INTERACTIVE_PRIORITY, None))
    background = asyncio.ensure_future(gate.acquire(chat_frontend.BULK_PRIORITY, 1))
    await asyncio.sleep(0.05)
    await gate.release()
    await asyncio.sleep(0.05)
    check(waiting.done() and not background.done(), "a freed sequence goes to the waiting interactive request first")
    await gate.release()
    await gate.release()
    await asyncio.sleep(0.05)
    check(not background.done(), "background work never takes more than its share, even with sequences free")
    await gate.release()
    await background
    check(gate.state() == {"sequences": 4, "in_use": 2, "waiting_interactive": 0}, f"background work gets a sequence within its share: {gate.state()}")


async def run(tmp, check):
    await gate_rules(check)
    async with fixture.Harness(tmp) as harness:
        engine = harness.engine
        engine.reply(CALL_REPLY)
        status, headers, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "Read a.txt"}]))
        reply = json.loads(text)
        call = reply["choices"][0]["message"]["tool_calls"][0]
        check(status == 200 and call["function"]["name"] == "read" and reply["sparkpipe"]["model"] == "alpha" and headers.get("X-SparkPipe-Request") == reply["sparkpipe"]["request"],
              f"chat completions carries the sparkpipe object and the request header: {text[:300]}")
        engine.reply(ANSWER)
        history = [{"role": "user", "content": "Read a.txt"}, {"role": "assistant", "content": "Checking.", "tool_calls": [call]},
                   {"role": "tool", "tool_call_id": call["id"], "content": "hello"}]
        status, headers, text = await harness.post("/v1/chat/completions", chat(history, stream=True, stream_options={"include_usage": True}))
        chunks = [data for name, data in fixture.sse_events(text) if isinstance(data, dict)]
        check(engine.requests[1]["prompt_token_ids"][:len(engine.committed[0])] == engine.committed[0],
              "a chat completions tool call round trip replays the generated tokens, keyed by tool_calls[].id")
        final = [chunk for chunk in chunks if chunk.get("choices") and chunk["choices"][0].get("finish_reason")]
        check(final and final[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == len(engine.committed[0]) and final[0]["sparkpipe"]["replayed_tokens"] == len(engine.committed[0]),
              f"the final chunk reports the cached tokens and the sparkpipe object: {final}")
        check(chunks[-1].get("sparkpipe") is not None and chunks[-1]["choices"] == [], "the usage chunk, the last before [DONE], carries the sparkpipe object too")

        chat_frontend.KEEPALIVE_SECONDS = 0.1
        engine.delay = 0.35
        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "slow"}], stream=True))
        engine.delay = 0.0
        events = fixture.sse_events(text)
        check(status == 200 and ("comment", None) in events[:4] and events[-1] == (None, "[DONE]"), f"a long prefill sends SSE comment lines on chat completions: {events[:5]}")

        status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}], model="beta"))
        check(status == 409 and json.loads(text)["error"]["code"] == "model_not_loaded", f"chat completions refuses a swap with 409 too: {status} {text}")
        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}], model="gpt-4o"))
        check(status == 200 and json.loads(text)["sparkpipe"]["model"] == "alpha", "chat completions resolves gpt-* aliases")
        status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}], model="alpha+core"))
        check(status == 404 and json.loads(text)["error"]["code"] == "pack_not_found", f"a pack alias names a pack that must exist: {status} {text}")

        body = {"model": "alpha", "messages": [{"role": "user", "content": "Count me"}], "tools": TOOLS}
        status, _, text = await harness.post("/v1/tokenize", body)
        counted = json.loads(text)
        engine.reply(ANSWER)
        await harness.post("/v1/chat/completions", body)
        check(status == 200 and counted["tokens"] == engine.requests[-1]["prompt_token_ids"] and counted["count"] == len(counted["tokens"]),
              "tokenize renders messages with the model's template to exactly the engine's prompt")
        status, _, text = await harness.post("/v1/tokenize", {"model": "alpha", "text": "hello world"})
        tokens = json.loads(text)["tokens"]
        status, _, text = await harness.post("/v1/detokenize", {"model": "alpha", "tokens": tokens})
        check(status == 200 and json.loads(text)["text"] == "hello world", f"detokenize inverts tokenize: {text}")
        status, _, text = await harness.post("/v1/detokenize", {"model": "alpha", "tokens": ["x"]})
        check(status == 400, "detokenize refuses what is not a token id")

        async with harness.session.get(harness.base + "/v1/models", headers=harness.headers()) as response:
            listing = await response.json()
        resident = {entry["id"]: entry["resident"] for entry in listing["data"]}
        check(listing["data"][0]["id"] == "alpha" and resident == {"alpha": True, "alpha-bulk": True, "beta": False, "beta-bulk": False} and
              listing["data"][0]["type"] == "model" and listing["data"][0]["context_window"] == 65536 and listing["has_more"] is False and
              listing["aliases"] == {"claude-*": None, "gpt-*": None}, f"the model list puts resident models first and shows the alias table: {listing}")
        async with harness.session.put(harness.base + "/v1/models/aliases", json={"claude-*": "nope"}, headers=harness.headers()) as response:
            check(response.status == 400, "an alias to a model that is not served is refused")
        async with harness.session.put(harness.base + "/v1/models/aliases", json={"claude-*": "beta", "o3": "alpha"}, headers=harness.headers()) as response:
            check(response.status == 200, "an alias table naming served models is kept")
        async with harness.session.get(harness.base + "/v1/models/aliases", headers=harness.headers()) as response:
            table = await response.json()
        check(table == {"aliases": {"claude-*": "beta", "o3": "alpha"}, "first_resident": "alpha"}, f"the alias table reads back with the first resident model: {table}")
        async with harness.session.get(harness.base + "/v1/models/aliases", headers=harness.headers(swap=True)) as response:
            other = await response.json()
        check(other["aliases"] == {"claude-*": None, "gpt-*": None}, "each account has its own alias table, starting from the default")

        harness.frontend.replay.db.execute("DELETE FROM prefill")
        engine.delay = 0.3
        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/chat/completions", {"model": "alpha", "messages": [{"role": "user", "content": LONG}]})
        cold = json.loads(text)["sparkpipe"]
        engine.delay = 0.0
        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/chat/completions", {"model": "alpha", "messages": [{"role": "user", "content": LONG + " And one more."}]})
        warm = json.loads(text)["sparkpipe"]
        rate = cold["prefill_ms"] / cold["prefill_tokens"]
        expected = warm["cached_tokens"] * rate
        check(cold["prefill_tokens"] >= 512 and warm["cached_tokens"] > 0 and warm["cold_prefill_ms_estimate"] is not None and
              abs(warm["cold_prefill_ms_estimate"] - expected) <= 0.1 * expected + 1,
              f"cold_prefill_ms_estimate applies the measured prefill rate at this length to the cached tokens: {cold} {warm}")
        check(cold["cold_prefill_ms_estimate"] == 0 and warm["ttft_ms"] < cold["ttft_ms"], "a cold request saved nothing and the warm one starts sooner")

        chat_frontend.BUSY_WAIT_SECONDS = 0.2
        engine.hold = asyncio.Event()
        held = [asyncio.ensure_future(harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hold"}]))) for _ in range(2)]
        await asyncio.sleep(0.1)
        status, headers, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "more"}]))
        check(status == 429 and headers.get("Retry-After") == str(chat_frontend.BUSY_RETRY_AFTER_SECONDS) and headers.get("x-should-retry") == "true",
              f"a keyed chat request past 60 s of busy sequences gets 429 with retry-after: {status} {headers}")
        inside = asyncio.ensure_future(harness.session.post(harness.base + "/v1/chat/completions", json=chat([{"role": "user", "content": "inside"}])))
        await asyncio.sleep(0.4)
        check(not inside.done(), "an internal request waits for a sequence without a time limit, as the door has always queued")
        engine.hold.set()
        await asyncio.gather(*held)
        response = await inside
        check(response.status == 200, "the internal request is served once a sequence frees")
        response.release()
        engine.hold = None


def main():
    check = fixture.Checks("test_chat_frontend_fleet_api")
    captured = io.StringIO()
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(captured):
        asyncio.run(run(Path(directory), check))
    if check.failures:
        print(captured.getvalue()[-4000:])
    check.finish("background requests leave two sequences for interactive ones; chat completions replays tool calls by id, reports cached tokens and the "
                 "sparkpipe object, sends keep-alive comments, resolves aliases and refuses swaps with 409; tokenize matches the engine's prompt and "
                 "detokenize inverts it; /v1/models lists resident models first with the alias table; aliases are per account; cold_prefill_ms_estimate "
                 "uses the measured rate; keyed requests get 429 after the busy wait while internal ones queue")


if __name__ == "__main__":
    main()
