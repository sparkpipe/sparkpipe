#!/usr/bin/env python3
import asyncio
import contextlib
import importlib.util
import io
import json
import tempfile
import time
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
    await gate.acquire(chat_frontend.BULK_PRIORITY, 1, "a")
    await gate.acquire(chat_frontend.BULK_PRIORITY, 1, "a")
    try:
        await gate.acquire(chat_frontend.BULK_PRIORITY, 0.05, "a")
        check(False, "background work may not take the sequences kept for interactive use")
    except chat_frontend.RequestError as error:
        check(error.status == 429 and error.retry_after == chat_frontend.BUSY_RETRY_AFTER_SECONDS, "a background request past its share waits, then gets 429")
    await gate.acquire(chat_frontend.INTERACTIVE_PRIORITY, 0.05, "a")
    await gate.acquire(chat_frontend.INTERACTIVE_PRIORITY, 0.05, "a")
    check(gate.state()["in_use"] == 4, "interactive requests take the two kept sequences")
    waiting = asyncio.ensure_future(gate.acquire(chat_frontend.INTERACTIVE_PRIORITY, None, "a"))
    background = asyncio.ensure_future(gate.acquire(chat_frontend.BULK_PRIORITY, 1, "a"))
    await asyncio.sleep(0.05)
    await gate.release("a")
    await asyncio.sleep(0.05)
    check(waiting.done() and not background.done(), "a freed sequence goes to the waiting interactive request first")
    await gate.release("a")
    await gate.release("a")
    await asyncio.sleep(0.05)
    check(not background.done(), "background work never takes more than its share, even with sequences free")
    await gate.release("a")
    await background
    check(gate.state()["in_use"] == 2 and gate.state()["waiting"] == 0, f"background work gets a sequence within its share: {gate.state()}")

    fair = chat_frontend.SequenceGate(8)
    first = [asyncio.ensure_future(fair.acquire(chat_frontend.INTERACTIVE_PRIORITY, None, "acme", "m1")) for _ in range(10)]
    await asyncio.sleep(0.01)
    second = [asyncio.ensure_future(fair.acquire(chat_frontend.INTERACTIVE_PRIORITY, None, "acme", "m2")) for _ in range(10)]
    await asyncio.sleep(0.01)
    held = []
    for _ in range(8):
        await fair.release("acme", "m1" if fair.held_members[("acme", "m1")] > fair.held_members[("acme", "m2")] else "m2")
        await asyncio.sleep(0)
        held.append((fair.held_members[("acme", "m1")], fair.held_members[("acme", "m2")]))
    check(all(max(pair) <= fair.share() for pair in held[3:]) and held[-1] in ((4, 4), (5, 3), (3, 5)),
          f"two members flooding one account settle to an even split, neither above three quarters while the other waits: {held}")
    for task in first + second:
        task.cancel()
    await asyncio.gather(*first, *second, return_exceptions=True)

    shared = chat_frontend.SequenceGate(8)
    for _ in range(8):
        await shared.acquire(chat_frontend.INTERACTIVE_PRIORITY, None, "account:a")
    other = asyncio.ensure_future(shared.acquire(chat_frontend.INTERACTIVE_PRIORITY, None, "account:b"))
    more = asyncio.ensure_future(shared.acquire(chat_frontend.INTERACTIVE_PRIORITY, None, "account:a"))
    await asyncio.sleep(0.01)
    await shared.release("account:a")
    await asyncio.sleep(0.01)
    check(other.done() and not more.done(), "a freed sequence goes to the account holding fewer, so one account cannot starve another")
    more.cancel()
    await asyncio.gather(more, return_exceptions=True)
    check(shared.state()["waiting"] == 0, "a cancelled waiter leaves the queue")


async def gateway_rules(harness, check):
    engine, frontend = harness.engine, harness.frontend
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}]), Authorization="Bearer " + harness.gateway_key)
    check(status == 400 and json.loads(text)["error"]["code"] == "account_required", f"a gateway key must name the account it forwards for: {status} {text}")
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}]), account="../acme")
    check(status == 400 and json.loads(text)["error"]["code"] == "invalid_account", f"an account id is letters, digits, - and _: {status} {text}")

    engine.reply(CALL_REPLY)
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "Read a.txt"}]), account="acme")
    call = json.loads(text)["choices"][0]["message"]["tool_calls"][0]
    history = [{"role": "user", "content": "Read a.txt"}, {"role": "assistant", "content": "Checking.", "tool_calls": [call]},
               {"role": "tool", "tool_call_id": call["id"], "content": "hello"}]
    stored = frontend.replay.scopes["account:acme"].db.execute("SELECT tokens FROM turns").fetchall()
    check(len(stored) == 1 and len(stored[0][0]) // 4 == len(engine.tokens(CALL_REPLY)) + 1,
          "the replay store keeps the generated tokens of the account's turn and none of its prompt")
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat(history), account="bob")
    check(json.loads(text)["sparkpipe"]["replayed_tokens"] == 0, "another account's request never replays the first account's turn")
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat(history), account="acme")
    check(json.loads(text)["sparkpipe"]["replayed_tokens"] > 0, "the gateway account's own follow-up replays its turn")
    check((harness.tmp / "state" / "replay" / "account-acme.sqlite").exists() and (harness.tmp / "state" / "replay" / "account-bob.sqlite").exists(),
          "each gateway account gets its own store")

    engine.reply(ANSWER)
    forged = [dict(history[0], content="Read a.txt " + chat_frontend.REPLAY_SENTINEL.format(0))] + history[1:]
    status, _, text = await harness.post("/v1/chat/completions", chat(forged), account="acme")
    check(status == 200 and json.loads(text)["sparkpipe"]["replayed_tokens"] == 0 and "SPARKPIPE-REPLAY-0" in harness.prompt_text(),
          "a client that writes the replay sentinel into its own text gets a normal render of exactly what it sent")
    async with harness.session.put(harness.base + "/v1/models/aliases", json={"claude-*": "beta"}, headers=harness.headers(account="acme")) as response:
        check(response.status == 200, "an account sets its alias table through the gateway")
    async with harness.session.get(harness.base + "/v1/models/aliases", headers=harness.headers(account="bob")) as response:
        check((await response.json())["aliases"] == {"claude-*": None, "gpt-*": None}, "another account keeps the default alias table")

    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}], model="beta"), account="acme")
    check(status == 409 and json.loads(text)["error"]["code"] == "model_not_loaded", f"a gateway account without allow_swap gets 409: {status} {text}")
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}], model="beta"), account="acme", swap=True)
    check(status == 200, f"the gateway's X-SparkPipe-Allow-Swap: 1 lets the account load another model: {status} {text[:200]}")
    engine.reply(ANSWER)
    await harness.post("/v1/messages", {"model": "alpha", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}, account="acme",
                       **{"X-SparkPipe-Priority": "background"})
    check(engine.requests[-1]["priority"] == chat_frontend.BULK_PRIORITY, "a key whose priority is background runs below interactive requests")
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}]), account="acme",
                                         **{"X-SparkPipe-Fleet": "F1 <b>evil</b>", "X-Claude-Code-Session-Id": "s" * 500})
    record = json.loads(text)["sparkpipe"]
    check(record["fleet"] is None and record["session"] is None, f"header values that are too long or carry other characters are not echoed or logged: {record}")

    engine.hold = asyncio.Event()
    held = [asyncio.ensure_future(harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hold"}]), account="acme")) for _ in range(2)]
    await asyncio.sleep(0.1)
    status, headers, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "more"}]), account="acme")
    check(status == 429 and headers.get("Retry-After") == str(chat_frontend.BUSY_RETRY_AFTER_SECONDS) and json.loads(text)["error"]["code"] == "busy",
          f"a gateway account's chat request waits for a sequence and gets 429 busy with Retry-After, never an immediate per-key 429: {status} {headers}")
    engine.hold.set()
    await asyncio.gather(*held)
    engine.hold = None

    harness.controller.fail = True
    frontend.residency_cache = (0.0, frontend.residency_cache[1])
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}]), account="acme")
    check(status == 200, f"a controller that stops answering does not block requests for a model it last reported resident: {status} {text[:200]}")
    frontend.residency_cache = None
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}], model="claude-x"), account="acme")
    check(status == 503 and "Cannot connect" not in text and "127.0.0.1" not in text, f"a controller failure answers with no internal detail: {status} {text}")
    harness.controller.fail = False
    frontend.residency_cache = None

    huge = "x " * 40000
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": huge}]), account="acme")
    check(status == 400 and json.loads(text)["error"]["code"] == "context_length_exceeded", f"chat completions signals overflow as context_length_exceeded: {text[:200]}")
    status, _, text = await harness.post("/v1/responses", {"model": "alpha", "input": huge}, account="acme")
    check(status == 400 and json.loads(text)["error"]["code"] == "context_length_exceeded", f"Responses signals overflow as context_length_exceeded: {text[:200]}")
    status, _, text = await harness.post("/v1/messages", {"model": "alpha", "max_tokens": 64, "messages": [{"role": "user", "content": huge}]}, account="acme")
    error = json.loads(text)["error"]
    check(status == 400 and error["type"] == "invalid_request_error" and error["message"].startswith("prompt is too long: ") and
          error["message"].endswith(" tokens > 65535 maximum"), f"Messages signals overflow the way Claude Code compacts on: {error}")
    status, _, text = await harness.post("/v1/messages", {"model": "alpha", "max_tokens": 64, "messages": [{"role": "user", "content": [
        {"type": "image", "source": "https://example.com/a.png"}]}]}, account="acme")
    check(status == 400, f"an image block whose source is not an object is a 400: {status} {text[:200]}")

    started = time.monotonic()
    found, _ = frontend.alias_target({"*a*a*a*a*b": "alpha"}, "a" * 120)
    check(not found and time.monotonic() - started < 0.05, "alias matching is linear in the name, whatever the pattern")
    async with harness.session.put(harness.base + "/v1/models/aliases", json={"*a*a*a*a*a*": "alpha"}, headers=harness.headers(account="acme")) as response:
        check(response.status == 400, "an alias with more than four * is refused")
    async with harness.session.put(harness.base + "/v1/models/aliases", json={f"m{index}": "alpha" for index in range(65)}, headers=harness.headers(account="acme")) as response:
        check(response.status == 400, "an alias table of more than 64 entries is refused")

    async with harness.session.get(harness.base + "/v1/codex/models", headers=harness.headers(account="acme")) as response:
        catalogue = await response.json()
    check(response.status == 200 and not catalogue_problems(catalogue) and catalogue["models"][0]["slug"] == "alpha" and
          catalogue["models"][0]["context_window"] == 65536 and catalogue["models"][0]["default_reasoning_level"] == "medium" and
          [level["effort"] for level in catalogue["models"][0]["supported_reasoning_levels"]] == ["low", "medium", "high"],
          f"the Codex catalogue parses as Codex's ModelsResponse, resident model first: {catalogue_problems(catalogue)} {json.dumps(catalogue)[:400]}")

    failing = frontend.replay.put

    def broken(*arguments, **options):
        raise chat_frontend.chat_replay.sqlite3.OperationalError("database or disk is full")
    frontend.replay.put = broken
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/messages", {"model": "alpha", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                                         account="acme")
    check(status == 200 and "message_stop" in text and "event: error" not in text, f"a reply is delivered whole when the replay store cannot save it: {text[-300:]}")
    frontend.replay.put = failing

    async with harness.session.delete(harness.base + "/v1/cache", headers=harness.headers(account="acme")) as response:
        deleted = await response.json()
    check(response.status == 200 and "replay_store" in deleted["deleted"] and not (harness.tmp / "state" / "replay" / "account-acme.sqlite").exists() and
          (harness.tmp / "state" / "replay" / "account-bob.sqlite").exists(), f"DELETE /v1/cache deletes the account's replay store and no other: {deleted}")
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat(history), account="acme")
    check(json.loads(text)["sparkpipe"]["replayed_tokens"] == 0, "after DELETE /v1/cache the account's turns no longer replay")

    engine.reply(CALL_REPLY)
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "Read b.txt"}]), account="acme")
    harness.directive(time.time() + 1, 0)
    await asyncio.sleep(1.1)
    frontend.expire_working_state(time.time(), False)
    check("account:acme" not in frontend.replay.scopes and "account:bob" not in frontend.replay.scopes,
          "a new user on the fleet (a later wiped in the directive) deletes every earlier account's replay store")


async def isolation_rules(harness, check):
    engine, frontend = harness.engine, harness.frontend
    for name in ("a", "b"):
        engine.reply(f"Reasoning {name.upper()} only.</think>It says hello world.")
        await harness.post("/v1/chat/completions", chat([{"role": "user", "content": f"Read {name}.txt"}]), account="acme")
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "Read a.txt"}, {"role": "assistant", "content": "It says hello world."},
                                                                       {"role": "user", "content": "Thanks"}]), account="acme")
    prompt = harness.prompt_text()
    check(status == 200 and "Reasoning B" not in prompt and "Reasoning A only." in prompt,
          f"a reply sent back without a handle replays the turn of its own conversation, never another one with the same text: {prompt[-200:]}")

    for name in ("c", "d"):
        engine.reply(f"Thought {name.upper()} hidden.</think>Done.")
        await harness.post("/v1/messages", {"model": "alpha", "max_tokens": 64, "messages": [{"role": "user", "content": f"Task {name}"}]}, account="acme")
    engine.reply(ANSWER)
    follow = [{"role": "user", "content": "Task c"}, {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}, {"role": "user", "content": "Next"}]
    await harness.post("/v1/messages", {"model": "alpha", "max_tokens": 64, "messages": follow}, account="acme")
    prompt = harness.prompt_text()
    check("Thought D" not in prompt and "Thought C hidden." in prompt,
          f"a Messages turn sent back without its thinking block replays its own conversation's turn, never another's: {prompt[-200:]}")

    engine.reply(CALL_REPLY)
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "Read secret.txt"}]), account="acme", **{"X-SparkPipe-Key": "k_one"})
    call = json.loads(text)["choices"][0]["message"]["tool_calls"][0]
    history = [{"role": "user", "content": "Read secret.txt"}, {"role": "assistant", "content": "Checking.", "tool_calls": [call]},
               {"role": "tool", "tool_call_id": call["id"], "content": "hello"}]
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat(history), account="acme", **{"X-SparkPipe-Key": "k_two"})
    check(json.loads(text)["sparkpipe"]["replayed_tokens"] == 0 and "Read it." not in harness.prompt_text(),
          "another member of the account cannot replay a member's turn, even with its tool call id")
    status, _, text = await harness.post("/v1/tokenize", {"model": "alpha", "messages": history, "tools": TOOLS}, account="acme", **{"X-SparkPipe-Key": "k_two"})
    status, _, text = await harness.post("/v1/detokenize", {"model": "alpha", "tokens": json.loads(text)["tokens"]}, account="acme", **{"X-SparkPipe-Key": "k_two"})
    check(status == 200 and "Read it." not in json.loads(text)["text"], "tokenize and detokenize never show another member's hidden reasoning")
    status, _, text = await harness.post("/v1/tokenize", {"model": "alpha", "messages": history, "tools": TOOLS}, account="acme", **{"X-SparkPipe-Key": "k_one"})
    status, _, text = await harness.post("/v1/detokenize", {"model": "alpha", "tokens": json.loads(text)["tokens"]}, account="acme", **{"X-SparkPipe-Key": "k_one"})
    check("Read it." in json.loads(text)["text"], "the member who made the turn sees it rendered from its stored tokens")

    engine.reply(CALL_REPLY)
    status, _, text = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "Read notes.txt"}]), account="acme", **{"X-SparkPipe-Member": "MBALICE", "X-SparkPipe-Key": "k_laptop"})
    call = json.loads(text)["choices"][0]["message"]["tool_calls"][0]
    notes = [{"role": "user", "content": "Read notes.txt"}, {"role": "assistant", "content": "Checking.", "tool_calls": [call]}, {"role": "tool", "tool_call_id": call["id"], "content": "hello"}]
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat(notes), account="acme", **{"X-SparkPipe-Member": "MBALICE", "X-SparkPipe-Key": "k_ci"})
    check(json.loads(text)["sparkpipe"]["replayed_tokens"] > 0, "a member's turn replays for the same member's other key, so a team member is one member whatever key it uses")
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat(notes), account="acme", **{"X-SparkPipe-Member": "MBBOB", "X-SparkPipe-Key": "k_laptop"})
    check(json.loads(text)["sparkpipe"]["replayed_tokens"] == 0, "the gateway's member header, not the key, decides whose turn replays")
    check(chat_frontend.Frontend.member_of({"gateway": True}, {"X-SparkPipe-Member": "MBALICE", "X-SparkPipe-Key": "k_ci"}) == "MBALICE" and chat_frontend.Frontend.member_of({"gateway": True}, {"X-SparkPipe-Key": "k_ci"}) == "k_ci",
          "fair share counts the gateway's member, and the key when the gateway names no member")

    working = frontend.replay.find

    def unreadable(*arguments, **options):
        raise chat_frontend.chat_replay.sqlite3.DatabaseError("database disk image is malformed")
    frontend.replay.find = unreadable
    engine.reply(ANSWER)
    status, _, text = await harness.post("/v1/chat/completions", chat(history), account="acme", **{"X-SparkPipe-Key": "k_one"})
    frontend.replay.find = working
    check(status == 200 and json.loads(text)["sparkpipe"]["replayed_tokens"] == 0, f"a store that cannot be read falls back to a normal render: {status} {text[:200]}")

    background_id, background_key = chat_frontend.load_sibling("api_keys").create(harness.tmp / "api_keys.json", 0, "background", priority="background")
    engine.reply(ANSWER)
    status, _, _ = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}]), Authorization="Bearer " + background_key)
    check(status == 200 and engine.requests[-1]["priority"] == chat_frontend.BULK_PRIORITY, "a key made with --priority background runs its requests below interactive ones")
    engine.reply(ANSWER)
    status, _, _ = await harness.post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}]), Authorization="Bearer " + background_key,
                                      **{"X-SparkPipe-Priority": "interactive"})
    check(engine.requests[-1]["priority"] == chat_frontend.INTERACTIVE_PRIORITY, "a request's own X-SparkPipe-Priority overrides its key's default")


def catalogue_problems(document):
    problems = []
    required = {"slug": str, "display_name": str, "supported_reasoning_levels": list, "shell_type": str, "visibility": str, "supported_in_api": bool,
                "priority": int, "support_verbosity": bool, "truncation_policy": dict, "experimental_supported_tools": list}
    enums = {"shell_type": {"unified_exec", "disabled", "default", "local", "shell_command"}, "visibility": {"list", "hide", "none"},
             "apply_patch_tool_type": {"freeform", None}}
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        return ["the catalogue is an object with a models array"]
    slugs = [entry.get("slug") for entry in document["models"] if isinstance(entry, dict)]
    if len(set(slugs)) != len(slugs) or not all(isinstance(slug, str) and slug.strip() for slug in slugs):
        problems.append("slugs are unique and not blank")
    for entry in document["models"]:
        for name, kind in required.items():
            if not isinstance(entry.get(name), kind) or (kind is int and isinstance(entry.get(name), bool)):
                problems.append(f"{entry.get('slug')}: {name} must be {kind.__name__}")
        for name, allowed in enums.items():
            if name in entry and entry[name] not in allowed:
                problems.append(f"{entry.get('slug')}: {name} {entry[name]!r} is not one of {sorted(map(str, allowed))}")
        if not all(isinstance(level, dict) and isinstance(level.get("effort"), str) and level["effort"] and isinstance(level.get("description"), str)
                   for level in entry.get("supported_reasoning_levels") or []):
            problems.append(f"{entry.get('slug')}: every reasoning level has an effort and a description")
        policy = entry.get("truncation_policy") or {}
        if policy.get("mode") not in ("bytes", "tokens") or not isinstance(policy.get("limit"), int):
            problems.append(f"{entry.get('slug')}: truncation_policy needs mode bytes or tokens and an integer limit")
        for name in ("context_window", "max_context_window", "auto_compact_token_limit"):
            if entry.get(name) is not None and (not isinstance(entry[name], int) or isinstance(entry[name], bool)):
                problems.append(f"{entry.get('slug')}: {name} must be an integer")
        if not isinstance(entry.get("base_instructions"), str) and not isinstance((entry.get("model_messages") or {}).get("instructions_template"), str):
            problems.append(f"{entry.get('slug')}: Codex refuses a model without base_instructions or model_messages.instructions_template")
        if entry.get("input_modalities") is not None and not set(entry["input_modalities"]) <= {"text", "image", "audio"}:
            problems.append(f"{entry.get('slug')}: input_modalities are text, image or audio")
    return problems


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
              listing["data"][0]["max_output_tokens"] == 4096 and listing["data"][0]["sequences"] == 2 and
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

        harness.frontend.replay.prefill.execute("DELETE FROM prefill")
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
        await gateway_rules(harness, check)
        await isolation_rules(harness, check)


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
                 "uses the measured rate; keyed requests get 429 after the busy wait while internal ones queue; accounts and members share sequences fairly; a "
                 "gateway key must name a valid account, each account has its own replay store and alias table, X-SparkPipe-Allow-Swap 1 allows a swap and "
                 "background keys run below interactive ones; bad header values are not echoed; a failed controller falls back to the last residency and "
                 "leaks nothing; overflow uses each shape's native signal; alias matching is linear and capped; the Codex catalogue parses as Codex's; a "
                 "store failure still delivers the reply; DELETE /v1/cache and a new user's wipe delete the replay store; a reply without a handle replays only "
                 "its own conversation's turn and only for its own member (the gateway's X-SparkPipe-Member, across that member's keys), also through tokenize; an unreadable store falls back to a normal render; a key "
                 "made with --priority background runs below interactive requests")


if __name__ == "__main__":
    main()
