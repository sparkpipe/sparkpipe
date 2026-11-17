#!/usr/bin/env python3
import asyncio
import contextlib
import importlib.util
import io
import json
import re
import sqlite3
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
chat_frontend = fixture.chat_frontend
CODEX_RETRY = re.compile(r"(?i)try again in\s*(\d+(?:\.\d+)?)\s*(s|ms|seconds?)")
SESSION = "6f1c2a9e-0d5b-4c1e-9a43-2b7f8e1d0c55"
PARENT = "0b9d7e21-5f4a-4e8c-8d1b-77aa3c9e4f10"
CLAUDE_TOOLS = [
    {"name": "Read", "description": "Reads a file from the local filesystem.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "offset": {"type": "number"}, "limit": {"type": "number"}},
                      "required": ["file_path"], "additionalProperties": False, "$schema": "http://json-schema.org/draft-07/schema#"}},
    {"name": "Bash", "description": "Executes a bash command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}, "description": {"type": "string"}},
                      "required": ["command"], "additionalProperties": False, "$schema": "http://json-schema.org/draft-07/schema#"}},
    {"name": "Task", "description": "Launch a new agent.",
     "input_schema": {"type": "object", "properties": {"description": {"type": "string"}, "prompt": {"type": "string"}, "subagent_type": {"type": "string"}},
                      "required": ["description", "prompt", "subagent_type"], "additionalProperties": False}, "cache_control": {"type": "ephemeral"}}]
CLAUDE_SYSTEM = [{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
                 {"type": "text", "text": "You are a subagent. Search the repository and report.", "cache_control": {"type": "ephemeral"}}]
CLAUDE_HEADERS = {"anthropic-version": "2023-06-01", "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14", "x-app": "cli",
                  "User-Agent": "claude-cli/2.1.308 (external, cli)", "X-Claude-Code-Session-Id": SESSION, "x-claude-code-request-class": "subagent",
                  "x-claude-code-agent-id": "agent-a7", "x-claude-code-parent-agent-id": "main"}
CALL_REPLY = "I should read the file first.</think>Reading it.<tool_call>Read<arg_key>file_path</arg_key><arg_value>/repo/a.txt</arg_value></tool_call>"
ANSWER = "The file is short.</think>It says hello world."
CODEX_TOOLS = [{"type": "function", "name": "shell", "description": "Runs a shell command", "strict": False,
                "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}}, "required": ["command"]}},
               {"type": "custom", "name": "apply_patch", "description": "Apply a patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}}]
CODEX_CALL = "List the files.</think><tool_call>shell<arg_key>command</arg_key><arg_value>[\"ls\"]</arg_value></tool_call>"
CODEX_ANSWER = "Two files.</think>There are two files."
ZERO_USAGE = {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}


def claude_body(messages, **extra):
    body = {"model": "claude-sonnet-4-5-20250929", "max_tokens": 32000, "stream": True, "system": CLAUDE_SYSTEM, "tools": CLAUDE_TOOLS,
            "messages": messages, "thinking": {"type": "enabled", "budget_tokens": 31999}, "temperature": 1,
            "metadata": {"user_id": json.dumps({"device_id": "d41d8cd98f00b204e9800998ecf8427e", "account_uuid": "", "session_id": SESSION,
                                                "parent_session_id": PARENT})}}
    body.update(extra)
    return body


def claude_user(text):
    return {"role": "user", "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]}


def codex_body(items, **extra):
    body = {"model": "gpt-5-codex", "instructions": "You are Codex, a coding agent.", "input": items, "tools": CODEX_TOOLS, "tool_choice": "auto",
            "parallel_tool_calls": False, "reasoning": {"effort": "medium", "summary": "auto"}, "store": False, "stream": True,
            "include": ["reasoning.encrypted_content"], "prompt_cache_key": SESSION, "text": {"verbosity": "low"}}
    body.update(extra)
    return body


def codex_user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def claude_code_usage(events):
    usage = dict(ZERO_USAGE)

    def merge(part):
        for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            if part.get(name) is not None and part[name] > 0:
                usage[name] = part[name]
        if part.get("output_tokens") is not None:
            usage["output_tokens"] = part["output_tokens"]
    for name, data in events:
        if name == "message_start":
            merge(data["message"]["usage"])
        elif name == "message_delta":
            merge(data["usage"])
    return usage


def input_sum(usage):
    return usage["input_tokens"] + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"]


def blocks_of(events):
    blocks, open_blocks = [], {}
    for name, data in events:
        if name == "content_block_start":
            open_blocks[data["index"]] = dict(data["content_block"])
        elif name == "content_block_delta":
            block, change = open_blocks[data["index"]], data["delta"]
            if change["type"] == "thinking_delta":
                block["thinking"] += change["thinking"]
            elif change["type"] == "signature_delta":
                block["signature"] = change["signature"]
            elif change["type"] == "text_delta":
                block["text"] += change["text"]
            elif change["type"] == "input_json_delta":
                block["input"] = json.loads(change["partial_json"])
        elif name == "content_block_stop":
            blocks.append(open_blocks.pop(data["index"]))
    return blocks


async def timed_stream(harness, path, body, **headers):
    frames, started = [], time.monotonic()
    async with harness.session.post(harness.base + path, json=body, headers=harness.headers(**headers)) as response:
        opened = time.monotonic() - started
        buffer = ""
        async for chunk in response.content.iter_any():
            buffer += chunk.decode()
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                frames.append((time.monotonic() - started, frame))
        return response.status, dict(response.headers), opened, frames


async def timed_body(harness, path, body, **headers):
    started = time.monotonic()
    async with harness.session.post(harness.base + path, json=body, headers=harness.headers(**headers)) as response:
        opened = time.monotonic() - started
        text = await response.text()
        return response.status, dict(response.headers), opened, time.monotonic() - started, text


def data_gaps(frames):
    moments = [moment for moment, frame in frames if any(line.startswith("data:") for line in frame.split("\n"))]
    return [after - before for before, after in zip(moments, moments[1:])]


async def socket_events(socket, body):
    await socket.send_json(dict(body, type="response.create"))
    events = []
    while True:
        message = await socket.receive(timeout=20)
        event = json.loads(message.data)
        events.append(event)
        if event["type"] in ("response.completed", "response.failed") or (event["type"] == "error" and "status" in event):
            return events


async def busy(harness, check):
    engine = harness.engine
    waited, interval = chat_frontend.BUSY_WAIT_SECONDS, chat_frontend.KEEPALIVE_SECONDS
    chat_frontend.BUSY_WAIT_SECONDS, chat_frontend.KEEPALIVE_SECONDS = 0.3, 5.0
    engine.hold = asyncio.Event()
    held = [asyncio.ensure_future(harness.post("/v1/messages", claude_body([claude_user("hold")], stream=False), **CLAUDE_HEADERS)) for _ in range(2)]
    await asyncio.sleep(0.15)
    started = time.monotonic()
    status, headers, text = await harness.post("/v1/messages", claude_body([claude_user("full")]), **CLAUDE_HEADERS)
    took = time.monotonic() - started
    body = json.loads(text)
    check(status == 429 and took >= 0.3 and headers.get("Retry-After", "").isdigit() and headers.get("x-should-retry") == "true" and
          body == {"type": "error", "error": {"type": "rate_limit_error", "message": body["error"]["message"]}},
          f"a Claude Code request that finds every sequence busy waits the full wait, then gets 429 rate_limit_error with retry-after and "
          f"x-should-retry, which the Anthropic SDK retries: {status} {took:.2f} {headers} {text}")
    status, headers, text = await harness.post("/v1/responses", codex_body([codex_user("full")]))
    error = json.loads(text)["error"]
    found = CODEX_RETRY.search(error["message"])
    check(status == 429 and headers.get("Retry-After") == str(chat_frontend.BUSY_RETRY_AFTER_SECONDS) and error["code"] == "rate_limit_exceeded" and
          error["type"] == "requests" and found is not None and found.group(1) == headers.get("Retry-After"),
          f"a Codex request gets 429 rate_limit_exceeded with retry-after, and a message whose delay Codex parses: {status} {headers} {text}")
    async with harness.socket() as socket:
        events = await socket_events(socket, codex_body([codex_user("full")]))
    check(len(events) == 1 and events[0]["status"] == 429 and events[0]["error"]["code"] == "rate_limit_exceeded" and
          events[0]["headers"].get("retry-after") == str(chat_frontend.BUSY_RETRY_AFTER_SECONDS),
          f"over the WebSocket the same refusal is an error event with status 429, the code and retry-after: {events}")
    engine.hold.set()
    results = await asyncio.gather(*held)
    engine.hold = None
    chat_frontend.BUSY_WAIT_SECONDS, chat_frontend.KEEPALIVE_SECONDS = waited, interval
    check(all(result[0] == 200 for result in results), "the requests that held the sequences finish")


async def long_prefill(harness, check):
    engine = harness.engine
    interval = chat_frontend.KEEPALIVE_SECONDS
    chat_frontend.KEEPALIVE_SECONDS = 0.1
    engine.delay = 0.6
    engine.reply(ANSWER)
    status, _, opened, frames = await timed_stream(harness, "/v1/messages", claude_body([claude_user("slow")]), **CLAUDE_HEADERS)
    events = fixture.sse_events("\n\n".join(frame for _, frame in frames) + "\n\n")
    names = [name for name, _ in events]
    start = names.index("message_start") if "message_start" in names else -1
    check(status == 200 and opened < 0.4 and start >= 3 and set(names[:start]) == {"ping"} and names[-1] == "message_stop",
          f"a Claude Code stream opens at the first keep-alive and sends ping events until the first token: {opened:.2f} {names[:start + 2]}")
    check(max(data_gaps(frames)) < 0.35, f"no gap between SSE events is longer than a few keep-alive intervals: {data_gaps(frames)}")

    engine.reply(CODEX_ANSWER)
    status, _, opened, frames = await timed_stream(harness, "/v1/responses", codex_body([codex_user("slow")]))
    events = fixture.sse_events("\n\n".join(frame for _, frame in frames) + "\n\n")
    names = [name for name, _ in events]
    check(status == 200 and names[:2] == ["response.created", "response.in_progress"] and names[2:5] == ["response.in_progress"] * 3 and
          all(name != "comment" for name in names) and [data["sequence_number"] for _, data in events] == list(range(len(events))),
          f"a Codex stream repeats response.in_progress, a data event Codex's SSE reader counts, never a comment it would skip: {names[:6]}")
    check(max(data_gaps(frames)) < 0.35, f"Codex's idle timer, which only data events reset, never sees a gap longer than a few intervals: {data_gaps(frames)}")

    engine.reply(ANSWER)
    status, headers, opened, done, text = await timed_body(harness, "/v1/messages", claude_body([claude_user("slow")], stream=False), **CLAUDE_HEADERS)
    reply = json.loads(text)
    check(status == 200 and opened < 0.4 and done >= 0.6 and text.startswith("\n") and text.lstrip() == text.lstrip("\n") and
          reply["type"] == "message" and reply["content"][-1]["text"] == "It says hello world." and headers.get("Content-Type", "").startswith("application/json"),
          f"a non-streamed Messages request gets its headers at the first keep-alive and newlines before the JSON body, which JSON parsers skip: "
          f"{opened:.2f} {done:.2f} {text[:40]!r}")
    engine.reply(CODEX_ANSWER)
    status, headers, opened, done, text = await timed_body(harness, "/v1/responses", codex_body([codex_user("slow")], stream=False))
    reply = json.loads(text)
    check(status == 200 and opened < 0.4 and done >= 0.6 and text.startswith("\n") and reply["object"] == "response" and reply["status"] == "completed",
          f"a non-streamed Responses request is kept open the same way: {opened:.2f} {done:.2f} {text[:40]!r}")
    engine.delay = 0.0
    engine.reply(ANSWER)
    status, _, _, _, text = await timed_body(harness, "/v1/messages", claude_body([claude_user("quick")], stream=False), **CLAUDE_HEADERS)
    check(status == 200 and not text.startswith("\n"), "a reply ready before the first keep-alive has no padding")

    engine.delay = 0.6
    engine.reply(CODEX_ANSWER)
    async with harness.socket() as socket:
        events = await socket_events(socket, codex_body([codex_user("slow")]))
    kinds = [event["type"] for event in events]
    check(kinds[:5] == ["response.created", "response.in_progress", "response.in_progress", "response.in_progress", "response.in_progress"] and
          kinds[-1] == "response.completed", f"over the WebSocket the same response.in_progress events keep the turn alive: {kinds[:6]}")
    engine.delay = 0.0
    chat_frontend.KEEPALIVE_SECONDS = interval


async def exact_usage(harness, check, public):
    engine = harness.engine
    where = "through the public door" if public else "inside the hub"

    async def send(path, body):
        if public:
            return await harness.post(path, body, **CLAUDE_HEADERS)
        async with harness.session.post(harness.base + path, json=body, headers=CLAUDE_HEADERS) as response:
            return response.status, dict(response.headers), await response.text()

    def read_ok(read, cached):
        return read == cached if not public else 0 <= cached - read < chat_frontend.PrefixLedger().block_tokens

    history = [claude_user(f"Read /repo/a.txt and summarise it {where}.")]
    status, _, text = await send("/v1/messages/count_tokens", claude_body(history, stream=None, max_tokens=None))
    counted = json.loads(text)["input_tokens"]
    engine.reply(CALL_REPLY)
    status, _, text = await send("/v1/messages", claude_body(history))
    events = fixture.sse_events(text)
    engine_usage = engine.usage[-1]
    seen = claude_code_usage(events)
    start = next(data for name, data in events if name == "message_start")["message"]["usage"]
    check(status == 200 and counted == engine_usage["prompt_tokens"] == len(engine.requests[-1]["prompt_token_ids"]),
          f"{where}, count_tokens on a Claude Code request with tools, thinking and subagent metadata equals the engine's prompt count: {counted} {engine_usage}")
    check(input_sum(seen) == engine_usage["prompt_tokens"] and read_ok(seen["cache_read_input_tokens"], engine_usage["cached_tokens"]) and
          seen["cache_creation_input_tokens"] == 0, f"{where}, Claude Code's merged usage sums to the engine's prompt length on the first turn: {seen} {engine_usage}")
    check(start == dict(seen, output_tokens=start["output_tokens"]),
          f"{where}, message_start already carries the engine's counts, for clients that read usage only there: {start} {seen}")

    blocks = blocks_of(events)
    history = history + [{"role": "assistant", "content": blocks},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": blocks[-1]["id"], "content": "hello world"}]}]
    status, _, text = await send("/v1/messages/count_tokens", claude_body(history, stream=None, max_tokens=None))
    counted = json.loads(text)["input_tokens"]
    engine.reply(ANSWER)
    status, _, text = await send("/v1/messages", claude_body(history))
    events = fixture.sse_events(text)
    engine_usage = engine.usage[-1]
    seen = claude_code_usage(events)
    check(counted == engine_usage["prompt_tokens"] and engine_usage["cached_tokens"] >= len(engine.committed[-2]) and
          input_sum(seen) == engine_usage["prompt_tokens"] and read_ok(seen["cache_read_input_tokens"], engine_usage["cached_tokens"]),
          f"{where}, on the replayed second turn count_tokens and the merged usage match the engine's prompt and cache counts: {counted} {seen} {engine_usage}")

    engine.cache_all = True
    engine.reply(ANSWER)
    status, _, text = await send("/v1/messages", claude_body(history))
    seen = claude_code_usage(fixture.sse_events(text))
    check(input_sum(seen) == engine.usage[-1]["prompt_tokens"] and (public or seen["input_tokens"] == 0),
          f"{where}, an engine that reports the whole prompt cached still sums to the prompt length, not twice it: {seen} {engine.usage[-1]}")
    engine.cache_all = False
    engine.early_usage = False
    engine.reply(ANSWER)
    status, _, text = await send("/v1/messages", claude_body(history))
    events = fixture.sse_events(text)
    seen = claude_code_usage(events)
    start = next(data for name, data in events if name == "message_start")["message"]["usage"]
    check(start == ZERO_USAGE and input_sum(seen) == engine.usage[-1]["prompt_tokens"],
          f"{where}, an engine that reports usage only at the end gets zeros at message_start, never a count the chat layer made up: {start} {seen}")
    engine.early_usage = True

    engine.reply(ANSWER)
    status, _, text = await send("/v1/messages", claude_body(history, stream=False))
    usage = json.loads(text)["usage"]
    check(input_sum(usage) == engine.usage[-1]["prompt_tokens"] and read_ok(usage["cache_read_input_tokens"], engine.usage[-1]["cached_tokens"]),
          f"{where}, a non-streamed reply's usage sums the same way: {usage} {engine.usage[-1]}")

    engine.reply(CODEX_ANSWER)
    status, _, text = await send("/v1/responses", codex_body([codex_user(f"Count {where}")], stream=False))
    usage = json.loads(text)["usage"]
    check(usage["input_tokens"] == engine.usage[-1]["prompt_tokens"] and read_ok(usage["input_tokens_details"]["cached_tokens"], engine.usage[-1]["cached_tokens"]),
          f"{where}, Responses usage reports the engine's prompt and cached counts: {usage} {engine.usage[-1]}")
    engine.reply(CODEX_ANSWER)
    status, _, text = await send("/v1/chat/completions", {"model": "alpha", "messages": [{"role": "user", "content": f"Count {where}"}]})
    usage = json.loads(text)["usage"]
    check(usage["prompt_tokens"] == engine.usage[-1]["prompt_tokens"] and read_ok(usage["prompt_tokens_details"]["cached_tokens"], engine.usage[-1]["cached_tokens"]),
          f"{where}, chat completions map prompt_tokens_details.cached_tokens from the engine the same way: {usage} {engine.usage[-1]}")


async def codex_catalogue(harness, check):
    async with harness.session.get(harness.base + "/v1/codex/models", headers=harness.headers(account="acme")) as response:
        catalogue = await response.json()
    entry = catalogue["models"][0]
    model = harness.frontend.models[entry["slug"]]
    check(response.status == 200 and entry["context_window"] == model.context_tokens == 65536 and
          entry["auto_compact_token_limit"] == model.context_tokens - model.max_output_tokens and entry["effective_context_window_percent"] == 100,
          f"the Codex catalogue gives the served window, compaction before the output limit and the whole window as usable: {entry}")
    check(chat_frontend.compact_limit(32768, 32768) == int(32768 * chat_frontend.CODEX_COMPACT_SHARE),
          "a model whose output limit is its whole window still gets a positive compaction limit")


async def stored(harness, check):
    engine = harness.engine
    engine.reply(CODEX_ANSWER)
    status, _, text = await harness.post("/v1/responses", codex_body([codex_user("Keep this")], store=True, stream=False), account="acme")
    kept = json.loads(text)
    async with harness.session.get(harness.base + "/v1/responses/" + kept["id"], headers=harness.headers(account="acme")) as response:
        got = await response.json()
        check(response.status == 200 and got == kept, f"GET returns a stored response to its own account, as it was answered: {response.status} {json.dumps(got)[:300]}")
    async with harness.session.get(harness.base + "/v1/responses/" + kept["id"], headers=harness.headers(account="bob")) as response:
        got = await response.json()
        check(response.status == 404 and got["error"]["code"] == "not_found", f"another account's id is a 404: {response.status} {got}")
    engine.reply(CODEX_ANSWER)
    status, _, text = await harness.post("/v1/responses", codex_body([codex_user("Keep this too")], store=True), account="acme")
    completed = fixture.sse_events(text)[-1][1]["response"]
    async with harness.session.get(harness.base + "/v1/responses/" + completed["id"], headers=harness.headers(account="acme")) as response:
        got = await response.json()
        check(response.status == 200 and got["id"] == completed["id"] and got["output"] == completed["output"] and got["usage"] == completed["usage"],
              f"a streamed stored response is retrieved as response.completed carried it: {json.dumps(got)[:300]}")
    engine.reply(CODEX_ANSWER)
    status, _, text = await harness.post("/v1/responses", codex_body([codex_user("Forget this")], store=False, stream=False), account="acme")
    async with harness.session.get(harness.base + "/v1/responses/" + json.loads(text)["id"], headers=harness.headers(account="acme")) as response:
        check(response.status == 404, "store false keeps nothing to retrieve")
    async with harness.session.delete(harness.base + "/v1/responses/" + kept["id"], headers=harness.headers(account="bob")) as response:
        check(response.status == 404, "another account cannot delete it")
    teammate = {"account": "acme", "X-SparkPipe-Member": "MBOTHER"}
    async with harness.session.get(harness.base + "/v1/responses/" + kept["id"], headers=harness.headers(**teammate)) as response:
        check(response.status == 404, f"another member of the account cannot read it: {response.status}")
    async with harness.session.delete(harness.base + "/v1/responses/" + kept["id"], headers=harness.headers(**teammate)) as response:
        check(response.status == 404, f"another member of the account cannot delete it: {response.status}")
    async with harness.session.delete(harness.base + "/v1/responses/" + kept["id"], headers=harness.headers(account="acme")) as response:
        got = await response.json()
        check(response.status == 200 and got == {"id": kept["id"], "object": "response", "deleted": True}, f"DELETE removes it: {got}")
    async with harness.session.get(harness.base + "/v1/responses/" + kept["id"], headers=harness.headers(account="acme")) as response:
        check(response.status == 404, "a deleted response is a 404")
    async with harness.session.delete(harness.base + "/v1/cache", headers=harness.headers(account="acme")) as response:
        check(response.status == 200, "DELETE /v1/cache answers")
    async with harness.session.get(harness.base + "/v1/responses/" + completed["id"], headers=harness.headers(account="acme")) as response:
        check(response.status == 404, "deleting the account's working state deletes its stored responses")
    engine.reply(CODEX_ANSWER)
    status, _, text = await harness.post("/v1/responses", codex_body([codex_user("Until the wipe")], store=True, stream=False), account="acme")
    wiped = json.loads(text)["id"]
    harness.directive(time.time() + 5, 0)
    harness.frontend.expire_working_state(time.time(), True)
    async with harness.session.get(harness.base + "/v1/responses/" + wiped, headers=harness.headers(account="acme")) as response:
        check(response.status == 404, "a wipe at a change of user removes stored responses by the working-state rule")
    harness.directive(0, 0)
    harness.frontend.expire_working_state(time.time(), True)
    engine.reply(CODEX_ANSWER)
    status, _, text = await harness.post("/v1/responses", codex_body([codex_user("Across a restart")], store=True, stream=False), account="acme")
    return json.loads(text)


async def same_turns(harness, transport):
    engine = harness.engine
    engine.reply(CODEX_CALL)
    first_body = codex_body([codex_user("What files are here?")], store=transport == "http")
    socket = None
    if transport == "websocket":
        socket = await harness.socket().__aenter__()
        events = await socket_events(socket, first_body)
    else:
        _, _, text = await harness.post("/v1/responses", first_body)
        events = [data for _, data in fixture.sse_events(text)]
    first = events[-1]["response"]
    call = first["output"][-1]
    engine.reply(CODEX_ANSWER)
    follow = codex_body([{"type": "function_call_output", "call_id": call["call_id"], "output": "a.txt\nb.txt"}], previous_response_id=first["id"])
    if socket is not None:
        second = await socket_events(socket, follow)
        await socket.close()
    else:
        _, _, text = await harness.post("/v1/responses", follow)
        second = [data for _, data in fixture.sse_events(text)]
    return events, second, harness.prompt_text()


async def websocket(harness, check):
    engine = harness.engine
    http_first, http_second, http_prompt = await same_turns(harness, "http")
    ws_first, ws_second, ws_prompt = await same_turns(harness, "websocket")
    check([event["type"] for event in ws_first] == [event["type"] for event in http_first] and
          [event["type"] for event in ws_second] == [event["type"] for event in http_second] and
          [event["sequence_number"] for event in ws_first] == list(range(len(ws_first))),
          f"the WebSocket sends the same events as SSE for the same two turns: {[event['type'] for event in ws_first]} {[event['type'] for event in http_first]}")
    check(ws_second[-1]["type"] == "response.completed" and ws_prompt == http_prompt and
          engine.requests[-1]["prompt_token_ids"][:len(engine.committed[-2])] == engine.committed[-2],
          f"on one connection previous_response_id with store false follows the connection's last response, renders the prompt the stored HTTP chain "
          f"renders, and replays the reply to its end: {[event['type'] for event in ws_second][-3:]}")
    record = ws_second[-1]["response"]["sparkpipe"]
    check(record["transport"] == "websocket" and isinstance(record["turn_ms"], int) and record["turn_ms"] >= 0 and
          http_second[-1]["response"]["sparkpipe"]["transport"] == "http" and isinstance(http_second[-1]["response"]["sparkpipe"]["turn_ms"], int),
          f"the telemetry record gives each turn's transport and time, so WebSocket and HTTP turns can be compared: {record}")

    engine.reply(CODEX_ANSWER)
    async with harness.socket() as socket:
        events = await socket_events(socket, codex_body([codex_user("hi")], previous_response_id="resp_unknown"))
    check(len(events) == 1 and events[0]["type"] == "error" and events[0]["status"] == 400 and events[0]["error"]["code"] == "previous_response_not_found",
          f"a previous_response_id the connection and the store do not hold is an error event with status 400: {events}")
    engine.replies.clear()

    async with harness.socket() as socket:
        warm = await socket_events(socket, codex_body([codex_user("Warm the session")], generate=False))
        before = len(engine.requests)
        engine.reply(CODEX_ANSWER)
        turn = await socket_events(socket, codex_body([codex_user("Now answer")], previous_response_id=warm[-1]["response"]["id"]))
    check([event["type"] for event in warm] == ["response.created", "response.in_progress", "response.completed"] and warm[-1]["response"]["output"] == [] and
          len(engine.requests) == before + 1 and turn[-1]["type"] == "response.completed" and "Warm the session" in harness.prompt_text(),
          f"generate false keeps the input on the connection without generating, and the next turn builds on it: {[event['type'] for event in warm]}")

    async with harness.socket() as socket:
        await socket.send_str("not json")
        refused = json.loads((await socket.receive(timeout=5)).data)
        await socket.send_json({"type": "response.cancel"})
        other = json.loads((await socket.receive(timeout=5)).data)
        engine.reply(CODEX_ANSWER)
        alive = await socket_events(socket, codex_body([codex_user("still here")]))
    check(refused["status"] == 400 and other["status"] == 400 and alive[-1]["type"] == "response.completed",
          f"a malformed message is refused with an error event and the connection keeps serving: {refused} {other}")
    async with harness.session.get(harness.base + "/v1/responses", headers=harness.headers()) as response:
        check(response.status == 405, f"a plain GET on /v1/responses names the two ways in: {response.status}")


def old_store(tmp, check):
    directory = tmp / "old-store"
    directory.mkdir()
    path = directory / "account-acme.sqlite"
    db = sqlite3.connect(str(path))
    db.execute("CREATE TABLE responses (id TEXT PRIMARY KEY, parent TEXT, model TEXT NOT NULL, member TEXT NOT NULL, messages TEXT NOT NULL, "
               "bytes INTEGER NOT NULL, used REAL NOT NULL)")
    db.execute("INSERT INTO responses VALUES ('resp_old', NULL, 'alpha', '', '[]', 10, ?)", (time.time(),))
    db.commit()
    db.close()
    path.chmod(0o600)
    store = chat_frontend.chat_replay.ReplayStore(directory)
    store.put_response("account:acme", None, "resp_new", None, "alpha", [{"role": "user", "content": "x"}], {"id": "resp_new", "object": "response"})
    check(store.response_body("account:acme", None, "resp_new") == {"id": "resp_new", "object": "response"} and store.response_body("account:acme", None, "resp_old") is None and
          store.response_messages("account:acme", None, "resp_old") == [], "a replay store written before stored bodies gains the column and keeps its responses")
    store.close()


async def run(tmp, check):
    async with fixture.Harness(tmp) as harness:
        await busy(harness, check)
        await long_prefill(harness, check)
        await exact_usage(harness, check, True)
        await exact_usage(harness, check, False)
        await codex_catalogue(harness, check)
        survivor = await stored(harness, check)
        await websocket(harness, check)
    async with fixture.Harness(tmp) as harness:
        async with harness.session.get(harness.base + "/v1/responses/" + survivor["id"], headers=harness.headers(account="acme")) as response:
            check(response.status == 200 and (await response.json())["id"] == survivor["id"], "a stored response survives a chat layer restart")
    old_store(tmp, check)


def main():
    check = fixture.Checks("test_chat_frontend_client_fidelity")
    captured = io.StringIO()
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(captured):
        asyncio.run(run(Path(directory), check))
    if check.failures:
        print(captured.getvalue()[-4000:])
    check.finish("replayed Claude Code and Codex requests: a full fleet waits, then answers 429 rate_limit_error or rate_limit_exceeded with retry-after on "
                 "HTTP and WebSocket; long prefills send ping events, repeated response.in_progress events and leading newlines on non-streamed bodies; "
                 "count_tokens and Claude Code's merged usage equal the engine's prompt and cache counts on every turn; the Codex catalogue gives the "
                 "served window and a compaction limit; stored responses are retrievable by their own account only, until deleted or wiped, and across a "
                 "restart; the WebSocket transport sends the same events as SSE, follows the connection's last response, warms with generate false and "
                 "reports each turn's time")


if __name__ == "__main__":
    main()
