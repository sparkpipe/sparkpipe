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
TOOLS = [{"name": "read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
          "cache_control": {"type": "ephemeral"}}]
SYSTEM = [{"type": "text", "text": "You are a coding agent."}, {"type": "text", "text": "Work in /repo.", "cache_control": {"type": "ephemeral"}}]
TOOL_REPLY = "The user wants a.txt read.</think>Checking.<tool_call>read<arg_key>path</arg_key><arg_value>a.txt</arg_value></tool_call>"
ANSWER = "The file says hello.</think>It says hello world."


def stream_body(**extra):
    body = {"model": "alpha", "max_tokens": 1024, "stream": True, "system": SYSTEM, "tools": TOOLS,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Read a.txt", "cache_control": {"type": "ephemeral"}}]}]}
    body.update(extra)
    return body


def kinds(events):
    return [(name, (data or {}).get("content_block", {}).get("type") or (data or {}).get("delta", {}).get("type")) for name, data in events]


async def run(tmp, check):
    async with fixture.Harness(tmp) as harness:
        engine = harness.engine
        engine.reply(TOOL_REPLY)
        status, headers, text = await harness.post("/v1/messages", stream_body(metadata={"user_id": "u1"}, service_tier="auto", unknown_field={"x": 1}),
                                                   **{"anthropic-beta": "interleaved-thinking-2025-05-14", "anthropic-version": "2023-06-01"})
        events = fixture.sse_events(text)
        names = [name for name, _ in events]
        check(status == 200 and headers.get("Content-Type", "").startswith("text/event-stream"), f"a streamed Messages request answers an SSE stream: {status} {text[:200]}")
        check(names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"], f"the stream opens with message_start and ends with message_delta, message_stop: {names}")
        check([kind for kind in kinds(events) if kind[0] != "content_block_delta" or True][1:] and
              [k for k in kinds(events)][1] == ("content_block_start", "thinking") and ("content_block_delta", "signature_delta") in kinds(events) and
              ("content_block_start", "text") in kinds(events) and ("content_block_start", "tool_use") in kinds(events) and
              ("content_block_delta", "input_json_delta") in kinds(events), f"thinking, text and tool_use blocks stream with their deltas: {kinds(events)}")
        order = [kind for kind in kinds(events) if kind[0] in ("content_block_start", "content_block_stop")]
        check(order == [("content_block_start", "thinking"), ("content_block_stop", None), ("content_block_start", "text"), ("content_block_stop", None),
                        ("content_block_start", "tool_use"), ("content_block_stop", None)], f"every block is opened and closed in order: {order}")
        signature_at = kinds(events).index(("content_block_delta", "signature_delta"))
        check(names[signature_at + 1] == "content_block_stop", "the signature is the last delta of its thinking block")
        delta = next(data for name, data in events if name == "message_delta")
        usage = delta["usage"]
        prompt = len(engine.requests[0]["prompt_token_ids"])
        check(delta["delta"]["stop_reason"] == "tool_use", f"a tool call stops with tool_use: {delta}")
        check(usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"] == prompt and usage["output_tokens"] > 0,
              f"the three input fields sum to the prompt length {prompt}: {usage}")
        record = delta.get("sparkpipe") or {}
        check(set(record) >= {"request", "fleet", "model", "pack", "prompt_tokens", "cached_tokens", "prefill_tokens", "queue_ms", "prefill_ms", "ttft_ms",
                              "output_tokens", "decode_tps", "draft_accept", "cold_prefill_ms_estimate"} and record["model"] == "alpha" and record["prompt_tokens"] == prompt,
              f"message_delta carries the sparkpipe telemetry object: {record}")
        check(headers.get("X-SparkPipe-Request") == record.get("request") and headers.get("X-SparkPipe-Prompt-Tokens") == str(prompt),
              f"headers carry the request id and prompt length before the first byte: {headers}")
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
        thinking, said, call = blocks
        check(thinking["thinking"] == "The user wants a.txt read." and thinking["signature"].startswith("sp1."), f"the thinking block carries the reasoning and an opaque signature: {thinking}")
        check(said["text"] == "Checking." and call["name"] == "read" and call["input"] == {"path": "a.txt"} and call["id"].startswith("toolu_"),
              f"the text and the parsed tool call: {said} {call}")
        first_prompt = engine.requests[0]["prompt_token_ids"]
        first_text = harness.prompt_text(0)
        check(first_text.startswith("<|im_start|>system\n# Tools\n") and "<|im_start|>system\nYou are a coding agent.\n\nWork in /repo.<|im_end|>" in first_text,
              f"system blocks render as the system turn, after the template's tool block: {first_text[:300]}")
        check(engine.requests[0]["priority"] == chat_frontend.INTERACTIVE_PRIORITY, "a main request runs at interactive priority")

        engine.reply(ANSWER)
        history = [{"role": "user", "content": [{"type": "text", "text": "Read a.txt", "cache_control": {"type": "ephemeral"}}]},
                   {"role": "assistant", "content": blocks},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": [{"type": "text", "text": "hello world"}]}]}]
        status, headers, text = await harness.post("/v1/messages", stream_body(messages=history))
        events = fixture.sse_events(text)
        second = engine.requests[1]["prompt_token_ids"]
        generated = engine.committed[0][len(first_prompt):]
        check(status == 200 and second[:len(engine.committed[0])] == engine.committed[0],
              "the next turn's prompt is the previous prompt plus the exact generated tokens, up to the end of the previous reply")
        delta = next(data for name, data in events if name == "message_delta")
        check(delta["usage"]["cache_read_input_tokens"] == len(engine.committed[0]),
              f"the cache matches the whole previous turn, {len(engine.committed[0])} tokens: {delta['usage']}")
        check(delta["sparkpipe"]["replayed_tokens"] == len(first_prompt) + len(generated) and headers.get("X-SparkPipe-Replayed-Tokens") == str(len(first_prompt) + len(generated)),
              f"telemetry says how many prompt tokens came from the replay store: {delta['sparkpipe']}")
        check(harness.prompt_text(1).endswith("<|im_start|>tool\nhello world<|im_end|>\n<|im_start|>assistant\n<think>"),
              f"what follows the replayed turn is rendered by the template: {harness.prompt_text(1)[-120:]}")
        model = harness.frontend.models["alpha"]
        internal = chat_frontend.chat_shapes.messages_request(stream_body(messages=history))
        _, naive = model.render(chat_frontend.normalize_messages(internal["messages"]), internal["tools"], {})
        check(fixture.common_prefix(naive, engine.committed[0]) < len(first_prompt) + 2,
              "a plain re-render of the same history would miss the cache from the start of the previous reply, which the replay avoids")

        edited = json.loads(json.dumps(history))
        edited[1]["content"][1]["text"] = "Checking carefully."
        engine.reply(ANSWER)
        await harness.post("/v1/messages", stream_body(messages=edited))
        check("Checking carefully." in harness.prompt_text(2) and "The user wants a.txt read." not in harness.prompt_text(2),
              "a turn the client changed is rendered from the client's copy, never replaced by stored tokens")

        dropped = json.loads(json.dumps(history))
        dropped[1]["content"] = dropped[1]["content"][1:]
        engine.reply(ANSWER)
        await harness.post("/v1/messages", stream_body(messages=dropped))
        check(engine.requests[3]["prompt_token_ids"][:len(engine.committed[0])] == engine.committed[0],
              "a client that drops the thinking block still gets the turn replayed from its tool_use ids and text")

        engine.reply(ANSWER)
        mid = history + [{"role": "system", "content": "The user is now on the release branch."}, {"role": "user", "content": "Continue."}]
        status, _, text = await harness.post("/v1/messages", stream_body(messages=mid, stream=False))
        reply = json.loads(text)
        check(status == 200 and "<|im_end|>\n<|im_start|>system\nThe user is now on the release branch.<|im_end|>\n<|im_start|>user\nContinue.<|im_end|>" in harness.prompt_text(4),
              f"a mid-conversation system message renders in place as a system turn: {harness.prompt_text(4)[-260:]}")
        check(reply["type"] == "message" and reply["stop_reason"] == "end_turn" and [block["type"] for block in reply["content"]] == ["thinking", "text"] and
              reply["content"][0]["signature"].startswith("sp1.") and reply["content"][1]["text"] == "It says hello world." and "sparkpipe" in reply,
              f"a non-streamed reply carries the thinking block with its signature, the text, end_turn and telemetry: {reply}")
        check(engine.requests[4]["prompt_token_ids"][:len(engine.committed[0])] == engine.committed[0],
              "the mid-conversation system turn follows the replayed reply without breaking the cache")

        status, _, text = await harness.post("/v1/messages/count_tokens", {"model": "alpha", "system": SYSTEM, "tools": TOOLS, "messages": mid})
        check(status == 200 and json.loads(text) == {"input_tokens": len(engine.requests[4]["prompt_token_ids"])},
              f"count_tokens returns the engine's exact prompt length, replay included: {text}")

        status, _, text = await harness.post("/v1/messages", dict(stream_body(), model="beta", stream=False))
        error = json.loads(text)
        check(status == 409 and error["type"] == "error" and error["error"]["type"] == "model_not_loaded" and error["error"]["resident"] == ["alpha"],
              f"a model that would need the resident one unloaded is refused with 409 model_not_loaded naming the resident models: {status} {text}")
        before = len(engine.requests)
        status, _, text = await harness.post("/v1/messages", dict(stream_body(), model="beta", stream=False), swap=True)
        check(status == 200 and len(engine.requests) == before + 1, f"a key with allow_swap may load it: {status} {text[:200]}")

        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/messages", dict(stream_body(), model="claude-sonnet-4-5-20250929", stream=False))
        reply = json.loads(text)
        check(status == 200 and reply["model"] == "claude-sonnet-4-5-20250929" and reply["sparkpipe"]["model"] == "alpha",
              f"a claude-* name goes to the first resident model and the reply echoes the requested name: {text[:300]}")
        harness.controller.status["models"] = [{"id": "alpha", "active": True}, {"id": "beta", "active": True}]
        harness.frontend.residency_cache = None
        async with harness.session.put(harness.base + "/v1/models/aliases", json={"claude-haiku-*": "beta", "claude-*": None}, headers=harness.headers()) as response:
            check(response.status == 200, f"the alias table takes a haiku pattern: {response.status} {await response.text()}")
        status, _, text = await harness.post("/v1/messages", dict(stream_body(), model="claude-haiku-4-5-20251001", stream=False))
        check(status == 200 and json.loads(text)["sparkpipe"]["model"] == "beta", f"a claude-haiku-* name goes to the small resident model: {text[:300]}")
        status, _, text = await harness.post("/v1/messages", dict(stream_body(), model="claude-opus-4-1", stream=False))
        check(status == 200 and json.loads(text)["sparkpipe"]["model"] == "alpha", "other claude-* names still go to the first resident model")

        status, _, text = await harness.post("/v1/messages", dict(stream_body(), stream=False, messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]))
        check(status == 400 and json.loads(text)["error"]["type"] == "image_not_supported", f"an image to a model without images is refused: {status} {text}")
        status, _, text = await harness.post("/v1/messages", dict(stream_body(), stream=False, tools=[dict(TOOLS[0], defer_loading=True)]))
        check(status == 400, "defer_loading (MCP tool search) is refused, since the setup turns that beta off")

        async with harness.session.post(harness.base + "/v1/messages", json=dict(stream_body(), stream=False),
                                        headers={fixture.HEADER: "1", "X-Api-Key": harness.key}) as response:
            check(response.status == 200, f"X-Api-Key authenticates like a bearer key: {response.status}")
        async with harness.session.post(harness.base + "/v1/messages", json=dict(stream_body(), stream=False), headers={fixture.HEADER: "1"}) as response:
            body = await response.json()
            check(response.status == 401 and body["type"] == "error" and body["error"]["type"] == "invalid_key", f"a missing key gets the Anthropic error body: {body}")
        async with harness.session.head(harness.base + "/api/hello", headers={fixture.HEADER: "1"}) as response:
            check(response.status == 200, "HEAD /api/hello answers 200 without a key")

        engine.reply(ANSWER)
        await harness.post("/v1/messages", dict(stream_body(), stream=False), **{"x-claude-code-request-class": "compaction"})
        check(engine.requests[-1]["priority"] == chat_frontend.BULK_PRIORITY, "a compaction request runs at background priority")
        engine.reply(ANSWER)
        await harness.post("/v1/messages", dict(stream_body(), stream=False), **{"x-claude-code-request-class": "auxiliary", "X-Claude-Code-Session-Id": "s-1",
                                                                                   "x-claude-code-agent-id": "a-2"})
        check(engine.requests[-1]["priority"] == chat_frontend.BULK_PRIORITY, "an auxiliary request runs at background priority")
        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/messages", dict(stream_body(), stream=False), **{"x-claude-code-request-class": "subagent", "X-Claude-Code-Session-Id": "s-1",
                                                                                                     "x-claude-code-agent-id": "a-2"})
        record = json.loads(text)["sparkpipe"]
        check(engine.requests[-1]["priority"] == chat_frontend.INTERACTIVE_PRIORITY and record["session"] == "s-1" and record["agent"] == "a-2" and record["priority"] == "interactive",
              f"a subagent request stays interactive and its session and agent ids reach the record: {record}")

        chat_frontend.KEEPALIVE_SECONDS = 0.1
        engine.delay = 0.45
        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/messages", stream_body())
        engine.delay = 0.0
        names = [name for name, _ in fixture.sse_events(text)]
        check(status == 200 and names[0] == "message_start" and names[1:3] == ["ping", "ping"] and names[-1] == "message_stop",
              f"a long prefill sends ping events after message_start until the first token: {names}")

        chat_frontend.BUSY_WAIT_SECONDS = 0.3
        engine.hold = asyncio.Event()
        held = [asyncio.ensure_future(harness.post("/v1/messages", dict(stream_body(), stream=False))) for _ in range(2)]
        await asyncio.sleep(0.1)
        status, headers, text = await harness.post("/v1/messages", dict(stream_body(), stream=False))
        check(status == 429 and headers.get("Retry-After", "").isdigit() and headers.get("x-should-retry") == "true" and json.loads(text)["error"]["type"] == "busy",
              f"when every sequence is in use the request waits, then gets 429 with an integer retry-after and x-should-retry: {status} {headers} {text}")
        engine.hold.set()
        results = await asyncio.gather(*held)
        engine.hold = None
        check(all(result[0] == 200 for result in results), "the requests that held the sequences finish")
    async with fixture.Harness(tmp) as harness:
        harness.engine.committed.append(list(engine.committed[0]))
        harness.engine.reply(ANSWER)
        await harness.post("/v1/messages", stream_body(messages=history))
        check(harness.engine.requests[0]["prompt_token_ids"][:len(engine.committed[0])] == engine.committed[0],
              "the replay store survives a chat layer restart, so the first turn after a release is still exact")


def main():
    check = fixture.Checks("test_chat_frontend_messages")
    captured = io.StringIO()
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(captured):
        asyncio.run(run(Path(directory), check))
    if check.failures:
        print(captured.getvalue()[-4000:])
    check.finish("POST /v1/messages streams message_start, thinking/text/tool_use blocks with their deltas and the signature, message_delta with usage "
                 "summing to the prompt and the sparkpipe object, and message_stop; a tool-use round trip replays the generated tokens so the next prompt "
                 "matches the cache to the end of the previous reply, survives a dropped thinking block and a restart, and never overrides an edited turn; "
                 "mid-conversation system messages render in place; unknown fields, metadata, cache_control and anthropic-beta are ignored; count_tokens "
                 "is exact; 409 model_not_loaded unless allow_swap; claude-* and claude-haiku-* aliases; hint headers set priority and affinity; pings "
                 "keep a long prefill open; a full fleet answers 429 with retry-after")


if __name__ == "__main__":
    main()
