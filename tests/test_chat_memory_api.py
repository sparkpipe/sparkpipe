#!/usr/bin/env python3
"""API clients get chat memory only when they ask for it. A Messages or Responses request with X-SparkPipe-Memory: on and a session id (the Claude Code session header, X-Session-Id, the Responses prompt_cache_key) is laid out by the memory layer under the key's scope and member: the system note, the expand and search_history tools and the exchange ids. The turn ids come from the conversation itself, so the next turn binds the curator's line for the last answer through its parent and the ledger then holds it, also when the request names Open WebUI message ids: a session given its identity never takes message ids from headers. Without the header, or for the harness's own compaction requests, the request is served raw, and opting in without a session id is refused. A tool round inside a memory turn replays the model's exact tokens, so its prompt extends the previous round's prompt and reply in the engine's cache. Every foreground request, from the API or from chat, counts as foreground for the idle-only background work."""
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
MEMORY = importlib.util.spec_from_file_location("chat_memory", ROOT / "tools" / "serving" / "chat_memory.py")
cm = importlib.util.module_from_spec(MEMORY)
MEMORY.loader.exec_module(cm)
BUDGETS = {"open_budget": 1500, "line_tokens": 48, "group_size": 16, "working_tokens": 24576, "working_context_share": 0.1875, "ledger_tokens": 12288,
           "ledger_context_share": 0.1, "part_small_tokens": 1024, "part_cap_tokens": 12288, "keep_tokens": 4096, "target_tolerance": 0.25,
           "target_minimum_tokens": 16384}
NOTE = "The first user message starts with <ledger>"
TOOLS = [{"name": "read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
LINE = "The user asked to read a.txt; it says hello world."
SESSION = "5f0c2d1e-aaaa-4bbb-8ccc-0123456789ab"


def body(messages, **extra):
    return dict({"model": "alpha", "max_tokens": 256, "system": "You are a coding agent.", "tools": TOOLS, "messages": messages}, **extra)


def assistant(text):
    return {"role": "assistant", "content": json.loads(text)["content"]}


async def run(tmp, check):
    secret = tmp / "memory.secret"
    secret.write_text("s" * 48)
    memory = {"store": str(tmp / "memory-store"), "jwt_secret_file": str(secret), "idle_seconds": 0.01, "budgets": BUDGETS}
    async with fixture.Harness(tmp, extra={"chat_memory": memory}) as harness:
        engine, frontend = harness.engine, harness.frontend
        model = frontend.models["alpha"]
        first = [{"role": "user", "content": "Read a.txt"}]

        engine.reply("Done.</think>It says hello world.")
        status, _, text = await harness.post("/v1/messages", body(first))
        check(status == 200 and NOTE not in harness.prompt_text(), f"without the header an API request is served raw: {status} {text[:200]}")

        status, _, text = await harness.post("/v1/messages", body(first), **{"X-SparkPipe-Memory": "on"})
        check(status == 400 and "needs a session id" in json.loads(text)["error"]["message"], f"opting in without a session id is refused: {status} {text[:200]}")

        engine.reply("Done.</think>It says hello world.")
        status, _, text = await harness.post("/v1/messages", body(first), **{"X-SparkPipe-Memory": "on", "X-Claude-Code-Request-Class": "compaction",
                                                                            "X-Claude-Code-Session-Id": SESSION})
        check(status == 200 and NOTE not in harness.prompt_text(), "the harness's own compaction requests stay raw")

        engine.reply("Done.</think>It says hello world.")
        engine.reply(json.dumps({"lines": {"e1": LINE}, "open": []}))
        opted = {"X-SparkPipe-Memory": "on", "X-Claude-Code-Session-Id": SESSION}
        status, _, text = await harness.post("/v1/messages", body(first), **opted)
        prompt = harness.prompt_text()
        check(status == 200 and NOTE in prompt and '"expand"' in prompt and '"search_history"' in prompt and "[e1] Read a.txt" in prompt,
              f"an opted-in request is laid out by the memory layer: {status} {prompt[-400:]}")
        user = f"key:{harness.key_id}/{harness.key_id}"
        check(await frontend.memory.settled(model, user, SESSION, 10.0), "the curator for the finished answer settles")
        state = frontend.memory.store.load(user, SESSION)
        pending = [entry["line"] for entry in state["pending"].values()]
        check(pending == [LINE], f"the curator's line for the answer waits under the reply's turn id: {state['pending']}")

        second = first + [assistant(text), {"role": "user", "content": "Read b.txt"}]
        engine.reply("Reading b.</think>Checking.<tool_call>read<arg_key>path</arg_key><arg_value>b.txt</arg_value></tool_call>")
        named = {"X-OpenWebUI-User-Message-Id": "owui-user-2", "X-OpenWebUI-User-Message-Parent-Id": "owui-reply-1", "X-OpenWebUI-Message-Id": "owui-reply-2"}
        status, _, text = await harness.post("/v1/messages", body(second), **opted, **named)
        state = frontend.memory.store.load(user, SESSION)
        key = cm.exchange_key([{"role": "user", "content": "Read a.txt"}, {"role": "assistant", "content": "It says hello world."}])
        check(status == 200 and state["lines"].get(key) == LINE, f"the next turn binds the line through the parent derived from the conversation: {state['lines']}")
        latest, ids = max(state["turns"].items(), key=lambda item: item[1]["index"])
        check(latest.startswith(cm.API_TURN) and (ids["parent"] or "").startswith(cm.API_REPLY) and not set(named.values()) & {latest, ids["parent"], ids["message"]},
              f"and keeps the conversation's turn ids when the request also names Open WebUI message ids: {latest} {ids}")
        direct = frontend.memory.session(model, named, [{"role": "user", "content": "Read c.txt"}], [], {}, identity=(user, "direct"))
        turns = frontend.memory.store.load(user, "direct")["turns"]
        check(direct is not None and list(turns) == [cm.API_TURN + cm.turn_id([], {"role": "user", "content": "Read c.txt"})] and not set(named.values()) & set(turns),
              f"a session given its identity derives its turn ids from the conversation, whatever message ids the headers name: {turns}")
        call = json.loads(text)
        check(call["stop_reason"] == "tool_use", f"the model asked for a tool: {call}")
        round_prompt = list(engine.requests[-1]["prompt_token_ids"])
        tool_use = next(block for block in call["content"] if block["type"] == "tool_use")
        third = second + [{"role": "assistant", "content": call["content"]},
                          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "bye"}]}]
        engine.reply("Done.</think>It says bye.")
        status, _, said_bye = await harness.post("/v1/messages", body(third), **opted)
        cached = engine.usage[-1]["cached_tokens"]
        check(status == 200 and cached > len(round_prompt), f"the tool round's prompt extends the previous round's prompt and reply in the cache: {cached} of {len(round_prompt)}")

        engine.reply("Done.</think>It says hello world.")
        status, _, text = await harness.post("/v1/responses", {"model": "alpha", "input": "Read a.txt", "prompt_cache_key": "codex-thread-1", "max_output_tokens": 64},
                                             **{"X-SparkPipe-Memory": "on"})
        check(status == 200 and NOTE in harness.prompt_text(), f"a Responses request is identified by its prompt_cache_key: {status} {text[:200]}")

        engine.hold = asyncio.Event()
        engine.reply("Done.</think>It says hello world.")
        waiting = asyncio.ensure_future(harness.post("/v1/messages", body(first)))
        for _ in range(100):
            if frontend.memory.active.get("alpha"):
                break
            await asyncio.sleep(0.02)
        check(frontend.memory.active.get("alpha") == 1, f"an API request in flight counts as foreground: {frontend.memory.active}")
        engine.hold.set()
        await waiting
        engine.hold = None
        check(frontend.memory.active.get("alpha") == 0, f"and stops counting when it ends: {frontend.memory.active}")

        check(await frontend.memory.settled(model, user, SESSION, 10.0), "the session's background work settles before the next turn")
        fourth = third + [assistant(said_bye), {"role": "user", "content": "What did a.txt say?"}]
        engine.reply("Checking my notes.</think><tool_call>expand<arg_key>id</arg_key><arg_value>e1</arg_value></tool_call>")
        engine.reply("Found it.</think>It said hello world.")
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            status, _, text = await harness.post("/v1/messages", body(fourth), **opted)
        rounds = engine.usage[-2:]
        logged = [json.loads(line) for line in captured.getvalue().splitlines() if line.startswith('{"event": "chat"')]
        check(status == 200 and json.loads(text)["sparkpipe"]["engine_rounds"] == 2,
              f"an answer after the model's own expand call reports two engine rounds: {status} {text[-300:]}")
        check(len(logged) == 1 and logged[0]["engine_rounds"] == 2 and
              logged[0]["engine_prefill_tokens"] == sum(entry["prompt_tokens"] - entry["cached_tokens"] for entry in rounds) and
              logged[0]["prompt_tokens"] == rounds[-1]["prompt_tokens"] and logged[0]["cached_tokens"] <= logged[0]["prompt_tokens"],
              f"its log line sums the new prefill of both rounds and counts the last round's prompt: {logged} {rounds}")


def main():
    checks = fixture.Checks("test_chat_memory_api")
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp), checks))
    checks.finish("API requests opt in with X-SparkPipe-Memory and a session id, get the memory layout under their key's scope, bind lines through turn ids "
                  "derived from the conversation even beside Open WebUI message ids, keep exact-token replay for tool rounds, stay raw without the header or for compaction, and count as "
                  "foreground for background work")


if __name__ == "__main__":
    main()
