#!/usr/bin/env python3
"""The chat layer's miss ledger records, for every request on the Messages, Responses and chat completions shapes, how much of the prompt the fleet had carried before, where the engine resumed, and a cause for everything prefilled beyond the request's own new tail: a first or changed opening, a lost or restored session, a compaction, a history the client changed, a new pack profile, a model that was not resident or the engine's nearest resume point. It records counts, times, ids and digests only, never content, and it changes nothing the engine is asked to do."""
import asyncio
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
chat_frontend = fixture.chat_frontend
miss_ledger = chat_frontend.miss_ledger
chat_shapes = chat_frontend.chat_shapes

NOW = 1_800_000_000.0
BLOCK = 64
SECRET = "zebra-quartz-umbrella"
TOOLS = [{"name": "read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
SYSTEM = [{"type": "text", "text": "You are a coding agent. " * 40}, {"type": "text", "text": "Work in /repo. " * 40}]
CONTEXT = "Project instructions: keep the tests passing. " * 30
REPLY = "Thinking about it.</think>Done reading."


def facts(**changes):
    base = {"model": "m", "block_tokens": BLOCK, "resume_tokens": BLOCK, "scope": "own", "session": ("own", "s1", "a1", "main"), "prompt_tokens": 46400,
            "cached_tokens": 0, "engine": None, "seen_tokens": 0, "seen_of": 0, "seen_at": None, "opening": opening(), "first_prompt_tokens": 80,
            "model_wait_ms": None, "pack_profile_new": False, "request_class": "main", "agent_type": None, "compaction": None, "compacted": False, "now": NOW}
    base.update(changes)
    return base


def opening(first_request=True, **changes):
    parts = {"tools": "t1", "attribution": None, "system": "s1", "project_context": "c1", "first_request": first_request}
    parts.update(changes)
    return parts


def engine(matched, resumed, boot="b1", artifact="a1", **extra):
    return dict({"matched_tokens": matched, "resumed_tokens": resumed, "boot": boot, "artifact": artifact}, **extra)


def unit(check):
    ledger = miss_ledger.Ledger()
    first = facts(engine=engine(0, 0))
    entry = ledger.entry(first)
    check(entry["cause"] == "first_opening" and entry["tail_tokens"] == 80 and entry["opening_tokens"] == 46320 and entry["miss_tokens"] == 46320,
          f"a session's first request on an opening nobody carried is a first sight, and its own first prompt is the tail: {entry}")
    ledger.settle(first, 46500)
    turn = facts(prompt_tokens=46900, seen_tokens=46464, seen_of=46500, seen_at=NOW, engine=engine(46499, 46499), now=NOW + 5)
    entry = ledger.entry(turn)
    check(entry["cause"] is None and entry["tail_tokens"] == 400 and entry["miss_tokens"] == 1 and entry["session_turn"] == 2 and entry["session_tokens"] == 46500,
          f"a turn that extends its session and resumes at the reply's end has no cause: {entry}")
    ledger.settle(turn, 47000)
    anchored = ledger.entry(facts(prompt_tokens=47300, seen_tokens=46976, seen_of=47000, seen_at=NOW, engine=engine(46999, 32768), now=NOW + 9))
    check(anchored["cause"] == "resume_point" and anchored["recomputed_tokens"] == 14231 and anchored["lost_tokens"] == 0 and anchored["miss_tokens"] == 14232,
          f"tokens between the engine's resume point and its match are recomputed, not lost: {anchored}")
    for changes, reason in ((dict(engine=engine(0, 0)), "evicted"), (dict(engine=engine(0, 0, boot="b2")), "restart"),
                            (dict(engine=engine(0, 0, boot="b2", artifact="a2")), "release"), (dict(engine=engine(0, 0), wiped=NOW + 30), "wiped"),
                            (dict(engine=engine(0, 0), wiped=NOW - 30, now=NOW + 900000), "evicted")):
        lost = miss_ledger.Ledger()
        lost.entry(first)
        lost.settle(first, 46500)
        entry = lost.entry(facts(**dict(dict(prompt_tokens=46900, seen_tokens=46464, seen_of=46500, seen_at=NOW, now=NOW + 60), **changes)))
        check(entry["cause"] == "lost" and entry["lost_reason"] == reason and entry["lost_tokens"] == 46500 and entry["miss_tokens"] == 46500,
              f"a session the engine no longer matches is lost, by {reason}: {entry}")
    swapped = miss_ledger.Ledger()
    swapped.entry(first)
    swapped.settle(first, 46500)
    swapped.model_absent("m", NOW + 30)
    entry = swapped.entry(facts(prompt_tokens=46900, seen_tokens=46464, seen_of=46500, seen_at=NOW, engine=engine(0, 0, boot="b2"), now=NOW + 60))
    check(entry["lost_reason"] == "swap", f"a session lost while its model was away is lost to a swap: {entry}")
    waited = swapped.entry(facts(prompt_tokens=46900, seen_tokens=46464, seen_of=46500, seen_at=NOW, engine=engine(46499, 46499, boot="b2", restore_ms=2400.0),
                                 model_wait_ms=41000.0, now=NOW + 60))
    check(waited["cause"] == "model_not_resident" and waited["model_wait_ms"] == 41000.0 and waited["restore_ms"] == 2400.0,
          f"a request that waited for its model to load says so, with its restore time: {waited}")
    restored = ledger.entry(facts(prompt_tokens=47300, seen_tokens=46976, seen_of=47000, seen_at=NOW, engine=engine(46999, 46999, restore_ms=1800.0), now=NOW + 9))
    check(restored["cause"] == "restored" and restored["restore_ms"] == 1800.0 and restored["miss_tokens"] == 1, f"a session the engine restored from NVMe is named: {restored}")
    changed = ledger.entry(facts(prompt_tokens=47300, seen_tokens=20032, seen_of=47000, seen_at=NOW, engine=engine(20032, 16384), now=NOW + 9))
    check(changed["cause"] == "history_changed" and changed["cause_position"] == 20032 and changed["cause_block"] == "history" and changed["miss_tokens"] == 30916,
          f"a session whose prompt leaves its last prompt was changed by the client, at the first block nobody carried: {changed}")
    reply = ledger.entry(facts(prompt_tokens=47300, seen_tokens=46912, seen_of=47000, seen_at=NOW, engine=engine(46912, 46912), now=NOW + 9))
    check(reply["cause"] == "history_changed" and reply["cause_block"] == "reply", f"a change inside the last reply is told from a change to the history: {reply}")
    compaction = facts(prompt_tokens=47200, seen_tokens=46976, seen_of=47000, seen_at=NOW, engine=engine(46999, 46999), request_class="compaction",
                       session=("own", "s1", "a1", "compaction"), session_fallbacks=[("own", "s1", "a1", "main")], opening=opening(first_request=False), now=NOW + 10)
    entry = ledger.entry(compaction)
    check(entry["cause"] is None and entry["request_class"] == "compaction" and entry["tail_tokens"] == 200 and entry["session_tokens"] == 47000,
          f"the compaction request itself extends the main conversation: {entry}")
    ledger.settle(compaction, 47800)
    ledger.compacting([("own", "s1", "a1", "main"), ("own", "s1", "a1", None)])
    after = ledger.entry(facts(prompt_tokens=52000, seen_tokens=46400, seen_of=47800, seen_at=NOW, engine=engine(46400, 32768), now=NOW + 40))
    check(after["cause"] == "compaction" and after["cause_position"] == 46400 and after["miss_tokens"] == 19232, f"the first request after a compaction is named: {after}")
    headed = miss_ledger.Ledger().entry(facts(compacted=True, compaction="auto", prompt_tokens=52000, seen_tokens=46400, seen_of=47800, seen_at=NOW, engine=engine(46400, 32768)))
    check(headed["cause"] == "compaction" and headed["compacted"] is True and headed["compaction"] == "auto", f"the context-compacted header names a compaction by itself: {headed}")
    family = miss_ledger.Ledger()
    family.entry(first)
    family.settle(first, 46500)
    for changes, block in ((dict(tools="t2"), "tools"), (dict(attribution="x2"), "attribution"), (dict(system="s2"), "system"), (dict(project_context="c2"), "project_context"),
                           (dict(), "template")):
        entry = family.entry(facts(session=("own", "s2", "a1", "main"), opening=opening(**changes), seen_tokens=1024, seen_of=46500, seen_at=NOW, engine=engine(1024, 1024)))
        check(entry["cause"] == "opening_changed" and entry["cause_block"] == block and entry["opening_tokens"] == 45296, f"an opening that changed names its first changed block, {block}: {entry}")
    shared = family.entry(facts(session=("own", "s3", "a1", "main"), seen_tokens=46272, seen_of=46500, seen_at=NOW, engine=engine(46300, 32768)))
    check(shared["cause"] == "resume_point" and shared["opening_tokens"] == 48 and shared["recomputed_tokens"] == 13532,
          f"a second session on a known opening pays only from the engine's resume point: {shared}")
    other = family.entry(facts(session=("own", "s4", "a9", "subagent"), request_class="subagent", agent_type="Explore", engine=engine(0, 0)))
    check(other["cause"] == "first_opening" and other["agent_type"] == "Explore", f"each agent type has its own opening, so its first is a first sight: {other}")
    pack = miss_ledger.Ledger().entry(facts(pack_profile_new=True, seen_tokens=40000, seen_of=41000, seen_at=NOW, engine=engine(0, 0)))
    check(pack["cause"] == "lost", f"a carried prefix the engine lost is lost before it is a pack profile: {pack['cause']}")
    pack = miss_ledger.Ledger().entry(facts(pack_profile_new=True, engine=engine(0, 0)))
    check(pack["cause"] == "pack_profile", f"a pack prefilled for a new tool set or template setting is named: {pack}")
    older = miss_ledger.Ledger()
    older.entry(facts(cached_tokens=0))
    older.settle(facts(), 46500)
    entry = older.entry(facts(prompt_tokens=46900, seen_tokens=46464, seen_of=46500, seen_at=NOW, cached_tokens=32768, resume_tokens=16384, now=NOW + 5))
    check(entry["cause"] is None and entry["engine_reported"] is False and entry["matched_tokens"] is None and entry["resumed_tokens"] == 32768 and entry["recomputed_tokens"] is None,
          f"with an engine that reports only cached tokens, a resume within the model's checkpoint distance is not called lost: {entry}")
    entry = older.entry(facts(prompt_tokens=46900, seen_tokens=46464, seen_of=46500, seen_at=NOW, cached_tokens=0, resume_tokens=16384, now=NOW + 5))
    check(entry["cause"] == "lost" and entry["lost_reason"] is None, f"and a larger gap is lost, with no reason it cannot know: {entry}")
    loose = miss_ledger.Ledger().entry(facts(session=None, opening=opening(first_request=False), prompt_tokens=9000, seen_tokens=6400, seen_of=6500, seen_at=NOW, engine=engine(6400, 6400)))
    check(loose["cause"] is None and loose["tail_tokens"] is None and loose["miss_tokens"] is None, f"without a session id a follow-up's own tail is unknown, so nothing is blamed: {loose}")
    check(set(entry) == set(first_fields()) and all(not isinstance(value, (list, dict)) for value in entry.values()), "a ledger entry is a flat set of counts, times and names")


def first_fields():
    return ("cause", "cause_block", "cause_position", "lost_reason", "matched_tokens", "resumed_tokens", "recomputed_tokens", "tail_tokens", "miss_tokens",
            "opening_tokens", "lost_tokens", "seen_tokens", "seen_of", "seen_age_s", "session_tokens", "session_turn", "restore_ms", "background_wait_ms",
            "model_wait_ms", "request_class", "agent_type", "compaction", "compacted", "engine_reported")


def shapes(check):
    body = {"tools": TOOLS, "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=1;"}] + SYSTEM,
            "messages": [{"role": "user", "content": [{"type": "text", "text": CONTEXT}, {"type": "text", "text": "Fix the bug."}]}]}
    parts = chat_shapes.messages_opening(body)
    other = chat_shapes.messages_opening(dict(body, system=[{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=2;"}] + SYSTEM))
    check(parts["prompt"] == "Fix the bug." and parts["first_request"] is True and parts["attribution"] != other["attribution"] and
          all(parts[block] == other[block] for block in ("tools", "system", "project_context")),
          "a Messages opening keeps its attribution block apart from the system blocks, and the first message's last text block is the first prompt")
    later = chat_shapes.messages_opening(dict(body, messages=body["messages"] + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]))
    check(later["first_request"] is False and all(later[block] == parts[block] for block in miss_ledger.OPENING_BLOCKS), "later turns of a session have the same opening")
    codex = {"instructions": "Base instructions.", "tools": [{"type": "function", "name": "exec_command"}],
             "input": [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "permissions"}]},
                       {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "AGENTS.md and environment"}]},
                       {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix the bug."}]}]}
    parts = chat_shapes.responses_opening(codex)
    grown = chat_shapes.responses_opening(dict(codex, input=codex["input"] + [{"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": "{}"},
                                                                             {"type": "function_call_output", "call_id": "c1", "output": "done"}]))
    check(parts["prompt"] == "Fix the bug." and parts["first_request"] is True and grown["first_request"] is False and grown["project_context"] == parts["project_context"],
          "a Responses opening is the instructions, the tools and the items before the first prompt")
    check(chat_shapes.responses_opening(dict(codex, previous_response_id="resp_1")) is None, "a Responses request that continues a stored response carries no opening")
    chat = chat_shapes.chat_opening({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}]})
    check(chat["prompt"] == "Hello" and chat["first_request"] is True, "a chat completions opening is its leading system messages")
    check(all(SECRET not in str(value) for value in chat_shapes.messages_opening(dict(body, tools=[{"name": SECRET}])).values() if value != "Fix the bug."),
          "an opening is digests, not content")


def messages_body(text, history=(), **extra):
    body = {"model": "alpha", "max_tokens": 64, "stream": False, "system": SYSTEM, "tools": TOOLS,
            "messages": [{"role": "user", "content": [{"type": "text", "text": CONTEXT}, {"type": "text", "text": text}]}] + list(history)}
    body.update(extra)
    return body


def records(tmp, owner):
    folder = tmp / chat_frontend.RECORDS_DIRECTORY / owner
    return [json.loads(line) for path in sorted(folder.glob("*.jsonl")) for line in path.read_text().splitlines()]


async def served(tmp, check):
    async with fixture.Harness(tmp) as harness:
        engine = harness.engine
        own = {"X-Claude-Code-Session-Id": "session-one", "x-claude-code-agent-id": "agent-one", "x-claude-code-request-class": "main", "x-claude-code-agent-type": "general-purpose"}

        async def internal(path, body, headers):
            async with harness.session.post(harness.base + path, json=body, headers=headers) as response:
                return response.status, json.loads(await response.text())

        engine.reply(REPLY)
        status, reply = await internal("/v1/messages", messages_body(f"Find the {SECRET}."), own)
        first = reply.get("sparkpipe") or {}
        check(status == 200 and first.get("cause") == "first_opening" and first.get("seen_tokens") == 0 and first.get("request_class") == "main" and
              first.get("agent_type") == "general-purpose" and first.get("session_turn") == 1 and first.get("engine_reported") is False,
              f"a Messages request's telemetry carries the ledger with the client's class and agent type: {status} {first}")
        check("cache_points" not in engine.requests[-1], "the ledger declares no cache point on a Messages request")
        history = [{"role": "assistant", "content": reply["content"]}, {"role": "user", "content": [{"type": "text", "text": "And the next file."}]}]
        engine.cache = lambda prompt, cached: {"matched_tokens": cached, "resumed_tokens": cached - cached % 128, "restore_ms": 0, "background_wait_ms": 37.5, "boot": "b1", "artifact": "a1"}
        engine.reply(REPLY)
        status, reply = await internal("/v1/messages", messages_body(f"Find the {SECRET}.", history), own)
        second = reply.get("sparkpipe") or {}
        prompt = len(engine.requests[-1]["prompt_token_ids"])
        check(status == 200 and second.get("seen_tokens", 0) >= first["prompt_tokens"] // 64 * 64 and second.get("session_turn") == 2 and second.get("engine_reported") is True and
              second.get("matched_tokens") == engine.usage[-1]["cached_tokens"] and second.get("resumed_tokens") == second["matched_tokens"] - second["matched_tokens"] % 128 and
              second.get("recomputed_tokens") == second["matched_tokens"] % 128 and second.get("background_wait_ms") == 37.5 and second.get("tail_tokens") == prompt - second["session_tokens"],
              f"the next turn extends its session, and the engine's own match, resume point and waits are recorded: {second}")
        check(second.get("cause") in (None, "resume_point"), f"a turn that extends its session is blamed for nothing but the engine's resume point: {second.get('cause')}")
        engine.cache = None
        rewritten = [{"role": "assistant", "content": [{"type": "text", "text": "A different earlier reply. " * 30}]}, {"role": "user", "content": [{"type": "text", "text": "And the next file."}]}]
        status, reply = await internal("/v1/messages", messages_body(f"Find the {SECRET}.", rewritten), own)
        third = reply.get("sparkpipe") or {}
        check(third.get("cause") == "history_changed" and third.get("cause_block") == "reply" and third.get("cause_position") == third.get("seen_tokens"),
              f"a client that sends a different reply than the one it was given changed its history, at the reply: {third}")
        status, reply = await internal("/v1/messages", messages_body(f"Find the {SECRET}.", history), dict(own, **{"x-claude-code-request-class": "compaction", "x-claude-code-compaction": "auto"}))
        check((reply.get("sparkpipe") or {}).get("request_class") == "compaction" and reply["sparkpipe"].get("compaction") == "auto", f"the compaction headers are recorded: {reply.get('sparkpipe')}")
        status, reply = await internal("/v1/messages", messages_body("A summary of the work so far."), dict(own, **{"x-claude-code-context-compacted": "1"}))
        compacted = reply.get("sparkpipe") or {}
        check(compacted.get("compacted") is True and compacted.get("cause") in ("compaction", None), f"the context-compacted header is recorded: {compacted}")
        codex = {"model": "alpha", "instructions": "Base instructions. " * 60, "stream": False, "store": False, "prompt_cache_key": "codex-session",
                 "tools": [{"type": "function", "name": "exec_command", "description": "Run", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}],
                 "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": CONTEXT}]},
                           {"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"Find the {SECRET}."}]}]}
        engine.reply(REPLY)
        status, reply = await internal("/v1/responses", codex, {"session_id": "codex-session"})
        responses = reply.get("sparkpipe") or {}
        check(status == 200 and responses.get("cause") == "first_opening" and "seen_tokens" in responses and responses.get("session_turn") == 1,
              f"a Responses request's telemetry carries the ledger, keyed by its prompt cache key: {status} {responses}")
        check("cache_points" not in engine.requests[-1], "the ledger declares no cache point on a Responses request")
        status, reply = await internal("/v1/responses", dict(codex, input=codex["input"][:1] + [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Another task."}]}],
                                                             prompt_cache_key="codex-second"), {})
        shared = reply.get("sparkpipe") or {}
        check(shared.get("seen_tokens", 0) > 0 and shared.get("opening_tokens") is not None and shared.get("cause") != "first_opening",
              f"a second Codex session on the same opening is not a first sight: {shared}")
        chat = {"model": "alpha", "max_tokens": 32, "messages": [{"role": "system", "content": "Be brief. " * 80}, {"role": "user", "content": f"Where is the {SECRET}?"}]}
        status, reply = await internal("/v1/chat/completions", chat, {"X-Session-Id": "chat-session"})
        completions = reply.get("sparkpipe") or {}
        check(status == 200 and completions.get("cause") == "first_opening" and completions.get("session_turn") == 1, f"a chat completions request carries the ledger too: {status} {completions}")
        status, _, text = await harness.post("/v1/messages", messages_body("Fix the bug."), **{"X-Claude-Code-Session-Id": "keyed-session"})
        keyed = json.loads(text).get("sparkpipe") or {}
        check(status == 200 and "cause" not in keyed and "seen_tokens" not in keyed, f"a keyed request's own telemetry does not show what other accounts' requests carried: {sorted(keyed)}")
        stored = records(tmp, "own")
        key_stored = records(tmp, "key-" + harness.key_id)
        check(len(stored) == 8 and all(set(first_fields()) <= set(line) for line in stored) and {line["shape"] for line in stored} == {"messages", "responses", "chat"},
              f"every request's stored record holds the ledger, on all three shapes: {len(stored)} {sorted({line['shape'] for line in stored})}")
        check(len(key_stored) == 1 and set(first_fields()) <= set(key_stored[0]) and key_stored[0]["seen_tokens"] > 0 and key_stored[0]["session_turn"] == 1,
              f"a keyed request's ledger is kept in its fleet record: {key_stored}")
        check(all(not isinstance(value, (list, dict)) for line in stored + key_stored for value in line.values()), "a record stays flat: counts, times, ids and names")
        text = "".join(path.read_text() for path in (tmp / chat_frontend.RECORDS_DIRECTORY).rglob("*.jsonl"))
        check(SECRET not in text and "Project instructions" not in text and "exec_command" not in text, "no prompt, tool or instruction text reaches a record")


def main():
    check = fixture.Checks("test_chat_frontend_miss_ledger")
    unit(check)
    shapes(check)
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(served(Path(tmp), check))
    check.finish("every request on the Messages, Responses and chat completions shapes records what the fleet carried, where the engine resumed and a cause for what was "
                 "prefilled beyond its own tail, with the Claude Code class, agent type and compaction headers, as flat counts and names with no content")


if __name__ == "__main__":
    main()
