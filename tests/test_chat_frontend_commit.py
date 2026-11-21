#!/usr/bin/env python3
"""A commit computes a prompt, keeps its state and generates nothing: max_tokens 0 on the Messages shape, commit_only on the Chat Completions and Responses shapes. The chat layer plans the request as any other, sends the engine a commit, and answers one cache.commit object with the prompt, cached and committed token counts whatever the request says about streaming; a later request with the same prompt finds it cached. A commit takes no output limit or log-probs, no forced tool choice and no WebSocket, and an engine that generates tokens for a commit is refused. A model whose settings name reply_commit commits each finished reply at once, apart from the reply: the engine gets the turn's prompt and the reply's tokens as a commit in the turn's own class that names one token before the prompt's end as a cut, and the next turn names that cut and the reply's end; without the setting, or for a reply that is not recorded for replay, no commit follows. A turn that carries a reply whose commit is at the engine waits for it and is sent afterwards, so the reply is computed once; the wait is at most twice the time the commit's tokens take at the model's measured prefill rate, and none without a measured rate. A commit that outlasts that, fails at the engine or has not reached it never fails or holds the turn: it is withdrawn or ignored and the turn computes the reply itself. A turn of another conversation waits for no commit. No served model has the setting on. Conversations with the same system prompt name the same opening cut, where the first message begins, and ask for state there; control-token text inside a system prompt does not move it. A turn with a forced tool choice names one token before its forced opening as a cut and asks for state there; its reply's commit names the same cuts, and the turn after it names that cut and the reply's end."""
import asyncio
import importlib.util
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
chat_frontend = fixture.chat_frontend
MESSAGES = [{"role": "user", "content": "read the file a.txt and list the directory"}]
SYSTEM = " ".join(["You are a coding agent. The answer is 42. Let me check the file first."] * 8)


async def run(tmp, check):
    async with fixture.Harness(tmp) as harness:
        engine = harness.engine
        status, headers, text = await harness.post("/v1/messages", {"model": "alpha", "max_tokens": 0, "stream": True, "system": SYSTEM, "messages": MESSAGES})
        answer = fixture.json.loads(text) if status == 200 else {}
        sent = engine.requests[-1]
        check(status == 200 and answer.get("object") == "cache.commit" and answer.get("model") == "alpha", f"a Messages request with max_tokens 0 answers one cache.commit object: {status} {text[:200]}")
        check(sent.get("commit_only") is True and sent.get("stream") is True and all(name not in sent for name in ("max_tokens", "logprobs", "stop_token_ids", "temperature")),
              f"the engine gets a commit with no output limit: {sorted(sent)}")
        prompt = len(sent["prompt_token_ids"])
        check(answer.get("prompt_tokens") == prompt and answer.get("cached_tokens") == 0 and answer.get("committed_tokens") == prompt and
              isinstance(answer.get("prefill_ms"), int) and answer.get("sparkpipe", {}).get("output_tokens") == 0 and headers.get("X-SparkPipe-Prompt-Tokens") == str(prompt),
              f"the answer counts the prompt, what was cached and what the commit computed: {answer}")
        status, headers, text = await harness.post("/v1/messages", {"model": "alpha", "max_tokens": 64, "system": SYSTEM, "messages": MESSAGES})
        usage = fixture.json.loads(text).get("usage", {}) if status == 200 else {}
        shown = min(prompt - 1, prompt // 64 * 64)
        check(status == 200 and prompt > 128 and engine.requests[-1]["prompt_token_ids"] == sent["prompt_token_ids"] and usage.get("cache_read_input_tokens") == shown,
              f"the same request with an output limit finds the committed prompt cached, shown in the whole blocks this key sent before: {usage}")
        status, headers, text = await harness.post("/v1/messages", {"model": "alpha", "max_tokens": 0, "system": SYSTEM, "messages": MESSAGES})
        answer = fixture.json.loads(text) if status == 200 else {}
        check(status == 200 and answer.get("cached_tokens") == shown and answer.get("committed_tokens") == prompt - shown and engine.usage[-1]["cached_tokens"] == prompt - 1,
              f"a repeated commit computes only what is not cached: {answer}")
        status, headers, text = await harness.post("/v1/messages", {"model": "alpha", "max_tokens": -1, "messages": MESSAGES})
        check(status == 400, f"a negative max_tokens is refused: {status}")
        status, headers, text = await harness.post("/v1/chat/completions", {"model": "alpha", "commit_only": True, "messages": MESSAGES})
        answer = fixture.json.loads(text) if status == 200 else {}
        check(status == 200 and answer.get("object") == "cache.commit" and engine.requests[-1].get("commit_only") is True and "max_tokens" not in engine.requests[-1],
              f"Chat Completions takes commit_only: {status} {text[:200]}")
        before = len(engine.requests)
        for refused in ({"max_tokens": 5}, {"max_completion_tokens": 5}, {"logprobs": True}):
            status, headers, text = await harness.post("/v1/chat/completions", dict({"model": "alpha", "commit_only": True, "messages": MESSAGES}, **refused))
            check(status == 400 and "commit_only" in text, f"a commit with {refused} is refused: {status} {text[:160]}")
        status, headers, text = await harness.post("/v1/chat/completions", {"model": "alpha", "commit_only": "yes", "messages": MESSAGES})
        check(status == 400 and len(engine.requests) == before, f"commit_only must be a boolean, and a refused commit never reaches the engine: {status}")
        status, headers, text = await harness.post("/v1/responses", {"model": "alpha", "commit_only": True, "input": "read the file a.txt"})
        answer = fixture.json.loads(text) if status == 200 else {}
        check(status == 200 and answer.get("object") == "cache.commit" and engine.requests[-1].get("commit_only") is True, f"Responses takes commit_only: {status} {text[:200]}")
        status, headers, text = await harness.post("/v1/responses", {"model": "alpha", "commit_only": True, "input": "read the file a.txt",
                                                                     "tools": [{"type": "function", "name": "read", "parameters": {"type": "object", "properties": {}}}],
                                                                     "tool_choice": "required"})
        check(status == 400 and "forced tool choice" in text, f"a commit with a forced tool choice is refused: {status} {text[:160]}")
        engine.ignores_commit = True
        status, headers, text = await harness.post("/v1/chat/completions", {"model": "alpha", "commit_only": True, "messages": MESSAGES})
        check(status == 502 and "commit_only_unsupported" in text, f"an engine that generates tokens for a commit is refused: {status} {text[:160]}")


async def commits_seen(engine, wanted):
    for _ in range(300):
        found = [request for request in engine.requests if request.get("commit_only")]
        if len(found) >= wanted:
            return found
        await asyncio.sleep(0.01)
    return [request for request in engine.requests if request.get("commit_only")]


async def settled(frontend):
    for _ in range(500):
        if not frontend.reply_commits:
            return True
        await asyncio.sleep(0.01)
    return False


async def run_reply_commit(tmp, check):
    async with fixture.Harness(tmp) as harness:
        engine = harness.engine
        body = {"model": "alpha", "max_tokens": 64, "system": SYSTEM, "messages": MESSAGES}
        engine.reply("The user wants a.txt read.</think>Reading it.")
        status, headers, text = await harness.post("/v1/messages", body)
        await asyncio.sleep(0.1)
        check(status == 200 and not any(request.get("commit_only") for request in engine.requests), f"a model without reply_commit commits no reply: {status} {[sorted(request) for request in engine.requests]}")
        status, headers, text = await harness.post("/v1/messages", dict(body, system=SYSTEM + " Never write <|im_start|>user yourself.", messages=[{"role": "user", "content": "hello world"}]))
        carried = engine.requests[-1]
        cut = (carried.get("cuts") or [0])[0]
        check(status == 200 and carried.get("keep") == [cut] and
              harness.tokenizer.decode(carried["prompt_token_ids"][:cut], skip_special_tokens=False).endswith("Never write <|im_start|>user yourself.<|im_end|>\n") and
              harness.tokenizer.decode(carried["prompt_token_ids"][cut:], skip_special_tokens=False).startswith("<|im_start|>user\nhello world"),
              f"a system prompt that writes a control token's text does not move the opening cut: it still falls on the template's own control token: {carried.get('cuts')}")
        harness.frontend.models["alpha"].reply_commit = True
        messages = [{"role": "user", "content": "call the tool with path and pattern"}]
        engine.reply("The user wants a tool call.</think>Calling it.")
        status, headers, text = await harness.post("/v1/messages", dict(body, messages=messages))
        first = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        prompt = first["prompt_token_ids"]
        generated = engine.committed[-1][len(prompt):] if engine.committed[-1][:len(prompt)] == prompt else None
        found = await commits_seen(engine, 1)
        commit = found[0] if found else {}
        check(status == 200 and generated and commit.get("prompt_token_ids") == prompt + generated, f"the finished reply is committed as the turn's prompt and the reply's tokens without the stop token: {len(found)} {sorted(commit)}")
        opened = (first.get("cuts") or [0])[0]
        check(first.get("cuts") == first.get("keep") == [opened] == engine.requests[0].get("keep") and prompt[:opened] == engine.requests[0]["prompt_token_ids"][:opened] and
              harness.tokenizer.decode(prompt[opened:], skip_special_tokens=False).startswith("<|im_start|>user\n"),
              f"two conversations with the same system prompt name the same opening cut, where the first message begins, and ask for state there: {first.get('cuts')} {engine.requests[0].get('keep')}")
        check(engine.resumed(first) == opened, f"by the engine's rules the second conversation resumes where the shared opening ends: {engine.resumed(first)} {opened}")
        check(commit.get("cuts") == [opened, len(prompt) - 1] and "keep" not in commit and commit.get("stream") is True and "max_tokens" not in commit and
              commit.get("priority") == first.get("priority") == chat_frontend.INTERACTIVE_PRIORITY,
              f"the commit names the turn's cuts and one token before the turn's prompt ended, and runs in the turn's own class: {commit.get('cuts')} {commit.get('priority')} {first.get('priority')}")
        reply = fixture.json.loads(text)
        engine.reply("The file is listed.</think>Done.")
        history = messages + [{"role": "assistant", "content": reply["content"]}, {"role": "user", "content": "list the directory"}]
        status, headers, text = await harness.post("/v1/messages", dict(body, messages=history))
        second = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        check(status == 200 and second["prompt_token_ids"][:len(prompt) + len(generated)] == prompt + generated and second.get("cuts") == [opened, len(prompt) - 1, len(prompt) + len(generated)],
              f"the next turn names that cut and where the committed reply ends, so it resumes at the commit's end: {second.get('cuts')} {len(prompt)} {len(generated)}")
        check(engine.resumed(commit) == len(prompt) - 1 and engine.resumed(second) == len(prompt) + len(generated),
              f"by the engine's rules the commit resumes one token before the turn's prompt ended and the next turn where the reply ends: {engine.resumed(commit)} {engine.resumed(second)}")
        found = await commits_seen(engine, 2)
        check(len(found) == 2 and found[1]["cuts"] == [opened, len(prompt) - 1, len(prompt) + len(generated), len(second["prompt_token_ids"]) - 1],
              f"the second reply's commit names every earlier cut and one token before its own prompt's end: {[request.get('cuts') for request in found]}")
        before = len([request for request in engine.requests if request.get("commit_only")])
        engine.reply("Cut short.</think>Before STOPHERE after")
        status, headers, text = await harness.post("/v1/messages", dict(body, messages=[{"role": "user", "content": "hello world"}], stop_sequences=["STOPHERE"]))
        await asyncio.sleep(0.15)
        after = len([request for request in engine.requests if request.get("commit_only")])
        check(status == 200 and fixture.json.loads(text).get("stop_reason") == "stop_sequence" and after == before,
              f"a reply cut at a stop sequence is not recorded for replay, so it is not committed either: the next turn could not name its cut: {status} {before} {after} {text[:160]}")
        frontend = harness.frontend
        body = dict(body, system=" ".join([SYSTEM] * 5))
        await settled(frontend)
        before = len(await commits_seen(engine, 0))
        engine.commit_hold = asyncio.Event()
        opening = [{"role": "user", "content": "read z.txt"}]
        engine.reply("The user wants z.txt read.</think>Reading z.")
        status, headers, text = await harness.post("/v1/messages", dict(body, messages=opening))
        reply = fixture.json.loads(text)
        held = await commits_seen(engine, before + 1)
        asked = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        follow = opening + [{"role": "assistant", "content": reply["content"]}, {"role": "user", "content": "now y.txt"}]
        engine.reply("Next.</think>Reading y.")
        began = time.monotonic()
        status, headers, text = await asyncio.wait_for(harness.post("/v1/messages", dict(body, messages=follow)), 10)
        unmeasured = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        check(status == 200 and len(held) >= before + 1 and time.monotonic() - began < 2 and engine.resumed(unmeasured) == len(asked["prompt_token_ids"]) - 1,
              f"with no measured prefill rate for the model a turn does not wait for its reply's commit: the commit is withdrawn and the turn computes the reply itself, "
              f"resuming one token before the earlier prompt's end: {status} {round(time.monotonic() - began, 2)} {engine.resumed(unmeasured)} {len(asked['prompt_token_ids'])}")
        withdrawn = [pending for pending in frontend.reply_commits if pending.ids == held[before]["prompt_token_ids"]]
        check(not withdrawn, f"the withdrawn commit is forgotten: {len(withdrawn)}")
        engine.commit_hold.set()
        check(await settled(frontend), "every commit ends once the engine answers")
        size = len(held[before]["prompt_token_ids"])
        frontend.replay.observe_prefill("alpha", size, size, size * 1000.0)
        check(size >= 512 and frontend.replay.prefill_rate("alpha", size) == 1000.0, f"the test's prompts are long enough to have a measured prefill rate: {size}")
        before = len(await commits_seen(engine, 0))
        engine.commit_hold = asyncio.Event()
        opening = [{"role": "user", "content": "read b.txt"}]
        engine.reply("The user wants b.txt read.</think>Reading b.")
        status, headers, text = await harness.post("/v1/messages", dict(body, messages=opening))
        reply = fixture.json.loads(text)
        held = await commits_seen(engine, before + 1)
        sent = len(engine.requests)
        follow = opening + [{"role": "assistant", "content": reply["content"]}, {"role": "user", "content": "now c.txt"}]
        engine.reply("Next.</think>Reading c.")
        turn = asyncio.ensure_future(harness.post("/v1/messages", dict(body, messages=follow)))
        await asyncio.sleep(0.2)
        check(status == 200 and len(held) == before + 1 and not turn.done() and len(engine.requests) == sent,
              f"a turn that carries a reply whose commit is at the engine is not sent while the commit runs: {status} {len(held)} {turn.done()} {len(engine.requests) - sent}")
        engine.reply("Elsewhere.</think>Other.")
        status, headers, text = await harness.post("/v1/messages", dict(body, messages=[{"role": "user", "content": "another conversation"}]))
        check(status == 200 and not turn.done(), f"a turn of another conversation does not wait for that commit: {status} {turn.done()}")
        engine.commit_hold.set()
        engine.commit_hold = None
        status, headers, text = await turn
        joined = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        check(status == 200 and joined.get("cuts", [None])[-1] == len(held[before]["prompt_token_ids"]) and joined["prompt_token_ids"][:joined["cuts"][-1]] == held[before]["prompt_token_ids"],
              f"the turn is sent once the commit is done and names the commit's end as its last cut: {status} {joined.get('cuts')} {len(held[before].get('prompt_token_ids', []))}")
        check(engine.resumed(joined) == len(held[before]["prompt_token_ids"]), f"by the engine's rules the turn that waited resumes where the commit ended: {engine.resumed(joined)}")
        await commits_seen(engine, before + 3)
        check(await settled(frontend), f"finished commits are forgotten: {len(frontend.reply_commits)}")

        before = len(await commits_seen(engine, 0))
        engine.commit_hold, engine.commit_fails = asyncio.Event(), True
        opening = [{"role": "user", "content": "read d.txt"}]
        engine.reply("The user wants d.txt read.</think>Reading d.")
        status, headers, text = await harness.post("/v1/messages", dict(body, messages=opening))
        reply = fixture.json.loads(text)
        held = await commits_seen(engine, before + 1)
        asked = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        follow = opening + [{"role": "assistant", "content": reply["content"]}, {"role": "user", "content": "now e.txt"}]
        engine.reply("Next.</think>Reading e.")
        turn = asyncio.ensure_future(harness.post("/v1/messages", dict(body, messages=follow)))
        await asyncio.sleep(0.2)
        waited = not turn.done()
        engine.commit_hold.set()
        status, headers, text = await asyncio.wait_for(turn, 10)
        engine.commit_hold, engine.commit_fails = None, False
        failed = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        check(await settled(frontend), "every commit ends after the failed one")
        check(status == 200 and waited and len(held) >= before + 1 and engine.resumed(failed) == len(asked["prompt_token_ids"]) - 1,
              f"a commit that fails at the engine does not fail the turn that waited for it: the turn computes the reply itself, resuming one token before the "
              f"earlier prompt's end: {status} {waited} {engine.resumed(failed)} {len(asked['prompt_token_ids'])}")

        before = len(await commits_seen(engine, 0))
        patience, floor = chat_frontend.REPLY_COMMIT_PATIENCE, chat_frontend.REPLY_COMMIT_FLOOR_SECONDS
        chat_frontend.REPLY_COMMIT_PATIENCE, chat_frontend.REPLY_COMMIT_FLOOR_SECONDS = 0.0, 0.4
        engine.commit_hold = asyncio.Event()
        opening = [{"role": "user", "content": "read f.txt"}]
        engine.reply("The user wants f.txt read.</think>Reading f.")
        status, headers, text = await harness.post("/v1/messages", dict(body, messages=opening))
        reply = fixture.json.loads(text)
        held = await commits_seen(engine, before + 1)
        asked = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        follow = opening + [{"role": "assistant", "content": reply["content"]}, {"role": "user", "content": "now g.txt"}]
        engine.reply("Next.</think>Reading g.")
        began = time.monotonic()
        turn = asyncio.ensure_future(harness.post("/v1/messages", dict(body, messages=follow)))
        await asyncio.sleep(0.15)
        waited = not turn.done()
        status, headers, text = await asyncio.wait_for(turn, 10)
        took = time.monotonic() - began
        chat_frontend.REPLY_COMMIT_PATIENCE, chat_frontend.REPLY_COMMIT_FLOOR_SECONDS = patience, floor
        stalled = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        check(status == 200 and waited and 0.3 <= took < 4 and engine.resumed(stalled) == len(asked["prompt_token_ids"]) - 1,
              f"a commit that does not finish in the time its own work should take is withdrawn, and the turn computes the reply itself: {status} {waited} {round(took, 2)} "
              f"{engine.resumed(stalled)} {len(asked['prompt_token_ids'])}")
        withdrawn = [pending for pending in frontend.reply_commits if pending.ids == held[before]["prompt_token_ids"]]
        check(not withdrawn, f"the commit that did not finish is forgotten: {len(withdrawn)}")
        engine.commit_hold.set()
        engine.commit_hold = None
        check(await settled(frontend), f"every commit ends: {len(frontend.reply_commits)}")
        served = fixture.json.loads((ROOT / "deployment" / "hub" / "chat_frontend.json").read_text())
        check(served["models"] and not any(spec.get("reply_commit") for spec in served["models"]),
              "no served model commits its replies: the setting stays off for a model until gate GX and the turn floor have run both forms on it")
        timed = chat_frontend.ReplyCommit("alpha", ("own", None), list(range(size)), 10)
        check(frontend.reply_patience(frontend.models["alpha"], timed) == 0.0, "a commit that has not reached the engine is waited for no time at all")
        timed.sent = time.monotonic()
        patient = frontend.reply_patience(frontend.models["alpha"], timed)
        check(abs(patient - 2.0 * 10 * 1000.0 / 1000) < 0.5,
              f"a commit at the engine is waited for at most twice the time its tokens take at the model's measured prefill rate: {patient}")
        timed.sent = time.monotonic() - 3600
        check(frontend.reply_patience(frontend.models["alpha"], timed) == 0.0, "a commit that is already past that time is not waited for")
        tools = [{"name": "read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
        asked = [{"role": "user", "content": "read the file a.txt"}]
        seen = len(await commits_seen(engine, 0))
        engine.reply("read<arg_key>path</arg_key><arg_value>a.txt</arg_value></tool_call>")
        status, headers, text = await harness.post("/v1/messages", dict(body, tools=tools, tool_choice={"type": "any"}, messages=asked))
        forced = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        prompt = forced["prompt_token_ids"]
        opening = len(engine.tokens("</think><tool_call>"))
        before = len(prompt) - opening - 1
        check(status == 200 and harness.tokenizer.decode(prompt[-opening:], skip_special_tokens=False) == "</think><tool_call>" and
              forced.get("cuts", [None])[-1] == before and forced.get("keep", [None])[-1] == before and len(forced["cuts"]) >= 2,
              f"a turn with a forced tool choice names one token before its forced opening as a cut and asks for state there: {forced.get('cuts')} {forced.get('keep')} {len(prompt)}")
        found = await commits_seen(engine, seen + 1)
        commit = found[-1] if len(found) > seen else {}
        check(commit.get("cuts") == forced["cuts"] and commit.get("prompt_token_ids", [])[:len(prompt)] == prompt,
              f"its reply's commit names the same cuts, with none at the forced opening's end, so its waves are the next turn's: {commit.get('cuts')} {forced.get('cuts')}")
        reply = fixture.json.loads(text)
        call = next((block for block in reply.get("content", []) if block.get("type") == "tool_use"), {})
        engine.reply("The file says hello.</think>It says hello.")
        follow = asked + [{"role": "assistant", "content": reply.get("content", [])}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.get("id"), "content": "hello"}]}]
        status, headers, text = await harness.post("/v1/messages", dict(body, tools=tools, messages=follow))
        after = next(request for request in reversed(engine.requests) if "max_tokens" in request)
        check(status == 200 and call.get("name") == "read" and after.get("cuts") == forced["cuts"] + [len(commit.get("prompt_token_ids", []))] and
              after["prompt_token_ids"][:len(commit.get("prompt_token_ids", []))] == commit.get("prompt_token_ids"),
              f"the turn after a forced tool call names that cut and the reply's end, so it resumes where the committed reply ends: {after.get('cuts')} {forced.get('cuts')}")
        check(engine.resumed(commit) == before and engine.resumed(after) == len(commit.get("prompt_token_ids", [])),
              f"by the engine's rules the forced turn's commit resumes one token before the forced opening and the next turn where the reply ends: {engine.resumed(commit)} {engine.resumed(after)}")
        await commits_seen(engine, seen + 2)
        for _ in range(300):
            if not frontend.reply_commits:
                break
            await asyncio.sleep(0.01)
        gate = ("account:a", "m1")
        waiting = chat_frontend.ReplyCommit("alpha", gate, [5, 6, 7])
        others = [chat_frontend.ReplyCommit("alpha", gate, [5, 6, 9]), chat_frontend.ReplyCommit("alpha", ("account:b", "m1"), [5, 6, 7]), chat_frontend.ReplyCommit("beta", gate, [5, 6, 7])]
        for pending in [waiting] + others:
            pending.task = asyncio.ensure_future(asyncio.sleep(30))
        frontend.reply_commits.update([waiting] + others)
        await frontend.reply_joined(frontend.models["alpha"], {"engine_body": {"prompt_token_ids": [5, 6, 7, 8, 9], "cuts": [2, 3]}, "gate": gate})
        await asyncio.sleep(0)
        check(waiting.task.cancelled() and not any(pending.task.done() for pending in others),
              f"a commit that has not reached the engine is withdrawn when its next turn arrives, and none of another reply, account or model is: {waiting.task.cancelled()} {[pending.task.done() for pending in others]}")
        for pending in others:
            pending.task.cancel()
        frontend.reply_commits.clear()


def main():
    check = fixture.Checks("test_chat_frontend_commit")
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp), check))
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run_reply_commit(Path(tmp), check))
    return check.finish("a commit computes and keeps a prompt on every shape and generates nothing; with reply_commit each finished reply is committed at once, the next turn names its end and waits for a commit that is running, no longer than its work should take; a commit that fails, stalls or has not started never fails or holds the turn; no served model has the setting on")


if __name__ == "__main__":
    raise SystemExit(main())
