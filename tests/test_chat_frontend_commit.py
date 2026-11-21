#!/usr/bin/env python3
"""A commit computes a prompt, keeps its state and generates nothing: max_tokens 0 on the Messages shape, commit_only on the Chat Completions and Responses shapes. The chat layer plans the request as any other, sends the engine a commit, and answers one cache.commit object with the prompt, cached and committed token counts whatever the request says about streaming; a later request with the same prompt finds it cached. A commit takes no output limit or log-probs, no forced tool choice and no WebSocket, and an engine that generates tokens for a commit is refused. A model whose settings name reply_commit commits each finished reply at once, apart from the reply: the engine gets the turn's prompt and the reply's tokens as a commit in the turn's own class that names one token before the prompt's end as a cut, and the next turn names that cut and the reply's end; without the setting, or for a reply that is not recorded for replay, no commit follows. A turn that carries a reply whose commit is at the engine waits for it and is sent afterwards, so the reply is computed once; a commit that has not reached the engine is withdrawn, and a turn of another conversation waits for neither. Conversations with the same system prompt name the same opening cut, where the first message begins, and ask for state there; control-token text inside a system prompt does not move it. A turn with a forced tool choice names one token before its forced opening as a cut and asks for state there; its reply's commit names the same cuts, and the turn after it names that cut and the reply's end."""
import asyncio
import importlib.util
import tempfile
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
        check(status == 200 and joined.get("cuts", [None])[-1] == len(held[-1]["prompt_token_ids"]) and joined["prompt_token_ids"][:joined["cuts"][-1]] == held[-1]["prompt_token_ids"],
              f"the turn is sent once the commit is done and names the commit's end as its last cut: {status} {joined.get('cuts')} {len(held[-1].get('prompt_token_ids', []))}")
        await commits_seen(engine, before + 3)
        frontend = harness.frontend
        for _ in range(300):
            if not frontend.reply_commits:
                break
            await asyncio.sleep(0.01)
        check(not frontend.reply_commits, f"finished commits are forgotten: {len(frontend.reply_commits)}")
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
              forced.get("cuts", [None])[-1] == before and forced.get("keep", [None])[-1] == before and len(forced["cuts"]) == 2,
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
    return check.finish("a commit computes and keeps a prompt on every shape and generates nothing; with reply_commit each finished reply is committed at once, the next turn names its end and waits for a commit that is running")


if __name__ == "__main__":
    raise SystemExit(main())
