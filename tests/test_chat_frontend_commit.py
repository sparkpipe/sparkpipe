#!/usr/bin/env python3
"""A commit computes a prompt, keeps its state and generates nothing: max_tokens 0 on the Messages shape, commit_only on the Chat Completions and Responses shapes. The chat layer plans the request as any other, sends the engine a commit, and answers one cache.commit object with the prompt, cached and committed token counts whatever the request says about streaming; a later request with the same prompt finds it cached. A commit takes no output limit or log-probs, no forced tool choice and no WebSocket, and an engine that generates tokens for a commit is refused."""
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


def main():
    check = fixture.Checks("test_chat_frontend_commit")
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp), check))
    return check.finish("a commit computes and keeps a prompt on every shape and generates nothing")


if __name__ == "__main__":
    raise SystemExit(main())
