#!/usr/bin/env python3
"""An imported conversation pre-warms the base its next request renders. POST /v1/sparkpipe/memory/import takes a chat request's template settings (reasoning_effort, chat_template_kwargs), so on a template that renders the reasoning effort ahead of the conversation, as Qwen 3.8's does at the top of its system block, the next request at that effort extends the import's pre-warmed prompt token for token and finds it in the engine's cache. An effort the model does not offer is refused. The import checks its identity once, when it arrives, so an import that outlives the token it was sent with still imports every batch."""
import asyncio
import importlib.util
import json
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
cm = load("chat_memory", ROOT / "tools" / "serving" / "chat_memory.py")
qualify = load("chat_memory_qualify", ROOT / "tools" / "chat_memory_qualify.py")
SECRET = "s" * 48
BUDGETS = {"open_budget": 1500, "line_tokens": 48, "group_size": 16, "working_tokens": 24576, "working_context_share": 0.1875, "ledger_tokens": 12288,
           "ledger_context_share": 0.1, "part_small_tokens": 1024, "part_cap_tokens": 12288, "keep_tokens": 4096, "target_tolerance": 0.25,
           "target_minimum_tokens": 16384}
HISTORY = [{"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "Read a.txt"}, {"role": "assistant", "content": "It says hello world."},
           {"role": "user", "content": "Read b.txt"}, {"role": "assistant", "content": "It says bye."},
           {"role": "user", "content": "Read c.txt"}, {"role": "assistant", "content": "It is empty."}]


async def post(harness, path, body, headers):
    async with harness.session.post(harness.base + path, json=body, headers=headers) as response:
        return response.status, await response.text()


async def run(tmp, check):
    secret = tmp / "memory.secret"
    secret.write_text(SECRET)
    memory = {"store": str(tmp / "memory-store"), "jwt_secret_file": str(secret), "idle_seconds": 0.01, "budgets": BUDGETS}
    async with fixture.Harness(tmp, extra={"chat_memory": memory}) as harness:
        engine = harness.engine
        identity = {"X-OpenWebUI-User-Jwt": qualify.token(SECRET, "user-1"), "X-OpenWebUI-Chat-Id": "chat-import", "X-OpenWebUI-Task": ""}

        status, text = await post(harness, "/v1/sparkpipe/memory/import", {"model": "alpha", "messages": HISTORY, "batch": 8, "reasoning_effort": "highest"},
                                  identity)
        check(status == 400 and "reasoning_effort" in text, f"an effort the model does not offer is refused: {status} {text[:200]}")

        lifetime = 2.0
        expiring = dict(identity, **{"X-OpenWebUI-User-Jwt": qualify.token(SECRET, "user-1", lifetime), "X-OpenWebUI-Chat-Id": "chat-expiring"})
        minted, before = time.time(), len(engine.requests)
        engine.hold = asyncio.Event()
        pending = asyncio.ensure_future(post(harness, "/v1/sparkpipe/memory/import", {"model": "alpha", "messages": HISTORY, "batch": 1, "reasoning_effort": "low"},
                                             expiring))
        while len(engine.requests) == before:
            await asyncio.sleep(0.01)
        await asyncio.sleep(max(0.0, minted + lifetime - time.time()) + 0.2)
        engine.hold.set()
        engine.hold = None
        status, text = await pending
        check(status == 200 and json.loads(text)["exchanges"] == 3 and json.loads(text)["lined"] == 3,
              f"an import whose token expires after its first batch still imports every batch: {status} {text[:200]}")

        engine.reply(json.dumps({"lines": {"e1": "The user read a.txt.", "e2": "The user read b.txt.", "e3": "The user read c.txt."}, "open": []}))
        status, text = await post(harness, "/v1/sparkpipe/memory/import", {"model": "alpha", "messages": HISTORY, "batch": 8, "reasoning_effort": "low"},
                                  identity)
        check(status == 200 and json.loads(text)["lined"] == 3, f"the import lines every exchange: {status} {text[:200]}")
        warmed = list(engine.requests[-1]["prompt_token_ids"])
        check("<|im_start|>system\nReasoning effort: low<|im_end|>" in harness.prompt_text(), f"the import pre-warms at the effort it was given: {harness.prompt_text()[:160]}")

        last = cm.exchanges_of(HISTORY)[1][-1]
        turn = dict(identity, **{"X-OpenWebUI-User-Message-Id": "u-next", "X-OpenWebUI-Message-Id": "a-next",
                                 "X-OpenWebUI-User-Message-Parent-Id": "import-reply-" + cm.exchange_key(last)[:24]})
        engine.reply("Done.</think>It said hello world.")
        answer = len(engine.requests)
        status, text = await post(harness, "/v1/chat/completions", {"model": "alpha", "messages": HISTORY + [{"role": "user", "content": "What did a.txt say?"}],
                                                                    "reasoning_effort": "low", "max_tokens": 64}, turn)
        prompt = list(engine.requests[answer]["prompt_token_ids"])
        check(status == 200 and prompt[:len(warmed)] == warmed and len(prompt) > len(warmed),
              f"the next request at that effort extends the pre-warmed prompt token for token: {status} {len(prompt)} {len(warmed)}")
        check(engine.usage[answer]["cached_tokens"] >= len(warmed) - 1, f"and finds it in the cache: {engine.usage[answer]} of {len(warmed)} warmed")


def main():
    checks = fixture.Checks("test_chat_memory_import")
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp), checks))
    checks.finish("an import pre-warms the base its next request renders, at the template settings it was given; an effort the model does not offer is refused, and an import outlives its token")


if __name__ == "__main__":
    main()
