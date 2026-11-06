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
SPEC = importlib.util.spec_from_file_location("count_tokens_check", ROOT / "tools" / "count_tokens_check.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)
TOOLS = [{"name": "Read", "description": "Reads a file.", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}]


def body(text, **extra):
    request = {"model": "claude-sonnet-4-5", "max_tokens": 32000, "stream": True, "system": [{"type": "text", "text": "You are Claude Code."}], "tools": TOOLS,
               "thinking": {"type": "enabled", "budget_tokens": 8000}, "metadata": {"user_id": json.dumps({"session_id": "s-1"})},
               "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}
    request.update(extra)
    return request


async def run(tmp, check):
    bodies = tmp / "bodies"
    bodies.mkdir()
    for index, text in enumerate(["Read a.txt", "List the directory and read b.txt", "What does the weather tool say?"]):
        (bodies / f"{index:03d}.json").write_text(json.dumps(body(text)))
    async with fixture.Harness(tmp) as harness:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            status = await asyncio.to_thread(tool.main, ["--base", harness.base, str(bodies)])
        lines = printed.getvalue().splitlines()
        rows = [json.loads(line) for line in lines[:-1]]
        check(status == 0 and len(rows) == 3 and all(row["same"] and row["count_tokens"] == usage["prompt_tokens"] for row, usage in zip(rows, harness.engine.usage))
              and lines[-1].startswith("PASS"), f"every recorded body's count equals the engine's prompt length: {lines}")
        harness.frontend.models["alpha"].render = broken(harness.frontend.models["alpha"].render)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            status = await asyncio.to_thread(tool.main, ["--base", harness.base, str(bodies)])
        check(status == 1 and printed.getvalue().splitlines()[-1].startswith("FAIL"), f"a count that differs from the engine's fails the check: {printed.getvalue()[-300:]}")
        empty = tmp / "empty"
        empty.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            check(await asyncio.to_thread(tool.main, ["--base", harness.base, str(empty)]) == 1, "a directory with no bodies fails")


def broken(render):
    calls = []

    def wrapped(messages, tools, kwargs):
        calls.append(1)
        text, ids = render(messages, tools, kwargs)
        return (text, list(ids) + list(ids[-1:])) if len(calls) % 2 == 1 else (text, ids)
    return wrapped


def main():
    check = fixture.Checks("test_count_tokens_check")
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
        asyncio.run(run(Path(directory), check))
    check.finish("tools/count_tokens_check.py compares count_tokens with the engine's prompt length for recorded Claude Code bodies, and fails on a difference "
                 "or on an empty directory")


if __name__ == "__main__":
    main()
