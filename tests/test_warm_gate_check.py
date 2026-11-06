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
TOOL = importlib.util.spec_from_file_location("warm_gate_check", ROOT / "tools" / "warm_gate_check.py")
warm_gate_check = importlib.util.module_from_spec(TOOL)
TOOL.loader.exec_module(warm_gate_check)
COUNT = "\n".join(str(number) for number in range(1, 40))


async def measure(harness, tmp, contend):
    engine = harness.engine
    engine.pace, engine.commit_delay, engine.contend = 0.01, 0.4, contend
    engine.reply(COUNT, "length")
    engine.reply(COUNT, "length")
    corpus = tmp / "corpus.txt"
    corpus.write_text("hello world, the weather in Paris is sunny today. read the file a.txt and list the directory.")
    headers = harness.headers(account="acct-gate")
    config = tmp / "hub.json"
    config.write_text(json.dumps({"models": [{"id": "alpha", "cold_prefill": {"points": [[100, 10], [10000, 1000]], "source": "test"}}]}))
    arguments = ["--base", harness.base, "--model", "alpha", "--corpus", str(corpus), "--tokens", "600", "--decode-tokens", "64", "--lead-seconds", "0.05",
                 "--hub-config", str(config), "--out", str(tmp / f"gate-{contend}.json")] + [item for name, value in headers.items() for item in ("--header", f"{name}: {value}")]
    with contextlib.redirect_stdout(io.StringIO()):
        code = await asyncio.get_running_loop().run_in_executor(None, warm_gate_check.main, arguments)
    return code, json.loads((tmp / f"gate-{contend}.json").read_text())


async def run(tmp, check):
    async with fixture.Harness(tmp / "quiet") as harness:
        code, report = await measure(harness, tmp, 0.0)
    check(code == 0 and report["matched"] and report["yielded"] and report["interactive"]["overlapped_warm"] and report["warm"]["prompt_tokens"] >= 600 and
          report["real"]["cached_tokens"] == report["warm"]["committed_tokens"] > 0 and report["interactive"]["during_gaps"] > 0,
          f"a warm-up that leaves an interactive stream's pace alone passes, with the next request matching every warmed block: {code} {report}")
    seed = report["seed"]
    check(seed["measured_ms"] == 12 and seed["estimate_ms"] == round(0.1 * report["warm"]["prompt_tokens"]) and
          seed["error"] == round(seed["estimate_ms"] / 12 - 1, 4) and seed["within_10_percent"] is False,
          f"the report compares the model's cold_prefill seed with the warm-up's measured cold prefill: {seed}")
    async with fixture.Harness(tmp / "contended") as harness:
        code, report = await measure(harness, tmp, 0.05)
    check(code == 1 and report["matched"] and not report["yielded"] and report["interactive"]["delay_steps"] > 1,
          f"a warm-up that slows the interactive stream by more than one decode step fails the gate: {code} {report['interactive']}")


def main():
    check = fixture.Checks("test_warm_gate_check")
    captured = io.StringIO()
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(captured):
        asyncio.run(run(Path(directory), check))
    if check.failures:
        print(captured.getvalue()[-3000:])
    check.finish("tools/warm_gate_check.py warms a prompt of the asked length, checks the next real request caches every warmed block, and fails "
                 "when an interactive stream decoding through the warm-up is delayed by more than one decode step; it compares the model's cold_prefill seed "
                 "with the warm-up's measured cold prefill")


if __name__ == "__main__":
    main()
