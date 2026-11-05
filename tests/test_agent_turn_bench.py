#!/usr/bin/env python3
import asyncio
import http.server
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import chat_harness_fixture as fixture


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("agent_turn_bench", ROOT / "tools" / "agent_turn_bench.py")
watch = load("fleet_memory_watch", ROOT / "tools" / "fleet_memory_watch.py")

MESSAGES_REPLY = "The next file is due.</think>Read it next.<tool_call>Read<arg_key>file_path</arg_key><arg_value>/repo/b.c</arg_value></tool_call>"
RESPONSES_REPLY = "The next file is due.</think>Read it next.<tool_call>exec_command<arg_key>cmd</arg_key><arg_value>cat /repo/b.c</arg_value></tool_call>"


class Endless:
    def __init__(self, engine):
        self.engine = engine
        self.text = MESSAGES_REPLY

    def __bool__(self):
        return True

    def pop(self, index):
        return self.engine.tokens(self.text) + [self.engine.tokenizer.token_to_id("<|im_end|>")], "stop"


class Background:
    def __init__(self, tmp, sequences):
        self.tmp, self.sequences = tmp, sequences
        self.ready = threading.Event()
        self.loop = None

    def __enter__(self):
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()
        if not self.ready.wait(60):
            raise SystemExit("FAIL test_agent_turn_bench: the chat layer harness did not start")
        return self

    def serve(self):
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.hold())

    async def hold(self):
        self.stop = asyncio.Event()
        async with fixture.Harness(self.tmp, models=("alpha",), active="alpha", sequences=self.sequences) as harness:
            self.harness = harness
            self.replies = Endless(harness.engine)
            harness.engine.replies = self.replies
            self.ready.set()
            await self.stop.wait()

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(30)


def corpus(directory):
    directory.mkdir(parents=True, exist_ok=True)
    words = "hello world the weather in paris is sunny today read the file and list the directory".split()
    for index in range(12):
        lines = [" ".join(words[(index + line) % len(words):] + words[:(index + line) % len(words)]) for line in range(40)]
        (directory / f"part{index}.c").write_text("\n".join(lines) + "\n")
    (directory / "README.md").write_text("Project notes. " * 40)
    return directory


def run_bench(base, key_file, corpus_dir, output, shapes, sessions, label, extra=()):
    arguments = bench.build_parser().parse_args([
        "run", "--endpoint", base, "--key-file", str(key_file), "--model", "alpha", "--corpus", str(corpus_dir), "--sessions", sessions,
        "--shapes", shapes, "--turns", "3", "--max-tokens", "256", "--context-low", "12000", "--context-high", "40000", "--result-tokens", "150:300",
        "--preamble-tokens", "600", "--stagger", "0.2", "--jitter", "0", "--label", label, "--output", str(output), *extra])
    bench.run(arguments)
    return bench.load_records([output])


class RetryStub(http.server.BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        RetryStub.calls.append(time.monotonic())
        if len(RetryStub.calls) == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0.3")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        frames = [("message_start", {"type": "message_start", "message": {"usage": {}}}),
                  ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                  ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
                  ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                  ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                                     "usage": {"input_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 5, "output_tokens": 1},
                                     "sparkpipe": {"queue_ms": 7, "prefill_ms": 40, "prompt_tokens": 15, "cached_tokens": 5, "prefill_tokens": 10, "output_tokens": 1}}),
                  ("message_stop", {"type": "message_stop"})]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("X-SparkPipe-Queue-Ms", "7")
        self.end_headers()
        for name, data in frames:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)


def check_end_to_end(tmp, check):
    corpus_dir = corpus(tmp / "corpus")
    with Background(tmp / "chat", 4) as background:
        harness = background.harness
        key_file = tmp / "key"
        key_file.write_text(f"{harness.key_id} {harness.key}\n")
        key_file.chmod(0o640)
        try:
            bench.read_key(key_file)
            check(False, "a group-readable key file was accepted")
        except SystemExit as error:
            check("mode 640" in str(error), f"the key file refusal does not name its mode: {error}")
        key_file.chmod(0o600)
        check(bench.read_key(key_file) == harness.key, "read_key did not take the sk-sp- word from an api_keys.py line")
        records = run_bench(harness.base, key_file, corpus_dir, tmp / "messages", "messages", "1,3", "m")
        engine_requests = list(harness.engine.requests)
        background.replies.text = RESPONSES_REPLY
        responses = run_bench(harness.base, key_file, corpus_dir, tmp / "responses", "responses", "2", "r", ("--no-warm",))
        check(len(harness.engine.requests) - len(engine_requests) == 6, "the Responses run did not send six turns to the engine")
    run = records[0]
    check(run["phase"] == "run" and run["model_entry"]["id"] == "alpha" and run["public_key"] is True, f"the run record is wrong: {json.dumps(run)[:300]}")
    warms = [record for record in records if record["phase"] == "warm"]
    turns = [record for record in records if record["phase"] == "turn"]
    check(len(warms) == 3 and all(record["error"] is None and record["output_tokens"] == 1 for record in warms),
          f"the warm phase did not prefill each of three seeds with one output token: {[(r['error'], r['output_tokens']) for r in warms]}")
    check(all(warm["prompt_tokens"] >= 12000 * 0.8 for warm in warms), f"the seeds did not reach the low end of the context range: {[w['prompt_tokens'] for w in warms]}")
    check(len({warm["prompt_tokens"] for warm in warms}) == 3, "three sessions seeded the same context size")
    check(len(turns) == 3 + 9 and all(record["error"] is None for record in turns), f"expected 12 clean turns, got {len(turns)}: {[t['error'] for t in turns if t['error']]}")
    by_session = {}
    for record in turns:
        by_session.setdefault((record["arm"], record["session"]), []).append(record)
    check(sorted(by_session) == [(1, 0), (3, 0), (3, 1), (3, 2)], f"the arms ran the wrong sessions: {sorted(by_session)}")
    for (arm, session), group in by_session.items():
        group.sort(key=lambda record: record["turn"])
        warm = next(record for record in warms if record["session"] == session)
        check(group[0]["prompt_tokens"] == warm["prompt_tokens"], f"arm {arm} session {session} turn 0 did not resume the warmed seed")
        check(group[0]["prefill_tokens"] <= 64, f"arm {arm} session {session} prefilled {group[0]['prefill_tokens']} tokens after the warm, not a cache hit")
        for before, after in zip(group, group[1:]):
            check(after["cached_tokens"] >= before["prompt_tokens"], f"arm {arm} session {session} turn {after['turn']} lost the cached prefix "
                  f"({after['cached_tokens']} cached of a previous prompt of {before['prompt_tokens']})")
            check(150 * 0.5 <= after["prefill_tokens"] <= 300 * 3 + 200, f"turn {after['turn']} prefilled {after['prefill_tokens']} tokens for a 150-300 token result")
        check(all(record["tool_calls"] == 1 and record["tool_results"] == 1 and record["stop"] == "tool_use" for record in group),
              f"arm {arm} session {session} did not answer each tool_use with a tool_result")
        check(all(record["ttft_s"] is not None and 0 < record["ttft_s"] <= record["turn_s"] for record in group), "a turn lacks a first-token time inside the turn")
        check(all(record["queue_ms"] is not None and record["prefill_ms"] is not None and record["request"] for record in group),
              "a turn lacks the chat layer's queue, prefill or request telemetry")
        thinking = group[0]["assistant"][0]
        check(thinking["type"] == "thinking" and thinking["signature"], f"the assistant record dropped the thinking block or its signature: {thinking}")
    starts = sorted(group[0]["offset_s"] for (arm, _), group in by_session.items() if arm == 3)
    check(starts[1] - starts[0] >= 0.18 and starts[2] - starts[1] >= 0.18, f"arm 3 sessions did not arrive 0.2 s apart: {starts}")
    check(len(engine_requests) == 3 + 12, f"the chat layer sent {len(engine_requests)} engine requests for 3 warms and 12 turns")
    response_turns = [record for record in responses if record["phase"] == "turn"]
    check(len(response_turns) == 6 and all(record["error"] is None and record["shape"] == "responses" for record in response_turns),
          f"the Responses run did not complete 6 turns: {[r['error'] for r in response_turns]}")
    check(not any(record["phase"] == "warm" for record in responses), "--no-warm still warmed")
    for session in (0, 1):
        group = sorted((record for record in response_turns if record["session"] == session), key=lambda record: record["turn"])
        check(group[0]["prefill_tokens"] > 1000, "the first Responses turn without a warm was not a cold prefill")
        for before, after in zip(group, group[1:]):
            check(after["cached_tokens"] >= before["prompt_tokens"], f"Responses session {session} lost the cached prefix at turn {after['turn']}")
        check(all(record["tool_calls"] == 1 and record["tool_results"] == 1 for record in group), "a Responses turn did not answer its function_call")
        check(all(item.get("type") != "reasoning" or "encrypted_content" not in item for record in group for item in record["assistant"]),
              "the receipt keeps the reasoning item's encrypted content")
    summary = json.loads((tmp / "messages" / "summary.json").read_text())
    arm3 = next(arm for arm in summary["arms"] if arm["arm"] == 3)
    check(arm3["sessions"] == 3 and arm3["turns_ok"] == 9 and arm3["turns_with_tool_calls"] == 9, f"the arm 3 summary is wrong: {json.dumps(arm3)[:400]}")
    check(abs(arm3["share"]["waiting_for_first_token"] + arm3["share"]["decode"] - 1) < 1e-6, "the waiting and decode shares do not add to one")
    check(summary["warm"]["m"]["seeds"] == 3, "the warm summary does not count three seeds")
    table = bench.markdown(bench.summarize(bench.load_records([tmp / "messages", tmp / "responses"])))
    check("| m 3 |" in table and "| r 2 |" in table, f"the merged report does not label its runs:\n{table}")


def check_retry(check):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RetryStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = bench.Client(f"http://127.0.0.1:{server.server_address[1]}", "sk-sp-test", public=True, max_retries=2)
        result = client.stream("/v1/messages", {"model": "alpha"}, {}, "messages")
    finally:
        server.shutdown()
    check(result["error"] is None and result["attempts"] == 2 and result["statuses"] == [429, 200], f"a 429 was not retried once: {result.get('statuses')}")
    check(abs(result["retry_wait_s"] - 0.3) < 1e-9, f"the retry did not wait Retry-After: {result['retry_wait_s']}")
    check(result["first"] - result["started"] >= 0.3, "the first-token time does not count the retry wait from the first attempt")
    check(result["blocks"] == [{"type": "text", "text": "hi"}] and result["prompt_tokens"] == 15 and result["cached_tokens"] == 5,
          f"the stream was parsed wrong: {result['blocks']} {result['prompt_tokens']} {result['cached_tokens']}")


def check_summary(check):
    def turn(session, offset, ttft, seconds, output, prefill_ms, queue_ms, prompt):
        return {"session": session, "offset_s": offset, "end_offset_s": offset + seconds, "ttft_s": ttft, "turn_s": seconds, "decode_s": seconds - ttft,
                "output_tokens": output, "prefill_tokens": 1000, "prefill_ms": prefill_ms, "queue_ms": queue_ms, "retry_wait_s": 0, "attempts": 1,
                "prompt_tokens": prompt, "cached_tokens": prompt - 1000, "error": None, "tool_calls": 1, "shape": "messages"}
    turns = [turn(0, 0, 2, 10, 80, 1500, 0, 120000), turn(0, 10, 1, 5, 40, 1000, 0, 125000), turn(1, 4, 4, 10, 60, 2000, 1000, 210000)]
    summary = bench.summarize_arm(turns, 2)
    check(summary["wall_s"] == 15 and summary["output_tokens"] == 180 and summary["output_tok_s"] == 12, f"arm totals are wrong: {summary}")
    check(summary["share"] == {"waiting_for_first_token": 0.28, "prefill": 0.18, "queue": 0.04, "decode": 0.72}, f"shares are wrong: {summary['share']}")
    check(summary["all_sessions_active"]["seconds"] == 10, f"the together window is wrong: {summary['all_sessions_active']}")
    expected = (80 * 6 / 8 + 40 * 3 / 4 + 60) / 10
    check(abs(summary["all_sessions_active"]["output_tok_s"] - round(expected, 2)) < 1e-9, f"the together rate is wrong: {summary['all_sessions_active']}")
    check([bucket["from"] for bucket in summary["by_context"]] == [100000, 200000], f"context buckets are wrong: {summary['by_context']}")
    check([round(bench.start_fraction(index), 4) for index in range(4)] == [0.5, 0.25, 0.75, 0.125], "the first sessions do not spread over the context range")
    check(bench.percentile([1, 2, 3, 4], 0.5) == 2.5 and bench.percentile([], 0.5) is None, "percentile is wrong")


def check_memory_watch(tmp, check):
    meminfo = tmp / "meminfo"
    meminfo.write_text("MemTotal:       125000000 kB\nMemFree:          1000000 kB\nMemAvailable:    9000000 kB\n")
    sampled = subprocess.run([sys.executable, "-", "0.05", "1.2", str(meminfo)], input=watch.SAMPLER.encode(), capture_output=True, timeout=30)
    lines = [watch.parse_sample(line) for line in sampled.stdout.decode().splitlines()]
    check(sampled.returncode == 0 and lines and lines[0]["total"] == 125000000 * 1024 and lines[0]["low"] == 9000000 * 1024,
          f"the sampler did not report MemTotal and the lowest MemAvailable: {sampled.stdout!r} {sampled.stderr!r}")
    fake = tmp / "fake_ssh.py"
    fake.write_text("import sys, time\nhost = sys.argv[1]\nsys.stdin.read()\nlow = {'good': 9, 'thin': 3}[host] * (1 << 30)\n"
                    "for second in range(3):\n    print(time.time(), 100 << 30, low - second, low, flush=True)\n")
    watcher = watch.MemoryWatch(["good", "thin", "gone"], ssh=[sys.executable, str(fake)]).start()
    for process in watcher.processes.values():
        process.wait(timeout=30)
    summary = watcher.stop()
    nodes, verdict = summary["nodes"], summary["verdict"]
    line = int((100 << 30) * 0.02)
    check(nodes["good"]["samples"] == 3 and nodes["good"]["lowest_available_bytes"] == (9 << 30) - 2, f"the good node's samples are wrong: {nodes['good']}")
    check(nodes["good"]["earlyoom_line_bytes"] == line and nodes["good"]["passes"], "a node 6.9 GiB above earlyoom's line did not pass")
    check(not nodes["thin"]["passes"] and nodes["thin"]["lowest_above_line_bytes"] == (3 << 30) - 2 - line,
          "a node less than 1 GiB above earlyoom's line passed")
    check(verdict["missing"] == ["gone"] and verdict["failing"] == ["thin"] and not verdict["passes"], f"the verdict is wrong: {verdict}")
    check("gone" in summary["errors"], "a node whose sampler failed reported no error")


def main():
    check = fixture.Checks("test_agent_turn_bench")
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        check_end_to_end(tmp, check)
        check_retry(check)
        check_summary(check)
        check_memory_watch(tmp, check)
    check.finish("Messages and Responses sessions replay through the real chat layer: seeds warm once and resume from the cache, every turn's "
                 "tool_use or function_call gets its result appended so the next prompt hits the cache to the end of the reply, arms run 1 and 3 "
                 "staggered sessions, a 429 waits Retry-After and counts in the first-token time, the summary's rates, shares, together window and "
                 "context buckets are exact, and the memory watch judges each node against 1 GiB above earlyoom's line")


if __name__ == "__main__":
    main()
