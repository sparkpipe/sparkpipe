#!/usr/bin/env python3
"""Checks a tools/turn_floor.py run that tests/test_model_api_text.c made against a real sparkpipe_model_api and real rank daemons: two context levels with two tails, a plain and an as-served decode sample each, and one question to the engine about the session. The stub in tests/test_turn_floor.py is written by hand; this is the same tool reading what the API itself sends and logs."""
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
summary = json.loads((folder / "summary.json").read_text())
rows = [json.loads(line) for line in (folder / "parts.jsonl").read_text().splitlines()]
failures = []


def check(condition, what):
    if not condition:
        failures.append(what)


check(summary["requests"] == 10 and summary["unmatched"] == 0 and len(rows) == 10,
      f"every request finds its request_measurements line by the digest of its token ids: {summary['requests']} requests, {summary['unmatched']} unmatched")
check([row["kind"] for row in rows] == ["grow", "turn", "turn", "decode", "decode_served"] * 2 and [row["context"] for row in rows] == [16] * 5 + [32] * 5,
      f"two levels run a grow, two turns and both decode samples: {[(row['kind'], row['context']) for row in rows]}")
check(all(row.get("error") is None and isinstance(row.get("request_id"), int) and row["engine_ttft_ms"] > 0 and row["queue_ms"] >= 0 and row["restore_ms"] == 0 and
          row["prefill_ms"] >= 0 for row in rows), "each request has the engine's own stamps: accepted, first admitted dispatch and first token")
check(rows[0]["cached_tokens"] == 0 and rows[0]["prefill_tokens"] == 16 and all(row["output_tokens"] == (3 if row["kind"].startswith("decode") else 2) for row in rows),
      f"the stream's tokens and cached count are read as the API sends them: {[(row['cached_tokens'], row['output_tokens']) for row in rows]}")
session = [row for row in rows if row["kind"] in ("grow", "turn")]
check([row["cached_tokens"] for row in session] == [0] + [row["prompt_tokens"] - 1 for row in session[:-1]] and
      all(0 < row["prefill_tokens"] < row["prompt_tokens"] for row in session[1:]) and
      all(row["cached_tokens"] == before["prompt_tokens"] - 1 for before, row in zip(rows, rows[1:]) if row["kind"] == "decode") and
      all(row["cached_tokens"] == row["prompt_tokens"] - 1 for row in rows if row["kind"] == "decode_served"),
      "a turn names where each earlier reply began and ended, resumes one token before the previous prompt's end and computes that reply again; "
      f"a prompt the engine has whole computes its last token again: {[(row['kind'], row['prompt_tokens'], row['cached_tokens']) for row in rows]}")
boots = {row.get("cache_boot") for row in rows}
check(len(boots) == 1 and None not in boots and all(row["cache_resumed_tokens"] == row["cached_tokens"] and row["cache_matched_tokens"] >= row["cache_resumed_tokens"] and
                                                     row["cache_restore_ms"] == 0 and row["cache_background_wait_ms"] == 0 for row in rows),
      f"the cache object of each reply is kept: its resume point is the cached count, and one engine process answered all of them: {boots}")
asked = summary["cache_states"]
answer = asked[0]["answer"] if asked else {}
check(len(asked) == 1 and asked[0]["http_status"] == 200 and asked[0]["event"] == "after-the-turns" and answer.get("complete") is True and answer.get("ranks_failed") == 0 and
      answer.get("ranks") == answer.get("ranks_reported") and answer.get("ranks", 0) >= 1 and answer.get("prompt_tokens") == asked[0]["history_tokens"] and
      answer.get("matched_tokens") == asked[0]["history_tokens"] - 1 and answer.get("held_tokens") == answer.get("probed_tokens") == answer.get("matched_tokens") and
      answer.get("resident_tokens") == answer.get("probed_tokens") and answer.get("boot") in boots,
      f"the engine says what it holds of the session, from every rank, and it is the session the turns left: {asked}")
check(isinstance(summary["clock_shared"], bool) and (summary["clock_shared"] or all(row["send_ms"] is None and row["reply_ms"] is None for row in rows)) and
      (not summary["clock_shared"] or all(row["send_ms"] >= 0 and row["reply_ms"] >= 0 for row in rows)),
      f"send and reply times are kept only when the client's clock is the engine's: {summary['clock_shared']}")

for failure in failures:
    print("FAIL turn_floor_run_check: " + failure)
if failures:
    sys.exit(1)
print("ok turn_floor_run_check")
