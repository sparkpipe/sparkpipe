#!/usr/bin/env python3
"""Measure completed model_batch streams in arrival order, with bounded runtime."""
import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import tempfile
import threading
import time


def summarize(events, status):
    tokens, previous, completed, seen = [], {}, set(), set()
    errors = []
    for stamp, event in events:
        key = (event.get("request_id", 0), event.get("sequence_id", 0))
        kind = event.get("event")
        if event.get("status", 0) != 0 or kind in ("error", "cancelled"):
            errors.append("request failed or cancelled")
        if kind == "accepted":
            seen.add(key)
        elif kind == "completed":
            completed.add(key)
        elif kind == "token":
            index, token = event.get("token_index"), event.get("token_id")
            if type(index) is not int or type(token) is not int or index < 0 or token < 0:
                errors.append("invalid token")
                continue
            if key in completed or index != previous.get(key, -1) + 1:
                errors.append("duplicate or out-of-order token")
            previous[key] = index
            seen.add(key)
            tokens.append((stamp, token))
    if seen - completed:
        errors.append("missing request completion")
    if not tokens:
        errors.append("no tokens")
    result = {"valid": status == 0 and not errors, "process_status": status,
              "errors": sorted(set(errors)), "token_count": len(tokens),
              "sequence_count": len(previous), "token_order": "arrival"}
    if not result["valid"]:
        return result
    stamps, ids = zip(*tokens)
    duration = stamps[-1] - stamps[0]
    result.update(ttft_seconds=stamps[0], total_seconds=stamps[-1],
                  decode_seconds_after_first=duration, timed_intervals=len(tokens) - 1,
                  decode_tokens_per_second=(len(tokens) - 1) / duration if duration > 0 else None,
                  token_ids=ids, token_csv_newline_sha256=hashlib.sha256(
                      (",".join(map(str, ids)) + "\n").encode()).hexdigest(),
                  measurement="stdout arrival from first to last token")
    if len(previous) == 1 and len(tokens) > 1:
        intervals = [b - a for a, b in zip(stamps, stamps[1:])]
        result["inter_token_median_seconds"] = statistics.median(intervals)
        result["inter_token_p95_seconds"] = sorted(intervals)[int(.95 * len(intervals))]
    return result


def measure(command, timeout):
    events, failure = [], []
    start = time.monotonic()
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr,
                                   start_new_session=True)
        def stop():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        def expired():
            failure.append("benchmark deadline exceeded")
            stop()
        timer = threading.Timer(timeout, expired)
        timer.start()
        try:
            while True:
                line = process.stdout.readline(1048577)
                if not line:
                    break
                if len(line) > 1048576 or not line.endswith(b"\n"):
                    raise ValueError("oversized or incomplete output line")
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(event, dict):
                    events.append((time.monotonic() - start, event))
                if len(events) > 1000000:
                    raise ValueError("event limit exceeded")
            process.wait()
        except ValueError as error:
            failure.append(str(error))
            stop()
            process.wait(timeout=3)
        finally:
            timer.cancel()
            timer.join()
            process.stdout.close()
        result = summarize(events, process.returncode) if not failure else {
            "valid": False, "process_status": process.returncode, "errors": failure}
        stderr.seek(0, os.SEEK_END)
        stderr.seek(max(0, stderr.tell() - 500))
        result.update(command=command, stderr_tail=stderr.read().decode(errors="replace"))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.timeout_seconds <= 900:
        parser.error("provide a command and a finite timeout in (0, 900] seconds")
    result = measure(command, args.timeout_seconds)
    print(json.dumps(result, indent=1))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
