#!/usr/bin/env python3

import argparse
import json
import statistics
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if not arguments.command:
        parser.error("a benchmark command is required")
    started = time.monotonic_ns()
    process = subprocess.Popen(
        arguments.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    events = []
    token_times = []
    token_ids = []
    stderr_lines = []

    def drain_stderr():
        assert process.stderr is not None
        for line in process.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(target=drain_stderr)
    stderr_thread.start()
    assert process.stdout is not None
    for line in process.stdout:
        observed = time.monotonic_ns()
        event = json.loads(line)
        events.append({
            "elapsed_seconds": (observed - started) / 1e9,
            "event": event,
        })
        if event.get("event") == "token":
            token_times.append(observed)
            token_ids.append(event["token_id"])
    status = process.wait()
    stderr_thread.join()
    stderr = "".join(stderr_lines)
    ended = time.monotonic_ns()
    if status != 0 or len(token_times) < 2:
        raise SystemExit(
            f"benchmark failed status={status} tokens={len(token_times)} "
            f"stderr={stderr}")
    intervals = [(right - left) / 1e9
                 for left,right in zip(token_times,token_times[1:])]
    result = {
        "schema_version": 1,
        "command": arguments.command,
        "process_status": status,
        "token_count": len(token_times),
        "token_ids": token_ids,
        "ttft_seconds": (token_times[0] - started) / 1e9,
        "total_seconds": (ended - started) / 1e9,
        "decode_seconds_after_first":
            (token_times[-1] - token_times[0]) / 1e9,
        "decode_tokens_per_second": (len(token_times) - 1) * 1e9 /
            (token_times[-1] - token_times[0]),
        "inter_token_median_seconds": statistics.median(intervals),
        "inter_token_p95_seconds": sorted(intervals)[
            min(len(intervals) - 1,int(len(intervals) * 0.95))],
        "stderr": stderr,
        "events": events,
    }
    with open(arguments.output,"w",encoding="utf-8") as target:
        json.dump(result,target,indent=2)
        target.write("\n")
    print(json.dumps({key: result[key] for key in (
        "token_count","ttft_seconds","decode_seconds_after_first",
        "decode_tokens_per_second","inter_token_median_seconds",
        "inter_token_p95_seconds","total_seconds")}))


if __name__ == "__main__":
    main()
