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


def decode_window(sequences):
    start = max(rows[0][0] for rows in sequences.values())
    end = min(rows[-1][0] for rows in sequences.values())
    result = {"all_prefixes_ready_seconds": start,
              "first_sequence_last_token_seconds": end,
              "valid": end > start,
              "scope": "after every sequence first token, through earliest sequence last token"}
    if end > start:
        counts = [sum(start < stamp <= end for stamp, _ in rows)
                  for rows in sequences.values()]
        result.update(elapsed_seconds=end-start, token_count=sum(counts),
                      aggregate_tokens_per_second=sum(counts)/(end-start))
    return result


def summarize(events, status):
    tokens, previous, completed, seen = [], {}, set(), set()
    sequences = {}
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
            sequences.setdefault(key, []).append((stamp, token))
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
    result["sequences"] = [dict(request_id=key[0], sequence_id=key[1],
        token_ids=[row[1] for row in rows],
        arrival_seconds=[row[0] for row in rows]) for key, rows in sequences.items()]
    result["all_sequences_decode_window"] = decode_window(sequences)
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


def summarize_api_measurements(records):
    try:
        if not records or len({r['boot_pid'] for r in records}) != 1:
            raise ValueError('measurements must come from one persistent engine process')
        base = min(r['accepted_ns'] for r in records)
        if type(base) is not int or base <= 0:
            raise ValueError('missing engine acceptance timestamp')
        events, identities = [], set()
        for record in records:
            identity = record['request_id']
            if identity in identities or record['status'] != 0 or record['engine_completed'] != 1:
                raise ValueError('duplicate, failed or incomplete request')
            identities.add(identity)
            cached, prompt = record['cached_prompt_tokens'], record['prompt_tokens']
            if type(cached) is not int or type(prompt) is not int or not 0 <= cached <= prompt:
                raise ValueError('invalid cached prompt count')
            previous = record['accepted_ns']
            if type(previous) is not int or previous <= 0:
                raise ValueError('invalid engine acceptance timestamp')
            common = {'request_id':identity,'sequence_id':identity,'status':0}
            events.append(((previous-base)/1e9,dict(common,event='accepted')))
            if not record['tokens']:
                raise ValueError('no generated tokens')
            for index, (token, stamp) in enumerate(record['tokens']):
                if type(stamp) is not int or stamp <= 0 or stamp < previous:
                    raise ValueError('invalid or decreasing token timestamp')
                events.append(((stamp-base)/1e9,dict(common,event='token',token_index=index,token_id=token)))
                previous = stamp
            events.append(((previous-base)/1e9,dict(common,event='completed')))
        result = summarize(sorted(events,key=lambda event:event[0]),0)
        result.update(measurement='common engine token event timestamps; API records emitted after completion',
                      boot_pid=records[0]['boot_pid'],cached_prompt_tokens={str(r['request_id']):r['cached_prompt_tokens'] for r in records},
                      all_requests_have_prefix_hits=all(r['cached_prompt_tokens'] > 0 for r in records))
        by_id = {r['request_id']:r for r in records}
        for sequence in result.get('sequences',[]):
            record = by_id[sequence['request_id']]
            stamps = [stamp for _,stamp in record['tokens']]
            sequence.update(cached_prompt_tokens=record['cached_prompt_tokens'],
                            engine_ttft_seconds=(stamps[0]-record['accepted_ns'])/1e9,
                            decode_tokens_per_second=(len(stamps)-1)*1e9/(stamps[-1]-stamps[0]) if stamps[-1]>stamps[0] else None)
        return result
    except (KeyError, TypeError, ValueError) as error:
        return {'valid':False,'errors':[str(error)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--api-log")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.api_log:
        if command:
            parser.error('provide an API log or a command, not both')
        with open(args.api_log) as source:
            records = [json.loads(line) for line in source if line.startswith('{')]
        result = summarize_api_measurements([r for r in records if r.get('event') == 'request_measurements'])
        print(json.dumps(result,indent=1))
        return 0 if result['valid'] else 1
    if not command or not 0 < args.timeout_seconds <= 900:
        parser.error("provide a command and a finite timeout in (0, 900] seconds")
    result = measure(command, args.timeout_seconds)
    print(json.dumps(result, indent=1))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
