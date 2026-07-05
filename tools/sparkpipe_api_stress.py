#!/usr/bin/env python3
"""OpenAI-compatible SparkPipe API stress client.

This is intentionally dependency-free so it can run from a Mac, the public
website host, or spark0 without installing load-test packages.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import queue
import ssl
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://sparkpipe.ai/v1/chat/completions"
DEFAULT_PROMPTS = (
    "Reply with exactly one short sentence about batching.",
    "Give a terse answer: what is queue pressure?",
    "Return three comma-separated words about fast inference.",
    "In one sentence, explain why streaming exposes TTFT.",
    "Answer with a compact JSON object containing status ok.",
    "Say whether concurrent requests should batch together.",
)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.file = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        record.setdefault("time_unix", time.time())
        text = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self.lock:
            self.file.write(text + "\n")
            self.file.flush()

    def close(self) -> None:
        with self.lock:
            self.file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress SparkPipe OpenAI-compatible API and record JSONL progress."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--health-url", default="")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--launch-spacing-ms", type=float, default=0.0)
    parser.add_argument("--health-interval-s", type=float, default=2.0)
    parser.add_argument("--origin", default="https://sparkpipe.ai")
    parser.add_argument("--api-key-env", default="SPARKPIPE_API_KEY")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--record-output-limit", type=int, default=4096)
    parser.add_argument("--full-output", action="store_true")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--success-statuses", default="200,202")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    prompts.extend(args.prompt)
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
        prompts.append(prompt_text)
    if not prompts:
        prompts.extend(DEFAULT_PROMPTS)
    repeat = max(1, args.prompt_repeat)
    expanded: list[str] = []
    for prompt in prompts:
        for _ in range(repeat):
            expanded.append(prompt)
    return expanded


def build_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if args.stream else "application/json",
        "User-Agent": "sparkpipe-api-stress/1",
    }
    if args.origin:
        headers["Origin"] = args.origin
    api_key = os.environ.get(args.api_key_env, "")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def make_connection(parsed: urllib.parse.ParseResult, timeout_s: float) -> http.client.HTTPConnection:
    if parsed.scheme == "https":
        context = ssl.create_default_context()
        return http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=timeout_s, context=context)
    if parsed.scheme == "http":
        return http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
    raise ValueError(f"unsupported URL scheme: {parsed.scheme}")


def request_path(parsed: urllib.parse.ParseResult) -> str:
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return path


def parse_sse_events(
    chunks: list[tuple[bytes, float]],
) -> tuple[int, int, str, list[dict[str, Any]], float | None]:
    buffer = b""
    token_events = 0
    done_events = 0
    text_parts: list[str] = []
    events: list[dict[str, Any]] = []
    first_token_seen = False
    first_token_offset_s: float | None = None
    for chunk, chunk_offset_s in chunks:
        buffer += chunk
        while b"\n\n" in buffer:
            raw_event, buffer = buffer.split(b"\n\n", 1)
            event_name = ""
            data_lines: list[str] = []
            for raw_line in raw_event.splitlines():
                line = raw_line.decode("utf-8", errors="replace")
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            data_text = "\n".join(data_lines)
            parsed_data: dict[str, Any] = {}
            if data_text:
                try:
                    value = json.loads(data_text)
                    if isinstance(value, dict):
                        parsed_data = value
                except json.JSONDecodeError:
                    parsed_data = {"raw_data": data_text}
            if event_name == "token":
                token_events += 1
                if not first_token_seen:
                    first_token_seen = True
                    first_token_offset_s = chunk_offset_s
                text = parsed_data.get("text", "")
                if isinstance(text, str):
                    text_parts.append(text)
            elif event_name == "done":
                done_events += 1
            events.append({"event": event_name, "data": parsed_data})
    return token_events, done_events, "".join(text_parts), events, first_token_offset_s


def run_one(
    args: argparse.Namespace,
    parsed: urllib.parse.ParseResult,
    headers: dict[str, str],
    prompt: str,
    request_index: int,
) -> dict[str, Any]:
    body = {
        "model": args.model,
        "stream": bool(args.stream),
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    encoded_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
    start = time.monotonic()
    first_byte_s: float | None = None
    response_body = b""
    chunks: list[tuple[bytes, float]] = []
    status = 0
    reason = ""
    content_type = ""
    error = ""
    try:
        connection = make_connection(parsed, args.timeout_s)
        connection.request("POST", request_path(parsed), body=encoded_body, headers=headers)
        response = connection.getresponse()
        status = response.status
        reason = response.reason
        content_type = response.getheader("content-type", "")
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            if first_byte_s is None:
                first_byte_s = time.monotonic() - start
            chunks.append((chunk, time.monotonic() - start))
            response_body += chunk
        connection.close()
    except Exception as exc:  # load client must report all failures, not abort the run
        error = f"{type(exc).__name__}: {exc}"
    finish = time.monotonic()
    token_events = 0
    done_events = 0
    output_text = ""
    parsed_events: list[dict[str, Any]] = []
    ttft_s: float | None = None
    if chunks and "text/event-stream" in content_type:
        token_events, done_events, output_text, parsed_events, ttft_s = parse_sse_events(chunks)
        if not output_text and status >= 400:
            output_text = response_body.decode("utf-8", errors="replace")
    else:
        output_text = response_body.decode("utf-8", errors="replace")
    output_limit = -1 if args.full_output else args.record_output_limit
    output_record = output_text
    truncated = False
    if output_limit >= 0 and len(output_record) > output_limit:
        output_record = output_record[:output_limit]
        truncated = True
    return {
        "type": "request_result",
        "request_index": request_index,
        "status": status,
        "reason": reason,
        "content_type": content_type,
        "error": error,
        "prompt_bytes": len(prompt.encode("utf-8")),
        "request_bytes": len(encoded_body),
        "elapsed_s": finish - start,
        "first_byte_s": first_byte_s,
        "ttft_s": ttft_s,
        "token_events": token_events,
        "done_events": done_events,
        "response_bytes": len(response_body),
        "output_chars": len(output_text),
        "output_truncated": truncated,
        "output": output_record,
        "events": parsed_events if args.full_output else [],
    }


def fetch_health(health_url: str, timeout_s: float) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(health_url)
    start = time.monotonic()
    result: dict[str, Any] = {"type": "health_sample", "url": health_url}
    try:
        connection = make_connection(parsed, timeout_s)
        connection.request("GET", request_path(parsed), headers={"User-Agent": "sparkpipe-api-stress/1"})
        response = connection.getresponse()
        body = response.read(1024 * 1024)
        connection.close()
        result["status"] = response.status
        result["elapsed_s"] = time.monotonic() - start
        result["body_bytes"] = len(body)
        text = body.decode("utf-8", errors="replace")
        try:
            result["json"] = json.loads(text)
        except json.JSONDecodeError:
            result["body"] = text[:4096]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_s"] = time.monotonic() - start
    return result


def health_loop(args: argparse.Namespace, writer: JsonlWriter, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        writer.write(fetch_health(args.health_url, args.timeout_s))
        stop_event.wait(max(0.1, args.health_interval_s))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def parse_statuses(text: str) -> set[int]:
    statuses: set[int] = set()
    for value in text.split(","):
        stripped = value.strip()
        if stripped:
            statuses.add(int(stripped))
    return statuses


def summarize(results: list[dict[str, Any]], elapsed_s: float, success_statuses: set[int]) -> dict[str, Any]:
    latencies = [float(r["elapsed_s"]) for r in results if not r.get("error")]
    ttfts = [float(r["ttft_s"]) for r in results if r.get("ttft_s") is not None]
    statuses: dict[str, int] = {}
    total_tokens = 0
    http_failure_count = 0
    for result in results:
        status = int(result.get("status", 0))
        key = str(status)
        statuses[key] = statuses.get(key, 0) + 1
        total_tokens += int(result.get("token_events", 0))
        if status not in success_statuses:
            http_failure_count += 1
    return {
        "type": "summary",
        "request_count": len(results),
        "status_counts": statuses,
        "error_count": sum(1 for r in results if r.get("error")),
        "http_failure_count": http_failure_count,
        "elapsed_s": elapsed_s,
        "requests_per_s": (len(results) / elapsed_s) if elapsed_s > 0.0 else None,
        "token_events": total_tokens,
        "token_events_per_s": (total_tokens / elapsed_s) if elapsed_s > 0.0 else None,
        "latency_avg_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 50.0),
        "latency_p95_s": percentile(latencies, 95.0),
        "latency_p99_s": percentile(latencies, 99.0),
        "ttft_avg_s": statistics.fmean(ttfts) if ttfts else None,
        "ttft_p50_s": percentile(ttfts, 50.0),
        "ttft_p95_s": percentile(ttfts, 95.0),
    }


def main() -> int:
    args = parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        print("requests and concurrency must be positive", file=sys.stderr)
        return 2
    parsed = urllib.parse.urlparse(args.url)
    if parsed.hostname is None:
        print("url must include a hostname", file=sys.stderr)
        return 2
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_jsonl = Path(args.output_jsonl or f"/private/tmp/sparkpipe_api_stress_{timestamp}.jsonl")
    summary_json = Path(args.summary_json or f"/private/tmp/sparkpipe_api_stress_{timestamp}.summary.json")
    prompts = load_prompts(args)
    headers = build_headers(args)
    success_statuses = parse_statuses(args.success_statuses)
    writer = JsonlWriter(output_jsonl)
    stop_event = threading.Event()
    health_thread: threading.Thread | None = None
    start = time.monotonic()
    results: list[dict[str, Any]] = []
    writer.write(
        {
            "type": "run_start",
            "url": args.url,
            "health_url": args.health_url,
            "model": args.model,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_completion_tokens": args.max_completion_tokens,
            "stream": args.stream,
            "success_statuses": sorted(success_statuses),
            "output_jsonl": str(output_jsonl),
            "summary_json": str(summary_json),
        }
    )
    if args.health_url:
        health_thread = threading.Thread(target=health_loop, args=(args, writer, stop_event), daemon=True)
        health_thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures: list[concurrent.futures.Future[dict[str, Any]]] = []
            for request_index in range(args.requests):
                prompt = prompts[request_index % len(prompts)]
                future = executor.submit(run_one, args, parsed, headers, prompt, request_index)
                futures.append(future)
                if args.launch_spacing_ms > 0.0:
                    time.sleep(args.launch_spacing_ms / 1000.0)
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                writer.write(result)
    finally:
        stop_event.set()
        if health_thread is not None:
            health_thread.join(timeout=max(1.0, args.health_interval_s + 1.0))
    elapsed_s = time.monotonic() - start
    summary = summarize(results, elapsed_s, success_statuses)
    writer.write(summary)
    writer.close()
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"jsonl={output_jsonl}")
    print(f"summary={summary_json}")
    if summary["error_count"] == 0 and summary["http_failure_count"] == 0:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
