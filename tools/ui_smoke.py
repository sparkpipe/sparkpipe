#!/usr/bin/env python3
"""Chat, streaming, tool-call and reasoning smokes through an OpenAI-compatible endpoint.

Checks (each one request to /v1/chat/completions of --endpoint for --model,
temperature 0):
  chat          one-sentence answer that names Paris, finish_reason stop, no
                tool calls, no raw template markers in content
  stream        the same question streamed; the assembled content names Paris
  tool          a get_weather tool offered; the reply is a get_weather call
                whose JSON arguments name Paris, finish_reason tool_calls
  tool_result   the call and its result sent back; the answer quotes 18
  reasoning_off --off-effort; content holds 391, no reasoning_content
  reasoning_on  --on-effort; reasoning_content is non-empty, content holds 391
--reasoning none skips the two reasoning checks.

usage:
  ui_smoke.py --endpoint http://127.0.0.1:4000 --model hy4-preview --reasoning both \\
      --out qualification/hy4/ui-smoke-<run-id>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

MARKERS = ("<|", "|>", "<｜", "｜>", "<think", "</think", "<tool_call", "</tool_call", "<arg_")
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "location": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["location"]}}}
CAPITAL = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
WEATHER_ASK = [{"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}]
PRODUCT = [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}]


def post(endpoint: str, body: dict, timeout: int):
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout), None
    except urllib.error.HTTPError as error:
        with error:
            return None, f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:400]}"
    except OSError as error:
        return None, f"{type(error).__name__}: {error}"


def complete(endpoint: str, body: dict, timeout: int) -> dict:
    started = time.monotonic()
    resp, error = post(endpoint, body, timeout)
    if error:
        return {"error": error, "elapsed_s": round(time.monotonic() - started, 2)}
    with resp:
        payload = json.loads(resp.read())
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {"payload": payload, "content": message.get("content") or "",
            "reasoning": message.get("reasoning_content") or "", "tool_calls": message.get("tool_calls") or [],
            "finish": choice.get("finish_reason"), "elapsed_s": round(time.monotonic() - started, 2)}


def stream(endpoint: str, body: dict, timeout: int) -> dict:
    started = time.monotonic()
    resp, error = post(endpoint, dict(body, stream=True), timeout)
    if error:
        return {"error": error, "elapsed_s": round(time.monotonic() - started, 2)}
    content, reasoning, finish, chunks = [], [], None, 0
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunks += 1
            choice = (json.loads(line[5:]).get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            finish = choice.get("finish_reason") or finish
    return {"content": "".join(content), "reasoning": "".join(reasoning), "tool_calls": [], "finish": finish,
            "chunks": chunks, "elapsed_s": round(time.monotonic() - started, 2)}


def leaks(text: str) -> list:
    return [marker for marker in MARKERS if marker in text]


def judge(name: str, reply: dict) -> list:
    if "error" in reply:
        return [reply["error"]]
    problems = []
    content, reasoning, calls, finish = reply["content"], reply["reasoning"], reply["tool_calls"], reply["finish"]
    if leaks(content):
        problems.append(f"raw markers in content: {leaks(content)}")
    if name in ("chat", "stream"):
        if "paris" not in content.lower():
            problems.append("content does not name Paris")
        if finish != "stop":
            problems.append(f"finish_reason {finish!r}")
        if calls:
            problems.append("unexpected tool call")
    elif name == "tool":
        if finish != "tool_calls":
            problems.append(f"finish_reason {finish!r}")
        if not calls:
            problems.append("no tool call parsed")
        else:
            function = calls[0].get("function") or {}
            if function.get("name") != "get_weather":
                problems.append(f"called {function.get('name')!r}")
            try:
                arguments = json.loads(function.get("arguments") or "")
            except json.JSONDecodeError:
                arguments = None
            if not isinstance(arguments, dict) or "paris" not in str(arguments.get("location", "")).lower():
                problems.append(f"arguments {function.get('arguments')!r}")
    elif name == "tool_result":
        if "18" not in content:
            problems.append("answer does not quote 18")
        if finish != "stop":
            problems.append(f"finish_reason {finish!r}")
    elif name == "reasoning_off":
        if "391" not in content:
            problems.append("content does not hold 391")
        if reasoning.strip():
            problems.append("reasoning_content present with reasoning off")
    elif name == "reasoning_on":
        if not reasoning.strip():
            problems.append("no reasoning_content with reasoning on")
        if "391" not in content:
            problems.append("content does not hold 391")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning", required=True, choices=("both", "none"))
    ap.add_argument("--off-effort", default="minimal")
    ap.add_argument("--on-effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--reasoning-max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens}
    checks = [("chat", complete, dict(base, messages=CAPITAL)),
              ("stream", stream, dict(base, messages=CAPITAL)),
              ("tool", complete, dict(base, messages=WEATHER_ASK, tools=[WEATHER]))]
    results = []
    for name, run, body in checks:
        reply = run(args.endpoint, body, args.timeout)
        results.append({"check": name, "request": body, "reply": reply, "problems": judge(name, reply)})
        if name == "tool" and reply.get("tool_calls"):
            call = reply["tool_calls"][0]
            follow = WEATHER_ASK + [{"role": "assistant", "content": reply["content"] or None, "tool_calls": [call]},
                                    {"role": "tool", "tool_call_id": call.get("id"),
                                     "content": json.dumps({"location": "Paris", "temperature_c": 18, "condition": "sunny"})}]
            body = dict(base, messages=follow, tools=[WEATHER])
            reply = complete(args.endpoint, body, args.timeout)
            results.append({"check": "tool_result", "request": body, "reply": reply, "problems": judge("tool_result", reply)})
        elif name == "tool":
            results.append({"check": "tool_result", "request": None, "reply": None, "problems": ["skipped: no tool call to answer"]})
    if args.reasoning == "both":
        for name, effort in (("reasoning_off", args.off_effort), ("reasoning_on", args.on_effort)):
            body = dict(base, messages=PRODUCT, reasoning_effort=effort, max_tokens=args.reasoning_max_tokens)
            reply = complete(args.endpoint, body, args.timeout)
            results.append({"check": name, "request": body, "reply": reply, "problems": judge(name, reply)})
    for result in results:
        reply = result["reply"] or {}
        print(f"{result['check']:13s} {'PASS' if not result['problems'] else 'FAIL'} "
              f"{'; '.join(result['problems'])} | finish={reply.get('finish')!r} {reply.get('elapsed_s')}s "
              f"content={reply.get('content', '')[:90]!r} reasoning={len(reply.get('reasoning', ''))} chars", flush=True)
    passed = sum(1 for result in results if not result["problems"])
    summary = {"format": "sparkpipe-ui-smoke-v1", "generated_at": time.strftime("%FT%T%z"), "endpoint": args.endpoint,
               "model": args.model, "passed": passed, "total": len(results), "results": results}
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=1)
    print(f"UI-SMOKE {args.model}: {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
