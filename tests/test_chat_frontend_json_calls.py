#!/usr/bin/env python3
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)


class JsonModel:
    reasoning = {"start": "<think>", "end": "</think>"}
    tool_calls = {"format": "json", "start": "<tool_call>", "end": "</tool_call>"}
    content_markers = []
    end_markers = []


def run(text, chunk, tools=None, reasoning_open=True):
    parser = chat_frontend.OutputParser(JsonModel(), tools, reasoning_open, [])
    events = []
    for start in range(0, len(text), chunk):
        events.extend(parser.feed(text[start:start + chunk]))
    events.extend(parser.finish())
    joined = {"reasoning": "", "content": "", "bad_call": []}
    calls = []
    for kind, value in events:
        if kind == "tool_call":
            calls.append(value)
        elif kind == "bad_call":
            joined["bad_call"].append(value)
        else:
            joined[kind] += value
    return joined, calls


def main():
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}}}}}]
    reply = ("날씨를 확인합니다.</think>\n\n서울 날씨를 찾아볼게요.\n<tool_call>\n"
             "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"서울 <중구>\", \"days\": 3}}\n</tool_call>")
    for chunk in (1, 4, 9, len(reply)):
        joined, calls = run(reply, chunk, tools)
        name = f" (chunks of {chunk})"
        check(joined["reasoning"] == "날씨를 확인합니다.", "reasoning ends at the think close marker" + name)
        check(joined["content"].strip() == "서울 날씨를 찾아볼게요.", "content before the call is kept" + name)
        check(len(calls) == 1, "one call is parsed" + name)
        if calls:
            check(calls[0]["function"]["name"] == "get_weather", "the name comes from the object" + name)
            check(json.loads(calls[0]["function"]["arguments"]) == {"city": "서울 <중구>", "days": 3}, "the arguments object is kept as sent" + name)
    joined, calls = run("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \"{\\\"city\\\": \\\"부산\\\"}\"}\n</tool_call>", 3, tools, False)
    check(len(calls) == 1 and calls and json.loads(calls[0]["function"]["arguments"]) == {"city": "부산"}, "arguments sent as a JSON string decode")
    joined, calls = run("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \n</tool_call>", 5, tools, False)
    check(not calls and len(joined["bad_call"]) == 1 and joined["bad_call"][0]["kind"] == "unparsed" and "get_weather" not in joined["content"],
          "a malformed call is reported as a bad call, never returned as content")
    joined, calls = run("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}, \"extra\": 1}\n</tool_call>", 7, tools, False)
    check(not calls and len(joined["bad_call"]) == 1, "an object with fields other than name and arguments is refused")
    if failures:
        for failure in failures:
            print("FAIL", failure)
        return 1
    print("PASS chat frontend json calls: the name and arguments object parse in any chunking, string arguments decode, a malformed or unexpected object is reported, never shown as content")
    return 0


if __name__ == "__main__":
    sys.exit(main())
