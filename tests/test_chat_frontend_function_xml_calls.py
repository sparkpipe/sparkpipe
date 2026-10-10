#!/usr/bin/env python3
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)


class FunctionXmlModel:
    reasoning = {"start": "<think>", "end": "</think>"}
    tool_calls = {"format": "function_xml", "start": "<tool_call>", "end": "</tool_call>", "attribute_end": ">",
                  "function_start": "<function=", "function_end": "</function>",
                  "parameter_start": "<parameter=", "parameter_end": "</parameter>"}
    content_markers = []
    end_markers = []


def run(text, chunk, tools=None, reasoning_open=True):
    parser = chat_frontend.OutputParser(FunctionXmlModel(), tools, reasoning_open, [])
    events = []
    for start in range(0, len(text), chunk):
        events.extend(parser.feed(text[start:start + chunk]))
    events.extend(parser.finish())
    joined = {"reasoning": "", "content": ""}
    calls = []
    for kind, value in events:
        if kind == "tool_call":
            calls.append(value)
        else:
            joined[kind] += value
    return joined, calls


def main():
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    tools = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "lines": {"type": "integer"}, "body": {"type": "string"}}}}}]
    reply = ("Plan the write.</think>\n\nWriting it now.\n<tool_call>\n<function=write_file>\n"
             "<parameter=path>\na <b>/path.txt\n</parameter>\n<parameter=lines>\n12\n</parameter>\n"
             "<parameter=body>\nline one\nline two\n</parameter>\n</function>\n</tool_call>")
    for chunk in (1, 4, 9, len(reply)):
        joined, calls = run(reply, chunk, tools)
        name = f" (chunks of {chunk})"
        check(joined["reasoning"] == "Plan the write.", "reasoning ends at the think close marker" + name)
        check(joined["content"].strip() == "Writing it now.", "content before the call is kept" + name)
        check(len(calls) == 1, "one call is parsed" + name)
        if calls:
            arguments = json.loads(calls[0]["function"]["arguments"])
            check(calls[0]["function"]["name"] == "write_file", "the function name is the attribute" + name)
            check(arguments == {"path": "a <b>/path.txt", "lines": 12, "body": "line one\nline two"},
                  "parameters keep inner text, drop one wrapping newline and type non-strings" + name)
    joined, calls = run("<tool_call>\n<function=write_file>\n<parameter=path>\nx\n</function>\n</tool_call>", 5, tools, False)
    check(not calls and "<function=write_file>" in joined["content"], "a malformed call is returned as content, not dropped")
    if failures:
        for failure in failures:
            print("FAIL", failure)
        return 1
    print("PASS chat frontend function_xml calls: name and parameters parse in any chunking; one wrapping newline is dropped; non-string parameters decode; a malformed call stays visible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
