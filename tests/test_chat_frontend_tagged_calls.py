#!/usr/bin/env python3
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)

O, C, S = "<|open|>", "<|close|>", "<|sep|>"


class TaggedModel:
    reasoning = {"start": O + "think" + S, "end": C + "think" + S}
    tool_calls = {"format": "tagged", "start": O + "call", "end": C + "call" + S, "attribute_end": S,
                  "argument_start": O + "argument", "argument_end": C + "argument" + S,
                  "object_start": O + "json", "object_end": C + "json" + S}
    content_markers = [O + "response" + S, C + "response" + S, O + "tools" + S, C + "tools" + S]
    end_markers = [C + "message" + S]


D = "\uff5cDSML\uff5c"


class AttributeModel:
    reasoning = {"start": "<think>", "end": "</think>"}
    tool_calls = {"format": "tagged", "start": "\n<" + D + " invoke", "end": "</" + D + " invoke>", "attribute_end": ">",
                  "argument_start": "<" + D + " parameter", "argument_end": "</" + D + " parameter>",
                  "name_attribute": "name", "key_attribute": "name", "string_attribute": "string"}
    content_markers = ["<" + D + " calls>", "\n</" + D + " calls>"]
    end_markers = []


def run(text, chunk, tools=None, reasoning_open=True, model=None):
    parser = chat_frontend.OutputParser(model or TaggedModel(), tools, reasoning_open, [])
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
    return joined, calls, parser


def main():
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    tools = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "lines": {"type": "integer"}, "overwrite": {"type": "boolean"}}}}}]
    reply = ("I should write the file." + C + "think" + S + O + "response" + S + "Writing it now." + C + "response" + S +
             O + "tools" + S + O + 'call tool="write_file" index="1"' + S +
             O + 'argument key="path" type="string"' + S + 'a "quoted" & raw/path.txt' + C + "argument" + S +
             O + 'argument key="lines" type="number"' + S + "12" + C + "argument" + S +
             O + 'argument key="overwrite" type="boolean"' + S + "true" + C + "argument" + S +
             C + "call" + S +
             O + 'call tool="write_file" index="2"' + S + O + 'json type="object"' + S + '{"path": "b.txt", "lines": 3}' + C + "json" + S + C + "call" + S +
             C + "tools" + S + C + "message" + S + "never shown")
    for chunk in (1, 3, 7, len(reply)):
        joined, calls, parser = run(reply, chunk, tools)
        name = f" (chunks of {chunk})"
        check(joined["reasoning"] == "I should write the file.", "reasoning ends at the think close marker" + name)
        check(joined["content"] == "Writing it now.", "response and tools wrappers never reach the content" + name)
        check(len(calls) == 2 and parser.calls == 2, "both calls are parsed" + name)
        if len(calls) == 2:
            first = json.loads(calls[0]["function"]["arguments"])
            second = json.loads(calls[1]["function"]["arguments"])
            check(calls[0]["function"]["name"] == "write_file" and first == {"path": 'a "quoted" & raw/path.txt', "lines": 12, "overwrite": True},
                  "typed arguments decode: string kept raw, number and boolean as JSON" + name)
            check(second == {"path": "b.txt", "lines": 3}, "a JSON object argument merges into the call" + name)
        check(parser.stopped and parser.ended and "never shown" not in joined["content"], "the message end marker ends the reply and marks it as the model's own end" + name)

    plain = O + "response" + S + "Paris" + C + "response" + S + C + "message" + S
    joined, calls, _ = run(plain, 2, reasoning_open=False)
    check(joined == {"reasoning": "", "content": "Paris"} and not calls, "a reply without thinking keeps only the response text")

    broken = "x" + C + "think" + S + O + "call no attributes" + C + "call" + S
    joined, calls, _ = run(broken, 4)
    check(not calls and "no attributes" in joined["content"], "a call that does not parse is returned as visible text, never dropped")

    def invoke(name, *arguments):
        body = "".join("\n<" + D + ' parameter name="' + key + '" string="' + flag + '">' + value + "</" + D + " parameter>" for key, flag, value in arguments)
        return "\n<" + D + ' invoke name="' + name + '">' + body + "\n</" + D + " invoke>"

    attributed = ("Plan it.</think>Checking both.\n\n<" + D + " calls>" +
                  invoke("write_file", ("path", "true", "x > y.txt"), ("lines", "false", "7")) +
                  invoke("write_file", ("path", "true", "12"), ("overwrite", "false", "false")) +
                  "\n</" + D + " calls>")
    for chunk in (1, 5, len(attributed)):
        joined, calls, _ = run(attributed, chunk, tools, model=AttributeModel())
        name = f" (attribute names, chunks of {chunk})"
        check(joined == {"reasoning": "Plan it.", "content": "Checking both.\n\n"}, "the call block wrapper and call separators never reach the content" + name)
        check(len(calls) == 2, "every call in one block is parsed" + name)
        if len(calls) == 2:
            check(calls[0]["function"]["name"] == "write_file" and json.loads(calls[0]["function"]["arguments"]) == {"path": "x > y.txt", "lines": 7},
                  "the configured name attribute names the tool and the string flag keeps strings raw" + name)
            check(json.loads(calls[1]["function"]["arguments"]) == {"path": "12", "overwrite": False},
                  "a string-flagged value stays a string even when it looks like JSON; an unflagged value decodes as JSON" + name)

    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print("PASS chat frontend tagged calls: reasoning, response and tool wrappers stream in any chunking; typed and JSON arguments decode; the message end stops; unparseable calls stay visible; configured name, key and string-flag attributes parse multi-call blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
