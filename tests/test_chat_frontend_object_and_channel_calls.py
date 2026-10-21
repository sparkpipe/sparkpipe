#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)

Q = '<|"|>'


class ObjectModel:
    reasoning = {"start": "<|channel>thought\n", "end": "<channel|>"}
    tool_calls = {"format": "object", "start": "<|tool_call>", "end": "<tool_call|>", "name_prefix": "call:", "string_delimiter": Q}
    content_markers = []
    end_markers = []
    channels = None


class ChannelModel:
    tool_calls = {"format": "tagged", "start": "<atem:invoke", "end": "</atem:invoke>", "attribute_end": ">",
                  "argument_start": "<atem:parameter", "argument_end": "</atem:parameter>", "name_attribute": "name", "key_attribute": "name"}
    content_markers = ["<atem:function_calls>", "</atem:function_calls>"]
    end_markers = []
    channels = {"message_start": "<|start|>assistant", "header_end": "<|message|>", "message_end": "<|eom|>",
                "recipient_prefix": " to=", "reasoning_recipient": "self", "content_recipient": "user"}


def collect(parser, text, chunk):
    events = []
    for start in range(0, len(text), chunk):
        events.extend(parser.feed(text[start:start + chunk]))
    events.extend(parser.finish())
    joined = {"reasoning": "", "content": ""}
    calls = []
    collect.bad = []
    for kind, value in events:
        if kind == "tool_call":
            calls.append(value)
        elif kind == "bad_call":
            collect.bad.append(value["text"])
        else:
            joined[kind] += value
    return joined, calls


def main():
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    weather = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}]
    nested = ("<|tool_call>call:plan{count:3,flag:true,missing:null,ratio:-1.5e-3,tags:[" + Q + "a" + Q + ",2,[false]]," +
              "where:{city:" + Q + "Paris, {France}" + Q + ",zip:" + Q + "75001" + Q + "}," + Q + "odd key" + Q + ":" + Q + Q + "}<tool_call|>")
    gemma = "<|channel>thought\nCompare both.<channel|><|tool_call>call:get_weather{location:" + Q + "Paris" + Q + "}<tool_call|>" + nested
    for chunk in (1, 4, len(gemma)):
        name = f" (chunks of {chunk})"
        joined, calls = collect(chat_frontend.OutputParser(ObjectModel(), weather, False, []), gemma, chunk)
        check(joined == {"reasoning": "Compare both.", "content": ""}, "a thought block the model opens itself becomes reasoning_content" + name)
        check(len(calls) == 2, "consecutive object calls are both parsed" + name)
        if len(calls) == 2:
            check(calls[0]["function"]["name"] == "get_weather" and json.loads(calls[0]["function"]["arguments"]) == {"location": "Paris"},
                  "a delimited string argument decodes raw" + name)
            check(json.loads(calls[1]["function"]["arguments"]) == {"count": 3, "flag": True, "missing": None, "ratio": -0.0015,
                                                                   "tags": ["a", 2, [False]], "where": {"city": "Paris, {France}", "zip": "75001"}, "odd key": ""},
                  "numbers, booleans, null, arrays, nested objects and delimited keys decode as the publisher writes them" + name)

    joined, calls = collect(chat_frontend.OutputParser(ObjectModel(), None, False, []), "\n391", 1)
    check(joined == {"reasoning": "", "content": "\n391"} and not calls, "a reply without a thought block keeps its leading whitespace as content")
    joined, _ = collect(chat_frontend.OutputParser(ObjectModel(), None, False, []), "391 <|channel>thought\nlate<channel|>", 3)
    check(joined["reasoning"] == "" and joined["content"].startswith("391 <|channel>"), "a thought marker after the answer started is not reasoning")
    joined, _ = collect(chat_frontend.OutputParser(ObjectModel(), None, True, []), "Done.<channel|>391", 2)
    check(joined == {"reasoning": "Done.", "content": "391"}, "a thought block opened by the prompt still ends at the close marker")
    for broken in ("call:get_weather{location:Paris}", "call:get-weather{location:" + Q + "Paris" + Q + "}", "call:f{a:" + Q + "x}", "call:f{a:1} tail"):
        joined, calls = collect(chat_frontend.OutputParser(ObjectModel(), None, False, []), "<|tool_call>" + broken + "<tool_call|>", 5)
        check(not calls and broken not in joined["content"] and any(broken in text for text in collect.bad),
              "an object call outside the published grammar is reported as a bad call, never shown as content: " + broken)

    muse_tool = (" to=self<|message|>Call it.<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n"
                 "<atem:parameter name=\"location\">Paris</atem:parameter>\n<atem:parameter name=\"days\">3</atem:parameter>\n</atem:invoke>\n</atem:function_calls>")
    muse_chat = " to=self<|message|>One sentence.<|eom|><|start|>assistant to=user<|message|>The capital of France is Paris."
    for chunk in (1, 6, len(muse_tool)):
        name = f" (chunks of {chunk})"
        joined, calls = collect(chat_frontend.ChannelParser(ChannelModel(), weather, "<|start|>assistant", []), muse_tool, chunk)
        check(joined == {"reasoning": "Call it.", "content": ""}, "the self channel is reasoning and protocol wrappers never leak" + name)
        check(len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" and json.loads(calls[0]["function"]["arguments"]) == {"location": "Paris", "days": 3},
              "a tool channel yields a call with schema-typed arguments" + name)
        joined, calls = collect(chat_frontend.ChannelParser(ChannelModel(), weather, "<|start|>assistant", []), muse_chat, chunk)
        check(joined == {"reasoning": "One sentence.", "content": "The capital of France is Paris."} and not calls, "the user channel is content" + name)
    joined, _ = collect(chat_frontend.ChannelParser(ChannelModel(), None, "<|start|>assistant", ["Paris"]), muse_chat, 3)
    check(joined["content"] == "The capital of France is " and joined["reasoning"] == "One sentence.", "a request stop string ends the user channel")
    joined, _ = collect(chat_frontend.ChannelParser(ChannelModel(), None, "x<|eot|>", []), "stray<|start|>assistant to=user<|message|>ok", 4)
    check(joined["content"] == "strayok", "text outside any message is shown, not dropped")

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        (base / "chat_template.jinja").write_text("{% for tool in tools %}{{ tool.function.description | tojson }}{% endfor %}")
        (base / "tokenizer_config.json").write_text("{}")
        renderer = chat_frontend.TemplateRenderer(base, {}, None)
        try:
            renderer.render([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "f"}}], {})
            check(False, "a template that cannot render a request raises")
        except chat_frontend.RequestError as error:
            check(error.status == 400 and "TypeError" in error.message, "a template TypeError becomes a 400 that names the error")

    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print("PASS chat frontend object and channel calls: publisher object calls with delimited strings, nested values and unquoted keys parse in any chunking; a thought block opened by the model is reasoning; channel replies split into reasoning, content and tool calls; template errors are 400s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
