#!/usr/bin/env python3
import asyncio
import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
chat_frontend = fixture.chat_frontend
TOOLS = [{"type": "function", "name": "shell", "description": "Run a command", "strict": False,
          "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}}, "required": ["command"]}},
         {"type": "web_search"},
         {"type": "custom", "name": "apply_patch", "description": "Apply a patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}}]
INSTRUCTIONS = "You are Codex, a coding agent."
CALL_REPLY = "List the files first.</think><tool_call>shell<arg_key>command</arg_key><arg_value>[\"ls\"]</arg_value></tool_call>"
ANSWER = "Two files.</think>There are two files: a.txt and b.txt."


def request(input_items, **extra):
    body = {"model": "alpha", "instructions": INSTRUCTIONS, "input": input_items, "tools": TOOLS, "tool_choice": "auto", "parallel_tool_calls": False,
            "reasoning": {"effort": "high", "summary": "auto"}, "store": True, "stream": True, "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "session-7", "text": {"verbosity": "low"}, "service_tier": "auto", "unknown_field": [1, 2]}
    body.update(extra)
    return body


def user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


async def run(tmp, check):
    async with fixture.Harness(tmp) as harness:
        engine = harness.engine
        engine.reply(CALL_REPLY)
        status, headers, text = await harness.post("/v1/responses", request([user("What files are here?")]))
        events = fixture.sse_events(text)
        names = [name for name, _ in events]
        check(status == 200 and headers.get("Content-Type", "").startswith("text/event-stream"), f"a streamed Responses request answers an SSE stream: {status} {text[:200]}")
        expected = ["response.created", "response.in_progress", "response.output_item.added", "response.reasoning_summary_part.added"]
        check(names[:4] == expected, f"the stream opens with created, in_progress and the reasoning item: {names}")
        check("response.reasoning_summary_text.delta" in names and names.index("response.reasoning_summary_text.done") < names.index("response.reasoning_summary_part.done") <
              names.index("response.output_item.done"), f"reasoning summary deltas, then done events, then the item is done: {names}")
        tail = names[names.index("response.output_item.done") + 1:]
        check(tail == ["response.output_item.added", "response.function_call_arguments.delta", "response.function_call_arguments.done",
                       "response.output_item.done", "response.completed"], f"the function call streams its arguments, then response.completed ends the stream: {tail}")
        sequence = [data["sequence_number"] for _, data in events]
        check(sequence == list(range(len(events))), f"every event carries an increasing sequence_number: {sequence}")
        completed = events[-1][1]["response"]
        reasoning, call = completed["output"]
        check(reasoning["type"] == "reasoning" and reasoning["summary"][0]["text"] == "List the files first." and reasoning["encrypted_content"].startswith("sp1."),
              f"the reasoning item carries the reasoning and an opaque encrypted_content handle: {reasoning}")
        check(call["type"] == "function_call" and call["name"] == "shell" and json.loads(call["arguments"]) == {"command": ["ls"]} and call["call_id"].startswith("call_"),
              f"the function call item: {call}")
        usage = completed["usage"]
        prompt = len(engine.requests[0]["prompt_token_ids"])
        check(usage["input_tokens"] == prompt and "cached_tokens" in usage["input_tokens_details"] and usage["output_tokens"] > 0 and
              usage["output_tokens_details"]["reasoning_tokens"] > 0, f"usage reports input, cached, output and reasoning tokens: {usage}")
        record = completed["sparkpipe"]
        check(record["model"] == "alpha" and record["prompt_tokens"] == prompt and record["session"] == "session-7",
              f"response.completed carries the sparkpipe object, with Codex's prompt_cache_key as the session: {record}")
        first = harness.prompt_text(0)
        check(first.startswith("<|im_start|>system\n# Tools\n") and '"name": "shell"' in first and '"name": "apply_patch"' in first and "web_search" not in first and
              "start: /.+/" in first, "function and custom tools reach the template; tools the fleet cannot run are left out")
        check(f"<|im_start|>system\n{INSTRUCTIONS}<|im_end|>\n<|im_start|>user\nWhat files are here?<|im_end|>\n<|im_start|>assistant\n<think>" in first,
              f"instructions become the system turn: {first[-200:]}")
        check(engine.requests[0]["priority"] == chat_frontend.INTERACTIVE_PRIORITY, "a Codex request runs at interactive priority")

        engine.reply(ANSWER)
        output = [{"type": "function_call_output", "call_id": call["call_id"], "output": "a.txt\nb.txt"}]
        full = [user("What files are here?"), reasoning, call] + output
        status, _, text = await harness.post("/v1/responses", request(full, stream=False))
        reply = json.loads(text)
        check(status == 200 and engine.requests[1]["prompt_token_ids"][:len(engine.committed[0])] == engine.committed[0],
              "a function call round trip with the reasoning item sent back replays the generated tokens to the end of the previous reply")
        check(reply["usage"]["input_tokens_details"]["cached_tokens"] == len(engine.committed[0]) and reply["sparkpipe"]["replayed_tokens"] == len(engine.committed[0]),
              f"the cache serves the whole previous turn: {reply['usage']} {reply['sparkpipe']}")
        check(reply["status"] == "completed" and [item["type"] for item in reply["output"]] == ["reasoning", "message"] and
              reply["output"][1]["content"][0]["text"] == "There are two files: a.txt and b.txt.", f"a non-streamed reply lists its output items: {reply}")
        check(harness.prompt_text(1).endswith("<|im_end|>\n<|im_start|>tool\na.txt\nb.txt<|im_end|>\n<|im_start|>assistant\n<think>"),
              f"the function_call_output renders as a tool turn after the replayed reply: {harness.prompt_text(1)[-120:]}")
        stripped = [user("What files are here?"), call] + output
        engine.reply(ANSWER)
        await harness.post("/v1/responses", request(stripped, stream=False))
        check(engine.requests[2]["prompt_token_ids"][:len(engine.committed[0])] == engine.committed[0],
              "without the reasoning item the call_id still identifies the turn")

        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/responses", request(output, previous_response_id=completed["id"], stream=False))
        check(status == 200 and engine.requests[3]["prompt_token_ids"] == engine.requests[1]["prompt_token_ids"],
              f"previous_response_id rebuilds the conversation from the stored response: {status} {text[:200]}")
        follow = json.loads(text)
        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/responses", request([user("And the sizes?")], previous_response_id=follow["id"], stream=False))
        check(status == 200 and engine.requests[4]["prompt_token_ids"][:len(engine.committed[3])] == engine.committed[3],
              "a chain of previous_response_id requests keeps replaying each reply")

        engine.reply(CALL_REPLY)
        status, _, text = await harness.post("/v1/responses", request([user("Again, unstored.")], store=False, stream=False))
        unstored = json.loads(text)
        status, _, text = await harness.post("/v1/responses", request(output, previous_response_id=unstored["id"], stream=False))
        check(status == 400 and json.loads(text)["error"]["code"] == "previous_response_not_found", f"store false keeps no response to refer to: {status} {text}")
        engine.reply(ANSWER)
        again = [user("Again, unstored.")] + unstored["output"] + [{"type": "function_call_output", "call_id": unstored["output"][1]["call_id"], "output": "a.txt"}]
        await harness.post("/v1/responses", request(again, store=False, stream=False))
        check(engine.requests[-1]["prompt_token_ids"][:len(engine.committed[5])] == engine.committed[5],
              "store false still keeps the generated tokens for replay")

        engine.reply(ANSWER)
        developer = [{"role": "developer", "content": "Prefer short answers."}, user("Hi")] + [reasoning, call] + output + \
                    [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "The sandbox is now read-only."}]}, user("Go on")]
        status, _, text = await harness.post("/v1/responses", request(developer, stream=False))
        check(status == 200 and "<|im_start|>tool\na.txt\nb.txt<|im_end|>\n<|im_start|>system\nThe sandbox is now read-only.<|im_end|>\n<|im_start|>user\nGo on<|im_end|>" in harness.prompt_text(),
              f"a mid-conversation developer message renders in place as a system turn: {harness.prompt_text()[-240:]}")

        engine.reply("Patch it.</think><tool_call>apply_patch<arg_key>input</arg_key><arg_value>*** Begin Patch</arg_value></tool_call>")
        status, _, text = await harness.post("/v1/responses", request([user("Patch a.txt")], stream=False))
        items = json.loads(text)["output"]
        check(items[-1]["type"] == "custom_tool_call" and items[-1]["name"] == "apply_patch" and items[-1]["input"] == "*** Begin Patch",
              f"a call of a custom tool comes back as a custom_tool_call with its input: {items}")

        status, _, text = await harness.post("/v1/responses", request([user("hi")], model="beta", stream=False))
        error = json.loads(text)["error"]
        check(status == 409 and error["code"] == "model_not_loaded" and error["resident"] == ["alpha"], f"409 model_not_loaded in the OpenAI error body: {status} {text}")
        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/responses", request([user("hi")], model="gpt-5-codex", stream=False))
        check(status == 200 and json.loads(text)["model"] == "gpt-5-codex" and json.loads(text)["sparkpipe"]["model"] == "alpha",
              f"a gpt-* name goes to the first resident model: {text[:200]}")
        status, _, text = await harness.post("/v1/responses", request([user("hi")], text={"format": {"type": "json_schema", "schema": {}}}, stream=False))
        check(status == 400, "text.format json_schema is refused until grammar-constrained sampling lands")
        status, _, text = await harness.post("/v1/responses", request([{"type": "local_shell_call", "call_id": "x"}], stream=False))
        check(status == 400 and "local_shell_call" in json.loads(text)["error"]["message"], "an input item the fleet cannot render is refused by name")

        chat_frontend.KEEPALIVE_SECONDS = 0.1
        engine.delay = 0.45
        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/responses", request([user("slow")]))
        engine.delay = 0.0
        events = fixture.sse_events(text)
        check(status == 200 and [name for name, _ in events][:4] == ["response.created", "response.in_progress", "response.in_progress", "response.in_progress"] and
              [data["sequence_number"] for _, data in events] == list(range(len(events))) and events[-1][0] == "response.completed",
              f"a long prefill repeats response.in_progress, an event Codex counts against its idle timeout, until the first token: {[name for name, _ in events]}")

        engine.reply(ANSWER)
        await harness.post("/v1/responses", request([user("later")], stream=False), **{"X-SparkPipe-Priority": "background"})
        check(engine.requests[-1]["priority"] == chat_frontend.BULK_PRIORITY, "X-SparkPipe-Priority background runs the request at background priority")

        interleaved = [user("Look around"),
                       {"type": "reasoning", "id": "rs_x1", "summary": [{"type": "summary_text", "text": "First I say what I do."}]},
                       {"type": "message", "role": "assistant", "id": "msg_x1", "content": [{"type": "output_text", "text": "Listing the files."}]},
                       {"type": "reasoning", "id": "rs_x2", "summary": [{"type": "summary_text", "text": "Now the call."}]},
                       {"type": "function_call", "id": "fc_x1", "call_id": "call_x1", "name": "shell", "arguments": "{\"command\": [\"ls\"]}"},
                       {"type": "function_call_output", "call_id": "call_x1", "output": "a.txt"}]
        engine.reply(ANSWER)
        status, _, text = await harness.post("/v1/responses", request(interleaved, stream=False))
        prompt = harness.prompt_text()
        turn = prompt[prompt.index("<|im_start|>user\nLook around"):]
        check(status == 200 and turn.count("<|im_start|>assistant\n") == 2 and turn.count("<tool_call>shell") == 1 and "Listing the files." in turn,
              f"a turn of reasoning, message, reasoning and function call keeps every item as one assistant turn with its call once: {turn[:400]}")
        translated = chat_frontend.chat_shapes.responses_request(request(interleaved))["input_messages"]
        check([message["role"] for message in translated] == ["user", "assistant", "tool"] and translated[1]["tool_calls"][0]["id"] == "call_x1" and
              set(translated[1][chat_frontend.chat_shapes.HANDLES]) == {"rs_x1", "msg_x1", "rs_x2", "fc_x1", "call_x1"},
              f"every item of the turn is kept, with its handles: {translated}")
    async with fixture.Harness(tmp) as harness:
        for subagent, wanted in (("compact", chat_frontend.BULK_PRIORITY), ("review", chat_frontend.INTERACTIVE_PRIORITY), (None, chat_frontend.INTERACTIVE_PRIORITY)):
            harness.engine.reply(ANSWER)
            status, _, text = await harness.post("/v1/responses", request([user("Summarize the session.")], stream=False), **({"x-openai-subagent": subagent} if subagent else {}))
            check(status == 200 and harness.engine.requests[-1]["priority"] == wanted and json.loads(text)["sparkpipe"]["session"] == "session-7",
                  f"a Codex request with x-openai-subagent {subagent} runs at priority {wanted} and keeps its prompt_cache_key session: {harness.engine.requests[-1]['priority']}")


def main():
    check = fixture.Checks("test_chat_frontend_responses")
    captured = io.StringIO()
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(captured):
        asyncio.run(run(Path(directory), check))
    if check.failures:
        print(captured.getvalue()[-4000:])
    check.finish("POST /v1/responses streams created, in_progress, reasoning summary, function call argument and output text events with sequence numbers and "
                 "ends with response.completed carrying usage and the sparkpipe object; a function call round trip, with or without the reasoning item, and a "
                 "previous_response_id chain replay the generated tokens so the next prompt matches the cache to the end of the previous reply; store false keeps "
                 "replay but no response; instructions and developer messages render as system turns in place; custom tools round trip; unknown fields and tools "
                 "are ignored; 409 model_not_loaded and gpt-* aliases; repeated response.in_progress events keep a long prefill open; a turn of reasoning, message, reasoning and "
                 "function call keeps every item as one assistant turn")


if __name__ == "__main__":
    main()
