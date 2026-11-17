#!/usr/bin/env python3
import http.client
import importlib.util
import json
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("agent_endpoint_stub", ROOT / "tools" / "agent_endpoint_stub.py")
stub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stub)

PATCH = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
SOLVED = {"instance_id": "calc__calc-1", "problem_statement": "add() subtracts instead of adding. add(2, 3) returns -1.", "patch": PATCH}
OPEN = {"instance_id": "calc__calc-2", "problem_statement": "mul() is missing from calc.py and should be added.", "patch": PATCH}
KEY = "sk-sp-" + "z" * 43


def request(port, method, path, body=None, headers=None, chunked=False):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    sent = dict(headers or {}, **{"Content-Type": "application/json"})
    if chunked:
        connection.putrequest(method, path)
        for name, value in dict(sent, **{"Transfer-Encoding": "chunked"}).items():
            connection.putheader(name, value)
        connection.endheaders()
        for start in range(0, len(data), 7):
            piece = data[start:start + 7]
            connection.send(b"%x\r\n%s\r\n" % (len(piece), piece))
        connection.send(b"0\r\n\r\n")
    else:
        connection.request(method, path, data, sent)
    response = connection.getresponse()
    status, kind, text = response.status, response.getheader("Content-Type"), response.read().decode()
    connection.close()
    return status, kind, text


def events(text):
    found = []
    for frame in text.split("\n\n"):
        lines = [line for line in frame.splitlines() if line]
        if lines:
            name = next(line[7:] for line in lines if line.startswith("event: "))
            found.append((name, json.loads(next(line[6:] for line in lines if line.startswith("data: ")))))
    return found


def main():
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        dataset = tmp / "dataset.jsonl"
        dataset.write_text(json.dumps(SOLVED) + "\n" + json.dumps(OPEN) + "\n")
        log = tmp / "requests.jsonl"
        server = stub.serve("127.0.0.1", 0, "stub-model", 131072, stub.Script(str(dataset), ["calc__calc-1"]), str(log))
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            claude_tools = [{"name": "Read", "input_schema": {}}, {"name": "Bash", "input_schema": {}}]
            first = {"model": "stub-model", "stream": True, "tools": claude_tools, "messages": [{"role": "user", "content": [{"type": "text", "text": SOLVED["problem_statement"] + "\n\nFix it."}]}]}
            status, kind, text = request(port, "POST", "/v1/messages", first, {"Authorization": "Bearer " + KEY, "X-SparkPipe-Public": "1"}, chunked=True)
            frames = events(text)
            check(status == 200 and kind == "text/event-stream", f"a streamed Messages turn answered {status} {kind}")
            check([name for name, _ in frames] == ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "content_block_start",
                                                   "content_block_delta", "content_block_stop", "message_delta", "message_stop"],
                  f"the Messages stream has the wrong events: {[name for name, _ in frames]}")
            tool = frames[4][1]["content_block"]
            command = json.loads(frames[5][1]["delta"]["partial_json"])["command"]
            check(tool["type"] == "tool_use" and tool["name"] == "Bash" and frames[7][1]["delta"]["stop_reason"] == "tool_use", "the first turn is not a Bash tool_use")
            check(command.startswith("cd /testbed && git apply -v <<'SPARKPIPE_PATCH_END'\n") and PATCH in command and command.endswith("\nSPARKPIPE_PATCH_END"),
                  f"a solved task's command does not apply its gold patch: {command!r}")
            second = dict(first, stream=False, messages=first["messages"] + [{"role": "assistant", "content": [{"type": "tool_use", "id": tool["id"], "name": "Bash", "input": {"command": command}}]},
                                                                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool["id"], "content": "Applied"}]}])
            status, kind, text = request(port, "POST", "/v1/messages", second, {"X-Api-Key": KEY})
            message = json.loads(text)
            check(status == 200 and message["stop_reason"] == "end_turn" and message["content"] == [{"type": "text", "text": stub.FINAL_TEXT}],
                  f"the turn after the tool result does not end: {message}")
            open_first = dict(first, stream=False, messages=[{"role": "user", "content": OPEN["problem_statement"]}])
            command = json.loads(request(port, "POST", "/v1/messages", open_first)[2])["content"][1]["input"]["command"]
            check(command == "pwd && ls && git status --short", f"a task the stub does not solve changed files: {command!r}")

            codex_tools = [{"type": "function", "name": "exec_command", "parameters": {}}, {"type": "custom", "name": "apply_patch"}]
            responses = {"model": "stub-model", "stream": True, "tools": codex_tools, "instructions": "You are a coding agent.",
                         "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": SOLVED["problem_statement"]}]}]}
            status, kind, text = request(port, "POST", "/v1/responses", responses, {"Authorization": "Bearer " + KEY})
            frames = events(text)
            names = [name for name, _ in frames]
            check(status == 200 and names[0] == "response.created" and names[-1] == "response.completed" and "response.function_call_arguments.delta" in names,
                  f"the Responses stream has the wrong events: {names}")
            call = frames[-1][1]["response"]["output"][0]
            check(call["type"] == "function_call" and call["name"] == "exec_command" and PATCH in json.loads(call["arguments"])["cmd"],
                  f"Codex's first turn does not run the patch through exec_command: {call}")
            check([data["sequence_number"] for _, data in frames] == list(range(len(frames))), "the Responses sequence numbers are not consecutive")
            followed = dict(responses, stream=False, input=responses["input"] + [call, {"type": "function_call_output", "call_id": call["call_id"], "output": "ok"}])
            done = json.loads(request(port, "POST", "/v1/responses", followed)[2])
            check(done["status"] == "completed" and done["output"][0]["content"][0]["text"] == stub.FINAL_TEXT, f"the Responses turn after the output does not end: {done}")
            shell = dict(responses, stream=False, tools=[{"type": "function", "name": "shell", "parameters": {}}])
            arguments = json.loads(json.loads(request(port, "POST", "/v1/responses", shell)[2])["output"][0]["arguments"])
            check(arguments["command"][:2] == ["bash", "-lc"], f"the shell tool did not get an argv: {arguments}")

            status, _, text = request(port, "GET", "/v1/codex/models")
            catalog = json.loads(text)["models"][0]
            check(status == 200 and catalog["slug"] == "stub-model" and catalog["context_window"] == 131072 and catalog["shell_type"] == "unified_exec",
                  f"the Codex catalogue is wrong: {catalog}")
            check(json.loads(request(port, "GET", "/v1/models")[2])["data"][0]["id"] == "stub-model", "/v1/models lacks the model")
            check(json.loads(request(port, "POST", "/v1/messages/count_tokens", first)[2])["input_tokens"] > 0, "count_tokens did not count")
            check(request(port, "GET", "/api/hello")[0] == 200 and request(port, "GET", "/nothing")[0] == 404, "/api/hello or the 404 is wrong")
        finally:
            server.shutdown()
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        turns = [entry for entry in entries if entry["shape"] in ("messages", "responses")]
        check([entry["step"] for entry in turns[:2]] == [0, 1] and turns[0]["instance"] == "calc__calc-1" and turns[0]["action"] == "Bash"
              and turns[1]["action"] == "final", f"the log does not record steps and actions: {turns[:2]}")
        check(turns[0]["public"] is True and turns[1]["public"] is False, "the log does not record the public header")
        check(turns[0]["key_sha256_12"] == turns[1]["key_sha256_12"] and len(turns[0]["key_sha256_12"]) == 12 and KEY not in log.read_text(),
              "the log does not record a key hash for both header forms, or it holds the key")
    for failure in failures:
        print(f"FAIL test_agent_endpoint_stub: {failure}")
    if failures:
        raise SystemExit(1)
    print("PASS test_agent_endpoint_stub: the stub answers Messages and Responses, streamed and whole, to chunked and sized bodies; a named task's "
          "first turn applies its gold patch through Bash, exec_command or shell, any other task's only looks, and the next turn ends; the Codex "
          "catalogue, models, count_tokens and hello answer; the log records step, action, the public header and a key hash, never the key")


if __name__ == "__main__":
    main()
