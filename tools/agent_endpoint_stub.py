#!/usr/bin/env python3
import argparse
import hashlib
import http.server
import json
import threading
import time
import uuid
from pathlib import Path

PATCH_END = "SPARKPIPE_PATCH_END"
PROBLEM_PREFIX = 200
FINAL_TEXT = "The change is complete."
SHELL_TOOLS = (("Bash", "command", "text"), ("exec_command", "cmd", "text"), ("shell", "command", "argv"))
CALL_OUTPUTS = frozenset({"function_call_output", "custom_tool_call_output"})


def handle(prefix):
    return prefix + uuid.uuid4().hex[:24]


def sse(event, data):
    return ("event: " + event + "\ndata: " + json.dumps(data) + "\n\n").encode()


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def presented_key(headers):
    authorization = headers.get("Authorization", "")
    key = authorization[len("Bearer "):].strip() if authorization.startswith("Bearer ") else headers.get("X-Api-Key", "").strip()
    return hashlib.sha256(key.encode()).hexdigest()[:12] if key else None


class Script:
    def __init__(self, dataset, solve):
        self.problems = {}
        if dataset:
            for line in Path(dataset).read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.problems[row["problem_statement"].strip()[:PROBLEM_PREFIX]] = row
        self.solve = set(solve)

    def instance(self, texts):
        joined = "\n".join(texts)
        for prefix, row in self.problems.items():
            if prefix and prefix in joined:
                return row
        return None

    def command(self, row):
        if row is not None and row["instance_id"] in self.solve:
            patch = row["patch"] if row["patch"].endswith("\n") else row["patch"] + "\n"
            if any(line == PATCH_END for line in patch.splitlines()):
                raise ValueError(f"{row['instance_id']}'s patch holds the heredoc delimiter")
            return f"cd /testbed && git apply -v <<'{PATCH_END}'\n{patch}{PATCH_END}"
        return "pwd && ls && git status --short"


def shell_tool(tools):
    names = {}
    for tool in tools or []:
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        if name:
            names[name] = tool
    for name, field, kind in SHELL_TOOLS:
        if name in names:
            return name, field, kind
    return None


def arguments_for(tool, command):
    name, field, kind = tool
    return {field: ["bash", "-lc", command] if kind == "argv" else command}


class Stub(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "sparkpipe-agent-stub"

    def log_message(self, *args):
        pass

    def record(self, entry):
        entry.update(t=round(time.time(), 3), path=self.path.split("?")[0], public=self.headers.get("X-SparkPipe-Public") == "1",
                     key_sha256_12=presented_key(self.headers))
        with self.server.lock:
            with open(self.server.log, "a") as handle:
                handle.write(json.dumps(entry) + "\n")

    def reply_json(self, status, document):
        body = json.dumps(document).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self, frames):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for frame in frames:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(frame), frame))
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def read_body(self):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            parts = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return b"".join(parts)
                parts.append(self.rfile.read(size))
                self.rfile.readline()
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self):
        path = self.path.split("?")[0]
        model = self.server.model
        if path == "/api/hello":
            self.record({"shape": "hello"})
            return self.reply_json(200, {"ok": True})
        if path == "/v1/models":
            self.record({"shape": "models"})
            return self.reply_json(200, {"object": "list", "data": [{"id": model, "object": "model", "context_window": self.server.context,
                                                                    "max_output_tokens": 32768, "sequences": 8}]})
        if path == "/v1/codex/models":
            self.record({"shape": "codex_models"})
            return self.reply_json(200, {"models": [{
                "slug": model, "display_name": model, "description": "stub", "default_reasoning_level": "medium",
                "supported_reasoning_levels": [{"effort": "medium", "description": "Balanced speed and reasoning depth"}],
                "shell_type": "unified_exec", "visibility": "list", "supported_in_api": True, "priority": 0, "support_verbosity": False,
                "truncation_policy": {"mode": "tokens", "limit": 10000}, "experimental_supported_tools": [], "apply_patch_tool_type": "freeform",
                "context_window": self.server.context, "max_context_window": self.server.context, "input_modalities": ["text"],
                "base_instructions": "You are a coding agent running in a terminal."}]})
        self.reply_json(404, {"error": {"type": "not_found", "message": f"{path} is not served by the stub"}})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            body = json.loads(self.read_body() or b"{}")
        except json.JSONDecodeError:
            return self.reply_json(400, {"error": {"type": "invalid_request_error", "message": "the body is not JSON"}})
        if path == "/v1/messages/count_tokens":
            self.record({"shape": "count_tokens"})
            return self.reply_json(200, {"input_tokens": max(1, len(json.dumps(body)) // 4)})
        if path == "/v1/messages":
            return self.messages_api(body)
        if path == "/v1/responses":
            return self.responses_api(body)
        self.reply_json(404, {"error": {"type": "not_found", "message": f"{path} is not served by the stub"}})

    def plan(self, texts, tools, step):
        tool = shell_tool(tools)
        row = self.server.script.instance(texts)
        if tool is None or step > 0:
            return None, None, row
        return tool, arguments_for(tool, self.server.script.command(row)), row

    def messages_api(self, body):
        messages = body.get("messages") or []
        texts = [text_of(body.get("system"))] + [text_of(message.get("content")) for message in messages if message.get("role") == "user"]
        step = sum(message.get("role") == "assistant" for message in messages)
        tool, arguments, row = self.plan(texts, body.get("tools"), step)
        self.record({"shape": "messages", "model": body.get("model"), "stream": bool(body.get("stream")), "step": step,
                     "instance": row["instance_id"] if row else None, "action": tool[0] if tool else "final",
                     "tools": [item.get("name") for item in body.get("tools") or []]})
        identifier, usage = handle("msg_"), {"input_tokens": max(1, len(json.dumps(body)) // 4), "output_tokens": 12}
        blocks = [{"type": "text", "text": "Running the next step." if tool else FINAL_TEXT}]
        if tool:
            blocks.append({"type": "tool_use", "id": handle("toolu_"), "name": tool[0], "input": arguments})
        stop = "tool_use" if tool else "end_turn"
        if not body.get("stream"):
            return self.reply_json(200, {"id": identifier, "type": "message", "role": "assistant", "model": body.get("model"), "content": blocks,
                                         "stop_reason": stop, "stop_sequence": None, "usage": usage})
        frames = [sse("message_start", {"type": "message_start", "message": {"id": identifier, "type": "message", "role": "assistant", "model": body.get("model"),
                                                                           "content": [], "stop_reason": None, "stop_sequence": None,
                                                                           "usage": dict(usage, output_tokens=0)}})]
        for index, block in enumerate(blocks):
            shown = dict(block, text="") if block["type"] == "text" else dict(block, input={})
            frames.append(sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": shown}))
            delta = {"type": "text_delta", "text": block["text"]} if block["type"] == "text" else {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            frames.append(sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}))
            frames.append(sse("content_block_stop", {"type": "content_block_stop", "index": index}))
        frames.append(sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 12}}))
        frames.append(sse("message_stop", {"type": "message_stop"}))
        self.stream(frames)

    def responses_api(self, body):
        items = body.get("input") or []
        items = [{"type": "message", "role": "user", "content": items}] if isinstance(items, str) else items
        texts = [text_of(body.get("instructions"))] + [text_of(item.get("content")) for item in items if item.get("type", "message") == "message"]
        step = sum(item.get("type") in CALL_OUTPUTS for item in items)
        tool, arguments, row = self.plan(texts, body.get("tools"), step)
        self.record({"shape": "responses", "model": body.get("model"), "stream": bool(body.get("stream")), "step": step,
                     "instance": row["instance_id"] if row else None, "action": tool[0] if tool else "final",
                     "tools": [item.get("name") for item in body.get("tools") or []]})
        identifier = handle("resp_")
        if tool:
            item = {"type": "function_call", "id": handle("fc_"), "call_id": handle("call_"), "name": tool[0], "arguments": json.dumps(arguments), "status": "completed"}
        else:
            item = {"type": "message", "id": handle("msg_"), "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": FINAL_TEXT, "annotations": []}]}
        usage = {"input_tokens": max(1, len(json.dumps(body)) // 4), "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 12,
                 "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": max(1, len(json.dumps(body)) // 4) + 12}
        response = {"id": identifier, "object": "response", "created_at": int(time.time()), "model": body.get("model"), "status": "completed",
                    "output": [item], "usage": usage, "error": None, "incomplete_details": None}
        if not body.get("stream"):
            return self.reply_json(200, response)
        sequence = iter(range(1 << 30))

        def frame(kind, payload):
            return sse(kind, dict(payload, type=kind, sequence_number=next(sequence)))

        pending = dict(item, status="in_progress")
        frames = [frame("response.created", {"response": dict(response, status="in_progress", output=[], usage=None)}),
                  frame("response.in_progress", {"response": dict(response, status="in_progress", output=[], usage=None)})]
        if tool:
            frames.append(frame("response.output_item.added", {"output_index": 0, "item": dict(pending, arguments="")}))
            frames.append(frame("response.function_call_arguments.delta", {"item_id": item["id"], "output_index": 0, "delta": item["arguments"]}))
            frames.append(frame("response.function_call_arguments.done", {"item_id": item["id"], "output_index": 0, "arguments": item["arguments"]}))
        else:
            part = item["content"][0]
            frames.append(frame("response.output_item.added", {"output_index": 0, "item": dict(pending, content=[])}))
            frames.append(frame("response.content_part.added", {"item_id": item["id"], "output_index": 0, "content_index": 0, "part": dict(part, text="")}))
            frames.append(frame("response.output_text.delta", {"item_id": item["id"], "output_index": 0, "content_index": 0, "delta": part["text"]}))
            frames.append(frame("response.output_text.done", {"item_id": item["id"], "output_index": 0, "content_index": 0, "text": part["text"]}))
            frames.append(frame("response.content_part.done", {"item_id": item["id"], "output_index": 0, "content_index": 0, "part": part}))
        frames.append(frame("response.output_item.done", {"output_index": 0, "item": item}))
        frames.append(frame("response.completed", {"response": response}))
        self.stream(frames)


def serve(host, port, model, context, script, log):
    server = http.server.ThreadingHTTPServer((host, port), Stub)
    server.model, server.context, server.script, server.log, server.lock = model, context, script, log, threading.Lock()
    return server


def main():
    parser = argparse.ArgumentParser(description="A canned Anthropic Messages and OpenAI Responses endpoint for checking agent harnesses end to end")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18433)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--dataset")
    parser.add_argument("--solve", default="")
    parser.add_argument("--log", required=True)
    arguments = parser.parse_args()
    server = serve(arguments.host, arguments.port, arguments.model, arguments.context,
                   Script(arguments.dataset, [value for value in arguments.solve.split(",") if value]), arguments.log)
    print(f"agent_endpoint_stub listening on http://{arguments.host}:{server.server_address[1]}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
