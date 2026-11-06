#!/usr/bin/env python3
"""Batch speculation gate: B concurrent greedy requests with speculation off and on must emit identical tokens; reports aggregate tokens per second both ways.

Prompts are rendered by the chat layer's own template from --chat-dir, so the
engine sees exactly what a chat would send. Three task kinds: "edit" asks for a
source file back with one rename (lookup drafts copy it), "prose" asks for an
essay (model drafts), and "agent" is a coding agent's turn: tool definitions,
a Read call whose result is a source file, then the model's reasoning and the
Edit calls it makes (lookup drafts copy the code it edits, model drafts the
rest). Each (kind, batch, repeat) runs off then on.
"""
import argparse
import hashlib
import http.client
import importlib.util
import json
import re
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path

TOPICS = ["the history of the printing press", "how bridges carry load", "the water cycle", "why volcanoes erupt",
          "the life of honeybees", "how vaccines train immunity", "the rise of container shipping", "how compilers work",
          "the physics of sailing", "the economics of coffee", "how glaciers shape valleys", "the invention of the telephone",
          "how birds navigate", "the chemistry of bread", "the story of the metric system", "how tides work"]


def load_model(chat_dir, model_id):
    sys.path.insert(0, str(chat_dir))
    spec = importlib.util.spec_from_file_location("chat_frontend", chat_dir / "chat_frontend.py")
    frontend = importlib.util.module_from_spec(spec)
    sys.modules["chat_frontend"] = frontend
    spec.loader.exec_module(frontend)
    config = json.loads((chat_dir / "config.json").read_text())
    return frontend, frontend.Model(next(entry for entry in config["models"] if entry["id"] == model_id), chat_dir)


def edit_tasks(sources, count):
    files = [path for path in sorted(Path(sources).rglob("*.c")) if 3000 <= path.stat().st_size <= 9000]
    if len(files) < count:
        raise SystemExit(f"batch_speculation_gate: {sources} has {len(files)} C files of 3-9 KB, need {count}")
    step = len(files) // count
    return [{"role": "user", "content": "Below is a source file. Reproduce it exactly, changing only the name of the first function "
             "defined in it by appending _v2 everywhere that name appears. Output only the file, inside one ```c block.\n\n```c\n"
             + files[index * step].read_text(errors="replace") + "\n```"} for index in range(count)]


AGENT_TOOLS = [
    {"type": "function", "function": {"name": "Read", "description": "Read a file from the repository.",
     "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Path of the file"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "Edit", "description": "Replace one exact occurrence of old_string with new_string in a file.",
     "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string", "description": "Exact text to replace"},
                                                     "new_string": {"type": "string", "description": "Replacement text"}},
                    "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "Bash", "description": "Run a shell command in the repository.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
FUNCTION = re.compile(r"^(?:static\s+)?(?:inline\s+)?[A-Za-z_][A-Za-z0-9_ \t\*]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{]*\)\s*\{", re.M)


def agent_tasks(sources, count):
    files = []
    for path in sorted(Path(sources).rglob("*.c")):
        if 3000 <= path.stat().st_size <= 9000:
            text = path.read_text(errors="replace")
            names = [match.group(1) for match in FUNCTION.finditer(text) if match.group(1) not in ("if", "for", "while", "switch", "return")]
            if len(names) >= 2:
                files.append((path, text, names))
    if len(files) < count:
        raise SystemExit(f"batch_speculation_gate: {sources} has {len(files)} C files of 3-9 KB with two functions, need {count}")
    step = len(files) // count
    tasks = []
    for index in range(count):
        path, text, names = files[index * step]
        relative = str(path.relative_to(sources))
        tasks.append([
            {"role": "system", "content": "You are a coding agent working in a C repository. Think about the change, then make it with the tools. "
                                          "Each Edit replaces one exact occurrence, so quote enough surrounding code to make old_string unique."},
            {"role": "user", "content": f"In {relative}, the function {names[1]} should check its pointer arguments for NULL at the top and return early. "
                                        f"Also rename {names[0]} to {names[0]}_checked everywhere in that file. Read the file first."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_read_0", "type": "function",
                                                                 "function": {"name": "Read", "arguments": json.dumps({"file_path": relative})}}]},
            {"role": "tool", "tool_call_id": "call_read_0", "content": text},
        ])
    return tasks


def prose_tasks(count):
    return [{"role": "user", "content": f"Write a detailed essay of about 600 words on {TOPICS[index % len(TOPICS)]}."} for index in range(count)]


def complete(endpoint, body, timeout):
    url = urllib.parse.urlparse(endpoint)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    started = time.monotonic()
    connection.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    response = connection.getresponse()
    data = response.read()
    seconds = time.monotonic() - started
    connection.close()
    if response.status != 200:
        return {"error": {"http_status": response.status, "body": data[:400].decode(errors="replace")}, "seconds": seconds, "tokens": []}
    reply = json.loads(data)
    return {"error": reply.get("error"), "seconds": seconds, "tokens": list(reply.get("tokens") or []), "finish_reason": reply.get("finish_reason")}


def arm(arguments, prompts, stop, speculation):
    results = [None] * len(prompts)

    def worker(index):
        body = {"prompt_token_ids": prompts[index], "max_tokens": arguments.max_tokens, "temperature": 0.0, "stop_token_ids": stop}
        if not speculation:
            body["speculation"] = False
        results[index] = complete(arguments.engine, body, arguments.timeout)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(len(prompts))]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - started
    tokens = sum(len(result["tokens"]) for result in results)
    return {"speculation": speculation, "wall_s": round(wall, 2), "tokens": tokens, "aggregate_tok_s": round(tokens / wall, 2) if wall > 0 else None,
            "errors": [result["error"] for result in results if result["error"]], "streams": results}


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--engine", required=True)
    parser.add_argument("--chat-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--kinds", default="edit,prose,agent")
    parser.add_argument("--template-kwargs", default="{}", help="JSON object passed to the chat template, e.g. a model's thinking switch")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--streams", help="JSONL file that receives every request's prompt and output token ids, for offline drafter replay")
    arguments = parser.parse_args()
    frontend, model = load_model(arguments.chat_dir, arguments.model)
    batches = [int(value) for value in arguments.batches.split(",")]
    summary = {}
    with open(arguments.output, "a") as handle:
        for kind in arguments.kinds.split(","):
            if kind == "agent":
                conversations, tools = agent_tasks(arguments.sources, max(batches)), AGENT_TOOLS
            elif kind in ("edit", "prose"):
                conversations, tools = [[message] for message in (edit_tasks(arguments.sources, max(batches)) if kind == "edit" else prose_tasks(max(batches)))], None
            else:
                raise SystemExit(f"batch_speculation_gate: unknown kind {kind}")
            prompts = [model.render(frontend.normalize_messages(conversation), tools, json.loads(arguments.template_kwargs))[1] for conversation in conversations]
            for count in batches:
                for repeat in range(arguments.repeats):
                    off = arm(arguments, prompts[:count], model.stop_token_ids, False)
                    on = arm(arguments, prompts[:count], model.stop_token_ids, True)
                    differences = [first_difference(a["tokens"], b["tokens"]) for a, b in zip(off["streams"], on["streams"])]
                    record = {"label": arguments.label, "kind": kind, "batch": count, "repeat": repeat,
                              "off_tok_s": off["aggregate_tok_s"], "on_tok_s": on["aggregate_tok_s"], "off_wall_s": off["wall_s"], "on_wall_s": on["wall_s"],
                              "off_tokens": off["tokens"], "on_tokens": on["tokens"], "identical": sum(1 for value in differences if value is None),
                              "first_differences": differences, "errors": off["errors"] + on["errors"],
                              "streams": [{"off_sha": hashlib.sha256(json.dumps(a["tokens"]).encode()).hexdigest()[:12],
                                           "on_sha": hashlib.sha256(json.dumps(b["tokens"]).encode()).hexdigest()[:12],
                                           "off_tokens": len(a["tokens"]), "on_tokens": len(b["tokens"]),
                                           "off_finish": a.get("finish_reason"), "on_finish": b.get("finish_reason")} for a, b in zip(off["streams"], on["streams"])],
                              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    if arguments.streams:
                        with open(arguments.streams, "a") as streams:
                            for index, (a, b) in enumerate(zip(off["streams"], on["streams"])):
                                streams.write(json.dumps({"label": arguments.label, "kind": kind, "batch": count, "repeat": repeat, "stream": index,
                                                          "prompt": prompts[index], "off": a["tokens"], "on": b["tokens"]}) + "\n")
                    print(json.dumps({key: record[key] for key in ("kind", "batch", "repeat", "off_tok_s", "on_tok_s", "identical", "first_differences", "errors")}), flush=True)
                    summary.setdefault((kind, count), []).append(record)
                    if record["errors"]:
                        print(f"GATE {arguments.label} STOPPED at {kind} B{count} repeat {repeat}: {json.dumps(record['errors'][0])[:300]}", flush=True)
                        return
    for (kind, count), records in summary.items():
        off = statistics.median(record["off_tok_s"] for record in records)
        on = statistics.median(record["on_tok_s"] for record in records)
        exact = sum(record["identical"] for record in records)
        ratio = f"{on / off:.2f}x" if off else "n/a"
        print(f"GATE {arguments.label} {kind} B{count}: off {off:.1f} tok/s, on {on:.1f} tok/s ({ratio}), identical streams {exact}/{count * len(records)}", flush=True)


if __name__ == "__main__":
    main()
