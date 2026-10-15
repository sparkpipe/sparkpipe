import argparse
import http.client
import importlib.util
import json
import re
import struct
import sys
import time
import urllib.parse
from pathlib import Path

DEFINITIONS = [re.compile(r"^def ([A-Za-z_][A-Za-z0-9_]*)\(", re.M),
               re.compile(r"^static[^;{(]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M),
               re.compile(r"^[A-Za-z_][A-Za-z0-9_ \*]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*$", re.M)]
STYLES = {
    "editfull": "Reply with the complete updated file and nothing else: no explanation and no code fences.",
    "editsr": ("Reply only with SEARCH/REPLACE blocks, one per changed region, each written as\n<<<<<<< SEARCH\n(the exact original lines)\n=======\n"
               "(the new lines)\n>>>>>>> REPLACE\nCopy the SEARCH lines exactly, with enough unchanged lines around each change to make it unique."),
}


def first_definition(text):
    for pattern in DEFINITIONS:
        for match in pattern.finditer(text):
            name = match.group(1)
            if name not in ("if", "for", "while", "switch", "return", "sizeof") and len(re.findall(r"\b" + re.escape(name) + r"\b", text)) >= 2:
                return name
    return None


def build(args):
    root = Path(args.root)
    tasks = []
    for relative in args.files:
        text = (root / relative).read_text()
        name = first_definition(text)
        if name is None:
            raise SystemExit(f"{relative}: no function definition that is also used")
        for style, rule in STYLES.items():
            prompt = (f"Here is the file {relative}:\n\n{text}\n\nRename the function {name} to {name}_v2 at its definition and at every use "
                      f"in this file, and change nothing else. {rule}")
            tasks.append({"name": f"{style}_{len(tasks) // 2 + 1:02d}", "file": relative, "function": name, "style": style,
                          "messages": [{"role": "user", "content": prompt}], "file_chars": len(text)})
    Path(args.out).write_text(json.dumps(tasks, indent=1))
    print(f"{len(tasks)} tasks from {len(args.files)} files")


def load_frontend(directory):
    spec = importlib.util.spec_from_file_location("chat_frontend", Path(directory) / "chat_frontend.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["chat_frontend"] = module
    spec.loader.exec_module(module)
    return module


def stream(endpoint, body):
    url = urllib.parse.urlparse(endpoint)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=3600)
    connection.request("POST", "/v1/completions", json.dumps(dict(body, stream=True)), {"Content-Type": "application/json"})
    response = connection.getresponse()
    tokens, buffer, finish = [], b"", None
    while True:
        chunk = response.read1(65536)
        if not chunk:
            break
        buffer += chunk
        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            line = event.strip()
            if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                document = json.loads(line[5:])
                tokens.extend(document.get("tokens", []))
                finish = (document.get("choices") or [{}])[0].get("finish_reason") or finish
    connection.close()
    return tokens, finish


def generate(args):
    frontend = load_frontend(args.chat_dir)
    config = json.loads((Path(args.chat_dir) / "config.json").read_text())
    spec = next(model for model in config["models"] if model["id"] == args.model)
    model = frontend.Model(spec, Path(args.chat_dir))
    kwargs = dict(model.reasoning_effort.get(args.reasoning_effort, {})) if args.reasoning_effort else {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for task in json.loads(Path(args.corpus).read_text()):
        _, prompt_ids = model.render(frontend.normalize_messages(task["messages"]), None, kwargs)
        budget = min(args.max_tokens, int(task["file_chars"] / 2.5) + 512)
        started = time.monotonic()
        output, finish = stream(args.engine, {"prompt_token_ids": prompt_ids, "max_tokens": budget, "temperature": 0, "stop_token_ids": model.stop_token_ids})
        seconds = time.monotonic() - started
        output = [token for token in output if token not in model.stop_token_ids]
        tokens = list(prompt_ids) + output
        (out / f"{task['name']}.u32").write_bytes(struct.pack(f"<II{len(tokens)}I", len(prompt_ids), len(tokens), *tokens))
        row = {"name": task["name"], "file": task["file"], "prompt_tokens": len(prompt_ids), "output_tokens": len(output), "finish": finish,
               "seconds": round(seconds, 1)}
        report.append(row)
        print(json.dumps(row), flush=True)
    (out / "generation.json").write_text(json.dumps({"model": args.model, "reasoning_effort": args.reasoning_effort, "tasks": report}, indent=1))


def main():
    parser = argparse.ArgumentParser(description="Build a code-edit corpus from repository files and record greedy no-speculation streams for offline draft replay.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", default=".")
    b.add_argument("--out", required=True)
    b.add_argument("files", nargs="+")
    g = sub.add_parser("generate")
    g.add_argument("--corpus", required=True)
    g.add_argument("--chat-dir", required=True, help="the chat layer directory with chat_frontend.py, config.json and the model directories")
    g.add_argument("--model", required=True)
    g.add_argument("--engine", required=True)
    g.add_argument("--reasoning-effort", default="minimal")
    g.add_argument("--max-tokens", type=int, default=6144)
    g.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args) if args.command == "build" else generate(args)


if __name__ == "__main__":
    main()
