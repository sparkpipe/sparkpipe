#!/usr/bin/env python3
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

FILLER = ("The river bends past the old mill, where the miller keeps a ledger of every sack of grain. "
          "Each morning the carts arrive, and each evening the ledger is balanced against the stores. ")
CLASSES = {
    "prose": "Write a short story of about three hundred words about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that parses an ISO 8601 date string without using the datetime module, with unit tests.",
    "structured": "Return a JSON array of ten objects describing fictional books, each with title, author, year, genre and pages fields.",
}


class Tokenizer:
    def __init__(self, channel):
        channel = Path(channel).expanduser()
        self.template = json.loads((channel / "model_resident.json").read_text())["chat_template"]
        self.tokenizer = channel / "runtime/tokenizer/tokenizer.json"
        self.tool = channel / "bin/sparkpipe_tokenize_prompt"

    def encode(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as handle:
            handle.write(text)
            handle.flush()
            out = subprocess.run([str(self.tool), "--tokenizer-json", str(self.tokenizer), "--text-file", handle.name],
                                 capture_output=True, text=True, check=True).stdout
        return json.loads(out)["token_ids"] if out.strip().startswith("{") else [int(x) for x in out.replace(",", " ").split()]

    def chat(self, body):
        t = self.template
        return self.encode(t["prefix"] + t["user"] + body + t["turn_suffix"] + t["generation"])


def filler_prompt(tok, target, salt):
    head = f"Document {salt}. Read the following records and then summarize them in one sentence.\n"
    tail = "\nSummarize the records above in one sentence."
    unit = len(tok.encode(FILLER * 8)) / 8
    repeats = max(1, int((target - len(tok.chat(head + tail))) / unit))
    ids = tok.chat(head + FILLER * repeats + tail)
    while len(ids) > target:
        repeats -= max(1, int((len(ids) - target) / unit))
        ids = tok.chat(head + FILLER * repeats + tail)
    return ids


def strip_expect(case, budget, label):
    return {"label": label, "prompt": case["prompt"], "budget": budget}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True)
    ap.add_argument("--base-sessions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tokenizer(a.channel)
    base = Path(a.base_sessions)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = {}
    for name, kind in (("t1", "t1"), ("b1", "b1"), ("b8", "decode"), ("seq6", "seq6"), ("compsec17", "compsec17")):
        s = json.loads((base / f"{name}.json").read_text())
        s["kind"] = kind
        sessions[name] = s
    compsec = sessions["compsec17"]["cases"]
    for rows in (16, 32, 64, 128):
        cases = [strip_expect(compsec[i % len(compsec)], 128, f"b{rows}-{i:03d}-{compsec[i % len(compsec)]['label']}") for i in range(rows)]
        sessions[f"b{rows}"] = {"label": f"b{rows}", "kind": "decode", "sequential": False, "cases": cases}
    classes = []
    for name, text in CLASSES.items():
        ids = tok.chat(text)
        classes += [{"label": f"b1c-{name}-r{r}", "class": name, "prompt": ids, "budget": 256} for r in (1, 2)]
    sessions["b1c"] = {"label": "b1c", "kind": "b1", "sequential": True, "cases": classes}
    for target, label, repeats in ((1024, "ttft1k", 3), (4096, "ttft4k", 2), (16384, "ttft16k", 1), (32768, "ttft32k", 1)):
        cases = []
        for r in range(repeats):
            ids = filler_prompt(tok, target - 16, f"{label}-{r}")
            cases.append({"label": f"{label}-r{r + 1}", "prompt": ids, "budget": 4})
        sessions[label] = {"label": label, "kind": "ttft", "sequential": True, "max_context_tokens": target, "cases": cases}
    for rows in (8, 16):
        cases = [{"label": f"pl{rows}-{i:02d}", "prompt": filler_prompt(tok, 1536, f"pl{rows}-{i}"), "budget": 8} for i in range(rows)]
        sessions[f"pl{rows}"] = {"label": f"pl{rows}", "kind": "prefill", "sequential": False, "cases": cases}
    for name, s in sessions.items():
        (out / f"{name}.json").write_text(json.dumps(s))
        lengths = [len(c["prompt"]) for c in s["cases"]]
        print(f"{name}: {len(lengths)} cases, prompt {min(lengths)}-{max(lengths)} tokens, kind {s['kind']}")


if __name__ == "__main__":
    main()
