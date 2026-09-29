#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "qualification" / "spec_bakeoff" / "corpus" / "sources"
CLASSES = ("prose", "code", "repetitive", "chat", "long", "thinking", "tool_json", "chinese")
TEMPLATES = {
    "glmflash": {"prefix": "[gMASK]<sop>", "user": "<|user|>\n", "assistant": "<|assistant|>\n<think></think>\n", "thinking": "<|assistant|>\n<think>"},
    "glmfull": {"prefix": "[gMASK]<sop>", "user": "<|user|>\n", "assistant": "<|assistant|>\n<think></think>\n", "thinking": "<|assistant|>\n<think>"},
}
SENTENCES = [
    "The {noun} committee met on the {ordinal} day of the quarter to review the {topic} budget.",
    "Section {section} records that the {topic} allocation rose by {percent} percent over the previous plan.",
    "A footnote in section {section} names {name} as the officer responsible for the {topic} audit.",
    "The {noun} report lists {count} open findings, of which {small} concern the {topic} ledger.",
    "Every {ordinal} paragraph repeats the reference code {code} so that readers can cross-check the appendix.",
    "Reviewers asked whether the {topic} forecast accounted for the {noun} transition finished in {year}.",
]
NOUNS = ["harbor", "transit", "library", "water", "energy", "housing", "forestry", "fisheries"]
TOPICS = ["maintenance", "procurement", "staffing", "compliance", "logistics", "research"]
NAMES = ["Alvarez", "Okafor", "Lindqvist", "Nakamura", "Petrova", "Haddad"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha256(ids: list[int]) -> str:
    import struct
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


class Tokenizer:
    def __init__(self, command: list[str], tokenizer_json: Path):
        self.command = command
        self.tokenizer_json = tokenizer_json
        self.sha256 = file_sha256(tokenizer_json)

    def encode(self, text: str) -> list[int]:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write(text)
            path = handle.name
        try:
            completed = subprocess.run([*self.command, "--tokenizer-json", str(self.tokenizer_json), "--prompt-file", path],
                                       capture_output=True, text=True, check=True)
        finally:
            Path(path).unlink(missing_ok=True)
        ids = [int(line) for line in completed.stdout.split() if line.strip()]
        if not ids:
            raise RuntimeError(f"the tokenizer produced no ids for {text[:60]!r}: {completed.stderr[-200:]}")
        return ids


def render(template: dict, turns: list[tuple[str, str | None]], thinking: bool) -> str:
    text = template["prefix"]
    for index, (user, assistant) in enumerate(turns):
        text += template["user"] + user
        if assistant is not None:
            text += template["assistant"] + assistant
        elif index == len(turns) - 1:
            text += template["thinking"] if thinking else template["assistant"]
    return text


def long_document(seed: int, paragraphs: int) -> tuple[str, list[dict]]:
    import random
    rng = random.Random(seed)
    facts = []
    parts = []
    for section in range(1, paragraphs + 1):
        sentences = []
        code = f"RC-{seed:02d}-{section:04d}"
        percent = rng.randint(1, 40)
        name = rng.choice(NAMES)
        for template in SENTENCES:
            sentences.append(template.format(noun=rng.choice(NOUNS), ordinal=rng.choice(["first", "second", "third", "fourth"]), topic=rng.choice(TOPICS),
                                             section=section, percent=percent, name=name, count=rng.randint(3, 60), small=rng.randint(0, 3), code=code, year=rng.randint(1990, 2025)))
        parts.append(f"Section {section}.\n" + " ".join(sentences))
        if section % 25 == 7:
            facts.append({"section": section, "code": code, "percent": percent, "name": name})
    return "\n\n".join(parts), facts


def long_code_file(seed: int, functions: int) -> str:
    lines = ["import json", "import math", "", ""]
    for index in range(functions):
        lines.extend([f"def metric_{seed}_{index}(values, scale={index % 7 + 1}):",
                      f"    total = sum(v * scale for v in values) + {index}",
                      "    if not values:",
                      "        return 0.0",
                      f"    return math.sqrt(abs(total)) / (len(values) + {index % 3})", "", ""])
    lines.append("REGISTRY = {name: obj for name, obj in globals().items() if name.startswith('metric_')}")
    return "\n".join(lines) + "\n"


def grow_to_tokens(tokenizer: Tokenizer, make: callable, target: int, start: int, step_tolerance: float = 0.1) -> tuple[str, int, dict]:
    count = start
    for _ in range(12):
        text, extra = make(count)
        ids = tokenizer.encode(text)
        if abs(len(ids) - target) <= target * step_tolerance:
            return text, len(ids), extra
        count = max(1, int(count * target / max(1, len(ids))))
    raise RuntimeError(f"could not reach {target} tokens (last {len(ids)})")


def build_class(model: str, content_class: str, source: dict, tokenizer: Tokenizer, template: dict) -> dict:
    thinking = bool(source.get("thinking"))
    prompts = []
    if content_class == "chat":
        for index, conversation in enumerate(source["conversations"]):
            turns = conversation["turns"]
            text = render(template, [(turns[0], None)], thinking)
            ids = tokenizer.encode(text)
            prompts.append({"index": index, "text": turns[0], "follow_up_turns": turns[1:], "prompt_text": text, "prompt_token_ids": ids,
                            "prompt_tokens": len(ids), "output_tokens": source["output_tokens"], "sha256": ids_sha256(ids)})
    elif content_class == "long":
        for index, item in enumerate(source["prompts"]):
            target = int(item["target_tokens"])
            if item["generator"] == "document_qa":
                def make(count, seed=item["seed"]):
                    document, facts = long_document(seed, count)
                    question = item["question"].format(**facts[-1]) if facts else item["question"]
                    return render(template, [(document + "\n\n" + question, None)], thinking), {"facts": facts}
                text, count, extra = grow_to_tokens(tokenizer, make, target, max(4, target // 260))
            elif item["generator"] == "code_edit":
                def make(count, seed=item["seed"]):
                    return render(template, [(item["instruction"] + "\n\n```python\n" + long_code_file(seed, count) + "```", None)], thinking), {}
                text, count, extra = grow_to_tokens(tokenizer, make, target, max(2, target // 45))
            else:
                raise ValueError(f"unknown long-context generator {item['generator']}")
            ids = tokenizer.encode(text)
            prompts.append({"index": index, "text": item.get("question") or item.get("instruction"), "generator": item["generator"], "target_tokens": target,
                            "prompt_text": text, "prompt_token_ids": ids, "prompt_tokens": len(ids), "output_tokens": source["output_tokens"],
                            "sha256": ids_sha256(ids), **extra})
    else:
        for index, item in enumerate(source["prompts"]):
            user = item["text"] if isinstance(item, dict) else item
            text = render(template, [(user, None)], thinking)
            ids = tokenizer.encode(text)
            prompts.append({"index": index, "text": user, "prompt_text": text, "prompt_token_ids": ids, "prompt_tokens": len(ids),
                            "output_tokens": (item.get("output_tokens") if isinstance(item, dict) else None) or source["output_tokens"], "sha256": ids_sha256(ids)})
    combined = hashlib.sha256("".join(prompt["sha256"] for prompt in prompts).encode()).hexdigest()
    return {"model": model, "class": content_class, "thinking": thinking, "tokenizer_sha256": tokenizer.sha256, "template": template,
            "prompts": prompts, "sha256": combined}


def check_file(path: Path) -> dict:
    data = json.loads(path.read_text())
    for prompt in data["prompts"]:
        ids = prompt.get("prompt_token_ids")
        if not ids:
            raise ValueError(f"{path}: prompt {prompt.get('index')} has text but no token ids; the harness refuses text-only prompts")
        if ids_sha256(ids) != prompt["sha256"]:
            raise ValueError(f"{path}: prompt {prompt['index']} ids do not match their sha256")
    combined = hashlib.sha256("".join(prompt["sha256"] for prompt in data["prompts"]).encode()).hexdigest()
    if combined != data["sha256"]:
        raise ValueError(f"{path}: class sha256 does not match its prompts")
    return data


def build(args: argparse.Namespace) -> int:
    if args.model not in TEMPLATES:
        raise SystemExit(f"no chat template registered for {args.model}; known: {sorted(TEMPLATES)}. Add it before building a corpus for that model")
    tokenizer = Tokenizer(args.tokenize_command.split(), Path(args.tokenizer_json))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for content_class in args.classes.split(","):
        if content_class not in CLASSES:
            raise SystemExit(f"unknown class {content_class}; classes are {CLASSES}")
        source = json.loads((Path(args.sources) / f"{content_class}.json").read_text())
        data = build_class(args.model, content_class, source, tokenizer, TEMPLATES[args.model])
        path = out / f"{content_class}.json"
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        written[content_class] = {"prompts": len(data["prompts"]), "prompt_tokens": [p["prompt_tokens"] for p in data["prompts"]], "sha256": data["sha256"]}
        print(f"{content_class}: {len(data['prompts'])} prompts, prompt tokens {written[content_class]['prompt_tokens']}, sha256 {data['sha256'][:16]}")
    (out / "MANIFEST.json").write_text(json.dumps({"model": args.model, "tokenizer_sha256": tokenizer.sha256, "classes": written}, indent=1))
    return 0


def check(args: argparse.Namespace) -> int:
    directory = Path(args.corpus)
    files = sorted(directory.glob("*.json"))
    files = [path for path in files if path.name != "MANIFEST.json"]
    if not files:
        raise SystemExit(f"no class files under {directory}")
    for path in files:
        data = check_file(path)
        print(f"{path.name}: {len(data['prompts'])} prompts ok, class sha256 {data['sha256'][:16]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spec_bakeoff_corpus", description="Build the content-class corpora with a model's tokenizer and chat template; prompt ids are fixed per model and carry their sha256.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--model", required=True)
    p.add_argument("--sources", default=str(SOURCES))
    p.add_argument("--classes", default=",".join(CLASSES))
    p.add_argument("--tokenize-command", required=True, help="e.g. build/sparkpipe_tokenize_prompt")
    p.add_argument("--tokenizer-json", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(function=build)
    p = sub.add_parser("check")
    p.add_argument("--corpus", required=True)
    p.set_defaults(function=check)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
