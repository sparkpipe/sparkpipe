#!/usr/bin/env python3
"""Frozen, SHA-pinned token corpora for teacher-forced A/B scoring (design §3.4, critic §10).

Documents arrive as JSONL records {id, stratum, text | token_ids, source{...}}
from the extract-* subcommands (or from any other producer), are tokenized with
the pinned tokenizer through build/sparkpipe_tokenize_prompt, cut to exactly
--length tokens, and accepted in input order unless they are too short or share
a 64-token block with an already accepted document. A shared block would let
the engine's prefix reuse serve cached KV (C4: cached_prompt_tokens must stay
0), so the finished corpus is re-checked and refused if any aligned full block
repeats across documents.

Outputs, both SHA-pinned in the index and in the PLAN:
  <out>/<name>.tokens.u32    little-endian uint32 token ids, documents back to back
  <out>/<name>.index.json    sparkpipe-ab-corpus-v1: tokenizer pin, per-document
                             id, stratum, offset, length, tokens_sha256, source,
                             optional score_from and probe_spans, skipped candidates

usage:
  ab_corpus_build.py extract-arxiv PARQUET --out DOCS.jsonl --stratum NAME --id-prefix P
      [--licenses L,..] [--min-chars N] [--max-chars N] [--limit N] [--seed S]
  ab_corpus_build.py extract-code REPO COMMIT --out DOCS.jsonl [--group-by-directory] [--min-chars N]
  ab_corpus_build.py retrieval DOCS.jsonl --out DOCS.jsonl --keys N --queries N --seed S
  ab_corpus_build.py build --name NAME --length TOKENS --out DIR --tokenizer-json PATH
      --tokenizer-bin PATH --quota STRATUM=N[,..] DOCS.jsonl [DOCS.jsonl ...]
  ab_corpus_build.py check INDEX.json          re-verify SHAs and block uniqueness
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FORMAT = "sparkpipe-ab-corpus-v1"
BLOCK = 64
CODE_SUFFIXES = (".c", ".h", ".cu", ".cuh")
RETRIEVAL_MARK = "\n\nRecall check.\n"


class CorpusError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens_bytes(tokens) -> bytes:
    return np.asarray(tokens, dtype="<u4").tobytes()


def block_keys(tokens) -> list:
    data = np.asarray(tokens, dtype="<u4")
    return [sha256_bytes(data[start:start + BLOCK].tobytes()) for start in range(0, len(data) - BLOCK + 1, BLOCK)]


def collisions(documents: list) -> list:
    owner, found = {}, []
    for index, tokens in enumerate(documents):
        for block, key in enumerate(block_keys(tokens)):
            if key in owner and owner[key][0] != index:
                found.append({"block_sha256": key, "first": owner[key], "second": (index, block)})
            else:
                owner.setdefault(key, (index, block))
    return found


class Tokenizer:
    def __init__(self, tokenizer_json: Path, binary: Path, workdir: Path):
        self.binary = binary
        self.json_sha256 = sha256_file(tokenizer_json)
        self.compiled = workdir / "tokenizer.compiled"
        self.workdir = workdir
        probe = workdir / "probe.txt"
        probe.write_text("tokenizer probe\n")
        self._run(["--tokenizer-json", str(tokenizer_json), "--text-file", str(probe), "--disable-special-token-match",
                   "--save-compiled-tokenizer", str(self.compiled)])

    def _run(self, arguments: list) -> list:
        result = subprocess.run([str(self.binary), *arguments], capture_output=True, text=True)
        if result.returncode != 0:
            raise CorpusError(f"tokenizer failed ({result.returncode}): {result.stderr.strip()[-400:]}")
        return [int(line) for line in result.stdout.split()]

    def encode(self, text: str) -> list:
        path = self.workdir / "text.txt"
        path.write_bytes(text.encode("utf-8"))
        return self._run(["--tokenizer-compiled", str(self.compiled), "--text-file", str(path), "--disable-special-token-match"])


def latex_body(text: str) -> str:
    marker = text.find("\\begin{document}")
    return text[marker + len("\\begin{document}"):] if marker >= 0 else text


def arxiv_month(paper_id: str):
    match = re.match(r"(\d{2})(\d{2})\.\d+", paper_id)
    if match:
        return f"20{match.group(1)}-{match.group(2)}"
    match = re.match(r"[a-z-]+(?:\.[A-Z]{2})?/(\d{2})(\d{2})\d+", paper_id)
    if match:
        year = int(match.group(1))
        return f"{1900 + year if year > 90 else 2000 + year}-{match.group(2)}"
    return None


def extract_arxiv(args) -> int:
    import pyarrow.parquet as pq
    table = pq.read_table(args.parquet, columns=["paper_id", "text", "text_sha256", "license", "primary_category", "title"])
    licenses = set(args.licenses.split(",")) if args.licenses else None
    prefix = re.compile(args.id_prefix)
    rows = []
    for record in table.to_pylist():
        if not prefix.match(record["paper_id"]):
            continue
        if licenses is not None and (record["license"] or "none") not in licenses:
            continue
        body = latex_body(record["text"] or "")
        if len(body) < args.min_chars:
            continue
        rows.append(record)
    order = sorted(rows, key=lambda record: sha256_bytes(f"{args.seed}:{record['paper_id']}".encode()))
    written = 0
    with open(args.out, "w") as handle:
        for record in order[:args.limit]:
            body = latex_body(record["text"])[:args.max_chars]
            handle.write(json.dumps({"id": f"arxiv:{record['paper_id']}", "stratum": args.stratum, "text": body,
                                     "source": {"dataset": args.dataset, "paper_id": record["paper_id"], "month": arxiv_month(record["paper_id"]),
                                                "text_sha256": record["text_sha256"], "license": record["license"],
                                                "primary_category": record["primary_category"], "cut": "after \\begin{document}"}},
                                    sort_keys=True) + "\n")
            written += 1
    print(f"extract-arxiv: {written} of {len(rows)} eligible documents -> {args.out}")
    return 0


def strip_includes(text: str) -> str:
    lines = text.split("\n")
    index = 0
    while index < len(lines) and (not lines[index].strip() or lines[index].lstrip().startswith(("#include", "#pragma", "#ifndef", "#define", "#ifdef", "#endif", "extern \"C\"", "{", "}"))):
        index += 1
    return "\n".join(lines[index:])


def git(repo: str, *arguments) -> str:
    return subprocess.run(["git", "-C", repo, *arguments], capture_output=True, text=True, check=True).stdout


def extract_code(args) -> int:
    commit = git(args.repo, "rev-parse", args.commit).strip()
    paths = [path for path in git(args.repo, "ls-tree", "-r", "--name-only", commit).split("\n")
             if path.endswith(CODE_SUFFIXES) and not path.startswith(("tests/", "experiments/"))]
    groups = {}
    for path in sorted(paths):
        key = os.path.dirname(path) if args.group_by_directory else path
        groups.setdefault(key, []).append(path)
    candidates = []
    for key, members in groups.items():
        texts = []
        for path in members:
            text = git(args.repo, "show", f"{commit}:{path}")
            texts.append(strip_includes(text) if not args.group_by_directory else f"// {path}\n{text}")
        body = "\n".join(texts)
        if len(body) >= args.min_chars:
            candidates.append((key, members, body))
    candidates.sort(key=lambda item: sha256_bytes(f"{args.seed}:{item[0]}".encode()))
    with open(args.out, "w") as handle:
        for key, members, body in candidates[:args.limit]:
            handle.write(json.dumps({"id": f"code:{key}", "stratum": args.stratum, "text": body[:args.max_chars],
                                     "source": {"repository": "sparkpipe", "commit": commit, "paths": members}}, sort_keys=True) + "\n")
    print(f"extract-code: {min(len(candidates), args.limit)} of {len(candidates)} candidates at {commit[:12]} -> {args.out}")
    return 0


def retrieval(args) -> int:
    rng = random.Random(args.seed)
    written = 0
    with open(args.docs) as source, open(args.out, "w") as handle:
        for line in source:
            record = json.loads(line)
            keys = [f"K{rng.randrange(16 ** 6):06X}" for _ in range(args.keys)]
            values = [f"{rng.randrange(10 ** 8):08d}" for _ in range(args.keys)]
            registry = "Access code registry for this document.\n" + "".join(f"{k} = {v}\n" for k, v in zip(keys, values)) + "\n"
            asked = rng.sample(range(args.keys), args.queries)
            tail = RETRIEVAL_MARK + "".join(f"The access code for {keys[i]} is {values[i]}.\n" for i in asked)
            handle.write(json.dumps({"id": f"retrieval:{record['id']}", "stratum": "retrieval", "head": registry, "text": record["text"],
                                     "tail": tail, "answers": [values[i] for i in asked],
                                     "source": dict(record["source"], retrieval_seed=args.seed, keys=args.keys, queries=args.queries)},
                                    sort_keys=True) + "\n")
            written += 1
    print(f"retrieval: {written} documents -> {args.out}")
    return 0


def document_tokens(record: dict, tokenizer: Tokenizer, length: int) -> tuple:
    if "token_ids" in record:
        tokens = list(record["token_ids"])
        extra = {"score_from": int(record.get("score_from", 0))}
        return (tokens[:length] if len(tokens) >= length or record.get("allow_short") else None), extra
    if "tail" in record:
        head = tokenizer.encode(record["head"])
        tail = tokenizer.encode(record["tail"])
        body = tokenizer.encode(record["text"])
        room = length - len(head) - len(tail)
        if room <= 0 or len(body) < room:
            return None, {}
        tokens = head + body[:room] + tail
        spans, cursor = [], len(head) + room
        for answer in record["answers"]:
            answer_tokens = tokenizer.encode(answer)
            for start in range(cursor, len(tokens) - len(answer_tokens) + 1):
                if tokens[start:start + len(answer_tokens)] == answer_tokens:
                    spans.append([start, start + len(answer_tokens)])
                    cursor = start + len(answer_tokens)
                    break
            else:
                raise CorpusError(f"{record['id']}: answer {answer} does not tokenize to a findable span")
        return tokens, {"probe_spans": spans, "tail_from": len(head) + room}
    tokens = tokenizer.encode(record["text"])
    return (tokens[:length] if len(tokens) >= length else None), {}


def build(args) -> int:
    quota = {}
    for item in args.quota.split(","):
        stratum, _, count = item.partition("=")
        quota[stratum] = int(count)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    accepted, meta, skipped = [], [], []
    owner = set()
    counts = {stratum: 0 for stratum in quota}
    with tempfile.TemporaryDirectory() as workdir:
        tokenizer = Tokenizer(Path(args.tokenizer_json), Path(args.tokenizer_bin), Path(workdir))
        for docs_path in args.docs:
            for line in open(docs_path):
                record = json.loads(line)
                stratum = record["stratum"]
                if stratum not in quota or counts[stratum] >= quota[stratum]:
                    continue
                tokens, extra = document_tokens(record, tokenizer, args.length)
                if tokens is None:
                    skipped.append({"id": record["id"], "reason": "shorter than the document length"})
                    continue
                keys = block_keys(tokens)
                if any(key in owner for key in keys):
                    skipped.append({"id": record["id"], "reason": "shares a 64-token block with an accepted document"})
                    continue
                owner.update(keys)
                counts[stratum] += 1
                accepted.append(tokens)
                meta.append({"id": record["id"], "stratum": stratum, "source": record["source"], **extra})
    short = {stratum: quota[stratum] - counts[stratum] for stratum in quota if counts[stratum] < quota[stratum]}
    if short:
        raise CorpusError(f"not enough documents for strata {short}")
    found = collisions(accepted)
    if found:
        raise CorpusError(f"{len(found)} 64-token blocks repeat across documents; refusing the corpus")
    tokens_path = out / f"{args.name}.tokens.u32"
    offset = 0
    with open(tokens_path, "wb") as handle:
        for tokens, entry in zip(accepted, meta):
            data = tokens_bytes(tokens)
            handle.write(data)
            entry.update({"offset": offset, "length": len(tokens), "tokens_sha256": sha256_bytes(data)})
            offset += len(tokens)
    index = {
        "format": FORMAT, "name": args.name, "block_tokens": BLOCK, "document_tokens": args.length,
        "corpus_sha256": sha256_file(tokens_path), "total_tokens": offset,
        "tokenizer_sha256": tokenizer.json_sha256, "tokenizer": args.tokenizer_label,
        "builder_commit": git(str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD").strip(),
        "strata": counts, "documents": meta, "skipped": skipped, "notes": args.note or [],
    }
    index_path = out / f"{args.name}.index.json"
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    print(f"build: {args.name} {len(accepted)} documents, {offset} tokens, corpus_sha256 {index['corpus_sha256']}, "
          f"index_sha256 {sha256_file(index_path)}, skipped {len(skipped)}")
    return 0


def load_documents(index_path: Path) -> tuple:
    index = json.loads(index_path.read_text())
    tokens_path = index_path.with_name(f"{index['name']}.tokens.u32")
    if sha256_file(tokens_path) != index["corpus_sha256"]:
        raise CorpusError(f"{tokens_path}: sha256 does not match the index")
    data = np.fromfile(tokens_path, dtype="<u4")
    documents = []
    for entry in index["documents"]:
        tokens = data[entry["offset"]:entry["offset"] + entry["length"]]
        if sha256_bytes(tokens.tobytes()) != entry["tokens_sha256"]:
            raise CorpusError(f"{entry['id']}: tokens_sha256 mismatch")
        documents.append(tokens)
    return index, documents


def check(args) -> int:
    index, documents = load_documents(Path(args.index))
    found = collisions(documents)
    if found:
        raise CorpusError(f"{len(found)} 64-token blocks repeat across documents")
    ids = [entry["id"] for entry in index["documents"]]
    if len(set(ids)) != len(ids):
        raise CorpusError("document ids repeat")
    print(f"check: {index['name']} OK {len(documents)} documents {index['total_tokens']} tokens, no shared 64-token block")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract-arxiv")
    p.add_argument("parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--stratum", required=True)
    p.add_argument("--id-prefix", required=True)
    p.add_argument("--dataset", default="secemp9/arxiv-complete@cee894837962fede5612cccf2a4c7cacf49b4c3a")
    p.add_argument("--licenses")
    p.add_argument("--min-chars", type=int, default=12000)
    p.add_argument("--max-chars", type=int, default=24000)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260929)
    p = sub.add_parser("extract-code")
    p.add_argument("repo")
    p.add_argument("commit")
    p.add_argument("--out", required=True)
    p.add_argument("--stratum", default="code")
    p.add_argument("--group-by-directory", action="store_true")
    p.add_argument("--min-chars", type=int, default=8000)
    p.add_argument("--max-chars", type=int, default=24000)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260929)
    p = sub.add_parser("retrieval")
    p.add_argument("docs")
    p.add_argument("--out", required=True)
    p.add_argument("--keys", type=int, default=32)
    p.add_argument("--queries", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260929)
    p = sub.add_parser("build")
    p.add_argument("--name", required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer-json", required=True)
    p.add_argument("--tokenizer-label", default="zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0 tokenizer.json")
    p.add_argument("--tokenizer-bin", required=True)
    p.add_argument("--quota", required=True)
    p.add_argument("--note", action="append")
    p.add_argument("docs", nargs="+")
    p = sub.add_parser("check")
    p.add_argument("index")
    args = parser.parse_args()
    try:
        return {"extract-arxiv": extract_arxiv, "extract-code": extract_code, "retrieval": retrieval,
                "build": build, "check": check}[args.command](args)
    except CorpusError as error:
        print(f"ab_corpus_build: REFUSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
