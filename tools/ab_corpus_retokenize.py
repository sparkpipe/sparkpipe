#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--source-tokenizer", required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--minimum-tokens", type=int, default=1265)
    parser.add_argument("--out-index", required=True)
    parser.add_argument("--out-tokens", required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    source = Tokenizer.from_file(args.source_tokenizer)
    target = Tokenizer.from_file(args.target_tokenizer)
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    data = Path(args.tokens).read_bytes()
    if hashlib.sha256(data).hexdigest() != index["corpus_sha256"]:
        raise SystemExit(f"{args.tokens}: SHA-256 differs from the index's corpus_sha256")
    tokens = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
    limit = index["document_tokens"]
    body = bytearray()
    documents, skipped = [], list(index.get("skipped", []))
    for record in index["documents"]:
        text = source.decode(tokens[record["offset"]:record["offset"] + record["length"]], skip_special_tokens=False)
        ids = target.encode(text, add_special_tokens=False).ids[:limit]
        if len(ids) < args.minimum_tokens:
            skipped.append({"id": record["id"], "reason": f"{len(ids)} target tokens, fewer than {args.minimum_tokens}"})
            continue
        raw = b"".join(token.to_bytes(4, "little") for token in ids)
        documents.append(dict(record, offset=len(body) // 4, length=len(ids), tokens_sha256=hashlib.sha256(raw).hexdigest()))
        body += raw
    out = dict(index, documents=documents, skipped=skipped, corpus_sha256=hashlib.sha256(bytes(body)).hexdigest(),
               total_tokens=len(body) // 4, tokenizer=args.target_name, tokenizer_sha256=hashlib.sha256(Path(args.target_tokenizer).read_bytes()).hexdigest(),
               notes=list(index.get("notes", [])) + [f"retokenized from {index['tokenizer']} by decoding each document and encoding it with {args.target_name}, cut to {limit} tokens"])
    out["strata"] = {name: sum(1 for document in documents if document["stratum"] == name) for name in index["strata"]}
    Path(args.out_tokens).write_bytes(bytes(body))
    Path(args.out_index).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{len(documents)} documents, {len(skipped) - len(index.get('skipped', []))} skipped, strata {out['strata']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
