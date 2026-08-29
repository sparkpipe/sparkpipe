#!/usr/bin/env python3
"""glm5_next COMPSEC-17 quality gate (ds4_eval protocol, local endpoint).

Fires the 17 COMPSEC cases (ids compsec-076..092 of
qualification/ds4_eval/quality-fixtures-glm5.3-flash.json, pre-tokenized)
at a live /v1/completions endpoint sequentially, temperature 0, and archives
the run in the canonical ds4_eval format (REPORT.md + INTEGRITY.json +
summary.json + cases.json + responses/*.json).

Grading rule (stated in the run's REPORT.md): the model's completion is
trimmed; PASS iff the fixture `answer` string (e.g. "17-20", "3,13-15")
occurs as a substring of the completion's first non-empty line. Raw text is
preserved in every response file for independent regrading.

usage:
  glm5_next_compsec17.py --endpoint http://spark0:8433 \
      --fixture qualification/ds4_eval/quality-fixtures-glm5.3-flash.json \
      --out qualification/ds4_eval/runs/glm5-next-tp16-<date>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

COMPSEC_IDS = [f"compsec-{i:03d}" for i in range(76, 93)]

# GPT-2 style byte-level BPE reverse alphabet (the GLM tokenizer.json vocab
# uses the same byte-unicode mapping).
def _bytes_to_unicode() -> dict:
    bs = (list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_TO_CHAR = _bytes_to_unicode()
_CHAR_TO_BYTE = {v: k for k, v in _BYTE_TO_CHAR.items()}


def load_decoder(tokenizer_path: Path):
    """Return fn(token_ids) -> text (vocab pieces joined + byte-decoded)."""
    tok = json.loads(tokenizer_path.read_text())
    model = tok.get("model", {})
    vocab = model.get("vocab") or {}
    id_to_piece = {v: k for k, v in vocab.items()}

    def decode(ids: list) -> str:
        out = bytearray()
        for i in ids:
            p = id_to_piece.get(int(i))
            if p is None:
                out += b"\xef\xbf\xbd"  # U+FFFD
                continue
            out += bytes(_CHAR_TO_BYTE.get(ch, 0xFF) for ch in p)
        return out.decode("utf-8", errors="replace")

    return decode


def call(endpoint: str, prompt_ids: list, max_tokens: int, temperature: float,
         timeout: int = 600) -> dict:
    body = json.dumps({
        "prompt_token_ids": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return {"elapsed_s": time.monotonic() - t0, "payload": payload}


def grade(text: str, answer: str) -> tuple:
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    passed = answer in first_line
    return passed, first_line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8433")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer.json (vocab used to decode token ids)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    decode = load_decoder(Path(args.tokenizer))

    fixture = json.loads(Path(args.fixture).read_text())
    cases = [c for c in fixture["cases"] if c["id"] in COMPSEC_IDS]
    if len(cases) != 17:
        print(f"FATAL: expected 17 COMPSEC cases, found {len(cases)}", file=sys.stderr)
        return 2
    cases.sort(key=lambda c: c["id"])

    out = Path(args.out)
    (out / "responses").mkdir(parents=True, exist_ok=True)

    results = []
    for i, c in enumerate(cases, 1):
        r = call(args.endpoint, c["prompt_token_ids"], args.max_tokens,
                 args.temperature, args.timeout)
        payload = r["payload"]
        tokens = payload.get("tokens") or []
        text = decode(tokens) if tokens else (payload.get("text") or "")
        passed, extracted = grade(text, c["answer"])
        rec = {
            "index": i,
            "id": c["id"],
            "source": "COMPSEC",
            "domain": c["domain"],
            "expected": c["answer"],
            "extracted": extracted,
            "passed": passed,
            "elapsed_ms": round(r["elapsed_s"] * 1000, 1),
            "generated_token_count": len(tokens),
            "status": payload.get("status"),
        }
        results.append(rec)
        resp = {
            "id": c["id"],
            "request": {
                "endpoint": args.endpoint + "/v1/completions",
                "prompt_token_count": len(c["prompt_token_ids"]),
                "prompt_sha256": hashlib.sha256(
                    json.dumps(c["prompt_token_ids"]).encode()).hexdigest(),
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
            "response": payload,
            "decoded_text": text,
            "grade": rec,
        }
        fname = f"{i:03d}-{c['id']}.json"
        (out / "responses" / fname).write_text(json.dumps(resp, indent=1))
        mark = "PASS" if passed else "FAIL"
        print(f"[{i:2d}/17] {c['id']} {c['domain'][:24]:24s} {mark} "
              f"expected={c['answer']!r} got={extracted[:60]!r} "
              f"({r['elapsed_s']:.1f}s, {len(tokens)} tok)", flush=True)

    passed_n = sum(1 for r in results if r["passed"])
    summary = {
        "format": "ds4-eval-glm5-next-compsec17-summary-v1",
        "generated_at": time.strftime("%FT%T%z"),
        "endpoint": args.endpoint,
        "fixture": Path(args.fixture).name,
        "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
        "parameters": {"max_tokens": args.max_tokens,
                       "temperature": args.temperature, "concurrency": 1},
        "grading_rule": ("PASS iff the fixture answer string occurs in the "
                         "completion's first non-empty line"),
        "completed": len(results),
        "passed": passed_n,
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    cases_out = {
        "format": "ds4-eval-glm5-next-compsec17-cases-v1",
        "note": "cases exactly as selected from the fixture (pre-tokenized)",
        "cases": cases,
    }
    (out / "cases.json").write_text(json.dumps(cases_out, indent=1))

    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    hashes = {}
    lines = []
    for rel in files:
        if rel in ("INTEGRITY.json", "REPORT.md"):
            continue
        h = hashlib.sha256((out / rel).read_bytes()).hexdigest()
        hashes[rel] = h
        if rel.startswith("responses/"):
            lines.append(f"{h}  {rel}\n")
    integrity = {
        "format": "ds4-eval-run-integrity-v1",
        "file_sha256": hashes,
        "response_count": len(results),
        "response_stream_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "response_stream_paths": "run-relative-posix",
    }
    (out / "INTEGRITY.json").write_text(json.dumps(integrity, indent=1))

    print(f"\nCOMPSEC-17 RESULT: {passed_n}/17 "
          f"({', '.join(r['id'] + ('+ok' if r['passed'] else '+X') for r in results if not r['passed']) or 'all pass'})")
    return 0 if passed_n >= 15 else 1


if __name__ == "__main__":
    sys.exit(main())
