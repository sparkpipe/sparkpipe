#!/usr/bin/env python3
"""COMPSEC-17 quality gate (ds4_eval protocol, local endpoint), any chat model.

Fires the 17 COMPSEC cases (ids compsec-076..092 of
qualification/ds4_eval/quality-fixtures-glm5.3-flash.json) at a live
/v1/completions endpoint sequentially, temperature 0, and archives the run in
the canonical ds4_eval format (INTEGRITY.json + summary.json + cases.json +
responses/*.json).

Each case's question is pre-tokenized with the fixture's GLM vocabulary; it is
decoded to text with --tokenizer (the fixture's tokenizer.json) and wrapped
in the served model's chat template (--template), rendered as its publisher's
template renders one user turn with the generation prompt:
  glm:    [gMASK]<sop><|user|>\n{question}<|assistant|>\n then <think></think>\n
          (--thinking off) or <think> (--thinking on)
  gemma4: <bos><|turn>user\n{question}<turn|>\n<|turn>model\n then
          <|channel>thought\n<channel|> (--thinking off); --thinking on adds the
          <|turn>system\n<|think|>\n<turn|>\n turn before the user turn
The endpoint tokenizes the prompt, so the role markers become their special
tokens, and returns the completion text through its own tokenizer.

Grading is the canonical ds4_eval rule from
qualification/ds4_eval/compare_runs.py: the answer is taken from the last
"Answer:" line after any </think>, and a case passes iff its line set is a
non-empty subset of the fixture's expected lines.

--compare REFERENCE CANDIDATE checks row invariance instead: it loads two
archived runs (for example --concurrency 1 and --concurrency 17 against the same
root) and requires every case's generated token ids to be identical. It prints
the first differing token index per case and exits 0 when all 17 cases match,
1 when any differs, and 2 when a run is missing, malformed or incomplete.

usage:
  compsec17.py --template glm --endpoint http://127.0.0.1:8433 --thinking off \
      --fixture qualification/ds4_eval/quality-fixtures-glm5.3-flash.json \
      --tokenizer <runtime>/tokenizer/tokenizer.json \
      --out qualification/ds4_eval/runs/glm5-next-tp16-<date>
  compsec17.py --compare RUN_SEQUENTIAL RUN_CONCURRENT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qualification" / "ds4_eval"))
from compare_runs import answer_matches, extract_answer

COMPSEC_IDS = [f"compsec-{i:03d}" for i in range(76, 93)]
CHAT_TEMPLATES = {
    "glm": {
        "off": ("[gMASK]<sop><|user|>\n", "<|assistant|>\n<think></think>\n"),
        "on": ("[gMASK]<sop><|user|>\n", "<|assistant|>\n<think>"),
    },
    "gemma4": {
        "off": ("<bos><|turn>user\n",
                "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
        "on": ("<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n",
               "<turn|>\n<|turn>model\n"),
    },
}
SUMMARY_FORMATS = {
    "glm": ("ds4-eval-glm5-next-compsec17-summary-v1", "ds4-eval-glm5-next-compsec17-cases-v1"),
    "gemma4": ("ds4-eval-compsec17-summary-v1", "ds4-eval-compsec17-cases-v1"),
    "ling": ("ds4-eval-compsec17-summary-v1", "ds4-eval-compsec17-cases-v1"),
}

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
                out += b"\xef\xbf\xbd"
                continue
            out += bytes(_CHAR_TO_BYTE.get(ch, 0xFF) for ch in p)
        return out.decode("utf-8", errors="replace")

    return decode


def build_prompt(question: str, thinking: str, template: str) -> str:
    prefix, suffix = CHAT_TEMPLATES[template][thinking]
    return prefix + question + suffix


def call(endpoint: str, prompt: str, max_tokens: int, temperature: float,
         timeout: int = 600) -> dict:
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read().decode("utf-8", "replace")
        raise RunError(f"HTTP {error.code} from {req.full_url}: {detail}") from error
    return {"elapsed_s": time.monotonic() - t0, "payload": payload}


def require_text_endpoint(endpoint: str, timeout: int = 30) -> dict:
    url = endpoint.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            health = json.loads(resp.read())
    except (OSError, ValueError) as error:
        raise RunError(f"{url}: {error}") from error
    if not isinstance(health, dict) or health.get("tokenizer") is not True:
        reported = health.get("tokenizer") if isinstance(health, dict) else health
        raise RunError(f"{url} reports tokenizer={reported!r}: the API has no tokenizer sidecar, "
                       "so every text prompt is answered HTTP 400 tokenizer_unavailable; start it "
                       "from a deployment whose \"tokenizer\" block names the asset staged under "
                       "its runtime root")
    return health


def grade(text: str, answer: str) -> tuple:
    case = {"source": "COMPSEC", "choices": [], "answer": answer}
    extracted, _ = extract_answer(case, text, "")
    return answer_matches(case, extracted), extracted


class RunError(Exception):
    pass


def load_run_tokens(run: Path) -> dict:
    tokens, statuses = {}, {}
    responses = sorted((run / "responses").glob("*.json"))
    if not responses:
        raise RunError(f"{run}: no responses/*.json")
    for path in responses:
        try:
            record = json.loads(path.read_text())
            case = record["id"]
            ids = record["response"]["tokens"]
            status = record["response"].get("status")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise RunError(f"{path}: {error!r}") from error
        if case not in COMPSEC_IDS:
            raise RunError(f"{path}: case {case!r} is not a COMPSEC-17 case")
        if not isinstance(ids, list) or not all(isinstance(value, int) for value in ids):
            raise RunError(f"{path}: response.tokens is not a list of token ids")
        if not ids:
            raise RunError(f"{path}: response.tokens is empty (status {status!r})")
        if case in tokens:
            raise RunError(f"{path}: duplicate case {case}")
        tokens[case] = ids
        statuses[case] = status
    missing = [case for case in COMPSEC_IDS if case not in tokens]
    if missing:
        raise RunError(f"{run}: missing cases {', '.join(missing)}")
    return tokens, statuses


def first_difference(left: list, right: list) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return -1 if len(left) == len(right) else min(len(left), len(right))


def compare_runs(reference: Path, candidate: Path, out=sys.stdout) -> int:
    try:
        left, left_status = load_run_tokens(reference)
        right, right_status = load_run_tokens(candidate)
    except RunError as error:
        print(f"COMPSEC-COMPARE-ERROR {error}", file=out)
        return 2
    differing = 0
    for case in COMPSEC_IDS:
        index = first_difference(left[case], right[case])
        if index < 0 and left_status[case] != right_status[case]:
            differing += 1
            print(f"COMPSEC-COMPARE {case} DIFFERS status reference_status={left_status[case]!r} "
                  f"candidate_status={right_status[case]!r} tokens={len(left[case])}", file=out)
            continue
        if index < 0:
            print(f"COMPSEC-COMPARE {case} IDENTICAL tokens={len(left[case])}", file=out)
            continue
        differing += 1
        a = left[case][index] if index < len(left[case]) else None
        b = right[case][index] if index < len(right[case]) else None
        print(f"COMPSEC-COMPARE {case} DIFFERS first_token_index={index} reference_token={a} candidate_token={b} "
              f"reference_tokens={len(left[case])} candidate_tokens={len(right[case])}", file=out)
    print(f"COMPSEC-COMPARE RESULT identical={17 - differing}/17", file=out)
    return 0 if differing == 0 else 1


def arguments(description: str, thinking_modes) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8433")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer.json (vocab used to decode token ids)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", required=True, choices=sorted(thinking_modes))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--pass-threshold", type=int, default=14)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=600)
    return ap


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        if len(sys.argv) != 4:
            print("usage: compsec17.py --compare REFERENCE_RUN CANDIDATE_RUN", file=sys.stderr)
            return 2
        return compare_runs(Path(sys.argv[2]), Path(sys.argv[3]))
    ap = arguments(__doc__, ("off", "on"))
    ap.add_argument("--template", required=True, choices=sorted(CHAT_TEMPLATES))
    args = ap.parse_args()
    decode = load_decoder(Path(args.tokenizer))
    template = args.template
    return run(args, lambda question, thinking: build_prompt(question, thinking, template), template,
               decode, decode if template == "glm" else None)


def run(args, prompt_builder, chat_template: str, decode, decode_output) -> int:
    fixture = json.loads(Path(args.fixture).read_text())
    cases = [c for c in fixture["cases"] if c["id"] in COMPSEC_IDS]
    if len(cases) != 17:
        print(f"FATAL: expected 17 COMPSEC cases, found {len(cases)}", file=sys.stderr)
        return 2
    cases.sort(key=lambda c: c["id"])

    try:
        require_text_endpoint(args.endpoint, args.timeout)
    except RunError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2

    out = Path(args.out)
    (out / "responses").mkdir(parents=True, exist_ok=True)

    prompts = [prompt_builder(decode(c["prompt_token_ids"]), args.thinking) for c in cases]
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            replies = list(pool.map(lambda p: call(args.endpoint, p, args.max_tokens,
                                                   args.temperature, args.timeout), prompts))
    except RunError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    wall_s = time.monotonic() - started

    results = []
    for i, (c, prompt, r) in enumerate(zip(cases, prompts, replies), 1):
        payload = r["payload"]
        tokens = payload.get("tokens") or []
        choices = payload.get("choices") or [{}]
        text = choices[0].get("text")
        if text is None and decode_output is not None:
            text = decode_output(tokens)
        if text is None:
            raise SystemExit(f"{c['id']}: the endpoint returned no completion text; "
                             "tokens of a non-fixture vocabulary cannot be decoded here")
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
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "generated_token_count": len(tokens),
            "status": payload.get("status"),
        }
        results.append(rec)
        resp = {
            "id": c["id"],
            "request": {
                "endpoint": args.endpoint + "/v1/completions",
                "fixture_prompt_token_count": len(c["prompt_token_ids"]),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "thinking": args.thinking,
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
        "format": SUMMARY_FORMATS[chat_template][0],
        "generated_at": time.strftime("%FT%T%z"),
        "endpoint": args.endpoint,
        "fixture": Path(args.fixture).name,
        "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
        "wall_s": round(wall_s, 2),
        "parameters": {"max_tokens": args.max_tokens,
                       "temperature": args.temperature,
                       "chat_template": chat_template, "thinking": args.thinking,
                       "concurrency": args.concurrency,
                       "pass_threshold": args.pass_threshold},
        "grading_rule": ("qualification/ds4_eval/compare_runs.py: last Answer: line after "
                         "</think>; PASS iff its line set is a non-empty subset of the expected lines"),
        "completed": len(results),
        "passed": passed_n,
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    cases_out = {
        "format": SUMMARY_FORMATS[chat_template][1],
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
    return 0 if passed_n >= args.pass_threshold else 1



if __name__ == "__main__":
    sys.exit(main())
