#!/usr/bin/env python3
"""The 92-case ds4-eval suite against any model served through the chat frontend.

A deployment qualification in three phases, run in order:
  warm    every case's prompt once with max_tokens 1, one at a time, so the
          KV prefix cache holds all 92 prompts before scoring starts
  serial  every case with the full completion budget, one at a time
  batch   all cases submitted at once (--concurrency, default 92); the
          engine's continuous batching admits them as lanes free up

Requests go to the chat frontend's OpenAI chat endpoint with the model id,
so each model is prompted through its own chat template, exactly as the UI
prompts it. The cases (rendered prompts, answers, system prompt) come from
the retained K3 API run; answers are extracted and matched with
qualification/ds4_eval/compare_runs.py. Each phase writes one response file
per case plus summary.json; the run directory gets REPORT.md comparing the
phases (score, cached prompt tokens, wall time, completion tok/s, and how
many serial and batch answers agree).

usage:
  ds4_eval92.py --endpoint http://127.0.0.1:8433 --model qwen3.8-27b \\
      --max-tokens 16000 --temperature 0 --out qualification/ds4_eval/runs/<run-id>
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qualification" / "ds4_eval"))
from compare_runs import answer_matches, extract_answer, family_name  # noqa: E402

DEFAULT_CASES = ROOT / "qualification" / "ds4_eval" / "runs" / "kimi-k3-api-20260728" / "cases.json"
PHASES = ("warm", "serial", "batch")


def request(endpoint: str, body: dict, timeout: int) -> tuple[dict, float]:
    data = json.dumps(body).encode()
    started = time.monotonic()
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        payload = {"error": {"status": error.code, "body": error.read().decode(errors="replace")[:2000]}}
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        payload = {"error": {"status": None, "body": repr(error)}}
    return payload, time.monotonic() - started


def messages(cases: dict, case: dict) -> list:
    out = []
    if cases.get("system_prompt"):
        out.append({"role": "system", "content": cases["system_prompt"]})
    out.append({"role": "user", "content": case["rendered_prompt"]})
    return out


def body_for(args, cases: dict, case: dict, max_tokens: int) -> dict:
    body = {"model": args.model, "messages": messages(cases, case), "max_tokens": max_tokens,
            "temperature": args.temperature}
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    return body


def score(case: dict, payload: dict, elapsed: float) -> dict:
    record = {"id": case["id"], "index": case["index"], "source": case["source"],
              "family": family_name(case["source"]), "expected": case["answer"], "elapsed_s": round(elapsed, 3)}
    if "error" in payload or not payload.get("choices"):
        record.update(status="error", passed=False, extracted="", error=payload.get("error"))
        return record
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    extracted, _ = extract_answer(case, content, reasoning)
    usage = payload.get("usage") or {}
    record.update(status="ok", finish_reason=choice.get("finish_reason"), extracted=extracted,
                  passed=bool(answer_matches(case, extracted)), prompt_tokens=usage.get("prompt_tokens"),
                  completion_tokens=usage.get("completion_tokens"),
                  cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    return record


def run_phase(args, cases: dict, phase: str, out: Path) -> dict:
    responses = out / phase / "responses"
    responses.mkdir(parents=True, exist_ok=True)
    selected = cases["cases"]
    max_tokens = 1 if phase == "warm" else args.max_tokens
    workers = args.concurrency if phase == "batch" else 1

    def one(case: dict) -> dict:
        payload, elapsed = request(args.endpoint, body_for(args, cases, case, max_tokens), args.timeout)
        (responses / f"{case['index']:03d}-{case['id']}.json").write_text(
            json.dumps({"id": case["id"], "phase": phase, "elapsed_s": elapsed, "response": payload}, indent=1) + "\n")
        record = score(case, payload, elapsed)
        print(json.dumps({"phase": phase, "index": case["index"], "status": record["status"],
                          "passed": record["passed"], "extracted": record["extracted"][:40],
                          "completion_tokens": record.get("completion_tokens"),
                          "cached_tokens": record.get("cached_tokens"), "s": record["elapsed_s"]}), flush=True)
        return record

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(one, selected))
    wall = time.monotonic() - started
    by_family = {}
    for record in records:
        family = by_family.setdefault(record["family"], {"completed": 0, "passed": 0})
        family["completed"] += record["status"] == "ok"
        family["passed"] += record["passed"]
    completion = sum(record.get("completion_tokens") or 0 for record in records)
    summary = {"format": "sparkpipe-ds4-eval92-phase-v1", "phase": phase, "model": args.model,
               "endpoint": args.endpoint, "max_tokens": max_tokens, "temperature": args.temperature,
               "reasoning_effort": args.reasoning_effort, "concurrency": workers, "wall_s": round(wall, 1),
               "completed": sum(record["status"] == "ok" for record in records),
               "errors": sum(record["status"] != "ok" for record in records),
               "passed": sum(record["passed"] for record in records), "by_family": by_family,
               "prompt_tokens": sum(record.get("prompt_tokens") or 0 for record in records),
               "cached_tokens": sum(record.get("cached_tokens") or 0 for record in records),
               "completion_tokens": completion,
               "completion_tok_s": round(completion / wall, 2) if wall > 0 else None,
               "finish_reasons": {reason: sum(record.get("finish_reason") == reason for record in records)
                                  for reason in sorted({str(record.get("finish_reason")) for record in records})},
               "results": records}
    (out / phase / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def report(out: Path, args, cases_path: Path, summaries: dict) -> None:
    lines = [f"# ds4-eval 92 — {args.model}", "",
             f"- Endpoint `{args.endpoint}`, temperature `{args.temperature}`, max_tokens `{args.max_tokens}`,"
             f" reasoning_effort `{args.reasoning_effort or 'model default'}`",
             f"- Cases `{cases_path.relative_to(ROOT) if cases_path.is_relative_to(ROOT) else cases_path}`"
             f" sha256 `{hashlib.sha256(cases_path.read_bytes()).hexdigest()}`", "",
             "| Phase | Concurrency | Passed | Errors | Prompt tok | Cached tok | Completion tok | Wall s | Completion tok/s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for phase, s in summaries.items():
        lines.append(f"| {phase} | {s['concurrency']} | {s['passed']}/92 | {s['errors']} | {s['prompt_tokens']} |"
                     f" {s['cached_tokens']} | {s['completion_tokens']} | {s['wall_s']} | {s['completion_tok_s']} |")
    for phase in ("serial", "batch"):
        if phase in summaries:
            lines += ["", f"## {phase} by family", "", "| Family | Passed | Completed |", "|---|---:|---:|"]
            for family, value in sorted(summaries[phase]["by_family"].items()):
                lines.append(f"| {family} | {value['passed']} | {value['completed']} |")
    if "serial" in summaries and "batch" in summaries:
        serial = {r["id"]: r for r in summaries["serial"]["results"]}
        batch = {r["id"]: r for r in summaries["batch"]["results"]}
        same = sum(serial[i]["extracted"] == batch[i]["extracted"] for i in serial if i in batch)
        lines += ["", f"Serial and batch extracted answers agree on {same}/92 cases."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high"), default=None)
    parser.add_argument("--concurrency", type=int, default=92)
    parser.add_argument("--phases", default=",".join(PHASES))
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    phases = [phase for phase in args.phases.split(",") if phase]
    if not phases or any(phase not in PHASES for phase in phases):
        raise SystemExit(f"--phases is a comma list of {', '.join(PHASES)}")
    cases = json.loads(args.cases.read_text())
    if len(cases.get("cases", [])) != 92:
        raise SystemExit(f"{args.cases} holds {len(cases.get('cases', []))} cases, not the 92-case suite")
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for phase in phases:
        summaries[phase] = run_phase(args, cases, phase, args.out)
        print(json.dumps({k: v for k, v in summaries[phase].items() if k != "results"}), flush=True)
        report(args.out, args, args.cases, summaries)
    return 0 if all(s["errors"] == 0 for s in summaries.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
