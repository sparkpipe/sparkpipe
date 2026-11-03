#!/usr/bin/env python3
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("chat_memory_qualify", ROOT / "tools" / "chat_memory_qualify.py")
QUALIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFY)


def chat(arm, prompts, jobs, correct=5):
    rows = [{"kind": "turn" if index < len(prompts) - 5 else "probe", "prompt_tokens": prompt, "cached_tokens": max(0, prompt - 300)}
            for index, prompt in enumerate(prompts)]
    return {"arm": arm, "chat": arm, "rows": rows, "probes": [{"correct": index < correct} for index in range(5)],
            "stale": {"turns": 0, "background": 0}, "jobs": jobs}


def run(memory_prompts, memory_jobs):
    full = [1000 * (index + 1) for index in range(22)]
    results = [chat("memory", memory_prompts, memory_jobs), chat("full", full, [])]
    checks, _ = QUALIFY.qualify(results, "memory", "full", 512, 0.9)
    control, _ = QUALIFY.qualify(results, "full", "full", 512, 0.9)
    checks["memory_off_control_fails"] = not all(control.values())
    return checks


def main():
    failures = []
    curated = [{"kind": "curator", "outcome": "done", "prompt_tokens": 9000, "cached_tokens": 8800, "seconds": 3.0},
               {"kind": "prewarm", "outcome": "done", "prompt_tokens": 8000, "cached_tokens": 7900, "seconds": 0.5}]
    compact = [min(1000 * (index + 1), 12000) for index in range(22)]
    good = run(compact, curated)
    if not all(good.values()):
        failures.append(f"a memory arm that curated and kept its prompts small must qualify: {good}")
    never = run([1000 * (index + 1) for index in range(22)], [])
    for check in ("memory_curated_every_chat", "memory_prompt_clearly_smaller"):
        if never[check]:
            failures.append(f"a memory arm whose memory never engaged must fail {check}")
    if all(never.values()):
        failures.append("a memory arm whose memory never engaged must not qualify")
    prewarm_only = run(compact, curated[1:])
    if prewarm_only["memory_curated_every_chat"]:
        failures.append("prewarm jobs alone are not a curated memory")
    if not good["memory_off_control_fails"]:
        failures.append("the memory-off control must fail the same checks")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS chat memory qualification: a curated, compact memory arm qualifies; one whose memory never engaged, or that only prewarmed, does not; the memory-off control fails the same checks")


if __name__ == "__main__":
    main()
