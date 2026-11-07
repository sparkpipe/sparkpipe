#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "metal_stage_run"


def run(work, spec_tokens, prefill_rows):
    prompts = work / "prompts.json"
    spec = work / "spec.json"
    prompts.write_text(json.dumps({"prompts": [{"name": "gate", "prompt_token_ids": list(range(1, 200))}]}))
    spec.write_text(json.dumps({"prompts": [{"name": "short", "prompt_token_ids": list(range(1, spec_tokens + 1))}]}))
    result = subprocess.run([str(TOOL), "--model", "qwen3.8-27b", "--pack", str(work / "absent.pack"), "--prompts", str(prompts),
                             "--spec-prompts", str(spec), "--mode", "spec", "--out", str(work / "out"), "--prefill-rows", str(prefill_rows)],
                            capture_output=True, text=True, timeout=120)
    return result.returncode, result.stderr


def main():
    if not TOOL.exists():
        print("SKIP test_metal_stage_run_inputs (build/metal_stage_run is not built on this host)")
        return 0
    failures = 0
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        cases = [
            ("a 10-token first speculation prompt is refused before the model loads", 10, 1024, "it needs at least 64"),
            ("--prefill-rows 32 is refused in spec mode", 200, 32, "--prefill-rows must be at least 64"),
        ]
        for what, tokens, rows, message in cases:
            code, stderr = run(work, tokens, rows)
            ok = code != 0 and message in stderr
            print(("PASS " if ok else "FAIL ") + what + ("" if ok else f" (exit {code}, stderr {stderr.strip()[-300:]!r})"))
            failures += 0 if ok else 1
        code, stderr = run(work, 64, 64)
        ok = code != 0 and "it needs at least 64" not in stderr and "--prefill-rows must be at least" not in stderr
        print(("PASS " if ok else "FAIL ") + "a 64-token prompt with 64 prefill rows passes both checks and stops only at the absent pack" +
              ("" if ok else f" (exit {code}, stderr {stderr.strip()[-300:]!r})"))
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
