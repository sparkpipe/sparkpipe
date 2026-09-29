#!/usr/bin/env python3
"""glm5_next COMPSEC-17 quality gate: tools/compsec17.py with the GLM chat
template. Same arguments as compsec17.py without --template, including
--compare REFERENCE CANDIDATE.

usage:
  glm5_next_compsec17.py --endpoint http://127.0.0.1:8433 --thinking off \
      --fixture qualification/ds4_eval/quality-fixtures-glm5.3-flash.json \
      --tokenizer <runtime>/tokenizer/tokenizer.json \
      --out qualification/ds4_eval/runs/glm5-next-tp16-<date>
  glm5_next_compsec17.py --compare RUN_SEQUENTIAL RUN_CONCURRENT
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compsec17
from compsec17 import (COMPSEC_IDS, RunError, arguments, call, compare_runs, first_difference, grade,
                       load_decoder, load_run_tokens, require_text_endpoint, run)

__all__ = ["COMPSEC_IDS", "RunError", "arguments", "build_prompt", "call",
           "compare_runs", "first_difference", "grade", "load_decoder", "load_run_tokens", "main",
           "require_text_endpoint", "run"]


def build_prompt(question: str, thinking: str) -> str:
    return compsec17.build_prompt(question, thinking, "glm")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        if len(sys.argv) != 4:
            print("usage: glm5_next_compsec17.py --compare REFERENCE_RUN CANDIDATE_RUN", file=sys.stderr)
            return 2
        return compare_runs(Path(sys.argv[2]), Path(sys.argv[3]))
    args = arguments(__doc__, compsec17.CHAT_TEMPLATES["glm"]).parse_args()
    decode = load_decoder(Path(args.tokenizer))
    return run(args, build_prompt, "glm", decode, decode)


if __name__ == "__main__":
    sys.exit(main())
