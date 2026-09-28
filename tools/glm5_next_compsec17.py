#!/usr/bin/env python3
"""glm5_next COMPSEC-17 quality gate: tools/compsec17.py with the GLM chat
template. Same arguments as compsec17.py without --template.

usage:
  glm5_next_compsec17.py --endpoint http://127.0.0.1:8433 --thinking off \
      --fixture qualification/ds4_eval/quality-fixtures-glm5.3-flash.json \
      --tokenizer <runtime>/tokenizer/tokenizer.json \
      --out qualification/ds4_eval/runs/glm5-next-tp16-<date>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compsec17
from compsec17 import grade, load_decoder

__all__ = ["build_prompt", "grade", "load_decoder", "main"]


def build_prompt(question: str, thinking: str) -> str:
    return compsec17.build_prompt(question, thinking, "glm")


def main() -> int:
    sys.argv[1:1] = ["--template", "glm"]
    return compsec17.main()


if __name__ == "__main__":
    sys.exit(main())
