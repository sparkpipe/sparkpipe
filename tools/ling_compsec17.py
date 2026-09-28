#!/usr/bin/env python3
import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("glm5_next_compsec17", TOOLS / "glm5_next_compsec17.py")
compsec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compsec)

LING_SYSTEM = "<role>SYSTEM</role>detailed thinking {mode}<|role_end|>"
LING_USER = "<role>HUMAN</role>{question}<|role_end|>"
LING_ASSISTANT = {"off": "<role>ASSISTANT</role>\n<think></think>", "on": "<role>ASSISTANT</role>\n<think>"}


def build_prompt(question: str, thinking: str) -> str:
    return LING_SYSTEM.format(mode=thinking) + LING_USER.format(question=question) + LING_ASSISTANT[thinking]


def load_output_decoder(tokenizer_path: Path):
    document = json.loads(tokenizer_path.read_text())
    byte_decoder = compsec.load_decoder(tokenizer_path)
    added = {int(token["id"]): token["content"] for token in document.get("added_tokens", [])}

    def decode(ids) -> str:
        pieces, run = [], []
        for token in ids:
            if int(token) in added:
                if run:
                    pieces.append(byte_decoder(run))
                    run = []
                pieces.append(added[int(token)])
            else:
                run.append(int(token))
        if run:
            pieces.append(byte_decoder(run))
        return "".join(pieces)

    return decode


def main() -> int:
    parser = compsec.arguments("Ling-3.0 COMPSEC-17 quality gate: the glm5_next_compsec17 protocol with the Ling chat template", LING_ASSISTANT)
    parser.add_argument("--output-tokenizer", required=True, help="Ling tokenizer.json, used when the endpoint returns token ids without text")
    args = parser.parse_args()
    question_decoder = compsec.load_decoder(Path(args.tokenizer))
    return compsec.run(args, build_prompt, "ling", question_decoder, load_output_decoder(Path(args.output_tokenizer)))


if __name__ == "__main__":
    sys.exit(main())
