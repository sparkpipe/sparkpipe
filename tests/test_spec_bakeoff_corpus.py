#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import spec_bakeoff_corpus as corpus  # noqa: E402

FAKE_TOKENIZER = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
text = open(args[args.index('--prompt-file') + 1], encoding='utf-8').read()
for byte in text.encode('utf-8'):
    print(byte + 1)
"""


def refused(argv) -> bool:
    try:
        corpus.main(argv)
    except SystemExit as error:
        return error.code not in (0, None)
    except (ValueError, RuntimeError):
        return True
    return False


def main() -> int:
    for path in sorted(corpus.SOURCES.glob("*.json")):
        data = json.loads(path.read_text())
        assert data["class"] == path.stem and path.stem in corpus.CLASSES, path
        if path.stem == "chat":
            assert len(data["conversations"]) == 4 and all(len(item["turns"]) == 3 for item in data["conversations"])
        elif path.stem == "long":
            assert sorted(item["target_tokens"] for item in data["prompts"]) == [8192, 32768]
        else:
            assert len(data["prompts"]) >= 3
    assert sorted(path.stem for path in corpus.SOURCES.glob("*.json")) == sorted(corpus.CLASSES)
    template = corpus.TEMPLATES["glmflash"]
    assert corpus.render(template, [("hi", None)], False) == "[gMASK]<sop><|user|>\nhi<|assistant|>\n<think></think>\n"
    assert corpus.render(template, [("hi", None)], True).endswith("<|assistant|>\n<think>")
    assert corpus.render(template, [("a", "b"), ("c", None)], False) == "[gMASK]<sop><|user|>\na<|assistant|>\n<think></think>\nb<|user|>\nc<|assistant|>\n<think></think>\n"
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        fake = directory / "fake_tokenize.py"
        fake.write_text(FAKE_TOKENIZER)
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        tokenizer_json = directory / "tokenizer.json"
        tokenizer_json.write_text("{}")
        out = directory / "glmflash"
        classes = "prose,code,repetitive,chat,thinking,tool_json,chinese"
        assert corpus.main(["build", "--model", "glmflash", "--classes", classes, "--tokenize-command", f"{sys.executable} {fake}", "--tokenizer-json", str(tokenizer_json), "--out", str(out)]) == 0
        built = json.loads((out / "prose.json").read_text())
        assert built["prompts"][0]["prompt_token_ids"][0] == ord("[") + 1 and built["prompts"][3]["output_tokens"] == 2048 and built["prompts"][0]["output_tokens"] == 512
        assert built["tokenizer_sha256"] == corpus.file_sha256(tokenizer_json)
        chat = json.loads((out / "chat.json").read_text())
        assert chat["prompts"][0]["follow_up_turns"] and chat["prompts"][0]["prompt_text"].endswith("<think></think>\n")
        thinking = json.loads((out / "thinking.json").read_text())
        assert thinking["thinking"] and thinking["prompts"][0]["prompt_text"].endswith("<think>")
        chinese = json.loads((out / "chinese.json").read_text())
        assert len(chinese["prompts"][0]["prompt_token_ids"]) > len(chinese["prompts"][0]["text"])
        manifest = json.loads((out / "MANIFEST.json").read_text())
        assert set(manifest["classes"]) == set(classes.split(","))
        assert corpus.main(["check", "--corpus", str(out)]) == 0
        long_source = {"class": "long", "output_tokens": 256, "prompts": [
            {"generator": "document_qa", "seed": 7, "target_tokens": 6000, "question": "Which officer is named in section {section} with code {code}?"},
            {"generator": "code_edit", "seed": 11, "target_tokens": 4000, "instruction": "Add metric_mean."}]}
        sources = directory / "sources"
        sources.mkdir()
        (sources / "long.json").write_text(json.dumps(long_source))
        assert corpus.main(["build", "--model", "glmflash", "--classes", "long", "--sources", str(sources), "--tokenize-command", f"{sys.executable} {fake}", "--tokenizer-json", str(tokenizer_json), "--out", str(out)]) == 0
        long_built = json.loads((out / "long.json").read_text())
        assert abs(long_built["prompts"][0]["prompt_tokens"] - 6000) <= 600 and abs(long_built["prompts"][1]["prompt_tokens"] - 4000) <= 400
        assert long_built["prompts"][0]["facts"] and "RC-07-" in long_built["prompts"][0]["prompt_text"] and "def metric_11_0" in long_built["prompts"][1]["prompt_text"]
        prose = json.loads((out / "prose.json").read_text())
        prose["prompts"][1]["prompt_token_ids"] = []
        (out / "prose.json").write_text(json.dumps(prose))
        assert refused(["check", "--corpus", str(out)])
        prose["prompts"][1]["prompt_token_ids"] = [5, 6, 7]
        (out / "prose.json").write_text(json.dumps(prose))
        assert refused(["check", "--corpus", str(out)])
        assert refused(["build", "--model", "k3", "--classes", "prose", "--tokenize-command", f"{sys.executable} {fake}", "--tokenizer-json", str(tokenizer_json), "--out", str(out)])
        assert refused(["build", "--model", "glmflash", "--classes", "poetry", "--tokenize-command", f"{sys.executable} {fake}", "--tokenizer-json", str(tokenizer_json), "--out", str(out)])
    print("PASS spec_bakeoff_corpus: eight content classes build with a tokenizer command and chat template, ids carry sha256, text without ids or a "
          "wrong digest is refused, long-context prompts reach their token targets, and unknown models or classes are refused")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main())
