import json
import subprocess
import sys
import tempfile
from pathlib import Path

service = Path(sys.argv[1])
data = json.loads((service / "check" / "corpus.json").read_text())
suffix = json.loads((service / "rendered" / "model_resident.json").read_text())["tokenizer"].get("text_prompt_suffix_token_ids", [])
prefix = json.loads((service / "rendered" / "model_resident.json").read_text())["tokenizer"].get("text_prompt_prefix_token_ids", [])
wrong = []
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "text"
    for index, (text, ids) in enumerate(zip(data["texts"], data["ids"])):
        path.write_text(text, encoding="utf-8")
        out = subprocess.run([str(service / "fw" / "sparkpipe_tokenize_prompt"), "--tokenizer-compiled", str(service / "tokenizer.compiled"), "--text-file", str(path)],
                             capture_output=True, text=True)
        ours = prefix + [int(line) for line in out.stdout.split()] + suffix
        if out.returncode != 0 or ours != ids:
            wrong.append({"text": index, "tokens": len(ids), "ours": len(ours), "rc": out.returncode,
                          "first_difference": next((i for i, (a, b) in enumerate(zip(ours, ids)) if a != b), min(len(ours), len(ids)))})
print(json.dumps({"step": "tokenizer", "texts": len(data["texts"]), "prefix": prefix, "suffix": suffix, "different": len(wrong), "examples": wrong[:6]}))
