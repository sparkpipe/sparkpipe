#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "hy4_reference" / "hy4_compare_runs.py"
HIDDEN = 6144
LAYERS = 78
PROMPT = {"name": "p", "text": "x", "prompt_token_ids": [5, 6, 7], "new_tokens": 3, "capture_layers": [0, 77]}


def write_run(directory, generated, heads, streams, route_ids):
    os.makedirs(directory, exist_ok=True)
    lines = ["prompt 5 6 7", "generated " + " ".join(str(token) for token in generated)]
    for position, pairs in heads.items():
        lines.append(f"head {position} " + " ".join(f"{token}:{score:.6e}" for token, score in pairs))
    Path(directory, "p.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = 3 + len(generated)
    ids = np.zeros((total, LAYERS - 1, 8), dtype=np.int32) + np.arange(8, dtype=np.int32)
    ids[:, :, 0] = route_ids
    weights = np.full((total, LAYERS - 1, 8), 0.3, dtype=np.float32)
    with open(os.path.join(directory, "p.routes.bin"), "wb") as fh:
        fh.write(ids.tobytes())
        fh.write(weights.tobytes())
    streams.astype(np.float32).tofile(os.path.join(directory, "p.streams.bin"))


def run(prompts, reference, candidate, *extra):
    result = subprocess.run([sys.executable, str(TOOL), "--prompts", prompts, "--reference", reference, "--candidate", candidate, *extra], capture_output=True, text=True)
    return result.returncode, result.stdout


def main():
    rng = np.random.default_rng(7)
    failures = []
    with tempfile.TemporaryDirectory() as work:
        prompts = os.path.join(work, "prompts.json")
        Path(prompts).write_text(json.dumps({"prompts": [PROMPT]}), encoding="utf-8")
        streams = rng.standard_normal((2, 6, 4 * HIDDEN)).astype(np.float32)
        heads = {2: [(10, 20.0), (11, 19.9)], 3: [(12, 18.0), (13, 15.0)], 4: [(14, 17.0), (15, 16.95)]}
        write_run(os.path.join(work, "ref"), [10, 12, 14], heads, streams, 100)
        write_run(os.path.join(work, "same"), [10, 12, 14], heads, streams * 1.001, 100)
        code, out = run(prompts, os.path.join(work, "ref"), os.path.join(work, "same"))
        if code != 0 or "RESULT: PASS" not in out:
            failures.append("identical runs within the band must pass:\n" + out)
        noisy = streams.copy()
        noisy[1, 0, 17] += 5.0
        write_run(os.path.join(work, "noisy"), [10, 12, 14], heads, noisy, 100)
        code, out = run(prompts, os.path.join(work, "ref"), os.path.join(work, "noisy"))
        if code == 0 or "FAIL p layer 77" not in out:
            failures.append("a stream outside the band must fail naming its layer:\n" + out)
        write_run(os.path.join(work, "drift"), [10, 13, 14], heads, streams, 100)
        code, out = run(prompts, os.path.join(work, "ref"), os.path.join(work, "drift"))
        if code == 0 or "FAIL p tokens" not in out:
            failures.append("a different greedy token must fail:\n" + out)
        write_run(os.path.join(work, "routes"), [10, 12, 14], heads, streams, 101)
        code, out = run(prompts, os.path.join(work, "ref"), os.path.join(work, "routes"))
        if "NOTE p routes: 0/" not in out:
            failures.append("different expert sets must be counted:\n" + out)
        tie_heads = {2: [(11, 19.95), (10, 19.9)], 3: [(12, 18.0), (13, 15.0)], 4: [(14, 17.0), (15, 16.95)]}
        write_run(os.path.join(work, "tie"), [10, 12, 14], tie_heads, streams, 100)
        code, out = run(prompts, os.path.join(work, "ref"), os.path.join(work, "tie"), "--teacher-forced")
        if code != 0 or "TIE  p head 2" not in out or "2/3 equal, 1 reference near-ties" not in out:
            failures.append("a flipped near-tie under teacher forcing must pass as a tie:\n" + out)
        miss_heads = {2: [(10, 20.0), (11, 19.9)], 3: [(13, 18.0), (12, 17.0)], 4: [(14, 17.0), (15, 16.95)]}
        write_run(os.path.join(work, "miss"), [10, 12, 14], miss_heads, streams, 100)
        code, out = run(prompts, os.path.join(work, "ref"), os.path.join(work, "miss"), "--teacher-forced")
        if code == 0 or "MISS p head 3" not in out:
            failures.append("a flip at a 3.0 logit margin must fail under teacher forcing:\n" + out)
    for failure in failures:
        print("FAIL", failure)
    print("test_hy4_reference_compare:", "FAIL" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
