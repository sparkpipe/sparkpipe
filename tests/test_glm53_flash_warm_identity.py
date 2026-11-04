#!/usr/bin/env python3
import json
import pathlib
import re
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = ROOT / "modules" / "glm53_flash_tp16_resident_decode_stage" / "include" / "sparkpipe" / "spark_glm53_flash_tp16_stage_model.h"


def main():
    failures = []
    text = HEADER.read_text()
    model = re.search(r'WeightdModel\(void\) \{ return "([^"]+)"; \}', text)
    revision = re.search(r'WeightdRevision\(void\) \{ return "([^"]+)"; \}', text)
    if not model or not revision:
        print(f"FAIL {HEADER.name} no longer spells WeightdModel and WeightdRevision as string returns; update this test with it")
        return 1
    swap = {entry["id"]: entry for entry in json.loads((ROOT / "deployment" / "hub" / "model_swap.json").read_text())["models"]}
    warm = swap["glm-5.3-flash"].get("node_warm", "")
    words = shlex.split(re.sub(r"\$\([^)]*\)", "SUBSTITUTION", warm.split("&&")[-1].split(">", 1)[0]))
    if not words or not words[0].endswith("weightd_warm"):
        failures.append(f"the glm-5.3-flash swap entry has no weightd_warm command: {warm!r}")
    else:
        positional = [word for word in words[1:] if not word.startswith("--")]
        if len(positional) < 5 or positional[3] != revision.group(1):
            failures.append(f"the glm-5.3-flash warm passes revision {positional[3] if len(positional) > 3 else None!r}, but the stage registers its expert "
                            f"arena with weightd as {model.group(1)} revision {revision.group(1)!r} ({HEADER.name}); a shared warm with another revision "
                            f"finds no arena and the swap fails")
        if "--model" in words or "--family" in words:
            failures.append("the glm-5.3-flash warm must take weightd_warm's default module tag, which is the stage's glm5_next_stage")
        if model.group(1) != "glm5_next_stage":
            failures.append(f"the stage registers as {model.group(1)}, not weightd_warm's default glm5_next_stage; the warm needs --model")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print(f"PASS glm-5.3-flash warm identity: the swap controller's warm asks weightd for the {model.group(1)} arena under the revision the stage registers ({revision.group(1)[:8]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
