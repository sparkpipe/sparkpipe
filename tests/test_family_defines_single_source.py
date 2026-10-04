#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFINE = re.compile(r"^[ \t]*#define[ \t]+([A-Z][A-Z0-9_]*)[ \t]+([^\\\n]+)$", re.M)
LITERAL = re.compile(r"^(?:[-+]?(?:0[xX][0-9a-fA-F]+|[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?)[uUlLfF]*|\"[^\"]*\")$")


def literals(path):
    found = {}
    for match in DEFINE.finditer(path.read_text()):
        value = match.group(2).strip()
        if LITERAL.match(value):
            found[match.group(1)] = value
    return found


def main():
    failures = []
    families = 0
    for llm in sorted(ROOT.glob("model-families/*/include/sparkpipe/llm_defines.h")):
        family = llm.parents[2].name
        model = llm.parent / f"spark_{family}_model.h"
        if not model.exists():
            continue
        families += 1
        prefix = f"SPARK_{family.upper()}_MODEL_"
        llm_values = literals(llm)
        model_values = literals(model)
        for name in sorted(set(llm_values) & set(model_values)):
            failures.append(f"{family}: {name} is a literal in both {model.name} and llm_defines.h")
        for name, value in sorted(model_values.items()):
            if name.startswith(prefix) and "SPARK_LLM_" + name[len(prefix):] in llm_values:
                failures.append(f"{family}: {name} = {value} repeats SPARK_LLM_{name[len(prefix):]} instead of naming it")
        for name, value in sorted(llm_values.items()):
            if name.startswith("SPARK_LLM_") and prefix + name[len("SPARK_LLM_"):] in model_values:
                failures.append(f"{family}: {name} = {value} repeats {prefix}{name[len('SPARK_LLM_'):]} instead of naming it")
    if families == 0:
        failures.append("no family has both llm_defines.h and a model header")
    for failure in failures:
        print("FAIL " + failure, file=sys.stderr)
    if failures:
        return 1
    print(f"PASS family defines single source: {families} families keep each model value as one literal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
