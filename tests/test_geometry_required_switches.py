#!/usr/bin/env python3
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    ("model-families/glm52/source/glm52_stage.cuh", "GlmDefaults"): ("kExpertCodec", "kLinearCodec", "kIndexPairing"),
    ("model-families/ling/include/sparkpipe/spark_ling_family.h", "LingDefaults"): ("kAttention", "kNormEpsilon", "kKdaDoubledBeta"),
    ("model-families/gemma4/include/sparkpipe/spark_gemma4_family.h", "GemmaDefaults"): ("kNormEpsilon", "kPostEpsilon", "kQkNormWeighted", "kQueryMultiply",
        "kValueNorm", "kAttentionScale", "kFullRope", "kGeluMlp", "kLayerScalar", "kPreNorm", "kSigmoidRouter", "kRoutedScale", "kRouterFloat"),
    ("model-families/qwen38/include/sparkpipe/spark_qwen38_family.h", "QwenHybridDefaults"): ("kGdnSigmoidGate",),
}
SOURCES = (sorted(ROOT.glob("modules/*/source/*.cu")) + sorted(ROOT.glob("modules/*/include/sparkpipe/*.h")) +
           sorted(ROOT.glob("model-families/*/source/*.cuh")))


def block(text, header):
    match = re.search(re.escape(header) + r"\n\{\n(.*?)\n\};", text, re.S)
    return match.group(1) if match else None


def main():
    failures, geometries = [], 0
    for (path, defaults), names in REQUIRED.items():
        body = block((ROOT / path).read_text(), f"struct {defaults}")
        if body is None:
            failures.append(f"{path}: struct {defaults} not found")
            continue
        for name in names:
            if re.search(rf"\b{name}\s*=", body):
                failures.append(f"{defaults} defaults the arithmetic switch {name}; a geometry must choose it")
        for source in SOURCES:
            text = source.read_text()
            for match in re.finditer(rf"struct (\w+) : {defaults}\n\{{\n(.*?)\n\}};", text, re.S):
                geometries += 1
                missing = [name for name in names if not re.search(rf"\b{name}\s*=", match.group(2))]
                if missing:
                    failures.append(f"{source.relative_to(ROOT)}: {match.group(1)} does not define {', '.join(missing)}")
    if geometries < 12:
        failures.append(f"only {geometries} geometries found; the scan no longer sees the stage geometries")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print(f"PASS geometry switches: the four stage defaults structs default no arithmetic switch, and all {geometries} geometries choose every one")


if __name__ == "__main__":
    sys.exit(main())
