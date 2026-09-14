#!/usr/bin/env python3
"""llm_defines/spark_driver_defines module gate (mod-infra-mid wave, 2026-09-13).

The single-source law: every driver value lives in the family llm_defines.h
and every derived key is produced once by spark_driver_defines.h. This gate

  1. compiles a vector TU against ling's llm_defines.h whose _Static_asserts
     pin the derived keys to hand-derived values (vectors FROM the same
     single source the driver builds from), and
  2. runs negative controls: each flipped or deleted parameter must FAIL
     compilation with an error naming that parameter (the checklist is the
     compiler, the K3A oracle pattern at the defines layer).
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON_INCLUDE = ROOT / "model-families/common/include"
LING_INCLUDE = ROOT / "model-families/ling/include"
LLM_DEFINES = LING_INCLUDE / "sparkpipe/llm_defines.h"

VECTOR_TU = r"""
#include "sparkpipe/llm_defines.h"

_Static_assert(SPARK_LLM_HIDDEN_DIMENSION == 2560u, "vector hidden");
_Static_assert(SPARK_LLM_LAYER_COUNT == 42u, "vector layers");
_Static_assert(SPARK_LLM_OUTPUT_VOCAB_COUNT == 157184u, "vector vocab");
_Static_assert(SPARK_LLM_MAXIMUM_CONTEXT_TOKENS == 262144u, "vector context");
_Static_assert(SPARK_LLM_WEIGHT_LAYER_COUNT == 43u, "vector weight layers");
_Static_assert(SPARK_LLM_MLA_QK_HEAD_DIMENSION == 192u, "vector qk head dim");
_Static_assert(SPARK_LLM_MLA_QUERY_DIMENSION == 32u * 192u, "vector query dim");
_Static_assert(SPARK_LLM_MLA_KV_A_DIMENSION == 576u, "vector kv a dim");
_Static_assert(SPARK_LLM_MLA_KV_B_DIMENSION == 32u * 256u, "vector kv b dim");
_Static_assert(SPARK_LLM_KV_SLOT_BYTES == 576u * 16u / 8u, "vector kv slot bytes");
_Static_assert(SPARK_LLM_KDA_QK_DIM == 32u * 128u, "vector kda qk dim");
_Static_assert(SPARK_LLM_KDA_VALUE_DIM == 32u * 128u, "vector kda value dim");
_Static_assert(SPARK_LLM_KDA_STATE_BYTES_PER_LAYER == 32u * 128u * 128u * 4u, "vector kda state bytes");
_Static_assert(SPARK_LLM_KDA_CONV_WINDOW_BYTES_PER_LAYER == 3u * 4096u * 4u * 2u, "vector kda conv window bytes");
_Static_assert(SPARK_LLM_MOE_ROUTED_GATE_UP_DIMENSION == 768u * 2u, "vector gate up dim");
_Static_assert(SPARK_LLM_ROUTED_LAYERS == 40u, "vector routed layers");
_Static_assert(SPARK_LLM_LAYER_IS_FULL(5u) && SPARK_LLM_LAYER_IS_FULL(41u), "vector full phases");
_Static_assert(SPARK_LLM_LAYER_IS_LINEAR(0u) && SPARK_LLM_LAYER_IS_LINEAR(4u), "vector linear phases");
_Static_assert(SPARK_LLM_ATTENTION_KIND == SPARK_LLM_ATTENTION_KIND_LATENT_LINEAR, "vector kind");

int main(void) { return 0; }
"""

NEGATIVE_CONTROLS = [
    ("kda_conv_kernel_deleted", r"#define SPARK_LLM_KDA_CONV_KERNEL .*\n", None,
     "SPARK_LLM_KDA_CONV_KERNEL"),
    ("latent_deleted", r"#define SPARK_LLM_MLA_LATENT_DIMENSION .*\n", None,
     "SPARK_LLM_MLA_LATENT_DIMENSION"),
    ("kv_page_slots_deleted", r"#define SPARK_LLM_KV_PAGE_SLOTS .*\n", None,
     "SPARK_LLM_KV_PAGE_SLOTS"),
    ("head_count_flip", r"#define SPARK_LLM_MLA_HEAD_COUNT .*\n",
     "#define SPARK_LLM_MLA_HEAD_COUNT 33u\n", "divide by 16"),
    ("expert_count_flip", r"#define SPARK_LLM_MOE_EXPERT_COUNT .*\n",
     "#define SPARK_LLM_MOE_EXPERT_COUNT 500u\n", "divide into 16 ranks"),
    ("latent_layout_flip", r"#define SPARK_LLM_MLA_LATENT_DIMENSION .*\n",
     "#define SPARK_LLM_MLA_LATENT_DIMENSION 640u\n", "latent cache slot layout"),
    ("router_subset_flip", r"#define SPARK_LLM_MOE_ROUTER_TOP_GROUPS .*\n",
     "#define SPARK_LLM_MOE_ROUTER_TOP_GROUPS 9u\n", "cannot exceed"),
    ("period_tiling_flip", r"#define SPARK_LLM_ATTENTION_PERIOD .*\n",
     "#define SPARK_LLM_ATTENTION_PERIOD 5u\n", "tile the stack exactly"),
    ("kind_flip_to_gqa", r"#define SPARK_LLM_ATTENTION_KIND .*\n",
     "#define SPARK_LLM_ATTENTION_KIND SPARK_LLM_ATTENTION_KIND_GQA\n",
     "SPARK_LLM_FULL_QUERY_HEAD_COUNT"),
]


def compile_tu(source, include_dir):
    with tempfile.TemporaryDirectory() as workdir:
        tu = Path(workdir) / "tu.c"
        tu.write_text(source)
        command = [
            "cc", "-std=c11", "-Wall", "-Werror", "-fsyntax-only",
            f"-I{include_dir}", f"-I{COMMON_INCLUDE}",
            f"-DSPARK_BATCH_BUCKET=1024u",
            str(tu),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        return result.returncode, result.stderr


def transformed_defines(pattern, replacement):
    text = LLM_DEFINES.read_text()
    if not re.search(pattern, text):
        return None, f"pattern absent: {pattern!r}"
    substitute = replacement if replacement is not None else ""
    return re.sub(pattern, substitute, text, count=1), None


def main():
    failures = []
    print("llm_defines module gate:")
    with tempfile.TemporaryDirectory() as workdir:
        include_dir = Path(workdir)
        (include_dir / "sparkpipe").mkdir(parents=True)
        (include_dir / "sparkpipe" / "llm_defines.h").write_text(LLM_DEFINES.read_text())
        code, stderr = compile_tu(VECTOR_TU, include_dir)
        if code != 0:
            failures.append(f"vector TU failed against unmodified llm_defines.h:\n{stderr}")
        else:
            print("  PASS vectors: 19 derived-key assertions hold against ling llm_defines.h")
        for name, pattern, replacement, expected_text in NEGATIVE_CONTROLS:
            mutated, error = transformed_defines(pattern, replacement)
            if error:
                failures.append(f"{name}: fixture construction failed: {error}")
                continue
            (include_dir / "sparkpipe" / "llm_defines.h").write_text(mutated)
            code, stderr = compile_tu(VECTOR_TU, include_dir)
            (include_dir / "sparkpipe" / "llm_defines.h").unlink()
            if code == 0:
                failures.append(f"{name}: flipped/deleted parameter still compiles — the gate has no teeth")
            elif expected_text not in stderr:
                failures.append(
                    f"{name}: compile failed but error does not name"
                    f" '{expected_text}':\n{stderr}"
                )
            else:
                print(f"  PASS negative control {name}: rejected, error names '{expected_text}'")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"FAIL ({len(failures)}) llm_defines gate violations")
        return 1
    print("PASS single-source defines: vectors hold, negative controls bite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
