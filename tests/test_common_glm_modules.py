#!/usr/bin/env python3
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFINES = ROOT / "model-families/glm52/include/sparkpipe/llm_defines.h"
INCLUDES = (
    "-I", str(ROOT),
    "-I", str(ROOT / "include"),
    "-I", str(ROOT / "model-families/glm52/include"),
    "-I", str(ROOT / "model-families/common/include"),
    "-I", str(ROOT / "tests/cuda_stub"),
)
CFLAGS = ("-std=c11", "-Wall", "-Wextra", "-Werror", "-O1")


def parse_defines(path):
    values = {}
    joined = re.sub(r"\\\n\s*", " ", path.read_text())
    for line in joined.splitlines():
        match = re.match(r"^#define SPARK_LLM_(\w+) (.+)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def resolve(name, values, depth=0):
    if depth > 16:
        raise AssertionError(f"llm_defines.h define cycle at LLM_{name}")
    text = values[name]
    if re.fullmatch(r"\d+u", text):
        return int(text[:-1])
    def substitute(match):
        key = match.group(1)
        if key not in values:
            raise AssertionError(f"llm_defines.h vector surface references missing LLM_{key}")
        return f"({resolve(key, values, depth + 1)})"
    expression = re.sub(r"\bSPARK_LLM_(\w+)\b", substitute, text)
    expression = expression.replace("u", "")
    if not re.fullmatch(r"[0-9()+\-*/ ]+", expression):
        raise AssertionError(f"LLM_{name} is not an integer vector: {text}")
    return eval(expression)


def vectors(values):
    required = (
        "LAYER_COUNT", "FIRST_ROUTED_LAYER", "MOE_INTERMEDIATE_DIMENSION", "MOE_W1_COMPONENT_COUNT",
        "MLA_LATENT_DIMENSION", "MLA_QK_ROPE_HEAD_DIMENSION", "MTP_LAYER_INDEX", "KV_BITS", "KV_A_DIMENSION",        "MOE_TOP_K", "MOE_EXPERT_COUNT", "DSA_INDEX_FULL_FIRST_LAYER",
        "DSA_INDEX_FULL_GROUP_LAYER", "DSA_INDEX_SHARE_GROUP_LAYER_COUNT",
        "LAYER_THREADS", "ATTN_THREADS", "KV_BYTES_PER_SCALAR",
        "KV_BLOCK_TOKEN_COUNT", "DSA_INDEX_HEAD_COUNT", "DSA_INDEX_HEAD_DIMENSION",
        "KV_PAGE_SLOTS",
    )
    for key in required:
        if key not in values:
            raise AssertionError(f"llm_defines.h is missing vector source LLM_{key}")
    out = {
        "ROUTED_LAYERS": resolve("LAYER_COUNT", values) - resolve("FIRST_ROUTED_LAYER", values),
        "GATE_UP_DIM": resolve("MOE_INTERMEDIATE_DIMENSION", values) * resolve("MOE_W1_COMPONENT_COUNT", values),
        "LATENT_ROW": resolve("MLA_LATENT_DIMENSION", values) + resolve("MLA_QK_ROPE_HEAD_DIMENSION", values),
        "WEIGHT_LAYERS": resolve("MTP_LAYER_INDEX", values) + 1,
        "KV_SLOT_BYTES": resolve("KV_A_DIMENSION", values) * resolve("KV_BITS", values) // 8,
        "KV_BYTES_PER_SCALAR": resolve("KV_BYTES_PER_SCALAR", values),
        "LAYER_THREADS": resolve("LAYER_THREADS", values),
        "ATTN_THREADS": resolve("ATTN_THREADS", values),
        "DSA_QUERY_DIM": resolve("DSA_INDEX_HEAD_COUNT", values) * resolve("DSA_INDEX_HEAD_DIMENSION", values),
        "KV_PAGE_SLOTS": resolve("KV_PAGE_SLOTS", values),
    }
    top_k = resolve("MOE_TOP_K", values)
    experts = resolve("MOE_EXPERT_COUNT", values)
    for bucket in (1, 128, 1024):
        out[f"ROWS_PER_EXPERT_B{bucket}"] = (bucket * top_k + experts - 1) // experts
    first = resolve("DSA_INDEX_FULL_FIRST_LAYER", values)
    group = resolve("DSA_INDEX_FULL_GROUP_LAYER", values)
    share = resolve("DSA_INDEX_SHARE_GROUP_LAYER_COUNT", values)
    for layer in (0, 2, 3, 5, 6, 9, 10):
        full = 1 if (layer < first or (layer >= group and (layer - group) % share == 0)) else 0
        out[f"FULL_INDEXER_L{layer}"] = full
    return out


def build_and_run(source, flags):
    with tempfile.TemporaryDirectory() as workdir:
        binary = str(Path(workdir) / "probe")
        command = ["cc", *CFLAGS, *INCLUDES, *flags, str(ROOT / source), "-o", binary]
        build = subprocess.run(command, capture_output=True, text=True)
        if build.returncode != 0:
            return ("build-failed", build.stderr)
        run = subprocess.run([binary], capture_output=True, text=True)
        if run.returncode != 0:
            return ("run-failed", run.stdout + run.stderr)
        return ("passed", run.stdout)


def expect_pass(source, flags, needle):
    status, output = build_and_run(source, flags)
    assert status == "passed", f"{source} positive case {status}:\n{output}"
    assert needle in output, f"{source} positive case missing '{needle}':\n{output}"


def expect_fail(source, flags, needle):
    status, output = build_and_run(source, flags)
    assert status in ("run-failed", "build-failed"), \
        f"{source} negative control unexpectedly {status}:\n{output}"
    assert needle in output, f"{source} negative control did not name '{needle}':\n{output}"


def vector_flags(vectors_map, overrides=None):
    injected = dict(vectors_map)
    if overrides:
        injected.update(overrides)
    return [f"-DVECTOR_{key}={value}u" for key, value in sorted(injected.items())]


def main():
    original = DEFINES.read_text()
    for key in ("SPARK_LLM_KV_BYTES_PER_SCALAR", "SPARK_LLM_MOE_TOP_K", "SPARK_LLM_KV_BLOCK_TOKEN_COUNT"):
        assert re.search(rf"^#define {key} \d+u$", original, re.M), \
            f"{key} must stay a plain integer vector source"
    values = parse_defines(DEFINES)
    vectors_original = vectors(values)
    flags_original = vector_flags(vectors_original)
    expect_pass("tests/test_common_glm_cuda_tree.c", flags_original,
                "PASS common_glm_cuda_tree")
    expect_pass("tests/test_common_kv_geometry.c", flags_original,
                "PASS common_kv_geometry")
    expect_pass("tests/test_common_glm_stage_module.c", (),
                "PASS common_glm_stage_module")

    expect_fail("tests/test_common_kv_geometry.c",
                vector_flags(vectors_original,
                             {"KV_BYTES_PER_SCALAR": vectors_original["KV_BYTES_PER_SCALAR"] + 1}),
                "bytes_per_scalar tracks llm_defines")
    expect_fail("tests/test_common_glm_cuda_tree.c",
                vector_flags(vectors_original,
                             {"ROWS_PER_EXPERT_B128": vectors_original["ROWS_PER_EXPERT_B128"] + 1}),
                "rows/expert at B128")

    try:
        flipped = re.sub(r"^#define SPARK_LLM_KV_BLOCK_TOKEN_COUNT \d+u$",
                         "#define SPARK_LLM_KV_BLOCK_TOKEN_COUNT 63u",
                         original, count=1, flags=re.M)
        assert flipped != original
        DEFINES.write_text(flipped)
        expect_fail("tests/test_common_kv_geometry.c", flags_original,
                    "SPARK_GLM_KV_BLOCK_TOKEN_COUNT must be a power of two")
    finally:
        DEFINES.write_text(original)
    assert DEFINES.read_text() == original, "llm_defines.h must be restored verbatim"
    print("PASS common GLM modules: vectors from llm_defines.h, negative controls named")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
