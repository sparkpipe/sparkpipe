#!/usr/bin/env python3
"""Stage-zero receipts for the qwen38_27b DFlash2 draft path (complexity lane).

Pins the 2026-08-28 eviction in SparkQwen38_27bModuleRunDsparkBlockForward
(159 -> 75 CCN):
1. the six env flag NAMES remain the config source, parsed exactly once in
   SparkQwen38_27bDflash2ConfigLoad (typed SparkQwen38_27bDflash2Config);
2. the block-forward body itself contains ZERO getenv sites;
3. the 55-line inline /tmp ctxdump block and the unconditional one-shot
   /tmp parity dump are gone (the gated ctxrun/l0 captures remain, feeding
   tools/qwen36_dflash2_round_parity.py et al.);
4. the CTX_TAIL fail-loud clamp lives in the loader.
"""
import pathlib
import re
import sys

MODULE = (pathlib.Path(__file__).resolve().parents[1]
          / "modules/qwen38_27b_resident_decode_stage/source"
          / "spark_qwen38_27b_resident_decode_stage_module.c")

ENV_FLAGS = (
    "SPARK_QWEN38_27B_DFLASH2_WINDOW",
    "SPARK_QWEN38_27B_DFLASH2_BLOCK_KV",
    "SPARK_QWEN38_27B_DFLASH2_CTX_TAIL",
    "SPARK_QWEN38_27B_DFLASH2_CTX_CACHE",
    "SPARK_QWEN38_27B_DFLASH2_CTX_DUMP",
    "SPARK_QWEN38_27B_DSPARK_SEL_CHECK",
)

DELETED_DUMP_PATHS = (
    "/tmp/ctxdump_taps_last.bin",
    "/tmp/ctxdump_fc_last.bin",
    "/tmp/ctxdump_normed_last.bin",
    "/tmp/ctxwin_taps.bin",
    "/tmp/ctxwin.meta",
    "/tmp/ctxwin_anchor",
    "/tmp/dflash2_taps.bin",
    "/tmp/dflash2_c0.bin",
    "/tmp/dflash2_logits.bin",
    "/tmp/dflash2_hidden.bin",
)

KEPT_DUMP_PATHS = (
    "/tmp/l0_sample_rows.txt",
    "/tmp/l0_kv_k.bin",
    "/tmp/l0_kv_v.bin",
    "/tmp/l0_q.bin",
    "/tmp/l0_attn.bin",
)


def function_body(text: str, signature: str) -> str:
    start = text.index(signature)
    open_brace = text.index("{", start)
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace:index]
    raise AssertionError(f"unbalanced braces after {signature}")


def main() -> int:
    text = MODULE.read_text(encoding="utf-8", errors="surrogateescape")

    # 1. the typed config struct + one-time loader exist
    struct_name = "SparkQwen38_27bDflash2Config"
    loader_name = "SparkQwen38_27bDflash2ConfigLoad"
    assert f"typedef struct {struct_name}" in text, f"missing typed config struct {struct_name}"
    loader = function_body(text, f"static SparkStatus {loader_name}(")

    # 2. every env NAME is still the config source, read exactly once (compat)
    for env in ENV_FLAGS:
        count = text.count(f'"{env}"')
        assert count == 1, f"{env}: expected exactly 1 occurrence in the loader, found {count}"
        assert env in loader, f"{env} not parsed by {loader_name}"
    assert "getenv(" in loader, "loader must read the env names itself"

    # 3. the forward path is getenv-free
    forward = function_body(text, "static SparkStatus SparkQwen38_27bModuleRunDsparkBlockForward(")
    assert "getenv" not in forward, "getenv re-entered the DSpark block forward"
    assert "state->dflash2_config" in forward or "cfg->" in forward, \
        "the forward must consume the typed config struct"

    # 4. the CTX_TAIL clamp is fail-loud in the loader (frame-bound headroom)
    assert "SPARK_QWEN38_27B_DFLASH2_FRAME_KV_ROWS - 2048u" in loader, \
        "CTX_TAIL range clamp missing from the loader"
    assert "dflash2_ctx_tail_out_of_range" in loader, "CTX_TAIL fail-loud log missing"
    assert "SPARK_STATUS_CAPACITY_EXCEEDED" in loader, "CTX_TAIL must fail capacity"

    # 5. the deleted inline /tmp dumps stay deleted; the gated captures stay
    for path in DELETED_DUMP_PATHS:
        assert path not in text, f"deleted dump path reappeared: {path}"
    for path in KEPT_DUMP_PATHS:
        assert path in text, f"gated parity-capture path missing: {path}"

    print(f"qwen38_27b dflash2 stage-zero receipts OK: 6 env flags -> {struct_name} "
          f"(loader {loader_name}), forward getenv-free, inline /tmp dumps deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
