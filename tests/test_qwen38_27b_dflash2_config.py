#!/usr/bin/env python3
"""Stage-zero receipts for the qwen38_27b DFlash2 draft path (complexity lane).

Pins the 2026-08-28 eviction in SparkQwen38_27bModuleRunDsparkBlockForward
(159 -> 75 CCN):
1. DFlash2 configuration is read into the typed SparkQwen38_27bDflash2Config
   by SparkQwen38_27bDflash2ConfigLoad only: no DFlash2/DSpark variable is
   read anywhere else in the module;
2. the block-forward body itself contains ZERO getenv sites;
3. the 55-line inline /tmp ctxdump block and the unconditional one-shot
   /tmp parity dump stay deleted;
4. the CTX_TAIL fail-loud clamp lives in the loader.
It does not require the variables or the remaining /tmp captures to stay:
having the same loader read validated adapter configuration instead, and
moving the captures under DEBUG (I03, I04, I22), must not fail this test.
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

    struct_name = "SparkQwen38_27bDflash2Config"
    loader_name = "SparkQwen38_27bDflash2ConfigLoad"
    if f"typedef struct {struct_name}" not in text:
        raise AssertionError(f"missing typed config struct {struct_name}")
    loader = function_body(text, f"static SparkStatus {loader_name}(")

    outside_loader = text.replace(loader, "")
    for env in ENV_FLAGS:
        if f'"{env}"' in outside_loader:
            raise AssertionError(f"{env} is read outside {loader_name}")

    forward = function_body(text, "static SparkStatus SparkQwen38_27bModuleRunDsparkBlockForward(")
    if "getenv" in forward:
        raise AssertionError("getenv re-entered the DSpark block forward")
    if "state->dflash2_config" not in forward and "cfg->" not in forward:
        raise AssertionError("the forward must consume the typed config struct")

    for needle, message in (
            ("SPARK_QWEN38_27B_DFLASH2_FRAME_KV_ROWS - 2048u", "CTX_TAIL range clamp missing from the loader"),
            ("dflash2_ctx_tail_out_of_range", "CTX_TAIL fail-loud log missing"),
            ("SPARK_STATUS_CAPACITY_EXCEEDED", "CTX_TAIL must fail capacity")):
        if needle not in loader:
            raise AssertionError(message)

    for path in DELETED_DUMP_PATHS:
        if path in text:
            raise AssertionError(f"deleted dump path reappeared: {path}")

    print(f"qwen38_27b dflash2 stage-zero receipts OK: configuration read only by "
          f"{loader_name} into {struct_name}, forward getenv-free, inline /tmp dumps deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
