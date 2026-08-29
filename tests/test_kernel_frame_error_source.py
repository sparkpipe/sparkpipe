#!/usr/bin/env python3
"""Source contract: corruption fails the frame, never the context (K1/K2/K4).

The fail-frame discipline (docs/BUG_LEDGER.md, kernel-crew lane): kernels
respond to corrupt inputs by recording into the per-frame error record
(inference/kernels/frame_error.cuh) and returning a bounded result. Device
`trap` terminates the whole CUDA context, so it is banned as a data-corruption
response in inference/kernels and modules - the only permitted spellings are
the compile-time capability asserts in mma.cuh's #else arms, which fire when
a build configuration is wrong, never when a frame's data is.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_TRAP_FILES = {
    # compile-time capability asserts (unsupported PTX on the target), not
    # data-corruption responses: they fire for a misconfigured build.
    "mma.cuh",
}


def main() -> int:
    failures = []
    kernels = ROOT / "inference" / "kernels"
    for path in sorted(kernels.glob("*.cuh")):
        body = path.read_text()
        if "asm volatile" in body and "trap" in body:
            if path.name not in ALLOWED_TRAP_FILES:
                failures.append(
                    f"{path.relative_to(ROOT)}: device trap remains as a "
                    f"data-corruption response")
    for path in sorted((ROOT / "modules").rglob("*.cu")):
        body = path.read_text()
        for line in body.splitlines():
            if "asm volatile" in line and "trap" in line:
                failures.append(
                    f"{path.relative_to(ROOT)}: device trap remains: "
                    f"{line.strip()}")

    # the fixed corruption sites report through the frame record
    required = {
        "inference/kernels/tile.cuh": "LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE",
        "inference/kernels/activation.cuh": "LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE",
        "inference/kernels/gemm.cuh": "frame_error",
        "modules/qwen4_flash_resident_decode_stage/source/"
        "spark_qwen4_flash_resident_decode_stage_cuda.cu":
            "LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE",
        "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_cuda.cu":
            "LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE",
        "modules/qwen38_27b_resident_decode_stage/source/"
        "spark_qwen38_27b_resident_decode_stage_cuda.cu":
            "LM_FRAME_ERROR_PAYLOAD_WINDOW_OUT_OF_RANGE",
    }
    for relative, needle in required.items():
        if needle not in (ROOT / relative).read_text():
            failures.append(f"{relative}: missing frame-error report '{needle}'")

    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        return 1
    print("frame-error source contract holds: no device traps on corruption "
          "paths, every fixed site reports into the frame record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
