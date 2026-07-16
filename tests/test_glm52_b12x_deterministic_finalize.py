#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require_contains(path: str, needle: str) -> None:
    text = (ROOT / path).read_text()
    if needle not in text:
        raise AssertionError(f"missing {needle!r} in {path}")


def forbid_contains(path: str, needle: str) -> None:
    text = (ROOT / path).read_text()
    if needle in text:
        raise AssertionError(f"forbidden {needle!r} in {path}")


def main() -> int:
    require_contains(
        "modules/glm52_sm121_b12x_compiled_backend/include/sparkpipe/spark_glm52_sm121_b12x_generated_kernel_table.h",
        "SPARK_GLM52_SM121_B12X_GENERATED_MANIFEST_FLAG_ROUTE_SLICE_OUTPUT",
    )
    require_contains(
        "modules/glm52_sm121_b12x_compiled_backend/include/sparkpipe/spark_glm52_sm121_b12x_generated_kernel_table.h",
        "route_output_slice_count",
    )
    require_contains(
        "modules/glm52_sm121_b12x_compiled_backend/source/spark_flashinfer_b12x_compiled_moe_backend.cu",
        "SparkGlm52B12xDeterministicFc2FinalizeKernel",
    )
    require_contains(
        "modules/glm52_sm121_b12x_compiled_backend/source/spark_flashinfer_b12x_compiled_moe_backend.cu",
        "route_row * (uint64_t)route_output_slice_count",
    )
    require_contains(
        "modules/glm52_sm121_b12x_compiled_backend/source/spark_flashinfer_b12x_compiled_moe_backend.cu",
        "generated_arguments.output_bf16 =\n        state->workspace.route_output_bf16;",
    )
    forbid_contains(
        "modules/glm52_sm121_b12x_compiled_backend/source/spark_flashinfer_b12x_compiled_moe_backend.cu",
        "state->workspaces",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/cute_dsl/fp4_common.py",
        "def store_bf16x2",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/cute_dsl/fp4_common.py",
        "st.global.b32 [$0], packed;",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/cute_dsl/fp4_common.py",
        "def store_v4_bf16x2",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/cute_dsl/fp4_common.py",
        "st.global.v4.b32 [$0], {p0, p1, p2, p3};",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py",
        "scatter_total = total_pairs * Int32(self.output_tile_count_n) * cols",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py",
        "route_slice_row = (",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py",
        "routed_rows * route_output_slice_count",
    )
    require_contains(
        "third_party/flashinfer/flashinfer/fused_moe/cute_dsl/b12x_moe.py",
        "_sparkpipe_b12x_deterministic_route_output_row_count",
    )
    require_contains(
        "tools/glm52_b12x_aot_compile.py",
        "route_slice_output",
    )
    require_contains(
        "tools/glm52_b12x_aot_compile.py",
        "token_count = int(key[5])",
    )
    forbid_contains(
        "tools/glm52_b12x_aot_compile.py",
        "token_count = int(key[4])",
    )
    forbid_contains(
        "tools/glm52_b12x_aot_compile.py",
        "_MICRO_KERNEL_CACHE",
    )
    forbid_contains(
        "tools/glm52_b12x_aot_compile.py",
        "_DYNAMIC_KERNEL_CACHE",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
