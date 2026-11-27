#!/usr/bin/env python3
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKL2C = ROOT / "tools" / "skl2c" / "skl2c.py"
SKL2_DIR = ROOT / "inference" / "skl2"

sys.path.insert(0, str(ROOT / "tools" / "skl2c"))
sys.path.append(str(ROOT / "tools" / "sklc"))

from skl2_frontend import Skl2CompileError, load_unit, parse_skl2_text
from skl2_lower import build_lockfile, compile_unit, resolve_models, select_primitive_variant, verify_lockfile


def expect_error(src, expected_code):
    try:
        u = parse_skl2_text(src, file="neg_test.skl2")
        compile_unit(u, arch="sm121")
    except Skl2CompileError as err:
        assert err.code == expected_code, f"expected {expected_code}, got {err.code}: {err}"
        return err
    raise AssertionError(f"expected {expected_code}, but compilation succeeded")


def test_anti_slop_hard_errors():
    dup_block = "\n".join(f"    reg u32 v{i} = tid + {i + 1};" for i in range(10))
    dup_sum = " + ".join(f"v{i}" for i in range(10))
    err_dup = expect_error(
        f"""
kernel[128, 32] k_a(global u32* dst) {{
    reg u32 tid = tid_x();
{dup_block}
    stg32(ptr_add(dst, tid, 4), {dup_sum});
}}
kernel[128, 32] k_b(global u32* dst) {{
    reg u32 tid = tid_x();
{dup_block.replace("v", "w")}
    stg32(ptr_add(dst, tid, 4), {dup_sum.replace("v", "w")});
}}
""",
        "E_DUP_SUBTREE",
    )
    assert "neg_test.skl2:" in str(err_dup), str(err_dup)

    ifs = "\n".join(f"    if (tid == {i}) {{ acc = acc + {i + 1}; }}" for i in range(17))
    expect_error(
        f"""
kernel[128, 32] too_complex(global u32* dst) {{
    reg u32 tid = tid_x();
    reg u32 acc = 0;
{ifs}
    stg32(ptr_add(dst, tid, 4), acc);
}}
""",
        "E_COMPLEXITY",
    )

    expect_error(
        """
kernel[128, 32] too_nested(global u32* dst, u32 n) {
    reg u32 tid = tid_x();
    if (tid < n) {
        if (tid > 0) {
            if (tid < 64) {
                if (tid > 1) {
                    if (tid != 5) {
                        stg32(ptr_add(dst, tid, 4), tid);
                    }
                }
            }
        }
    }
}
""",
        "E_COMPLEXITY",
    )

    long_body = "\n".join(f"    acc = acc + {i + 1};" for i in range(82))
    expect_error(
        f"""
kernel[128, 32] too_long(global u32* dst) {{
    reg u32 tid = tid_x();
    reg u32 acc = tid;
{long_body}
    stg32(ptr_add(dst, tid, 4), acc);
}}
""",
        "E_KERNEL_TOO_LONG",
    )

    expect_error(
        """
kernel[128, 16] reg_spill(global u32* dst, global u32* src) {
    reg u32 tid = tid_x();
    reg u32x8 a = ldg256_stream(ptr_add(src, tid, 32));
    reg u32x8 b = ldg256_stream(ptr_add(src, tid + 1, 32));
    reg u32x8 c = ldg256_stream(ptr_add(src, tid + 2, 32));
    stg32(ptr_add(dst, tid, 4), a[0] + b[0] + c[0]);
}
""",
        "E_REG_SPILL",
    )

    expect_error(
        """
macro never_used(x) {
    x = x + 1;
}
kernel[128, 32] k_ok(global u32* dst) {
    reg u32 tid = tid_x();
    stg32(ptr_add(dst, tid, 4), tid);
}
""",
        "E_DEAD_CODE",
    )

    expect_error(
        """
kernel[128, 32] dead_local(global u32* dst) {
    reg u32 tid = tid_x();
    reg u32 unused_var = tid + 1;
    stg32(ptr_add(dst, tid, 4), tid);
}
""",
        "E_DEAD_CODE",
    )

    expect_error(
        """
template kernel[128, 32] uninst<N>(global u32* dst) {
    reg u32 tid = tid_x();
    stg32(ptr_add(dst, tid, 4), tid + N);
}
kernel[128, 32] k_live(global u32* dst) {
    reg u32 tid = tid_x();
    stg32(ptr_add(dst, tid, 4), tid);
}
""",
        "E_DEAD_CODE",
    )

    expect_error(
        """
@pdl
kernel[128, 32] pdl_store_early(global u32* dst) {
    reg u32 tid = tid_x();
    stg32(ptr_add(dst, tid, 4), tid);
    acqbulk();
    preexit();
}
""",
        "E_PDL_HAZARD",
    )

    expect_error(
        """
@pdl
kernel[128, 32] pdl_no_preexit(global u32* dst, u32 n) {
    acqbulk();
    reg u32 tid = tid_x();
    if (tid >= n) { return; }
    stg32(ptr_add(dst, tid, 4), tid);
    preexit();
}
""",
        "E_PDL_HAZARD",
    )


def test_stdlib_policy_versioning_and_lockfile():
    unit = load_unit([str(SKL2_DIR / "models" / "families.skl2")], search_roots=[str(SKL2_DIR)])

    s_small = select_primitive_variant(unit, "std.allreduce", "latest", {"msg_bytes": 10240, "ranks": 16, "topology": "hybrid_mode0"})
    assert s_small["variant"] == "hybrid_inline_128", s_small
    s_2d = select_primitive_variant(unit, "std.allreduce", "latest", {"msg_bytes": 65536, "ranks": 16, "topology": "2d_4x4"})
    assert s_2d["variant"] == "two_d_4x4", s_2d
    s_ring = select_primitive_variant(unit, "std.allreduce", "latest", {"msg_bytes": 65536, "ranks": 16, "topology": "ring"})
    assert s_ring["variant"] == "hybrid_dma_ring", s_ring

    ver_src = """
primitive std.allreduce @v1 {
    variant hybrid_inline_128 { max_inline_bytes = 16384; }
    policy { default => hybrid_inline_128; }
}
primitive std.allreduce @v2 {
    variant dual_rail_200g { max_inline_bytes = 32768; }
    policy { default => dual_rail_200g; }
}
model ReferenceTransformer @v1 {
    hidden = 4096;
    use std.allreduce@latest;
}
model dev_model extends ReferenceTransformer@latest {
    hidden = 5120;
}
model pinned_model extends ReferenceTransformer@v1 {
    hidden = 5120;
    use std.allreduce@v1;
}
"""
    u_ver = parse_skl2_text(ver_src, "version_test.skl2")
    plan_ver = resolve_models(u_ver, ["dev_model", "pinned_model"])
    assert plan_ver["models"]["dev_model"]["resolved_primitives"]["std.allreduce"] == "v2:dual_rail_200g", plan_ver
    assert plan_ver["models"]["pinned_model"]["resolved_primitives"]["std.allreduce"] == "v1:hybrid_inline_128", plan_ver

    lock_doc = build_lockfile(unit)
    assert lock_doc["schema"] == "skl2.lock.v1"
    assert "std.allreduce" in lock_doc["primitives"]
    assert "std.linear" in lock_doc["primitives"]
    verify_lockfile(unit, lock_doc)

    bad_lock = json.loads(json.dumps(lock_doc))
    bad_lock["primitives"]["std.allreduce"]["sha256"] = "0" * 64
    try:
        verify_lockfile(unit, bad_lock)
        raise AssertionError("expected E_LOCK_MISMATCH")
    except Skl2CompileError as err:
        assert err.code == "E_LOCK_MISMATCH", err


def test_differential_models_and_dedup():
    unit = load_unit([str(SKL2_DIR / "models" / "families.skl2")], search_roots=[str(SKL2_DIR)])
    plan = resolve_models(unit)
    expected_models = {"qwen38_27b", "deepseek_v41_flash", "kimi_k31_pro", "glm53_flash"}
    assert set(plan["models"].keys()) == expected_models, plan["models"].keys()
    assert plan["models"]["deepseek_v41_flash"]["inheritance"] == [
        "deepseek_v41_flash@v1",
        "ReferenceMlaTransformer@v1",
        "ReferenceMoeTransformer@v1",
        "ReferenceTransformer@v1",
    ]
    assert plan["models"]["qwen38_27b"]["spec"]["hidden"] == 5120
    assert plan["models"]["kimi_k31_pro"]["spec"]["moe_experts"] == 896
    assert plan["models"]["glm53_flash"]["spec"]["layers"] == 45
    assert plan["dedup_ratio"] >= 0.50, f"expected >=50% cross-model dedup, got {plan['dedup_ratio']}"
    assert "allreduce.hybrid_inline_128_fused_rmsnorm" in plan["shared_kernels"]

    families_lines = len((SKL2_DIR / "models" / "families.skl2").read_text(encoding="utf-8").splitlines())
    assert families_lines <= 80, f"expected all 4 differential models in <=80 lines, got {families_lines}"


def test_full_cubin_emission_and_cli():
    assert not list((ROOT / "inference" / "kernels" / "skl").glob("*.skl")), "legacy .skl v1 files must be retired"
    inputs = [
        str(SKL2_DIR / "stdlib" / "allreduce.skl2"),
        *(str(p) for p in sorted((SKL2_DIR / "kernels").glob("*.skl2"))),
    ]
    for arch, expected_sm in (("sm121", 121), ("sm120", 120)):
        with tempfile.TemporaryDirectory() as tmp:
            cubin_path = pathlib.Path(tmp) / f"skl2_{arch}.cubin"
            manifest_path = pathlib.Path(tmp) / f"skl2_{arch}.manifest"
            lock_path = pathlib.Path(tmp) / "skl2.lock"
            res = subprocess.run(
                [sys.executable, str(SKL2C), "--arch", arch, *inputs, "-o", str(cubin_path), "--manifest", str(manifest_path), "--stats"],
                capture_output=True, text=True, check=False,
            )
            assert res.returncode == 0, f"skl2c failed for {arch}: {res.stderr}"
            blob = cubin_path.read_bytes()
            assert blob[:4] == b"\x7fELF"
            assert struct.unpack_from("<H", blob, 18)[0] == 190
            assert ((struct.unpack_from("<I", blob, 48)[0] >> 8) & 0xFF) == expected_sm
            mtxt = manifest_path.read_text(encoding="utf-8")
            assert f"cubin {hashlib.sha256(blob).hexdigest()} {len(blob)}" in mtxt
            for required in (
                "skinny_bf16_b1", "skinny_fp8_b1", "skinny_mxfp4_b1", "skinny_nvfp4_b1",
                "skinny_bf16_b2", "skinny_bf16_b4", "skinny_bf16_b1_residual", "skinny_bf16_b1_silu",
                "mma_bf16_gemm_bf16", "mma_bf16_gemm_f32",
                "rmsnorm_bf16_vec16", "fused_residual_rmsnorm_bf16x8", "rmsnorm_rope_bf16", "rmsnorm_rope_fp8_bf16",
                "allreduce_wqe_stg128", "allreduce_combine_rmsnorm_bf16",
                "latent_attn_decode_512_64", "gqa_split_reduce_sigmoid_gate_pdl_exact_bf16",
                "delta_rule_decode1_256_128_128_4_pdl", "gdn_conv_split_tiled_4_128_pdl",
                "causal_conv_serial_256_4_swish_bf16_pdl", "causal_conv_tiled_4_swish_bf16_pdl",
                "router_softmax_256", "router_softmax_bf16_256", "mx_quantize_e4m3_ue8m0",
                "head_certified_reduce_f32", "head_certified_rescore_batched_f32_bf16",
                "head_certified_fp8_score_1024", "head_candidate_256_1024_pdl_bf16", "head_softmax_pdl_f32",
                "stage_embedding_lookup_pdl_bf16x8", "stage_row_dot_pdl_bf16x8",
                "tp_f32_to_bf16_vec16", "device_program_copy_exact", "device_program_fill_exact",
            ):
                assert f"kernel {required} " in mtxt, f"missing {required} in {arch} manifest"
            assert "spill_stores 0 spill_loads 0" in mtxt
            res_lock = subprocess.run(
                [sys.executable, str(SKL2C), "lock", str(SKL2_DIR / "models" / "families.skl2"), "-o", str(lock_path)],
                capture_output=True, text=True, check=False,
            )
            assert res_lock.returncode == 0, res_lock.stderr
            res_v = subprocess.run(
                [sys.executable, str(SKL2C), "--verify-lock", str(lock_path), "--model", "all", str(SKL2_DIR / "models" / "families.skl2")],
                capture_output=True, text=True, check=False,
            )
            assert res_v.returncode == 0, res_v.stderr


def main():
    test_anti_slop_hard_errors()
    test_stdlib_policy_versioning_and_lockfile()
    test_differential_models_and_dedup()
    test_full_cubin_emission_and_cli()
    print("PASS skl2c anti-slop gates, hill-climbed stdlib policy/versioning/lockfile, differential models, and 91 sm_121a/sm_120a kernels (0 legacy .skl files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
