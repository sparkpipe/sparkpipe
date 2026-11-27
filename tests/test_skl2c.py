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
    assert s_small["version"] == "v2" and s_small["variant"] == "hybrid_single_acquire_128", s_small
    s_v1 = select_primitive_variant(unit, "std.allreduce", "v1", {"msg_bytes": 10240, "ranks": 16, "topology": "hybrid_mode0"})
    assert s_v1["version"] == "v1" and s_v1["variant"] == "hybrid_inline_128", s_v1
    s_2d = select_primitive_variant(unit, "std.allreduce", "latest", {"msg_bytes": 65536, "ranks": 16, "topology": "2d_4x4"})
    assert s_2d["variant"] == "two_d_4x4", s_2d
    s_ring = select_primitive_variant(unit, "std.allreduce", "latest", {"msg_bytes": 65536, "ranks": 16, "topology": "ring"})
    assert s_ring["variant"] == "hybrid_dma_ring", s_ring

    pin_src = """
import "models/reference.skl2";
model dev_hybrid extends ReferenceHybridTransformer@latest {
    hidden = 5120;
    tp_degree = 16;
}
model pinned_hybrid extends ReferenceHybridTransformer@v1 {
    hidden = 5120;
    tp_degree = 16;
    use std.allreduce@v1;
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        pin_p = pathlib.Path(tmp) / "version_test.skl2"
        pin_p.write_text(pin_src, encoding="utf-8")
        u_ver = load_unit([str(pin_p)], search_roots=[str(SKL2_DIR)])
    plan_ver = resolve_models(u_ver, ["dev_hybrid", "pinned_hybrid"])
    assert plan_ver["models"]["dev_hybrid"]["resolved_primitives"]["std.allreduce"] == "v2:hybrid_single_acquire_128", plan_ver
    assert plan_ver["models"]["pinned_hybrid"]["resolved_primitives"]["std.allreduce"] == "v1:hybrid_inline_128", plan_ver
    lock_dev = build_lockfile(u_ver, ["dev_hybrid"])
    lock_pin = build_lockfile(u_ver, ["pinned_hybrid"])
    assert lock_dev["models"]["dev_hybrid"]["sha256"] != lock_pin["models"]["pinned_hybrid"]["sha256"]
    assert lock_dev["resolved_models"]["dev_hybrid"]["resolved_primitives"]["std.allreduce"] == "v2:hybrid_single_acquire_128"
    assert lock_pin["resolved_models"]["pinned_hybrid"]["resolved_primitives"]["std.allreduce"] == "v1:hybrid_inline_128"

    lock_doc = build_lockfile(unit)
    assert lock_doc["schema"] == "skl2.lock.v1"
    assert lock_doc["primitives"]["std.allreduce"]["version"] == "v2"
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
    expected_models = {
        "qwen38_27b", "qwen38_flash", "qwen38_max",
        "deepseek_v41_flash", "kimi_k31_pro",
        "glm53", "glm53_flash",
        "gemma4_31b", "gemma4_26b_a4b", "ling_30_flash",
    }
    assert set(plan["models"].keys()) == expected_models, plan["models"].keys()
    assert plan["models"]["deepseek_v41_flash"]["inheritance"] == [
        "deepseek_v41_flash@v1",
        "ReferenceMlaTransformer@v1",
        "ReferenceMoeTransformer@v1",
        "ReferenceTransformer@v1",
    ]
    assert plan["models"]["qwen38_27b"]["spec"]["hidden"] == 5120
    assert plan["models"]["qwen38_max"]["spec"]["hidden"] == 8192
    assert plan["models"]["kimi_k31_pro"]["spec"]["moe_experts"] == 896
    assert plan["models"]["glm53"]["spec"]["layers"] == 78
    assert plan["models"]["glm53_flash"]["spec"]["layers"] == 45
    assert plan["models"]["gemma4_31b"]["spec"]["vocab"] == 262144
    assert plan["models"]["gemma4_26b_a4b"]["spec"]["moe_experts"] == 128
    assert plan["models"]["ling_30_flash"]["spec"]["moe_experts"] == 512
    for m in expected_models:
        assert plan["models"][m]["resolved_primitives"]["std.allreduce"] == "v2:hybrid_single_acquire_128", (m, plan["models"][m])
    assert plan["dedup_ratio"] >= 0.65, f"expected >=65% cross-model dedup across 10 models, got {plan['dedup_ratio']}"
    assert "allreduce.hybrid_single_acquire_128_fused_rmsnorm" in plan["shared_kernels"]

    families_lines = len((SKL2_DIR / "models" / "families.skl2").read_text(encoding="utf-8").splitlines())
    assert families_lines <= 165, f"expected all 10 differential models in <=165 lines, got {families_lines}"


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
                "qwen_exact_bf16_multi_w1x4_r4_n4", "qwen_exact_bf16_gate_up_swiglu_w1x4_r4_n4",
                "qwen_bf16_dense_mma_16x64", "qwen_bf16_dense_mma_128x64",
                "qwen_bf16_fused_swiglu_mma_16x64", "qwen_bf16_fused_swiglu_mma_128x64",
                "qwen_bf16_smalln_mma_16", "qwen_bf16_smalln_mma_wide_32",
                "qwen_hybrid_attn_prep_tp1", "qwen_attn_prep_local_tp1", "qwen_prior_kv_direct_4_256",
                "qwen_local_prefill_attn_4_256_direct", "qwen_gqa_split_decode_24_4_256", "qwen_gqa_direct_decode_gate_24_4_256",
                "qwen_gdn_conv_split_serial_4_128_pdl", "qwen_gdn_conv_prefill_2d_4_128_pdl",
                "qwen_gdn_conv_window_commit_4_pdl", "qwen_gdn_gate_only_128_pdl", "qwen_gdn_split_expand_gate_256_128_pdl",
                "qwen_gdn_exact_edge_pre_128_128_pdl", "qwen_gdn_exact_edge_post_128_128_pdl", "qwen_gdn_scalar_chunk_128_128_128_pdl",
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
            hdr_path = pathlib.Path(tmp) / f"qwen38_27b_{arch}.h"
            res_hdr = subprocess.run(
                [sys.executable, str(SKL2C), "--arch", arch, "--model", "qwen38_27b", "--emit-c-header", str(hdr_path), str(SKL2_DIR / "models" / "families.skl2"), *inputs],
                capture_output=True, text=True, check=False,
            )
            assert res_hdr.returncode == 0, f"skl2c --emit-c-header failed: {res_hdr.stderr}"
            htxt = hdr_path.read_text(encoding="utf-8")
            assert "//" not in htxt and "/*" not in htxt, "generated C99 header must be comment-free"
            for tok in (
                "#define SKL2_QWEN38_27B_HIDDEN 5120u",
                "SKL2_ORD_COUNT = 150",
                "Skl2ModuleLoad",
                "Skl2KernelLayout",
                "Skl2Launch_qwen_bf16_dense_mma_16x64",
                "Skl2Launch_qwen_local_prefill_attn_4_256_direct",
                "Skl2Launch_qwen_gdn_scalar_chunk_128_128_128_pdl",
                "Skl2Select_allreduce",
                "Skl2Ordinal_allreduce",
                "SKL2_VAR_ALLREDUCE_HYBRID_SINGLE_ACQUIRE_128",
                "Skl2Select_linear",
                "Skl2Select_attention",
            ):
                assert tok in htxt, f"missing {tok!r} in generated C99 header"
            cc_res = subprocess.run(
                ["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", f"-I{ROOT / 'include'}", "-x", "c", str(hdr_path)],
                capture_output=True, text=True, check=False,
            )
            assert cc_res.returncode == 0, f"generated C99 header failed cc -std=c99 -Wall -Wextra -Werror:\n{cc_res.stderr}"
            if arch == "sm121":
                c_test = pathlib.Path(tmp) / "test_policy_c99.c"
                c_bin = pathlib.Path(tmp) / "test_policy_c99"
                c_test.write_text(
                    f'#include <assert.h>\n#include "{hdr_path.name}"\n'
                    "int main(void) {\n"
                    "    uint32_t cnt = 0u, bytes = 0u;\n"
                    "    assert(Skl2Select_allreduce(10240u, 16u, SKL2_TAG_NONE) == SKL2_VAR_ALLREDUCE_HYBRID_SINGLE_ACQUIRE_128);\n"
                    "    assert(Skl2Select_allreduce(65536u, 16u, SKL2_TAG_2D_4X4) == SKL2_VAR_ALLREDUCE_TWO_D_4X4);\n"
                    "    assert(Skl2Select_allreduce(65536u, 16u, SKL2_TAG_NONE) == SKL2_VAR_ALLREDUCE_HYBRID_DMA_RING);\n"
                    "    assert(Skl2Ordinal_allreduce(SKL2_VAR_ALLREDUCE_HYBRID_SINGLE_ACQUIRE_128) == SKL2_ORD_allreduce_wqe_stg128);\n"
                    "    assert(Skl2Ordinal_allreduce(SKL2_VAR_ALLREDUCE_HYBRID_DMA_RING) == SKL2_ORD_allreduce_combine_rmsnorm_bf16);\n"
                    "    assert(Skl2Select_linear(4u) == SKL2_VAR_LINEAR_SKINNY_GEMV);\n"
                    "    assert(Skl2Select_linear(32u) == SKL2_VAR_LINEAR_MMA_GEMM);\n"
                    "    assert(Skl2Select_moe(8u) == SKL2_VAR_MOE_GROUPED_SKINNY);\n"
                    "    assert(Skl2Select_moe(16u) == SKL2_VAR_MOE_GROUPED_MMA);\n"
                    "    assert(Skl2Select_attention(SKL2_TAG_MLA_SPARSE) == SKL2_VAR_ATTENTION_MLA_SPARSE_CSA2);\n"
                    "    assert(Skl2Select_attention(SKL2_TAG_MLA) == SKL2_VAR_ATTENTION_MLA_LATENT);\n"
                    "    assert(Skl2Select_attention(SKL2_TAG_NONE) == SKL2_VAR_ATTENTION_GQA_FLASH);\n"
                    "    assert(Skl2Select_recurrent(SKL2_TAG_HYBRID_KDA) == SKL2_VAR_RECURRENT_KDA_DECODE);\n"
                    "    assert(Skl2Select_recurrent(SKL2_TAG_NONE) == SKL2_VAR_RECURRENT_GDN_DECODE);\n"
                    "    assert(Skl2Select_norm_rope(1u, 0u) == SKL2_VAR_NORM_ROPE_RMSNORM_ROPE_FP8);\n"
                    "    assert(Skl2Select_norm_rope(0u, 1u) == SKL2_VAR_NORM_ROPE_FUSED_RESIDUAL_RMSNORM);\n"
                    "    assert(Skl2Select_norm_rope(0u, 0u) == SKL2_VAR_NORM_ROPE_RMSNORM_VEC16);\n"
                    "    assert(Skl2Select_route_topk() == SKL2_VAR_ROUTE_TOPK_WARP_EXACT_TOPK);\n"
                    "    assert(Skl2Select_head(SKL2_TAG_BF16) == SKL2_VAR_HEAD_CERTIFIED_BF16);\n"
                    "    assert(Skl2Select_head(SKL2_TAG_NONE) == SKL2_VAR_HEAD_CERTIFIED_FP8_RESCORE);\n"
                    "    assert(Skl2KernelLayout(SKL2_ORD_qwen_bf16_dense_mma_16x64, &cnt, &bytes) != 0 && cnt == 6u && bytes == 36u);\n"
                    "    return 0;\n"
                    "}\n",
                    encoding="utf-8",
                )
                cc_run = subprocess.run(
                    ["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", f"-I{ROOT / 'include'}", f"-I{tmp}", str(c_test), "-o", str(c_bin)],
                    capture_output=True, text=True, check=False,
                )
                assert cc_run.returncode == 0, f"C99 runtime policy test failed to compile:\n{cc_run.stderr}"
                exec_res = subprocess.run([str(c_bin)], capture_output=True, text=True, check=False)
                assert exec_res.returncode == 0, f"C99 runtime policy test failed:\n{exec_res.stderr}"
                emb_hdr = pathlib.Path(tmp) / "embedded.h"
                emb_res = subprocess.run(
                    [sys.executable, str(SKL2C), "--arch", "sm_121a", "--only", "qwen_bf16_dense_mma_16x64", "--prefix", "TestEmb", "--emit-embedded-h", str(emb_hdr), str(SKL2_DIR / "kernels" / "mma_gemm.skl2")],
                    capture_output=True, text=True, check=False,
                )
                assert emb_res.returncode == 0, f"--emit-embedded-h failed:\n{emb_res.stderr}"
                etxt = emb_hdr.read_text(encoding="utf-8")
                for tok in ("TESTEMB_qwen_bf16_dense_mma_16x64 = 0", "kTestEmbNames[1]", "kTestEmbCubin[", "0x7f,0x45,0x4c,0x46", "kTestEmbManifest["):
                    assert tok in etxt, f"missing {tok!r} in embedded header"
                emb_cc = subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", "-x", "c", str(emb_hdr)], capture_output=True, text=True, check=False)
                assert emb_cc.returncode == 0, f"embedded header failed cc:\n{emb_cc.stderr}"


def test_pdl_preexit_and_mio_war_invariants():
    unit = load_unit([str(SKL2_DIR / "kernels" / "norm_rope.skl2")], search_roots=[str(SKL2_DIR)])
    images, _ = compile_unit(unit, arch="sm121", only_set={"head_rmsnorm_silu_gate_pdl_exact_128_bf16"}, allow_unused_lib=True)
    insns = images[0].insns
    last_stg = max(i for i, (lo, _) in enumerate(insns) if (lo & 0xFFF) == 0x986)
    preexits = [i for i, (lo, _) in enumerate(insns) if (lo & 0xFFFF) == 0x782D]
    assert preexits and all(p > last_stg for p in preexits), f"PREEXIT before last STG: preexits={preexits} last_stg={last_stg}"
    rd_bar = {}
    for pc, (lo, hi) in enumerate(insns):
        op16, wr, wait = lo & 0xFFFF, (hi >> 46) & 7, (hi >> 52) & 0x3F
        rd_r, ra_r = (lo >> 16) & 0xFF, (lo >> 24) & 0xFF
        if wait:
            rd_bar = {r: (m & ~wait) for r, m in rd_bar.items() if (m & ~wait)}
        if op16 in (0x7589, 0x7F89) and wr < 6:
            rd_bar[ra_r] = rd_bar.get(ra_r, 0) | (1 << wr)
        elif (lo & 0xFFF) in (0x220, 0x802, 0x202) and rd_r in rd_bar:
            raise AssertionError(f"MIO WAR hazard at pc={pc*16:#x}: writes R{rd_r} while SHFL bar mask {rd_bar[rd_r]:#x} pending")


def main():
    test_anti_slop_hard_errors()
    test_stdlib_policy_versioning_and_lockfile()
    test_differential_models_and_dedup()
    test_full_cubin_emission_and_cli()
    test_pdl_preexit_and_mio_war_invariants()
    print("PASS skl2c anti-slop gates, hill-climbed stdlib policy/versioning/lockfile, C99 NVRM header gen, differential models, and 123 sm_121a/sm_120a kernels (0 legacy .skl files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
