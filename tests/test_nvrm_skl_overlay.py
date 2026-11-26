#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKLC = ROOT / "tools" / "sklc" / "sklc.py"
NORM_ACT_SKL = ROOT / "inference" / "kernels" / "skl" / "norm_act.skl"
STAGE_OPS_SKL = ROOT / "inference" / "kernels" / "skl" / "stage_ops.skl"
TP_MESH_MEM_SKL = ROOT / "inference" / "kernels" / "skl" / "tp_mesh_mem.skl"
SKINNY_DECODE_SKL = ROOT / "inference" / "kernels" / "skl" / "skinny_decode.skl"
LINEAR_ATTN_SKL = ROOT / "inference" / "kernels" / "skl" / "linear_attn.skl"
HOSTED_BIN = ROOT / "build" / "test_nvrm_hosted_launch"
NVRM_LIB = ROOT / "build" / "runtime_libs" / "libsparkpipe_device_nvrm.so"
FORBIDDEN_LIBS = ("libcuda", "libcudart", "libnvidia")
PROD_OVERLAY_KERNELS = (
    ("skinny_bf16_dense_l32_r1_n1_pdl", "_Z14LmSkinnyKernelI12LmBf16FormatLj32ELj1ELj1EEv17LmSkinnyArguments"),
    ("skinny_bf16_multi_l32_r1_n1_pdl", "_Z19LmSkinnyMultiKernelI12LmBf16FormatLj32ELj1ELj1EEv22LmSkinnyMultiArguments"),
    ("skinny_bf16_split_r1_pdl", "_Z19LmSkinnySplitKernelI12LmBf16FormatLj1EEv17LmSkinnyArguments"),
    ("delta_rule_decode1_256_128_128_4_pdl", "_Z24LmDeltaRuleDecode1KernelILj256ELj128ELj128ELj4EEvPhjPKjS2_PKtS4_S4_PKfS6_PtjjjjS2_"),
    ("gdn_conv_split_tiled_4_128_pdl", "_Z27QwenGdnConvSplitTiledKernelILj4ELj128EEvPtPKjS2_S2_PKtS4_S0_S0_S0_S0_jjjjjjS2_"),
    ("causal_conv_serial_256_4_swish_bf16_pdl", "_Z18LmCausalConvKernelILj256ELj4ELj1EtLj1EEvPtPKjS2_S2_PKtPKT2_S0_jjjS2_S2_"),
    ("causal_conv_tiled_4_swish_bf16_pdl", "_Z23LmCausalConvTiledKernelILj4ELj1EtLj1EEvPtPKjS2_S2_PKtPKT1_S0_jjjS2_"),
    ("swiglu_pdl_exact_bf16x8", "_Z17QwenSiluMulKernelILj256EEvPKtPtj"),
    ("rmsnorm_pdl_exact_5120_bf16x8", "_Z17QwenRmsNormKernelILj5120ELb0EEvPtPKtPKfS0_f"),
    ("fused_residual_rmsnorm_pdl_exact_5120_bf16x8", "_Z17QwenRmsNormKernelILj5120ELb1EEvPtPKtPKfS0_f"),
    ("head_rmsnorm_silu_gate_pdl_exact_128_bf16", "_Z28QwenGdnHeadRmsNormGateKernelILb0EEvPKtS1_S1_Ptjjjj"),
    ("head_rmsnorm_sigmoid_gate_pdl_exact_128_bf16", "_Z28QwenGdnHeadRmsNormGateKernelILb1EEvPKtS1_S1_Ptjjjj"),
    ("fused_residual_rmsnorm_f32_generic_pdl_exact_bf16x8", "_Z28LmFusedResidualRmsNormKernelILj256EfLb0EEvPKtS1_PKT0_PtS5_jjf"),
    ("fused_residual_rmsnorm_f32_round_pdl_bf16x8", "_Z28LmFusedResidualRmsNormKernelILj256EfLb1EEvPKtS1_PKT0_PtS5_jjf"),
    ("fused_residual_rmsnorm_bf16_pdl_bf16x8", "_Z28LmFusedResidualRmsNormKernelILj256EtLb0EEvPKtS1_PKT0_PtS5_jjf"),
    ("fused_residual_rmsnorm_bf16_round_pdl_bf16x8", "_Z28LmFusedResidualRmsNormKernelILj256EtLb1EEvPKtS1_PKT0_PtS5_jjf"),
    ("bf16_rmsnorm_pdl_bf16x8", "_Z19LmBf16RmsNormKernelILj256EEvPKtS1_Ptjjf"),
    ("centered_rmsnorm_pdl_bf16x8", "_Z23LmCenteredRmsNormKernelILj256EEvPKtS1_Ptjjf"),
    ("lm_silu_mul_pdl_bf16x8", "_Z15LmSiluMulKernelILj256EEvPKtPtjb"),
    ("head_rmsnorm_f32_pdl_bf16", "_Z19LmHeadRmsNormKernelILj256EfLb0EEvPKtPKT0_Ptjjjff"),
    ("head_rmsnorm_bf16_pdl_bf16", "_Z19LmHeadRmsNormKernelILj256EtLb0EEvPKtPKT0_Ptjjjff"),
    ("head_rmsnorm_silu_gate_generic_pdl", "_Z27LmHeadRmsNormSiluGateKernelILj256EEvPKtS1_S1_Ptjjjf"),
    ("head_rmsnorm_sigmoid_gate_generic_pdl", "_Z30LmHeadRmsNormSigmoidGateKernelILj256EEvPKtS1_S1_Ptjjjf"),
    ("head_rmsnorm_sigmoid_gate_once_generic_pdl", "_Z34LmHeadRmsNormSigmoidGateOnceKernelILj256EEvPKtS1_S1_Ptjjjf"),
    ("lm_gather_rows_pdl_bf16x8", "_Z18LmGatherRowsKernelILj256EEvPKtPKjPtjj"),
    ("lm_scatter_rows_pdl_bf16x8", "_Z19LmScatterRowsKernelILj256EEvPKtPKjPtjj"),
    ("lm_slice_columns_pdl_bf16x8", "_Z20LmSliceColumnsKernelILj256EEvPKtPtjjjj"),
    ("lm_concat_rows_pdl_bf16x8", "_Z18LmConcatRowsKernelPKtS0_Ptjjjj"),
    ("lm_rank_major_rows_pdl_bf16x8", "_Z21LmRankMajorRowsKernelILj256EEvPKtPtjjjj"),
    ("lm_scale_elements_bf16_pdl", "_Z21LmScaleElementsKernelItEvPT_mf"),
    ("lm_scale_elements_f32_pdl", "_Z21LmScaleElementsKernelIfEvPT_mf"),
    ("lm_output_gate_pdl_exact_bf16x8", "_Z18LmOutputGateKernelILj256ELb0EEvPtPKtj"),
    ("lm_sigmoid_rows_pdl_bf16x8", "_Z19LmSigmoidRowsKernelILj256EEvPKtPfj"),
    ("lm_add_rows_pdl_bf16x8", "_Z15LmAddRowsKernelILj256EEvPKtS1_Ptjj"),
    ("lm_copy_rows_pdl_bf16x8", "_Z16LmCopyRowsKernelILj256EEvPKtPtjj"),
    ("lm_add_rows_strided_pdl_bf16x8", "_Z22LmAddRowsStridedKernelILj256EEvPKtS1_Ptjjm"),
    ("lm_copy_rows_strided_pdl_bf16x8", "_Z23LmCopyRowsStridedKernelILj256EEvPKtPtjjm"),
    ("lm_widen_bf16_to_f32_pdl", "_Z22LmWidenBf16ToF32KernelPKtPfm"),
    ("attn_tp1_merge_sigmoid_gate_pdl_exact_bf16", "_Z28QwenHybridTp1MergeGateKernelILj24ELj256ELj14336EEvPKfPKtPtj"),
    ("add_elements_pdl_exact_bf16x8", "_Z19LmAddElementsKernelPtPKtm"),
    ("gqa_split_reduce_sigmoid_gate_pdl_exact_bf16", "_Z28QwenGqaSplitReduceGateKernelILj24ELj256ELj14336EEvPKfPKtPKjS5_S5_Ptj"),
    ("head_certified_reduce_f32", "_Z32SparkLmHeadCertifiedReduceKernelPKfPKjPjPfj"),
    ("head_certified_rescore_f32_bf16", "_Z33SparkLmHeadCertifiedRescoreKernelPKvS0_PKjS2_PfPjj"),
    ("head_certified_fp8_score_1024", "_Z34SparkLmHeadCertifiedFp8ScoreKernelPKvPKhPKfS4_S4_PfS5_jj"),
    ("head_candidate_256_1024_pdl_bf16", "_Z21LmHeadCandidateKernelILj256ELj1024EEvPKtS1_PKjPfPjjjj"),
    ("head_commit_argmax_pdl_f32", "_Z18LmHeadCommitKernelILj256EEvPKfPKjjPjPfj"),
    ("head_softmax_pdl_f32", "_Z19LmHeadSoftmaxKernelILj256EEvPfjjf"),
    ("stage_embedding_lookup_pdl_bf16x8", "_Z22LmStageEmbeddingKernelILj256EEvPKtPKjPtjjj"),
    ("stage_embedding_alias_pdl_bf16x8", "_Z27LmStageEmbeddingAliasKernelILj256EEvPKtPKjPtjjjjjj"),
    ("stage_embedding_whole_pdl_bf16x8", "_Z27LmStageEmbeddingWholeKernelILj256EEvPKtPKjPtjjjjj"),
    ("stage_store_outputs_u32_f32", "_Z29StageRunnerStoreOutputsKernelPjPfPKjPKfS2_j"),
    ("stage_chain_advance_u32", "_Z29StageRunnerChainAdvanceKernelPjPKjS_S_S1_jS_S_j"),
    ("stage_last_rows_gather_pdl_bf16x8", "_Z27LmStageLastRowsGatherKernelPKjS0_PKtPtj"),
    ("stage_last_rows_scatter_pdl_u32_f32", "_Z28LmStageLastRowsScatterKernelPKjS0_S0_PKfPjPfj"),
    ("stage_rows_gather_pdl_bf16x8", "_Z23LmStageRowsGatherKernelPKjPKtPtj"),
    ("stage_head_rank_token_pdl_u32", "_Z26LmStageHeadRankTokenKernelPjjj"),
    ("stage_maxloc_pack_pdl_f32_u32", "_Z23LmStageMaxlocPackKernelPKfPKjPmj"),
    ("stage_maxloc_unpack_pdl_u64", "_Z25LmStageMaxlocUnpackKernelPKmPjPfj"),
    ("stage_combine_pair_bf16", "_Z28StageRunnerCombinePairKernelPKtPtj"),
    ("stage_combine_u64_max", "_Z30StageRunnerCombineU64MaxKernelPKmPmj"),
    ("stage_tap_copy_bf16", "_Z24StageRunnerTapCopyKernelPKtPtjjm"),
    ("stage_row_lanes_u32", "_Z16LmRowLanesKernelPKjS0_PjS1_"),
    ("stage_row_dot_pdl_bf16x8", "_Z22QwenHybridRowDotKernelILj256EEvPKtS1_Ptj"),
    ("device_program_copy_exact", "SparkDeviceProgramCopy"),
    ("device_program_fill_exact", "SparkDeviceProgramFill"),
)
PROD_OVERLAY_ABI_BYTES = {
    "_Z14LmSkinnyKernelI12LmBf16FormatLj32ELj1ELj1EEv17LmSkinnyArguments": (1, 216),
    "_Z19LmSkinnyMultiKernelI12LmBf16FormatLj32ELj1ELj1EEv22LmSkinnyMultiArguments": (1, 888),
    "_Z19LmSkinnySplitKernelI12LmBf16FormatLj1EEv17LmSkinnyArguments": (1, 216),
    "_Z24LmDeltaRuleDecode1KernelILj256ELj128ELj128ELj4EEvPhjPKjS2_PKtS4_S4_PKfS6_PtjjjjS2_": (15, 104),
    "_Z27QwenGdnConvSplitTiledKernelILj4ELj128EEvPtPKjS2_S2_PKtS4_S0_S0_S0_S0_jjjjjjS2_": (17, 112),
    "_Z18LmCausalConvKernelILj256ELj4ELj1EtLj1EEvPtPKjS2_S2_PKtPKT2_S0_jjjS2_S2_": (12, 88),
    "_Z23LmCausalConvTiledKernelILj4ELj1EtLj1EEvPtPKjS2_S2_PKtPKT1_S0_jjjS2_": (11, 80),
    "_Z17QwenSiluMulKernelILj256EEvPKtPtj": (3, 20),
    "_Z17QwenRmsNormKernelILj5120ELb0EEvPtPKtPKfS0_f": (5, 36),
    "_Z17QwenRmsNormKernelILj5120ELb1EEvPtPKtPKfS0_f": (5, 36),
    "_Z28QwenGdnHeadRmsNormGateKernelILb0EEvPKtS1_S1_Ptjjjj": (8, 48),
    "_Z28QwenGdnHeadRmsNormGateKernelILb1EEvPKtS1_S1_Ptjjjj": (8, 48),
    "_Z28LmFusedResidualRmsNormKernelILj256EfLb0EEvPKtS1_PKT0_PtS5_jjf": (8, 52),
    "_Z28LmFusedResidualRmsNormKernelILj256EfLb1EEvPKtS1_PKT0_PtS5_jjf": (8, 52),
    "_Z28LmFusedResidualRmsNormKernelILj256EtLb0EEvPKtS1_PKT0_PtS5_jjf": (8, 52),
    "_Z28LmFusedResidualRmsNormKernelILj256EtLb1EEvPKtS1_PKT0_PtS5_jjf": (8, 52),
    "_Z19LmBf16RmsNormKernelILj256EEvPKtS1_Ptjjf": (6, 36),
    "_Z23LmCenteredRmsNormKernelILj256EEvPKtS1_Ptjjf": (6, 36),
    "_Z15LmSiluMulKernelILj256EEvPKtPtjb": (4, 21),
    "_Z19LmHeadRmsNormKernelILj256EfLb0EEvPKtPKT0_Ptjjjff": (8, 44),
    "_Z19LmHeadRmsNormKernelILj256EtLb0EEvPKtPKT0_Ptjjjff": (8, 44),
    "_Z27LmHeadRmsNormSiluGateKernelILj256EEvPKtS1_S1_Ptjjjf": (8, 48),
    "_Z30LmHeadRmsNormSigmoidGateKernelILj256EEvPKtS1_S1_Ptjjjf": (8, 48),
    "_Z34LmHeadRmsNormSigmoidGateOnceKernelILj256EEvPKtS1_S1_Ptjjjf": (8, 48),
    "_Z18LmGatherRowsKernelILj256EEvPKtPKjPtjj": (5, 32),
    "_Z19LmScatterRowsKernelILj256EEvPKtPKjPtjj": (5, 32),
    "_Z20LmSliceColumnsKernelILj256EEvPKtPtjjjj": (6, 32),
    "_Z18LmConcatRowsKernelPKtS0_Ptjjjj": (7, 40),
    "_Z21LmRankMajorRowsKernelILj256EEvPKtPtjjjj": (6, 32),
    "_Z21LmScaleElementsKernelItEvPT_mf": (3, 20),
    "_Z21LmScaleElementsKernelIfEvPT_mf": (3, 20),
    "_Z18LmOutputGateKernelILj256ELb0EEvPtPKtj": (3, 20),
    "_Z19LmSigmoidRowsKernelILj256EEvPKtPfj": (3, 20),
    "_Z15LmAddRowsKernelILj256EEvPKtS1_Ptjj": (5, 32),
    "_Z16LmCopyRowsKernelILj256EEvPKtPtjj": (4, 24),
    "_Z22LmAddRowsStridedKernelILj256EEvPKtS1_Ptjjm": (6, 40),
    "_Z23LmCopyRowsStridedKernelILj256EEvPKtPtjjm": (5, 32),
    "_Z22LmWidenBf16ToF32KernelPKtPfm": (3, 24),
    "_Z28QwenHybridTp1MergeGateKernelILj24ELj256ELj14336EEvPKfPKtPtj": (4, 28),
    "_Z19LmAddElementsKernelPtPKtm": (3, 24),
    "_Z28QwenGqaSplitReduceGateKernelILj24ELj256ELj14336EEvPKfPKtPKjS5_S5_Ptj": (7, 52),
    "_Z32SparkLmHeadCertifiedReduceKernelPKfPKjPjPfj": (5, 36),
    "_Z33SparkLmHeadCertifiedRescoreKernelPKvS0_PKjS2_PfPjj": (7, 52),
    "_Z34SparkLmHeadCertifiedFp8ScoreKernelPKvPKhPKfS4_S4_PfS5_jj": (9, 64),
    "_Z21LmHeadCandidateKernelILj256ELj1024EEvPKtS1_PKjPfPjjjj": (8, 52),
    "_Z18LmHeadCommitKernelILj256EEvPKfPKjjPjPfj": (6, 44),
    "_Z19LmHeadSoftmaxKernelILj256EEvPfjjf": (4, 20),
    "_Z22LmStageEmbeddingKernelILj256EEvPKtPKjPtjjj": (6, 36),
    "_Z27LmStageEmbeddingAliasKernelILj256EEvPKtPKjPtjjjjjj": (9, 48),
    "_Z27LmStageEmbeddingWholeKernelILj256EEvPKtPKjPtjjjjj": (8, 44),
    "_Z29StageRunnerStoreOutputsKernelPjPfPKjPKfS2_j": (6, 44),
    "_Z29StageRunnerChainAdvanceKernelPjPKjS_S_S1_jS_S_j": (9, 68),
    "_Z27LmStageLastRowsGatherKernelPKjS0_PKtPtj": (5, 36),
    "_Z28LmStageLastRowsScatterKernelPKjS0_S0_PKfPjPfj": (7, 52),
    "_Z23LmStageRowsGatherKernelPKjPKtPtj": (4, 28),
    "_Z26LmStageHeadRankTokenKernelPjjj": (3, 16),
    "_Z23LmStageMaxlocPackKernelPKfPKjPmj": (4, 28),
    "_Z25LmStageMaxlocUnpackKernelPKmPjPfj": (4, 28),
    "_Z28StageRunnerCombinePairKernelPKtPtj": (3, 20),
    "_Z30StageRunnerCombineU64MaxKernelPKmPmj": (3, 20),
    "_Z24StageRunnerTapCopyKernelPKtPtjjm": (5, 32),
    "_Z16LmRowLanesKernelPKjS0_PjS1_": (4, 32),
    "_Z22QwenHybridRowDotKernelILj256EEvPKtS1_Ptj": (4, 28),
    "SparkDeviceProgramCopy": (3, 24),
    "SparkDeviceProgramFill": (3, 24),
}

SKL_EXACT = """kernel[128, 32] _Z10HostedStepILj7EEvPjj(global u32* output, u32 base) {
    reg u32 index = ctaid_x() * 128 + tid_x();
    reg u32 val = base + index * 7;
    stg32(ptr_add(output, index, 4), val);
}
"""

SKL_SENTINEL = """kernel[128, 32] _Z10HostedStepILj7EEvPjj(global u32* output, u32 base) {
    reg u32 index = ctaid_x() * 128 + tid_x();
    reg u32 val = base + index * 7 + 1;
    stg32(ptr_add(output, index, 4), val);
}
"""

SKL_BAD_ABI = """kernel[128, 32] _Z10HostedStepILj7EEvPjj(global u32* output, u32 base, u32 extra) {
    reg u32 index = ctaid_x() * 128 + tid_x();
    reg u32 val = base + index * 7 + extra;
    stg32(ptr_add(output, index, 4), val);
}
"""


def compile_overlay(directory, stem, source):
    skl_path = directory / f"{stem}.skl"
    cubin_path = directory / f"{stem}.cubin"
    manifest_path = directory / f"{stem}.manifest"
    skl_path.write_text(source, encoding="utf-8")
    res = subprocess.run(
        [sys.executable, str(SKLC), "--arch", "sm121", str(skl_path), "-o", str(cubin_path), "--manifest", str(manifest_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"sklc failed for {stem}: {res.stderr}"
    skl_path.unlink()


def check_zero_nvidia_ldd(path):
    res = subprocess.run(["ldd", str(path)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"ldd failed for {path}: {res.stderr}"
    lower = res.stdout.lower()
    for bad in FORBIDDEN_LIBS:
        assert bad not in lower, f"forbidden library {bad} linked in {path}:\n{res.stdout}"


def run_hosted(env_extra):
    env = dict(os.environ)
    env.pop("SPARKPIPE_KERNEL_PROVIDER", None)
    env.pop("SPARKPIPE_SKL_CUBIN_DIR", None)
    env.pop("SPARKPIPE_SKL_OVERLAY_STATS", None)
    env.update(env_extra)
    return subprocess.run([str(HOSTED_BIN)], env=env, capture_output=True, text=True, check=False)


def verify_multi_kernel_overlay_bundle(directory):
    cubin_path = directory / "prod_overlay.cubin"
    manifest_path = directory / "prod_overlay.manifest"
    cmd = [sys.executable, str(SKLC), "--arch", "sm121a"]
    for src_name, dst_name in PROD_OVERLAY_KERNELS:
        cmd.extend(["--only", src_name, "--rename", f"{src_name}={dst_name}"])
    cmd.extend([str(SKINNY_DECODE_SKL), str(LINEAR_ATTN_SKL), str(NORM_ACT_SKL), str(STAGE_OPS_SKL), str(TP_MESH_MEM_SKL), "-o", str(cubin_path), "--manifest", str(manifest_path)])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"sklc multi-kernel overlay bundle failed: {res.stderr}"
    mtxt = manifest_path.read_text(encoding="utf-8")
    params_by_kernel = {}
    for line in mtxt.splitlines():
        if line.startswith("param "):
            parts = line.split()
            kname = parts[1]
            pbytes = int(parts[7])
            poff = int(parts[9])
            params_by_kernel.setdefault(kname, []).append((poff, pbytes))
    for _, dst_name in PROD_OVERLAY_KERNELS:
        assert f"kernel {dst_name} " in mtxt, f"missing {dst_name} in multi-kernel overlay manifest"
        assert f"spill_stores 0 spill_loads 0" in [l for l in mtxt.splitlines() if l.startswith(f"kernel {dst_name} ")][0], (
            f"unexpected spills in {dst_name}"
        )
        plist = params_by_kernel.get(dst_name, [])
        exp_count, exp_end = PROD_OVERLAY_ABI_BYTES[dst_name]
        actual_end = max(o + b for o, b in plist) if plist else 0
        assert (len(plist), actual_end) == (exp_count, exp_end), (
            f"{dst_name}: expected ABI ({exp_count} params, {exp_end}B), got ({len(plist)}, {actual_end}B)"
        )


def main():
    if not pathlib.Path("/dev/nvidiactl").exists():
        with tempfile.TemporaryDirectory(prefix="spark-skl-overlay-bundle-") as tmp:
            verify_multi_kernel_overlay_bundle(pathlib.Path(tmp))
        print(f"PASS skl-overlay production bundle verified {len(PROD_OVERLAY_KERNELS)} kernels with exact ABI and 0 spills")
        print("SKIP live skl-overlay silicon test (no /dev/nvidiactl on this host)")
        return 0
    if not HOSTED_BIN.exists() or not NVRM_LIB.exists():
        built = subprocess.run(["make", "-s", "build/test_nvrm_hosted_launch"], cwd=ROOT, capture_output=True, text=True, check=False)
        assert built.returncode == 0, f"failed to build build/test_nvrm_hosted_launch:\n{built.stdout}\n{built.stderr}"
    assert HOSTED_BIN.exists(), f"missing {HOSTED_BIN}; build build/test_nvrm_hosted_launch first"
    assert NVRM_LIB.exists(), f"missing {NVRM_LIB}; build nvrm-device first"
    check_zero_nvidia_ldd(HOSTED_BIN)
    check_zero_nvidia_ldd(NVRM_LIB)

    with tempfile.TemporaryDirectory(prefix="spark-skl-overlay-") as tmp:
        root = pathlib.Path(tmp)
        exact_dir = root / "exact"
        sentinel_dir = root / "sentinel"
        bad_abi_dir = root / "bad_abi"
        exact_dir.mkdir()
        sentinel_dir.mkdir()
        bad_abi_dir.mkdir()
        compile_overlay(exact_dir, "hosted_step", SKL_EXACT)
        compile_overlay(sentinel_dir, "hosted_step", SKL_SENTINEL)
        compile_overlay(bad_abi_dir, "hosted_step", SKL_BAD_ABI)
        verify_multi_kernel_overlay_bundle(exact_dir)

        res_exact = run_hosted({
            "SPARKPIPE_KERNEL_PROVIDER": "skl-overlay",
            "SPARKPIPE_SKL_CUBIN_DIR": str(exact_dir),
            "LD_DEBUG": "files",
        })
        assert res_exact.returncode == 0 and "PASS LM_LAUNCH" in res_exact.stdout, (
            f"exact skl-overlay run failed:\nstdout={res_exact.stdout}\nstderr={res_exact.stderr}"
        )
        assert "skl-overlay stats:" not in res_exact.stderr, f"stats printed while unset:\n{res_exact.stderr}"
        lower_dbg = res_exact.stderr.lower()
        for bad in FORBIDDEN_LIBS:
            assert bad not in lower_dbg, f"forbidden library {bad} mapped at runtime:\n{res_exact.stderr}"

        res_stats = run_hosted({
            "SPARKPIPE_KERNEL_PROVIDER": "skl-overlay",
            "SPARKPIPE_SKL_CUBIN_DIR": str(exact_dir),
            "SPARKPIPE_SKL_OVERLAY_STATS": "1",
        })
        assert res_stats.returncode == 0 and "PASS LM_LAUNCH" in res_stats.stdout, (
            f"stats skl-overlay run failed:\nstdout={res_stats.stdout}\nstderr={res_stats.stderr}"
        )
        assert "nvrm: skl-overlay: bound _Z10HostedStepILj7EEvPjj ->" in res_stats.stderr, (
            f"missing overlay bind line:\n{res_stats.stderr}"
        )
        assert "nvrm: skl-overlay stats: kernel=_Z10HostedStepILj7EEvPjj direct=3 program=0 total=3" in res_stats.stderr, (
            f"missing overlay stats launch count:\n{res_stats.stderr}"
        )

        res_sentinel = run_hosted({
            "SPARKPIPE_KERNEL_PROVIDER": "skl-overlay",
            "SPARKPIPE_SKL_CUBIN_DIR": str(sentinel_dir),
        })
        assert res_sentinel.returncode != 0 and "FAIL the template kernel ran on the queue it was given" in res_sentinel.stdout, (
            f"sentinel skl-overlay did not replace the kernel:\nstdout={res_sentinel.stdout}\nstderr={res_sentinel.stderr}"
        )

        res_bad_abi = run_hosted({
            "SPARKPIPE_KERNEL_PROVIDER": "skl-overlay",
            "SPARKPIPE_SKL_CUBIN_DIR": str(bad_abi_dir),
        })
        assert res_bad_abi.returncode != 0 and "ABI mismatch" in (res_bad_abi.stdout + res_bad_abi.stderr), (
            f"bad ABI overlay did not fail closed:\nstdout={res_bad_abi.stdout}\nstderr={res_bad_abi.stderr}"
        )

        res_default = run_hosted({
            "SPARKPIPE_KERNEL_PROVIDER": "default",
            "SPARKPIPE_SKL_CUBIN_DIR": str(sentinel_dir),
        })
        assert res_default.returncode == 0 and "PASS LM_LAUNCH" in res_default.stdout, (
            f"default provider did not bypass sentinel overlay:\nstdout={res_default.stdout}\nstderr={res_default.stderr}"
        )

    print("PASS skl-overlay transparently replaces nvcc kernels at runtime with 0 bit diffs, fail-closed ABI checks, and 0 libcuda/libcudart/libnvidia")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
