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
HOSTED_BIN = ROOT / "build" / "test_nvrm_hosted_launch"
NVRM_LIB = ROOT / "build" / "runtime_libs" / "libsparkpipe_device_nvrm.so"
FORBIDDEN_LIBS = ("libcuda", "libcudart", "libnvidia")
PROD_OVERLAY_KERNELS = (
    ("swiglu_pdl_exact_bf16x8", "_Z17QwenSiluMulKernelILj256EEvPKtPtj"),
    ("rmsnorm_pdl_exact_5120_bf16x8", "_Z17QwenRmsNormKernelILj5120ELb0EEvPtPKtPKfS0_f"),
    ("fused_residual_rmsnorm_pdl_exact_5120_bf16x8", "_Z17QwenRmsNormKernelILj5120ELb1EEvPtPKtPKfS0_f"),
    ("head_rmsnorm_silu_gate_pdl_exact_128_bf16", "_Z28QwenGdnHeadRmsNormGateKernelILb0EEvPKtS1_S1_Ptjjjj"),
    ("head_rmsnorm_sigmoid_gate_pdl_exact_128_bf16", "_Z28QwenGdnHeadRmsNormGateKernelILb1EEvPKtS1_S1_Ptjjjj"),
    ("fused_residual_rmsnorm_f32_generic_pdl_exact_bf16x8", "_Z28LmFusedResidualRmsNormKernelILj256EfLb0EEvPKtS1_PKT0_PtS5_jjf"),
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
    ("head_commit_argmax_pdl_f32", "_Z18LmHeadCommitKernelILj256EEvPKfPKjjPjPfj"),
    ("stage_embedding_lookup_pdl_bf16x8", "_Z22LmStageEmbeddingKernelILj256EEvPKtPKjPtjjj"),
    ("stage_embedding_alias_pdl_bf16x8", "_Z27LmStageEmbeddingAliasKernelILj256EEvPKtPKjPtjjjjjj"),
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
    "_Z17QwenSiluMulKernelILj256EEvPKtPtj": (3, 20),
    "_Z17QwenRmsNormKernelILj5120ELb0EEvPtPKtPKfS0_f": (5, 36),
    "_Z17QwenRmsNormKernelILj5120ELb1EEvPtPKtPKfS0_f": (5, 36),
    "_Z28QwenGdnHeadRmsNormGateKernelILb0EEvPKtS1_S1_Ptjjjj": (8, 48),
    "_Z28QwenGdnHeadRmsNormGateKernelILb1EEvPKtS1_S1_Ptjjjj": (8, 48),
    "_Z28LmFusedResidualRmsNormKernelILj256EfLb0EEvPKtS1_PKT0_PtS5_jjf": (8, 52),
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
    "_Z18LmHeadCommitKernelILj256EEvPKfPKjjPjPfj": (6, 44),
    "_Z22LmStageEmbeddingKernelILj256EEvPKtPKjPtjjj": (6, 36),
    "_Z27LmStageEmbeddingAliasKernelILj256EEvPKtPKjPtjjjjjj": (9, 48),
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
    cmd.extend([str(NORM_ACT_SKL), str(STAGE_OPS_SKL), str(TP_MESH_MEM_SKL), "-o", str(cubin_path), "--manifest", str(manifest_path)])
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
