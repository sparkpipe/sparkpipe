import hashlib
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKLC = ROOT / "tools" / "sklc" / "sklc.py"
SKL_DIR = ROOT / "inference" / "kernels" / "skl"
SUITES = [
    "skinny_decode.skl",
    "mx_gemm.skl",
    "norm_act.skl",
    "stage_ops.skl",
    "attn_decode.skl",
    "moe_route.skl",
    "tp_mesh_mem.skl",
]
EXPECTED_KERNELS = [
    "skinny_nvfp4_b1",
    "skinny_mxfp4_b1",
    "skinny_bf16_b1",
    "mx_quantize_e4m3_ue8m0",
    "rmsnorm_bf16_vec16",
    "fused_residual_rmsnorm_bf16x8",
    "swiglu_packed_bf16x8",
    "swiglu_pdl_exact_bf16x8",
    "rmsnorm_pdl_exact_5120_bf16x8",
    "fused_residual_rmsnorm_pdl_exact_5120_bf16x8",
    "head_rmsnorm_silu_gate_pdl_exact_128_bf16",
    "head_rmsnorm_sigmoid_gate_pdl_exact_128_bf16",
    "fused_residual_rmsnorm_f32_generic_pdl_exact_bf16x8",
    "lm_output_gate_pdl_exact_bf16x8",
    "lm_sigmoid_rows_pdl_bf16x8",
    "lm_add_rows_pdl_bf16x8",
    "lm_copy_rows_pdl_bf16x8",
    "lm_add_rows_strided_pdl_bf16x8",
    "lm_copy_rows_strided_pdl_bf16x8",
    "lm_widen_bf16_to_f32_pdl",
    "attn_tp1_merge_sigmoid_gate_pdl_exact_bf16",
    "add_elements_pdl_exact_bf16x8",
    "gqa_split_reduce_sigmoid_gate_pdl_exact_bf16",
    "head_certified_reduce_f32",
    "head_commit_argmax_pdl_f32",
    "stage_embedding_lookup_pdl_bf16x8",
    "stage_embedding_alias_pdl_bf16x8",
    "stage_store_outputs_u32_f32",
    "stage_chain_advance_u32",
    "stage_last_rows_gather_pdl_bf16x8",
    "stage_last_rows_scatter_pdl_u32_f32",
    "stage_rows_gather_pdl_bf16x8",
    "stage_head_rank_token_pdl_u32",
    "stage_maxloc_pack_pdl_f32_u32",
    "stage_maxloc_unpack_pdl_u64",
    "stage_combine_pair_bf16",
    "stage_combine_u64_max",
    "stage_tap_copy_bf16",
    "stage_row_lanes_u32",
    "qwen_row_dot_pdl_bf16x8",
    "latent_attn_decode_512_64",
    "router_softmax_256",
    "router_softmax_bf16_256",
    "tp_f32_to_bf16_vec16",
    "tp_bf16_to_f32_vec16",
    "device_program_copy_vec256",
    "device_program_copy_exact",
    "device_program_fill_exact",
]


def verify_arch(arch, expected_sm):
    with tempfile.TemporaryDirectory() as tmp:
        cubin_path = pathlib.Path(tmp) / f"hot6_{arch}.cubin"
        manifest_path = pathlib.Path(tmp) / f"hot6_{arch}.manifest"
        cmd = [
            sys.executable,
            str(SKLC),
            "--arch",
            arch,
            *[str(SKL_DIR / s) for s in SUITES],
            "-o",
            str(cubin_path),
            "--manifest",
            str(manifest_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"sklc failed for {arch}: {res.stderr}"
        blob = cubin_path.read_bytes()
        assert blob[:4] == b"\x7fELF", f"bad ELF magic for {arch}"
        e_machine = struct.unpack_from("<H", blob, 18)[0]
        assert e_machine == 190, f"expected EM_CUDA=190, got {e_machine}"
        e_flags = struct.unpack_from("<I", blob, 48)[0]
        assert ((e_flags >> 8) & 0xFF) == expected_sm, f"expected sm {expected_sm}, got {(e_flags >> 8) & 0xFF}"
        mtxt = manifest_path.read_text()
        digest = hashlib.sha256(blob).hexdigest()
        assert f"cubin {digest} {len(blob)}" in mtxt, f"manifest digest/size mismatch for {arch}"
        for kname in EXPECTED_KERNELS:
            assert f"kernel {kname} " in mtxt, f"missing kernel {kname} in {arch} manifest"
        if arch == "sm121":
            for tool, args in (("cuobjdump", ["--dump-sass", str(cubin_path)]), ("nvdisasm", [str(cubin_path)])):
                exe = shutil.which(tool)
                if exe is None:
                    continue
                dis = subprocess.run([exe, *args], capture_output=True, text=True, check=False)
                assert dis.returncode == 0, f"{tool} failed on {arch} cubin: {dis.stderr}"
                for line in dis.stdout.splitlines():
                    upper = line.upper()
                    assert not any(tok in upper for tok in ("UNKNOWN", "ILLEGAL", "INVALID", "???")), (
                        f"{tool} found invalid sm_121 opcode: {line}"
                    )


def verify_long_name_manifest():
    long_name = "k_" + ("a" * 80)
    src = f"kernel[256, 32] {long_name}(global u32* dst, global u32* src, u32 n) {{ reg u32 i = ctaid_x() * 256 + tid_x(); if (i >= n) {{ return; }} reg u32 v = ldg32_stream(ptr_add(src, i, 4)); stg32(ptr_add(dst, i, 4), v); }}\n"
    expected_short = f"{long_name[:51]}_{hashlib.sha256(long_name.encode('utf-8')).hexdigest()[:12]}"
    with tempfile.TemporaryDirectory() as tmp:
        skl_path = pathlib.Path(tmp) / "long.skl"
        cubin_path = pathlib.Path(tmp) / "long.cubin"
        manifest_path = pathlib.Path(tmp) / "long.manifest"
        skl_path.write_text(src)
        cmd = [sys.executable, str(SKLC), "--arch", "sm121", str(skl_path), "-o", str(cubin_path), "--manifest", str(manifest_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"sklc failed for long kernel name: {res.stderr}"
        mtxt = manifest_path.read_text()
        assert f"kernel {long_name} short {expected_short} " in mtxt, f"expected NVRM hashed short name {expected_short} in manifest:\n{mtxt}"
    mangled = "_Z26SampleLongTemplateKernelOpILj24ELj256ELj14336EEvPKfPKtPKjS5_S5_Ptj"
    expected_mangled = f"SampleLongTemplateKernelOp_{hashlib.sha256(mangled.encode('utf-8')).hexdigest()[:12]}"
    with tempfile.TemporaryDirectory() as tmp:
        skl_path = pathlib.Path(tmp) / "mangled.skl"
        cubin_path = pathlib.Path(tmp) / "mangled.cubin"
        manifest_path = pathlib.Path(tmp) / "mangled.manifest"
        skl_path.write_text(f"kernel[256, 32] {mangled}(global u32* dst, global u32* src, u32 n) {{ reg u32 i = ctaid_x() * 256 + tid_x(); if (i >= n) {{ return; }} reg u32 v = ldg32_stream(ptr_add(src, i, 4)); stg32(ptr_add(dst, i, 4), v); }}\n")
        cmd = [sys.executable, str(SKLC), "--arch", "sm121", str(skl_path), "-o", str(cubin_path), "--manifest", str(manifest_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"sklc failed for mangled long kernel name: {res.stderr}"
        mtxt = manifest_path.read_text()
        assert f"kernel {mangled} short {expected_mangled} " in mtxt, f"expected NVRM demangled short name {expected_mangled} in manifest:\n{mtxt}"


def verify_codegen_invariants():
    sys.path.insert(0, str(ROOT / "tools" / "sklc"))
    import isa_sm121 as isa
    from codegen import compile_program
    from frontend import parse_skl

    try:
        isa.encode_ldc(2, isa.C0_PARAM_BASE, width=128)
        raise AssertionError("encode_ldc must reject width=128 (LDC.INVALID6 on sm_121a)")
    except KeyError:
        pass

    for suite in SUITES:
        text = (SKL_DIR / suite).read_text(encoding="utf-8")
        kernels = compile_program(parse_skl(text))
        for k in kernels:
            assert k.stack_bytes == 0, f"unexpected stack usage in {k.name}: {k.stack_bytes}"
            if k.name == "latent_attn_decode_512_64":
                assert k.registers <= 80, f"expected <=80 regs for {k.name}, got {k.registers}"
            for idx, (lo, hi) in enumerate(k.insns):
                wr = (hi >> 46) & 0x7
                rd = (hi >> 49) & 0x7
                wait = (hi >> 52) & 0x3F
                if wr < 6:
                    assert (wait & (1 << wr)) == 0, f"self-wait on wr barrier {wr} in {k.name} at insn {idx}"
                if rd < 6:
                    assert (wait & (1 << rd)) == 0, f"self-wait on rd barrier {rd} in {k.name} at insn {idx}"
                if (lo & 0xFFFF) == 0x7B1D:
                    assert (hi & 0x10000) == 0, f"BAR.SYNC must not set DEFER_BLOCKING in {k.name}"
                    assert idx + 1 < len(k.insns) and (k.insns[idx + 1][0] & 0xFFFF) == 0x7918, f"BAR.SYNC must be followed by NOP in {k.name}"


def verify_new_builtins():
    src = """kernel[256, 32] builtin_smoke(global u32* out, u32 n, u8 flag) {
    reg u32 gy = grid_y();
    reg u32 bz = ctaid_z();
    reg u32 ny = ntid_y();
    reg u32 i = ctaid_x() * ntid_x() + tid_x();
    if (i >= n) { return; }
    reg f32 f = u32_to_f32(i + gy + bz + ny);
    reg f32 u = shfl_up(f, 1);
    reg u32 lo = f32_as_u32(u);
    reg u32 hi = flag & 0xFF;
    stg64(ptr_add(out, i, 8), pack_u32x2(lo, hi));
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        skl_path = pathlib.Path(tmp) / "builtin.skl"
        cubin_path = pathlib.Path(tmp) / "builtin.cubin"
        manifest_path = pathlib.Path(tmp) / "builtin.manifest"
        skl_path.write_text(src, encoding="utf-8")
        res = subprocess.run(
            [sys.executable, str(SKLC), "--arch", "sm121", str(skl_path), "-o", str(cubin_path), "--manifest", str(manifest_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"builtin_smoke compile failed: {res.stderr}"
        mtxt = manifest_path.read_text(encoding="utf-8")
        assert "param builtin_smoke 0 ptr align 8 bytes 8 offset 0" in mtxt, mtxt
        assert "param builtin_smoke 1 u32 align 4 bytes 4 offset 8" in mtxt, mtxt
        assert "param builtin_smoke 2 u8 align 1 bytes 1 offset 12" in mtxt, mtxt
        blob = cubin_path.read_bytes()
        assert struct.pack("<IHHI", 0, 2, 12, (1 << 18) | (0x1F << 12)) in blob, "expected exact 1-byte EIATTR_KPARAM_INFO size encoding"


def verify_rename_and_only():
    with tempfile.TemporaryDirectory() as tmp:
        cubin_path = pathlib.Path(tmp) / "overlay.cubin"
        manifest_path = pathlib.Path(tmp) / "overlay.manifest"
        cmd = [
            sys.executable,
            str(SKLC),
            "--arch",
            "sm121a",
            "--only",
            "swiglu_pdl_exact_bf16x8",
            "--only",
            "fused_residual_rmsnorm_pdl_exact_5120_bf16x8",
            "--only",
            "head_rmsnorm_silu_gate_pdl_exact_128_bf16",
            "--rename",
            "swiglu_pdl_exact_bf16x8=_Z17TestSiluMulKernelILj256EEvPKtPtj",
            "--rename",
            "fused_residual_rmsnorm_pdl_exact_5120_bf16x8=_Z17TestRmsNormKernelILj5120ELb1EEvPtPKtPKfS0_f",
            "--rename",
            "head_rmsnorm_silu_gate_pdl_exact_128_bf16=_Z28TestHeadRmsNormGateKernelILb0EEvPKtS1_S1_Ptjjjj",
            str(SKL_DIR / "norm_act.skl"),
            "-o",
            str(cubin_path),
            "--manifest",
            str(manifest_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"sklc --only/--rename failed: {res.stderr}"
        mtxt = manifest_path.read_text()
        kernel_lines = [line for line in mtxt.splitlines() if line.startswith("kernel ")]
        assert len(kernel_lines) == 3, f"expected exactly 3 selected kernels, got {len(kernel_lines)}:\n{mtxt}"
        for target in (
            "_Z17TestSiluMulKernelILj256EEvPKtPtj",
            "_Z17TestRmsNormKernelILj5120ELb1EEvPtPKtPKfS0_f",
            "_Z28TestHeadRmsNormGateKernelILb0EEvPKtS1_S1_Ptjjjj",
        ):
            assert f"kernel {target} " in mtxt, f"missing renamed kernel {target} in manifest:\n{mtxt}"
        assert "rmsnorm_bf16_vec16" not in mtxt, f"unselected kernel leaked into manifest:\n{mtxt}"


def main():
    verify_arch("sm121", 121)
    verify_arch("sm120", 120)
    verify_long_name_manifest()
    verify_rename_and_only()
    verify_codegen_invariants()
    verify_new_builtins()
    print(f"PASS sklc compiled {len(EXPECTED_KERNELS)} kernels cleanly for sm121 and sm120")


if __name__ == "__main__":
    main()
