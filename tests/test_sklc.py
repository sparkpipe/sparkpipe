import hashlib
import pathlib
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
    "latent_attn_decode_512_64",
    "router_softmax_256",
    "router_softmax_bf16_256",
    "tp_f32_to_bf16_vec16",
    "tp_bf16_to_f32_vec16",
    "device_program_copy_vec256",
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


def main():
    verify_arch("sm121", 121)
    verify_arch("sm120", 120)
    print(f"PASS sklc compiled {len(EXPECTED_KERNELS)} kernels cleanly for sm121 and sm120")


if __name__ == "__main__":
    main()
