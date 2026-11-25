#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKLC = ROOT / "tools" / "sklc" / "sklc.py"
HOSTED_BIN = ROOT / "build" / "test_nvrm_hosted_launch"
NVRM_LIB = ROOT / "build" / "runtime_libs" / "libsparkpipe_device_nvrm.so"
FORBIDDEN_LIBS = ("libcuda", "libcudart", "libnvidia")

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
    env.update(env_extra)
    return subprocess.run([str(HOSTED_BIN)], env=env, capture_output=True, text=True, check=False)


def main():
    if not pathlib.Path("/dev/nvidiactl").exists():
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

        res_exact = run_hosted({
            "SPARKPIPE_KERNEL_PROVIDER": "skl-overlay",
            "SPARKPIPE_SKL_CUBIN_DIR": str(exact_dir),
            "LD_DEBUG": "files",
        })
        assert res_exact.returncode == 0 and "PASS LM_LAUNCH" in res_exact.stdout, (
            f"exact skl-overlay run failed:\nstdout={res_exact.stdout}\nstderr={res_exact.stderr}"
        )
        lower_dbg = res_exact.stderr.lower()
        for bad in FORBIDDEN_LIBS:
            assert bad not in lower_dbg, f"forbidden library {bad} mapped at runtime:\n{res_exact.stderr}"

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
