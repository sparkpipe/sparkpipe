"""Every checked-in generated-config tree must byte-match its generator.

The K3 rot (Aug 2026): a checked-in config silently diverged from what its
generator emits, and nothing caught it until a human diffed by hand. This
gate is the permanent version of that hand diff: for EVERY family that
commits both a generator and the generator's output, regenerate into a
scratch directory and byte-compare. A failure means one of two things and
the fixer must say which:

  - the config is stale (the generator moved) - regenerate the config; or
  - the generator is stale (the config moved, e.g. a lane hand-edited the
    committed tree) - update the generator.

Stdlib only; regenerates into a temp dir; touches nothing.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEX = "0123456789abcdef"


def run(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(argv)} failed rc={result.returncode}: "
            f"{result.stdout[-400:]} {result.stderr[-400:]}")


def diff_bytes(generated: Path, committed: Path) -> str | None:
    if generated.read_bytes() == committed.read_bytes():
        return None
    return (f"{committed} diverges from its generator:\n"
            f"  regenerated: {generated}\n"
            f"  classify: (a) stale config - regenerate it, or (b) stale "
            f"generator - update the generator")


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory(prefix="cfgdrift") as scratch:
        scratch = Path(scratch)

        # --- K3: model_resident.json + the 16 per-rank adapter configs ---
        k3_gen = scratch / "k3"
        k3_gen.mkdir()
        run(["bash", "tools/k3_gen_deployment.sh",
             str(k3_gen / "model_resident.json")])
        run(["bash", "tools/k3_gen_adapter_configs.sh",
             str(k3_gen / "adapters"), "4"])
        k3_tree = ROOT / "modules/k3_resident_decode_stage/configs"
        generated = {("model_resident.json"): k3_gen / "model_resident.json"}
        for i in range(16):
            name = f"spark{HEX[i]}.json"
            generated[name] = k3_gen / "adapters" / name
        for name, path in generated.items():
            problem = diff_bytes(path, k3_tree / name)
            if problem:
                failures.append(problem)

    if failures:
        for failure in failures:
            print("FAIL " + failure)
        return 1
    print("deployment config drift PASS: k3 (17)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
