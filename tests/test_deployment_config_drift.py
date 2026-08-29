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

It also enforces the exact-member contracts the serving adapters apply at
load: the glm52 and glm5_next adapters validate their stage-config members
EXACTLY (SparkJsonValidateObjectMembersExact), so a generator that emits
one member fewer than the adapter's list produces configs the residentd
REJECTS (SCHEMA_ERROR) - that is the R3 flash-decode drift
(decode_split_context_threshold, 2026-08-29) this class of check catches
mechanically.

Stdlib only; regenerates into a temp dir; touches nothing.
"""
import json
import re
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


def adapter_members(relative_source: str, symbol: str) -> list[str]:
    """Extract the exact-member list the adapter validates against."""
    text = (ROOT / relative_source).read_text(encoding="utf-8")
    match = re.search(rf"{symbol}\s*\[\s*\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not match:
        raise RuntimeError(f"{symbol} not found in {relative_source}")
    return re.findall(r'"([^"]+)"', match.group(1))


def check_member_list(failures: list, family: str, source: str, symbol: str,
                      emitted: dict) -> None:
    want = adapter_members(source, symbol)
    got = list(emitted.keys())
    if got != want:
        failures.append(
            f"{family}: generator stage-config members {got} != adapter "
            f"exact-member list {want} ({source}:{symbol}); the adapter "
            f"rejects the generator's output SCHEMA_ERROR at load")


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

        # --- glm5_next: deployment/glm5_next_tp16 (17 JSONs) ---
        glm5_gen = scratch / "glm5_next"
        run(["python3", "tools/glm5_next_gen_deployment.py",
             "--output", str(glm5_gen)])
        glm5_tree = ROOT / "deployment/glm5_next_tp16"
        for relative in ["model_resident.json"] + [
                "config/stage_%02d.json" % i for i in range(16)]:
            problem = diff_bytes(glm5_gen / relative, glm5_tree / relative)
            if problem:
                failures.append(problem)
        stage = json.loads((glm5_gen / "config/stage_00.json").read_text())
        check_member_list(failures, "glm5_next",
                          "modules/glm5_next_resident_decode_stage/source/"
                          "spark_glm5_next_serving_adapter.c",
                          "SparkGlm5NextServingConfigurationMembers", stage)

        # --- glm52: no committed tree; the generator's output must still
        # satisfy the adapter's exact-member list (r3-flashdecode drift) ---
        glm52_gen = scratch / "glm52"
        run(["python3", "tools/glm52_gen_deployment.py", str(glm52_gen)])
        stage = json.loads(
            (glm52_gen / "spark8/config/glm52_stage.json").read_text())
        check_member_list(failures, "glm52",
                          "modules/glm52_resident_decode_stage/source/"
                          "spark_glm52_serving_adapter.c",
                          "SparkGlm52ServingConfigurationMembers", stage)

        # --- the generic spec family: every committed spec must still
        # generate (schema drift detector), and the two committed generator
        # twins must byte-match ---
        for spec in sorted((ROOT / "examples/deployments").glob("*.spec.json")):
            base = spec.name[:-len(".spec.json")]
            out = scratch / ("spec-%s.json" % base)
            run(["python3", "tools/generate_model_resident_deployment.py",
                 "--specification", str(spec), "--output", str(out)])
        for twin in ("dsv4_flash_pp13_host_rdma", "qwen36_pp13_host_rdma"):
            generated = scratch / ("spec-%s.json" % twin)
            committed = ROOT / "examples/deployments" / (twin + ".json")
            problem = diff_bytes(generated, committed)
            if problem:
                failures.append(problem)

    if failures:
        for failure in failures:
            print("FAIL " + failure)
        return 1
    print("deployment config drift PASS: k3 (17), glm5_next (17), glm52 "
          "adapter members, %d deployment specs + 2 committed twins"
          % len(list((ROOT / "examples/deployments").glob("*.spec.json"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
