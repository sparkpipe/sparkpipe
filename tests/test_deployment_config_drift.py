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
import os
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


def adapter_optional_members(relative_source: str, function: str) -> list[str]:
    """Extract the optional members the adapter appends to its exact list."""
    text = (ROOT / relative_source).read_text(encoding="utf-8")
    match = re.search(rf"\b{function}\s*\([^)]*\)\s*\{{(.*?)\n\}}", text, re.S)
    if not match:
        raise RuntimeError(f"{function} not found in {relative_source}")
    return re.findall(r'list\[count\+\+\]\s*=\s*"([^"]+)"', match.group(1))


def check_member_list(failures: list, family: str, source: str, symbol: str,
                      emitted: dict, optional: list[str] | None = None) -> None:
    want = adapter_members(source, symbol)
    got = list(emitted.keys())
    extra = got[len(want):]
    allowed = [name for name in (optional or []) if name in extra]
    if got[:len(want)] != want or extra != allowed:
        failures.append(
            f"{family}: generator stage-config members {got} != adapter "
            f"exact-member list {want} plus optional {optional or []} "
            f"({source}:{symbol}); the adapter rejects the generator's "
            f"output SCHEMA_ERROR at load")


def firmware_constant(relative_header: str, name: str) -> int:
    text = (ROOT / relative_header).read_text(encoding="utf-8")
    match = re.search(rf"#define {name} (\d+)u", text)
    if not match:
        raise RuntimeError(f"{name} not found in {relative_header}")
    return int(match.group(1))


def check_glm5_next_kv_shard(failures: list, tree: Path) -> None:
    required = firmware_constant(
        "modules/glm5_next_resident_decode_stage/include/sparkpipe/"
        "spark_glm5_next_resident_decode_stage_firmware.h",
        "SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_KV_SHARD_REQUIRED_DEGREE")
    for path in sorted((tree / "config").glob("stage_*.json")):
        stage = json.loads(path.read_text())
        if stage["tp_degree"] < required:
            continue
        if stage.get("kv_shard") != 1 or \
                stage.get("dsa_index_context_parallel") != 1:
            failures.append(
                f"{path}: tp_degree {stage['tp_degree']} >= {required} "
                f"needs kv_shard=1 and dsa_index_context_parallel=1; the "
                f"module refuses GLM-KV-SHARD-REQUIRED at init")


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
                          "SparkGlm5NextServingConfigurationMembers", stage,
                          adapter_optional_members(
                              "modules/glm5_next_resident_decode_stage/"
                              "source/spark_glm5_next_serving_adapter.c",
                              "SparkGlm5NextServingConfigurationList"))
        check_glm5_next_kv_shard(failures, glm5_gen)
        check_glm5_next_kv_shard(failures, glm5_tree)

        for relative in ["config/stage_%02d.json" % i for i in range(16)]:
            committed = json.loads((glm5_tree / relative).read_text())
            leaked = sorted(k for k in committed if k.startswith("score_"))
            if leaked:
                failures.append(f"glm5_next: production {relative} carries score-dump members {leaked}")
        score_source = (ROOT / "modules/glm5_next_resident_decode_stage/source/"
                        "spark_glm5_next_serving_adapter.c").read_text(encoding="utf-8")
        score_match = re.search(r"SparkGlm5NextServingScoreMembers\s*\[\s*\]\s*=\s*\{(.*?)\};", score_source, re.S)
        adapter_score = re.findall(r'"([^"]+)"', score_match.group(1)) if score_match else []
        refused = subprocess.run(
            ["python3", "tools/glm5_next_gen_deployment.py", "--output", str(scratch / "glm5_score_prod"),
             "--score-dump-directory", "score"], cwd=ROOT, capture_output=True, text=True)
        if refused.returncode == 0:
            failures.append("glm5_next: the generator accepted score-dump members for the production root")
        dev = scratch / "glm5_score_dev"
        dev_env = dict(os.environ, GLM5_NEXT_ROOT_NAME="glm53flash.arm-dev.tp16")
        accepted = subprocess.run(
            ["python3", "tools/glm5_next_gen_deployment.py", "--output", str(dev),
             "--score-dump-directory", "score", "--score-probe-path", "score/probe.bin",
             "--score-tier2-rows-path", "score/tier2.bin"], cwd=ROOT, capture_output=True, text=True, env=dev_env)
        if accepted.returncode != 0:
            failures.append(f"glm5_next: the generator refused score-dump members for a dev root: {accepted.stderr[-300:]}")
        else:
            dev_stage = json.loads((dev / "config/stage_00.json").read_text())
            check_glm5_next_kv_shard(failures, dev)
            optional = adapter_optional_members("modules/glm5_next_resident_decode_stage/source/spark_glm5_next_serving_adapter.c",
                                                "SparkGlm5NextServingConfigurationList")
            want = adapter_members("modules/glm5_next_resident_decode_stage/source/spark_glm5_next_serving_adapter.c",
                                   "SparkGlm5NextServingConfigurationMembers") + \
                [name for name in optional if name in dev_stage] + adapter_score
            if list(dev_stage.keys()) != want or len(adapter_score) != 3:
                failures.append(f"glm5_next: dev score-dump stage members {list(dev_stage.keys())} != adapter "
                                f"base + present optional members + score members {want}")
        refusals = {
            "production empty directory": ({}, ["--score-dump-directory", ""]),
            "dev empty directory": (dev_env, ["--score-dump-directory", ""]),
            "dev empty probe": (dev_env, ["--score-dump-directory", "score", "--score-probe-path", ""]),
            "dev probe without directory": (dev_env, ["--score-probe-path", "score/probe.bin"]),
            "dev absolute": (dev_env, ["--score-dump-directory", "/tmp/score"]),
            "dev parent": (dev_env, ["--score-dump-directory", "score/../packs"]),
            "dev dot": (dev_env, ["--score-dump-directory", "./score"]),
            "dev double slash": (dev_env, ["--score-dump-directory", "score//a"]),
            "dev trailing slash": (dev_env, ["--score-dump-directory", "score/"]),
        }
        for label, (env, extra) in refusals.items():
            target = scratch / ("glm5_score_refused_" + label.replace(" ", "_"))
            outcome = subprocess.run(["python3", "tools/glm5_next_gen_deployment.py", "--output", str(target), *extra],
                                     cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, **env))
            if outcome.returncode == 0 or (target / "config").exists():
                failures.append(f"glm5_next: the generator accepted the {label} score-dump configuration")

        glm52_gen = scratch / "glm52"
        run(["python3", "tools/glm53full_lane.py", "--lane", "6", "--codec", "fp8", "--socket", "/tmp/weightd.sock",
             "--kv-backing-bytes", str(1 << 30), "--kv-snapshot-bytes", str(1 << 30), "--max-sequence-positions", "4096", "--execution-row-capacity", "16",
             "--sequences", "8", "--inflight", "1", "--output", str(glm52_gen)])
        stage = json.loads((glm52_gen / "config/stage_08.json").read_text())
        resident = json.loads((glm52_gen / "model_resident.json").read_text())
        limits = resident["runtime_limits"]
        if limits["kv_logical_page_capacity"] <= limits["kv_physical_page_capacity"]:
            failures.append("glm52: the lane renderer leaves no spill pages (logical <= physical)")
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
