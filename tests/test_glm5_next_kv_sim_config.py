#!/usr/bin/env python3
"""The KV sim stage-config members load only into the experiment adapter.

The production adapter must refuse every KV sim member (I22); the experiment
adapter (-DSPARK_KV_QUANT_SIM_EXPERIMENT) must pack each valid configuration
into the node-context field the module reads and refuse invalid ones.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "model_contracts/glm53_flash_authoritative.json"

HARNESS = r"""
#include <stdio.h>
#include <string.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_serving_adapter.c"
int main(int argc, char **argv)
{
    static SparkGlm5NextServingState state;
    uint32_t msp = 0, erc = 0, dsct = 0, tpd = 0, tpr = 0;
    SparkStatus rc;
    if ( argc != 3 )
        return(2);
    memset(&state, 0, sizeof(state));
    rc = SparkGlm5NextServingLoadConfiguration(argv[1], argv[2], &state, &msp, &erc, &dsct, &tpd, &tpr);
#if defined(SPARK_KV_QUANT_SIM_EXPERIMENT)
    printf("rc=%d kv_sim=%u flags_mask=%u\n", (int)rc, state.kv_sim,
        (unsigned)SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KV_SIM_MASK);
#else
    printf("rc=%d\n", (int)rc);
#endif
    return(0);
}
"""

ACCEPTED = [
    ({}, 0),
    ({"kv_latent_codec": "bf16", "kv_index_codec": "bf16", "kv_state_codec": "fp32"}, 0),
    ({"kv_latent_codec": "fp8_e4m3", "kv_codec_group": 128}, 1),
    ({"kv_latent_codec": "fp8_e4m3", "kv_codec_group": 64}, 1 | 4),
    ({"kv_latent_codec": "mxfp4"}, 2),
    ({"kv_index_codec": "fp8_e4m3", "kv_codec_group": 128}, 1 << 3),
    ({"kv_state_codec": "bf16"}, 1 << 6),
    ({"kv_latent_codec": "fp8_e4m3", "kv_index_codec": "fp8_e4m3", "kv_codec_group": 64}, 5 | (5 << 3)),
]
REFUSED = [
    {"kv_codec_group": 128},
    {"kv_latent_codec": "fp8_e4m3"},
    {"kv_latent_codec": "fp8_e4m3", "kv_codec_group": 32},
    {"kv_latent_codec": "mxfp4", "kv_codec_group": 32},
    {"kv_latent_codec": "fp8"},
    {"kv_state_codec": "fp16"},
    {"kv_latent_codec": 1},
    {"kv_quant_sim": "fp8_e4m3"},
]


def fail(message):
    print("FAIL", message)
    raise SystemExit(1)


def build(tmpdir, revision, experiment):
    harness = tmpdir / "harness.c"
    harness.write_text(HARNESS)
    binary = tmpdir / ("harness_experiment" if experiment else "harness_production")
    archives = [ROOT / "build" / name for name in
                ("libsparkpipe_runtime.a", "libsparkpipe_model_common.a", "libsparkpipe_core.a")]
    missing = [str(path.relative_to(ROOT)) for path in archives if not path.is_file()]
    if missing:
        fail("the adapter harness links " + ", ".join(missing) + "; run make all first")
    cmd = ["cc", "-std=c11",
           "-I" + str(ROOT), "-I" + str(ROOT / "include"),
           "-I" + str(ROOT / "src"), "-I" + str(ROOT / "tests/cuda_stub"),
           "-I" + str(ROOT / "model-families/common/include"),
           "-I" + str(ROOT / "model-families/glm5_next/include"),
           "-I" + str(ROOT / "modules/glm5_next_resident_decode_stage/include"),
           "-I" + str(ROOT / "modules/glm5_next_resident_decode_stage/source"),
           "-O0", "-D_GNU_SOURCE", "-DSPARK_BATCH_BUCKET=1024u",
           "-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=5",
           "-DGLM5_NEXT_EXPERT_CODEC_NAME=\"fp8\"",
           "-DGLM5_NEXT_MODEL_REVISION=\"" + revision + "\"",
           "-DGLM5_NEXT_CONTRACT_SHA256=\"" + hashlib.sha256(CONTRACT.read_bytes()).hexdigest() + "\"",
           *(["-DSPARK_KV_QUANT_SIM_EXPERIMENT=1"] if experiment else []),
           str(harness), *[str(path) for path in archives],
           "-o", str(binary), "-ldl", "-lpthread"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        fail("adapter harness did not compile: " + result.stderr[-800:])
    return binary


def load(binary, base, members, tmpdir):
    config = dict(base)
    config.update(members)
    path = tmpdir / "stage.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    run = subprocess.run([str(binary), str(path), str(ROOT)], capture_output=True, text=True, timeout=60)
    if run.returncode != 0:
        fail(f"harness exited {run.returncode}: {run.stderr[-300:]}")
    return dict(item.split("=") for item in run.stdout.split())


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        subprocess.run([sys.executable, str(ROOT / "tools/glm5_next_gen_deployment.py"),
                        "--output", str(tmpdir / "deploy")], check=True, capture_output=True)
        base = json.loads((tmpdir / "deploy/config/stage_00.json").read_text())
        production = build(tmpdir, base["model_revision"], False)
        experiment = build(tmpdir, base["model_revision"], True)
        if load(production, base, {}, tmpdir)["rc"] != "0":
            fail("the production adapter refuses the generated stage config")
        for members, _ in ACCEPTED[1:]:
            if load(production, base, members, tmpdir)["rc"] == "0":
                fail(f"the production adapter accepted KV sim members {members}")
        for members, packed in ACCEPTED:
            got = load(experiment, base, members, tmpdir)
            if got["rc"] != "0" or int(got["kv_sim"]) != packed:
                fail(f"experiment adapter on {members}: {got}, want rc=0 kv_sim={packed}")
            if (packed << 8) & ~int(got["flags_mask"]):
                fail(f"packed value {packed} does not fit the node-context KV sim field")
        for members in REFUSED:
            got = load(experiment, base, members, tmpdir)
            if got["rc"] == "0":
                fail(f"experiment adapter accepted invalid KV sim members {members}")
    print(f"PASS the production adapter refuses every KV sim member; the experiment adapter packs "
          f"{len(ACCEPTED)} configurations into the node-context field and refuses {len(REFUSED)} invalid ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
