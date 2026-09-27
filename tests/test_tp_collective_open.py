#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZERO_SHA = "0" * 64
MESH_RENAMES = (
    "Create",
    "Attach",
    "AttachMesh",
    "PrepareReceiveBf16",
    "Destroy",
)
OBSOLETE_ENVIRONMENT = (
    "TP_BACKEND_PATH",
    "TP_IDENTIFIER",
    "TP_PORT_BASE",
    "TP_HOSTS",
    "TP_LOCAL_HOST",
    "TP_SESSION_PORTS",
)
LAZY_PACK = """
static SparkWeightdLazyPack TpOpenLazyPack;
TpOpenLazyPack.attached.mesh_send_buffer_addr = (uint64_t)(uintptr_t)&TpOpenLazyPack;
state.lazy_pack = &TpOpenLazyPack;
"""
CONTEXT = """
static {context} context;
context.tp_connect_timeout_milli = 30000u;
context.tp_operation_timeout_milli = 30000u;
context.tp_collective_backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
context.tp_collective_topology.rank_count = 16u;
state.tp_degree = 16u;
state.tp_rank = 3u;
"""


def environment_setup(prefix: str, degree: str, rank: str) -> str:
    lines = [f'unsetenv("SPARK_{prefix}_STAGE_{name}");' for name in OBSOLETE_ENVIRONMENT]
    lines.append(f'setenv("SPARK_{prefix}_{degree}","4",1);')
    lines.append(f'setenv("SPARK_{prefix}_{rank}","1",1);')
    lines.append(f'unsetenv("SPARK_{prefix}_TP_STANDALONE");')
    lines.append(f'assert(Spark{{tag}}ModuleConfigureTp(&state) == SPARK_STATUS_OK);')
    lines.append('assert(state.tp_degree == 4u && state.tp_rank == 1u && state.tp_standalone == 0u);')
    return "\n".join(lines)


def glm_cluster(family: str, tag: str, codec: str, lazy: bool, extra: str = "", creates: int = 1, u64: int = 1, common: int = 1) -> dict:
    return {
        "family": family,
        "codec": codec,
        "include": f"modules/{family}_resident_decode_stage/source/spark_{family}_resident_decode_stage_module.c",
        "setup": f"static Spark{tag}ModuleState state;\n"
        + CONTEXT.format(context=f"Spark{tag}ResidentDecodeStageNodeContext")
        + "state.execution_row_capacity = 64u;\n"
        + extra
        + (LAZY_PACK if lazy else ""),
        "call": f"Spark{tag}ModuleInitializeTpCollective(&state,&context)",
        "creates": creates,
        "regions": creates if lazy else 0,
        "u64": u64,
        "common": common,
    }


def qwen_cluster(family: str, tag: str, prefix: str, codec: str, lazy: bool, degree: str, rank: str) -> dict:
    return {
        "family": family,
        "codec": codec,
        "include": f"modules/{family}_resident_decode_stage/source/spark_{family}_resident_decode_stage_module.c",
        "setup": f"static Spark{tag}ModuleState state;\n"
        + environment_setup(prefix, degree, rank).replace("{tag}", tag)
        + "\n"
        + (LAZY_PACK if lazy else ""),
        "call": f"Spark{tag}ModuleInitializeTpCollective(&state)",
        "creates": 1,
        "regions": 1 if lazy else 0,
        "u64": 1,
        "common": 1,
    }


CASES = (
    {
        "family": "dsv4",
        "codec": "fp8",
        "include": "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c",
        "setup": "static SparkDsv4ModuleState state;\n" + CONTEXT.format(context="SparkDsv4ResidentDecodeStageNodeContext"),
        "call": "SparkDsv4ModuleInitializeTpCollective(&state,&context)",
        "creates": 1,
        "regions": 0,
        "u64": 1,
        "common": 1,
    },
    glm_cluster("glm52", "Glm52", "fp8", True),
    glm_cluster("ling", "Ling", "fp8", False),
    glm_cluster("laguna", "Laguna", "fp8", True),
    glm_cluster("glm5_next", "Glm5Next", "fp8", True, "state.pipeline_slot_count = 2u;\nstate.lane_client = (SparkWeightdClient *)&state;\n", 2, 1, 0),
    qwen_cluster("qwen38_max", "Qwen38Max", "QWEN38_MAX", "fp8", True, "STAGE_TP_DEGREE", "STAGE_TP_RANK"),
    qwen_cluster("qwen4_flash", "Qwen4Flash", "QWEN4_FLASH", "fp8", True, "TP_DEGREE", "TP_RANK"),
    qwen_cluster("gemma4", "Gemma4", "GEMMA4", "bf16", False, "TP_DEGREE", "TP_RANK"),
    qwen_cluster("muse_glimmer", "MuseGlimmer", "MUSE_GLIMMER", "bf16", False, "STAGE_TP_DEGREE", "STAGE_TP_RANK"),
    qwen_cluster("minimax", "Minimax", "MINIMAX", "bf16", False, "TP_DEGREE", "TP_RANK"),
    {
        "family": "qwen38_27b",
        "codec": "bf16",
        "include": "modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_tp.c",
        "setup": "static SparkQwen38_27bTpState state;\nunsetenv(\"SPARK_QWEN38_27B_TP_SESSION_PORTS\");\nunsetenv(\"SPARK_QWEN38_27B_TP_STANDALONE\");\n",
        "call": "SparkQwen38_27bTpInitialize(&state,4u,1u,16u,2u,(void *)&state)",
        "creates": 1,
        "regions": 0,
        "u64": 1,
        "common": 1,
    },
)

HARNESS = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "@INCLUDE@"
#include "sparkpipe/spark_tp_mesh_register.h"
static uint32_t TpOpenCreates,TpOpenAttaches,TpOpenRegions;
SparkStatus SparkTpOpenMeshCreate(const SparkTpDeviceCollectiveConfig *config,SparkTpDeviceCollective *collective_out);
SparkStatus SparkTpDeviceCollectiveCreate(const SparkTpDeviceCollectiveConfig *config,SparkTpDeviceCollective *collective_out)
{
	SparkStatus status;
	uint32_t fp32,common;
	status = SparkTpOpenMeshCreate(config,collective_out);
	fp32 = config->combine_fused_bf16_function != 0 && config->combine_f32_seed_function != 0 && config->combine_f32_add_function != 0 && config->round_f32_function != 0 && config->combine_bf16_function != 0;
	common = config->combine_fused_bf16_function == SparkTpMeshCombineFusedBf16 && config->combine_f32_seed_function == SparkTpMeshCombineF32Seed && config->combine_f32_add_function == SparkTpMeshCombineF32Add && config->round_f32_function == SparkTpMeshRoundF32 && config->combine_bf16_function == SparkTpMeshCombineBf16 && config->combine_u64_max_function == SparkTpMeshCombineU64Max;
	printf("TP-OPEN create mesh_status=%d fp32_combines=%u common_combines=%u u64_combine=%u\n",(int)status,fp32,common,config->combine_u64_max_function != 0 ? 1u : 0u);
	if ( status != SPARK_STATUS_UNSUPPORTED || fp32 == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	collective_out->implementation = &TpOpenCreates;
	TpOpenCreates++;
	return(SPARK_STATUS_OK);
}
SparkStatus SparkTpDeviceCollectiveAttach(SparkTpDeviceCollective *collective,void *mesh_region)
{
	if ( collective == 0 || collective->implementation != &TpOpenCreates )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	TpOpenAttaches++;
	TpOpenRegions += mesh_region != 0 ? 1u : 0u;
	return(SPARK_STATUS_OK);
}
SparkStatus SparkTpDeviceCollectiveAttachMesh(SparkTpDeviceCollective *collective)
{
	return(SparkTpDeviceCollectiveAttach(collective,0));
}
SparkStatus SparkTpDeviceCollectivePrepareReceiveBf16(SparkTpDeviceCollective *collective,void *receive_device,uint32_t active_sequence_count,uint32_t hidden_dimension,uint32_t step_index,void *cuda_stream)
{
	(void)active_sequence_count;
	(void)hidden_dimension;
	(void)step_index;
	(void)cuda_stream;
	return(receive_device != 0 ? SparkTpDeviceCollectiveAttach(collective,receive_device) : SPARK_STATUS_INVALID_ARGUMENT);
}
void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
	(void)collective;
}
int main(void)
{
	SparkStatus status;
	unsetenv("SPARK_WEIGHTD_SOCKET");
@SETUP@
	status = @CALL@;
	printf("TP-OPEN status=%d creates=%u attaches=%u regions=%u\n",(int)status,TpOpenCreates,TpOpenAttaches,TpOpenRegions);
	return(0);
}
'''


def module_flags(family: str, codec: str) -> list[str]:
    directory = ROOT / "modules" / f"{family}_resident_decode_stage"
    result = subprocess.run(
        ["make", "-s", "-C", str(directory), "--eval",
         "tp-open-flags: ; @echo $(MODULE_COMPILE_FLAGS) $(MODULE_INCLUDE_FLAGS)", "tp-open-flags",
         f"EXPERT_CODEC={codec}", "MODEL_REVISION=test", f"CONTRACT_SHA256={ZERO_SHA}", "CUDA_HOME=/nonexistent"],
        capture_output=True, text=True, check=True)
    flags = []
    for token in result.stdout.split():
        if token.startswith("-I") and not os.path.isabs(token[2:]):
            token = "-I" + str((directory / token[2:]).resolve())
        flags.append(token)
    return flags


def undefined_symbols(output: str) -> list[str]:
    names = set(re.findall(r"undefined reference to `([A-Za-z_][A-Za-z0-9_]*)'", output))
    names |= set(re.findall(r'"_([A-Za-z_][A-Za-z0-9_]*)", referenced from', output))
    return sorted(names)


def link(command: list[str], stubs: Path) -> subprocess.CompletedProcess:
    built = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    missing = undefined_symbols(built.stderr + built.stdout)
    if built.returncode == 0 or not missing:
        return built
    stubs.write_text("#include <stdlib.h>\n" + "".join(f"void {name}(void) {{ abort(); }}\n" for name in missing))
    return subprocess.run(command + [str(stubs)], cwd=ROOT, capture_output=True, text=True)


def run_case(case: dict, mesh_object: Path, directory: Path) -> str | None:
    source = directory / f"{case['family']}.c"
    binary = directory / case["family"]
    harness = HARNESS.replace("@INCLUDE@", str(ROOT / case["include"]))
    harness = harness.replace("@SETUP@", "\n".join("\t" + line for line in case["setup"].strip().split("\n")))
    harness = harness.replace("@CALL@", case["call"])
    source.write_text(harness)
    command = ["cc", "-std=c11", "-D_GNU_SOURCE", "-O0", "-ffunction-sections", "-fdata-sections",
               "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
               "-I" + str(ROOT), "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src"),
               "-I" + str(ROOT / "runtime"), "-I" + str(ROOT / "tests/cuda_stub"),
               *module_flags(case["family"], case["codec"]),
               str(source), str(mesh_object), str(ROOT / "runtime/stage_module_common.c"),
               str(ROOT / "tests/cuda_stub/cuda_runtime_stub.c"),
               str(ROOT / "build/libsparkpipe_runtime.a"), str(ROOT / "build/libsparkpipe_core.a"),
               "-pthread", "-lm", "-o", str(binary)]
    built = link(command, directory / f"{case['family']}_stubs.c")
    if built.returncode != 0:
        return f"{case['family']}: harness did not build\n{built.stderr[-1500:]}"
    ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    expected = f"TP-OPEN status=0 creates={case['creates']} attaches={case['creates']} regions={case['regions']}"
    u64 = ran.stdout.count("u64_combine=1")
    common = ran.stdout.count("common_combines=1")
    if ran.returncode != 0 or expected not in ran.stdout or u64 != case["u64"] or common != case["common"]:
        return f"{case['family']}: expected '{expected}', {case['u64']} U64 max combine(s) and {case['common']} common combine set(s)\n{ran.stdout[-800:]}{ran.stderr[-800:]}"
    return None


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory(prefix="tp-open-") as temporary:
        directory = Path(temporary)
        mesh_object = directory / "tp_device_collective_mesh.o"
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-O0", "-c", "-ffunction-sections", "-fdata-sections",
                        *[f"-DSparkTpDeviceCollective{name}=SparkTpOpenMesh{name}" for name in MESH_RENAMES],
                        "-I" + str(ROOT), "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src"),
                        "-I" + str(ROOT / "tests/cuda_stub"), "-I" + str(ROOT / "model-families/common/include"),
                        str(ROOT / "ring/transport/tp_device_collective.c"), "-o", str(mesh_object)],
                       cwd=ROOT, check=True)
        for case in CASES:
            failure = run_case(case, mesh_object, directory)
            if failure is not None:
                failures.append(failure)
            else:
                print(f"  ok {case['family']}: {case['creates']} collective(s) pass the mesh's own validation, reduce in FP32 and attach")
    if failures:
        print("\n".join("FAIL " + failure for failure in failures))
        return 1
    print(f"PASS every TP module ({len(CASES)}) opens the weightd mesh with only the settings the mesh reads and the FP32 combines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
