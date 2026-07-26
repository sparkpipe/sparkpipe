#!/usr/bin/env python3

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON_RULES_PATH = ROOT / "modules/resident_decode_stage_rules.mk"

FAMILIES = {
    "dsv4": {
        "model_header": "sparkpipe/spark_dsv4_model.h",
        "source": "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c",
        "module_include": "modules/dsv4_resident_decode_stage/include",
        "family_include": "model-families/dsv4/include",
        "description": "examples/model_descriptions/dsv4_resident_decode_stage_firmware.json",
        "makefile": "modules/dsv4_resident_decode_stage/Makefile",
        "prefix": "SparkDsv4ResidentDecodeStage",
        "unqualified_environment": "SPARK_DSV4_ALLOW_UNQUALIFIED_EXECUTION",
        "required_fragments": (),
    },
    "k3": {
        "model_header": "sparkpipe/spark_k3_model.h",
        "source": "modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c",
        "module_include": "modules/k3_resident_decode_stage/include",
        "family_include": "model-families/k3/include",
        "description": "examples/model_descriptions/k3_resident_decode_stage_firmware.json",
        "makefile": "modules/k3_resident_decode_stage/Makefile",
        "prefix": "SparkK3ResidentDecodeStage",
        "unqualified_environment": "K3_ALLOW_UNQUALIFIED_EXECUTION",
        "required_fragments": (
            "SparkK3ModuleValidateBlockCoverage",
            "logical_block < block_count",
            "row_capacity",
        ),
    },
    "mimo25": {
        "model_header": "sparkpipe/spark_mimo25_model.h",
        "source": "modules/mimo25_resident_decode_stage/source/spark_mimo25_resident_decode_stage_module.c",
        "module_include": "modules/mimo25_resident_decode_stage/include",
        "family_include": "model-families/mimo25/include",
        "description": "examples/model_descriptions/mimo25_resident_decode_stage_firmware.json",
        "makefile": "modules/mimo25_resident_decode_stage/Makefile",
        "prefix": "SparkMimo25ResidentDecodeStage",
        "unqualified_environment": "SPARK_MIMO25_ALLOW_UNQUALIFIED_EXECUTION",
        "required_fragments": (
            "context->mtp_draft_depth != 0u || context->mtp_draft_tokens != 0",
            "state->mtp_armed != 0u && state->owns_final_head != 0u",
        ),
    },
    "qwen36": {
        "model_header": "sparkpipe/spark_qwen36_model.h",
        "source": "modules/qwen36_resident_decode_stage/source/spark_qwen36_resident_decode_stage_module.c",
        "module_include": "modules/qwen36_resident_decode_stage/include",
        "family_include": "model-families/qwen36/include",
        "description": "examples/model_descriptions/qwen36_resident_decode_stage_firmware.json",
        "makefile": "modules/qwen36_resident_decode_stage/Makefile",
        "prefix": "SparkQwen36ResidentDecodeStage",
        "unqualified_environment": "SPARK_QWEN36_ALLOW_UNQUALIFIED_EXECUTION",
        "required_fragments": (),
    },
}

REQUIRED_SOURCE_FRAGMENTS = (
    "SparkFirmwareModuleValidateInitialization",
    "SparkStageModuleIndexSetClaim",
    "SparkStageModuleIndexSetRelease",
    "SparkStageModuleWaitForSlots",
    "SPARK_STATUS_MODULE_NOT_VALIDATED",
    "descriptor_bytes",
    "lanes_claimed",
)

ALLOWED_SCHEDULING_FLAGS = {
    "stream_ordered",
    "driver_owns_resident_state",
    "driver_owns_kv_cache",
    "jit_kv_cache",
    "zero_copy_node_context",
    "private_queue_pressure",
    "no_host_staging",
    "fixed_firmware",
    "validated_latency",
    "captured_cuda_graph",
    "no_device_memcpy",
    "driver_private_expert_queues",
    "stream_event_dependencies",
    "residency_affinity_required",
    "batch_shape_fixed",
    "requires_hidden_transport",
    "no_file_transport",
    "no_shell_transport",
    "bulk_prefill",
}


def run(command):
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    if completed.returncode != 0:
        print(f"FAIL command: {' '.join(command)}", file=sys.stderr)
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        return False
    return True


def strict_host_syntax_check(family, contract):
    compiler = os.environ.get("CC", "cc")
    command = [
        compiler,
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-D_POSIX_C_SOURCE=200809L",
        "-D_FILE_OFFSET_BITS=64",
        "-fsyntax-only",
        "-Itests/cuda_stub",
        "-Iinclude",
        "-Imodel-families/common/include",
        f"-I{contract['family_include']}",
        f"-I{contract['module_include']}",
        "-include",
        contract["model_header"],
        contract["source"],
    ]
    if not run(command):
        return [f"{family}: strict host syntax compilation failed"]
    return []


def validate_source_contract(family, contract, required_environment):
    source = (ROOT / contract["source"]).read_text()
    failures = []
    for fragment in REQUIRED_SOURCE_FRAGMENTS + contract["required_fragments"]:
        if fragment not in source:
            failures.append(f"{family}: source lacks {fragment}")
    for environment_name in required_environment:
        if environment_name not in source:
            failures.append(
                f"{family}: source does not consume declared environment {environment_name}")
    if "SparkStageModuleEnvironmentOptionalBoolean" in source:
        failures.append(
            f"{family}: source silently defaults an optional runtime configuration")
    if contract["unqualified_environment"] not in source:
        failures.append(
            f"{family}: source lacks controlled bring-up guard "
            f"{contract['unqualified_environment']}")
    for suffix in ("Initialize", "Execute", "Admit", "Snapshot", "Destroy"):
        symbol = contract["prefix"] + suffix
        if symbol not in source:
            failures.append(f"{family}: missing exported entry point {symbol}")
    rejects_unknown_frame_flags = (
        "frame->flags & ~known_frame_flags" in source)
    rejects_unknown_admission_flags = (
        "request->frame_flags & ~known_frame_flags" in source or
        ("request->frame_flags &" in source and
         "~SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL" in source))
    if not rejects_unknown_frame_flags or not rejects_unknown_admission_flags:
        failures.append(
            f"{family}: frame and admission flags are not visibly fail-closed")
    if "frame->driver_dispatch_slot" in source:
        failures.append(
            f"{family}: execution reads the generic dispatch ticket as model state")
    return failures


def validate_description_contract(family, contract):
    description = json.loads((ROOT / contract["description"]).read_text())
    failures = []
    if description.get("schema_version") != 1:
        failures.append(f"{family}: description schema_version is not 1")
    metadata = description.get("metadata", {})
    qualification = metadata.get("qualification", {})
    if qualification.get("status") != "NOT_MEASURED":
        failures.append(f"{family}: description is not explicitly NOT_MEASURED")
    if qualification.get("production_ready") is not False:
        failures.append(f"{family}: description does not reject production readiness")

    stages = description.get("stages")
    if not isinstance(stages, list) or len(stages) != 1:
        failures.append(f"{family}: description must contain exactly one stage example")
        return failures, description, set()
    programs = stages[0].get("programs")
    if not isinstance(programs, list) or len(programs) != 1:
        failures.append(f"{family}: description must contain exactly one program example")
        return failures, description, set()
    program = programs[0]
    if program.get("completion") != "submit_return":
        failures.append(f"{family}: synchronous module must use submit_return completion")
    operations = program.get("operations")
    if not isinstance(operations, list) or len(operations) != 1 or not isinstance(operations[0], dict):
        failures.append(f"{family}: operations must use the current object schema")
    elif not operations[0].get("module"):
        failures.append(f"{family}: operation lacks a module identity")
    flags = program.get("scheduling", {}).get("flags", [])
    unknown_flags = sorted(set(flags) - ALLOWED_SCHEDULING_FLAGS)
    if unknown_flags:
        failures.append(f"{family}: unknown scheduling flags: {', '.join(unknown_flags)}")
    serialized = json.dumps(description).lower()
    for phrase in (
        '"driver_dispatch_slot": "required and valid, it is the lane"',
        '"production_ready": true',
        '"fallback_allowed": true',
    ):
        if phrase in serialized:
            failures.append(f"{family}: stale or unsafe descriptor phrase: {phrase}")
    runtime_contract = metadata.get("runtime_contract", {})
    if runtime_contract.get("model_driver_abi") != 6:
        failures.append(f"{family}: description does not bind model-driver ABI 6")
    if runtime_contract.get("firmware_module_abi") != 4:
        failures.append(f"{family}: description does not bind firmware-module ABI 4")
    if runtime_contract.get("fallback_allowed") is not False:
        failures.append(f"{family}: runtime contract does not fail closed")
    if runtime_contract.get("runtime_backend_selection") != "forbidden":
        failures.append(f"{family}: runtime backend selection is not forbidden")
    required_environment = runtime_contract.get("required_environment", [])
    if not isinstance(required_environment, list) or not required_environment:
        failures.append(f"{family}: runtime contract lacks required_environment")
        required_environment_set = set()
    else:
        required_environment_set = set(required_environment)
        if len(required_environment_set) != len(required_environment):
            failures.append(f"{family}: required_environment contains duplicates")
    return failures, description, required_environment_set


def parse_make_assignment(makefile_text, variable_name):
    match = re.search(
        rf"^{re.escape(variable_name)}\s*:?=\s*(\S+)\s*$",
        makefile_text,
        re.MULTILINE)
    return match.group(1) if match else None


def validate_module_makefile_contract(family, contract, description, required_environment):
    makefile_text = (ROOT / contract["makefile"]).read_text()
    failures = []
    runtime_environment = set(
        re.findall(r"\b([A-Z][A-Z0-9_]+)=\$\(", makefile_text))
    if runtime_environment != required_environment:
        missing = sorted(required_environment - runtime_environment)
        extra = sorted(runtime_environment - required_environment)
        if missing:
            failures.append(
                f"{family}: Makefile omits declared environment: {', '.join(missing)}")
        if extra:
            failures.append(
                f"{family}: Makefile publishes undeclared environment: {', '.join(extra)}")
    if "include ../resident_decode_stage_rules.mk" not in makefile_text:
        failures.append(f"{family}: Makefile bypasses common resident-stage rules")
    if "ALLOW_UNQUALIFIED_EXECUTION ?= 0" not in makefile_text:
        failures.append(f"{family}: unqualified execution is not disabled by default")

    stage = description["stages"][0]
    operation = stage["programs"][0]["operations"][0]
    if parse_make_assignment(makefile_text, "MODULE_IDENTIFIER") != operation["module"]:
        failures.append(f"{family}: Makefile module identity differs from description")
    if parse_make_assignment(makefile_text, "MODULE_TARGET") != stage["target"]:
        failures.append(f"{family}: Makefile target differs from description")

    validation_recipe_lines = [
        line for line in makefile_text.splitlines()
        if line.lstrip().startswith("VALIDATION_RECIPE")]
    if any("reference" in line.lower() for line in validation_recipe_lines):
        failures.append(f"{family}: CPU/reference path appears in validation recipe")
    return failures


def validate_common_publication_rules():
    text = COMMON_RULES_PATH.read_text()
    failures = []
    required_fragments = (
        "require_gpu_validator",
        "test -x \"$(GPU_VALIDATOR)\"",
        "--validator $(GPU_VALIDATOR)",
        "VALIDATION_CONFIGURATION_SHA256",
        "--validator-arg $(VALIDATION_CONFIGURATION_SHA256)",
        "require_stage_pack",
        "CUDA_ARCH must be sm_121a",
        "-D_POSIX_C_SOURCE=200809L",
        "-D_FILE_OFFSET_BITS=64",
    )
    for fragment in required_fragments:
        if fragment not in text:
            failures.append(f"common module rules lack {fragment}")
    return failures


def validate_schema_flags():
    schema = json.loads((ROOT / "schema/model_description.schema.json").read_text())
    enum = set(
        schema["properties"]["stages"]["items"]["properties"]["programs"]
        ["items"]["properties"]["scheduling"]["properties"]["flags"]
        ["items"]["enum"])
    missing = sorted(ALLOWED_SCHEDULING_FLAGS - enum)
    return [f"schema omits parser scheduling flags: {', '.join(missing)}"] if missing else []


def main():
    failures = []
    failures.extend(validate_common_publication_rules())
    for family, contract in FAMILIES.items():
        description_failures, description, required_environment = \
            validate_description_contract(family, contract)
        failures.extend(description_failures)
        failures.extend(strict_host_syntax_check(family, contract))
        failures.extend(
            validate_source_contract(family, contract, required_environment))
        if description_failures:
            continue
        failures.extend(
            validate_module_makefile_contract(
                family,
                contract,
                description,
                required_environment))
    failures.extend(validate_schema_flags())

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(
        "PASS model-driver contracts: DSV4, K3, MiMo 2.5, and Qwen 3.6 "
        "compile under strict host syntax, match their package descriptors, "
        "and publish only through fail-closed ABI-4/ABI-6 rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
