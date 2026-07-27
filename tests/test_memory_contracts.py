#!/usr/bin/env python3
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from glm52_model_contract import load_model_contract, render_c_header

SOURCE_SUFFIXES = {".c", ".cu", ".h", ".py"}
SKIP_PARTS = {".git", "__pycache__", "build", "third_party"}
CALL_NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
MEMORY_CALL = re.compile(
    r"alloc|malloc|calloc|realloc|memcpy|memset|copy|zero", re.I)
RAW_BYTE_MULTIPLIER = re.compile(r"\*\s*(2|4|8)(?:u|ul|ull)?\b")
RAW_TYPED_FIELD_ARGUMENT = re.compile(r"(?:1|2|4|8)(?:u|ul|ull)?")
RAW_DTYPE_BYTES = re.compile(
    r'"(?:BF16|F32|F8_E4M3)"\s*,\s*(?:1|2|4)(?:u|ul|ull)?\b', re.S)
RAW_HEADER_OFFSET = re.compile(r"\bheader_bytes\s*\+\s*(?:16|80|88)u\b")
RAW_REGION_INDEX = re.compile(r"\bheader\.regions\[(?:0|1|2|3)\]")
RAW_DIMENSION_DEFINE = re.compile(
    r"#define\s+SPARK_GLM52_RESIDENT_DECODE_STAGE_"
    r"(?:QUERY_B_DIMENSION|KV_A_DIMENSION|KV_B_DIMENSION|QK_HEAD_DIMENSION|"
    r"CACHE_TOKEN_ELEMENTS|QUERY_ROPE_PROJECTION_DIMENSION)\s+[0-9]+u\b")
RAW_MODEL_SCALAR_ASSIGNMENT = re.compile(
    r"\b(?:qk_scale|rms_norm_epsilon|index_softmax_scale|"
    r"moe_routed_scaling_factor)\s*=\s*"
    r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?f?\b",
    re.I)
RAW_SCALAR_WIDTH_ASSIGNMENT = re.compile(
    r"\b(?:bytes_per_scalar|bytes_per_element|element_bytes)\s*=\s*"
    r"(?:1|2|4|8)(?:u|ul|ull)?\b")
RAW_PROTOCOL_BUFFER = re.compile(
    r"\b(?:read|write|packet|header|event|payload|message|wire|frame)_buffer"
    r"\s*\[\s*[0-9]+u?\s*\]")
RAW_TYPED_BYTE_ARRAY = re.compile(
    r"\buint8_t\s+(?:bytes|encoded|header_length_bytes)"
    r"\s*\[\s*[0-9]+u?\s*\]")
RAW_SIDEBAND_BYTE_ASSIGNMENT = re.compile(
    r"\bsideband_bytes_per_sequence\s*=\s*[1-9][0-9]*u?\b")
RAW_PACKET_BYTE_SLACK = re.compile(
    r"\bmax_packet_bytes\s*=\s*[1-9][0-9]*u?\s*\+")
SIZEOF_CONTRACT_SUFFIXES = (
    "_COMPLETION_BYTES",
    "_DESCRIPTOR_BYTES",
    "_DYNAMIC_LIBRARY_BYTES",
    "_INTERFACE_BYTES",
    "_PACKET_BYTES",
    "_STATISTICS_BYTES",
)
MODEL_LITERAL_ALLOWLIST = {
    "6144": {"model-families/glm52/include/sparkpipe/spark_glm52_model.h"},
    "154880": {"model-families/glm52/include/sparkpipe/spark_glm52_model.h"},
    "28672": set(),
    "576": set(),
    "12288": {"model-families/glm52/include/sparkpipe/spark_glm52_model.h"},
}
TYPED_FIELD_CALLS = {"ALLOC_FIELD", "ALLOC_FIELD_MAPPED", "ZERO_FIELD"}
NON_GLM_MODEL_PREFIXES = (
    "model-families/dsv4/",
    "model-families/k3/",
    "model-families/mimo25/",
    "model-families/qwen36/",
    "modules/dsv4_",
    "modules/k3_",
    "modules/mimo25_",
    "modules/qwen36_",
)
REQUIRED_SYMBOLIC_DEFINES = {
    "model-families/glm52/src/spark_glm52_ring_service_backend.c": {
        "SPARK_GLM52_RING_SERVICE_BACKEND_PIPELINE_COHORT_CAPACITY":
            "SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT",
        "SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_CAPACITY":
            "SPARK_GLM52_RING_SERVICE_BACKEND_PIPELINE_COHORT_CAPACITY",
        "SPARK_GLM52_RING_SERVICE_BACKEND_DEFAULT_LOGICAL_BLOCK_COUNT":
            "SPARK_GLM52_RING_SERVICE_BACKEND_GPU_BLOCK_COUNT",
    },
    "model-families/glm52/include/sparkpipe/spark_glm52_ring_runtime.h": {
        "SPARK_GLM52_RING_RUNTIME_STAGE_COUNT":
            "SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT",
        "SPARK_GLM52_RING_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE":
            "SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES",
        "SPARK_GLM52_RING_RUNTIME_INDEXSHARE_SIDEBAND_BYTES_PER_SEQUENCE":
            "SPARK_GLM52_MODEL_DSA_SELECTED_INDEX_BYTES",
    },
    "model-families/glm52/include/sparkpipe/spark_glm52_production_topology.h": {
        "SPARK_GLM52_PRODUCTION_TOPOLOGY_FIRST_FULL_INDEXER_LAYER_COUNT":
            "SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER",
        "SPARK_GLM52_PRODUCTION_TOPOLOGY_INDEXSHARE_GROUP_LAYER_COUNT":
            "SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT",
        "SPARK_GLM52_PRODUCTION_TOPOLOGY_INDEX_SKIP_TOPK_OFFSET":
            "SPARK_GLM52_MODEL_DSA_INDEX_SKIP_TOPK_OFFSET",
    },
    "model-families/glm52/include/sparkpipe/spark_glm52_request_api.h": {
        "SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT":
            "SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT",
    },
    "modules/glm52_resident_decode_stage/include/sparkpipe/"
    "spark_glm52_resident_decode_stage_firmware.h": {
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_BLOCK_SIZE":
            "SPARK_GLM52_MODEL_FP8_SCALE_BLOCK",
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ROUTED_STAGE_SLICE_LAYER_COUNT":
            "SPARK_GLM52_STAGE_PLAN_MAX_ROUTED_LAYERS_PER_STAGE",
    },
}
SIZE_ARGUMENTS_BY_CALL = {
    "cudaHostAlloc": (1,),
    "cudaMalloc": (1,),
    "cudaMallocManaged": (1,),
    "cudaMemcpy": (2,),
    "cudaMemcpyAsync": (2,),
    "cudaMemset": (2,),
    "cudaMemsetAsync": (2,),
    "fread": (1, 2),
    "fwrite": (1, 2),
    "malloc": (0,),
    "memcmp": (2,),
    "memcpy": (2,),
    "memmove": (2,),
    "memset": (2,),
    "mmap": (1,),
    "munmap": (1,),
    "read": (2,),
    "realloc": (1,),
    "recv": (2,),
    "send": (2,),
    "SparkSha256Update": (2,),
    "strncmp": (2,),
    "write": (2,),
}
RAW_DIRECT_BYTE_COUNT = re.compile(
    r"(?:\([^)]+\))?\s*(?:[2-9]|[1-9][0-9]+)(?:u|ul|ull)?")


def glm_model_literal_scope(relative_path):
    return not relative_path.startswith(NON_GLM_MODEL_PREFIXES)


def source_paths():
    for path in sorted(ROOT.rglob("*")):
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        yield path


def calls(text):
    for match in CALL_NAME.finditer(text):
        depth = 1
        index = match.end()
        quote = None
        escaped = False
        while index < len(text) and depth != 0:
            value = text[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif value == "\\":
                    escaped = True
                elif value == quote:
                    quote = None
            elif value in {"\"", "'"}:
                quote = value
            elif value == "(":
                depth += 1
            elif value == ")":
                depth -= 1
            index += 1
        if depth == 0:
            yield match.start(), match.group(1), text[match.end():index - 1]


def split_arguments(arguments):
    result = []
    begin = 0
    depth = 0
    for index, value in enumerate(arguments):
        if value == "(":
            depth += 1
        elif value == ")":
            depth -= 1
        elif value == "," and depth == 0:
            result.append(arguments[begin:index].strip())
            begin = index + 1
    result.append(arguments[begin:].strip())
    return result


def report(violations, path, text, offset, message):
    line = text.count("\n", 0, offset) + 1
    violations.append(f"{path.relative_to(ROOT)}:{line}: {message}")


def logical_preprocessor_lines(text):
    lines = text.splitlines(keepends=True)
    offset = 0
    index = 0
    while index < len(lines):
        begin = offset
        value = lines[index]
        offset += len(lines[index])
        index += 1
        while value.rstrip().endswith("\\") and index < len(lines):
            value += lines[index]
            offset += len(lines[index])
            index += 1
        yield begin, value


def main():
    violations = []
    for path in source_paths():
        relative = str(path.relative_to(ROOT))
        text = path.read_text(errors="replace")
        for offset, name, arguments in calls(text):
            if MEMORY_CALL.search(name) and RAW_BYTE_MULTIPLIER.search(arguments):
                report(violations, path, text, offset,
                    f"{name} uses a raw byte-width multiplier")
            if name in TYPED_FIELD_CALLS:
                call_arguments = split_arguments(arguments)
                if call_arguments and RAW_TYPED_FIELD_ARGUMENT.fullmatch(
                        call_arguments[-1]):
                    report(violations, path, text, offset,
                        f"{name} uses an untyped element width")
            size_argument_indices = SIZE_ARGUMENTS_BY_CALL.get(name)
            if size_argument_indices is not None:
                call_arguments = split_arguments(arguments)
                for size_argument_index in size_argument_indices:
                    if size_argument_index < len(call_arguments) and \
                            RAW_DIRECT_BYTE_COUNT.fullmatch(
                                call_arguments[size_argument_index]):
                        report(violations, path, text, offset,
                            f"{name} uses a direct size/count literal")
        for match in RAW_DTYPE_BYTES.finditer(text):
            report(violations, path, text, match.start(),
                "tensor dtype is paired with a raw byte width")
        for match in RAW_HEADER_OFFSET.finditer(text):
            report(violations, path, text, match.start(),
                "wire header uses a numeric field offset")
        for match in RAW_REGION_INDEX.finditer(text):
            report(violations, path, text, match.start(),
                "FP8 pack region uses a numeric field index")
        for match in RAW_DIMENSION_DEFINE.finditer(text):
            report(violations, path, text, match.start(),
                "derived GLM dimension is defined as a literal")
        for match in RAW_MODEL_SCALAR_ASSIGNMENT.finditer(text):
            report(violations, path, text, match.start(),
                "GLM model scalar is assigned from a raw literal")
        for match in RAW_SCALAR_WIDTH_ASSIGNMENT.finditer(text):
            report(violations, path, text, match.start(),
                "scalar element width is assigned as a byte literal")
        for match in RAW_PROTOCOL_BUFFER.finditer(text):
            report(violations, path, text, match.start(),
                "protocol buffer capacity is a raw numeric extent")
        for match in RAW_TYPED_BYTE_ARRAY.finditer(text):
            report(violations, path, text, match.start(),
                "typed byte array extent is a raw numeric width")
        for match in RAW_SIDEBAND_BYTE_ASSIGNMENT.finditer(text):
            report(violations, path, text, match.start(),
                "sideband byte width is a raw numeric value")
        for match in RAW_PACKET_BYTE_SLACK.finditer(text):
            report(violations, path, text, match.start(),
                "transport packet capacity contains numeric slack")
        for offset, logical_line in logical_preprocessor_lines(text):
            match = re.match(r"\s*#define\s+([A-Z0-9_]+)\s+", logical_line)
            if match is None or not match.group(1).endswith(
                    SIZEOF_CONTRACT_SUFFIXES):
                continue
            if "sizeof(" not in logical_line:
                report(violations, path, text, offset,
                    f"{match.group(1)} is not derived from its wire type")
        for literal, allowed_paths in MODEL_LITERAL_ALLOWLIST.items():
            if not glm_model_literal_scope(relative) or relative in allowed_paths:
                continue
            if path.suffix == ".py" and relative.startswith("tests/"):
                continue
            if path.suffix == ".py":
                pattern = re.compile(rf"(?<![A-Za-z0-9_]){literal}(?![A-Za-z0-9_])")
            else:
                pattern = re.compile(rf"\b{literal}(?:u|ul|ull)\b")
            for match in pattern.finditer(text):
                report(violations, path, text, match.start(),
                    f"GLM model literal {literal} is outside the model contract")
    module_header = ROOT / (
        "modules/glm52_sm121_flashinfer_b12x_moe/include/sparkpipe/"
        "spark_glm52_sm121_flashinfer_b12x_moe.h")
    if module_header.exists():
        violations.append("B12x public ABI header has a duplicate module copy")
    final_event_definitions = []
    final_event_pattern = re.compile(
        r"typedef\s+struct\s+SparkGlm52Ring[A-Za-z0-9_]*FinalEvent\s*\{")
    for path in source_paths():
        text = path.read_text(errors="replace")
        for match in final_event_pattern.finditer(text):
            final_event_definitions.append(str(path.relative_to(ROOT)))
    if final_event_definitions != [
            "model-families/glm52/include/sparkpipe/spark_glm52_ring_runtime.h"]:
        violations.append(
            "RING final-event wire type is not single-source: " +
            ", ".join(final_event_definitions))
    model_contract = load_model_contract(ROOT)
    model_header = ROOT / "model-families/glm52/include/sparkpipe/spark_glm52_model.h"
    if model_header.read_text() != render_c_header(model_contract):
        violations.append("generated GLM-5.2 C model contract is stale")
    fp8_header = ROOT / (
        "modules/glm52_resident_decode_stage/include/sparkpipe/"
        "spark_glm52_resident_decode_stage_fp8_moe_plan.h")
    if "fields[" in fp8_header.read_text():
        violations.append("FP8 MoE wire header uses anonymous indexed fields")
    all_source_text = "\n".join(
        path.read_text(errors="replace") for path in source_paths()
        if path.suffix in {".c", ".cu", ".h"})
    if "SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_PREFILL_TILE_ROWS" in all_source_text:
        violations.append("DSA score tile rows have a duplicate prefill constant")
    if "dsa_prefill_scores_f32" in all_source_text:
        violations.append("DSA score storage retains a prefill-only ABI field")
    backend_text = (
        ROOT / "model-families/glm52/src/spark_glm52_ring_service_backend.c").read_text()
    if re.search(
            r"message->control_generation\s*=\s*state->session_id_base\s*;",
            backend_text) is None:
        violations.append("attached decode IPC omits the gateway generation")
    if re.search(
            r"if \(state->mtp_enabled != 0u\)\s*"
            r"request_api_configuration\.configuration_flags \|=\s*"
            r"SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT \|\s*"
            r"SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;",
            backend_text) is None:
        violations.append("MTP backend does not force MTP request scheduling")
    residentd_text = (
        ROOT / "node/residentd.c").read_text()
    if re.search(
            r"packet->control_generation\s*=\s*"
            r"message->control_generation\s*;",
            residentd_text) is None:
        violations.append("resident decode drops the attached control generation")
    if re.search(
            r"kv_logical_block_capacity\s*<\s*"
            r"SPARK_GLM52_RING_SERVICE_BACKEND_GPU_BLOCK_COUNT",
            backend_text):
        violations.append(
            "attached KV geometry is coupled to the local compile-time pool")
    if re.search(
            r"cuda_resident_socket_path\s*!=\s*0\s*&&\s*"
            r"configuration->kv_logical_block_capacity\s*==\s*0u",
            backend_text) is None:
        violations.append("attached KV geometry can silently default")
    service_pump = backend_text.find(
        "service_status = SparkGlm52ServicePump(")
    release_pump = backend_text.find(
        "release_status = SparkGlm52RingServiceBackendPumpSequenceReleases(",
        service_pump)
    work_flush = backend_text.find(
        "work_status = SparkGlm52RingServiceBackendPumpWorkOutput(",
        release_pump)
    if (service_pump < 0 or release_pump < 0 or work_flush < 0 or
            "SparkGlm52RingServiceBackendDrainSequenceReleases" in backend_text):
        violations.append(
            "service pump does not progress sequence releases without a global drain")
    model_description_path = ROOT / (
        "examples/model_descriptions/glm52_resident_decode_stage_firmware.json")
    model_description = json.loads(model_description_path.read_text())
    geometry = model_description["metadata"]["hf_config_geometry"]
    expected_geometry = {
        "hidden_size": model_contract["hidden_dimension"],
        "num_attention_heads": model_contract["head_count"],
        "kv_lora_rank": model_contract["latent_dimension"],
        "qk_rope_head_dim": model_contract["rope_dimension"],
        "index_topk": model_contract["dsa_selected_token_count"],
        "num_hidden_layers": model_contract["layer_count"],
        "vocab_size": model_contract["output_vocab_count"],
        "n_routed_experts": model_contract["moe_expert_count"],
        "num_experts_per_tok": model_contract["moe_top_k"],
        "qk_nope_head_dim": model_contract["qk_nope_head_dimension"],
        "qk_head_dim": (
            model_contract["qk_nope_head_dimension"] +
            model_contract["rope_dimension"]),
        "v_head_dim": model_contract["value_head_dimension"],
        "rope_theta": model_contract["rope_theta"],
        "rope_interleave": model_contract["rope_interleave"],
        "indexer_rope_interleave": model_contract["dsa_rope_interleave"],
        "rms_norm_eps": model_contract["rms_norm_epsilon"],
        "routed_scaling_factor": model_contract["moe_routed_scaling_factor"],
    }
    for key, expected in expected_geometry.items():
        if geometry.get(key) != expected:
            violations.append(
                f"GLM model description {key} disagrees with model contract")
    for relative, definitions in REQUIRED_SYMBOLIC_DEFINES.items():
        text = (ROOT / relative).read_text()
        for name, source_name in definitions.items():
            pattern = re.compile(
                rf"#define\s+{re.escape(name)}\s+(?:\\\s*)?"
                rf"{re.escape(source_name)}\b",
                re.S)
            if pattern.search(text) is None:
                violations.append(
                    f"{relative}: {name} does not alias {source_name}")
    if violations:
        raise SystemExit("\n".join(violations))


if __name__ == "__main__":
    main()
