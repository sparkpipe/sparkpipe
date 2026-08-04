"""The naming law, enforced.

A model's name may appear in exactly two places: inference/llms/<model>/ and
model-families/<model>/. The common paths - include/sparkpipe, node, ring,
serving - carry no model name, because generic machinery wearing one model's
name is how the next model copies ten thousand lines instead of writing a
table. docs/DRY_LEDGER.md is the argument; this gate is the enforcement.

The glm52 references still present in common paths are the DECLARED debt of
the two seams the ledger names (kv-cache A3, tp-shard A2) and the glm data
tables those seams read. Each is budgeted EXACTLY, per file, with its reason:
a new reference fails, and a reference removed fails too until the budget is
lowered - the table stays truthful in both directions, which is the whole
point of a ledger. Any other model token in a common path - k3, kimi, qwen,
dsv4, mimo - has no budget at all: those models were born after the law.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON = ("include/sparkpipe", "node", "ring", "serving", "api",
          "cache", "scheduler", "text", "src", "runtime", "deployment",
          "inference/stage")

GLM_BUDGET = {
    "serving/spark_tp_shard.c": (9, "stagepack tensor spec (pack decision)"),
    "include/sparkpipe/spark_tp_shard.h": (11, "stagepack tensor spec (pack decision)"),
    "include/sparkpipe/spark_cuda_resident_ipc.h": (4, "drafter wiring (glm data); kv-cache seam (A3)"),
    "include/sparkpipe/spark_http_gateway.h": (2, "compat surface"),
    "include/sparkpipe/spark_long_context.h": (5, "model header (glm data)"),
    "include/sparkpipe/spark_mtp_tree.h": (3, "model header (glm data)"),
    "include/sparkpipe/spark_resident_decode_stage.h": (24, "weight/plan config-flow pending (A4)"),
    "include/sparkpipe/spark_prefix_cache.h": (3, "kv-cache seam (A3)"),
    "include/sparkpipe/spark_production_topology.h": (7, "drafter wiring (glm data); model header (glm data)"),
    "include/sparkpipe/spark_prompt_pipeline.h": (3, "kv-cache seam (A3)"),
    "include/sparkpipe/spark_request_api.h": (4, "drafter wiring (glm data); kv-cache seam (A3)"),
    "include/sparkpipe/spark_ring_node_context_builder.h": (4, "drafter wiring (glm data)"),
    "include/sparkpipe/spark_ring_runtime.h": (8, "drafter wiring (glm data); model header (glm data); shape-config contract (glm data)"),
    "include/sparkpipe/spark_ring_work_control.h": (10, "drafter wiring (glm data); kv-cache seam (A3); model header (glm data)"),
    "include/sparkpipe/spark_row_allocator.h": (2, "seam includes"),
    "include/sparkpipe/spark_scheduler.h": (4, "seam includes"),
    "include/sparkpipe/spark_service_backend.h": (0, "generic loader ABI"),
    "include/sparkpipe/spark_serving_engine.h": (2, "seam includes"),
    "include/sparkpipe/spark_stage_kv_client.h": (1, "seam includes"),
    "include/sparkpipe/spark_stage_plan.h": (0, "model header (glm data)"),
    "node/backend.c": (42, "drafter wiring (glm data); kv-cache seam (A3); stage firmware (A4)"),
    "node/rank_daemon.c": (14, "drafter wiring (glm data); kv-cache seam (A3); stage firmware (A4)"),
    "node/rank_runtime.c": (26, "seam includes"),
    "node/residentd.c": (23, "kv-cache seam (A3)"),
    "serving/spark_production_topology.c": (7, "seam includes"),
    "serving/spark_ring_node_context_builder.c": (2, "seam includes"),
    "serving/spark_service_backend.c": (0, "generic loader ABI"),
    "api/compat_api.c": (130, "compat surface; template data"),
    "api/gateway/http_server.c": (4, "seam includes"),
    "api/http_gateway.c": (49, "seam includes"),
    "api/request.c": (2, "drafter wiring (glm data); kv-cache seam (A3)"),
    "api/service.c": (0, "seam includes"),
    "api/serving_engine.c": (0, "seam includes"),
    "cache/kv_cache.c": (9, "kv-cache seam (A3)"),
    "cache/prefix_cache.c": (27, "kv-cache seam (A3)"),
    "cache/store/stage_kv_client.c": (1, "seam includes"),
    "scheduler/long_context.c": (0, "seam includes"),
    "scheduler/scheduler.c": (4, "seam includes"),
    "scheduler/stage_plan.c": (0, "seam includes"),
    "scheduler/work_control.c": (9, "seam includes"),
    "text/chat_template.c": (79, "template data"),
    "text/prompt.c": (15, "template data"),
    "text/prompt_pipeline.c": (9, "seam includes"),
    "text/tokenize_main.c": (36, "model header (glm data); template data"),
    "runtime/launch.h": (0, "seam includes"),
    "runtime/pack/artifact_check.c": (70, "stage firmware (A4)"),
    "runtime/pack/fp8_resident_pack.py": (5, "stage firmware (A4)"),
    "runtime/pack/stage_pack.py": (2, "seam includes"),
    "runtime/pack/stagepack.c": (57, "stagepack format (pack decision)"),
    "deployment/src/spark_release.c": (14, "seam includes"),
    "inference/stage/dispatch.cu": (12, "stage firmware (A4)"),
    "inference/stage/module.c": (6, "drafter wiring (glm data); stage firmware (A4)"),
    "inference/stage/runner.c": (20, "stage firmware (A4)"),
    "inference/stage/serving_adapter.cu": (25, "stage firmware (A4)"),
}

FORBIDDEN = re.compile(r"kimi|_k3_|K3[A-Z]|k3_|qwen|dsv4|mimo25|Qwen36|Dsv4|Mimo25",
                       re.IGNORECASE)
GLM = re.compile(r"[Gg]lm52|GLM52")


def main():
    failures = 0
    seen = set()
    for root in COMMON:
        for path in sorted((ROOT / root).rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            rel = str(path.relative_to(ROOT))
            text = path.read_text(errors="surrogateescape")
            if FORBIDDEN.search(text) or FORBIDDEN.search(rel):
                print(f"  FAIL {rel}: a model token with no budget in a "
                      f"common path")
                failures += 1
            hits = len(GLM.findall(text))
            entry = GLM_BUDGET.get(rel)
            if entry is None and hits:
                print(f"  FAIL {rel}: {hits} glm references and no budget "
                      f"entry; the seam debt only shrinks")
                failures += 1
            if entry is not None:
                seen.add(rel)
                if hits != entry[0]:
                    print(f"  FAIL {rel}: {hits} glm references, budget "
                          f"says {entry[0]} ({entry[1]}); update the "
                          f"budget WITH the change, not after")
                    failures += 1
    for rel in set(GLM_BUDGET) - seen:
        print(f"  FAIL budget entry for missing file {rel}")
        failures += 1
    print(f"common files budgeted {len(GLM_BUDGET)}, "
          f"debt {sum(v[0] for v in GLM_BUDGET.values())} references")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nno model name in a common path beyond the declared, "
          "shrinking seam debt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
