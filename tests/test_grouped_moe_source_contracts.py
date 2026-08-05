#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing {label}: {needle}")


def reject(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise SystemExit(f"forbidden {label}: {needle}")


def main() -> None:
    route = read("inference/kernels/route.cuh")
    require(route, "packed_rows != expected_packed_rows", "route cardinality validation")
    require(route, "LmLaunchGroupedTileM(rows,top_k,EXPERTS)", "token-priced grouped tile")

    contracts = {
        "k3": (
            "inference/llms/kimi_k3/layer.cuh",
            "LM_TOPK_SCORE_SIGMOID",
            "K3_ROUTED_SCALE",
        ),
        "glm52": (
            "modules/glm52_resident_decode_stage/source/cuda/layer.cuh",
            "LM_TOPK_SCORE_SIGMOID",
            "GLM52_ROUTED_SCALE",
        ),
        "mimo25": (
            "inference/llms/mimo_2_5/layer.cuh",
            "LM_TOPK_SCORE_IDENTITY",
            "1.0f",
        ),
    }
    for family, (path, transform, scale) in contracts.items():
        text = read(path)
        require(text, "LmRouteBuild<", f"{family} device route build")
        require(text, "group_tile_prefix_w1", f"{family} W1 prefix")
        require(text, "group_tile_prefix_w2", f"{family} W2 prefix")
        require(text, "prefix_built = 1u", f"{family} prebuilt prefix")
        require(text, transform, f"{family} router transform")
        require(text, scale, f"{family} router scale")
        reject(text, "LmLaunchGroupedTileM(packed_rows", f"{family} route-priced tile")

    k3 = read("inference/llms/kimi_k3/layer.cuh")
    require(k3, "b->expert_w1_weight,packed_rows,rows,", "K3 W1 token count")
    require(k3, "b->expert_w2_weight,packed_rows,rows,", "K3 W2 token count")

    dsv4 = read(
        "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_module.c"
    )
    dsv4_cuda = read(
        "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_cuda.cu"
    )
    mimo = read("inference/llms/mimo_2_5/layer.cuh")
    require(dsv4, "float *moe_scores_f32;", "DSV4 FP32 router output")
    require(dsv4, "SparkDsv4LaunchMoeRoute", "DSV4 device grouping")
    require(dsv4, "SparkDsv4LaunchExpertUp", "DSV4 indirect grouped expert up")
    require(dsv4, "SparkDsv4LaunchExpertDown", "DSV4 grouped expert down")
    require(dsv4_cuda, "sqrtf(SparkLmSoftplus(accumulator))", "DSV4 sqrt-softplus router")
    require(dsv4, "SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR", "DSV4 router scale")
    require(mimo, "float *router_logits;", "MiMo FP32 router output")
    reject(dsv4, "for (expert", "DSV4 per-expert host loop")

    model = read("tests/studies/sparkpipe_glm52_batchplane_model.c")
    require(model, "expert_sweeps_per_active_expert = 1.0", "one expert sweep model")
    require(model, "replay/chunk expert-sweep multiplier: 1.0", "removed replay multiplier")
    reject(model, "BP_LAYERS * (BP_EXPERTS", "old queue-depth divisor")

    glm = read("modules/glm52_resident_decode_stage/source/cuda/layer.cuh")
    require(glm, "source_row_map = buffers->route_source_token",
            "GLM routed source-row map")
    require(glm, "LmGemmWeightOnlyIndirectLaunch<",
            "GLM indirect routed W1")
    require(glm, "LmGemmWeightOnlyLaunch<",
            "GLM packed routed W2")
    reject(glm, "LmGatherRowsKernel", "GLM gathered routed activation")
    for relative in (
        "model-families/glm52/include/sparkpipe/spark_glm52_expert_queue.h",
        "model-families/glm52/src/spark_glm52_expert_queue.c",
    ):
        if (ROOT / relative).exists():
            raise SystemExit(f"forbidden legacy GLM expert queue: {relative}")

    print("PASS grouped-MoE source contracts")


if __name__ == "__main__":
    main()
