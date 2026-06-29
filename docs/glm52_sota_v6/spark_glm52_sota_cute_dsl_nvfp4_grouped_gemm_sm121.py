from dataclasses import dataclass


@dataclass(frozen=True)
class Glm52GroupedNvfp4Shape:
    hidden: int = 6144
    moe_intermediate: int = 2048
    experts: int = 256
    top_k: int = 8
    nvfp4_group: int = 16
    mma_m: int = 16
    mma_n: int = 8
    mma_k: int = 64
    cta_m: int = 128
    cta_n: int = 128
    cta_k: int = 128
    warps: int = 8


class SparkGlm52CuteDslGroupedNvfp4GemmSm121:
    def __init__(self, shape: Glm52GroupedNvfp4Shape = Glm52GroupedNvfp4Shape()):
        self.shape = shape

    def emit_expected_kernel_manifest(self) -> dict:
        return {
            "target": "sm_121",
            "family": "sm120f",
            "module": "spark.glm52.grouped_nvfp4_expert_gemm.sm121.cutedsl.v1",
            "tiles": {
                "mma": [self.shape.mma_m, self.shape.mma_n, self.shape.mma_k],
                "cta": [self.shape.cta_m, self.shape.cta_n, self.shape.cta_k],
                "warps": self.shape.warps,
            },
            "operands": {
                "a": "nvfp4_e2m1_payload_with_ue4m3_group16_scales",
                "b": "nvfp4_e2m1_payload_with_ue4m3_group16_scales",
                "accumulator": "f32",
                "gate_up_output": "bf16_or_fused_nvfp4_intermediate",
                "down_output": "f32_weighted_accumulate_then_bf16",
            },
            "fixed_glm52_shapes": {
                "hidden": self.shape.hidden,
                "moe_intermediate": self.shape.moe_intermediate,
                "experts": self.shape.experts,
                "top_k": self.shape.top_k,
            },
            "scheduling": {
                "expert_major_grouped": True,
                "persistent_cta": True,
                "route_slot_cache_required": True,
                "empty_expert_launches": False,
            },
        }


if __name__ == "__main__":
    import json
    print(json.dumps(SparkGlm52CuteDslGroupedNvfp4GemmSm121().emit_expected_kernel_manifest(), indent=2))
