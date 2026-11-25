#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RAIS_VERSION = "1.1"
DEFAULT_BASKET_ID = "v1.0-initial"
DEFAULT_DIVISOR = 1.0
ALPHA_REF = 4.0
SLA_MIN_RATIO = 0.25
GIB_BYTES = float(1 << 30)
PILLAR_ORDER = ("P1", "P2", "P3", "P4", "P5")

SUBMETRIC_SPECS = (
    ("p1a_short_b1", "P1", "1a Short-ctx B1 (C=256)", 0.108),
    ("p1b_long_b1", "P1", "1b Long-ctx B1 (C=64K-128K)", 0.081),
    ("p1c_tail_jitter", "P1", "1c Tail Jitter (p50/p99 ITL)", 0.036),
    ("p2a_spec_acceptance", "P2", "2a Spec Acceptance (alpha vs 4.0)", 0.108),
    ("p2b_verify_b1", "P2", "2b-i B1 Verify & Draft Eff (1/V)", 0.072),
    ("p2b_verify_b8", "P2", "2b-ii B8 Batched Verify Eff (1/V)", 0.045),
    ("p3a_interactive_batch", "P3", "3a Interactive Batch (B=4,8,16 + SLA)", 0.090),
    ("p3b_high_concurrency", "P3", "3b High-Concurrency (B=32,64)", 0.081),
    ("p3c_long_ctx_batch", "P3", "3c Long-Ctx Batch (B=8 @ 32K-64K)", 0.054),
    ("p4a_cold_prefill_8k", "P4", "4a Cold Prefill (8K)", 0.072),
    ("p4b_long_cold_prefill", "P4", "4b Long Cold Prefill (32K/128K+)", 0.081),
    ("p4c_multiturn_prefix_ttft", "P4", "4c Multi-Turn Prefix Reuse TTFT", 0.072),
    ("p5a_cold_boot", "P5", "5a Cold Boot / Model Swap Eff", 0.050),
    ("p5b_memory_efficiency", "P5", "5b Usable KV Memory Eff", 0.030),
    ("p5c_batch_invariance", "P5", "5c Batch-Size Bit-Invariance", 0.020),
)

SUBMETRIC_WEIGHTS = {key: weight for key, _, _, weight in SUBMETRIC_SPECS}
SUBMETRIC_PILLAR = {key: pillar for key, pillar, _, _ in SUBMETRIC_SPECS}
SUBMETRIC_LABELS = {key: label for key, _, label, _ in SUBMETRIC_SPECS}
PILLAR_NAMES = {
    "P1": "Single-Stream Decode",
    "P2": "Speculation",
    "P3": "Batch & Concurrency Scaling",
    "P4": "Prefill & Multi-Turn Agent Loops",
    "P5": "Operational & Systems Physics",
}


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    description: str
    ranks: int
    mem_bw_gb_s_per_rank: float
    bf16_tflops_per_rank: float
    fp8_tflops_per_rank: float
    fp4_tflops_per_rank: float
    net_bw_gb_s_per_rank: float
    net_hop_us: float
    vram_gb_per_rank: float
    nvme_bw_gb_s_per_rank: float
    ram_gib_per_rank: float
    os_reserve_gib_per_rank: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    tier: int
    fleet_weight: float
    hidden: int
    layers: int
    kv_layers: int
    kv_bytes_per_token: int
    state_bytes_per_seq: int
    non_expert_params_b: float
    active_expert_params_b: float
    non_expert_bytes_per_param: float
    expert_bytes_per_param: float
    expert_count: int
    experts_per_token: int
    dtype: str
    compute_precision_non_expert: str
    compute_precision_expert: str
    attn_flops_per_tok_ctx: float
    sparse_ctx_cap: int
    context_tokens: int
    max_sequences: int
    index_bytes_per_token: int = 0
    swa_window: int = 0
    swa_kv_bytes_per_token: int = 0
    canonical_model: str = ""

    @property
    def active_params_b(self) -> float:
        return self.non_expert_params_b + self.active_expert_params_b

    @property
    def active_weight_bytes_b1(self) -> float:
        return (
            self.non_expert_params_b * 1e9 * self.non_expert_bytes_per_param
            + self.active_expert_params_b * 1e9 * self.expert_bytes_per_param
        )

    @property
    def total_weight_bytes(self) -> float:
        spine = self.non_expert_params_b * 1e9 * self.non_expert_bytes_per_param
        if self.expert_count > 0 and self.experts_per_token > 0 and self.active_expert_params_b > 0.0:
            pool = (
                self.active_expert_params_b
                * 1e9
                * (self.expert_count / float(self.experts_per_token))
                * self.expert_bytes_per_param
            )
        else:
            pool = 0.0
        return spine + pool

    @property
    def allreduce_ops_per_step(self) -> int:
        return 2 * self.layers


@dataclass(frozen=True)
class RooflineBreakdown:
    t_compute_ms: float
    t_memory_ms: float
    t_network_ms: float
    t_floor_ms: float
    t_serial_ms: float
    overlap_gain_max: float
    binding_bottleneck: str
    floor_tok_s: float
    mem_bytes_total: float
    mem_bytes_per_rank: float
    flops_total: float
    net_bytes_per_rank: float


HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "16x-gb10-1x100g": HardwareProfile(
        name="16x-gb10-1x100g",
        description="16x NVIDIA DGX Spark (GB10, 128 GB, 273 GB/s) over 1x100GbE RoCE",
        ranks=16,
        mem_bw_gb_s_per_rank=273.0,
        bf16_tflops_per_rank=90.0,
        fp8_tflops_per_rank=210.0,
        fp4_tflops_per_rank=420.0,
        net_bw_gb_s_per_rank=12.5,
        net_hop_us=16.0,
        vram_gb_per_rank=128.0,
        nvme_bw_gb_s_per_rank=5.0,
        ram_gib_per_rank=128.0,
        os_reserve_gib_per_rank=12.0,
    ),
    "16x-gb10-2x100g": HardwareProfile(
        name="16x-gb10-2x100g",
        description="16x NVIDIA DGX Spark (GB10, 128 GB, 273 GB/s) over dual 100GbE + direct pair rail",
        ranks=16,
        mem_bw_gb_s_per_rank=273.0,
        bf16_tflops_per_rank=90.0,
        fp8_tflops_per_rank=210.0,
        fp4_tflops_per_rank=420.0,
        net_bw_gb_s_per_rank=25.0,
        net_hop_us=16.0,
        vram_gb_per_rank=128.0,
        nvme_bw_gb_s_per_rank=5.0,
        ram_gib_per_rank=128.0,
        os_reserve_gib_per_rank=12.0,
    ),
    "4x-gb10-1x100g": HardwareProfile(
        name="4x-gb10-1x100g",
        description="4x NVIDIA DGX Spark (GB10, 128 GB, 273 GB/s) over 1x100GbE RoCE",
        ranks=4,
        mem_bw_gb_s_per_rank=273.0,
        bf16_tflops_per_rank=90.0,
        fp8_tflops_per_rank=210.0,
        fp4_tflops_per_rank=420.0,
        net_bw_gb_s_per_rank=12.5,
        net_hop_us=12.0,
        vram_gb_per_rank=128.0,
        nvme_bw_gb_s_per_rank=5.0,
        ram_gib_per_rank=128.0,
        os_reserve_gib_per_rank=12.0,
    ),
    "4x-gb10-2x100g": HardwareProfile(
        name="4x-gb10-2x100g",
        description="4x NVIDIA DGX Spark (GB10, 128 GB, 273 GB/s) over dual 100GbE RoCE",
        ranks=4,
        mem_bw_gb_s_per_rank=273.0,
        bf16_tflops_per_rank=90.0,
        fp8_tflops_per_rank=210.0,
        fp4_tflops_per_rank=420.0,
        net_bw_gb_s_per_rank=25.0,
        net_hop_us=12.0,
        vram_gb_per_rank=128.0,
        nvme_bw_gb_s_per_rank=5.0,
        ram_gib_per_rank=128.0,
        os_reserve_gib_per_rank=12.0,
    ),
    "1x-gb10": HardwareProfile(
        name="1x-gb10",
        description="Single NVIDIA DGX Spark (GB10, 128 GB, 273 GB/s)",
        ranks=1,
        mem_bw_gb_s_per_rank=273.0,
        bf16_tflops_per_rank=90.0,
        fp8_tflops_per_rank=210.0,
        fp4_tflops_per_rank=420.0,
        net_bw_gb_s_per_rank=0.0,
        net_hop_us=0.0,
        vram_gb_per_rank=128.0,
        nvme_bw_gb_s_per_rank=5.0,
        ram_gib_per_rank=128.0,
        os_reserve_gib_per_rank=12.0,
    ),
    "1x-m5-ultra": HardwareProfile(
        name="1x-m5-ultra",
        description="Single Apple Mac Studio M5 Ultra (256 GB, 1144 GB/s unified memory)",
        ranks=1,
        mem_bw_gb_s_per_rank=1144.0,
        bf16_tflops_per_rank=95.0,
        fp8_tflops_per_rank=190.0,
        fp4_tflops_per_rank=190.0,
        net_bw_gb_s_per_rank=0.0,
        net_hop_us=0.0,
        vram_gb_per_rank=256.0,
        nvme_bw_gb_s_per_rank=7.0,
        ram_gib_per_rank=256.0,
        os_reserve_gib_per_rank=32.0,
    ),
    "8x-h200-nvlink": HardwareProfile(
        name="8x-h200-nvlink",
        description="8x NVIDIA H200 SXM (141 GB HBM3e, 4.8 TB/s, 900 GB/s NVLink)",
        ranks=8,
        mem_bw_gb_s_per_rank=4800.0,
        bf16_tflops_per_rank=989.0,
        fp8_tflops_per_rank=1979.0,
        fp4_tflops_per_rank=1979.0,
        net_bw_gb_s_per_rank=450.0,
        net_hop_us=2.0,
        vram_gb_per_rank=141.0,
        nvme_bw_gb_s_per_rank=7.0,
        ram_gib_per_rank=141.0,
        os_reserve_gib_per_rank=4.0,
    ),
}


MODEL_SPECS: dict[str, ModelSpec] = {
    "kimi-k3.1-pro": ModelSpec(
        name="kimi-k3.1-pro",
        family="k3",
        tier=1,
        fleet_weight=0.13,
        hidden=7168,
        layers=93,
        kv_layers=24,
        kv_bytes_per_token=27648,
        state_bytes_per_seq=69 * 96 * 128 * 128 * 4 * 2,
        non_expert_params_b=55.6,
        active_expert_params_b=48.6,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=0.53125,
        expert_count=896,
        experts_per_token=16,
        dtype="BF16+MXFP4",
        compute_precision_non_expert="bf16",
        compute_precision_expert="fp4",
        attn_flops_per_tok_ctx=2.0 * 24 * 96 * (192 + 128),
        sparse_ctx_cap=0,
        context_tokens=1048576,
        max_sequences=8,
    ),
    "glm-5.3": ModelSpec(
        name="glm-5.3",
        family="glm52",
        tier=1,
        fleet_weight=0.12,
        hidden=6 * 1024,
        layers=78,
        kv_layers=78,
        kv_bytes_per_token=89856,
        state_bytes_per_seq=0,
        non_expert_params_b=17.5,
        active_expert_params_b=22.7,
        non_expert_bytes_per_param=1.0,
        expert_bytes_per_param=1.0,
        expert_count=256,
        experts_per_token=8,
        dtype="FP8",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=2.0 * 78 * 64 * (192 + 128),
        sparse_ctx_cap=2048,
        context_tokens=1048576,
        max_sequences=8,
        index_bytes_per_token=21 * 256,
    ),
    "glm-5.3-fp8": ModelSpec(
        name="glm-5.3-fp8",
        family="glm52",
        tier=1,
        fleet_weight=0.0,
        hidden=6 * 1024,
        layers=78,
        kv_layers=78,
        kv_bytes_per_token=89856,
        state_bytes_per_seq=0,
        non_expert_params_b=17.5,
        active_expert_params_b=22.7,
        non_expert_bytes_per_param=1.0,
        expert_bytes_per_param=1.0,
        expert_count=256,
        experts_per_token=8,
        dtype="FP8",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=2.0 * 78 * 64 * (192 + 128),
        sparse_ctx_cap=2048,
        context_tokens=1048576,
        max_sequences=8,
        index_bytes_per_token=21 * 256,
        canonical_model="glm-5.3",
    ),
    "glm-5.3-flash": ModelSpec(
        name="glm-5.3-flash",
        family="glm5_next",
        tier=1,
        fleet_weight=0.11,
        hidden=4096,
        layers=45,
        kv_layers=11,
        kv_bytes_per_token=11264,
        state_bytes_per_seq=34 * 64 * 128 * 128 * 4 * 2,
        non_expert_params_b=7.0,
        active_expert_params_b=11.0,
        non_expert_bytes_per_param=1.0,
        expert_bytes_per_param=1.0,
        expert_count=288,
        experts_per_token=8,
        dtype="FP8",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=2.0 * 11 * 64 * (256 + 256),
        sparse_ctx_cap=2048,
        context_tokens=131072,
        max_sequences=16,
        index_bytes_per_token=11 * 256,
    ),
    "glm-5.3-flash-fp8": ModelSpec(
        name="glm-5.3-flash-fp8",
        family="glm5_next",
        tier=1,
        fleet_weight=0.0,
        hidden=4096,
        layers=45,
        kv_layers=11,
        kv_bytes_per_token=11264,
        state_bytes_per_seq=34 * 64 * 128 * 128 * 4 * 2,
        non_expert_params_b=7.0,
        active_expert_params_b=11.0,
        non_expert_bytes_per_param=1.0,
        expert_bytes_per_param=1.0,
        expert_count=288,
        experts_per_token=8,
        dtype="FP8",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=2.0 * 11 * 64 * (256 + 256),
        sparse_ctx_cap=2048,
        context_tokens=131072,
        max_sequences=16,
        index_bytes_per_token=11 * 256,
        canonical_model="glm-5.3-flash",
    ),
    "deepseek-v4.1-flash": ModelSpec(
        name="deepseek-v4.1-flash",
        family="dsv41_flash",
        tier=1,
        fleet_weight=0.10,
        hidden=5120,
        layers=40,
        kv_layers=38,
        kv_bytes_per_token=3200,
        state_bytes_per_seq=0,
        non_expert_params_b=6.2,
        active_expert_params_b=8.5,
        non_expert_bytes_per_param=1.03125,
        expert_bytes_per_param=0.53125,
        expert_count=384,
        experts_per_token=6,
        dtype="FP8+MXFP4",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp4",
        attn_flops_per_tok_ctx=2.0 * 38 * 64 * (512 + 512),
        sparse_ctx_cap=512,
        context_tokens=1048576,
        max_sequences=8,
        index_bytes_per_token=8 * 128,
        swa_window=128,
        swa_kv_bytes_per_token=2 * 1280 * 2,
    ),
    "kimi-k3": ModelSpec(
        name="kimi-k3",
        family="k3",
        tier=1,
        fleet_weight=0.09,
        hidden=7168,
        layers=93,
        kv_layers=24,
        kv_bytes_per_token=27648,
        state_bytes_per_seq=69 * 96 * 128 * 128 * 4 * 2,
        non_expert_params_b=55.6,
        active_expert_params_b=48.6,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=0.53125,
        expert_count=896,
        experts_per_token=16,
        dtype="BF16+MXFP4",
        compute_precision_non_expert="bf16",
        compute_precision_expert="fp4",
        attn_flops_per_tok_ctx=2.0 * 24 * 96 * (192 + 128),
        sparse_ctx_cap=0,
        context_tokens=1048576,
        max_sequences=8,
    ),
    "qwen3.8-27b": ModelSpec(
        name="qwen3.8-27b",
        family="qwen38",
        tier=2,
        fleet_weight=0.050,
        hidden=5120,
        layers=64,
        kv_layers=16,
        kv_bytes_per_token=65536,
        state_bytes_per_seq=48 * 48 * 128 * 128 * 4 * 2,
        non_expert_params_b=27.0,
        active_expert_params_b=0.0,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=2.0,
        expert_count=0,
        experts_per_token=0,
        dtype="BF16",
        compute_precision_non_expert="bf16",
        compute_precision_expert="bf16",
        attn_flops_per_tok_ctx=4.0 * 16 * 24 * 256,
        sparse_ctx_cap=0,
        context_tokens=262144,
        max_sequences=8,
    ),
    "qwen3.8-27b-fp8": ModelSpec(
        name="qwen3.8-27b-fp8",
        family="qwen38",
        tier=2,
        fleet_weight=0.0,
        hidden=5120,
        layers=64,
        kv_layers=16,
        kv_bytes_per_token=65536,
        state_bytes_per_seq=48 * 48 * 128 * 128 * 4 * 2,
        non_expert_params_b=27.0,
        active_expert_params_b=0.0,
        non_expert_bytes_per_param=1.0,
        expert_bytes_per_param=1.0,
        expert_count=0,
        experts_per_token=0,
        dtype="FP8",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=4.0 * 16 * 24 * 256,
        sparse_ctx_cap=0,
        context_tokens=262144,
        max_sequences=8,
        canonical_model="qwen3.8-27b",
    ),
    "qwen3.8-max": ModelSpec(
        name="qwen3.8-max",
        family="qwen38",
        tier=2,
        fleet_weight=0.040,
        hidden=8192,
        layers=92,
        kv_layers=23,
        kv_bytes_per_token=94208,
        state_bytes_per_seq=69 * 128 * 128 * 128 * 4 * 2,
        non_expert_params_b=48.7,
        active_expert_params_b=46.3,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=0.5625,
        expert_count=512,
        experts_per_token=10,
        dtype="BF16+NVFP4",
        compute_precision_non_expert="bf16",
        compute_precision_expert="fp4",
        attn_flops_per_tok_ctx=4.0 * 23 * 64 * 256,
        sparse_ctx_cap=0,
        context_tokens=32768,
        max_sequences=16,
    ),
    "mimo-v2.6-pro": ModelSpec(
        name="mimo-v2.6-pro",
        family="mimo26",
        tier=2,
        fleet_weight=0.035,
        hidden=6 * 1024,
        layers=70,
        kv_layers=10,
        kv_bytes_per_token=51200,
        state_bytes_per_seq=0,
        non_expert_params_b=19.2,
        active_expert_params_b=20.8,
        non_expert_bytes_per_param=1.03125,
        expert_bytes_per_param=0.53125,
        expert_count=384,
        experts_per_token=8,
        dtype="FP8+MXFP4",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp4",
        attn_flops_per_tok_ctx=2.0 * 10 * 128 * (192 + 128),
        sparse_ctx_cap=0,
        context_tokens=32768,
        max_sequences=16,
        swa_window=128,
        swa_kv_bytes_per_token=60 * 8 * (192 + 128) * 2,
    ),
    "hy4-preview": ModelSpec(
        name="hy4-preview",
        family="hy4",
        tier=2,
        fleet_weight=0.035,
        hidden=6 * 1024,
        layers=78,
        kv_layers=78,
        kv_bytes_per_token=89856,
        state_bytes_per_seq=0,
        non_expert_params_b=24.0,
        active_expert_params_b=25.0,
        non_expert_bytes_per_param=1.03125,
        expert_bytes_per_param=1.03125,
        expert_count=256,
        experts_per_token=8,
        dtype="MXFP8",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=2.0 * 78 * 64 * (256 + 256),
        sparse_ctx_cap=2048,
        context_tokens=32768,
        max_sequences=16,
        index_bytes_per_token=21 * 256,
    ),
    "k-exaone-2.0": ModelSpec(
        name="k-exaone-2.0",
        family="kexaone2",
        tier=2,
        fleet_weight=0.030,
        hidden=6 * 1024,
        layers=78,
        kv_layers=20,
        kv_bytes_per_token=81920,
        state_bytes_per_seq=0,
        non_expert_params_b=14.5,
        active_expert_params_b=23.0,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=2.0,
        expert_count=256,
        experts_per_token=8,
        dtype="BF16",
        compute_precision_non_expert="bf16",
        compute_precision_expert="bf16",
        attn_flops_per_tok_ctx=4.0 * 20 * 64 * 128,
        sparse_ctx_cap=0,
        context_tokens=262144,
        max_sequences=4,
        swa_window=128,
        swa_kv_bytes_per_token=58 * 8 * 128 * 4,
    ),
    "a.x-k2": ModelSpec(
        name="a.x-k2",
        family="axk2",
        tier=2,
        fleet_weight=0.030,
        hidden=7168,
        layers=61,
        kv_layers=61,
        kv_bytes_per_token=70272,
        state_bytes_per_seq=0,
        non_expert_params_b=16.0,
        active_expert_params_b=21.1,
        non_expert_bytes_per_param=1.03125,
        expert_bytes_per_param=1.03125,
        expert_count=256,
        experts_per_token=8,
        dtype="FP8",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=2.0 * 61 * 64 * (192 + 128),
        sparse_ctx_cap=2048,
        context_tokens=262144,
        max_sequences=8,
        index_bytes_per_token=61 * 256,
    ),
    "solar-open-2": ModelSpec(
        name="solar-open-2",
        family="solar_open2",
        tier=2,
        fleet_weight=0.030,
        hidden=4096,
        layers=48,
        kv_layers=12,
        kv_bytes_per_token=49152,
        state_bytes_per_seq=36 * 64 * 128 * 128 * 4 * 2,
        non_expert_params_b=6.5,
        active_expert_params_b=6.0,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=2.0,
        expert_count=320,
        experts_per_token=8,
        dtype="BF16",
        compute_precision_non_expert="bf16",
        compute_precision_expert="bf16",
        attn_flops_per_tok_ctx=4.0 * 12 * 64 * 128,
        sparse_ctx_cap=0,
        context_tokens=262144,
        max_sequences=16,
    ),
    "mimo-v2.6-flash": ModelSpec(
        name="mimo-v2.6-flash",
        family="mimo26",
        tier=3,
        fleet_weight=0.035,
        hidden=4096,
        layers=48,
        kv_layers=9,
        kv_bytes_per_token=23040,
        state_bytes_per_seq=0,
        non_expert_params_b=5.5,
        active_expert_params_b=9.5,
        non_expert_bytes_per_param=1.03125,
        expert_bytes_per_param=0.53125,
        expert_count=256,
        experts_per_token=8,
        dtype="FP8+MXFP4",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp4",
        attn_flops_per_tok_ctx=2.0 * 9 * 64 * (192 + 128),
        sparse_ctx_cap=0,
        context_tokens=32768,
        max_sequences=16,
        swa_window=128,
        swa_kv_bytes_per_token=39 * 8 * (192 + 128) * 2,
    ),
    "gemma-4-31b": ModelSpec(
        name="gemma-4-31b",
        family="gemma4",
        tier=3,
        fleet_weight=0.030,
        hidden=5376,
        layers=60,
        kv_layers=10,
        kv_bytes_per_token=81920,
        state_bytes_per_seq=0,
        non_expert_params_b=30.7,
        active_expert_params_b=0.0,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=2.0,
        expert_count=0,
        experts_per_token=0,
        dtype="BF16",
        compute_precision_non_expert="bf16",
        compute_precision_expert="bf16",
        attn_flops_per_tok_ctx=4.0 * 10 * 32 * 512,
        sparse_ctx_cap=0,
        context_tokens=32768,
        max_sequences=16,
        swa_window=1024,
        swa_kv_bytes_per_token=50 * 16 * 256 * 4,
    ),
    "gemma-4-26b-a4b": ModelSpec(
        name="gemma-4-26b-a4b",
        family="gemma4",
        tier=3,
        fleet_weight=0.030,
        hidden=2816,
        layers=30,
        kv_layers=5,
        kv_bytes_per_token=20480,
        state_bytes_per_seq=0,
        non_expert_params_b=1.9,
        active_expert_params_b=1.9,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=2.0,
        expert_count=128,
        experts_per_token=8,
        dtype="BF16",
        compute_precision_non_expert="bf16",
        compute_precision_expert="bf16",
        attn_flops_per_tok_ctx=4.0 * 5 * 16 * 512,
        sparse_ctx_cap=0,
        context_tokens=262144,
        max_sequences=16,
        swa_window=1024,
        swa_kv_bytes_per_token=25 * 8 * 256 * 4,
    ),
    "ling-3.0-flash": ModelSpec(
        name="ling-3.0-flash",
        family="ling",
        tier=3,
        fleet_weight=0.030,
        hidden=2560,
        layers=42,
        kv_layers=7,
        kv_bytes_per_token=8064,
        state_bytes_per_seq=35 * 32 * 128 * 128 * 4 * 2,
        non_expert_params_b=2.6,
        active_expert_params_b=2.5,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=2.0,
        expert_count=512,
        experts_per_token=8,
        dtype="BF16",
        compute_precision_non_expert="bf16",
        compute_precision_expert="bf16",
        attn_flops_per_tok_ctx=2.0 * 7 * 32 * (192 + 128),
        sparse_ctx_cap=0,
        context_tokens=262144,
        max_sequences=16,
    ),
    "ling-3.0-flash-fin": ModelSpec(
        name="ling-3.0-flash-fin",
        family="ling",
        tier=3,
        fleet_weight=0.0,
        hidden=2560,
        layers=42,
        kv_layers=7,
        kv_bytes_per_token=8064,
        state_bytes_per_seq=35 * 32 * 128 * 128 * 4 * 2,
        non_expert_params_b=2.6,
        active_expert_params_b=2.5,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=2.0,
        expert_count=512,
        experts_per_token=8,
        dtype="BF16",
        compute_precision_non_expert="bf16",
        compute_precision_expert="bf16",
        attn_flops_per_tok_ctx=2.0 * 7 * 32 * (192 + 128),
        sparse_ctx_cap=0,
        context_tokens=262144,
        max_sequences=16,
        canonical_model="ling-3.0-flash",
    ),
    "muse-glimmer-30b": ModelSpec(
        name="muse-glimmer-30b",
        family="muse_glimmer",
        tier=3,
        fleet_weight=0.025,
        hidden=6656,
        layers=52,
        kv_layers=13,
        kv_bytes_per_token=13312,
        state_bytes_per_seq=0,
        non_expert_params_b=29.2,
        active_expert_params_b=0.0,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=2.0,
        expert_count=0,
        experts_per_token=0,
        dtype="BF16",
        compute_precision_non_expert="bf16",
        compute_precision_expert="bf16",
        attn_flops_per_tok_ctx=4.0 * 13 * 32 * 128,
        sparse_ctx_cap=0,
        context_tokens=131072,
        max_sequences=16,
        swa_window=2048,
        swa_kv_bytes_per_token=39 * 2 * 128 * 4,
    ),
    "laguna-s-2.1": ModelSpec(
        name="laguna-s-2.1",
        family="laguna",
        tier=3,
        fleet_weight=0.025,
        hidden=3072,
        layers=48,
        kv_layers=12,
        kv_bytes_per_token=49152,
        state_bytes_per_seq=0,
        non_expert_params_b=3.1,
        active_expert_params_b=4.4,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=2.0,
        expert_count=256,
        experts_per_token=10,
        dtype="BF16",
        compute_precision_non_expert="bf16",
        compute_precision_expert="bf16",
        attn_flops_per_tok_ctx=4.0 * 12 * 48 * 128,
        sparse_ctx_cap=0,
        context_tokens=32768,
        max_sequences=16,
        swa_window=512,
        swa_kv_bytes_per_token=36 * 8 * 128 * 4,
    ),
    "qwen3.8-flash": ModelSpec(
        name="qwen3.8-flash",
        family="qwen38",
        tier=3,
        fleet_weight=0.025,
        hidden=2560,
        layers=48,
        kv_layers=12,
        kv_bytes_per_token=24576,
        state_bytes_per_seq=36 * 48 * 128 * 128 * 4 * 2,
        non_expert_params_b=3.0,
        active_expert_params_b=2.4,
        non_expert_bytes_per_param=2.0,
        expert_bytes_per_param=1.03125,
        expert_count=512,
        experts_per_token=10,
        dtype="BF16+FP8",
        compute_precision_non_expert="bf16",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=4.0 * 12 * 24 * 256,
        sparse_ctx_cap=2048,
        context_tokens=262144,
        max_sequences=16,
        index_bytes_per_token=12 * 128,
    ),
    "hy4-fp8": ModelSpec(
        name="hy4-fp8",
        family="hy4",
        tier=2,
        fleet_weight=0.0,
        hidden=6 * 1024,
        layers=78,
        kv_layers=78,
        kv_bytes_per_token=89856,
        state_bytes_per_seq=0,
        non_expert_params_b=24.0,
        active_expert_params_b=25.0,
        non_expert_bytes_per_param=1.40,
        expert_bytes_per_param=1.0,
        expert_count=256,
        experts_per_token=8,
        dtype="BF16+FP8",
        compute_precision_non_expert="bf16",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=2.0 * 78 * 64 * (256 + 256),
        sparse_ctx_cap=2048,
        context_tokens=1048576,
        max_sequences=16,
        index_bytes_per_token=21 * 256,
    ),
    "ling-3.1-flash": ModelSpec(
        name="ling-3.1-flash",
        family="ling",
        tier=1,
        fleet_weight=0.0,
        hidden=4096,
        layers=72,
        kv_layers=9,
        kv_bytes_per_token=10368,
        state_bytes_per_seq=63 * 64 * 128 * 128 * 4 * 2,
        non_expert_params_b=15.7,
        active_expert_params_b=8.6,
        non_expert_bytes_per_param=1.0,
        expert_bytes_per_param=1.0,
        expert_count=512,
        experts_per_token=8,
        dtype="FP8",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=2.0 * 9 * 64 * (192 + 128),
        sparse_ctx_cap=0,
        context_tokens=262144,
        max_sequences=8,
    ),
    "ling-3.2-air": ModelSpec(
        name="ling-3.2-air",
        family="ling",
        tier=3,
        fleet_weight=0.0,
        hidden=2560,
        layers=42,
        kv_layers=7,
        kv_bytes_per_token=8064,
        state_bytes_per_seq=35 * 32 * 128 * 128 * 4 * 2,
        non_expert_params_b=2.6,
        active_expert_params_b=2.5,
        non_expert_bytes_per_param=1.0,
        expert_bytes_per_param=1.0,
        expert_count=512,
        experts_per_token=8,
        dtype="FP8",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=2.0 * 7 * 32 * (192 + 128),
        sparse_ctx_cap=0,
        context_tokens=262144,
        max_sequences=16,
    ),
    "sage-s2.1-flash": ModelSpec(
        name="sage-s2.1-flash",
        family="glm5_next",
        tier=1,
        fleet_weight=0.0,
        hidden=4096,
        layers=45,
        kv_layers=11,
        kv_bytes_per_token=11264,
        state_bytes_per_seq=34 * 64 * 128 * 128 * 4 * 2,
        non_expert_params_b=7.0,
        active_expert_params_b=11.0,
        non_expert_bytes_per_param=1.0,
        expert_bytes_per_param=1.0,
        expert_count=288,
        experts_per_token=8,
        dtype="FP8",
        compute_precision_non_expert="fp8",
        compute_precision_expert="fp8",
        attn_flops_per_tok_ctx=2.0 * 11 * 64 * (256 + 256),
        sparse_ctx_cap=2048,
        context_tokens=131072,
        max_sequences=16,
        index_bytes_per_token=11 * 256,
    ),
}

MODEL_ALIASES: dict[str, str] = {
    "glm53_full": "glm-5.3",
    "glm53_full_fp8": "glm-5.3-fp8",
    "glmfull": "glm-5.3",
    "glm53_flash": "glm-5.3-flash",
    "glm53_flash_fp8": "glm-5.3-flash-fp8",
    "glmflash": "glm-5.3-flash",
    "sage21_flash": "sage-s2.1-flash",
    "sage_s21_flash": "sage-s2.1-flash",
    "k3": "kimi-k3",
    "k31_pro": "kimi-k3.1-pro",
    "qwen38_27b": "qwen3.8-27b",
    "qwen38_27b_fp8": "qwen3.8-27b-fp8",
    "dsv41_flash": "deepseek-v4.1-flash",
    "dsv41_flash_rows16": "deepseek-v4.1-flash",
    "qwen38_max": "qwen3.8-max",
    "qwen38_flash": "qwen3.8-flash",
    "gemma4_31b": "gemma-4-31b",
    "gemma4_26b": "gemma-4-26b-a4b",
    "ling3_flash": "ling-3.0-flash",
    "ling3_fin": "ling-3.0-flash-fin",
    "ling31_flash": "ling-3.1-flash",
    "ling32_air": "ling-3.2-air",
    "hy4": "hy4-preview",
    "hy4_fp8": "hy4-fp8",
    "muse_glimmer": "muse-glimmer-30b",
    "laguna": "laguna-s-2.1",
    "mimo26_flash": "mimo-v2.6-flash",
    "mimo26_pro": "mimo-v2.6-pro",
    "kexaone2": "k-exaone-2.0",
    "axk2": "a.x-k2",
    "solar_open2": "solar-open-2",
}


def resolve_model_name(name: str) -> str:
    key = name.strip().lower()
    if key in MODEL_SPECS:
        return key
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_SPECS)}")


def resolve_hardware(name: str) -> HardwareProfile:
    key = name.strip().lower()
    if key in HARDWARE_PROFILES:
        return HARDWARE_PROFILES[key]
    raise KeyError(f"unknown hardware profile {name!r}; known: {sorted(HARDWARE_PROFILES)}")


def _tflops_for_precision(hw: HardwareProfile, precision: str) -> float:
    if precision == "fp4":
        return hw.fp4_tflops_per_rank
    if precision == "fp8":
        return hw.fp8_tflops_per_rank
    return hw.bf16_tflops_per_rank


def touched_expert_fraction(rows: int, expert_count: int, experts_per_token: int) -> float:
    if expert_count <= 0 or experts_per_token <= 0 or rows <= 0:
        return 0.0
    p = min(1.0, experts_per_token / float(expert_count))
    return 1.0 - (1.0 - p) ** rows


def compute_roofline(
    model: ModelSpec,
    hw: HardwareProfile,
    batch: int,
    ctx: int,
    rows_per_seq: int = 1,
    is_prefill: bool = False,
    prefix_ctx: int = 0,
) -> RooflineBreakdown:
    total_rows = max(1, batch * rows_per_seq)
    ranks = max(1, hw.ranks)
    non_expert_flops = 2.0 * model.non_expert_params_b * 1e9 * total_rows
    expert_flops = 2.0 * model.active_expert_params_b * 1e9 * total_rows
    if is_prefill:
        mean_ctx = prefix_ctx + 0.5 * rows_per_seq
        eff_ctx = min(mean_ctx, float(model.sparse_ctx_cap)) if model.sparse_ctx_cap > 0 else mean_ctx
        attn_flops = total_rows * eff_ctx * model.attn_flops_per_tok_ctx
    else:
        eff_ctx = float(min(ctx, model.sparse_ctx_cap)) if model.sparse_ctx_cap > 0 else float(ctx)
        attn_flops = total_rows * eff_ctx * model.attn_flops_per_tok_ctx
    flops_total = non_expert_flops + expert_flops + attn_flops

    t_c_non_expert = non_expert_flops / (ranks * _tflops_for_precision(hw, model.compute_precision_non_expert) * 1e12)
    t_c_expert = expert_flops / (ranks * _tflops_for_precision(hw, model.compute_precision_expert) * 1e12)
    t_c_attn = attn_flops / (ranks * hw.bf16_tflops_per_rank * 1e12)
    t_compute_s = t_c_non_expert + t_c_expert + t_c_attn

    spine_bytes = model.non_expert_params_b * 1e9 * model.non_expert_bytes_per_param
    if model.expert_count > 0 and model.experts_per_token > 0 and model.active_expert_params_b > 0.0:
        pool_bytes = (
            model.active_expert_params_b
            * 1e9
            * (model.expert_count / float(model.experts_per_token))
            * model.expert_bytes_per_param
        )
        expert_bytes = pool_bytes * touched_expert_fraction(total_rows, model.expert_count, model.experts_per_token)
    else:
        expert_bytes = 0.0

    if is_prefill:
        kv_read_ctx = min(prefix_ctx, model.sparse_ctx_cap) if (model.sparse_ctx_cap > 0 and prefix_ctx > 0) else prefix_ctx
        index_bytes = batch * (prefix_ctx + rows_per_seq) * model.index_bytes_per_token
        swa_bytes = batch * min(prefix_ctx + rows_per_seq, model.swa_window) * model.swa_kv_bytes_per_token
        kv_bytes = (
            batch * kv_read_ctx * model.kv_bytes_per_token
            + 2.0 * total_rows * model.kv_bytes_per_token
            + index_bytes
            + swa_bytes
            + batch * model.state_bytes_per_seq
        )
    else:
        kv_eff_ctx = min(ctx, model.sparse_ctx_cap) if model.sparse_ctx_cap > 0 else ctx
        index_bytes = batch * ctx * model.index_bytes_per_token
        swa_bytes = batch * min(ctx, model.swa_window) * model.swa_kv_bytes_per_token
        kv_bytes = (
            batch * kv_eff_ctx * model.kv_bytes_per_token
            + index_bytes
            + swa_bytes
            + batch * model.state_bytes_per_seq
        )

    mem_bytes_total = spine_bytes + expert_bytes + kv_bytes
    mem_bytes_per_rank = mem_bytes_total / ranks
    t_memory_s = mem_bytes_total / (ranks * hw.mem_bw_gb_s_per_rank * 1e9)

    if ranks == 1 or hw.net_bw_gb_s_per_rank <= 0.0:
        net_bytes_per_rank = 0.0
        t_network_s = 0.0
    else:
        ops = model.allreduce_ops_per_step
        payload_bytes = total_rows * model.hidden * 2.0
        wire_factor = 2.0 * (ranks - 1) / float(ranks)
        net_bytes_per_rank = ops * wire_factor * payload_bytes
        t_net_lat_s = ops * (hw.net_hop_us * 1e-6)
        t_net_bw_s = net_bytes_per_rank / (hw.net_bw_gb_s_per_rank * 1e9)
        t_network_s = t_net_lat_s + t_net_bw_s

    t_compute_ms = t_compute_s * 1000.0
    t_memory_ms = t_memory_s * 1000.0
    t_network_ms = t_network_s * 1000.0

    triples = (
        ("compute", t_compute_ms),
        ("memory", t_memory_ms),
        ("network", t_network_ms),
    )
    binding_bottleneck, t_floor_ms = max(triples, key=lambda item: item[1])
    t_serial_ms = t_compute_ms + t_memory_ms + t_network_ms
    overlap_gain_max = t_serial_ms / t_floor_ms if t_floor_ms > 0.0 else 1.0
    floor_tok_s = (total_rows * 1000.0 / t_floor_ms) if t_floor_ms > 0.0 else 0.0

    return RooflineBreakdown(
        t_compute_ms=round(t_compute_ms, 4),
        t_memory_ms=round(t_memory_ms, 4),
        t_network_ms=round(t_network_ms, 4),
        t_floor_ms=round(t_floor_ms, 4),
        t_serial_ms=round(t_serial_ms, 4),
        overlap_gain_max=round(overlap_gain_max, 4),
        binding_bottleneck=binding_bottleneck,
        floor_tok_s=round(floor_tok_s, 3),
        mem_bytes_total=round(mem_bytes_total, 1),
        mem_bytes_per_rank=round(mem_bytes_per_rank, 1),
        flops_total=round(flops_total, 1),
        net_bytes_per_rank=round(net_bytes_per_rank, 1),
    )


def _clamp_score(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 0.0
    return min(100.0, value)


def _eff_from_tok_s(actual_tok_s: float | None, floor_tok_s: float) -> tuple[float, float]:
    if actual_tok_s is None or actual_tok_s <= 0.0 or floor_tok_s <= 0.0:
        return 0.0, 0.0
    raw = 100.0 * actual_tok_s / floor_tok_s
    return _clamp_score(raw), round(raw, 2)


def _eff_from_ms(actual_ms: float | None, floor_ms: float) -> tuple[float, float]:
    if actual_ms is None or actual_ms <= 0.0 or floor_ms <= 0.0:
        return 0.0, 0.0
    raw = 100.0 * floor_ms / actual_ms
    return _clamp_score(raw), round(raw, 2)


def evaluate_gates(record: dict[str, Any]) -> dict[str, Any]:
    gates = record.get("gates")
    if not isinstance(gates, dict) or not gates:
        return {
            "status": "UNVERIFIED",
            "valid": False,
            "failed_gates": [],
            "prose_exact": None,
            "code_exact": None,
            "prefix_exact": None,
            "errors": None,
        }
    prose_exact = gates.get("prose_exact", gates.get("spec_exact_prose"))
    code_exact = gates.get("code_exact", gates.get("spec_exact_code"))
    prefix_exact = gates.get("prefix_exact", gates.get("gate_gx_exact"))
    errors = gates.get("errors", 0)
    ooms = gates.get("ooms", 0)
    xids = gates.get("xids", 0)
    failed: list[str] = []
    if prose_exact is False:
        failed.append("spec_exact_prose")
    if code_exact is False:
        failed.append("spec_exact_code")
    if prefix_exact is False:
        failed.append("prefix_exact_gx")
    if isinstance(errors, (int, float)) and errors > 0:
        failed.append(f"errors={int(errors)}")
    if isinstance(ooms, (int, float)) and ooms > 0:
        failed.append(f"ooms={int(ooms)}")
    if isinstance(xids, (int, float)) and xids > 0:
        failed.append(f"xids={int(xids)}")

    if failed:
        status = "INVALID"
        valid = False
    elif prose_exact is True and code_exact is True and prefix_exact is True and errors == 0:
        status = "VALID"
        valid = True
    else:
        status = "UNVERIFIED"
        valid = False

    return {
        "status": status,
        "valid": valid,
        "failed_gates": failed,
        "prose_exact": prose_exact,
        "code_exact": code_exact,
        "prefix_exact": prefix_exact,
        "errors": errors,
    }


def _extract_batch_map(metrics: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw = metrics.get("batch")
    out: dict[int, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for key, entry in raw.items():
            try:
                b = int(str(key).lstrip("bB"))
            except ValueError:
                continue
            if isinstance(entry, dict):
                out[b] = entry
            elif isinstance(entry, (int, float)):
                out[b] = {"aggregate_tok_s": float(entry)}
    for b in (1, 2, 4, 8, 16, 32, 64):
        flat_key = f"b{b}_tok_s"
        if b not in out and isinstance(metrics.get(flat_key), (int, float)):
            out[b] = {"aggregate_tok_s": float(metrics[flat_key])}
    return out


def _best_working_aggregate(batch_map: dict[int, dict[str, Any]], b1_tok_s: float | None, target_b: int) -> tuple[float, int | None, bool]:
    entry = batch_map.get(target_b)
    if entry and not entry.get("unsupported") and not entry.get("failed"):
        agg = entry.get("aggregate_tok_s")
        if isinstance(agg, (int, float)) and agg > 0.0:
            return float(agg), target_b, False
    best_agg = 0.0
    best_b: int | None = None
    if b1_tok_s and b1_tok_s > 0.0:
        best_agg = float(b1_tok_s)
        best_b = 1
    for b in sorted(batch_map):
        if b >= target_b:
            continue
        item = batch_map[b]
        if item.get("unsupported") or item.get("failed"):
            continue
        agg = item.get("aggregate_tok_s")
        if isinstance(agg, (int, float)) and agg > best_agg:
            best_agg = float(agg)
            best_b = b
    return best_agg, best_b, True


def score_model(record: dict[str, Any], hardware_name: str) -> dict[str, Any]:
    model_name = resolve_model_name(record["model"])
    model = MODEL_SPECS[model_name]
    hw = resolve_hardware(hardware_name)
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else record

    submetrics: dict[str, dict[str, Any]] = {}

    b1_short_tok_s = metrics.get("b1_short_tok_s")
    if b1_short_tok_s is None:
        b1_short_tok_s = metrics.get("b1_nospec_tok_s")
    short_ctx = int(metrics.get("b1_short_ctx") or 256)
    rf_1a = compute_roofline(model, hw, batch=1, ctx=short_ctx)
    s_1a, raw_1a = _eff_from_tok_s(b1_short_tok_s, rf_1a.floor_tok_s)
    submetrics["p1a_short_b1"] = {
        "score": round(s_1a, 2),
        "raw_efficiency_pct": raw_1a,
        "weight": SUBMETRIC_WEIGHTS["p1a_short_b1"],
        "measured_tok_s": b1_short_tok_s,
        "ctx": short_ctx,
        "roofline": asdict(rf_1a),
        "bottleneck": rf_1a.binding_bottleneck,
    }

    default_long_ctx = min(model.context_tokens - 512, 65536) if model.context_tokens >= 65536 else max(1024, model.context_tokens - 512)
    long_ctx = int(metrics.get("b1_long_ctx") or default_long_ctx)
    b1_long_tok_s = metrics.get("b1_long_tok_s")
    rf_1b = compute_roofline(model, hw, batch=1, ctx=long_ctx)
    s_1b, raw_1b = _eff_from_tok_s(b1_long_tok_s, rf_1b.floor_tok_s)
    submetrics["p1b_long_b1"] = {
        "score": round(s_1b, 2),
        "raw_efficiency_pct": raw_1b,
        "weight": SUBMETRIC_WEIGHTS["p1b_long_b1"],
        "measured_tok_s": b1_long_tok_s,
        "ctx": long_ctx,
        "roofline": asdict(rf_1b),
        "bottleneck": rf_1b.binding_bottleneck,
    }

    itl_p50 = metrics.get("itl_p50_ms")
    itl_p99 = metrics.get("itl_p99_ms")
    if isinstance(itl_p50, (int, float)) and isinstance(itl_p99, (int, float)) and itl_p50 > 0.0 and itl_p99 > 0.0:
        s_1c = _clamp_score(100.0 * (itl_p50 / itl_p99))
    else:
        s_1c = 0.0
    submetrics["p1c_tail_jitter"] = {
        "score": round(s_1c, 2),
        "weight": SUBMETRIC_WEIGHTS["p1c_tail_jitter"],
        "itl_p50_ms": itl_p50,
        "itl_p99_ms": itl_p99,
        "bottleneck": "jitter",
    }

    spec = metrics.get("speculation") if isinstance(metrics.get("speculation"), dict) else {}
    alpha_domains: list[float] = []
    for domain in ("prose", "code", "edit"):
        val = spec.get(f"alpha_{domain}")
        if isinstance(val, (int, float)) and val >= 1.0:
            alpha_domains.append(float(val))
    if alpha_domains:
        alpha_mean = sum(alpha_domains) / len(alpha_domains)
    elif isinstance(spec.get("alpha"), (int, float)) and spec["alpha"] >= 1.0:
        alpha_mean = float(spec["alpha"])
    elif isinstance(metrics.get("spec_alpha"), (int, float)) and metrics["spec_alpha"] >= 1.0:
        alpha_mean = float(metrics["spec_alpha"])
    else:
        alpha_mean = 1.0
    s_2a = _clamp_score(100.0 * max(0.0, alpha_mean - 1.0) / (ALPHA_REF - 1.0))
    submetrics["p2a_spec_acceptance"] = {
        "score": round(s_2a, 2),
        "weight": SUBMETRIC_WEIGHTS["p2a_spec_acceptance"],
        "alpha_mean": round(alpha_mean, 3),
        "alpha_ref": ALPHA_REF,
        "alpha_prose": spec.get("alpha_prose"),
        "alpha_code": spec.get("alpha_code"),
        "alpha_edit": spec.get("alpha_edit"),
        "bottleneck": "speculation",
    }

    verify_v_b1 = spec.get("verify_v_b1", spec.get("verify_v"))
    if verify_v_b1 is None:
        b1_spec_tok_s = spec.get("b1_spec_tok_s", metrics.get("b1_spec_tok_s"))
        if (
            isinstance(b1_spec_tok_s, (int, float))
            and isinstance(b1_short_tok_s, (int, float))
            and b1_spec_tok_s > 0.0
            and b1_short_tok_s > 0.0
            and alpha_mean > 1.0
        ):
            speedup = b1_spec_tok_s / b1_short_tok_s
            verify_v_b1 = alpha_mean / speedup
    spec_active_b1 = alpha_mean > 1.001 or bool(spec.get("verify_b1_enabled"))
    if spec_active_b1 and isinstance(verify_v_b1, (int, float)) and verify_v_b1 > 0.0:
        inv_v_b1 = 1.0 / float(verify_v_b1)
        s_2b_b1 = _clamp_score(100.0 * inv_v_b1)
    else:
        inv_v_b1 = 0.0
        s_2b_b1 = 0.0
    verify_k = int(spec.get("verify_k") or 8)
    rf_2b_b1 = compute_roofline(model, hw, batch=1, ctx=short_ctx, rows_per_seq=verify_k + 1)
    submetrics["p2b_verify_b1"] = {
        "score": round(s_2b_b1, 2),
        "weight": SUBMETRIC_WEIGHTS["p2b_verify_b1"],
        "verify_v": round(float(verify_v_b1), 4) if isinstance(verify_v_b1, (int, float)) else None,
        "inv_v": round(inv_v_b1, 4),
        "verify_k": verify_k,
        "roofline": asdict(rf_2b_b1),
        "bottleneck": "speculation" if s_2b_b1 == 0.0 else rf_2b_b1.binding_bottleneck,
    }

    verify_v_b8 = spec.get("verify_v_b8")
    alpha_b8 = spec.get("alpha_b8", alpha_mean if spec.get("verify_b8_enabled") else 1.0)
    spec_active_b8 = bool(spec.get("verify_b8_enabled")) or (isinstance(verify_v_b8, (int, float)) and verify_v_b8 > 0.0)
    if spec_active_b8 and isinstance(verify_v_b8, (int, float)) and verify_v_b8 > 0.0:
        inv_v_b8 = 1.0 / float(verify_v_b8)
        s_2b_b8 = _clamp_score(100.0 * inv_v_b8)
    else:
        inv_v_b8 = 0.0
        s_2b_b8 = 0.0
    rf_2b_b8 = compute_roofline(model, hw, batch=8, ctx=short_ctx, rows_per_seq=verify_k + 1)
    submetrics["p2b_verify_b8"] = {
        "score": round(s_2b_b8, 2),
        "weight": SUBMETRIC_WEIGHTS["p2b_verify_b8"],
        "verify_v": round(float(verify_v_b8), 4) if isinstance(verify_v_b8, (int, float)) else None,
        "inv_v": round(inv_v_b8, 4),
        "alpha_b8": round(float(alpha_b8), 3) if isinstance(alpha_b8, (int, float)) else None,
        "roofline": asdict(rf_2b_b8),
        "bottleneck": "speculation" if s_2b_b8 == 0.0 else rf_2b_b8.binding_bottleneck,
    }

    batch_map = _extract_batch_map(metrics)
    p3a_details: dict[str, Any] = {}
    p3a_scores: list[float] = []
    p3a_bottleneck = "memory"
    for b in (4, 8, 16):
        rf_b = compute_roofline(model, hw, batch=b, ctx=short_ctx)
        eff_agg, source_b, capped = _best_working_aggregate(batch_map, b1_short_tok_s, b)
        eff_score, raw_eff = _eff_from_tok_s(eff_agg if eff_agg > 0.0 else None, rf_b.floor_tok_s)
        per_stream = eff_agg / float(b) if eff_agg > 0.0 else 0.0
        sla_target = (SLA_MIN_RATIO * float(b1_short_tok_s)) if (b1_short_tok_s and b1_short_tok_s > 0.0) else 0.0
        sla_met = (per_stream >= sla_target) if sla_target > 0.0 else True
        sla_factor = min(1.0, per_stream / sla_target) if (sla_target > 0.0 and not sla_met) else 1.0
        b_score = eff_score * sla_factor
        p3a_scores.append(b_score)
        p3a_details[f"B{b}"] = {
            "score": round(b_score, 2),
            "raw_efficiency_pct": raw_eff,
            "aggregate_tok_s": round(eff_agg, 2),
            "per_stream_tok_s": round(per_stream, 2),
            "sla_target_tok_s": round(sla_target, 2),
            "sla_met": sla_met,
            "flat_capped_from_b": source_b if capped else None,
            "roofline": asdict(rf_b),
        }
        if capped:
            p3a_bottleneck = "capacity_cap"
        elif b == 8:
            p3a_bottleneck = rf_b.binding_bottleneck
    s_3a = sum(p3a_scores) / len(p3a_scores)
    submetrics["p3a_interactive_batch"] = {
        "score": round(s_3a, 2),
        "weight": SUBMETRIC_WEIGHTS["p3a_interactive_batch"],
        "points": p3a_details,
        "bottleneck": p3a_bottleneck,
    }

    p3b_details: dict[str, Any] = {}
    p3b_scores: list[float] = []
    p3b_capped = False
    p3b_rf_bottleneck = "memory"
    for b in (32, 64):
        rf_b = compute_roofline(model, hw, batch=b, ctx=short_ctx)
        eff_agg, source_b, capped = _best_working_aggregate(batch_map, b1_short_tok_s, b)
        b_score, raw_eff = _eff_from_tok_s(eff_agg if eff_agg > 0.0 else None, rf_b.floor_tok_s)
        p3b_scores.append(b_score)
        if capped:
            p3b_capped = True
        p3b_rf_bottleneck = rf_b.binding_bottleneck
        p3b_details[f"B{b}"] = {
            "score": round(b_score, 2),
            "raw_efficiency_pct": raw_eff,
            "aggregate_tok_s": round(eff_agg, 2),
            "flat_capped_from_b": source_b if capped else None,
            "roofline": asdict(rf_b),
        }
    s_3b = sum(p3b_scores) / len(p3b_scores)
    submetrics["p3b_high_concurrency"] = {
        "score": round(s_3b, 2),
        "weight": SUBMETRIC_WEIGHTS["p3b_high_concurrency"],
        "points": p3b_details,
        "bottleneck": "capacity_cap" if p3b_capped else p3b_rf_bottleneck,
    }

    default_batch_long_ctx = min(model.context_tokens - 512, 32768) if model.context_tokens >= 32768 else max(1024, model.context_tokens - 512)
    b8_long_ctx = int(metrics.get("b8_long_ctx") or default_batch_long_ctx)
    b8_long_tok_s = metrics.get("b8_long_tok_s")
    rf_3c = compute_roofline(model, hw, batch=8, ctx=b8_long_ctx)
    s_3c, raw_3c = _eff_from_tok_s(b8_long_tok_s, rf_3c.floor_tok_s)
    submetrics["p3c_long_ctx_batch"] = {
        "score": round(s_3c, 2),
        "raw_efficiency_pct": raw_3c,
        "weight": SUBMETRIC_WEIGHTS["p3c_long_ctx_batch"],
        "measured_tok_s": b8_long_tok_s,
        "ctx": b8_long_ctx,
        "roofline": asdict(rf_3c),
        "bottleneck": rf_3c.binding_bottleneck,
    }

    prefill_8k_tokens = int(metrics.get("prefill_8k_tokens") or 8192)
    prefill_8k_tok_s = metrics.get("prefill_8k_tok_s")
    if prefill_8k_tok_s is None and isinstance(metrics.get("prefill_8k_ms"), (int, float)) and metrics["prefill_8k_ms"] > 0.0:
        prefill_8k_tok_s = prefill_8k_tokens * 1000.0 / float(metrics["prefill_8k_ms"])
    rf_4a = compute_roofline(model, hw, batch=1, ctx=prefill_8k_tokens, rows_per_seq=prefill_8k_tokens, is_prefill=True)
    s_4a, raw_4a = _eff_from_tok_s(prefill_8k_tok_s, rf_4a.floor_tok_s)
    submetrics["p4a_cold_prefill_8k"] = {
        "score": round(s_4a, 2),
        "raw_efficiency_pct": raw_4a,
        "weight": SUBMETRIC_WEIGHTS["p4a_cold_prefill_8k"],
        "measured_tok_s": prefill_8k_tok_s,
        "tokens": prefill_8k_tokens,
        "roofline": asdict(rf_4a),
        "bottleneck": rf_4a.binding_bottleneck,
    }

    p4b_parts: list[tuple[str, float, float, dict[str, Any]]] = []
    pf_32k_tokens = int(metrics.get("prefill_32k_tokens") or min(32768, model.context_tokens - 256))
    pf_32k_tok_s = metrics.get("prefill_32k_tok_s")
    if pf_32k_tok_s is None and isinstance(metrics.get("prefill_32k_ms"), (int, float)) and metrics["prefill_32k_ms"] > 0.0:
        pf_32k_tok_s = pf_32k_tokens * 1000.0 / float(metrics["prefill_32k_ms"])
    rf_4b_32k = compute_roofline(model, hw, batch=1, ctx=pf_32k_tokens, rows_per_seq=pf_32k_tokens, is_prefill=True)
    if isinstance(pf_32k_tok_s, (int, float)) and pf_32k_tok_s > 0.0:
        sc32, r32 = _eff_from_tok_s(pf_32k_tok_s, rf_4b_32k.floor_tok_s)
        p4b_parts.append(("32K", 0.4, sc32, {"tokens": pf_32k_tokens, "measured_tok_s": pf_32k_tok_s, "score": round(sc32, 2), "raw_efficiency_pct": r32, "roofline": asdict(rf_4b_32k)}))

    pf_128k_tokens = int(metrics.get("prefill_128k_tokens") or min(131072, model.context_tokens - 256))
    pf_128k_tok_s = metrics.get("prefill_128k_tok_s")
    if pf_128k_tok_s is None and isinstance(metrics.get("prefill_128k_ms"), (int, float)) and metrics["prefill_128k_ms"] > 0.0:
        pf_128k_tok_s = pf_128k_tokens * 1000.0 / float(metrics["prefill_128k_ms"])
    rf_4b_128k = compute_roofline(model, hw, batch=1, ctx=pf_128k_tokens, rows_per_seq=pf_128k_tokens, is_prefill=True)
    if isinstance(pf_128k_tok_s, (int, float)) and pf_128k_tok_s > 0.0:
        sc128, r128 = _eff_from_tok_s(pf_128k_tok_s, rf_4b_128k.floor_tok_s)
        p4b_parts.append(("128K", 0.6, sc128, {"tokens": pf_128k_tokens, "measured_tok_s": pf_128k_tok_s, "score": round(sc128, 2), "raw_efficiency_pct": r128, "roofline": asdict(rf_4b_128k)}))

    if p4b_parts:
        w_sum = sum(w for _, w, _, _ in p4b_parts)
        s_4b = sum(w * sc for _, w, sc, _ in p4b_parts) / w_sum
        rf_4b_main = rf_4b_128k if pf_128k_tok_s else rf_4b_32k
    else:
        s_4b = 0.0
        rf_4b_main = rf_4b_32k
    submetrics["p4b_long_cold_prefill"] = {
        "score": round(s_4b, 2),
        "weight": SUBMETRIC_WEIGHTS["p4b_long_cold_prefill"],
        "points": {label: detail for label, _, _, detail in p4b_parts},
        "roofline": asdict(rf_4b_main),
        "bottleneck": rf_4b_main.binding_bottleneck,
    }

    p4c_parts: list[tuple[str, float, float, dict[str, Any]]] = []
    warm_8k_prefix = int(metrics.get("warm_8k_prefix_tokens") or 8192)
    warm_8k_new = int(metrics.get("warm_8k_new_tokens") or 128)
    warm_8k_ttft_ms = metrics.get("warm_8k_ttft_ms")
    rf_4c_8k = compute_roofline(model, hw, batch=1, ctx=warm_8k_prefix + warm_8k_new, rows_per_seq=warm_8k_new, is_prefill=True, prefix_ctx=warm_8k_prefix)
    if isinstance(warm_8k_ttft_ms, (int, float)) and warm_8k_ttft_ms > 0.0:
        sc_w8, r_w8 = _eff_from_ms(warm_8k_ttft_ms, rf_4c_8k.t_floor_ms)
        p4c_parts.append(("8K+128", 0.4, sc_w8, {"prefix_tokens": warm_8k_prefix, "new_tokens": warm_8k_new, "ttft_ms": warm_8k_ttft_ms, "score": round(sc_w8, 2), "raw_efficiency_pct": r_w8, "roofline": asdict(rf_4c_8k)}))

    warm_100k_prefix = int(metrics.get("warm_100k_prefix_tokens") or min(100000, model.context_tokens - 512))
    warm_100k_new = int(metrics.get("warm_100k_new_tokens") or 128)
    warm_100k_ttft_ms = metrics.get("warm_100k_ttft_ms")
    rf_4c_100k = compute_roofline(model, hw, batch=1, ctx=warm_100k_prefix + warm_100k_new, rows_per_seq=warm_100k_new, is_prefill=True, prefix_ctx=warm_100k_prefix)
    if isinstance(warm_100k_ttft_ms, (int, float)) and warm_100k_ttft_ms > 0.0:
        sc_w100, r_w100 = _eff_from_ms(warm_100k_ttft_ms, rf_4c_100k.t_floor_ms)
        p4c_parts.append(("100K+128", 0.6, sc_w100, {"prefix_tokens": warm_100k_prefix, "new_tokens": warm_100k_new, "ttft_ms": warm_100k_ttft_ms, "score": round(sc_w100, 2), "raw_efficiency_pct": r_w100, "roofline": asdict(rf_4c_100k)}))

    if p4c_parts:
        w_sum = sum(w for _, w, _, _ in p4c_parts)
        s_4c = sum(w * sc for _, w, sc, _ in p4c_parts) / w_sum
        rf_4c_main = rf_4c_100k if warm_100k_ttft_ms else rf_4c_8k
    else:
        s_4c = 0.0
        rf_4c_main = rf_4c_8k
    submetrics["p4c_multiturn_prefix_ttft"] = {
        "score": round(s_4c, 2),
        "weight": SUBMETRIC_WEIGHTS["p4c_multiturn_prefix_ttft"],
        "points": {label: detail for label, _, _, detail in p4c_parts},
        "roofline": asdict(rf_4c_main),
        "bottleneck": rf_4c_main.binding_bottleneck,
    }

    weight_bytes_per_rank = model.total_weight_bytes / float(max(1, hw.ranks))
    t_boot_floor_s = (
        max(1.0, weight_bytes_per_rank / (hw.nvme_bw_gb_s_per_rank * 1e9))
        if hw.nvme_bw_gb_s_per_rank > 0.0
        else 1.0
    )
    cold_boot_s = metrics.get("cold_boot_s")
    if cold_boot_s is None:
        cold_boot_s = metrics.get("swap_s")
    if cold_boot_s is None:
        cold_boot_s = record.get("cold_boot_s", record.get("swap_s"))
    if isinstance(cold_boot_s, (int, float)) and cold_boot_s > 0.0:
        raw_5a = 100.0 * t_boot_floor_s / float(cold_boot_s)
        s_5a = _clamp_score(raw_5a)
    else:
        raw_5a = 0.0
        s_5a = 0.0
    submetrics["p5a_cold_boot"] = {
        "score": round(s_5a, 2),
        "raw_efficiency_pct": round(raw_5a, 2),
        "weight": SUBMETRIC_WEIGHTS["p5a_cold_boot"],
        "cold_boot_s": round(float(cold_boot_s), 3) if isinstance(cold_boot_s, (int, float)) else None,
        "t_boot_floor_s": round(t_boot_floor_s, 3),
        "weight_gb_per_rank": round(weight_bytes_per_rank / 1e9, 2),
        "nvme_bw_gb_s_per_rank": hw.nvme_bw_gb_s_per_rank,
        "bottleneck": "storage",
    }

    weights_gib_per_rank = weight_bytes_per_rank / GIB_BYTES
    m_free_floor_gib = max(1.0, hw.ram_gib_per_rank - weights_gib_per_rank - hw.os_reserve_gib_per_rank)
    usable_kv_gib = metrics.get("usable_kv_gib_per_rank")
    if usable_kv_gib is None:
        usable_kv_gib = metrics.get("usable_kv_gib")
    if usable_kv_gib is None and isinstance(metrics.get("kv_cache_bytes_per_rank"), (int, float)):
        usable_kv_gib = float(metrics["kv_cache_bytes_per_rank"]) / GIB_BYTES
    if usable_kv_gib is None:
        usable_kv_gib = record.get("usable_kv_gib_per_rank")
    if isinstance(usable_kv_gib, (int, float)) and usable_kv_gib > 0.0:
        raw_5b = 100.0 * float(usable_kv_gib) / m_free_floor_gib
        s_5b = _clamp_score(raw_5b)
    else:
        raw_5b = 0.0
        s_5b = 0.0
    submetrics["p5b_memory_efficiency"] = {
        "score": round(s_5b, 2),
        "raw_efficiency_pct": round(raw_5b, 2),
        "weight": SUBMETRIC_WEIGHTS["p5b_memory_efficiency"],
        "usable_kv_gib_per_rank": round(float(usable_kv_gib), 3) if isinstance(usable_kv_gib, (int, float)) else None,
        "m_free_floor_gib": round(m_free_floor_gib, 3),
        "weights_gib_per_rank": round(weights_gib_per_rank, 3),
        "ram_gib_per_rank": hw.ram_gib_per_rank,
        "os_reserve_gib_per_rank": hw.os_reserve_gib_per_rank,
        "bottleneck": "memory_footprint",
    }

    batch_invariant_exact = metrics.get("batch_invariant_exact")
    if batch_invariant_exact is None and isinstance(record.get("gates"), dict):
        batch_invariant_exact = record["gates"].get("batch_invariant_exact")
    if batch_invariant_exact is None:
        batch_invariant_exact = record.get("batch_invariant_exact")
    s_5c = 100.0 if batch_invariant_exact is True else 0.0
    submetrics["p5c_batch_invariance"] = {
        "score": round(s_5c, 2),
        "weight": SUBMETRIC_WEIGHTS["p5c_batch_invariance"],
        "batch_invariant_exact": batch_invariant_exact if isinstance(batch_invariant_exact, bool) else None,
        "bottleneck": "determinism",
    }

    pillar_scores: dict[str, float] = {}
    for pillar in PILLAR_ORDER:
        keys = [k for k, p, _, _ in SUBMETRIC_SPECS if p == pillar]
        p_weight = sum(SUBMETRIC_WEIGHTS[k] for k in keys)
        p_sum = sum(SUBMETRIC_WEIGHTS[k] * submetrics[k]["score"] for k in keys)
        pillar_scores[pillar] = round(p_sum / p_weight, 2)

    raw_rais = round(sum(SUBMETRIC_WEIGHTS[k] * submetrics[k]["score"] for k in SUBMETRIC_WEIGHTS), 2)
    base_roofline_r = round((pillar_scores["P1"] + pillar_scores["P3"] + pillar_scores["P4"]) / 3.0, 2)

    gate_info = evaluate_gates(record)
    certified_rais = None if gate_info["status"] == "INVALID" else raw_rais

    raw_badge = record.get("stack_badge")
    stack_badge = raw_badge.strip() if isinstance(raw_badge, str) and raw_badge.strip() else None

    roi_items = []
    for key, pillar, label, weight in SUBMETRIC_SPECS:
        sc = submetrics[key]["score"]
        headroom = round(weight * (100.0 - sc), 3)
        roi_items.append({
            "model": model_name,
            "submetric": key,
            "pillar": pillar,
            "label": label,
            "weight_pct": round(weight * 100.0, 1),
            "current_score": sc,
            "headroom_rais_pts": headroom,
            "bottleneck": submetrics[key].get("bottleneck", "unknown"),
        })
    roi_items.sort(key=lambda item: item["headroom_rais_pts"], reverse=True)

    return {
        "rais_version": RAIS_VERSION,
        "model": model_name,
        "engine": record.get("engine", "unknown"),
        "stack_badge": stack_badge,
        "tier_run": record.get("tier", "custom"),
        "hardware": hw.name,
        "model_tier": model.tier,
        "fleet_weight": model.fleet_weight,
        "canonical_model": model.canonical_model or model.name,
        "dtype": model.dtype,
        "active_params_b": round(model.active_params_b, 2),
        "gates": gate_info,
        "status": gate_info["status"],
        "certified_rais": certified_rais,
        "raw_rais": raw_rais,
        "base_roofline_r": base_roofline_r,
        "speculation_summary": {
            "alpha": round(alpha_mean, 3),
            "inv_v_b1": round(inv_v_b1, 4),
            "inv_v_b8": round(inv_v_b8, 4),
            "p2_score": pillar_scores["P2"],
        },
        "pillars": pillar_scores,
        "submetrics": submetrics,
        "roi_table": roi_items,
    }


def default_fleet_weights() -> dict[str, float]:
    return {name: spec.fleet_weight for name, spec in MODEL_SPECS.items() if spec.fleet_weight > 0.0}


def load_weights_file(path: Path) -> tuple[dict[str, float], str, float | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "weights" in raw:
        weights_raw = raw["weights"]
        basket_id = str(raw.get("basket_id") or path.stem)
        divisor = float(raw["divisor"]) if raw.get("divisor") is not None else None
    elif isinstance(raw, dict):
        weights_raw = raw
        basket_id = path.stem
        divisor = None
    else:
        raise ValueError(f"invalid weights file: {path}")
    weights: dict[str, float] = {}
    for k, v in weights_raw.items():
        canon = resolve_model_name(str(k))
        fv = float(v)
        if fv < 0.0:
            raise ValueError(f"negative weight for {canon}: {fv}")
        if fv > 0.0:
            weights[canon] = fv
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"basket weights must sum to 1.0 (got {total:.6f})")
    return weights, basket_id, divisor


def score_fleet(
    records: list[dict[str, Any]],
    hardware_name: str,
    weights_override: dict[str, float] | None = None,
    divisor: float = DEFAULT_DIVISOR,
    basket_id: str = DEFAULT_BASKET_ID,
    stack_badge_override: str | None = None,
) -> dict[str, Any]:
    if divisor <= 0.0:
        raise ValueError(f"index divisor must be positive (got {divisor})")
    hw = resolve_hardware(hardware_name)
    badge_override = (
        stack_badge_override.strip()
        if isinstance(stack_badge_override, str) and stack_badge_override.strip()
        else None
    )
    per_model = []
    for rec in records:
        scored = score_model(rec, hw.name)
        if badge_override is not None:
            scored["stack_badge"] = badge_override
        per_model.append(scored)
    by_canonical: dict[str, dict[str, Any]] = {}
    for item in per_model:
        canon = item["canonical_model"]
        if canon not in by_canonical or item["model"] == canon:
            by_canonical[canon] = item

    weights = weights_override if weights_override is not None else default_fleet_weights()
    covered_weight = 0.0
    weighted_sum = 0.0
    strict_sum = 0.0
    any_invalid = False
    fleet_roi: list[dict[str, Any]] = []

    for canon, item in by_canonical.items():
        w = float(weights.get(canon, 0.0))
        if w <= 0.0:
            continue
        covered_weight += w
        if item["status"] == "INVALID":
            any_invalid = True
        score_val = 0.0 if item["certified_rais"] is None else item["certified_rais"]
        weighted_sum += w * score_val
        strict_sum += w * score_val
        for roi in item["roi_table"]:
            fleet_roi.append({
                "model": item["model"],
                "submetric": roi["submetric"],
                "pillar": roi["pillar"],
                "label": roi["label"],
                "model_weight_pct": round(w * 100.0, 2),
                "current_score": roi["current_score"],
                "model_headroom_pts": roi["headroom_rais_pts"],
                "fleet_headroom_pts": round(w * roi["headroom_rais_pts"], 4),
                "bottleneck": roi["bottleneck"],
            })

    fleet_roi.sort(key=lambda entry: entry["fleet_headroom_pts"], reverse=True)
    raw_covered_exact = (weighted_sum / covered_weight) if covered_weight > 0.0 else 0.0
    fleet_rais_covered = round(raw_covered_exact, 2)
    fleet_rais_strict = round(strict_sum, 2)
    fleet_rais_raw_index = round(raw_covered_exact / divisor, 2)
    unique_badges = list(dict.fromkeys(m["stack_badge"] for m in per_model if m.get("stack_badge")))

    return {
        "rais_version": RAIS_VERSION,
        "basket_id": basket_id,
        "index_divisor": round(divisor, 8),
        "hardware": asdict(hw),
        "stack_badge": unique_badges[0] if len(unique_badges) == 1 else (", ".join(unique_badges) if unique_badges else None),
        "model_count": len(per_model),
        "fleet_coverage_pct": round(covered_weight * 100.0, 2),
        "fleet_status": "INVALID" if any_invalid else ("VALID" if all(m["status"] == "VALID" for m in per_model) and per_model else "UNVERIFIED"),
        "fleet_rais_covered": None if any_invalid else fleet_rais_covered,
        "fleet_rais_strict": None if any_invalid else fleet_rais_strict,
        "fleet_rais_index": None if any_invalid else fleet_rais_raw_index,
        "fleet_rais_raw_covered": fleet_rais_covered,
        "fleet_rais_raw_index": fleet_rais_raw_index,
        "models": per_model,
        "fleet_roi_top": fleet_roi[:20],
    }


def reconstitute_basket(
    overlap_records: list[dict[str, Any]],
    hardware_name: str,
    new_weights: dict[str, float],
    prev_weights: dict[str, float] | None = None,
    prev_divisor: float = DEFAULT_DIVISOR,
    prev_basket_id: str = DEFAULT_BASKET_ID,
    new_basket_id: str = "reconstituted",
    allow_unverified: bool = False,
) -> dict[str, Any]:
    if prev_divisor <= 0.0:
        raise ValueError(f"prev_divisor must be positive (got {prev_divisor})")
    old_w = prev_weights if prev_weights is not None else default_fleet_weights()
    for label, wmap in (("prev_weights", old_w), ("new_weights", new_weights)):
        total = sum(wmap.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(f"{label} must sum to 1.0 (got {total:.6f})")

    hw = resolve_hardware(hardware_name)
    scored = [score_model(rec, hw.name) for rec in overlap_records]
    by_canon: dict[str, dict[str, Any]] = {}
    for item in scored:
        canon = item["canonical_model"]
        if canon not in by_canon or item["model"] == canon:
            by_canon[canon] = item

    required = sorted({k for k, v in old_w.items() if v > 0.0} | {k for k, v in new_weights.items() if v > 0.0})
    missing = [m for m in required if m not in by_canon]
    if missing:
        raise ValueError(f"overlap receipt missing required basket models: {', '.join(missing)}")

    invalid = [m for m in required if by_canon[m]["status"] == "INVALID"]
    if invalid:
        raise ValueError(f"cannot reconstitute across INVALID models: {', '.join(invalid)}")
    if not allow_unverified:
        unverified = [m for m in required if by_canon[m]["status"] != "VALID"]
        if unverified:
            raise ValueError(f"overlap receipt contains unverified models: {', '.join(unverified)}")

    old_mean = sum(old_w[m] * by_canon[m]["raw_rais"] for m in old_w if old_w[m] > 0.0)
    new_mean = sum(new_weights[m] * by_canon[m]["raw_rais"] for m in new_weights if new_weights[m] > 0.0)
    if old_mean <= 0.0 or new_mean <= 0.0:
        raise ValueError(f"overlap basket means must be positive (old={old_mean:.4f}, new={new_mean:.4f})")

    new_divisor = prev_divisor * (new_mean / old_mean)
    index_before = old_mean / prev_divisor
    index_after = new_mean / new_divisor

    old_set = {k for k, v in old_w.items() if v > 0.0}
    new_set = {k for k, v in new_weights.items() if v > 0.0}
    retired = sorted(old_set - new_set)
    added = sorted(new_set - old_set)
    reweighted = {
        m: {"old": round(old_w[m], 6), "new": round(new_weights[m], 6)}
        for m in sorted(old_set & new_set)
        if not math.isclose(old_w[m], new_weights[m], abs_tol=1e-9)
    }

    return {
        "rais_version": RAIS_VERSION,
        "hardware": hw.name,
        "prev_basket_id": prev_basket_id,
        "new_basket_id": new_basket_id,
        "prev_divisor": round(prev_divisor, 8),
        "new_divisor": round(new_divisor, 8),
        "raw_mean_before": round(old_mean, 4),
        "raw_mean_after": round(new_mean, 4),
        "index_before": round(index_before, 4),
        "index_after": round(index_after, 4),
        "continuity_gap": round(abs(index_after - index_before), 8),
        "retired_models": retired,
        "added_models": added,
        "reweighted_models": reweighted,
        "weights": {k: round(v, 6) for k, v in sorted(new_weights.items()) if v > 0.0},
    }


def compare_reports(report_a: dict[str, Any], report_b: dict[str, Any], label_a: str, label_b: str) -> dict[str, Any]:
    map_a = {m["model"]: m for m in report_a["models"]}
    map_b = {m["model"]: m for m in report_b["models"]}
    common = sorted(set(map_a) & set(map_b))
    rows = []
    for model_name in common:
        ma = map_a[model_name]
        mb = map_b[model_name]
        rows.append({
            "model": model_name,
            "status_a": ma["status"],
            "status_b": mb["status"],
            "stack_badge_a": ma.get("stack_badge"),
            "stack_badge_b": mb.get("stack_badge"),
            "rais_a": ma["raw_rais"],
            "rais_b": mb["raw_rais"],
            "delta_rais": round(mb["raw_rais"] - ma["raw_rais"], 2),
            "r_a": ma["base_roofline_r"],
            "r_b": mb["base_roofline_r"],
            "delta_r": round(mb["base_roofline_r"] - ma["base_roofline_r"], 2),
            "pillars_delta": {
                p: round(mb["pillars"][p] - ma["pillars"][p], 2) for p in PILLAR_ORDER
            },
            "submetrics_delta": {
                k: round(mb["submetrics"][k]["score"] - ma["submetrics"][k]["score"], 2) for k in SUBMETRIC_WEIGHTS
            },
        })
    return {
        "rais_version": RAIS_VERSION,
        "hardware": report_a["hardware"]["name"],
        "label_a": label_a,
        "label_b": label_b,
        "stack_badge_a": report_a.get("stack_badge"),
        "stack_badge_b": report_b.get("stack_badge"),
        "fleet_rais_a": report_a["fleet_rais_raw_covered"],
        "fleet_rais_b": report_b["fleet_rais_raw_covered"],
        "delta_fleet_rais": round(report_b["fleet_rais_raw_covered"] - report_a["fleet_rais_raw_covered"], 2),
        "models": rows,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def _infer_model_from_filename(path: Path) -> str | None:
    stem = path.name.split(".")[0].lower()
    if stem in MODEL_SPECS or stem in MODEL_ALIASES:
        return resolve_model_name(stem)
    for candidate in sorted(MODEL_SPECS, key=len, reverse=True):
        if candidate in path.name.lower():
            return candidate
    return None


def import_hillclimb_records(path: Path, default_model: str | None = None, engine: str = "sparkpipe") -> list[dict[str, Any]]:
    raw = load_jsonl(path)
    if not raw:
        return []
    grouped: dict[str, dict[str, Any]] = {}
    inferred = default_model or _infer_model_from_filename(path)

    for entry in raw:
        if "metrics" in entry and "model" in entry:
            grouped[resolve_model_name(entry["model"])] = entry
            continue
        model_raw = entry.get("model") or entry.get("label") or inferred
        try:
            model_name = resolve_model_name(str(model_raw))
        except KeyError:
            if inferred:
                model_name = resolve_model_name(inferred)
            else:
                raise ValueError(f"cannot determine model name for record in {path}; pass --model")

        rec = grouped.setdefault(
            model_name,
            {
                "rais_version": RAIS_VERSION,
                "model": model_name,
                "engine": entry.get("engine", engine),
                "tier": "imported",
                "metrics": {"batch": {}, "speculation": {}},
            },
        )
        m = rec["metrics"]

        if "case" in entry:
            case = str(entry["case"]).lower()
            if case.startswith("b1_o") or (case == "b1" and entry.get("speculation") is False):
                rate = entry.get("median_decode_tok_s") or entry.get("aggregate_tok_s")
                prompt_tok = 256
                if entry.get("per_stream") and isinstance(entry["per_stream"], list):
                    prompt_tok = int(entry["per_stream"][0].get("prompt_tokens") or 256)
                elif entry.get("repeats") and isinstance(entry["repeats"], list) and entry["repeats"][0].get("usage"):
                    prompt_tok = int(entry["repeats"][0]["usage"].get("prompt_tokens") or 256)
                if rate and prompt_tok >= 16384:
                    m["b1_long_tok_s"] = float(rate)
                    m["b1_long_ctx"] = prompt_tok
                elif rate:
                    m["b1_short_tok_s"] = float(rate)
                    m["b1_short_ctx"] = max(256, prompt_tok)
                if entry.get("ttft_s_max") and prompt_tok >= 4096:
                    ttft_ms = float(entry["ttft_s_max"]) * 1000.0
                    if prompt_tok >= 65536:
                        m["warm_100k_prefix_tokens"] = prompt_tok
                        m["warm_100k_new_tokens"] = 128
                        m["warm_100k_ttft_ms"] = ttft_ms
                    else:
                        m["warm_8k_prefix_tokens"] = prompt_tok
                        m["warm_8k_new_tokens"] = 128
                        m["warm_8k_ttft_ms"] = ttft_ms
            elif case == "b1" and entry.get("speculation") is True:
                rate = entry.get("aggregate_tok_s") or entry.get("stream_tok_s_median")
                if rate:
                    m["speculation"]["b1_spec_tok_s"] = float(rate)
            elif case.startswith("ttft"):
                pf = entry.get("prefill_tok_s")
                pt = int(entry.get("prompt_tokens") or 8192)
                if pf:
                    if pt >= 24000:
                        m["prefill_32k_tokens"] = pt
                        m["prefill_32k_tok_s"] = float(pf)
                    else:
                        m["prefill_8k_tokens"] = pt
                        m["prefill_8k_tok_s"] = float(pf)
            elif case == "warm" and entry.get("prompt_tokens") and entry.get("seconds"):
                pt = int(entry["prompt_tokens"])
                pf = pt / float(entry["seconds"])
                if pt >= 65536:
                    m["prefill_128k_tokens"] = pt
                    m["prefill_128k_tok_s"] = round(pf, 2)
                else:
                    m["prefill_32k_tokens"] = pt
                    m["prefill_32k_tok_s"] = round(pf, 2)
            elif case.startswith("streams") or (case.startswith("b") and case[1:].isdigit()):
                if case.startswith("streams"):
                    b_val = int(case[7:].split("_")[0].split("x")[0])
                else:
                    b_val = int(case[1:])
                agg = entry.get("aggregate_all_streams_decoding_tok_s") or entry.get("aggregate_tok_s")
                prompt_tok = 256
                if entry.get("per_stream") and isinstance(entry["per_stream"], list):
                    first = entry["per_stream"][0]
                    prompt_tok = int(first.get("prompt_tokens") or (first.get("usage") or {}).get("prompt_tokens") or 256)
                if agg and prompt_tok >= 16384 and b_val == 8:
                    m["b8_long_ctx"] = prompt_tok
                    m["b8_long_tok_s"] = float(agg)
                elif agg and b_val >= 2:
                    m["batch"][str(b_val)] = {"aggregate_tok_s": float(agg)}
        else:
            for src, dst in (
                ("b1_short_tok_s", "b1_short_tok_s"),
                ("b1_nospec_tok_s", "b1_short_tok_s"),
                ("b1_tok_s", "b1_short_tok_s"),
                ("b1_long_tok_s", "b1_long_tok_s"),
                ("b1_long_ctx", "b1_long_ctx"),
                ("itl_p50_ms", "itl_p50_ms"),
                ("itl_p99_ms", "itl_p99_ms"),
                ("b8_long_tok_s", "b8_long_tok_s"),
                ("b8_long_ctx", "b8_long_ctx"),
                ("prefill_8k_tok_s", "prefill_8k_tok_s"),
                ("prefill_8k_tokens", "prefill_8k_tokens"),
                ("prefill_32k_tok_s", "prefill_32k_tok_s"),
                ("prefill_32k_tokens", "prefill_32k_tokens"),
                ("prefill_128k_tok_s", "prefill_128k_tok_s"),
                ("prefill_128k_tokens", "prefill_128k_tokens"),
                ("warm_8k_ttft_ms", "warm_8k_ttft_ms"),
                ("warm_100k_ttft_ms", "warm_100k_ttft_ms"),
                ("cold_boot_s", "cold_boot_s"),
                ("swap_s", "swap_s"),
                ("usable_kv_gib_per_rank", "usable_kv_gib_per_rank"),
                ("usable_kv_gib", "usable_kv_gib_per_rank"),
                ("kv_cache_bytes_per_rank", "kv_cache_bytes_per_rank"),
                ("batch_invariant_exact", "batch_invariant_exact"),
            ):
                if entry.get(src) is not None:
                    m[dst] = entry[src]
            for b in (4, 8, 16, 32, 64):
                if entry.get(f"b{b}_tok_s") is not None:
                    m["batch"][str(b)] = {"aggregate_tok_s": float(entry[f"b{b}_tok_s"])}
            for s_key in ("alpha", "alpha_prose", "alpha_code", "alpha_edit", "verify_v_b1", "verify_v_b8", "verify_k", "b1_spec_tok_s", "verify_b1_enabled", "verify_b8_enabled"):
                if entry.get(s_key) is not None:
                    m["speculation"][s_key] = entry[s_key]
            if isinstance(entry.get("stack_badge"), str) and entry["stack_badge"].strip():
                rec["stack_badge"] = entry["stack_badge"].strip()
            if isinstance(entry.get("gates"), dict):
                rec["gates"] = entry["gates"]

    return list(grouped.values())


def sanitize_receipt_label(
    raw: str | Path | None,
    hardware: str | None = None,
    release: str | None = None,
) -> str:
    hw_suffix = f" ({hardware})" if hardware else ""
    if release:
        return f"release {release}{hw_suffix}"
    if raw is None:
        return f"live run{hw_suffix}".strip()
    text = str(raw).strip()
    if not text:
        return f"live run{hw_suffix}".strip()
    rel_match = re.search(r"(?:^|[\\/])releases[\\/]([0-9a-fA-F]{7,40})(?:[\\/]|$)", text)
    if rel_match:
        return f"release {rel_match.group(1)}{hw_suffix}"
    forbidden = ("qualification/", "tools/", "docs/", "/srv/", "/home/", "/Users/", "\\")
    if any(tok in text for tok in forbidden) or "/" in text:
        stem = Path(text).name
        for ext in (".jsonl", ".json", ".txt", ".log"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
        stem = re.sub(r"^rais-fleet-", "", stem)
        if hardware and hardware not in stem:
            return f"{stem}{hw_suffix}"
        return stem or f"live run{hw_suffix}".strip()
    return text


def render_markdown(report: dict[str, Any]) -> str:
    hw = report["hardware"]
    stack_line = f" | **Stack:** `{report['stack_badge']}`" if report.get("stack_badge") else ""
    lines = [
        f"# RAIS v{report['rais_version']} Scorecard — `{hw['name']}`",
        "",
        f"- **Hardware:** {hw['description']} (`{hw['ranks']}x`, {hw['mem_bw_gb_s_per_rank']:.0f} GB/s/rank, {hw['fp8_tflops_per_rank']:.0f} FP8 / {hw['bf16_tflops_per_rank']:.0f} BF16 TFLOPS/rank, {hw['net_bw_gb_s_per_rank']:.1f} GB/s/rank net @ {hw['net_hop_us']:.0f} us, {hw['nvme_bw_gb_s_per_rank']:.1f} GB/s NVMe, {hw['ram_gib_per_rank']:.0f} GiB RAM / {hw['os_reserve_gib_per_rank']:.0f} GiB OS reserve){stack_line}",
        f"- **Fleet Status:** `{report['fleet_status']}` | **Basket:** `{report['basket_id']}` (divisor `{report['index_divisor']:.6f}`) | **Coverage:** {report['fleet_coverage_pct']:.1f}% ({report['model_count']} models)",
        f"- **Fleet Aggregate RAIS (covered):** `{report['fleet_rais_covered'] if report['fleet_rais_covered'] is not None else 'INVALID'}` (raw: {report['fleet_rais_raw_covered']:.2f}) | **RAIS Index:** `{report['fleet_rais_index'] if report['fleet_rais_index'] is not None else 'INVALID'}` | **Strict Fleet RAIS:** `{report['fleet_rais_strict'] if report['fleet_rais_strict'] is not None else 'INVALID'}`",
        "",
        "## Model Summary",
        "",
        "| Model | Engine | Stack | Dtype | Status | RAIS | Base R | P1 Dec | P2 Spec (a / 1/V) | P3 Batch | P4 Prefill | P5 Sys |",
        "| --- | --- | --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in report["models"]:
        spec = m["speculation_summary"]
        rais_str = f"**{m['certified_rais']:.1f}**" if m["certified_rais"] is not None else f"INVALID ({m['raw_rais']:.1f})"
        badge_str = f"`{m['stack_badge']}`" if m.get("stack_badge") else "-"
        lines.append(
            f"| `{m['model']}` | {m['engine']} | {badge_str} | {m['dtype']} | `{m['status']}` | {rais_str} | {m['base_roofline_r']:.1f} | "
            f"{m['pillars']['P1']:.1f} | {m['pillars']['P2']:.1f} ({spec['alpha']:.2f} / {spec['inv_v_b1']:.2f}) | "
            f"{m['pillars']['P3']:.1f} | {m['pillars']['P4']:.1f} | {m['pillars']['P5']:.1f} |"
        )

    for m in report["models"]:
        lines.extend([
            "",
            f"### `{m['model']}` Sub-Metric & 3-Roofline Breakdown",
            "",
            "| Sub-Metric | Wt | Score | T_c (ms) | T_m (ms) | T_n (ms) | T_floor (ms) | Overlap Gain | Binding |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for key, _, label, weight in SUBMETRIC_SPECS:
            sm = m["submetrics"][key]
            rf = sm.get("roofline")
            if rf is None and isinstance(sm.get("points"), dict) and sm["points"]:
                first_pt = next(iter(sm["points"].values()))
                rf = first_pt.get("roofline")
            if rf:
                lines.append(
                    f"| {label} | {weight * 100:.1f}% | {sm['score']:.1f} | {rf['t_compute_ms']:.2f} | "
                    f"{rf['t_memory_ms']:.2f} | {rf['t_network_ms']:.2f} | {rf['t_floor_ms']:.2f} | "
                    f"{rf['overlap_gain_max']:.2f}x | `{sm['bottleneck']}` |"
                )
            else:
                lines.append(
                    f"| {label} | {weight * 100:.1f}% | {sm['score']:.1f} | - | - | - | - | - | `{sm['bottleneck']}` |"
                )
        if m["gates"]["failed_gates"]:
            lines.append("")
            lines.append(f"- **FAILED GATES:** {', '.join(m['gates']['failed_gates'])}")

    lines.extend([
        "",
        "## ROI Priority Table (Highest Remaining dRAIS Headroom)",
        "",
        "| Rank | Model | Sub-Metric | Current | Model dRAIS Max | Fleet dRAIS Max | Binding Bottleneck |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for idx, item in enumerate(report["fleet_roi_top"][:15], 1):
        lines.append(
            f"| {idx} | `{item['model']}` | {item['label']} | {item['current_score']:.1f} | "
            f"+{item['model_headroom_pts']:.2f} | +{item['fleet_headroom_pts']:.3f} | `{item['bottleneck']}` |"
        )
    return "\n".join(lines)


def render_compare_markdown(cmp_report: dict[str, Any]) -> str:
    la = cmp_report["label_a"]
    lb = cmp_report["label_b"]
    ba = cmp_report.get("stack_badge_a")
    bb = cmp_report.get("stack_badge_b")
    stack_note = ""
    if ba or bb:
        stack_note = f" | **Stacks:** `{ba or '-'}` vs `{bb or '-'}`"
    lines = [
        f"# RAIS v{cmp_report['rais_version']} Comparison — `{la}` vs `{lb}` on `{cmp_report['hardware']}`",
        "",
        f"- **Fleet Aggregate RAIS:** `{la}` = **{cmp_report['fleet_rais_a']:.2f}** vs `{lb}` = **{cmp_report['fleet_rais_b']:.2f}** (Δ = **{cmp_report['delta_fleet_rais']:+.2f}**){stack_note}",
        "",
        f"| Model | {la} RAIS | {lb} RAIS | dRAIS | dR | dP1 | dP2 | dP3 | dP4 | dP5 | Status ({la}/{lb}) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in cmp_report["models"]:
        pd = row["pillars_delta"]
        lines.append(
            f"| `{row['model']}` | {row['rais_a']:.1f} | {row['rais_b']:.1f} | {row['delta_rais']:+.1f} | "
            f"{row['delta_r']:+.1f} | {pd['P1']:+.1f} | {pd['P2']:+.1f} | {pd['P3']:+.1f} | {pd['P4']:+.1f} | {pd['P5']:+.1f} | "
            f"`{row['status_a']}` / `{row['status_b']}` |"
        )
    return "\n".join(lines)


def render_reconstitute_markdown(rec: dict[str, Any]) -> str:
    lines = [
        f"# RAIS v{rec['rais_version']} Index Reconstitution — `{rec['prev_basket_id']}` → `{rec['new_basket_id']}`",
        "",
        f"- **Hardware:** `{rec['hardware']}`",
        f"- **Divisor:** `{rec['prev_divisor']:.8f}` → **`{rec['new_divisor']:.8f}`**",
        f"- **Raw Basket Mean:** `{rec['raw_mean_before']:.2f}` → `{rec['raw_mean_after']:.2f}`",
        f"- **Continuous RAIS Index:** `{rec['index_before']:.2f}` → **`{rec['index_after']:.2f}`** (gap `{rec['continuity_gap']:.8f}`)",
        f"- **Retired:** {', '.join(f'`{m}`' for m in rec['retired_models']) or 'none'}",
        f"- **Added:** {', '.join(f'`{m}`' for m in rec['added_models']) or 'none'}",
    ]
    if rec["reweighted_models"]:
        lines.extend(["", "| Reweighted Model | Old Wt | New Wt |", "| --- | ---: | ---: |"])
        for model_name, pair in rec["reweighted_models"].items():
            lines.append(f"| `{model_name}` | {pair['old'] * 100:.2f}% | {pair['new'] * 100:.2f}% |")
    return "\n".join(lines)


def render_html_data(report: dict[str, Any], receipt_label: str | Path | None = None) -> str:
    hw_name = report["hardware"]["name"]
    clean_receipt = sanitize_receipt_label(receipt_label, hardware=hw_name) if receipt_label is not None else None
    fleet_obj: dict[str, Any] = {
        "status": report["fleet_status"],
        "coverage_pct": report["fleet_coverage_pct"],
        "rais_covered": report["fleet_rais_covered"],
        "rais_strict": report["fleet_rais_strict"],
        "rais_index": report["fleet_rais_index"],
        "rais_raw_covered": report["fleet_rais_raw_covered"],
        "rais_raw_index": report["fleet_rais_raw_index"],
    }
    if report.get("stack_badge"):
        fleet_obj["stack_badge"] = report["stack_badge"]
    if clean_receipt:
        fleet_obj["receipt"] = clean_receipt
    payload = {
        "rais_version": report["rais_version"],
        "basket_id": report["basket_id"],
        "index_divisor": report["index_divisor"],
        "hardware": report["hardware"],
        "fleet": fleet_obj,
        "models": [
            {
                "model": m["model"],
                "engine": m["engine"],
                "stack_badge": m.get("stack_badge"),
                "dtype": m["dtype"],
                "status": m["status"],
                "rais": m["certified_rais"],
                "raw_rais": m["raw_rais"],
                "base_r": m["base_roofline_r"],
                "spec": m["speculation_summary"],
                "pillars": [m["pillars"][p] for p in PILLAR_ORDER],
                "submetric_scores": {k: m["submetrics"][k]["score"] for k in SUBMETRIC_WEIGHTS},
                "failed_gates": m["gates"]["failed_gates"],
                **({"receipt": clean_receipt} if clean_receipt else {}),
            }
            for m in report["models"]
        ],
        "roi_top": report["fleet_roi_top"][:15],
        "submetrics_meta": [
            {"key": k, "pillar": p, "label": lbl, "weight_pct": round(w * 100.0, 1)}
            for k, p, lbl, w in SUBMETRIC_SPECS
        ],
    }
    return json.dumps(payload, indent=2)


def emit_output(text: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rais_score", description="RAIS v1.1 3-roofline + Pillar 5 reference scorer and ROI planner.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Score a RAIS JSONL file against a hardware profile.")
    p_score.add_argument("results", help="Path to RAIS results JSONL.")
    p_score.add_argument("--hardware", default="16x-gb10-1x100g", choices=sorted(HARDWARE_PROFILES))
    p_score.add_argument("--weights", help="Optional basket weights JSON file.")
    p_score.add_argument("--divisor", type=float, default=None, help="Optional S&P/DJIA continuity divisor.")
    p_score.add_argument("--basket-id", default=None, help="Optional basket identifier.")
    p_score.add_argument("--stack-badge", default=None, help="Optional display-only stack badge metadata.")
    p_score.add_argument("--format", default="markdown", choices=("markdown", "json", "html-data"))
    p_score.add_argument("--output", help="Optional output file path.")

    p_cmp = sub.add_parser("compare", help="Compare two RAIS JSONL runs on identical hardware.")
    p_cmp.add_argument("engine_a", help="Baseline / Engine A RAIS JSONL.")
    p_cmp.add_argument("engine_b", help="Candidate / Engine B RAIS JSONL.")
    p_cmp.add_argument("--hardware", default="16x-gb10-1x100g", choices=sorted(HARDWARE_PROFILES))
    p_cmp.add_argument("--weights", help="Optional basket weights JSON file.")
    p_cmp.add_argument("--divisor", type=float, default=None, help="Optional S&P/DJIA continuity divisor.")
    p_cmp.add_argument("--label-a", default="EngineA")
    p_cmp.add_argument("--label-b", default="EngineB")
    p_cmp.add_argument("--stack-badge-a", default=None, help="Optional display-only stack badge for Engine A.")
    p_cmp.add_argument("--stack-badge-b", default=None, help="Optional display-only stack badge for Engine B.")
    p_cmp.add_argument("--format", default="markdown", choices=("markdown", "json", "html-data"))
    p_cmp.add_argument("--output", help="Optional output file path.")

    p_rec = sub.add_parser("reconstitute", help="Compute S&P/DJIA continuity divisor for a basket reconstitution.")
    p_rec.add_argument("overlap", help="Path to overlap RAIS JSONL containing both old and new basket models.")
    p_rec.add_argument("--hardware", default="16x-gb10-1x100g", choices=sorted(HARDWARE_PROFILES))
    p_rec.add_argument("--new-weights", required=True, help="JSON file mapping new basket canonical models to weights summing to 1.0.")
    p_rec.add_argument("--prev-weights", help="Optional JSON file for previous basket weights (defaults to built-in catalog).")
    p_rec.add_argument("--prev-divisor", type=float, default=None, help="Previous continuity divisor (defaults to 1.0 or prev-weights divisor).")
    p_rec.add_argument("--basket-id", default=None, help="Identifier for the reconstituted basket.")
    p_rec.add_argument("--allow-unverified", action="store_true", help="Allow UNVERIFIED overlap records (still refuses INVALID).")
    p_rec.add_argument("--format", default="json", choices=("json", "markdown"))
    p_rec.add_argument("--output", help="Optional output file path.")

    p_imp = sub.add_parser("import-hillclimb", help="Convert legacy hillclimb or api_serving_perf JSONL to RAIS JSONL.")
    p_imp.add_argument("hillclimb", help="Path to hillclimb/baseline/after JSONL.")
    p_imp.add_argument("--model", help="Default model name when not embedded in records.")
    p_imp.add_argument("--engine", default="sparkpipe")
    p_imp.add_argument("--hardware", default="16x-gb10-1x100g", choices=sorted(HARDWARE_PROFILES))
    p_imp.add_argument("--score", action="store_true", help="Score imported records immediately instead of emitting JSONL.")
    p_imp.add_argument("--format", default="markdown", choices=("markdown", "json", "html-data"))
    p_imp.add_argument("--output", help="Optional output file path.")

    p_list = sub.add_parser("catalog", help="Print supported hardware profiles and fleet model geometries.")
    p_list.add_argument("--format", default="json", choices=("json", "markdown"))
    return parser


def render_catalog_markdown() -> str:
    lines = [
        f"# RAIS v{RAIS_VERSION} Hardware Profiles & Fleet Model Catalog",
        "",
        "## Hardware Profiles",
        "",
        "| Profile | Ranks | Mem BW/rank | BF16 / FP8 / FP4 TFLOPS | Net BW/rank | Hop Latency | NVMe BW/rank | RAM / OS Reserve |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for hw in HARDWARE_PROFILES.values():
        lines.append(
            f"| `{hw.name}` | {hw.ranks} | {hw.mem_bw_gb_s_per_rank:.0f} GB/s | "
            f"{hw.bf16_tflops_per_rank:.0f} / {hw.fp8_tflops_per_rank:.0f} / {hw.fp4_tflops_per_rank:.0f} | "
            f"{hw.net_bw_gb_s_per_rank:.1f} GB/s | {hw.net_hop_us:.0f} us | "
            f"{hw.nvme_bw_gb_s_per_rank:.1f} GB/s | {hw.ram_gib_per_rank:.0f} / {hw.os_reserve_gib_per_rank:.0f} GiB |"
        )
    lines.extend([
        "",
        "## Fleet Models (19 Models / 21 Lanes)",
        "",
        "| Lane | Tier | Fleet Wt | Active Params | Dtype | Hidden | Layers | KV B/tok | Context |",
        "| --- | :---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for m in MODEL_SPECS.values():
        wt = f"{m.fleet_weight * 100:.1f}%" if m.fleet_weight > 0 else f"variant ({m.canonical_model})"
        lines.append(
            f"| `{m.name}` | T{m.tier} | {wt} | {m.active_params_b:.1f}B | {m.dtype} | "
            f"{m.hidden} | {m.layers} | {m.kv_bytes_per_token} | {m.context_tokens} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "catalog":
        if args.format == "markdown":
            print(render_catalog_markdown())
        else:
            print(json.dumps({
                "hardware": {k: asdict(v) for k, v in HARDWARE_PROFILES.items()},
                "models": {k: asdict(v) for k, v in MODEL_SPECS.items()},
            }, indent=2))
        return 0

    if args.command == "score":
        records = load_jsonl(Path(args.results))
        weights_override = None
        basket_id = args.basket_id or DEFAULT_BASKET_ID
        divisor = DEFAULT_DIVISOR if args.divisor is None else args.divisor
        if args.weights:
            weights_override, file_basket, file_div = load_weights_file(Path(args.weights))
            if args.basket_id is None:
                basket_id = file_basket
            if args.divisor is None and file_div is not None:
                divisor = file_div
        report = score_fleet(
            records,
            args.hardware,
            weights_override=weights_override,
            divisor=divisor,
            basket_id=basket_id,
            stack_badge_override=args.stack_badge,
        )
        if args.format == "json":
            emit_output(json.dumps(report, indent=2), args.output)
        elif args.format == "html-data":
            emit_output(render_html_data(report, receipt_label=args.results), args.output)
        else:
            emit_output(render_markdown(report), args.output)
        return 0

    if args.command == "compare":
        weights_override = None
        basket_id = DEFAULT_BASKET_ID
        divisor = DEFAULT_DIVISOR if args.divisor is None else args.divisor
        if args.weights:
            weights_override, basket_id, file_div = load_weights_file(Path(args.weights))
            if args.divisor is None and file_div is not None:
                divisor = file_div
        rep_a = score_fleet(
            load_jsonl(Path(args.engine_a)),
            args.hardware,
            weights_override=weights_override,
            divisor=divisor,
            basket_id=basket_id,
            stack_badge_override=args.stack_badge_a,
        )
        rep_b = score_fleet(
            load_jsonl(Path(args.engine_b)),
            args.hardware,
            weights_override=weights_override,
            divisor=divisor,
            basket_id=basket_id,
            stack_badge_override=args.stack_badge_b,
        )
        cmp_rep = compare_reports(rep_a, rep_b, args.label_a, args.label_b)
        if args.format in ("json", "html-data"):
            emit_output(json.dumps(cmp_rep, indent=2), args.output)
        else:
            emit_output(render_compare_markdown(cmp_rep), args.output)
        return 0

    if args.command == "reconstitute":
        overlap = load_jsonl(Path(args.overlap))
        new_w, new_basket, _ = load_weights_file(Path(args.new_weights))
        prev_w = None
        prev_basket = DEFAULT_BASKET_ID
        prev_div = DEFAULT_DIVISOR if args.prev_divisor is None else args.prev_divisor
        if args.prev_weights:
            prev_w, prev_basket, file_div = load_weights_file(Path(args.prev_weights))
            if args.prev_divisor is None and file_div is not None:
                prev_div = file_div
        rec = reconstitute_basket(
            overlap,
            args.hardware,
            new_weights=new_w,
            prev_weights=prev_w,
            prev_divisor=prev_div,
            prev_basket_id=prev_basket,
            new_basket_id=args.basket_id or new_basket,
            allow_unverified=args.allow_unverified,
        )
        if args.format == "markdown":
            emit_output(render_reconstitute_markdown(rec), args.output)
        else:
            emit_output(json.dumps(rec, indent=2), args.output)
        return 0

    if args.command == "import-hillclimb":
        imported = import_hillclimb_records(Path(args.hillclimb), default_model=args.model, engine=args.engine)
        if args.score:
            report = score_fleet(imported, args.hardware)
            if args.format == "json":
                emit_output(json.dumps(report, indent=2), args.output)
            elif args.format == "html-data":
                emit_output(render_html_data(report, receipt_label=args.hillclimb), args.output)
            else:
                emit_output(render_markdown(report), args.output)
        else:
            emit_output("\n".join(json.dumps(rec) for rec in imported), args.output)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
