#!/usr/bin/env python3
"""Prefill/decode performance estimator for the six family drivers.

Extends the byte roofline of docs/archive/PERF_ROADMAP_2026-08-01.md (whose per-batch
step-byte law it REUSES by importing tools/nvme_kv_estimate.py - same Model
geometry, same coverage function, same step_gb) with the terms the roadmap
prices only in prose: launch overhead (eager vs CUDA-graph replay), the 6.5
TFLOPS compute wall, TP collectives and PP stage transport per topology, and
a chunked-prefill model. Docs write-up: docs/archive/PREFILL_DECODE_ESTIMATES.md;
invariants: tests/test_perf_estimate.py.

== ASSUMED CONSTANTS - measured nowhere in this tree, re-run with receipts ====

  LAUNCH_NS_DEFAULT   2000 ns     per-kernel launch/plan floor. Calibration
                                  doc: 2-5 us (docs/archive/PERF_ROADMAP_2026-08-01.md:587,
                                  docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md:36-44)
                                  and explicitly PENDING
                                  (docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md:155-157).
                                  Default = bottom of the range; --launch-ns.
  RING_AR_LATENCY_US  29.0        TP all-reduce latency floor per AR on the
                                  ring: the measured 29 us/hop floor
                                  (AUDIT_2026-07-29_BANDWIDTH_PATH.md:65)
                                  applied ONCE per AR, i.e. assuming hop-
                                  pipelined ring AR. ASSUMPTION.
  SWITCH_AR_LATENCY_US 15.0       per-AR latency floor on the dual-switch
                                  fabric (TP16_DUAL_RAIL_SPEED.md:36-37,
                                  "15 us allreduce latency floor per hop-pair
                                  assumed" - assumed there too).
  SWITCH_HOP_US       15.0        PP stage-boundary latency on the switch;
                                  same source, same assumed status.
  MFU_LO/HI           0.35/0.55   prefill compute efficiency range (task
                                  parameter; nothing in-tree pins prefill MFU).
  CHUNK_TOKENS        256         chunked-prefill dispatch cap
                                  (model_contracts/glm52.json:36,
                                  roadmap:763-768).
  GB10 SM specs (occupancy sanity check ONLY):
    REGS_PER_SM       65536       64K 32-bit registers/SM. ASSUMPTION (every
                                  NVIDIA SM since Volta; not GB10-confirmed).
    THREADS_PER_SM    2048        ASSUMPTION (sm_121 is GB100-lineage; consumer
                                  sm_120 is 1536. Second-order: registers bind
                                    first at every kernel below).
    MAX_BLOCKS_PER_SM 32          ASSUMPTION (architectural limit).
    SMEM_PER_SM       131072 B    NOT assumed: inference/kernels/layout.cuh:21
                                  (LM_SMEM_SM_TOTAL), matching the calibration
                                  doc's "128 KB L1/shared per SM" (:23).
    GEMM block        256 threads runtime/launch.h:201 (plan->block_threads).
    MXFP4 stored bits 4.25/weight group-32 E8M0 (roadmap:838-839); the GEMM
                                  smem estimate uses it for LmTileBytes.
    sm_121 dynamic-smem opt-in ceiling >= 66 KiB: ASSUMPTION. The delta rule
                                  carves 64 KiB dynamic + 1664 B static and
                                  the static_assert only bounds it by
                                  LM_SMEM_SM_TOTAL; whether
                                  cudaFuncSetAttribute grants >48 KiB on GB10
                                  is unverified (kimi_k3/layer.cuh:405-412).

== DERIVED CONSTANTS - cited ==================================================

  ETA_BW              0.80        three independent derivations
                                  (docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md:87-97)
  NODE_BW_GBPS        273.0       GB10 LPDDR5X (roadmap:29)
  NODE_TFLOPS         BF16 31 / FP8 250 / FP4 500 (roadmap:29-30, :433-434)
  WALL_TFLOPS         6.5/node    measured QKVO WMMA wall
                                  (docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md:66-73)
  RING_HOP_FLOOR_US   29.0        measured (roadmap:36-38)
  wire rates          ring 12.5 GB/s/link (1x 100 GbE, HARDWARE_TOPOLOGY.md
                      ring mode), dual-switch 25 GB/s/node (2x 100 GbE rails,
                      TP16_DUAL_RAIL_SPEED.md:8-11)
  launches/step       static LM_LAUNCH enumeration of
                      inference/llms/*/layer.cuh per layer kind x layer count
                      (derivation in PERF below). The roadmap's D1 hand count
                      for K3 is ~3,276 (roadmap:589-604) vs 1,974 static here -
                      it counts logical kernels the wrappers may fuse or loop;
                      launch counts carry the roadmap's own +-10% standing
                      uncertainty (roadmap:845-846) and the K3 gap is PENDING
                      reconciliation. --launches-multiplier for sensitivity.
  ptxas table         build server /opt/sparkpipe-topo/build/cuda13_sm121a_gate
                      /logs/inference__llms__*__unity.ptxas.txt (CUDA 13.3,
                      sm_121a, production flags, 2026-08-01). Baked in below;
                      --ptxas-dir re-parses the logs instead.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm52_model_contract import load_model_contract  # noqa: E402
from nvme_kv_estimate import (  # noqa: E402
    GB, MODELS_BY_NAME, BATCHES, coverage, step_gb)

# -- assumed constants (see module docstring) ---------------------------------
LAUNCH_NS_DEFAULT = 2000.0
RING_AR_LATENCY_US = 29.0
SWITCH_AR_LATENCY_US = 15.0
SWITCH_HOP_US = 15.0
RING_HOP_FLOOR_US = 29.0
MFU_LO = 0.35
MFU_HI = 0.55
MFU_DEFAULT = 0.45
GLM52_CONTRACT = load_model_contract()
CHUNK_TOKENS = GLM52_CONTRACT["max_prefill_tokens_per_dispatch"]
REGS_PER_SM = 65536
THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
SMEM_PER_SM = 131072
GEMM_BLOCK_THREADS = 256

# -- derived constants ---------------------------------------------------------
ETA_BW = 0.80
NODE_BW_GBPS = 273.0
NODE_TFLOPS = {"bf16": 31.0, "fp8": 250.0, "fp4": 500.0}   # roadmap:29-30, :433
WALL_TFLOPS_PER_NODE = 6.5   # docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md:66-73

GRAPH_LAUNCH_EXTRA = 1   # graph mode: 1 launch per stage + this many for the
                         # head/sample outside capture. ASSUMPTION.


class Topology:
    def __init__(self, name, nodes, wire_gbps, ar_latency_us, hop_us, mode):
        self.name = name
        self.nodes = nodes
        self.wire_gbps = wire_gbps
        self.ar_latency_us = ar_latency_us
        self.hop_us = hop_us
        self.mode = mode   # 'ring' | 'switch'

    @property
    def aggregate_gbps(self):
        """Effective aggregate memory bandwidth at the calibrated eta."""
        return self.nodes * NODE_BW_GBPS * ETA_BW


TOPOLOGIES = {
    # 13-node point-to-point ring, 1x 100 GbE per direction
    # (examples/topologies/ring_13node_bringup.json; roadmap:32-33)
    "ring13": Topology("ring13", 13, 12.5, RING_AR_LATENCY_US,
                       RING_HOP_FLOOR_US, "ring"),
    # 16-node dual-switch production, 2x 100 GbE rails
    # (examples/topologies/dual_switch_16node_production.json;
    # TP16_DUAL_RAIL_SPEED.md:8-11)
    "dual16": Topology("dual16", 16, 25.0, SWITCH_AR_LATENCY_US,
                       SWITCH_HOP_US, "switch"),
}


class Perf:
    """Per-model geometry the byte model in nvme_kv_estimate does not carry.

    gflops          decode FLOPs per token by precision, 2 x active params
                    (roadmap:423-430)
    launches        decode-step kernel launches, static LM_LAUNCH enumeration
                    (derivation comment per row)
    full_layers     layers with full (quadratic) attention
    swa_layers/win  sliding-window layers and window (their prefill attention
                    is linear in ctx past the window)
    score_dim       heads x head_dim summed per full-attention layer: the
                    quadratic term is full_layers x score_dim x ctx^2 FLOPs
                    (2 x d per (query, past) pair per matmul, causal halves
                    the pair count: 2 x 2 x d x C^2 / 2). BF16 rates.
    payload_kb      PP stage-boundary payload per row in KiB (K3's AttnRes
                    makes it 9 reps, not 1: config.h:205-213)
    shared_gb       shared-expert stream per step, which the coverage law
                    omits (it prices only routed draws). The roadmap's B1
                    tables include it (dsv4_pro: 7 x 61 x 33.0 MB = 14.1 GB
                    routed+shared vs 12.0 coverage-only, roadmap:371-378);
                    its B8+ tables omit it again - their inconsistency, not
                    this model's. Carried here so B1 is right; the B8+
                    shift is <2%.
    """

    def __init__(self, hidden, layers, gflops, launches, full_layers,
                 score_dim, swa_layers=0, swa_window=0, payload_kb=None,
                 shared_gb=0.0, launches_note=""):
        self.hidden = hidden
        self.layers = layers
        self.gflops = gflops
        self.launches = launches
        self.full_layers = full_layers
        self.score_dim = score_dim
        self.swa_layers = swa_layers
        self.swa_window = swa_window
        self.payload_kb = (payload_kb if payload_kb is not None
                           else hidden * 2 / 1024.0)
        self.shared_gb = shared_gb
        self.launches_note = launches_note


PERF = {
    # 69 KDA x 14 + 24 MLA x 7 + 93 LatentMoE x 7 + ~2 AttnRes x 93 + head 3
    # (kimi_k3/layer.cuh K3LayerKda/K3LayerMla/K3LayerLatentMoe/K3AttnRes/
    # K3Head). Roadmap D1 hand count 3,276 - see module docstring.
    "k3": Perf(7168, 93, {"bf16": 110.0, "fp4": 97.0}, 1974,
               full_layers=24, score_dim=96 * 192, payload_kb=126.0,
               launches_note="roadmap D1 counts 3,276 (PENDING gap)"),
    # 78 attn x 5 + 3 dense-MLP x 2 + 75 MoE x 5 + head 3 (glm5_2/layer.cuh)
    "glm52": Perf(
        GLM52_CONTRACT["hidden_dimension"],
        GLM52_CONTRACT["layer_count"],
        {"bf16": 30.5, "fp8": 45.2},
        774,
        full_layers=GLM52_CONTRACT["layer_count"],
        score_dim=(
            GLM52_CONTRACT["head_count"]
            * (GLM52_CONTRACT["latent_dimension"]
               + GLM52_CONTRACT["rope_dimension"])
        ),
    ),
    # 16 full-attn x 6 + 48 GDN x 6 + 64 dense-MLP x 2 + head 3
    # (qwen_3_6/layer.cuh; dense model, FFN every layer)
    "qwen38_27b": Perf(5120, 64, {"bf16": 50.2}, 515,
                   full_layers=16, score_dim=24 * 256),
    # 61 x (attn 9 + MoE 10) + head 3; pro shares deepseek_v4/layer.cuh
    "dsv4_pro": Perf(7168, 61, {"bf16": 3.7, "fp8": 77.0, "fp4": 56.4}, 1162,
                     full_layers=61, score_dim=128 * 512, shared_gb=2.0),
    # 43 x (attn 9 + MoE 10) + head 3; 2 SWA layers bounded at 128 slots
    # (roadmap:386-389)
    "dsv4_flash": Perf(4096, 43, {"bf16": 1.1, "fp8": 16.8, "fp4": 15.2}, 820,
                       full_layers=41, score_dim=64 * 512,
                       swa_layers=2, swa_window=128, shared_gb=0.54),
    # 48 attn x 8 + 47 MoE x 6 + 1 dense-MLP x 4 + head 3 (mimo_2_5/layer.cuh;
    # MIMO25_FIRST_ROUTED_LAYER is a GUESS, config.h:81)
    "mimo25": Perf(4096, 48, {"bf16": 1.3, "fp8": 27.6}, 673,
                   full_layers=7, score_dim=64 * 192,
                   swa_layers=41, swa_window=128),
}

# Top-3 kernels per model by register pressure, parsed from the build server's
# ptxas logs (see module docstring for provenance). smem = static smem from
# ptxas + known dynamic carve. The GEMMs' dynamic smem is recomputed from
# LmGemmSharedBytes (inference/kernels/gemm.cuh:40-47) by occupancy() below.
PTXAS_TOP = {
    "k3": [
        ("LmGemmKernel<Bf16,Mxfp4,M64,N128,K64,S2>", 109, 0,
         {"a_bits": 16, "b_bits": 4.25, "m": 64, "n": 128, "k": 64, "s": 2}),
        ("LmDeltaRuleKernel<256,128,128>", 94, 1664 + 65536, None),
        ("LmGemmKernel<Bf16,Bf16,M64,N128,K64,S2>", 87, 0,
         {"a_bits": 16, "b_bits": 16, "m": 64, "n": 128, "k": 64, "s": 2}),
    ],
    "glm52": [
        ("LmGemmKernel<Bf16,Fp8,M64,N128,K64,S2>", 107, 0,
         {"a_bits": 16, "b_bits": 8, "m": 64, "n": 128, "k": 64, "s": 2}),
        ("LmGemmKernel<Bf16,Bf16,M64,N128,K64,S2>", 87, 0,
         {"a_bits": 16, "b_bits": 16, "m": 64, "n": 128, "k": 64, "s": 2}),
        ("LmGemmKernel<Bf16,Fp8,M32,N128,K64,S2>", 72, 0,
         {"a_bits": 16, "b_bits": 8, "m": 32, "n": 128, "k": 64, "s": 2}),
    ],
    "qwen38_27b": [
        ("LmDeltaRuleKernel<256,128,128>", 94, 1664 + 65536, None),
        ("LmGemmKernel<Bf16,Bf16,M64,N128,K64,S2>", 87, 0,
         {"a_bits": 16, "b_bits": 16, "m": 64, "n": 128, "k": 64, "s": 2}),
        ("LmGemmKernel<Bf16,Bf16,M32,N128,K64,S2>", 64, 0,
         {"a_bits": 16, "b_bits": 16, "m": 32, "n": 128, "k": 64, "s": 2}),
    ],
    "dsv4_pro": [
        ("LmGemmKernel<Fp8,Fp8,M64,N128,K128,S2>", 123, 0,
         {"a_bits": 8, "b_bits": 8, "m": 64, "n": 128, "k": 128, "s": 2}),
        ("LmGemmKernel<Fp8,Mxfp4,M64,N128,K128,S2>", 123, 0,
         {"a_bits": 8, "b_bits": 4.25, "m": 64, "n": 128, "k": 128, "s": 2}),
        ("LmGemmKernel<Fp8,Mxfp4,M32,N128,K128,S2>", 83, 0,
         {"a_bits": 8, "b_bits": 4.25, "m": 32, "n": 128, "k": 128, "s": 2}),
    ],
    "dsv4_flash": [
        ("LmGemmKernel<Fp8,Fp8,M64,N128,K128,S2>", 123, 0,
         {"a_bits": 8, "b_bits": 8, "m": 64, "n": 128, "k": 128, "s": 2}),
        ("LmGemmKernel<Fp8,Mxfp4,M64,N128,K128,S2>", 123, 0,
         {"a_bits": 8, "b_bits": 4.25, "m": 64, "n": 128, "k": 128, "s": 2}),
        ("LmGemmKernel<Bf16,Bf16,M64,N128,K64,S2>", 87, 0,
         {"a_bits": 16, "b_bits": 16, "m": 64, "n": 128, "k": 64, "s": 2}),
    ],
    "mimo25": [
        ("LmGemmKernel<Fp8,Fp8,M64,N128,K128,S2>", 123, 0,
         {"a_bits": 8, "b_bits": 8, "m": 64, "n": 128, "k": 128, "s": 2}),
        ("LmGemmKernel<Int7,Int7,M64,N128,K256,S2>", 95, 0,
         {"a_bits": 7, "b_bits": 7, "m": 64, "n": 128, "k": 256, "s": 2}),
        ("LmGemmKernel<Bf16,Bf16,M64,N128,K64,S2>", 87, 0,
         {"a_bits": 16, "b_bits": 16, "m": 64, "n": 128, "k": 64, "s": 2}),
    ],
}

PREFILL_CONTEXTS = (2048, 32768, 131072)


# ---------------------------------------------------------------- decode ----

def gemm_smem_bytes(tile):
    """LmGemmSharedBytes (inference/kernels/gemm.cuh:40-47), decimal bits."""
    return (tile["s"] * (tile["m"] * tile["k"] * tile["a_bits"] / 8.0)
            + tile["s"] * (tile["n"] * tile["k"] * tile["b_bits"] / 8.0)
            + tile["s"] * 8)


def occupancy(regs, smem_bytes, block_threads=GEMM_BLOCK_THREADS):
    """Blocks resident per SM and occupancy vs THREADS_PER_SM. The SM
    constants are the flagged assumptions at the top of this file."""
    by_regs = REGS_PER_SM // (regs * block_threads)
    by_smem = (SMEM_PER_SM // smem_bytes) if smem_bytes > 0 else 64
    by_threads = THREADS_PER_SM // block_threads
    blocks = int(min(by_regs, by_smem, by_threads, MAX_BLOCKS_PER_SM))
    bound = "regs" if by_regs <= min(by_smem, by_threads) else (
        "smem" if by_smem <= by_threads else "threads")
    return blocks, blocks * block_threads / THREADS_PER_SM, bound


def perf_step_gb(model_name, batch, context, bf16_state=False):
    """The imported roadmap byte law plus the shared-expert stream the
    coverage law omits (see Perf.shared_gb)."""
    return (step_gb(MODELS_BY_NAME[model_name], batch, context, bf16_state)
            + PERF[model_name].shared_gb)


def memory_roofline_tok_s(model_name, batch, context=2048, nodes=13,
                          bf16_state=False):
    """The roadmap's per-batch number: aggregate bytes/s at eta over step
    bytes. The test holds this to the roadmap's tables to rounding."""
    agg = nodes * NODE_BW_GBPS * ETA_BW
    return agg / perf_step_gb(model_name, batch, context, bf16_state)


def compute_time_s(model_name, batch, nodes, mode="wall"):
    """Batch FLOPs against the ring's compute. wall: every GEMM class at the
    measured 6.5 TFLOPS/node QKVO figure (pessimistic for FP8/FP4 experts -
    roadmap:436-438, :862-864). peak: each precision at its unit peak."""
    pm = PERF[model_name]
    total = 0.0
    for precision, gf in pm.gflops.items():
        rate = (WALL_TFLOPS_PER_NODE if mode == "wall"
                else NODE_TFLOPS[precision])
        total += gf / (nodes * rate * 1000.0)   # GF / (TFLOP/s) -> ms*1e-3
    return batch * total


def launch_time_s(model_name, launch_mode, launch_ns, stages):
    if launch_mode == "graph":
        return (stages + GRAPH_LAUNCH_EXTRA) * launch_ns * 1e-9
    return PERF[model_name].launches * launch_ns * 1e-9


def collective_time_s(model_name, batch, topo):
    """TP: 2 all-reduces per layer of B x hidden BF16 partials (TP16_DUAL_
    RAIL_SPEED.md:22-24). Ring AR wire bytes use the 2(N-1)/N ring-AR factor;
    the switch figure assumes a hierarchical AR at full node injection -
    ASSUMPTION. Latency floors are the flagged per-AR constants."""
    pm = PERF[model_name]
    payload = batch * pm.hidden * 2
    if topo.mode == "ring":
        wire_bytes = 2.0 * (topo.nodes - 1) / topo.nodes * payload
    else:
        wire_bytes = 2.0 * payload
    t_ar = wire_bytes / (topo.wire_gbps * GB) + topo.ar_latency_us * 1e-6
    return 2 * pm.layers * t_ar


def transport_time_s(model_name, batch, topo):
    """PP: one stage boundary per hop, (stages - 1) boundaries, payload per
    row from PERF (K3's AttnRes 126 KiB; hidden x 2 B otherwise)."""
    pm = PERF[model_name]
    payload = batch * pm.payload_kb * 1024.0
    t_hop = payload / (topo.wire_gbps * GB) + topo.hop_us * 1e-6
    return (topo.nodes - 1) * t_hop


def decode_step(model_name, batch, context=2048, topo_name="ring13",
                launch_mode="eager", launch_ns=LAUNCH_NS_DEFAULT,
                compute_mode="wall", bf16_state=False):
    """Step time = max(bandwidth, compute) + launch + min(TP, PP) overlay.

    Returns every component so callers can rank bottlenecks by share."""
    topo = TOPOLOGIES[topo_name]
    mem_gb = perf_step_gb(model_name, batch, context, bf16_state)
    t_mem = mem_gb / topo.aggregate_gbps
    t_comp = compute_time_s(model_name, batch, topo.nodes, compute_mode)
    t_launch = launch_time_s(model_name, launch_mode, launch_ns, topo.nodes)
    t_coll = collective_time_s(model_name, batch, topo)
    t_trans = transport_time_s(model_name, batch, topo)
    overlay_tp = t_launch + t_coll
    overlay_pp = t_launch + t_trans
    recipe = "TP" if overlay_tp <= overlay_pp else "PP"
    step = max(t_mem, t_comp) + min(overlay_tp, overlay_pp)
    components = {
        "bandwidth": t_mem,
        "compute": t_comp,
        "launch": t_launch,
        "collective": t_coll if recipe == "TP" else 0.0,
        "transport": t_trans if recipe == "PP" else 0.0,
    }
    ranked = sorted(components.items(), key=lambda kv: -kv[1])
    return {
        "model": model_name, "batch": batch, "context": context,
        "topology": topo_name, "recipe": recipe, "step_s": step,
        "tok_s_seq": 1.0 / step, "tok_s_agg": batch / step,
        "mem_gb": mem_gb, "components": components, "ranked": ranked,
        "binding": ranked[0][0],
        "t_coll": t_coll, "t_trans": t_trans,
        "mbu": t_mem * ETA_BW / step if step > 0 else 0.0,
    }


def crossover_batch(model_name, topo_name="ring13", compute_mode="wall",
                    context=2048, max_batch=8192):
    """Smallest power-of-two-ish batch where compute time overtakes memory
    time; None if never within max_batch. Scans finely (the crossovers the
    roadmap names - B34, B130, B165, B258 - are not powers of two)."""
    topo = TOPOLOGIES[topo_name]
    bm = MODELS_BY_NAME[model_name]
    for batch in range(1, max_batch + 1):
        # The roadmap's crossover table (roadmap:423-430) is derived from the
        # coverage-law bytes WITHOUT the shared-expert correction; matching
        # it exactly means using the same bytes here.
        t_mem = step_gb(bm, batch, context) / topo.aggregate_gbps
        if compute_time_s(model_name, batch, topo.nodes, compute_mode) > t_mem:
            return batch
    return None


# --------------------------------------------------------------- prefill ----

def attention_quad_flops(model_name, context):
    """Total prefill attention FLOPs at a context: quadratic on full layers,
    linear-in-ctx past the window on SWA layers. BF16-rate work."""
    pm = PERF[model_name]
    quad = pm.full_layers * pm.score_dim * context * context
    if pm.swa_layers and context > pm.swa_window:
        quad += (pm.swa_layers * pm.score_dim * pm.swa_window
                 * (context - pm.swa_window))
    return float(quad)


def prefill(model_name, context, nodes=13, mfu=MFU_DEFAULT,
            launch_mode="eager", launch_ns=LAUNCH_NS_DEFAULT,
            chunk=CHUNK_TOKENS, topo_name=None):
    """Chunked prefill of ONE sequence. Per chunk: max(weight+KV-write stream,
    chunk compute incl. the attention quadratic slice) + launch + (TP
    collective | PP transport) when a topology is given. nodes=1 with
    topo_name=None is the single-node viability number."""
    bm = MODELS_BY_NAME[model_name]
    pm = PERF[model_name]
    topo = TOPOLOGIES[topo_name] if topo_name else None
    agg_bw = nodes * NODE_BW_GBPS * ETA_BW
    n_chunks = max(1, math.ceil(context / chunk))
    total = 0.0
    comp_total = mem_total = launch_total = comm_total = 0.0
    for i in range(n_chunks):
        past = i * chunk
        rows = min(chunk, context - past)
        mem_gb = (bm.fixed_gb + pm.shared_gb
                  + bm.pool_gb * coverage(rows, bm.n_experts, bm.top_k)
                  + rows * bm.kv_store_bpt / GB          # KV write traffic
                  + bm.state_stream_gb)                  # state R+W once/chunk
        t_mem = mem_gb / agg_bw
        quad_slice = (pm.full_layers * pm.score_dim
                      * ((past + rows) ** 2 - past ** 2))
        if pm.swa_layers and past + rows > pm.swa_window:
            quad_slice += (pm.swa_layers * pm.score_dim * pm.swa_window
                           * rows)
        t_comp = quad_slice / (nodes * NODE_TFLOPS["bf16"] * 1e12 * mfu)
        for precision, gf in pm.gflops.items():
            t_comp += (rows * gf * 1e9
                       / (nodes * NODE_TFLOPS[precision] * 1e12 * mfu))
        t_launch = launch_time_s(model_name, launch_mode, launch_ns,
                                 topo.nodes if topo else 1)
        t_comm = 0.0
        if topo is not None:
            t_comm = min(collective_time_s(model_name, rows, topo),
                         transport_time_s(model_name, rows, topo))
        total += max(t_mem, t_comp) + t_launch + t_comm
        mem_total += t_mem
        comp_total += t_comp
        launch_total += t_launch
        comm_total += t_comm
    binding = max((("bandwidth", mem_total), ("compute", comp_total),
                   ("launch", launch_total), ("collective/transport",
                                              comm_total)),
                  key=lambda kv: kv[1])[0]
    return {"model": model_name, "context": context, "nodes": nodes,
            "time_s": total, "tok_s": context / total, "binding": binding,
            "components": {"bandwidth": mem_total, "compute": comp_total,
                           "launch": launch_total,
                           "collective/transport": comm_total}}


# ------------------------------------------------------------- reporting ----

def fmt_tok_s(x):
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def decode_table(topo_name, launch_mode, launch_ns, compute_mode,
                 bf16_state=False):
    topo = TOPOLOGIES[topo_name]
    lines = [f"DECODE {topo.name} ({topo.nodes} nodes, "
             f"{topo.aggregate_gbps:,.0f} GB/s @eta {ETA_BW}), "
             f"{launch_mode} launches @ {launch_ns:.0f} ns, "
             f"compute {compute_mode}, ctx 2K",
             f"{'model':<11} {'B':>5} {'tok/s/seq':>10} {'agg tok/s':>11} "
             f"{'recipe':>6} {'binding':<10} top-3 bottlenecks (share)"]
    rows = []
    for name in PERF:
        for batch in BATCHES:
            r = decode_step(name, batch, 2048, topo_name, launch_mode,
                            launch_ns, compute_mode, bf16_state)
            top3 = ", ".join(
                f"{k} {v / r['step_s'] * 100:.0f}%"
                for k, v in r["ranked"][:3] if v > 0)
            lines.append(f"{name:<11} {batch:>5} "
                         f"{fmt_tok_s(r['tok_s_seq']):>10} "
                         f"{fmt_tok_s(r['tok_s_agg']):>11} "
                         f"{r['recipe']:>6} {r['binding']:<10} {top3}")
            rows.append(r)
        lines.append("")
    return lines, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topology", choices=sorted(TOPOLOGIES), default=None)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--launch-ns", type=float, default=LAUNCH_NS_DEFAULT)
    parser.add_argument("--launch-mode", choices=("eager", "graph"),
                        default="eager")
    parser.add_argument("--compute-mode", choices=("wall", "peak"),
                        default="wall")
    parser.add_argument("--mfu", type=float, default=MFU_DEFAULT)
    parser.add_argument("--k3-bf16-state", action="store_true")
    parser.add_argument("--launches-multiplier", type=float, default=1.0,
                        help="sensitivity: scale every model's static launch "
                             "count (K3's roadmap hand count is ~1.66x)")
    args = parser.parse_args(argv)
    if args.launches_multiplier != 1.0:
        for pm in PERF.values():
            pm.launches = int(round(pm.launches * args.launches_multiplier))

    topos = [args.topology] if args.topology else sorted(TOPOLOGIES)
    print(f"ASSUMED: launch {args.launch_ns:.0f} ns (PENDING, range 2-5 us), "
          f"eta_bw {ETA_BW}, wall {WALL_TFLOPS_PER_NODE} TFLOPS/node, "
          f"MFU {MFU_LO}-{MFU_HI}, AR latency ring {RING_AR_LATENCY_US:.0f}/"
          f"switch {SWITCH_AR_LATENCY_US:.0f} us, SM specs flagged in file")
    print()

    for topo_name in topos:
        lines, _ = decode_table(topo_name, args.launch_mode, args.launch_ns,
                                args.compute_mode, args.k3_bf16_state)
        print("\n".join(lines))

    # The 128K-context decode column: same model, KV term scaled.
    print("DECODE at 128K context, tok/s/seq (aggregate in parentheses), "
          f"{args.launch_mode} launches")
    print(f"{'model':<11} {'topo':<7} " + " ".join(f"{f'B{b}':>16}"
                                                  for b in BATCHES))
    for name in PERF:
        for topo_name in topos:
            cells = []
            for batch in BATCHES:
                r = decode_step(name, batch, 131072, topo_name,
                                args.launch_mode, args.launch_ns,
                                args.compute_mode, args.k3_bf16_state)
                cells.append(f"{fmt_tok_s(r['tok_s_seq']):>7}/"
                             f"{fmt_tok_s(r['tok_s_agg']):<8}")
            print(f"{name:<11} {topo_name:<7} " + " ".join(cells))
    print()

    # B1 MBU: does anything prevent 80% of the bandwidth roofline?
    print("B1 EFFECTIVE MBU (t_mem at eta / step), eager vs graph, ctx 2K")
    print(f"{'model':<11} {'topo':<7} {'eager':>7} {'graph':>7}  verdict")
    for name in PERF:
        for topo_name in topos:
            eager = decode_step(name, 1, 2048, topo_name, "eager",
                                args.launch_ns, args.compute_mode,
                                args.k3_bf16_state)
            graph = decode_step(name, 1, 2048, topo_name, "graph",
                                args.launch_ns, args.compute_mode,
                                args.k3_bf16_state)
            verdict = ("OK at graph" if graph["mbu"] >= 0.79
                       else "BELOW 80% even with graphs")
            print(f"{name:<11} {topo_name:<7} {eager['mbu']:>7.3f} "
                  f"{graph['mbu']:>7.3f}  {verdict}")
    print()

    print("BANDWIDTH->COMPUTE CROSSOVER BATCH (scan B=1..8192, ctx 2K)")
    print(f"{'model':<11} {'topo':<7} {'@wall':>7} {'@peak':>7}")
    for name in PERF:
        for topo_name in topos:
            wall = crossover_batch(name, topo_name, "wall")
            peak = crossover_batch(name, topo_name, "peak")
            print(f"{name:<11} {topo_name:<7} "
                  f"{str(wall):>7} {str(peak):>7}")
    print()

    print("COLLECTIVE (TP) vs TRANSPORT (PP) SHARE OF STEP, ctx 2K")
    print(f"{'model':<11} {'topo':<7} {'B':>5} {'TP coll %':>10} "
          f"{'PP trans %':>11} {'fabric>mem?':>11}")
    for name in PERF:
        for topo_name in topos:
            for batch in (1, 64, 1024):
                r = decode_step(name, batch, 2048, topo_name, "graph",
                                args.launch_ns, args.compute_mode,
                                args.k3_bf16_state)
                mem_bound = max(r["components"]["bandwidth"],
                                r["components"]["compute"])
                coll_pct = r["t_coll"] / r["step_s"] * 100
                trans_pct = r["t_trans"] / r["step_s"] * 100
                binds = "YES" if min(r["t_coll"], r["t_trans"]) > mem_bound \
                    else "no"
                print(f"{name:<11} {topo_name:<7} {batch:>5} "
                      f"{coll_pct:>9.1f}% {trans_pct:>10.1f}% {binds:>11}")
        print()

    print(f"PREFILL tok/s per sequence, MFU {args.mfu} "
          f"(range {MFU_LO}-{MFU_HI}), eager launches, PP overlay")
    print(f"{'model':<11} {'ctx':>8} {'ring13':>9} {'dual16':>9} "
          f"{'1 node':>9} {'128K 1-node time':>16}  binding(ring13)")
    for name in PERF:
        cells = {}
        for ctx in PREFILL_CONTEXTS:
            cells[ctx] = {t: prefill(name, ctx, TOPOLOGIES[t].nodes,
                                     args.mfu, topo_name=t)
                          for t in ("ring13", "dual16")}
            cells[ctx]["1node"] = prefill(name, ctx, 1, args.mfu)
        single = cells[131072]["1node"]
        hours = single["time_s"] / 3600.0
        t_str = (f"{single['time_s']:,.0f} s" if hours < 2
                 else f"{hours:,.1f} h")
        print(f"{name:<11} {2048:>8} {cells[2048]['ring13']['tok_s']:>9,.0f} "
              f"{cells[2048]['dual16']['tok_s']:>9,.0f} "
              f"{cells[2048]['1node']['tok_s']:>9,.0f} {'':>16}  "
              f"{cells[2048]['ring13']['binding']}")
        print(f"{'':<11} {32768:>8} {cells[32768]['ring13']['tok_s']:>9,.0f} "
              f"{cells[32768]['dual16']['tok_s']:>9,.0f} "
              f"{cells[32768]['1node']['tok_s']:>9,.0f} {'':>16}  "
              f"{cells[32768]['ring13']['binding']}")
        print(f"{'':<11} {131072:>8} "
              f"{cells[131072]['ring13']['tok_s']:>9,.0f} "
              f"{cells[131072]['dual16']['tok_s']:>9,.0f} "
              f"{single['tok_s']:>9,.0f} {t_str:>16}  "
              f"{cells[131072]['ring13']['binding']}")
    print()

    print("OCCUPANCY SANITY CHECK - top-3 kernels per model by ptxas register "
          "pressure")
    print(f"SM: {REGS_PER_SM} regs (ASSUMED), {SMEM_PER_SM} B smem "
          f"(layout.cuh:21), {THREADS_PER_SM} threads (ASSUMED), block "
          f"{GEMM_BLOCK_THREADS} (launch.h:201)")
    for name, kernels in PTXAS_TOP.items():
        print(f"  {name}")
        for kernel, regs, smem_static, tile in kernels:
            smem = gemm_smem_bytes(tile) if tile else smem_static
            blocks, occ, bound = occupancy(regs, smem)
            print(f"    {kernel:<44} {regs:>3} regs {smem:>7,.0f} B smem -> "
                  f"{blocks} blk/SM ({occ * 100:.0f}% occ, {bound}-bound)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
