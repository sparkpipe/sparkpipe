#!/usr/bin/env python3
"""Kimi K3 TP4xPP4 prefill + decode performance estimate (16 x DGX Spark).

ALL NUMBERS ARE ANALYTICAL ESTIMATES, NOT MEASUREMENTS, except the anchors
labelled measured below. Every projection is reported under TWO bandwidth
models so the analytical/measure boundary stays visible:

  convention  the repo roofline convention: 273 GB/s LPDDR5X x 0.65
              = 177.45 GB/s usable per spark (the dsv4-era default).
  calibrated  anchored to measured bandwidth points on this exact hardware
              class (see ANCHORS). This is the planning model; it lands the
              B1 decode projection ON the measured 55.5 ms stage step by
              construction, and lowers every large-batch projection ~12%.

Measured anchors, each with its source:

  A1  K3 itself: warm B1 stage-0 step on sparka = 55.5 ms (graph replay
      54.2 ms, bit-identical). docs/K3_PERF.md "2026-08-17". One stage,
      real rank pack, 24 layers. THE anchor for this model's path.
  A2  DRAM stream probe on sparkb (tools/devcycle/bw_probe.cu): 250.7 GB/s
      at 256 MB working set / 272.7 GB/s at 64 MB. The honest streaming
      ceiling is ~250 GB/s, not the 273 GB/s datasheet number.
  A3  Qwen3.8-27B TP4 band, spark0-3, quality-gated E2E
      (docs/QWEN36_TP4_PERF.md): measured device read bandwidth 225.9 GB/s;
      plain B1 runs at 72% of that weight floor -> 162.7 GB/s achieved
      end-to-end on a production decode path.

Calibration logic: the achieved per-spark bandwidth of a B1 weight-streaming
decode step is bytes_streamed / step_time (AR-latency subtracted first, same
as the model's stage equation). A1 gives 8.57 GB / 55.13 ms = 155.4 GB/s for
K3's mixed BF16-spine + MXFP4-expert stream; A3 gives 162.7 GB/s for qwen36's
all-BF16 stream. Two independent models,
same hardware class, within 4.5% of each other and at 62% of the A2 streaming
probe -- so the calibrated planning value is the CONSERVATIVE one (min),
155 GB/s-class, not the convention's optimistic 177 GB/s.

Ground truth inventory (read from the deployed rank pack manifest
k3.stage2.rank02.pack, pack v2, TP4 slice):

  per rank per KDA layer: qkv_beta 132.5 MB + decay 2.6 MB + gate 44.0 MB +
      out 44.0 MB + convs/bias ~0.2 MB + router 12.8 MB + shared_w1 44.0 MB +
      shared_w2 22.0 MB + routed_down/up 25.7 MB + norms ~0.1 MB  = 327.2 MB
  per rank per MLA layer: q_down 22.0 + q_up 42.5 + kv_a 8.3 + kv_b 3.1 +
      gate 44.0 + out 44.0 + shared 66.1 + router 12.8 + routed 25.7
      + norms ~0.1                                              = 268.7 MB
  per rank per layer, expert set (ALL 896 experts):
      w1 [896 k x 1536 out x 896 experts] = 655.1 MB
      w2 [896 x 3584 x 768]              = 1310.2 MB
      total                               = 1965.3 MB (MXFP4-E2M1 g32 + E8M0)
  per rank: embed + lm_head = 40960 x 7168 x 2 B = 587.2 MB each
  KDA recurrent state (fp32): slot 96x128x128x4 = 6.29 MB read+write plus
      three 4-wide conv windows 0.29 MB x2, per KDA layer per token,
      TP-sharded by heads.
  MLA KV per token per layer: 576 x 2 B = 1.2 KB (write).

Decode (output): B1 streams, per stage per rank:
      dense spine once + the routed experts at top-16/896 of their bytes +
      the KDA state read/write + (stage 3 only) lm_head,
  then two fused all-reduces per layer (NCCL tree latency on the critical
  path).  AR-latency sensitivity: the model uses 8 us/tree AR; the qwen36
  band measured ~27 us/op on its hidden-transport tier. Even at 27 us the
  whole-stage AR budget is 23 layers x 2 x 27 us = 1.24 ms against a ~55 ms
  step (~2%), so the projections are insensitive to the AR constant; it is
  NOT insensitive to the bandwidth term, which is why the model is calibrated.

Prefill: a batch of B tokens amortises the dense spine over B and the expert
  set over min(B * 18 / 896, 1) (each routed expert serves ~16B/896 tokens),
  while the KDA state read+write is strictly per token - at large B the state
  becomes the dominant term (the known K3 throughput lever, see
  inference/llms/kimi_k3/config.h).  PP4 pipelines micro-batches; steady-state
  throughput is B per stage step; a single isolated prompt pays the 4-stage
  fill bubble.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

CLASSIFICATION = "analytical estimate (bandwidth-calibrated to measured anchors)"
MEASURED = False

# -- machine ----------------------------------------------------------------
SPARK_BANDWIDTH_GB_PER_S = 273.0      # GB10 unified memory, per spark (datasheet)
USABLE_BANDWIDTH_FRACTION = 0.65      # repo convention (dsv4 roofline)
AR_LATENCY_US = 8.0                   # NCCL tree all-reduce, 4 ranks, ~86 KB
AR_COUNT_PER_LAYER = 2                # phase 0 + phase 1 (correctness-mandated)

# -- measured anchors (A1/A2/A3 above) ---------------------------------------
STREAM_PROBE_GB_PER_S = 250.7         # A2: bw_probe.cu @256 MB, sparkb
QWEN36_FLOOR_GB_PER_S = 225.9         # A3: measured device read bandwidth
QWEN36_PLAIN_FRACTION = 0.72          # A3: plain B1 vs its own weight floor
QWEN36_ACHIEVED_GB_PER_S = QWEN36_FLOOR_GB_PER_S * QWEN36_PLAIN_FRACTION

# -- model, per rank, TP4 slice ----------------------------------------------
LAYERS = 93
KDA_LAYERS = 69
MLA_LAYERS = 24
PP_STAGES = 4
TP_DEGREE = 4
KDA_DENSE_B = 327_200_000.0
MLA_DENSE_B = 268_700_000.0
EXPERT_SET_B = 1_965_293_568.0        # w1 + w2, all 896 experts
TOP_K = 16
SHARED_EXPERTS = 2
EXPERTS = 896
EMBED_B = 587_202_560.0               # 40960 x 7168 x 2
LM_HEAD_B = 587_202_560.0
KDA_STATE_B = 2.0 * (96 * 128 * 128 * 4 + ((2 * 96 * 128 + 96 * 128) * 4 * 2))
# per token per layer, per rank, TP-sharded by heads:
KDA_STATE_PER_RANK_B = KDA_STATE_B / TP_DEGREE
MLA_KV_B = (512 + 64) * 2
ACTIVATION_B = 88_000.0               # qkv/gate/latent/SiTU rounds + 2x43 KB AR wire

# -- measured anchor (A1) ------------------------------------------------------
MEASURED_STAGE_MS = 55.5               # warm B1 step, stage 0, one spark
MEASURED_TOKENS_PER_S = 1000.0 / MEASURED_STAGE_MS


def _stage_layers(first: int, count: int) -> tuple[int, int]:
    kda = mla = 0
    for layer in range(first, first + count):
        is_mla = (layer % 4 == 3) or layer == LAYERS - 1
        if is_mla:
            mla += 1
        else:
            kda += 1
    return kda, mla


def stage_slices() -> list[tuple[int, int, int]]:
    """[(first_layer, layer_count, kda_count), ...] for the 4 PP stages."""
    bounds = [(0, 24), (24, 23), (47, 23), (70, 23)]
    out = []
    for first, count in bounds:
        kda, mla = _stage_layers(first, count)
        out.append((first, count, kda))
    return out


def expert_fraction(batch: int) -> float:
    """Fraction of the 896-expert set streamed when every token routes top-16
    plus the 2 shared experts (shared are BF16, counted in the dense term)."""
    if batch <= 0:
        return 0.0
    return min(1.0, (TOP_K * batch) / EXPERTS)


def decode_stage_bytes(first: int, count: int, kda: int) -> float:
    mla = count - kda
    dense = kda * KDA_DENSE_B + mla * MLA_DENSE_B
    experts = count * EXPERT_SET_B * expert_fraction(1)
    state = kda * KDA_STATE_PER_RANK_B
    head = LM_HEAD_B if first + count == LAYERS else 0.0
    embed = 0.0  # embedding streams once per sequence, not per output token
    return dense + experts + state + head + ACTIVATION_B * count


def prefill_stage_bytes(first: int, count: int, kda: int, batch: int) -> float:
    mla = count - kda
    dense = kda * KDA_DENSE_B + mla * MLA_DENSE_B
    experts = count * EXPERT_SET_B * expert_fraction(batch)
    state = kda * KDA_STATE_PER_RANK_B * batch
    kv = mla * MLA_KV_B * batch
    act = ACTIVATION_B * batch * count
    embed = EMBED_B if first == 0 else 0.0
    head = LM_HEAD_B if first + count == LAYERS else 0.0
    return dense + experts + state + kv + act + embed + head


def _stage_ar_s(count: int) -> float:
    return AR_COUNT_PER_LAYER * AR_LATENCY_US * 1e-6 * count


def worst_decode_stage() -> tuple[int, int, int, float]:
    """(first, count, kda, bytes) of the byte-worst PP stage."""
    best = None
    for first, count, kda in stage_slices():
        byt = decode_stage_bytes(first, count, kda)
        if best is None or byt > best[3]:
            best = (first, count, kda, byt)
    return best


def calibrated_bandwidth() -> dict[str, Any]:
    """Derive the calibrated planning bandwidth from the measured anchors."""
    first, count, kda, byt = worst_decode_stage()
    ar_s = _stage_ar_s(count)
    k3_achieved = byt / ((MEASURED_STAGE_MS / 1000.0) - ar_s) / 1e9
    spread = abs(k3_achieved - QWEN36_ACHIEVED_GB_PER_S) / QWEN36_ACHIEVED_GB_PER_S
    return {
        "k3_achieved_gb_per_s": k3_achieved,          # from A1 + inventory
        "qwen36_achieved_gb_per_s": QWEN36_ACHIEVED_GB_PER_S,  # from A3
        "cross_check_spread_pct": 100.0 * spread,
        "stream_probe_fraction": min(k3_achieved, QWEN36_ACHIEVED_GB_PER_S)
        / STREAM_PROBE_GB_PER_S,
        "planning_gb_per_s": min(k3_achieved, QWEN36_ACHIEVED_GB_PER_S),
        "worst_stage_first_layer": first,
        "worst_stage_layers": count,
        "worst_stage_bytes": byt,
    }


CONVENTION_GB_PER_S = SPARK_BANDWIDTH_GB_PER_S * USABLE_BANDWIDTH_FRACTION
CALIBRATION = calibrated_bandwidth()
CALIBRATED_GB_PER_S = CALIBRATION["planning_gb_per_s"]


def project(effective_gb_per_s: float) -> dict[str, Any]:
    """All topology projections under one effective-bandwidth figure."""
    slices = stage_slices()
    ar_latency_s = AR_COUNT_PER_LAYER * AR_LATENCY_US * 1e-6

    decode_stages = []
    for first, count, kda in slices:
        byt = decode_stage_bytes(first, count, kda)
        ms = byt / effective_gb_per_s / 1e6 + ar_latency_s * count * 1e3
        decode_stages.append(
            {"first_layer": first, "layers": count, "stream_gb": byt / 1e9,
             "stage_ms": ms}
        )
    decode_model_ms = max(s["stage_ms"] for s in decode_stages)

    prefill = []
    for batch in (8, 16, 32, 56, 128, 256, 512, 1024):
        worst = 0.0
        for first, count, kda in slices:
            byt = prefill_stage_bytes(first, count, kda, batch)
            ms = byt / effective_gb_per_s / 1e6 + ar_latency_s * count * 1e3
            worst = max(worst, ms)
        prefill.append(
            {"batch": batch, "stage_ms": worst,
             "steady_state_tokens_per_s": batch * 1000.0 / worst,
             "single_prompt_latency_s": PP_STAGES * worst / 1000.0}
        )

    # TP16 (PP1): the same 16 sparks hold every layer.  The TP4-rank
    # baseline above already divides by 4, so the TP16 per-rank slice is
    # baseline / 4; the KDA state shards by heads (16-way) on the 69 KDA
    # layers only.  No pipeline fill; NCCL tree latency over 16 ranks.
    tp16_dense = (69 * KDA_DENSE_B + 24 * MLA_DENSE_B) / 4.0
    tp16_experts = LAYERS * EXPERT_SET_B / 4.0
    tp16_state_per_token = KDA_LAYERS * KDA_STATE_B / 16.0
    tp16_ar_ms = AR_COUNT_PER_LAYER * LAYERS * 15.0 / 1000.0
    tp16_decode_b = (tp16_dense + tp16_experts * expert_fraction(1)
                     + tp16_state_per_token + LM_HEAD_B / 4.0
                     + ACTIVATION_B * LAYERS)
    tp16_decode_ms = tp16_decode_b / effective_gb_per_s / 1e6 + tp16_ar_ms
    tp16_prefill = []
    for batch in (56, 128, 512, 1024):
        byt = (tp16_dense + tp16_experts * expert_fraction(batch)
               + tp16_state_per_token * batch + ACTIVATION_B * batch * LAYERS
               + EMBED_B / 4.0 + LM_HEAD_B / 4.0)
        ms = byt / effective_gb_per_s / 1e6 + tp16_ar_ms
        tp16_prefill.append(
            {"batch": batch, "stage_ms": ms,
             "steady_state_tokens_per_s": batch * 1000.0 / ms,
             "single_prompt_latency_s": ms / 1000.0}
        )

    return {
        "effective_bandwidth_gb_per_s": effective_gb_per_s,
        "decode_model_stage_ms": decode_model_ms,
        "decode_model_tokens_per_s": 1000.0 / decode_model_ms,
        "decode_stages": decode_stages,
        "prefill": prefill,
        "tp16": {
            "decode_tokens_per_s": 1000.0 / tp16_decode_ms,
            "decode_token_latency_ms": tp16_decode_ms,
            "prefill": tp16_prefill,
        },
    }


def estimate() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "measured": MEASURED,
        "model": "Kimi K3 (MXFP4 experts, BF16 spine)",
        "topology": "TP4 x PP4, 16 x DGX Spark",
        "anchors": {
            "k3_measured_stage_ms": MEASURED_STAGE_MS,
            "k3_measured_tokens_per_s": MEASURED_TOKENS_PER_S,
            "stream_probe_gb_per_s": STREAM_PROBE_GB_PER_S,
            "qwen36_floor_gb_per_s": QWEN36_FLOOR_GB_PER_S,
            "qwen36_achieved_gb_per_s": QWEN36_ACHIEVED_GB_PER_S,
            "calibration": CALIBRATION,
        },
        "convention": project(CONVENTION_GB_PER_S),
        "calibrated": project(CALIBRATED_GB_PER_S),
    }


def _print_project(label: str, r: dict[str, Any]) -> None:
    print(f"-- {label} model "
          f"({r['effective_bandwidth_gb_per_s']:.1f} GB/s/rank) --")
    print(f"  roofline stage step:     {r['decode_model_stage_ms']:.1f} ms")
    print(f"  roofline throughput:     {r['decode_model_tokens_per_s']:.1f} tok/s")
    print("  PREFILL (batch tokens per PP4 pipeline step):")
    print("    batch  stage_ms  steady-state tok/s  single-prompt latency")
    for p in r["prefill"]:
        print(f"    {p['batch']:>6}  {p['stage_ms']:>8.1f}  "
              f"{p['steady_state_tokens_per_s']:>18.0f}  "
              f"{p['single_prompt_latency_s']:>22.2f} s")
    t = r["tp16"]
    print(f"  TP16 (PP1): decode {t['decode_tokens_per_s']:.1f} tok/s, "
          f"token latency {t['decode_token_latency_ms']:.1f} ms")
    for p in t["prefill"]:
        print(f"    B={p['batch']:>5}: {p['stage_ms']:>8.1f} ms  "
              f"{p['steady_state_tokens_per_s']:>6.0f} tok/s  "
              f"latency {p['single_prompt_latency_s']:.2f} s")
    print()


def _print_human() -> None:
    c = CALIBRATION
    print("ESTIMATE ONLY - NOT A MEASUREMENT "
          "(bandwidth calibrated to measured anchors)")
    print("Kimi K3, TP4 x PP4, 16 x DGX Spark")
    print()
    print("MEASURED ANCHORS:")
    print(f"  A1 K3 warm B1 stage step:     {MEASURED_STAGE_MS:.1f} ms "
          f"({MEASURED_TOKENS_PER_S:.1f} tok/s), sparka, stage 0")
    print(f"  A2 DRAM stream probe:         {STREAM_PROBE_GB_PER_S:.1f} GB/s "
          "@256 MB (bw_probe.cu, sparkb)")
    print(f"  A3 qwen36 band achieved:      {QWEN36_ACHIEVED_GB_PER_S:.1f} GB/s "
          f"(= {QWEN36_FLOOR_GB_PER_S:.1f} x {QWEN36_PLAIN_FRACTION:.2f})")
    print(f"  calibration: K3 path implies {c['k3_achieved_gb_per_s']:.1f} GB/s "
          f"vs qwen36 {c['qwen36_achieved_gb_per_s']:.1f} GB/s "
          f"(spread {c['cross_check_spread_pct']:.1f}%); planning value "
          f"{c['planning_gb_per_s']:.1f} GB/s "
          f"= {100 * c['stream_probe_fraction']:.0f}% of the A2 stream probe")
    print()
    _print_project("convention", project(CONVENTION_GB_PER_S))
    _print_project("calibrated", project(CALIBRATED_GB_PER_S))
    print("Notes: the calibrated model reproduces the measured B1 stage step "
          "by construction; use IT for run-day accept windows and keep the")
    print("convention row only as the historical roofline. Expert stream "
          "saturates at B=56 (896/16); the fp32 KDA state read+write is")
    print("per-token and dominates large batches; PP4 bubbles count in the "
          "single-prompt latency only; TP16 trades the pipeline for ~4x")
    print("lower single-token latency at about the same throughput.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="emit the estimate as JSON")
    args = parser.parse_args(argv)
    if args.json:
        print(json.dumps(estimate(), indent=2, sort_keys=True))
    else:
        _print_human()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
