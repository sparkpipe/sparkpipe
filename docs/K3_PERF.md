# K3 serving-path performance audit and improvement plan

## 2026-08-17: the tail nondeterminism root cause (FIXED)

The fresh-run gate's known tail variance (hidden[6144..7167], ~14 ULP,
across runs AND across the TP4-vs-full equivalence) was the KDA o_proj
running its weight-only GEMM with output == input == attention_out_bf16.
The persistent GEMM stages A per k-tile while storing finished output tiles
into the output buffer, so a second-wave tile's A reads race the first-wave
stores of columns 0..7167 and its k-sums pick up another tile's outputs.
Localised by per-phase dumps: q/k/v, gate, decay logits, retention, beta
and the state pool were bit-identical across processes while the o_proj
output moved (8 runs, 6 distinct outputs) - and the pre-norm and post-gate
attention were bit-identical, which is what pinned the race to the
projection itself. Fixed in layer.cuh (o_proj lands in hidden_bf16, which
the MLP-side retrieval overwrites next; the tp1 boundary partial-set and
the TP4 phase-0 hook source follow the layer kind - the MLA path already
used distinct buffers). The gate's 64-ULP tail exemption is gone: fresh-run
determinism and capture fidelity now hold at 4 ULP everywhere, and the TP4
4-rank sum equals the full-stage run to bf16 rounding.

The earlier "plain BF16 GEMM breaks at K >= 512" was a test artifact, not
a kernel bug: the gate's plain probe filled its activation pattern with
(((int32_t)x % 11u) - 5), whose subtraction happens in UNSIGNED space and
wraps the intended negatives to ~4.29e9 (bf16 0x4E80). The GEMM was summing
poisoned test data faithfully - verified byte-exact across processes at
every real K extent (7168/12288/33792) by the determinism probes.

## 2026-08-23: equivalence re-proven on the classification-fixed sharder (#667 items 1a/1b/3 closed)

With the tensor-classification fix on `unified` (`2bc40c4`: `kda_decay_bias`
and `kda_head_log_scale` are HEAD_1D, `routed_norm_weight` LATENT_1D —
per-rank slices, not replicas), the layers-0-3 slice pack was re-sliced and
both post-fix receipts were re-taken on spark0 (GB10, nvcc 13.0,
sm_121a, tree `229526d`). The pack is `/tmp/k3_slice_0_4.pack` on spark0
(55.3 GB, manifest `first_layer: 0, layers: 4`); the fixed sharder cut it
into **87 tensors x 4 rank packs** (~13.9 GB each).

- **TP4-vs-full equivalence PASS.** Run in the gate's canonical form: four
  concurrent single-spark processes on 127.0.0.1 with their true `--tp4 <r>`
  rank (the real host-tier all-reduce), each dumping its POST-AR hidden,
  compared against a full-pack tp1 run via
  `tools/k3_tp4_equivalence_check.py --direct`. All four rank dumps are
  bit-identical to each other; against the full run, **worst relative
  deviation 0.02158 at column 2441, 0 of 7168 columns beyond the 0.03
  tolerance** (max absolute delta 0.00586 ≈ 3 BF16 ULP at that magnitude;
  1647 columns bit-exact). Operational note for re-runs: the sum-of-tp1-
  dumps mode is NOT valid with vocab-sliced embeds unless every leg runs
  its own true tp_rank — the runner offsets the embed row by
  `tp_rank * vocab_slice_rows`, so tp1 legs with `tp_rank=0` on ranks 1-3
  read the wrong token row and the offline sum diverges catastrophically
  (observed worst rel 1.97 before switching to `--tp4`).
- **Capture fidelity re-proven on the fixed path.** The single-spark step
  gate (`tests/test_k3_runner_step.cu`) PASSES on both the full pack and a
  rank pack: fresh-run determinism **0 mismatches beyond 4 ULP**, and the
  no-capture direct step-2 vs graph replay comparison at the same recurrent
  state is **0 mismatches** (full-pack warm replay 60.3 ms, rank-pack warm
  replay 13.2 ms). The fidelity failure #667 recorded was the GEMM bugs'
  symptom; it does not recur.

## Prefill + decode performance estimate

tools/k3_tp4pp4_perf_estimate.py derives both numbers from the deployed
rank-pack manifest inventory (per-layer tensor bytes) and reports every
projection under TWO bandwidth models, so the analytical/measure boundary
stays visible:

- **convention** (historical roofline): 273 GB/s x 0.65 = 177.45 GB/s.
- **calibrated** (planning model): anchored to three measured points on this
  hardware class - (A1) K3's own warm B1 stage step, 55.5 ms on sparka;
  (A2) `tools/devcycle/bw_probe.cu` DRAM stream 250.7 GB/s @256 MB on
  sparkb; (A3) the qwen36 TP4 band's measured device read bandwidth
  225.9 GB/s with plain B1 at 72% of its weight floor = 162.7 GB/s achieved.
  A1 + the byte inventory implies **155.4 GB/s** achieved on K3's mixed
  BF16-spine + MXFP4-expert stream; A3 independently gives 162.7 GB/s -
  two models, same hardware class, within 4.5%, both at ~62% of the A2
  stream probe. The calibrated planning value is the conservative one
  (min), and it lands the B1 decode projection ON the measured step by
  construction.

Projections (convention -> calibrated):

- Decode (output, B1): measured 18.0 tok/s (55.5 ms/stage); roofline
  20.6 -> 18.0 tok/s - under calibration the path IS its roofline, i.e.
  B1 decode is bandwidth-bound with no modeled headroom left; only byte
  reduction or speculation moves it (the ~32 tok/s absolute no-spec
  ceiling at 100% of 273 GB/s still bounds any kernel work).
- Prefill: 92/359/1537 tok/s at B8/B128/B1024 under the convention;
  **81/315/1346** calibrated (single-prompt latency 0.40 s -> 3.04 s at
  B1024). The expert stream saturates at B=56 (896/16) and the fp32 KDA
  state stays the dominant large-batch term (52% of B1024 bytes), so the
  BF16-state lever prices out at roughly +38% under either model.
- TP16 (PP1): decode 20.2 -> **17.8 tok/s** at 56.1 ms token latency
  (~4x lower latency than the pipelined TP4xPP4 at the same throughput);
  prefill parity (1323 tok/s at B1024 calibrated).

AR-latency sensitivity: the model uses 8 us per tree all-reduce; qwen36
measured ~27 us/op on its hidden-transport tier. Even at 27 us the
whole-stage AR budget is ~1.24 ms against a ~55 ms step (~2%), so the
projections are insensitive to the AR constant - they are sensitive to the
bandwidth term, which is why that term is the calibrated one.

Measured on sparka, real rank pack, stage 0 (24 layers), 1 token:

- Cold first step: ~2.5 s - one-time JIT, tensor-map encodes, shared-memory
  opt-ins. Amortises to nothing in steady state.
- Warm step: **55.5 ms** = 2.3 ms/layer. The slice issues ~35 kernels per
  layer (~840 per step, ~66 us of host enqueue each - the HOST spent ~55 ms
  enqueueing each step).
- Graph replay of the captured slice: **54.2 ms** (step 3 in the gate) -
  bit-identical output. The kernels themselves take ~54 ms, so the path is
  MEMORY-BANDWIDTH-BOUND on weight streaming, not launch-bound: each layer
  streams ~0.8-1 GB of BF16 spine + MXFP4 expert weights per token through
  the persistent GEMMs (LPDDR5X ~273 GB/s => ~2.3 ms/layer). The capture's
  win is real but wall-clock-side: it removes the ~55 ms serialized host
  enqueue from every submit, roughly halving the per-step wall time at B1
  and freeing the host thread for the residentd loop and the other ranks.
  Steady-state decode: ~55 ms per stage per token (the PP4 stages
  pipeline), i.e. ~18 tokens/s for the 16-spark K3.

## Improvements, in dependency order

1. **CUDA-graph capture** (the slice's own capture audit: legal once no
   host traffic remains on the layer path). The per-step dense-offset H2D
   is now a device-side kernel (K3RunnerDenseOffsetsKernel); the remaining
   host traffic is the collective tier's staging. Target: the ~3.3k
   launches/token become one graph replay per shape.
2. **The fused per-layer collective (LANDED)**: attention_out | hidden |
   shared_out pack into contiguous 14 KB segments instead of three
   separate exchanges - the two-phase split (item 5) ships attention_out
   alone before the MLP-side retrieval and hidden | shared_out after the
   MLP, so each exchange carries the largest frame its phase allows and
   the wire utilization stays high. The host-TCP tier implements it; the
   device tier replaces the whole block with stream-ordered combines (no
   sync at all).
3. **The device-direct tier (LANDED)**: the adapter parses a
   `device_collective` object (backend nccl | hidden_transport, peer
   hosts, ports, timeouts), applies the topology, and hands the runner a
   completed config. The runner creates the collective with the K3 combine
   kernels (hidden transport) or plain NCCL, and the per-layer hook issues
   ONE stream-ordered all-reduce of the fused 3x7168 buffer with a
   completion that folds the summed segments into the AttnRes partial on
   the same stream. No sync, no host staging, no H2D on the hot path.
   NCCL's unique id bootstraps through the shared control port (64620);
   the host TCP tier stays available as the TP4 fallback and is skipped
   for TP16 (its 4-rank cap).
4. **TP16-ready geometry (LANDED)**: the adapter derives the PP stage
   split from `world_size / tp_degree` (16/16 -> PP1), the module takes
   slice bounds from the pack manifest when the runner passes
   SPARK_K3_MODULE_DERIVE_SLICE, the bound-layer cap is 93, and the
   generator emits TP16 configs (16 peer hosts, no host tier). The degree
   divisibility audit below holds.
5. **Expert sharding, corrected (LANDED)**: two real findings from the
   audit. (a) The w1's K AXIS must slice too: the rank's latent slice
   addresses only its k-tiles, so the old whole-k shard paired ranks 1-3's
   activations with rank 0's weights - the sharder now takes the rank's
   k-tile range on w1 as well as the cell range. (b) The TP16 tile size
   is 32, not 64: the rank's SiTU intermediate slice IS contiguous (the
   gate|up halves share cell offsets), and 32 divides both the w1 k-slice
   (224 = 7 x 32) and the w2 k-slice (192 = 6 x 32); the packer's
   interleave_geometry already closes at tile_k 32 (16B payload row = 16
   rows x 1 scale byte). The packer now takes expert_tile_k (128 default,
   32 for TP16 packs) and the sharder refuses any degree the tile counts
   do not divide, naming the tile size. The TILE_K=32 INTERLEAVED_B GEMM
   variant is the remaining code wave; until then TP16 packs cannot be
   consumed by the serving tier. NOTE: the deployed TP4 rank packs were
   sliced by the pre-fix sharder and must be RE-SLICED before the
   end-to-end run (only rank 0's w1 is correct today).
5. **Two-phase layer collective (LANDED)**: the hook now fires after the
   attention half (phase 0) AND after the MLP half (phase 1). Phase 0 is
   required for correctness, not just speed: the MLP-side AttnRes retrieval
   is documented to read the POST-attention partial, and the old single
   post-MLP hook let the sharded path retrieve a partial missing the
   attention contribution. Each phase packs and all-reduces its own
   segment(s) with the fold landing on the submission's stream (no legacy
   default stream), and at tp_degree 1 the hook no-ops because the layer
   folds its own projections.
6. **Per-shape CUDA-graph capture (LANDED, gated)**: the runner captures
   the dense-offset kernel + the whole slice into a cudaGraphExec_t keyed
   by rows (warm-direct first submit, capture on the second, replay after),
   gated on a non-default stream and a capture-safe tier (NCCL device
   collective or tp_degree 1; the host tiers' syncs/staging are not
   replayable and self-disable the path). The slice audit (in slice.cuh)
   held: no host traffic on the layer path, host-resolved branches bake
   in, tensor-map encodes are steady-state cache hits. The single-spark
   gate now replays the captured slice and compares it against the direct
   launch (0 mismatches beyond the ULP limits).
7. **AR overhead audit, 2026-08-16 (second pass)**: the per-phase
   payloads are now SIZED to the phase (the submission carries the element
   count - phase 0 ships ONE 14 KB segment instead of the 3-segment 43 KB
   frame, cutting its wire bytes 3x), and the embedding exchange runs on
   the NCCL tier (one slot-encoded stream-ordered all-reduce; no sync, no
   host staging). Remaining, in order of value: (a) the slot-encoded
   full-width all-reduce moves 4x the minimal bytes - a reduce-scatter +
   all-gather pair would halve the per-rank wire traffic; (b) the head
   exchange still uses the host tier (its f32 slots have no NCCL f32
   collective - a bf16-splittable slot layout or an f32 NCCL op would move
   it); (c) the hidden-transport tier cannot narrow its pre-registered
   frame (the per-submission count is NCCL-only). The per-layer structure
   stays 2 ARs on the critical path (the correctness requirement), so the
   B1 AR budget is 2 x NCCL-tree latency (~5-15 us) per layer.
8. **Overlap** (revised): the per-layer ARs sit on the critical path for
   B1 decode by construction - the MLP-side retrieval needs the summed
   attention output and the next layer needs the summed MLP outputs - so
   there is no compute to hide them behind. The AR-overhead reductions are
   the fused messages, the device-direct tier, and the graph capture
   above. Remaining ideas: a per-submission payload width (the collective
   currently fixes local_hidden_dimension at create time, so a 1-segment
   phase still ships the 3-segment frame) and the balanced TP16 w2
   half-tile repack.

## TP16 readiness audit (degree 16 divisibility)

| tensor | full | /16 | status |
| --- | --- | --- | --- |
| heads (KDA + MLA) | 96 | 6 | ok |
| vocab (embed/lm_head) | 163840 | 10240 | ok |
| kda_out input | 128 | 8 | ok |
| mla_out input | 12288 | 768 | ok |
| routed latent | 3584 | 224 | ok |
| expert w1 output (cells) | 6144 (384) | 384 (24) | ok |
| expert w2 k-tiles | 24 | 1.5 | was REFUSED - now UNBALANCED SPLIT |
| shared w1 halves | 1024 | 64 | ok |
| shared w2 input | 2048 | 128 | ok |
| dense gate_up halves | 16896 | 1056 | ok |
| dense down input | 33792 | 2112 | ok |

The sharder now splits the w2 across TP16 UNBALANCED (the first 8 ranks
take 2 of the 24 k-tiles, the rest 1) - whole tiles, scales intact, the
per-rank dims the layer already reads make it explicit. The balanced
follow-up is the 64-element half-tile repack (a pack-format change the
GEMM would take as an INTERLEAVED_B TILE_K=64 variant with 32-byte cell
rows).
