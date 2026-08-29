# K3 serial-TP16: per-projection rework plan (embedding fixed; per-layer AR remains)

Status: the TILE_K=32 fallback + w2 fix unblocked the compute (replay runs
end-to-end). The embedding defect is FIXED in this diff (tp_rank = rank, so each
rank's embed slot offset is rank * vocab_slice_rows and an out-of-slice token
contributes zero). Re-run verdict with the fix: **6410/7168 columns beyond 0.06
rel, worst 1.278** (was 6683 / 2.619) — the embedding fix helped, and the
remaining error is exactly the intermediate-layer input-replication gap.

## Why the single whole-slice sweep cannot validate TP

K3's dataflow is: each layer reads the FULL (all-reduced) hidden as its input
(routed_down and every attention projection read normed_bf16 = hidden), computes
its rank slice, and the input-dimension-sharded projections (kda_out, mla_out,
routed_up, shared_w2, dense_down) emit a PARTIAL that the per-layer all-reduce
sums back into the full hidden. A single "run all 4 layers per rank, sum the
final hidden" sweep therefore feeds each layer's N+1 the rank's own partial, not
the sum. The harness's intended shape (docs/serial_tp_replay.md) is a
per-projection sweep + host all_reduce_sum BETWEEN projections, not a
whole-slice sweep.

## The AR points (9 for the 4-layer slice, not ~140)

- Embedding (slot-encoded sum) — FIXED by tp_rank = rank.
- Per layer: after the attention half (kda_out / mla_out partial) = phase 0.
- Per layer: after the MLP half (routed_up + shared_w2, or dense_down) = phase 1.

The runner already fires these two per-layer points through
`layer_collective_override` (phase 0/1), so the AR granularity is 2 per layer,
not per projection — the ~35 intra-half kernels are head-sliced (each rank owns
its heads) and need no collective.

## Required runner change: a "run one layer" mode

`K3LaunchSlice` (slice.cuh) runs first_layer..first_layer+layer_count in one
loop, maintaining the AttnRes bank + running partial + recurrent KDA state in
the K3SliceState/K3LayerBuffers. The rework adds a step mode that runs exactly
ONE layer per call with:

1. **explicit full hidden input** — the layer reads the full (host-summed)
   hidden, not the rank's partial. Reuse the PP boundary field
   `SparkK3StageRunnerDispatch.hidden_input_bf16` (already in the ABI, used by
   stages 1..3); stage-0 rank legs set it to the summed embedding.
2. **AttnRes state carry** — bank[0] = the summed embedding, the running partial
   = the summed output of the previous layer; both live in the K3SliceState and
   must persist across per-layer calls on the SAME rank (the runner already
   keeps recurrent state across submits; the bank/partial must be exposed or
   re-derived from the summed hiddens).
3. **phase-0/phase-1 partial capture** — the hook (or a step callback) copies
   the rank's attention partial (phase 0) and MLP partial (phase 1) to host so
   the harness can `spark_serial_tp_all_reduce_sum_bf16` them and write the
   summed hidden back for the next layer.

## Harness wiring (test_k3_serial_tp.cu)

For each layer L in 0..3:
1. `spark_serial_tp_sweep(16, K3_HIDDEN, full_hidden, ...)` — each rank runs
   layer L (one layer, attention half + phase-0 partial + MLP half + phase-1
   partial), writing its partial hidden.
2. `spark_serial_tp_all_reduce_sum_bf16(partials, 16, K3_HIDDEN, full_hidden)`
   — reconstruct the full hidden.
3. full_hidden becomes layer L+1's input; the golden compares full_hidden after
   layer 3 against the full slice run.

The embedding sweep is step 0 (slot-encoded, tp_rank = rank), summed once.

## Deliverable state now

- Embedding fix: applied (test_k3_serial_tp.cu, tp_rank = rank). Re-run shows
  6410 columns / worst 1.278 rel — the remaining error is the per-layer AR gap.
- The runner per-layer mode (items 1-3 above) is the remaining work; it is a
  slice.cuh/runner change, not a harness-only change, and I did not land it in
  this turn.

## Next step

Implement the runner per-layer mode (or an equivalent "capture phase-0/1
partials + replay with summed inputs" two-pass driver) so the harness can sweep
layer-by-layer. With that, the 17-leg replay becomes ~(1 embed + 8 per-layer)
sweeps and the sum-vs-golden should close at ~0.06 rel, matching the
K3_PERF.md golden tolerance.
