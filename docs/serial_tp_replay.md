# Serial-TP replay harness (correctness while the fleet is down)

Goal: prove a sharded resident stage is CORRECT on one spark by replaying its
tensor-parallel plan serially — one shard resident at a time, the collectives
emulated host-side — and comparing the final output against a golden reference
(token ids or a logits hash). Slow by design; it only has to be right and fit
the node.

## Placement + gates

The model-neutral core lives in `tests/` (`serial_tp_replay.{h,c}`) so it is
excluded from `test_code_size.py` (`tests/` and `docs/` are excluded,
`test_code_size.py:204-205`) and outside `test_dry_law.py`'s shared dirs
(`test_dry_law.py:12-27`). It is still token-free (no model name, no model
geometry) so it could move under `inference/kernels` later without tripping
dry-law. The only counted-source delta is the Makefile test wiring (+2 lines).

## What the core owns (model-neutral)

- `spark_serial_tp_sweep` — the serial loop: for `rank = 0..tp-1`, load the
  shard (budget-checked against the 108 GiB cap), run it, capture the partial,
  free the shard. Exactly one shard is resident at a time
  (`serial_tp_replay.c` sweep).
- Host collective emulation, pure C + deterministic:
  `all_reduce_sum_bf16`, `reduce_scatter_bf16`, `u64_maxloc` (fp32 sum,
  round-to-nearest-even bf16). These stand in for the device collective while
  it is absent.
- Golden compare: `compare_exact` (token ids) and `hash_elements` (FNV-1a
  64 over element bytes, for logits-hash goldens).

## What the caller wires (per model, out of the core)

The dsv4-pro and k3 agents implement the four `SparkSerialTpRankHooks` on
their module API:

1. **load/free/shard_device_bytes** — open the rank-r shard through the module
   load path and report its device bytes for the budget.
2. **run_rank** — execute the stage with the existing collective no-op so the
   reduces become no-ops and the PARTIAL output is captured. The reference
   pattern is the `TP_STANDALONE` escape hatch in
   `modules/qwen36_resident_decode_stage/source/spark_qwen36_tp.c:177-181`:
   when the env var is set, `SparkQwen36TpInitialize` returns early
   (`spark_qwen36_tp.c:177-182`) and `SparkQwen36TpReduceHidden` is already
   a no-op because `initialized == 0` (`spark_qwen36_tp.c:385-386`). The
   same shape is expected on the other families' TP state.
3. **the TP plan** — a list of sweeps + host collectives (`all_reduce_sum`
   after a row-parallel down/o projection, `reduce_scatter` after a
   column-parallel shard, `u64_maxloc` for the head argmax shard), expressed
   by calling `spark_serial_tp_sweep` then the matching emulation.
4. **golden** — token ids or a logits hash from the caller's reference run.

## Summation-order contract (a correctness footgun, stated once)

The host emulation sums the ranks in rank order in fp32 and rounds once to bf16.
The device combine folds each rank into the consumer buffer in its own order
(e.g. `SparkQwen36LaunchAccumAdd`, `spark_qwen36_tp.c:42-46`). If the golden
is bit-exact logits, the caller must make its `run_rank` partials reproduce the
device reduce order, or accept the hash comparison at the reduced precision.
This is the same accumulation-order contract the delta-rule kernels already
carry (`inference/kernels/linear_attn.cuh:429-435`).

## Deliverables

- `tests/serial_tp_replay.h` / `.c` — the shared core (host paths).
- `tests/test_serial_tp_replay.c` — pinning + self-test binary (builds with
  host `cc`, no nvcc); proves the host half the model agents build on.
- CUDA wiring (shard load + TP_STANDALONE no-op) is per-model and lives in the
  model agents' modules; it CI-compiles under the sm_121a compile gate when
  they wire it, not here (the fleet is down).
