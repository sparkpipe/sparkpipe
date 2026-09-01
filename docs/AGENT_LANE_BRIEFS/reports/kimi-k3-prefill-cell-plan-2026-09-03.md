# kimi-k3 prefill-equivalence cell — plan (2026-09-03)

PR #772's publish gate, scoped to a SINGLE-NODE queue cell (no exclusive
16-node window needed):

## The insight

The TP16 base pack (k3_tp16base.pack, full-width tile-32, 1.56 TB) is
UNSLICED — a tp_degree=1 runner run over it produces full-model
numerics (mmap-streamed; the TP4 equivalence ran this shape on a 52 GB
slice pack). So the run-vs-sequential comparison runs on ONE spark via
the queue, ahead of/parallel to the wave:

- host: spark8 (holds /home/spark8/k3-recovery/k3_tp16base.pack
  .recovered.local, sha b74328a1, and 2.4T free; nvcc CUDA 13)
- build: the cell binary compiles the k3 module sources directly
  (tools/k3_runner_compile_gate.sh's nvcc line + the runner .cu), or
  links against the staged adapter if simpler

## The cell program (tests/host-side pattern of test_k3_runner_step.cu)

1. init: SparkK3StageRunnerInitialize, tp_degree=1, PP1 derive-slice
   (93 layers), the full pack.
2. inputs: deterministic prompt of T tokens (T = 8, 64; LCG seeds), one
   KDA state slot.
3. PATH A (run form): ONE submit — rows=T, active_sequence_count=1,
   sequence_row_begin={0,T}, kda_state_index={slot}, positions p..p+T-1,
   context=pos+1. Record ALL per-row output tokens.
4. PATH B (sequential form): fresh runner state (or reset slot), T
   submits of rows=1 at the same positions/slot. Record.
5. COMPARE bit-exact per row + the final continuation token. Mismatch =
   RED stop (report, no timing).
6. REGRESSION leg (S2 analog): rows=8 DISTINCT slots, explicit
   runs-of-one prefix vs NULL-prefix dispatch — token-identical.
7. OPTIONAL timing AFTER exactness: wall-clock A vs B (the prefill
   speedup receipt — 16-row frames vs 16 dispatches).

## Queue shape

    k3-prefill-equiv-1  [spark8]  gpu  ttl 30  by lane-kimik3
    cmd-file: stage cell binary + run + emit receipt json
    (exactness verdict FIRST, timing only on PASS)

## Prereqs

- the cell binary (next unit; harness = tests/test_k3_runner_step.cu)
- #772 merged or its branch checked out on the build host (the runner
  dispatch carries sequence_row_begin)
- drop_caches before the 1.56T mmap pass (110 GiB law)
