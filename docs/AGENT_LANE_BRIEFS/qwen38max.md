# Qwen 3.8 Max (4-bit) lane brief — SOTA plan (re-orientation 09-07)

Model: Qwen3.8-2.4T-A95B, AMD Quark MXFP4 checkpoint (verbatim E2M1+g32
E8M0 experts, BF16 spine), 92 layers 3:1 GDN:attention, MoE 512+1
top-10. Deploy target: TP16 single stage, expert-sharded (32
experts/rank), attention kv-replication (4 kv heads over 16 ranks).
Branch lane/qwen38max-sota (current main 862a2fe).

## Where things are after the reset

Landed on main since the lane paused:
- kv-replication TP16 attention (macros + kernels + packer kv axis,
  harvested from #754; CPU-proven byte-exact) + the #777 tp1 guard fix
  and module link fix (merged).
- The qwen38_max module harvest repair (f01350c + cb3def9).
- weightd resident stack (W1-W3), UPDATE-file release protocol,
  repo-driven build/release script.
- The tree-allreduce world: tree engine (#807/#809) + #816 TP16
  allreduce — honest floor 147us sync/op (~140us fixed CPU protocol),
  d2a path 78us sync / 52us async pipelined, ack flow control, the
  nonce-offset u64 root cause fixed. Fleet verdict pending one clean
  run. Mimosa commit wall cleared; queue empty; fleet FREE.

Broken/missing on main (the lane's outstanding work):
1. VALIDATOR DRIFT: 8 stale launcher signatures in the qwen38max
   validation .cu (main dropped tp_degree/expert_mxfp4 params;
   MoeRoute's sits mid-signature -> kernel derefs 0x1 -> the moe_mxfp4
   `cuda invalid argument` reding the family GPU gate). Fix proven on
   my tree, never landed (Mimosa wall then).
2. moe_mxfp4 check stale: builds a tp4-shard expert view; main's
   wrapper requires the FULL 512-expert view. Fix proven on GPU
   (1.39e-3 / 0.999999034, S1-identical).
3. gdn_step_tp4 stale premise (kernel is global-geometry now): needs
   the dated honest skip; re-lands with GDN head-split.
4. head-split attn-slice loader (design branch): NOT on main. K/V row
   slices + strided o_proj column slices + the attention output
   all-reduce; unit-proven byte-exact (40/40) on the old lineage.
5. module_decode `nonfinite` at the module tier: unresolved — the A/B
   control never ran (fleet held, then the timer stopped). WIP module
   edits are tp1-byte-identical, so the control decides pre-existing
   vs introduced.
6. TP16 pack set: never built on the current lineage (the checkpoint
   is fully downloaded; the packer's tp16 kv axis is harvested).
7. Serving deployment tree: generator + publish recipe known, never
   materialized.

## The SOTA plan

SOTA = measured aggregate decode tok/s on the 16-node TP16 tree-
allreduce world, hill-climbed with exactness-first discipline.

Collective arithmetic (the binding constraint): ~92 MoE-delta + ~23
attention-output all-reduces per token = ~115 ops/token. At d2a 78us
sync the B1 floor is ~9ms/token (~110 tok/s ceiling); async pipelined
52us hides under expert compute at B8+. So B8+ is where SOTA lives,
and the climb is collective-op-count and expert-gather bound.

- M0 (now, fleet free): back-port the validator fixes (drift, full-
  width moe, tp4 skip) onto main; run the module_decode control cell
  on spark3. Un-reds the family gate; answers nonfinite. ~1 cell.
- M1: build the 16 TP16 rank packs from the AMD checkpoint with the
  current packer (96.2 GiB/rank, fits 110 GiB with the sliced spine;
  byte-exact verify per rank; place one per node). CPU work.
- M2: serving tree per node (generator + publish recipe + weightd),
  first 16-rank module load, B1 decode → nonfinite resolved in the
  real config; then the 8-token exactness gate.
- M3: rebase the head-split loader (K/V + o_proj slices, all-reduce
  placement) — required for attention compute to scale at TP16; its
  KV-cache shrink (4x vs TP4-era) also lifts the B-scale ceiling.
- M4: baseline ladder B{1,8,32} with telemetry+nvidia-smi receipts;
  profile where the 78us/52us collectives land against expert compute.
- M5: hill climb, in profile order: (a) collective shape — fuse/
  batch the per-layer reduces, bigger rows per op (d2a async), (b)
  expert-gather kernel width and tile shapes, (c) KV page + attention
  config, (d) speculation last (MTP-1 exists; the 5090 draft world +
  payoff probe #814 sizes it after non-spec is proven).

Kill-switch discipline unchanged: exactness before timing, mismatch =
RED stop, numbers recorded with context/batch/topology.
