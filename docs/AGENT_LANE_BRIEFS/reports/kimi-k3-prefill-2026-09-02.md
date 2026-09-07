# kimi-k3 — TP16 wave-ready + the prefill bug fixed (2026-09-02)

Operator directive: "a lot has changed in main; keep pushing non-speculative
inference speed; when the kimidev fixes the prefill bugs, you can also hill
climb prefill speed."

## The prefill bug, precisely (PR #766)

The engine ships multi-row prefill spans (max_prefill_rows =
runtime_limits.max_input_row_count, model_api.c:1044) and the shared
recurrence kernels are run-aware (sequence_row_begin prefix; state_index
indexed per SEQUENCE; the K3 step-input doc comment states the contract).
The K3 adapter defeated all of it: every row of a span was mapped to the
same resident slot and submitted with sequence_row_begin NULL and
sequences=rows — a T-token prefill frame ran T independent 1-row runs
against ONE KDA state slot (last-writer state; every row's output from the
stale pre-frame state; conv windows the same). Prefill numerics were wrong
for any frame wider than one row; that is the bug.

The fix is pure wiring (the ledger's PATTERN A), +73 lines:
- adapter derives the run prefix (consecutive same-slot rows = ONE chained
  run; decode batch = runs of one) + per-run state slots; fails loud
  (VALIDATION_FAILED) if the derived run count contradicts the
  submission's declared active_sequence_count;
- runner honors active_sequence_count + sequence_row_begin (validating
  count<=rows and prefix-requires-count).

Decode is untouched for distinct-slot rows (runs-of-one through the
explicit-prefix path is the kernels' own bit-exactness contract — flash
tier4a hardware receipt). Compile receipts: runner sm_121a gate PASS,
adapter clean build on sparke.

**Gate owed before a prefill perf claim:** the GPU tier3/tier4 equivalence
cell (N-row run vs N sequential 1-row waves through the live layer chain),
queued into the wave window — the flash lane's precedent. The host harness
(k3_layer_host.cu) is a dataflow probe with recorded GEMMs and cannot carry
a numerics claim.

**Unlocked next:** prefill width is a pure knob now (deployment
max_input_row_count; 16-row frames today; the 1024-row chunk bump with a
memory recheck is the wave-time follow-up), then prefill hill climb.

## TP16 is wave-ready (state consolidation)

- Base pack: recovered after the 08-30 build death at 47% (keepalive itself
  died with my interrupted ssh session — lesson: remote supervisors must
  not descend from an agent's ssh; use the queue). The recovery assembled
  the full 1,562,379,187,072-byte pack (journal-resumed + one inserted
  5.2 GB layer-65 segment, sha-insert receipt), ceph read-back verified,
  PUBLISHED to /mnt/model-warm/packbuild/k3_tp16base.pack
  (sha256 b74328a1...). Receipts: spark8:/home/spark8/k3-recovery/.
- Rank packs: ALL 16 deployed (99,566,844,288 B uniform) and
  coordinator-audited 09-01 (one stale Aug-25 rank01 found and replaced).
- Runtime trees: staged on all 16 nodes tonight
  (k3_stage_runtime.sh ... 16) — residentd 503,368 B, adapter 6,917,648 B
  (certified-FP8 screened head + this prefill port), hidden transport
  203,360 B; TP16 adapter configs (no host tier, 16 NCCL peers) + the
  generated deployment. spark0's earlier ssh banner-timeout recovered.
- Remaining for the wave: the exclusive 16-node window via the queue, then
  wave → B1 decode + the head-equivalence cell + the prefill equivalence
  cell; re-run k3_fleet_wave.sh check first (it enforces the memory
  envelope incl. spark2/3 MDS rule).

## Environment notes for the next pass

- The shared checkout's lane/kimik3-dev ref moved under this session
  (another agent pulling main into it / resetting the remote); this PR
  rides lane/kimik3-prefill off current main. Future kimi work: work on a
  fresh branch off main, never assume the lane ref.
- test_dsv4_serving_adapter aborts at current main in this environment
  (baseline receipt: fails identically with my edits stashed) — the dsv4
  lane or coordinator owns that; everything before it in offline-gates
  passes.
- Ratchet 237406 exact (tracked-file arithmetic; the untracked venv in
  this checkout still defeats the local code_size walk).
