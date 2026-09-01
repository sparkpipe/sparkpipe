# qwen38max non-spec decode ladder — the cell plan (2026-09-01)

The runbook-shaped experiment queue for this lane. Every cell: kill-switch
discipline (exactness verified BEFORE any timing; mismatch = RED stop,
not a data point), numbers recorded with context/batch/topology, weights
resident between cells (weightd), one variable per cell.

- **R1 — no-regress receipt** (`q38max-tp16k-regress2`, queued on
  spark8 behind the glm53full p0 wave): TP1 module tier + TP4 kernel
  tier on the merged #765 lineage; expect identical PASS to pre-change.
- **R2 — head-split equivalence cell** (4 live ranks, after the
  head-split PR's GPU gates): 4-layer synth pack, module decode at
  tp4 with load-time slices vs the tp1 oracle — KV cache 4× smaller,
  o_proj sliced, all-reduce in place; bit-exact expectation vs the
  oracle's tolerance class.
- **R3 — TP16 single-stage load cell** (16 nodes; NEEDS the residency
  ruling on #765, option (a)): real AMD pack, load-time expert-row
  slice + attention slices; receipts = load time, resident MiB/rank
  (must clear 110 GiB with headroom), first token.
- **R4 — B=1 decode baseline**: telemetry curl 127.0.0.1:8765 +
  node-local nvidia-smi pair; sustained generation (bursty runs hide
  from the 5s poll). This is the lane's first honest tok/s number.
- **R5 — hill climb**: batch width {1,8,32} first (the fleet's known
  B-sensitivity), then kernel variants / launch shapes; screened-head
  and split-K settings from the dormant-capabilities audit apply here.

MTP coverage note: BindMtp fills the mtp attention views from the same
sliced entries — rank-local shapes consistent with the module by
construction, and the MTP path is dormant anyway (operator ruling:
accuracy first, drafter wiring last). GPU cells will confirm.

Blocked-on: R1/R2 need fleet nodes (glm53full p0 wave running); R3+
need the #765 residency ruling.
