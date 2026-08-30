# Qwen 3.8 Max (4-bit) lane brief

Model: Qwen/Qwen3.8-2.4T-A95B via AMD Quark MXFP4 checkpoint (routed
experts E2M1+g32 E8M0, everything else BF16). Topology 16-node
TP4xPP4. Branch lane/qwen38max-dev (from main 879fc5a). Identity:
sparkpipe PAT via tools/sparkpipe_github_pat.sh.

## Inherited state (from reports/qwen38max-shard-2026-08-28.md)

- S1–S4 DONE on lane/qwen38max-shard (8 commits, NOT yet PR'd, NOT
  rebased on main): validator harness PASS on spark7, wire format v2,
  sharded loader, TP kernels, packer/verifier byte-exact round trip.
- S7 16-rank pack build was download-bound (AMD checkpoint ~1.25 TiB to
  /mnt/model-warm/packbuild/qwen38max/amd-mxfp4 on spark0/spark7).
- spark5 ceph still wedged (coordinator item); build on spark0/spark7.

## Plan

- M0: land the shard sprint. Rebase lane/qwen38max-shard onto main,
  resolve conflicts (loader/kernels moved under the publish-chain
  regime), run `make offline-gates` BY EXIT CODE, open the PR with the
  report in docs/AGENT_LANE_BRIEFS/reports/.
- M1: check S7 download status; when the queue accepts work, build +
  verify the 16 rank packs via tools/qwen38max_build_ranks.sh (queued,
  priority 5, staged scripts, ttl generous).
- M2: fleet window — 16-rank load, cross-rank numerics gate (the
  tp_probe_memory_mode fail-closed is expected single-node), first
  decode tokens, telemetry receipts at 127.0.0.1:8765.
- M3: serving stack (residentd/driver for TP4xPP4) + perf hill climb
  with kill-switch discipline (exactness before timing, RED on
  mismatch).

## Wait-state discipline (stagepacks/queue not fully live yet)

Every wakeup: check queue state (runs/queue.jsonl), telemetry, my PRs,
then advance the cheapest unblocked milestone. No hand-launched fleet
processes; no /tmp staging; TERM-only cleanup by cwd-filter.
