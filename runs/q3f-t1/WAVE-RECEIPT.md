# WAVE RECEIPT — Q3F-T1 shakedown + T1 attempt (2026-09-15, GPU window 17:49Z)

## What ran

8-rank TP8 wave (spark0..spark7), one `t1_q3f_harness` per rank under
sparkcap, staged from the spark7 build of lane/q3f-t1. Prompt: throwaway
shakedown ids (9707,11,1878,1878,6511 + 2 new tokens); the canonical
fixture prompts still wait on the ceph lease. Every rank attached to the
SHARED fleet weightsd (`/run/sparkpipe-weightsd/weightsd.sock`), lazy
attach of `~/sparkdata/qwen3flash.fp8.tp8/packs/qwenflash.tp8.fp8.rankXX.spstage`.

## What passed (MEASURED)

- Module archive + harness build green on spark7 (nvcc sm_121a, strict).
- weightsd lazy attach per rank: manifest check against the `.experts`
  sidecar PASS (rank-local 64-expert window, kinds 12-17), spine load
  (~20 GB) PASS, `initialize ok slice=0+48 gdn=36 attn=12
  owns_embedding=1 owns_head=1` on all 8 ranks.
- T1 dump hook active (`t1_dump_enabled dir=/tmp/t1q3f/dump rank=R`).

## What failed (MEASURED, honest)

- First TP collective on every rank:
  `tp_device_collective.c:833 status=19 (SPARK_STATUS_UNSUPPORTED)`
  — `implementation->mesh_buffer == 0`. The weightsd ATTACH_LAZY result
  carries no mesh fd because the fleet's weightsd daemons run
  MESH-DISABLED (no `--mesh-rank/--mesh-interface/--mesh-sgid-index`
  identity triple => `SparkWeightdMeshReady()==0` => no fd staged). The
  collective on main is mesh-only; there is no direct-transport fallback.
- Zero tokens decoded. The scale-plane pack defect (see
  STAMP-CHECK-RECEIPT.md) would make any decoded tokens wrong anyway.

## Findings escalated (this wave)

1. FLEET: weightsd stable channel mesh-disabled (blocks every TP>1 T1
   lane: Q3F, GFULL, HY4, GEMMA4, QMAX). Fix options reported to mgr2
   (enable mesh triple = recommended; production-socket detour = risky;
   direct-transport fallback = code change on main).
2. FAMILY: qwen4_flash module had zero weightsd lazy-attach wiring (the
   #1013 mesh-lanes adoption missed it) — ported on this lane; attach now
   green end-to-end up to the mesh handoff.
3. PACKS: fp8 scale planes corrupt (STAMP-CHECK-RECEIPT.md) — accuracy-
   fatal; decode would diverge even with the mesh up. Re-emission needed.

## Cleanup

Zero `t1_q3f_harness` processes on all 8 ranks after the wave (verified).
No daemons touched. MESH_LEASE + GPU window log updated honestly.
