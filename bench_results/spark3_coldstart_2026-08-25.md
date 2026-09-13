# spark3 qwen38-fp8.tp1 — cold-start measurement (2026-08-25)

Cell: ctx512_b1 (512-token prompt, output budget 128), DFlash2 active
(speculate=1, method=dflash2, draft_count=8, block_kv=0, window=2048, ctx_cache=1).
Method: kill daemon -> bash /tmp/launch_sp3.sh -> poll :17480 + ready banner ->
fire batch through a per-line `date +%s.%N` timestamper; parse event arrival times.

## Results

| Phase | Cold (fresh daemon) | Warm (2nd run, same cell) |
|---|---|---|
| Launch -> port listening | 13.08 s | - |
| Launch -> ready banner | 13.08 s | - |
| Ready/batch-start -> first token | 63.71 s | 72.94 s |
| **Launch -> first token** | **76.79 s** | n/a |
| Batch wall (128 tokens) | 91.53 s | 100.85 s |
| Steady decode after first token | 4.57 tok/s (127 tok / 27.82 s) | 4.55 tok/s |

## Findings

1. Weights/pack load is fast: control endpoint accepts + prints
   `model_residentd ready rank=0 ...` ~13 s after exec (page-cached FP8 pack).
2. The dominant term (~64-73 s) is FIRST-REQUEST processing, not daemon
   startup: it reproduces on a warm daemon. Daemon log shows no drafter-load /
   graph-compile lines after ready — the time goes into the request's
   prefill/spec pipeline itself.
3. Run-to-run variance on TTFT is ~±10 s (single sample each).
4. Caveat: this is process-restart cold (OS page cache warm). True boot-cold
   would add disk-cold read time for the 29 GB pack.

## Environment notes
- Daemon: ./bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index 0, cwd=/home/spark3/sparkdata/qwen38.fp8.tp1
- Launcher: /tmp/launch_sp3.sh (exports DFlash2 env, truncates /tmp/qwen38.log)
- Raw ts files: spark3:/tmp/coldstart_batch.ts, /tmp/warm_batch.ts
