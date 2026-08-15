# K3 TP4xPP4 — concurrency contract and preparation notes

## Fleet slotting (multi-model registry)

K3 TP4xPP4 is a FULL-16 topology, so it rides Big slot B: all hosts,
control 20480, collective 64620-64623, transport control 61700.
It never runs concurrently with another measured window; per the fleet
rule, residentd processes may be dev-active (up, idle) anytime, but a K3
measured window is EXCLUSIVE on the whole fleet — the always-on
DSV4 Flash TP4 (spark4-7) and Qwen 27B PP16 (spark0-3) dev-active
residentds pause for the window and are restored after. Promotion is
under 60 s, so pause/restore is cheap.

My assigned dev node is sparka: single-node CUDA smoke tests, kernel
probes, and packer runs there need no residentd and no ports, so they
are always allowed regardless of slot state.

## Preparation state

- Checkpoint: moonshotai/Kimi-K3-MXFP4, 96 shards (~1.6 TB), already on
  13 of 16 sparks under /home/<user>/srcdata/kimi_k3.mxfp4.pp13/ (the
  PP13-era layout: shards 1-93 split 7-per-node, 94-96 replicated).
  sparkd, sparke, sparkf are empty.
- TP4xPP4 shard map (from the pinned index): stage 0 (layers 0-23) =
  shards 1-24; stage 1 (layers 24-46) = 25-47; stage 2 (47-69) = 48-70;
  stage 3 (70-92) = 71-93; tail 94-96 = final norm/head/MTP, stage 3.
- Copy plan: every node gets its stage's full shard set + the tail on
  stage 3 + metadata; sha256-verified against the concatenated
  .sparkpipe-dataset.json table.
