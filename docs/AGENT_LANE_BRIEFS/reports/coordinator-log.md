
## 2026-08-28 09:2x — sweep: spark3 root cause revised; qwen-flash merged; 27B bests improved

- INCIDENT REVISED: spark3 reboot #2 coincided with a ~104+ GiB GPU KV-pool
  probe next to mds+2xOSD on the same unified memory — noisy-neighbor OOM,
  not (necessarily) flaky hardware. Escalation updated (move MDS off GPU
  nodes; systemd memory guards). NEW BINDING RULE: storage-host nodes
  (spark3, sparkc) host NO GPU lane work; knee lane granted one-time
  exception at the verified 71.1 GiB envelope, one-B-per-session.
- MERGED lane/qwen-flash (7c28c72): TP4 rank packs deployed+verified on
  spark4-7, M5 kernel port (TP standalone, sharded embedding/argmax,
  format-6 experts), ratchet reconciled 186844+448 → 187879. Scanners green.
- SCOREBOARD: 27B bests improved on exact main (f8f2ea0): B1 no-spec 8.45
  (was 8.03), B4 aggregate 55.44 (was 36.22 — continuous batching + kernel
  fixes; B=1/2/4 = 8.45/16.75/55.44, knee sweep K1, spark3 clean session).
- qwen38max-shard milestone: module PUBLISHED through the v2 validator on
  spark7 (wire-format-v2 chain green); driver compile blocked on firmware
  'stages' schema in examples/ — coordinator fix queued next sweep.
- Ceph canary: transient ENOENT + slow uncached reads = MDS cold after the
  08:59 reboot (expected, self-heals); kimi-k3 warm copy intact (96 shards).
