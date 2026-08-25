[lane G-glm52-bringup] attempt 1 17:16:46
[lane G-glm52-bringup] attempt 1 exited clean
[lane G-glm52-bringup] attempt 2 17:17:34
[2026-08-25 17:4x] STEP 1: re-probed band spark8,9,a,b,1,d,e,f — all up, GB10 GPU each, slot-A ports free, no residentd running, sparkc still unreachable (banner timeout); packs rank00-07 present (102835957760 B each); spark1 ALREADY seeded with full runtime tree + rank04 pack (shipped 2026-08-25 14:27), config tp_rank=4/stage_pack_path ok, model_resident.json node4→spark1:19484, kvcache dir exists; bench_b1.json on spark8 = 11-tok prompt budget 128; artifact: qualification/glm52/performance/tp8_bringup_20260825/step1_probe.md
[2026-08-25 17:24] STEP 2+3: spark1 runtime tree + rank04 pack ship found already staged (2026-08-25 14:27); sha256 gate PASS — sparke master == spark1 copy == a4f67c5eae0ddeedcc5efd651cf34333fd006db9306ac6bfa5efa39d9aee2c16; artifact: qualification/glm52/performance/tp8_bringup_20260825/step23_sha_gate.md
[lane G-glm52-bringup] attempt 2 exited clean
[lane G-glm52-bringup] attempt 3 17:28:52
[lane G-glm52-bringup] attempt 3 exited clean
[lane G-glm52-bringup] attempt 4 17:40:41
[2026-08-25 17:5x] STEP 4 (wip): solo boots spark8-rank0 & spark1-rank4 both fail schema_error(6) @adapter_initialize; deployed adapter==tp8 id, members/schema v4 walk clean -> instrumenting: traced adapter .so on sparke -> spark8-only redeploy + solo rerun; backups first
