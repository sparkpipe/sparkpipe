# Lane brief: GLM 5.3 Flash (glm5_next) — first serving build

Worktree: /tmp/lane-glm53 (git worktree, branch lane/glm53, synced to main)
Your nodes: spark2 (dev instance — light work only while the 27B dev
daemon is idle; coordinate via report). You INHERIT spark8..sparkf from
the glm52 lane when it exits — do not start heavy pack builds before
then. Cap TWO heavy jobs per node.

READ FIRST: docs/GLM53_FLASH_KERNEL_ASSESSMENT.md — the full component
mapping with tensor-name evidence. This brief is the execution plan; the
assessment is the why.

## Mission

Bring GLM 5.3 Flash (glm5_next, 306 GiB FP8, 45-layer hybrid) to its
first validated, serving-ready build as family `glm5_next`, assembled
from three donors — NOT new kernel research:

- glm52 module (`modules/glm52_resident_decode_stage/`): base skeleton,
  MLA projections, MoE 288+1 sigmoid/noaux_tc (identical config to 5.2
  except expert count), MTP, FP8 [128,128] quant path.
- k3 module (`modules/k3_resident_decode_stage/`): KDA linear-attention
  kernels (`kda_qkv_beta`, `kda_decay_{down,up}`, `kda_{q,k,v}_conv`,
  `kda_head_log_scale`, `kda_gate`, `kda_out_norm`) and the per-layer
  kda/full dispatch table pattern.
- dsv4 module (`modules/dsv4_resident_decode_stage/` + model-families/dsv4):
  lightning indexer (`indexer.wq_b/wk/weights_proj/k_norm`), the kpool
  compressor path (cache arena already carries `compressor_state_arena`),
  and hyper-connections (hc_mult 4, sinkhorn 20 — tensor names identical).

Source: /mnt/model-warm/glm-5.3-flash (warm, 62 shards, verified listed).
TEXT STACK ONLY — skip the vision tower entirely (same call as the
qwen-flash lane).

## The three real deltas (everything else is constants)

1. MLA with qk_rope_head_dim=0 (pure nope absorbed scoring). Check whether
   the glm52 MLA kernel hardcodes rope 64; add a compile-time
   MLA_ROPE_DIM=0 instantiation if so. Gate with kernel cosine vs CPU oracle.
2. Checkpoint→pack name mapping (kda_* <-> A_log/dt_bias/f_a_proj/...;
   compressor.wkv <-> wk + index_kpool_compress_*). A table in the pack
   tool + geometry header, with tests.
3. Hybrid dispatch geometry: 34 KDA + 11 DSA (layers 3,7,...,43), first 3
   dense, MTP layer 45 (eh_proj/enorm/hnorm), HC on every layer.

Verify K3's `use_full_rank_gate=true` vs glm53's omission before wiring
the KDA gate kernel — do not assume the default.

## Milestone ladder

M1 Contract freeze: model_contracts/glm53_flash_authoritative.json with
   pinned sha256s (all small files + strided shard sampling, note it).
M2 Geometry header + name mapping: model-families/glm5_next/ generated
   from config.json; mapping table with a round-trip test.
M3 Synthesized pack + module builds: family module publishes at
   mid-pipeline tier (STAGE_COUNT=2 STAGE_LAYER_COUNT=4 MTP_LAYER_COUNT=0)
   then whole-stack. Validator PASS, decode-vs-prefill bit-exact,
   determinism, kernel cosine gates per donor component.
M4 Real packs: 16 rank-sharded packs (start TP16, ~19.1 GB/rank; TP4xPP4
   is the hill-climb alternate). Deploy all-16 per README fleet policy.
M5 Serving + first perf cell: exact-32K B1 decode. Expectation DSV4-Flash
   class ~35-45 no-spec (see assessment); MTP-1 after no-spec is honest.

## Rules that bind you

docs/AGENT_LANE_BRIEFS/README.md in full — write set, build chain, truth
rules (every claim = command + raw output), script parameterization,
escalation. The glm52 lane owns spark8-f until it exits; coordinate in
your report, never touch another lane's daemons. Report to
docs/AGENT_LANE_BRIEFS/reports/glm53-<date>.md at every milestone.

## Coordinator update (2026-08-30): prefill width fixed on main — redeploy unit

Main 1554464 decouples execution_row_capacity from
resident_sequence_capacity and ships 1024-row chunk configs (was 16 —
the 10 tok/s prefill root cause; 32K prompt was 2048 sequential
submissions). NEXT REDEPLOY MUST SHIP TOGETHER: the rebuilt adapter/
driver .so from main@1554464 (the old binary SCHEMA_ERRORs the new
1024 configs) + the regenerated deployment/glm5_next_tp16/* configs.
Then re-measure prefill (expected floor ~600 tok/s from re-stream
arithmetic alone). Decode path unchanged (12.7 tok/s — P1 async port
+ #760-class transport remain the rocks).
