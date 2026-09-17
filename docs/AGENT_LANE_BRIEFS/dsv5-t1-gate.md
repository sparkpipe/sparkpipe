# dsv5-t1: reconciliation + structural gate + reference feasibility

Lane: DSV5-T1 (mgr2 dispatch), branch lane/dsv5-t1, night of 2026-09-15.
Machine-readable evidence: examples/release/dsv41_flash_tp8/t1_gate_receipt.json.

## 1. Reconciliation verdict: SP-2's receipt is TRUE; SMOKE-MD's audit is a false negative for dsv5

SMOKE-MD (PR 1015, docs/MULTI_DEV_SMOKE.md) claims "dsv5 (tp8) are NOT placed
on any sparkdata root (fleet-replicated listing)". Checked in the prescribed
order:

1. SP-2's placement receipt
   (examples/release/dsv41_flash_tp8/placement_r2_receipt.json @ 83b1a09,
   merged as PR 914) records 16 placements per the tp8 replica law
   (rank r -> spark r and spark r+8), all sha256_match=true and
   experts_byte_identical=true, at ~/sparkdata/dsv41flash.mxfp4.tp8/packs/.
2. Tonight's fleet listing confirms those exact paths on all 16 nodes:
   packs 39,158,874,624 B + .experts 552,976 B everywhere, mapping
   r0=spark0/spark8, r1=spark1/spark9, r2=spark2/sparka, r3=spark3/sparkb,
   r4=spark4/sparkc, r5=spark5/sparkd, r6=sparke/spark6, r7=spark7/sparkf.
3. On-disk sha256 re-verified byte-exact against the receipt on two bookends:
   spark0 rank0 (9079cc4b..., experts 83abd3c0...) and spark6 rank6
   (58a93a87..., experts 288446ba...). No re-placement was performed because
   nothing was missing.

The convention mismatch that produced the audit's false negative: the placed
root carries the model-family name dsv41flash.mxfp4.tp8 (packs law
~/sparkdata/<arm>/packs/, docs/STAGEPACK_NAMING.md), while the audit listed
for the lane shorthand "dsv5". This is stronger than laguna's resolution
(warm staging vs placed): for dsv5 the PLACED state exists with the full
receipt chain. For the record, laguna was checked tonight as a cross-lane
duty: no laguna root exists on any of the 16 nodes — SMOKE-MD is correct
there, and laguna's packs are warm-staging-only.

One placement defect found and repaired (content-preserving): spark6's own
cell (packs-r2/rank6.spstage.experts, local_placement=true per the receipt)
was root:root mode 0600 — unreadable by the spark6 user weightd runs as,
unlike the other 15 nodes (owner-owned 0664). chown/chmod to spark6:spark6
0664; sha256 before == after (288446ba...), so the receipt's
experts_byte_identical contract is intact.

## 2. Novel finding: weightd wire drift for any lane on a pre-PR-1007 base

The dsv5 lane tree (rebase base 50bd0d3) declares
SparkWeightdIpcAttachLazyResult.mesh_send_buffer_bytes as uint32_t; the
deployed stable-channel daemon (main >= da8a782, PR 1007) declares uint64_t.
The 4-byte struct-width difference shifts the manifest_sha256 field on the
wire: the client reads [mesh_send_buffer_bytes high dword][digest[0..27]] and
fails with LAZY-MANIFEST-MISMATCH + SPARK_STATUS_HASH_MISMATCH at attach.
Fixed here by merging origin/main (caa44f5) into the lane; init went green
immediately after. Any lane whose client predates the weightsd-channel merge
hits the same wall on the stable channel.

## 3. Structural gate on the r2 packs: PASS (module loads; decode fail-closed)

On spark6 (GB10 sm_121a), against
/home/spark6/sparkdata/dsv41flash.mxfp4.tp8/packs-r2/rank6.spstage
(rank 6 of 8, contract bb8adaf7..., config 8be45ce0..., 1038 tensors, 40
layers, hidden 5120, vocab 129280, 384 routed experts, codecs FP8_E4M3
linear + MXFP4 expert), via the stable weightsd channel
(/run/sparkpipe-weightsd/weightsd.sock), under a sparkcap scope
(MemoryMax=4096M/MemoryHigh=2900M):

- init_status=0 in 21.5 s — the module loads and lazy-attaches the r2 pack
  (spine 3,061,536,448 B + expert arena 36,097,228,800 B, 1038 entries,
  inventory validated) through weightd.
- decode ATTEMPTED: exec_status=19 SPARK_STATUS_UNSUPPORTED — the module's
  Execute is the fail-closed stub; decode kernels do not exist yet (the CUDA
  translation unit is the context-ensure only). This is the I01-conformant
  clean failure: the module has never served, and it says so explicitly.
- destroy_status=0, gate exit 0. Mesh weightd (T1-G53) and all co-resident
  lanes untouched; client-only attach.

The per-layer comparison target for later: the M7 oracle ladder receipts
(dsv5m7h-ladder0 684 pieces, dsv5m7h-ladder7c 935 pieces, layers
0/2/8/14/20/38/39) remain the captured reference-side dump to compare
against once driver decode exists.

## 4. Engram + reference feasibility: Engram is NOT the blocker; the full-chain decode engine is

- Engram tables: both layers (1: 384,006,168 rows; 14: 384,016,682 rows x
  256 B) are node-local on spark6 inside the warm checkpoint
  /mnt/model-warm/deepseek-v4.1-flash (48 shards, revision dba1be0a...).
  The M7 oracle's CheckpointEngramTable reads rows real-path from the
  checkpoint (per-id absolute seek into both planes) and those pieces are
  validated (commits 38478e4e, 7a4dcf5: engram addressing resolved,
  real-row cells in tolerance). The .spengram row-shard set is only 2/16
  cut (engram-r2 ranks 0-1 on spark6), but the OFFLINE reference does not
  need it. No ceph lease was needed or taken: the checkpoint is node-local.
- Verdict: a fixture-grade T1 offline reference is NOT deliverable tonight —
  not because of Engram, but because the full-chain decode orchestration has
  never been exercised by anyone: the oracle validates position-0/layer
  pieces, with the CSA2 Reuse layers and the Reindex layers (24/28/32/36)
  explicitly excluded from the validated ladders. Writing the orchestration
  tonight and committing its output as fixtures would validate it with
  nothing but itself — and the serving-gated half does not exist (driver
  decode is a fail-closed stub). Per the comparison contract, that fakes
  coverage. Not done.

## 5. Exact remaining work to T1

1. tools/t1_reference_dsv41_flash.py: lift the oracle's validated pieces
   (quant math, rope/yarn, hc, sparse attention, compressor, indexer, expert
   MLPs, engram row read) into full-chain orchestration: CED 20+20, CSA2
   Full/Reindex/Reuse cache chain, indexer top-512 over the 16384-position
   candidate pool, FP4 global KV rollout, Single-Pass mHC streams, greedy
   head over 129280.
2. Validate the new engine's position-0 against the committed layer-0 oracle
   (the contract's anchor rule), then commit fixtures (capital_of_france,
   count_up) + MANIFEST + negative controls.
3. Driver decode kernels for
   modules/dsv41_flash_resident_decode_stage (Execute is fail-closed today)
   + the T1R1 dump hook; decode both prompts via the shared weightsd.
4. Cut engram .spengram row-shards 2-15 (serving-path prerequisite; the
   offline reference does not need them).
5. tools/t1_reference_compare.py, T1 verdict (ids exact, streams banded,
   FIRST DIVERGENCE evidence, no loosening), then ONE measured B1 tok/s
   against the ~149 tok/s roofline ceiling.

## 6. Generator check (mgr2 directive)

No dsv41 deployment generator exists yet (no gen_deployment tool in the
lane). The gate probe used its own minimal node context
(execution_row_capacity=1); the module Makefile default is 64. Nothing in
the dsv41 tree carries the 1024 value: the laguna generator bug has no dsv5
counterpart today, and the future dsv41 generator must pin the mesh-law 128.
