# Wave-PR (SP-4R) stagepack repair wave — 2026-09-13

Agent: SP-4R (repair wave; this instance is itself a rerun after the prior
SP-4R process died in startup). Branch `lane/wave-pr-stagepack-sp4r` off
origin/main `c698e20`, fresh clone at `/Users/mac/sp4r`. Identity check
first: `tools/sparkpipe_github_pat.sh gh api user --jq .login` printed
`sparkpipe`; all GitHub-facing git/gh went through the PAT wrapper. Per-pack
details in `wave-p-sp4r-repair-ledger-2026-09-13.md` (same tree).

## Before/after matrix (arm x fleet, filesystem state; ✅ = new canonical
generation placed, sha-verified, .experts regenerated, .sha256 + receipt
written, chattr +i, purged; S = struct/digest only, ❌ = flagged)

| arm | before (Wave-P 09-13) | after (SP-4R) |
|---|---|---|
| glm53flash.fp8.tp8 | ❌ gen-A ec=1157 fleet-wide + gen-B ec=1160 s8/s9, content-unproven | ✅ 16/16 slots rebuilt through the current packer (1160 tensors, owns-shape, MTP-free), verify PASS 7/7 ranks sampled pre-placement (r0-r3, r5-r7), placed + locked + purged (r4 placement contingent on its rebuild, see blockers) |
| qwen3flash.fp8.tp8 | ❌ old-format struct-FAIL (offsets misaligned, scale_group 0), gen skews, 31 .tmp partials | rebuilt ranks staged through current packer on the official-fp8 recipe; placement gated on the stalled verifier (see blockers) |
| qwen3flash.fp8.tp4pp4 stage3 | ❌ rank12 byte-copy, ranks 13-15 two competing gens, 12 strays | ranks 12-15 rebuilt (305 tensors, 8,935,637,248 B); placement gated on the stalled verifier |
| k3.mxfp4.tp4pp4 | ❌R replica-canonicalization flagged | resolved as rebuild-mandatory: placed packs predate the 08-31 k3_shard manifest-reserve change and cannot be reproduced by the current pipeline; stage0-2 rebuilds launched, blocked mid-read by the dataset stall (see blockers) |
| qwen38_max strays | receipt-genuine strays | sparka rank00 + sparkf rank02 DELETED (93,243,509,760 B each, receipt-sha verified) |
| qwen3flash.bf16 strays | s4 duplicate rank0 in both bf16 arms | DELETED (46,333,527,808 + 41,818,367,232 B, Wave-P sha verified) |
| spark9 fp8 tp8 strays | rank03 + rank05 spstage | DELETED (30,518,614,272 B each, Wave-P sha verified) |
| spark2 qwen38_max .tmp | interrupted partial | DELETED (7,594,368,512 B, Wave-P sha verified) |
| spark0 qwen27b backup | stale-gen backup | DELETED (10,645,053,184 B, SP-5 sha verified) |
| sparkf glm tp8 compact.tmp | failed 09-05 compact strip | DELETED (13,156,747,648 + 7,896 B) with item-1 debris |

Deleted-bytes total (this wave, receipted): see ledger "Item 5" + per-node
old-generation debris lines; headline numbers in the ledger's final tally.

## Key findings this wave

1. GEN RULING (glm tp8): the current packer reproduces NEITHER placed
   generation. Fresh output (rank0 36cf4ae5…) differs from gen-A 578c3aab…
   and gen-B 3ad58198… despite matching gen-B's size/entry-count. gen-B's
   "interim canonical" status is void; the whole arm is replaced by the
   current-packer generation (dir_sha faf6ebe9… uniform).
2. RECIPE LAW (qwen fp8): the never-quantize rule decides the source —
   --expert-format fp8-official from qwen3.8-flash-next-fp8 verbatim. The
   first launch of ranks 0-3 used the BF16 quantize path, was caught in
   pre-flight, killed, and relaunched correctly; nothing from the wrong
   recipe was placed.
3. K3 REPLICA RESOLUTION: the 16 "unique digests" are the 16 distinct
   PP-stage×TP-rank shards (one pack per node, no replicas). The canonical
   question collapses to pack==packer, and it fails for all 16: k3_shard's
   08-31 manifest-reserve change (262128→1048560) alters the byte layout of
   every rank pack. All quartiles are rebuild-mandatory.
4. NAMING DENSITY: the tp8/qwen arms carry per-node lawful-name variants
   (padded rank01-style vs unpadded rank0). Placements preserve each node's
   recorded lawful name; stray name-classes are deleted, not renamed.
5. TOOLS: new file tools/qwen4_flash_experts_manifest.c (qwen4_flash wire
   variant of qwen38max_experts_manifest.c with argv tp_degree/tp_rank; the
   flash header carries no tp fields). v2 per-expert manifests,
   SparkWeightdManifestLoad self-validated; supersedes the v1 455-record
   sidecars from 09-05. Ships with the wave on node /home/<node>/
   sp4rtools/sparkpipe/tools/; committed in a follow-up code commit.

## Method notes

- Fleet budget: one heavy warm-read stream per node; builds run on placement
  targets where possible; mesh hops by user rsync + both-sides sha.
- Every placement: dest sha vs source sha vs receipt; .experts regenerated
  from placed bytes; .sha256 + receipt pair; chattr +i; per-node purge
  (sudo -n sync; echo 3 > /proc/sys/vm/drop_caches) with UTC timestamps in
  the ledger.
- Every deletion: lsof clean + lsattr clean (chattr -i first where set) +
  sha256 re-verification against the Wave-P/SP-5 ledger or the file's own
  receipt; a deletion was refused loudly whenever a re-check mismatched.

## Blockers / carry-forward

- Ceph warm-read stall window (began ~02:02Z node-time, still active at
  wave close for the qwen3.8-flash-next-fp8 and kimi-k3 datasets):
  - glm53flash.fp8.tp8 rank4: first build crawled post-stall (kill+retry
    per fail-fast); the retry ran healthy on spark1 (r4 is the only rank
    whose placement may land after the wave's ledger commit — see ledger
    addendum).
  - qwen verify gate (5 verifies) and k3 stage builds: D-stalled at
    deterministic offsets; k3 retried once per the fail-fast rule and
    reproduced the same stall point (journal sizes identical). Scopes left
    in place per the Wave-B1 playbook (no verdict killed mid-emit); they
    complete and self-purge when the window clears.
- 31 .tmp qwen partials: item-2 cleanup executes with the tp8 placements
  (finisher deletes per-node .qwenflash*.tmp after each pack lands).
