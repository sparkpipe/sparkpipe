# glm5-closeout — the fixed packs LAND on the fleet; generation STILL degenerate on a served=0 first request; evidence captured, STOP honored

Lane: `lane/glm5-closeout` (worktree /tmp/lane-g5close), based on main
f63eedf (packer one-liner 4758e68). Fleet: spark0..sparkf, glm5_next.tp16
runtime roots, api spark0:8433. Every claim = command + raw output. This
session held the ONE wave ownership (reservations: released the expired
lane-glm5rt / lane-glm5kda holds; reserved all 16 for
`lane-glm5-closeout`, TTL 360m, at session start).

## How this serves the scoreboard

The brief's gate: land the fleet on the fixed packs (4758e68), then the
quality gate (COMPSEC-17) and the M5 exact-32K B1 decode cell. RESULT: the
packs landed and verify 16/16 — a real, previously-missing deploy gate
(pack header provenance) was found and fixed on the way — but generation is
STILL degenerate on a clean first request, so per the brief's stop
condition ("If still degenerate: STOP, capture G5N-DBG + per-layer
checksum evidence, report (do not chase)") COMPSEC-17 and M5 are NOT RUN
and the evidence is archived for the next lane. The scoreboard cell stays
blocked on coherence, as designed.

## Pre-flight findings

1. Packer provenance: ranks 1-15 ran the FIXED packer
   (sha256 ~/g5rt2-src/tools/glm5_next_resident_stagepack.py =
   `90b970b58ad14e52...` == main 4758e68 on all of spark1..sparkf; spark0
   clean at 1dac68b — rank 0's completed pack is correct at rank 0 because
   start=0 makes m[0:count] accidentally right, and it independently
   verifies below).
2. The driver changed under the lane BEFORE the swap: all 16 nodes hold
   lib/model_driver.so sha256 prefix `6ca5f16b8b4259c5` (2,290,016 B, mtime
   2026-08-29 12:50:56 UTC) — the coordinator's deploy. The pre-swap live
   residentds started 13:29:51 UTC, i.e. ON that driver, so the BEFORE
   baseline and the AFTER wave differ ONLY in packs. LAUNCH-STATE.md's
   `0292a3e55f45b2fd` receipt is stale (updated at lane end).
3. The deployed packs are SYMLINKS
   (packs/<name>.g5nsp -> ~/glm53_packs/<name>.g5nsp): the swap is an
   atomic rename of symlinks; old pack FILES never touched (rollback =
   rename back).

## C1 — build monitor + per-rank verification

Controller: tools/glm5_next_pack_orchestrate.py; verifier:
tools/glm5_next_pack_verify.py (both committed on this branch, shipped to
nodes as a git bundle fetched into ~/g5rt2-src and `git archive`d to
/tmp/g5close-verify — never a hand-copied tools/ dir).

Per rank: header (magic/tp/file_bytes == rank-0 receipt 21,706,046,976),
1160 entries, layout bounds + payload byte-count rules, FULL plan diff
against the fixed packer's plan (all 1160 entries field-by-field), spot
payload round-trips regenerated from the checkpoint through the packer's
own produce closures, directory-sha cross-rank invariant.

    rank 0 (spark0, --deep): VERIFY-PASS ... dir_sha 421ef0989c054f67
    ranks 1..14: VERIFY-PASS, same dir_sha 421ef0989c054f67 (--deep on 1, 8)
    rank 15 (sparkf, --deep): VERIFY-PASS, same dir_sha
    rank 4: VERIFY-PASS --deep (see requeue below)
    per-node logs: /tmp/g5verify-r<r>.log

Two verifier bugs found and fixed during bring-up (mine, separate commits):
plan-vs-pack comparison must use the shared field subset; expert slabs'
produce streams payload+scale interleaved (compare the sized region).

Rank 4 requeue (the brief's wedged-rank rule): spark4's builder restarted
from scratch mid-lane (etime 1030s->307s, size 518MB->82MB — coordinator
action) and then crawled in ceph at ~60KB/s (fleet average ~4.5MB/s; ceph
`-s`: HEALTH_WARN, "3 OSD(s) experiencing slow operations in BlueStore",
19/19 up, PGs active+clean, ~947 MiB/s rd fleet-wide — spark4's path was
the slow one). Requeued SOLO on healthy spark9 from the committed branch
archive: build 12 min, `--deep` VERIFY-PASS (dir_sha 421ef098...), then
delivered to spark4 — sha256 identical on both ends:

    4a36178167143d4d126d46bed6eb91b81f6884be33a04c906350ad78c896a6b4  (spark9 /tmp/g5r4build)
    4a36178167143d4d126d46bed6eb91b81f6884be33a04c906350ad78c896a6b4  (spark4 canonical)

The stalled spark4 builder got TERM (my lane's python packer; D-state, died
on read return); its partial quarantined as
`glm5_next_stage.tp16.rank4.g5nsp.stalled-partial` (650,511,360 B).

## C2 — the swap + the header-provenance gate (the missing deploy step)

tools/glm5_next_pack_swap.sh: renames the old symlink to
`.pre-closeout-bak` (guarded — never overwritten on re-run), then
rename(2)s the new symlink into place. 16/16 swapped; outputs in the
session transcript. Old pack files remain in ~/glm53_packs/ for rollback.

FIRST WAVE FAILED: all 16 ranks died at adapter_initialize:

    GLM5_NEXT-ADAPTER LoadDriver rc=7
    model_residentd initialize=hash_mismatch status=7 phase=adapter_initialize

Root cause (modules/glm5_next_resident_decode_stage/source/
spark_glm5_next_resident_decode_stage_module.c:287): the module rejects a
pack unless header->contract_sha256 == hex(GLM5_NEXT_CONTRACT_SHA256) and
source_config_sha256 / pack_recipe_sha256 are NON-ZERO. The repack packer
(4758e68) emits ZEROS for all three (assemble_header's
CONTRACT_SHA256="0000...0", config/recipe = bytes(32)). The M4 bring-up
patched these "in place on every node" (glm53-2026-08-27 report: config
bb8f01c4..., recipe a1915eb3...) — that step was never re-run after the
repack. Byte proof (spark0, old vs new pack, header offset 161..257):

    old: a40e9ec5fbfb0c1a...8db4 | bb8f01c42cb92a52... | a1915eb3b1f7238b...
    new: 00000000... (all zeros)
    deployed adapter AND driver embed contract a40e9ec5fbfb0c1a...8db4 (`strings`)

Fix: tools/glm5_next_pack_header_patch.py (committed) copies the 96
provenance bytes from the OLD (loading) pack after asserting the contract
equals the deployed pin. 16/16 PATCHED (contract a40e9ec5..., config
bb8f01c4..., recipe a1915eb3...), then 16/16 re-verified:

    VERIFY-PASS (skip-spot) rank r: ... 1160 tensors, dir_sha 421ef0989c054f67  (all 16)

## C3 — the waves

Wave attempt 1 (mine, before the header fix): died at init (above).
Wave attempt 2: a simultaneous launch raced the coordinator's automation;
the fleet came up 16/16 on the fixed packs (~16:11 local) with the api
serving — verified `grep -c "model_residentd ready"` == 1 on 16/16 hosts.
Wave attempt 3 (clean, for the canonical curl): my
tools/glm5_next_closeout_wave.sh stop (cwd-scoped TERM, zero procs, 45s
settle) + staggered launch (2s; the 16-parallel ssh fan-out twice tripped
the controller's ssh proxy — "No route to host"/banner timeouts — the
stagger is the reliable form) + api + health gate. 16/16 ready,
api healthy. NOTE for future waves: the stagger (32s spread) is well
inside the 180s hidden-transport window.

## C4 — coherence: STILL DEGENERATE, on a served=0 first request

BEFORE (old packs, standing wave, driver 6ca5f16b):

    $ curl -s -X POST http://localhost:8433/v1/completions -H 'Content-Type: application/json' \
        -d '{"prompt_token_ids":[154819,11,1875,525],"max_tokens":24}'
    {"object":"text_completion","tokens":[108539,108539,108539,108539,129988,129988,99918,99918,99918,27967,27967,...x15],"status":0}

AFTER, dirty slots (coordinator probe traffic on the wave): 
    [3921,3921,...(x13),142967,3921,142967,3921,142967,3921,142967,3921,12026,12026,12026]

AFTER, the CANONICAL test — fresh wave, api health-gated served:0, the
literal first request, plus the coordinator's state-bleed pair:

    HEALTH-A: {"status":"ok","served":0}
    CURL-A (p=[154819,11,1875,525]): [66188 x21,3569,66188,3569]           <- STILL DEGENERATE
    CURL-B (p=[154819,11,1875]):     [9007 x11,49039 x13]
    CURL-C (p=[154819,11,1875,525] again): [114907 x8,3921 x16]

Reading (evidence, not chase):
  - Candidate 1 (KDA state persistence) is REAL but NOT primary: A≠C on the
    same prompt (66188-repeat vs 114907/3921-repeat) proves cross-request
    state dependence; but A — the FIRST request on a COLD fleet — is
    already degenerate, so a cold model is broken per se.
  - The first token after a cold wave is deterministic (66188 here; the
    coordinator independently got 66188 first on its cold reading; the old
    packs' cold receipt was 116315,41267x15) — the packs DID change the
    distribution; generation still collapses into a repeat attractor.
  - That leaves the coordinator's candidates 2 (conv-window left edge at
    position>0) and 3 (fused-section shapes vs kernel expectation at
    rank>0). Pack-side evidence: pack bytes == checkpoint per-section
    contract, deep-verified on ranks 0,1,8,15 and plan-verified on all 16
    — the pack CONTENT is proven right against the fixed packer's
    contract; if the kernel expects a different layout, that mismatch is
    module-side and needs the probe ladder or an oracle diff (below).

## STOP-clause evidence capture

1. G5N-DBG (preserved, clean wave, fixed packs):
   spark0:/tmp/g5n-clean-wave-r0.log (7,668 lines,
   sha256 59174635b7e42068...) and /tmp/g5n-clean-wave-api.log. Classes:
   72 admit-entry / 72 execute / 7360 tp completion (all status 0) /
   72 completion emit ("acc 4 raw_acc 4" on the prefill sub, then acc 1,
   "resid_zero 0" throughout — no zero-residual trips).
2. Per-layer checksums: UNAVAILABLE on the deployed driver. Arming
   SPARK_GLM5_NEXT_PROBE=1 fleet-wide (staggered wave) fails at init on
   every rank:

       GLM5_NEXT-ADAPTER LoadDriver rc=15
       model_residentd initialize=busy status=15 phase=adapter_initialize

   rc=15 = SPARK_STATUS_BUSY: the 6ca5f16b driver's create() refuses to
   run with the probe env — the kda-era diag ladder does not arm on this
   build. The probe wave was torn down; the fleet was restored clean.

## C5 / C6 — NOT RUN (blocked by the brief's gate)

Everything is staged for the moment coherence lands: COMPSEC-17 harness
(tools/glm5_next_compsec17.py, ds4_eval archive format, tokenizer.json
decoder for the token-id-only API), M5 batch generator
(tools/glm5_next_m5_batch.py; /tmp/glm5-m5-exact32k-b1.json on spark0,
prompt sha 468db1b8d2ff5d87, 32512+256 = exact 32768 positions), batch
client built from this branch at spark0:/tmp/g5close-build/build/
sparkpipe_model_batch, timing wrapper tools/glm5_next_bench_wrap.py.

## INTEGRATION REQUEST

1. Probe gate regression: driver 6ca5f16b returns BUSY from create() when
   SPARK_GLM5_NEXT_PROBE is armed (per-layer diag ladder unusable on the
   deployed build). The kda-era driver armed it. Restore the env-gated
   ladder path or ship the ladder receipts from a build where it works.
2. State reset on lane/client acquire (kda candidate 1, now MEASURED:
   same prompt, different outputs across requests — CURL-A vs CURL-C).
   The conv/recurrence state must reset when a resident slot's lease is
   re-acquired; SparkModelResidentdCloseClientLocked already clears slot
   BINDINGS, the recurrence state itself persists.
3. Packer should emit the provenance fields (contract/config/recipe) —
   a --contract-sha/--config-sha/--recipe-sha set of args + the M4 values
   documented, so a repack never again ships zeroed headers. Interim:
   tools/glm5_next_pack_header_patch.py (this branch) is a required step
   after every repack; recorded in LAUNCH-STATE.md.

## Next experiment (for the next lane)

Rank>0 fused-section kernel diff: run the module's TP1 validator/oracle
against a rank>0 pack's fused tensors (kda_qkv_beta rows[0:1540] of rank 1
re-assembled from the checkpoint) and compare the kernel's split-index
expectation (per-head local ids) against the per-section layout — the M3
gates were TP1-invariant, and both slicing layouts coincide there. That
plus a working probe ladder settles candidate 3 in one run.

## Commits (this branch)

58fe37b verify/monitor/wrapper · d1fd9cc plan-diff fix · fbd286e
orchestrator · 55aeb1e swap+wave · compsec17 + decode fix · m5 generator ·
report skeleton · header patch tool + --skip-spot.
