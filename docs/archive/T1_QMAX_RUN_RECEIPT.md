# T1-QMAX run receipt — 2026-09-15/16 night wave (re-stamp + engine repair)

> Archived 2026-09-28: a receipt from the 2026-09-15/16 waves. The TP16
> transport blocker it reports has since been fixed (GLM serves TP16 graph
> chains). Its evidence files and `runs/` are not in the tree, and its
> MESH_LEASE and worktree conventions are obsolete.

Lane lane/t1-qmax @ 3217be3 (+ tooling commits), worktree /Users/mac/t1qmaxn.
Mesh lease ACTIVE T1-QMAX window 16:21Z-~19:00Z; CEPH lease held from the
prior session (tail reads only; patch pass + release notes appended).

## VERDICT: T1 FAIL — execute step 0, routed-expert lease NOT_FOUND

The load path is REPAIRED: every placed pack now loads, validates,
attaches to the shared fleet weightsd, and the module initializes and
opens the TP16 collective on all 16 ranks. The decode still cannot run:
ranks whose resident expert window intersects the step-0 routing fail the
weightsd lease acquire (daemon NOT_FOUND), while ranks with an empty
window skip cleanly and time out on the collective. No tokens produced.
No tolerance touched. Phase 2 (measured B1 tok/s) correctly not attempted
- it is gated on T1.

## What this wave delivered (all MEASURED, receipts in this directory)

1. Re-stamp pipeline completed across the fleet. 16/16 placed packs now
   carry the codec-contract layout (weight_format 8 NVFP4_PACKED, group
   16, payload elements/2, scale elements/16 + resident*8 f32 tails):
   - ranks 0,1,2,3,4,6: warm rebuilds from the prior session
     (tool=qwen38_stagepack.py in each receipt)
   - ranks 5,8,9,7,a-f: in-place codec-contract patch by
     tools/qwen38max_patch_pack.py (tool=qwen38max_patch_pack.py,
     patched_from_output_sha256 records the superseded digest)
   - rank0 byte-identity re-proven by this agent:
     rank0-patch-identity-proof.json - sha256(patched_rank0.sp) ==
     placed warm-rebuild receipt sha c911156b...
   - uniform fleet verify 16/16 PASS with recomputed file digests
     (verify-rank{0..15}.log), packs relocked chattr +i after swap.
2. Pack-name hex bug fixed (ranka..rankf for ranks 10-15): wave 5 had
   silently lost ranks 10-15 to FileNotFoundError at the receipt read.
3. Seven engine defects fixed and committed (each commit cites evidence):
   - 027dff7 ValidateEntry accepts the stamped NVFP4_PACKED code on
     routed-expert entries
   - 3c99285 ValidateEntry expects the rank-sharded slab shape (the true
     wave-5 blocker: full 512-expert shape vs the 32-expert rank shard,
     ERBSITE module.c:410 in wave 5)
   - 59d7a3b + 484e4a7 sharded-shape expectations for every TP-sharded
     kind via kind masks (rows: routed slabs, shared gate/up, attn
     query/key/value, GDN qkv/gate; cols: shared down, GDN output, attn
     output; beta/decay/norms/router/conv replicated)
   - d9b3c17 grouped-scalar expert launch consumes the bound rank-slab
     view as-is (the copied tp_rank re-slice pushed ranks 1-15 past
     their slab)
   - 2b7ada8 shared expert consumes the rank shard with a TP allreduce
   - 510d6ab MTP completeness tracks the compiled MTP count (archive
     builds with MTP=0 against MTP-stripped packs)
   - 980afb1 acquire routed experts from the resident window only
     (rebased local prefix; zeroed expert output buffer; the hidden
     allreduce sums partials)
   - 3217be3 synchronize the slot stream before reading the routing
     prefix (the D2H used the default stream; keys were derived from
     stale offsets)
4. Attach evidence (single-rank, rank0): load ok (1682/1682 entries
   validated), lazy attach to the shared weightsd ok, tp_collective_open
   ok, MoE stage launches ok, run ends ONLY at the collective awaiting
   absent peers (attach3.log via attach runs).
5. Daemon path proven independently: weightd_execute_probe against the
   shared daemons on spark0 and spark5 acquires 8-key leases on the
   patched packs (EXECUTE-PROBE lines; run on rank0 and rank5 packs with
   correct per-rank shas).

## The remaining blocker (exact)

Wave 2/3: ranks with a nonempty resident-window intersection fail
SparkWeightdClientAcquire with daemon NOT_FOUND (ERRSITE
spark_weightd_map.c:393 status=3); rank0 (empty window at step 0) skips
the lease and times out on the collective (status=19). The probe
acquires identical-class keys (layer 0, resident experts) on the same
daemons with the same identity, through the same
SparkWeightdLazyPackCreate/CreateChecked code path. Un-tested
difference: the probe derives its keys from the loaded manifest groups;
the module derives them from routing. Next test: an instrumented wave
printing (layer, count, keys, raw status) per rank - the debug harness
for this exists (tree-patched build, not committed) - plus a
client/daemon ManifestFind parity check on the arena the module's
connection actually attached to. Wave 3 also hit MESH-REGISTER-FAIL
(tp_device_collective.c:960 status=4, NIC memory-band registration) on
relaunch - infrastructure state needs the night's leftover registrations
to drain before further waves.

## Evidence files

- patch-rank{1..15}.log (+ .fail1/.fail2 preserved first attempts):
  per-rank precondition sha, patch json, audit PASS, verify PASS, swap,
  chattr relock
- verify-rank{0..15}.log: uniform fleet verify, 16/16 verdict=PASS
- rank0-patch-identity-proof.json: byte-identity re-proof
- rank{0..15}.log: wave logs (workstation side)
- per-host /tmp/t1qmax/harness.log: the wave ERBSITE chains (wave 2:
  map.c:393 status=3 / collective 19; wave 3: MESH-REGISTER-FAIL)
- attach{,2,3,4}.log on spark0: single-rank attach progression
- fix commits 027dff7..3217be3 (8 commits) on lane/t1-qmax

## Honesty notes

- No tolerance was loosened; the compare stage was never reached.
- No fabricated tok/s: the only performance number tonight is the
  analytic ceiling 22.0 tok/s with NO measured companion (gated on T1).
- The fixture, its manifest sha, and the checkpoint identity chain are
  untouched from the prior wave's audit.

---

# T1-QMAX run receipt — 2026-09-15/16 night wave 2 (parity fix + first all-16 collective)

Lane lane/t1-qmax-wave4 @ dc4a16c (9 engine commits total), worktree
/Users/mac/t1qmaxw. Mesh lease ACTIVE T1-QMAX from 19:06:43Z.
Fixture untouched: capital_of_france.t1r sha256 74fc4bed5354b734... matches
the committed manifest.

## 1. ManifestFind PARITY TEST — verdict: MODULE side (key-derivation bug)

Tool committed this wave: tools/qwen38max_parity_probe.c (compiles on the
build node, runs offline against the placed experts sidecar; no daemon,
no lease, no CEPH). It derives module-style keys through
SparkWeightdRouteKeys on a synthetic full local window and diffs them
against the daemon-side id space (base-shifted) with
SparkWeightdManifestFind.

- rank5 manifest (window 160..191), layers 0/45/91:
  module-local keys 0/32 HIT, base-shifted keys 32/32 HIT,
  verdict basis=global every time
- rank0 manifest (window 0..31): both spaces coincide 32/32 HIT -
  the control that explains why rank0 alone never hit the acquire path

Root cause chain, both halves read in source:
- daemon map id space: tools/qwen38max_experts_manifest.c writes groups
  as (layer = entry->layer_index, expert = base + expert), base =
  tp_rank * per_rank; the lease table resolves keys with
  SparkWeightdManifestFind against that manifest (spark_weightd_lease.c
  prepare_groups)
- module derivation: RunMoe rebased the routing prefix on the rank
  window (correct for the window test and the RouteKeys schema check)
  but SparkWeightdRouteKeys emits the array index as the expert id -
  local 0..31, global on no rank except 0
- the probe path always worked because it copies (layer, expert)
  straight from manifest groups

Layer semantics were verified equal on both sides (module iterates
first_layer_index..first+count-1, the same space the pack entries and
manifest carry).

## 2. The fix (committed, minimal)

- module (spark_qwen38_max_resident_decode_stage_module.c): after
  RouteKeys, shift every derived key into the daemon's global id space
  (keys[k].expert += first); plus a standing route_keys print (layer,
  rank, base, total, count, exact ids) on every MoE lease
- daemon-side standing parity print (runtime/spark_weightd_lease.c):
  on any ManifestFind miss the daemon dumps the requested (layer,
  expert) and the group id range it actually holds for that layer
  (weightd_parity lease_miss)
- commits: 3edb3e1 (fix + prints), dc4a16c (parity probe + build
  wiring + runner retarget to /Users/mac/t1qmaxw and lane/t1-qmax-wave4)

## 3. Instrumented decode wave — two launches, one clean

Wave 1 (19:07:44Z): raced an external redeploy of the SHARED fleet
weightd daemons at 19:07:56Z (all 16 nodes, same second, new binary in
~/sparkdata/weightd, not this lane's build; no MESH_LEASE entry claims
it). Harness clients attached through the churn; every acquire then
died on a stale socket: map.c:393 status=4 (IO_ERROR), 8 ranks.

Mesh state before wave 2: all 16 daemons stable 7.5-9.5 min, then
weightd_execute_probe (rebuilt on the build node, run on spark5 against
the rank5 pack and the REDEPLOYED daemon) EXECUTE-PROBE waves=2 keys=8
waves_ns=813113913 - client and redeployed daemon are compatible and
acquire leases fine.

Wave 2 (19:21:41Z, clean):
- ALL routed-expert ranks derived correct GLOBAL keys, zero NOT_FOUND,
  zero weightd_parity lease_miss; step-0 routing (B1, top-8) touched
  exactly 8 windows: rank1 e51, rank5 e177, rank8 e256/269/287,
  rank9 e304, rank10 e329, rank11 e382, rank14 e469, rank15 e496 -
  every id inside its own window; the other 8 ranks skipped the lease
- the block has MOVED: all 16 ranks then failed at the FIRST TP hidden
  allreduce (16384 bytes = hidden dim bf16, round 0) with
  MESH-SPIN-TIMEOUT - every rank's own publication staged (pub
  seq=1024) but ZERO peer publications ever became visible (got=0)
  - evidence: runs/t1qmax/wave3-night/harness-rank{0..15}.log
- fabric counter-evidence: rocep1s0f1 PORT_ACTIVE on probed nodes, and
  the weightd mesh wired 15/15 peers over the same NIC class minutes
  before - the link is up; the tp transport's pairwise publication
  path is what does not deliver
- prior-session context: no wave of this lane ever reached a collective
  with all 16 ranks alive before the lease fix; the one independent
  other lane that got there (T1-G53, glm5_next, PR #1016) reports the
  same class of failure at its first TP collective

## VERDICT: T1 FAIL - honestly

The routed-expert lease NOT_FOUND blocker this lane was carrying is
FIXED and proven fleet-wide (parity basis=global, live route_keys
global ids, clean acquires, EXECUTE-PROBE green). The decode now runs
to the first TP hidden allreduce on all 16 ranks and the transport
delivers no peer publication; no tokens can be produced, the compare
stage was never reached, no tolerance touched, no rerun-until-pass.
Measured B1 tok/s correctly not attempted - gated on T1.

The standing blocker for the lane is now the TP16 transport publication
wire-up (ring/transport tp_device_collective + hidden_transport
rdma_verbs backend), a shared-fleet component: two independent lanes,
two model families, both stop at their first TP collective. It needs
its own evidence-driven repair pass (transport-level, not a backend
swap - required behavior cannot be waived by compatibility paths).
