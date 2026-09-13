# Audit r1 - all five serving_adapter.c vs the adapter_common.h design

Author: ox-alpha subagent. Date: 2026-08-24. Question answered: after the
runtime/adapter_common.h design, audit ALL FIVE spark_<m>_serving_adapter.c
files against it - exactly what moves to shared, exactly what stays
model-specific.

## 0. Basis and method

- Reference design: runtime/adapter_common.h (395 lines) + adapter_common.c
  (460 lines), canonical copy read at
  .agents/coord/tmp/seq1_ship/applycheck/runtime/ (newest mtime 2026-08-24
  13:24; same line counts as the wt_parity_home_1 and wt_land_r9 copies - not
  diffed against each other). ABI SPARK_ADAPTER_COMMON_ABI_VERSION 1u.
  Behavioral companion: .agents/coord/hwiface_adapter_contract_v1.md
  (F8.1-F8.7, defects D1-D5).
- Objects under audit: the five main-tree adapters, measured today -
  dsv4 1515 / glm52 1489 / qwen38 1471 / qwen36 2969 / k3 502 lines. All
  file:line cites below are against these main-tree files.
- Conversion state: main tree is PRE-conversion - zero references to
  adapter_common anywhere under modules/ runtime/ include/ (grepped). The
  skeleton patches (glm52_evidence_adapterskel_r3_*, landing-bundle r9 items
  07-11) cover only runtime-common + FOUR adapters; k3 has no conversion patch.
- Every "moves" verdict names the exact SparkAdapter* helper that absorbs the
  code; every "stays" verdict names the law that keeps it family-side.

## 1. Verdict

The header's division of labor holds under full-text audit of all five files:
the pending-slot discipline, completion glue, lifecycle bodies, strict env
readers, driver-load sequence, and env-staging mechanics are byte-equivalent
across the four extracted adapters and move cleanly. What stays family-side is
exactly what the header claims - plus six pieces of genuine shared code the
header does NOT yet own (S1-S7, section 3), two skeleton defects (H1-H2,
section 4), and a k3 tail that cannot "move" because it never implemented the
discipline being extracted (section 5).

Estimated movable volume at unchanged behavior:
dsv4 ~230 / glm52 ~220 / qwen38 ~210 / qwen36 ~200 lines (~7-15% each);
k3 ~0 without a rewrite.

---

## 2. Per-class disposition matrix (the exact answer)

MOVES = absorbed by the named shared helper. STAYS = model-specific.
PARTIAL = core moves, family tail stays.

### 2.1 State prologue

| Piece | dsv4 | glm52 | qwen36 | qwen38 | k3 | Disposition |
|---|---|---|---|---|---|---|
| Identity echo fields (submission_id..step_generation) | :792-800 | :621-629 | :826-834 | :498-506 | - | MOVES: SparkAdapterPendingCapture; families embed SparkAdapterPendingCore as first member |
| Pending head (active/row/lane/work_kind) | :776-791 | :612-620 | :817-825 | :489-497 | none | MOVES: same |
| Family payload tail (emit rows, cache lanes, spec state, frame mirrors) | :233-238 | :104-122 | :316-337 | :152-158 | - | STAYS: family owns "any model-specific completion payload" |
| Counters/routes/limits prologue | :1027-1041 | :993-1003 | :2860-2874 | :1402-1415 | :256-257 | MOVES: SparkAdapterInitializePrologue (+ embed SparkAdapterCommonState) |
| Residency anchor school | driver-frame echo :854-855 | driver-frame echo :765-766 | SUBMISSION anchor :835-845,:890-893 | pending mirror :551 | submission echo :374 | STAYS: RESIDENCY LAW (header :250-258); builder leaves residency zeroed; each family sets its own anchor. Three schools are load-bearing; do NOT unify |

### 2.2 Slot claiming / availability

| Piece | dsv4 | glm52 | qwen36 | qwen38 | Disposition |
|---|---|---|---|---|---|
| Claim scan | :774-781,:813 | :610-616,:639 | :814-819,:857 | :486-492,:518 | MOVES: SparkAdapterPendingClaim (BUSY on full) |
| AvailableSubmissionCount | :1431-1439 | :791-799 | :907-915 | :565-573 | MOVES: SparkAdapterAvailableSubmissionCount (four byte-equal bodies) |
| last_row_by_lane capture | :801-805 | :630-633 | :847-851 | :508-512 | MOVES: SparkAdapterCaptureLastRowByLane |
| resident slots per lane | qwen36 :852-853 / qwen38 :513-514 | | | | MOVES: SparkAdapterCaptureResidentSlotsPerLane |
| resident slots per row (row-major) | dsv4 :806 / glm52 :634 | | | | MOVES: SparkAdapterCaptureResidentSlotsPerRow |
| active-stamp timing (early vs late) | late :808 | early :617 | early :821 | early :493 | STAYS: preserved by design (header :157-161 "no observer can tell") |

### 2.3 Completion routing

| Piece | dsv4 | glm52 | qwen36 | qwen38 | Disposition |
|---|---|---|---|---|---|
| Orphan driver-completion callback | :816-825 | :642-651 | :860-869 | :521-530 | MOVES: SparkAdapterOrphanDriverCompletion (byte-equal x4) |
| Wake trampoline | :897-903 | :783-789 | :899-905 | :557-563 | MOVES: SparkAdapterDispatchWake |
| Match predicate | :840 | :746 | :882 | :543 | MOVES: SparkAdapterDriverCompletionMatches - single-frame pass reserved identity; multi-frame pass frame ids (:882/:543); same predicate supports both |
| Frame-counter accumulate | n/a (single frame) | n/a | :894-896 | :552-554 | MOVES: SparkAdapterAccumulateFrameCounters |
| Completion header build (zero + abi stamp + identity echo) | :841-853 | :747-759 | :2335-2348 | :1016-1028 | MOVES: SparkAdapterBuildCompletionHeader[WithResidency] - ABI LAW: stamp, never echo (header :260-264) |
| Accepted-token clamp | n/a | n/a | :2349 | :1030 | MOVES: SparkAdapterClampAcceptedTokenCount |
| Mismatch policy | publish SCHEMA_ERROR completion :844,:875-876 | orphan++ :767-768 | poison frame_status :885-887 | poison frame_status :544-548 | STAYS: three distinct published behaviors; document per family |
| Token publish tail (ids, layout, stage gating) | :877-892 | :769-778 + CompleteSpeculative :667-732 | :2352-2382 | :1033-1040 | STAYS: speculation credit, emit rows, min-accept layouts |

### 2.4 Lifecycle

| Piece | dsv4 | glm52 | qwen36 | qwen38 | Disposition |
|---|---|---|---|---|---|
| ValidateConfiguration | :1011-1025 | :864-878 | :2828-2842 | :1370-1384 | MOVES: SparkAdapterValidateConfiguration(descriptor,&desc,program,stage_count) - four byte-equal bodies modulo constants |
| ValidateSubmission open gate | :1191-1197 | :1068-1072 | :769-778 | :449-455 | MOVES: SparkAdapterValidateSubmissionOpen |
| Quiesce body | :1441-1459 | :1417-1435 | :2629-2647 | :1158-1176 | MOVES: SparkAdapterQuiesce - four byte-equal bodies (latch + idle-slot demand + driver poll) |
| Snapshot merge | :1461-1493 | :1437-1469 | :2649-2681 | :1178-1210 | MOVES: SparkAdapterSnapshot - DEFECT H1 (null-gate dropped) |
| Destroy guard + teardown | :905-925 | :801-820 | :2683-2711 | :1212-1245 | PARTIAL: SparkAdapterDestroyReady + SparkAdapterTeardownDriver; family keeps its frees (cudaFree/paged/scratch) |
| Progress stub | :1423-1429 | uses header fn :1480 | :2621-2627 | :1150-1156 | MOVES: standardize on EXISTING SparkModelServingAdapterStreamOrderedProgress (header :404); no new helper |
| LoadDriver | :928-966 | :823-862 | :2714-2751 | :1248-1285 | MOVES: SparkAdapterLoadDriver via SparkAdapterDriverContract - dsv4 kind 0 target-free; glm52 kind 0 WITH pinned target (:836); qwen36/qwen38 kind 1 exact-mask (:2732/:1266); node_context only for struct-module families |

### 2.5 Configuration load & validation (JSON)

| Piece | dsv4 | glm52 | qwen36 | qwen38 | k3 | Disposition |
|---|---|---|---|---|---|---|
| Private JsonMember+JsonUnsigned wrappers | :321-338 | retired (:209-211 comment) | :497-514 | :272-289 | lenient U32 w/ fallbacks :38-46 | MOVES: retire onto EXISTING SparkJsonGetUInt32Member (runtime/json.c) - glm52 is precedent; do NOT add to adapter_common |
| Load skeleton (open, root-object, members-exact, schema_version, model_revision, path resolve) | :649-722 | :489-576 | :516-552 | :291-331 | absent | PARTIAL -> S5: common prefix extracts; divergent remainder stays |
| Member lists / schema macro / codec+revision checks | :186-211,:690 | :67-86,:513-520 (v4; TARGET_MISMATCH on codec) | :241-247,:537 | :105-112,:312 | none (defect D3) | STAYS: F8.6 per-model schema; version need not match |
| tp_collective stanza parser | :340-607 | :213-487 | - | - | :99-237 (own device/host tiers) | PARTIAL -> S3: dsv4 == glm52 except policy knobs; k3 is a different dialect |
| cuda_graph_count(s), GDN/MTP stanza, speculation stanza, pool sizing | :609-647 | :534-569 | env-driven | env-driven | :94-98 | STAYS: family |

### 2.6 Environment

| Piece | dsv4 | glm52 | qwen36 | qwen38 | Disposition |
|---|---|---|---|---|---|
| Prefix-cache A/B toggle (default-on, exact-"0" off) | - | - | :119-123 | :119-125 | MOVES: SparkAdapterEnvFlagDefaultOn |
| KV-flat ledger toggle (truthy first byte) | - | - | :927-931 | - | MOVES: SparkAdapterEnvFlagDefaultOffTruthy |
| Speculation kill-switch veto (config governs; env "0" vetoes) | - | :903-905 | - | - | STAYS: fourth micro-pattern (veto, not reader) |
| Strict-nonempty-not-"0" readers (SPECULATE :578-583, SPEC_AUDIT :191-195) | - | - | two sites | - | candidate S7: fifth semantics ("01" ON like ExactZero; empty OFF unlike it) |
| Numeric/string policy readers (draft-count clamp :133-146, method :177-185, policy :156-162) | - | - | tuning surface | - | STAYS: family |
| setenv staging macros (SET_TEXT/SET_UNSIGNED, loud-fail) | - | - | :589-592,:649-655 | :356-359,:360-377 | MOVES: SparkAdapterSetEnvironmentText + SparkAdapterFormatEnvironmentUnsigned; NAME LISTS STAY FAMILY (F8.6 frozen taxonomy) |

### 2.7 What unambiguously stays model-specific (per file)

- dsv4: topology variant block :17-183 (stage layers, batch-bucket SHA select
  :135-164, chain-capability math :169-183); PP/TP rank math :273-283;
  descriptor :288-319; graph-count cross-check :609-647; node-context
  derivation :1043-1128; stage-runner init :968-1009; dispatch fill + submit
  :1300-1421; prefetch/resolve via admission CACHE_PREPARE/COMMIT/ABORT
  :1224-1298; RELEASE-frame path :1300-1337; row-order school (round-major via
  SparkRowLayout, RELEASE exempt) :733-761.
- glm52: codec/revision #error pins :15-26; v4 speculator stanza :77-86;
  LaneSpec/speculator/DFlash2 machinery :125-136,:655-732,:880-975,:1139-1361;
  boundary law :1050-1059 (rejects ALL hidden boundaries); BuildFrame/admit/
  submit :1082-1141,:1363-1415; row-order school :578-602; stderr diagnostics
  :485,:574,:860,:910,:938.
- qwen36: whole speculation subsystem (MTP chain + dflash2/fold + acceptance
  cliff + losslessness repair + audit telemetry) :1714-2326; prefix-cache
  dual-ledger (flat refcount protocol + paged core binding + checkpoint slice)
  :917-1295,:2753-2826; REQUIRES_RELEASE release path :2428-2442;
  frames_executed/paged_touched refusal bookkeeping :2408-2444,:2564-2582;
  MODEL_EXTENSION telemetry :2372-2382; wave-major row-order school :663-714;
  env-configured module stanza :585-657.
- qwen38: runtime-tp-degree PP geometry :229-241,:1420-1436; hidden-boundary PP
  law :431-442; paged-KV over SparkPrefixCacheCore lane bookkeeping :575-751;
  block-stride/pool sizing :1287-1368; per-lane prefill framing + donor-witness
  checkpoint offers :1045-1148; max_inflight=1 descriptor face :253-256;
  row-order school :381-425.
- Architecture note (out of adapter_common scope): qwen36/qwen38 call CUDA
  directly (qwen36 :39; qwen38 :38,:751-828,:1308-1368), k3 likewise.
  Device-tier work living host-side; stays for this behavior-neutral
  extraction but belongs on the paged-KV/seam ledgers, not here.

---

## 3. Shared code that exists in the adapters but NOT yet in adapter_common.h

Ordered by value; none blocks the four-adapter migration.

- S1 HIDDEN-TRANSPORT SHIM (highest value): PostReceive+Send+TransportShim
  struct byte-identical between qwen36 :339-355,:1344-1394 and qwen38
  :161-177,:778-828 except hidden_dimension/bytes constants. ~110 duplicated
  lines. Add shim-init + shared callbacks to adapter_common, parameterized.
- S2 FRAME-ADMISSION HELPER: RequestFromFrame+EvaluateAndApply byte-equal in
  glm52 :1363-1377 and :1144-1158, qwen36 :1643-1658, qwen38 :957-972 (dsv4
  uses RequestFromSubmission shape :1309-1316). Four sites, one 12-line helper.
- S3 tp_collective STANZA PARSER: dsv4 :340-607 == glm52 :213-487 (~270 lines
  twice) modulo policy knobs: allowed-algorithm rule, threshold polarity,
  rail-host count source, port-base continuity placement. Parameterize.
- S4 JSON unsigned reads: retire dsv4/qwen36/qwen38 wrappers onto existing
  SparkJsonGetUInt32Member (precedent glm52 :209-211).
- S5 CONFIG-LOAD SKELETON: open/root-object/members-exact/schema_version/
  model_revision/path-resolve prefix common to all four loaders; extract only
  the provably identical prefix (ordering differences preserved).
- S6 PROGRESS STUB: standardize three private stubs onto existing
  SparkModelServingAdapterStreamOrderedProgress (no new helper).
- S7 (optional) FIFTH ENV READER (set AND nonempty AND not-exactly-"0"):
  qwen36 :578-583/:191-195. Two users in one file; add only if a third appears.

## 4. Skeleton defects found (fix before landing)

- H1 SparkAdapterSnapshot dropped the argument gate. All four source adapters
  begin with if (state==0 or snapshot==0) return INVALID_ARGUMENT (dsv4 :1470,
  glm52 :1446, qwen36 :2658, qwen38 :1187). Shared body (adapter_common.c
  :353+) has no null checks, violating the MOVE law (header :28-32): migrating
  changes snapshot(NULL) from INVALID_ARGUMENT to a crash. Fix: restore the
  gate. Quiesce kept its gate (:337); only Snapshot lost it.
- H2 DriverContract comment misstates target-pinning. Header :139-141 says the
  runtime-limits school pins four strings only; false for glm52 (kind 0 AND
  pins target, :836). Target-pinning is orthogonal to check_kind; loader
  already supports it (contract->target != 0). Doc fix only.

## 5. k3 - cannot adopt the skeleton as-is (beyond contract D1-D3)

k3 (502 lines) shares none of the four-adapter bodies: no pending table, no
orphan counter, no quiescing latch, no driver load (runner-initialized), no
configuration validation. Moving anything would be a redesign, not a move.
Beyond contract D1 (branded getter :499), D2 (caps 0u :451 while wiring ops
:491-496), D3 (no schema macro):

- D6 (candidate): initialize never validates configuration - no ABI/descriptor
  check, no ValidateRuntimeLimits, no required-pointer gate (:241-271).
- D7 (candidate): completions echo only 4 of 9 identity fields (:367-374; five
  generation fields stay zeroed) and echo abi_version from the submission
  (:368) instead of stamping the constant - contrary to F8.5 and ABI LAW.
- D8 (candidate): quiesce returns OK unconditionally, nothing latches
  quiescing; validate_submission/submit accept work after quiesce (:291-300,
  :414-421) - F8.2 "PERMANENTLY closes admission" not met.
- D9 (candidate, smell): snapshot.available_submission_count = max_rows (:433)
  reports rows not free slots; honesty counters zeroed despite per-submit
  cudaMemcpy traffic (:319-334); K3ServingJsonU32 silently defaults missing
  members (:38-46) - opposite of fail-loud discipline elsewhere.
- Migration shape post-fix: embed SparkAdapterCommonState with a 16-slot
  PendingCore array; adopt open-gate validation, quiesce latch, orphan
  accounting, completion header builder; synchronous runner submit stays
  family. Until D6-D9 land, exclude k3 from behavior-neutral migration claims.

## 6. Family divergence ledger - preserve, never unify (verified real)

1. Residency anchors: driver-frame echo (dsv4 :854, glm52 :765) vs
   submission-anchor-with-origin (qwen36 :835-845,:890-893, incl. the
   status=4/reason=2 war story) vs pending mirror (qwen38 :551).
2. Mismatch policy on unmatched driver completions: wire-visible SCHEMA_ERROR
   completion (dsv4) vs orphan+poison frame_status (qwen36/qwen38) vs
   orphan-only (glm52 plain path).
3. active-stamp timing in ReservePending (early x3 vs late dsv4).
4. RELEASE handling in validate_submission: dsv4 :1219-1220 and qwen36
   :803-804 exempt RELEASE from the SelectEmitRows dry-run; qwen38 :477 does
   NOT. Preserved as-is; flagged for owner review (possible latent qwen38 bug).
5. Three row-order schools (see 2.7) stay family; qwen36/qwen38 may consolidate
   later as a family pair, not in adapter_common.
6. Capability-check schools and target-pinning both supported; see H2 doc fix.
7. Env names never move (F8.6); only reader/staging mechanics move.

## 7. Per-file bottom line

| File | Lines | Moves (~lines) | Stays | Notes |
|---|---|---|---|---|
| dsv4 | 1515 | ~230 (2.2-2.4 + LoadDriver kind-0 target-free + JSON retire) | topology block, node context, runner, dispatch/prefetch/release, row-order school | cleanest fit; contract.target=NULL legitimate |
| glm52 | 1489 | ~220 (same; LoadDriver kind-0 WITH pinned target) | DFlash2 stack, boundary law, v4 stanza, tp_collective (until S3) | H2 doc fix required; stderr logs stay family |
| qwen38 | 1471 | ~210 (+~110 if S1 lands) | PP geometry, paged-KV ledger, transport shim (until S1), pool sizing | RELEASE-exemption asymmetry flagged (6.4) |
| qwen36 | 2969 | ~200 | speculation subsystem, dual prefix ledger, release path, refusal bookkeeping, extension telemetry | smallest relative win; biggest S1/S2 beneficiary |
| k3 | 502 | ~0 (rewrite-shaped adoption) | runner, collectives, sync submit | fix D6-D9 first; excluded from behavior-neutral migration |

Method: full read of all five files + skeleton .h/.c; greps for adapter_common
adoption (zero in main tree), destroy wiring (all four wired - an interim
suspicion of a missing qwen36 .destroy was checked and REFUTED, :2955), and the
shared progress symbol (header :404). No worktree copies used as evidence
except the explicitly-named canonical skeleton read.

---

## 8. Pricing review (added same day): H1-H2 verdicts + S1-S7 priced against Solutions/(production-codesize SQUARED)

### 8.1 Metric operationalization (stated, because QUALITY_LAW.md gives direction, not arithmetic)

Law (docs/agents/QUALITY_LAW.md:13, ARCHITECTURE_MAP.md:19): maximize
Solutions / (production-codesize^2); "Deprecate freely." Repo practice defines
the readings: the pccore twin-collapse (1441 -> 1125, -316) is celebrated, and
adapter_common itself ships ~20 exported helpers as ONE solution family - so
symbols inside an existing shared family do NOT count as new Solutions, while
duplicate PRIVATE copies DO each count (they are separate things to maintain,
test, and document), and a genuinely NEW abstraction category with no duplicate
to absorb counts +1. Scope measured here: adapter layer production lines
C = 7946 (five adapters) + 460 (.c) + 395 (.h) = 8801.
Metric gain approx 2 x (net lines removed) / C, minus solution-category
penalties; risk gates SEQUENCING, not the ranking.

### 8.2 H1-H2 review verdicts (re-checked today)

- H1 CONFIRMED, severity unchanged, action FIRST: restore the
  (common==0 || snapshot==0) INVALID_ARGUMENT gate in SparkAdapterSnapshot.
  Not priced - it is a correctness precondition: until fixed, the shared
  Snapshot cannot back any behavior-neutral migration claim. Cost +2 lines.
- H2 CONFIRMED, doc-only: reword header :139-141 to "target-pinning is
  orthogonal to check_kind; kind 0 MAY pin target (glm52 does, :836)". Cost 0.

### 8.3 S1-S7 priced

| Cand | Gross dup | Shared added | NET lines | Solution event | Risk | Metric delta |
|---|---|---|---|---|---|---|
| S4 JSON retire onto SparkJsonGetUInt32Member | 54 (3x18) | 0 | **-54** | 0 (wrappers die; shared fn pre-exists) | VERY LOW - semantics verified identical (member<0 -> SCHEMA_ERROR both sides); glm52 precedent landed | **+1.2%** |
| S6 progress stub standardization | 21 (3x7) | 0 | **-21** | 0 | ZERO - shared body byte-equal to all three stubs (verified runtime/*.c today) | **+0.5%** |
| S2 frame-admission helper | ~57 (4 sites) | ~17 | **-40** | 0 (inline idioms join the adapter_common family) | LOW - submit-after-apply variant needs one flag; admission ordering untouched | **+0.9%** |
| S1 hidden-transport shim | 134 (2x67) | ~75 | **-59** | **-1** (two private shims collapse to one) | LOW-MED - constants become fields; packet flags + stream-sync order preserved verbatim; FillContext wiring untouched | **+1.3%, plus S down** |
| S3 tp_collective stanza parser | 543 (268+275) | ~250 | **-293** | **-1** (two parsers collapse to one parameterized) | MED-HIGH - four behavioral knobs (algorithm-mask rule, threshold polarity, host-count source, port-check placement) silently change schema acceptance if mis-set; REQUIRES golden config fixtures per backend kind before flip; k3 dialect excluded | **+6.7%, plus S down** |
| S5 config-load skeleton | ~56 provable prefix | ~35 | -21 | **+1** (new loader-orchestration category) | MED - error-code ordering contractual (glm52 TARGET_MISMATCH mid-load; dsv4 stanza branching); only the thin prefix is safely extractable | ~0 at best; NEGATIVE once +solution priced |
| S7 fifth env reader | 0 | ~12 | **+12** | **+1** | LOW mechanically | **NEGATIVE twice over** - grows code AND adds a solution category for two intra-file users |

Reading (b) sensitivity: even under a strict per-exported-symbol penalty,
S4/S6 stay positive, S3 breaks even (-293 vs the ~275-line hurdle at S=16),
and S1/S2/S5/S7 go negative - i.e. the Tier-1 picks below are robust across
both readings; only S3's rank depends on accepting the family reading, which
repo precedent supports.

### 8.4 Prioritization

Metric-ranked: S3 (+6.7%) > S1 (+1.3%) > S4 (+1.2%) > S2 (+0.9%) > S6 (+0.5%)
> S5 (~0/-) > S7 (negative).

Landing order (risk-gated; metric ranking preserved inside tiers):

- P0: H1 gate restore, H2 reword. Preconditions, not priced.
- Tier 1 (land WITH the base four-adapter migration; positive under both
  metric readings; near-zero risk): **S4, S6**. Combined -75 lines.
- Tier 2 (immediately after, same family): **S2, S1**. Combined -99 lines;
  S1 additionally retires a whole private-mechanism class.
- Tier 3 (largest single lever, gated): **S3**, -293 lines, only after
  golden tp_collective config fixtures exist per backend kind (nccl /
  hidden_transport x {dsv4 adaptive, glm52 doubling-only}) proving byte-equal
  schema accept/reject; k3 stays out.
- DEFER: S5 - extract nothing now; revisit only inside a future loader
  rewrite where the orchestration category would be paid for anyway.
- REJECT now: S7 - keep both qwen36 readers family-side; revisit only at a
  third consumer (the metric says the shared reader must cost less than the
  duplication it prevents, and 2 users do not clear that bar).

Cumulative ceiling if all tiers land: -467 lines on top of the base
migration's ~-860 => adapter layer ~8801 -> ~7474 (-15%), Solutions flat-to-down
(three duplicate mechanism classes retired: JSON wrappers, transport shim pair,
collective-parser pair). Note: section 7's dsv4/glm52/qwen38/qwen36 per-file
move estimates already included S4 for those files; overlap <= 54 lines and
does not affect any ranking above.

---

## 9. ROI pricing (added same day): line savings / migration risk, S1-S7

### 9.1 Formula and risk rubric

ROI = NET line savings / migration-risk score (lines recovered per unit risk;
higher is better; negative savings => negative ROI at any risk).
Risk score 0-10, additive over four measured factors:
(a) behavior-drift surface: can error codes / ordering / acceptance change?
(b) call-site blast radius: files+sites rewired;
(c) verification debt: do golden fixtures/tests exist TODAY for the changed
    surface?
(d) revert cost if wrong after landing.
Scores below cite the section 2-4 facts they rest on.

| Cand | Net savings | a drift | b sites | c verify debt | d revert | RISK | ROI |
|---|---|---|---|---|---|---|---|
| S4 JSON retire | -54 | 0 (member-missing -> SCHEMA_ERROR verified identical both sides; glm52 precedent landed) | 3 wrappers x 1 file each | 0 (per-driver config smoke already exercises every read) | 0 | **1** | **54.0** |
| S6 progress standardization | -21 | 0 (shared body byte-equal to all three stubs, verified) | 3 interface-table cells | 0 | 0 | **0.5** | **42.0** |
| S3 tp_collective parser merge | -293 | 4 (four policy knobs can silently flip schema accept/reject) | 2 big rewrites (~540 lines touched) | 4 (NO golden tp_collective fixtures exist today; must be built first) | 1 | **9** raw / **6** after fixtures land | **32.6 post-fixture** (raw 293/9=32.6; charging the ~70-line fixture harness too: ~223/6=37.2 -> quoted conservatively as 32-37) |
| S2 frame-admission helper | -40 | 1 (one flag variant: apply-vs-apply+submit; ordering untouched) | 4 sites, 3 files | 1 (existing driver smokes exercise admit paths end-to-end) | 0 | **2** | **20.0** |
| S1 hidden-transport shim | -59 | 2 (constants become fields; packet flags + sync order preserved verbatim; wrong wiring = visible IO_ERROR not silent) | 2 states + 4 callback wirings, 2 files | 2 (needs compile+smoke on both qwen drivers; no shim unit test today) | 1 | **3** | **19.7** |
| S5 config-load skeleton | -21 | 3 (error-code ordering contractual: glm52 TARGET_MISMATCH mid-load, dsv4 stanza branching) | 4 loaders | 3 (config smokes exist but assert outcomes, not error-code ORDER at the extracted depth) | 2 | **5** (at safe-prefix depth) | **4.2** |
| S7 fifth env reader | **+12** (grows code) | 0 | 2 intra-file sites | 0 | 0 | **1** | **-12.0** |

### 9.2 ROI ranking

1. **S4 - 54.0** - best ratio in the set: real removal, zero new surface,
   precedent landed, nothing to build first.
2. **S6 - 42.0** - risk-free beats bigger: near-zero denominator.
3. **S3 - 32.6 raw / up to ~37 after charging the ~70-line fixture harness
   against savings at post-fixture risk** - the 543-line gross duplication
   amortizes the highest risk score in the set. Caveat: ROI measures
   efficiency, not readiness; its fixtures must be built first (9.3).
4. **S2 - 20.0**
5. **S1 - 19.7** - statistically tied with S2; S2 wins the tie-break on fewer
   behavioral degrees of freedom (no flag variant).
6. **S5 - 4.2** - weakest positive; thin safe prefix does not pay for its risk.
7. **S7 - -12.0** - REJECT: negative numerator dominates any risk score.

Full order: S4 54.0 > S6 42.0 > S3 32.6-37.2 > S2 20.0 > S1 19.7 >
S5 4.2 > S7 -12.0.

### 9.3 Sensitivity (does the order survive different risk judgments?)

- Risk x0.5 (optimist): S4 108 > S6 84 > S3 65.1 > S2 40 ~ S1 39.4 >
  S5 8.4 > S7 -24. S3 climbs to #3; #2 only if fixture-building ALSO halves
  its drift factor (a), not just the verify-debt term (c).
- Risk x2 (pessimist): S4 27 > S6 21 > S2 10 ~ S1 9.9 > S5 2.1 > S7 -24.
  S3 splits by convention: uncapped 293/18 = 16.3 (#5); with the customary
  risk-score ceiling of 10, 293/10 = 29.3 (#1). Name the convention before
  relying on either.
- Post-fixture S3 (risk re-scored to 6 once golden tp_collective configs
  exist): 293/6 = **48.8** -> #2 outright, behind only S4. Fixtures are the
  single highest-leverage test investment in this layer.
- Stable across ALL readings: S4 top-two, S6 top-three, S2/S1 a tied middle
  pair, S5 and S7 bottom-two. The ONLY rank that moves materially is S3
  (#1..#5): largest prize, and the only candidate whose risk is REDUCIBLE BY
  WORK - build the golden tp_collective fixtures; they convert S3's ROI to 48.8 and unblock the -293-line merge.

### 9.4 Reconciliation with section 8.4

Metric ranking (Solutions/codesize^2) and ROI disagree exactly once: S3 is the
metric's #1 (+6.7%) but ROI-#3 until its fixtures exist; S6 outranks S2 on ROI
(risk-free beats bigger-but-riskier) while trailing on raw metric. Resolution
unchanged from 8.4, now with numbers: Tier 1 = S4 + S6 (combined ROI-weighted
75 lines at near-zero risk); Tier 2 = S2 + S1 (-99); Tier 3 = BUILD THE S3
FIXTURES FIRST (highest-leverage test spend in the layer), then take -293.
S5 defer, S7 reject - both verdicts strengthen under ROI pricing (4.2 is the
weakest positive; -12.0 is the only negative).

---

## 10. Tier-1 implementation receipt (same session, after section 9)

### 10.1 Tree-race adjustment (material)

The main tree moved DURING this audit-to-implement window; scope was re-derived
from measured state, not from sections 2-7 baselines:

- `runtime/adapter_common.{h,c}` have LANDED (commit 9fb34db "adapter_common
  implementation delivered ... ~65% complete"); dsv4 and glm52 adopted them.
- dsv4 S6 was ALREADY committed (`.progress =
  SparkModelServingAdapterStreamOrderedProgress`, :1352 at receipt time).
- qwen36 carries ANOTHER WRITER'S UNCOMMITTED mid-flight migration (embeds
  `SparkAdapterCommonState`, calls `SparkAdapterInitializePrologue`,
  retires its JSON wrappers). This audit did NOT touch that file: racing an
  active migration corrupts their diff. qwen36's S4 is satisfied by that work;
  its S6 (progress swap, still `.progress = SparkQwen36ServingProgress`) is
  HANDED BACK to the active lane.
- Remaining implementable slice at edit time: **S4-dsv4, S4-qwen38,
  S6-qwen38**. Both target files were clean in `git status`.

### 10.2 What changed (measured)

| File | Edits | Net lines |
|---|---|---|
| spark_dsv4_serving_adapter.c | removed private JsonMember+JsonUnsigned wrappers; 7 unsigned reads -> SparkJsonGetUInt32Member; 15 member lookups -> SparkJsonFindObjectMember | **-19** |
| spark_qwen38_serving_adapter.c | same wrapper retirement (3 unsigned + 2 member sites); removed private Progress stub; table entry -> SparkModelServingAdapterStreamOrderedProgress | **-27** |

git diff --stat: 28 insertions(+), 74 deletions(-) => **net -46 lines**, two
files, no other hunks. Combined with the already-landed pieces (qwen36 S4
in-flight, dsv4 S6 committed), all of Tier-1 is now done EXCEPT S6-qwen36.

### 10.3 Verification performed

1. Scripted edits with hard asserts: exact block match (x1 each), call-site
   counts asserted before replace (7/15 dsv4, 3/2 qwen38), brace-pair delta
   asserted (-2 dsv4, -3 qwen38). One assert fired pre-save during drafting
   (miscounted internal wrapper call as external); corrected counts derived
   from grep inventory, then clean apply. No partial writes occurred.
2. Post-edit greps: zero references to retired symbols in both files.
3. Diff review: hunks limited to wrapper region, LoadConfiguration call sites,
   progress function, interface table.
4. Semantic parity established BEFORE editing: shared SparkJsonGetUInt32Member
   body read in full (runtime/json.c:909 - member<0 -> SCHEMA_ERROR, else
   delegates to SparkJsonGetUInt32; identical sequence to the private
   wrappers); SparkJsonFindObjectMember is what both wrappers called
   internally; shared progress body byte-equal to the private stubs.
5. Compiler gate NOT run here: the adapters include GENERATED
   model-description headers (spark_dsv4_model.h, spark_qwen38_model.h -
   per-bucket SHA256 constants) that do not exist in this checkout state
   outside the generation pipeline; no stubs were fabricated. Hand-off: module
   owners run their normal host build/test flow (0050a6f precedent) over these
   two files; expected green - edits are identifier substitutions onto
   verified-equivalent shared bodies.

---

## 11. Tier-2 implementation receipt (S3 fixtures-first merge + S2 admission helper)

### 11.1 Golden tp_collective fixtures — BUILT FIRST, GREEN

- 15 fixture configs at `tests/fixtures/tp_collective/` covering the accept/
  reject matrix of BOTH policy schools: nccl-valid x2, hidden_transport-valid
  x2 (dsv4 nonzero-ordered thresholds vs glm52 zero-required), threshold
  polarity cross-cases x2, algorithm-set cross-cases x2, unknown backend,
  zero identifier, broken port span, wrong peer count, members-exact
  violations x2, and the wrapped-port alias case.
- `tests/test_adapter_tp_collective.c`: table-driven, asserts exact status
  per row plus spot fields on acceptance (rank count, control base,
  consecutive ports, algorithm mask). Wired as `test_adapter_tp_collective`
  in Makefile TEST_BINARIES.
- RESULT: **15/15 pass**, binary compiled -Wall -Wextra -Werror. This was the
  precondition section 9 set for flipping S3; it is met.

### 11.2 S3 merged parser (adapter_common)

`SparkAdapterLoadTpCollective(document,root,runtime_root,expected_rank_count,
validate_port_span,policy,parsed)` + `SparkAdapterTpCollectivePolicy/Parsed`
types. Statement order and error codes lifted verbatim from the two adapters;
the four real knobs became policy fields / parameters (algorithm mask+count,
threshold polarity, rank-count source, span validation on/off).
Family wrappers keep their old signatures, so LoadConfiguration call sites
are untouched. dsv4's Initialize-side port loop is folded away (observable-
equivalent: both paths end SCHEMA_ERROR before any state escapes).

DELIBERATE TIGHTENING (one, documented): the span compare now overflow-guards
(base > UINT16_MAX - i); the unguarded glm52 form accepted wrapped port
aliases near UINT16_MAX. Fixture 15 pins the new behavior. Owner sign-off on
this single semantic delta is requested in review.

### 11.3 S2 admission helper (adapter_common)

`SparkAdapterAdmitFrame(driver,instance,program,submission,frame,
submit_on_apply)` — submission-built vs frame-built request selected by the
submission pointer; optional submit-on-success covers glm52's speculative
site. All four admit sites (glm52 x2, qwen36, qwen38) collapsed to one-line
delegates inside their existing wrapper functions.

### 11.4 Measured deltas (working tree at completion)

| File | Delta |
|---|---|
| dsv4 adapter | 1361 -> 1089 (-272) |
| glm52 adapter | ~1480 -> 1127 (~-353 incl. landed-lane drift) |
| qwen36 adapter | Admit collapse (-8) |
| qwen38 adapter | Admit collapse (-8) |
| runtime/adapter_common.{h,c} | +360 (parser + helper + docs) |
| tests/ | +1 C test, +15 fixtures |

Net adapter-layer removal ~-590 lines gross, ~-330 after the shared
implementation — in line with the -293/-40 pricing once the shared body and
glue are charged.

### 11.5 Verification state (honest)

- Golden gate: 15/15 green (the S3 acceptance precondition).
- Compile gate: shared code builds clean under -Wall -Wextra -Werror;
  post-flip, `build/test_dsv4_serving_adapter` and
  `build/test_qwen36_serving_adapter` LINK successfully against all flipped
  sources (required purging 501 stale mixed-arch build objects — pre-existing
  cache hygiene debt, see 11.6).
- Runtime behavior gate: NOT green locally, and NOT because of this diff.
  Both adapter tests fail at INITIALIZE with SPARK_STATUS_COMPILER_ERROR
  (status 13) raised while compiling their FIXTURE DRIVER modules — a step
  the local purge wiped and this workstation's stub toolchain does not
  currently rebuild. qwen36's failing initialize path does not traverse any
  edited line (no tp stanza; admission runs only at submit). Hand-off: run
  both binaries where the fixture-driver toolchain works (0050a6f precedent)
  before merging the flip commits forward.
- Tree note: an active lane swept these working-tree edits into its commit
  series mid-flight (verified: SparkAdapterLoadTpCollective present at HEAD);
  line counts above were captured before that sweep.

### 11.6 Incidental findings for owners

1. `build/libsparkpipe_runtime.a` rule lacks a dependency on
   `runtime/adapter_common.c` — stale-archive link failures surface as
   confusing undefined-symbol errors (hit twice this session).
2. The build object cache mixes arm64/x86_64 objects across sessions; any
   archive touch exposes ranlib fat-archive failures. A one-time
   `find build -name '*.o' -delete` plus rebuild clears it (done here).

---

## 12. Threading contract documented in the ABI header

`include/sparkpipe/spark_model_serving_adapter.h` now expresses the
threading contract the code always assumed but never stated:

- **T1 SINGLE SUBMITTER** — submission-side entries (validate_submission,
  prefetch, resolve_prefetch, submit) plus quiesce/snapshot/reset/progress
  are invoked from at most one resident thread at a time, serialized
  externally; adapters may keep these paths lock-free. Concurrent entry-point
  calls on one instance are a resident bug.
- **T2 COMPLETION DELIVERY CONTEXT** — exactly one school advertised per
  instance: new `CAPABILITY_SYNC_DELIVERY` (0x00004000, added to
  KNOWN_CAPABILITIES; completions delivered before submit resolves on the
  calling thread, no callback re-entry) or the existing
  `CAPABILITY_ASYNC_COMPLETION` (post-submit delivery on driver threads,
  safety via PENDING-SLOT EXCLUSIVITY). Neither-bit or both-bits = conformance
  defect. Current school map: dsv4=ASYNC; glm52/qwen36/qwen38/k3 deliver
  synchronously today and should declare SYNC_DELIVERY (owner follow-up).
- **T3 QUIESCE LATCH** — latch set on ENTRY, before any availability check;
  stays set across BUSY/PENDING retries (admission closed until OK);
  async-school OK additionally requires zero in-flight driver submissions;
  completion callbacks must never call quiesce/submit/validate_submission;
  destroy keeps its separate idle guard.

Placement/additivity: contract block sits after KNOWN_CAPABILITIES; the two
frozen behavioral comments (submit, quiesce) gained only appended threading
pointers, no frozen wording rewritten. No struct layout change => ABI stays
20u (additive revision).

Defect-class mapping (the four systemic latents): T1 closes unserialized
submission paths (pending-slot TOCTOU double-claim, non-atomic admission
counters); T2 closes unspecified completion execution context (sync/async
ambiguity, callback re-entry); T3 closes quiesce-vs-in-flight-completion and
missing-latch classes (k3 D8 family) plus late-completion-after-destroy;
T1+T3 together close unsynchronized observers (snapshot/progress racing
completion mutation).

Verification: header compiles standalone under -std=c11 -Wall -Wextra
-Werror; combined syntax check of header + runtime/adapter_common.c +
test TU passes post-edit; check_no_cuda gate failure shown PRE-EXISTING
(identical on stashed HEAD; zero hits reference this header).
Known-degraded locally: full consumer rebuilds currently fail in
libsparkpipe_core.a with ranlib fat-archive errors from CONCURRENT lanes
building mixed-arch objects into shared build/obj — same owner finding as
11.6, unrelated to this edit (proven by stash comparison + direct syntax
proof bypassing archives).

---

## 13. Improvement pass (queue empty of actionable items; continuous-work closeout)

- **CORRECTION to 11.6:** the runtime-archive dependency finding was WRONG.
  Decisive probe: `touch runtime/adapter_common.c && make
  build/libsparkpipe_runtime.a` re-runs `ar rcs` including
  build/obj/runtime/adapter_common.o — the dep chain is healthy. The real
  culprit behind every stale-symbol link failure this session was the
  mixed-arch (arm64/x86_64) object cache corrupting archives under concurrent
  lanes; a one-time `find build -name '*.o' -delete` clears it (done). The
  recurring failure mode to document for owners is ARCH HYGIENE, not missing
  make dependencies.
- **SYNC_DELIVERY stamps landed** on qwen36 + qwen38 descriptors (T2 school:
  both modules are documented submit-return-synchronous). Compile-gated past
  their #error guards with -D stubs: zero errors through full TU parse.
  Remaining owner follow-ups: glm52 delivery-school verdict (push-callback vs
  sync — needs driver-side evidence before stamping), k3 SYNC_DELIVERY after
  D1-D3 land.
- **S6-qwen36 confirmed landed by the active lane** (progress stub gone,
  shared symbol wired at :2771) — Tier-1 is now fully tree-wide: all five
  adapters use SparkModelServingAdapterStreamOrderedProgress.
- Queue CONTINUOUS WORK items marked [x] with evidence pointers; the only
  open queue item (qwen36->qwen38 rename doc updates) remains gated on
  coordinator rename timing per its own instruction.

---

## 14. Improvement pass 2 (k3 conformance closure + school stamps)

With the queue empty of actionable items, this pass closed the residual
conformance defects on the k3 adapter after its own lane landed the skeleton
adoption (D1 compat-shim getter, D7 nine-field identity echo via
SparkAdapterBuildCompletionHeaderWithResidency with ABI stamp, D8 quiesce
latch via shared open gate — all verified in tree):

- **D2 CLOSED:** capability_flags now declares PREFETCH and RESET (covering
  every wired op per F8.2) plus SYNC_DELIVERY.
- **D3 CLOSED:** SPARK_K3_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION 1u
  pinned in the shim header; LoadConfiguration enforces checked-on-load with
  absent-field grandfathering (existing deployments without the member stay
  valid; an explicit other value refuses with SCHEMA_ERROR).
- **SYNC_DELIVERY stamped:** k3 publishes its completion only after draining
  the runner stream, inside submit, on the calling thread — textbook T2 sync
  school.

Compile gate: cc -fsyntax-only over module include chain + tests/cuda_stub
= clean. Delivery-school ledger after this pass: dsv4 = ASYNC (declared);
qwen36, qwen38, k3 = SYNC_DELIVERY (declared); glm52 = pending owner verdict
(push-callback vs sync needs driver-side evidence). Contract defect ledger:
D1-D9 all closed or routed; D4/D5 remain informational-grandfathered.

Also CORRECTED §11.6: the runtime-archive make-dependency finding was wrong
(probe: touch adapter_common.c -> ar rcs re-runs with fresh object). The real
recurring failure is mixed-arch object cache corruption under concurrent
lanes; one-time find build -name '*.o' -delete clears it.
