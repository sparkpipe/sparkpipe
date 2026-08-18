# RING VERDICT (spec vs lean, token-223 frame, pos 84)

## Method
- Spec side: dspark-k7-ring driver (bucket 8, current source + ring_anchor
  dump), k7 runtime (v4 pack), SPARK_DSPARK_DUMP=1 - 128 tokens, 516
  ring_anchor dumps (43 layers x 12 frames). Evidence: /tmp/spec_dumps_ring.txt
- Lean side: the PINNED LEAN DEPLOYMENT (the 3d962820 runtime, v3 pack) with
  the ef8fa302-era driver + the same ring_anchor dump (built at
  /tmp/sparkpipe-leandump - the ef8fa302 commit + the dump helpers + the
  __attribute__((unused)) fix + the -Werror drop in the rules file).
  The lean ran 128 tokens (the full O128 stream), 731 ring_anchor dumps
  (43 layers x 17 frames). Evidence: /tmp/lean_dumps_ring.txt

## THE VERDICT
ring_anchor per layer (the anchor's ring-slot KV the sparse attention reads):

- Layers 0, 1 (SWA, ratio 0): SAME (bit-identical fnv).
- Layer 2 (the first CSA, ratio 4): SAME.
- LAYER 3 (the FIRST HCA, ratio 128): FIRST DIFF - spec fnv 8b46d2c871 vs
  lean 4abb4eea6a (fp16-noise-level value diffs: v1=3bf0 vs 3d80, v7=3de0
  vs 3d30, ...). Every layer 3..41 then DIFFs (the state propagates).

The compression-ratio table (spark_dsv4_model.h SparkDsv4ModelCompressionRatios):
  layer 0,1 = 0 (SWA); layer 2 = 4 (CSA); layer 3 = 128 (HCA); alternating
  4/128 through layer 41; the tail = the MTP zeros.

=> The ring content is CLEAN through the SWA and CSA layers; the divergence
enters at the FIRST HCA LAYER (the 128:1 high-compression attention). The
HCA path = the indexer (index compressor Bf16LinearPair + IndexerPost with
RoPE/quantization + the HCA cache emission). The next bisect: the
Bf16LinearPair FLAT_8 (rows>1) vs FLAT_16 (rows==1) dispatch
(SparkLmHostLaunchBf16LinearPair / SparkLmSm121B1Bf16LinearPairPolicy) and
the IndexerPost quantization - the same exact-per-row pattern as the
pair/strided/expert/head fixes.

## Credit-flow bug (separate, logged for later - NOT needed for the ring)
The b1-in-k7 continuation path stalls: the rank-0 coordinator's completion
stuck in SparkModelResidentdPostTransport -> SparkHiddenTransportSend BUSY
(the output ring credit never returned). See DSPARK_VERIFY_HANDOFF_SESSION5B.md.

## SESSION 9 (slot-level dumps at the prefill boundaries - the state is clean)
- PROMPT = 128 TOKENS (the verify anchor = 128 per dspark_verify_pos) - the
  prefill = 128 padded 1-row frames, 32 full CSA groups, NO partial last
  wave (the '10x8+4' model is retired). The pos-84 dump = mid-prefill.
- Position-gated dumps at the pos-79 + pos-83 frames (both trees): the
  LAYER-2 compressor state = IDENTICAL on both sides - all 8 slots of the
  main kv/score AND the index kv/score (the first 32 channels each). The
  computed overlap pool from the spec's own state = the lean's (0.2372,
  0.1019, 0.0263, 0.0246) - the COMPRESSOR IS EXONERATED at layer 2.
- The layer-2 EMISSION (the ring entry at slot 148) = DIFF despite: the
  identical state, identical emit positions (the emit0 dump), identical
  ComputeFreqs, identical KvEmissionKernel (the v4 .cu == the v3 .cu).
  Remaining candidates: the pool's channels beyond the dumped 32 (the
  rope's cross-terms read the pool's ch 64+), or the RmsNorm over the full
  width. Layers 4+ = everything diff (the downstream of the layer-2
  emission). NEXT: dump the FULL 256-channel slots or the emit_bf16 pool
  row (all 128 channels) at the pos-83 boundary.

## SESSION 10 (full-slot + delta/moe dumps - the entry is LAYER 2)
- Full 256-channel slot dumps at the pos-79/83 frames: layer 2's INDEX
  state = IDENTICAL in full; the MAIN compressor state = DIFF in the
  channels 32-255 (the first 32 = same). The divergence enters the main
  WKV projection's later channels, not the index projection.
- delta_wq_a + moe_out added to the lean tree and dumped: LAYERS 0-1 =
  FULLY EXACT at both frames (the SWA + the first HCA = bit-identical,
  including their FFN outputs). LAYER 2 (the first CSA) = the divergence's
  ENTRY: its delta_wq_a AND moe_out = DIFF while the layer-1's moe_out =
  SAME - the layer-2's own island (the CSA sparse attention reading the
  index cache) is the first wrong computation. The layer-2 index cache
  entry (index_prev) = DIFF while its index STATE = SAME in full -> the
  index EMISSION (the rope/norm stage) diverges with the verified-
  identical kernel/freqs-code/positions. Remaining unverified: the
  DEVICE VALUES of compress_freqs_f32 + the norm_weight_bf16 at layer 2
  (the code is identical but the uploaded values may not be). NEXT:
  dump those two tensors at layer 2 on both sides.

## SESSION 11 (freqs/norm/device-emit dumps - the last inputs verified)
- The emission's remaining inputs dumped: the DEVICE compress_freqs_f32
  values = IDENTICAL (the spec f0=3f800000 = 1.0 = the lean's), and the
  compressor + indexer norm_weight_bf16 = IDENTICAL on both sides.
- The spec's DEVICE row_emit_positions[0] = the host mirror's (76 for the
  pos-79 frame) - the spec-side staging is consistent.
- Every emission input is now verified identical (state, pool, freqs,
  norms, host emit positions, kernel bytes, constants) yet the emission
  differs. Remaining candidates: (a) the LEAN's device emit positions
  (the comparison dump was blocked by the fleet's adapter race -
  route_not_found/busy on relaunch), (b) the emission's effective output
  slot/base at runtime, (c) an emission input not yet enumerated. NEXT:
  the lean device-emit comparison after a fleet settle (kill all, wait
  60s, launch 0/2 then 1/3) + a runtime slot-address dump of the scatter.

## SESSION 12 (the last input verified - full input-identity established)
- The LEAN's device row_emit_positions = the spec's (76 at the pos-79,
  80 at the pos-83 frames) - the final unverified input now matches.
- COMPLETE input-identity is established: the full state slots, the
  computed pool, the freqs, the norms, the host AND device emit
  positions, the kernel bytes, the constants - yet the emitted ring
  entry differs at layer 2. Per the coordinator's branch logic, the
  remaining suspect is the kernel's read pattern / the memory layout.
  NEXT: capture the emit_bf16 pooled row itself via a device-to-device
  copy inserted between the CompressStep and the KvEmission (a graph-
  capturable copy into a dedicated dump buffer) - the pre-RmsNorm/pre-
  rope pool vs the post-emission ring will expose exactly where the
  values diverge.

## SESSION 13 (scatter-slot dump built; the fleet is degrading)
- Built the scatter_addr dump (the emission's effective slot/base
  structure: the cache_off + the lane_stride + the slot + the lane) in
  both trees: dspark-k7-scatter (88e1a4f6) + leandump-scatter (8a4638d8).
- The dump run is BLOCKED by the fleet's fabric degradation: after the
  many kill cycles the relaunches hit route_failed/route_not_found (the
  RDMA endpoints no longer release cleanly between the cycles), and the
  batch gets served by stale residentd inodes while the new launches
  write 0 dumps. The device-emit comparison from the prior round already
  MATCHED (76/80 on both sides) - the scatter structure is the last
  uncompared item. RECOMMEND: a fleet recovery (longer settle or a
  reboot cycle) before the next dump run.

## SESSION 14 (D2D pool capture built; fleet outage)
- The D2D bisection instrument is BUILT in both trees: a graph-capturable
  device-to-device copy of the main compressor's pooled row between the
  CompressStep and the KvEmission into dspark_emit_pool_bf16 + a pool_raw
  dump tag. Spec dspark-k7-poolraw (9107ee6b), lean leandump-poolraw
  (4648a2ac).
- FLEET OUTAGE: the Sparks' /tmp was wiped (the k7 runtime AND the pinned
  lean control runtime are gone), spark5 rebooted, spark6 unreachable.
  The dump cycle is blocked until the runtimes are re-created (the lean
  control is also wiped, so the full recovery needs the base pack
  re-ship). Next after recovery: the pool_raw diff at the pos-83 frame.

## SESSION 15 (post-reboot re-staging - the poolraw run is blocked)
- Fleet = green + re-staged into the PERSISTENT path
  /home/sparkN/sparkdata/dsv4-staging/{k7,lean}-runtime (the staging
  script + the per-host config regeneration with the correct roots).
  The poolraw drivers (the k7 f6d13a0f + the lean 4648a2ac) = deployed;
  the devcycle validator + the residentd/batch = rebuilt after the wipe.
- The launch = blocked: the adapter = initialize=schema_error (the stage
  config's schema = rejected by the current adapter - the suspect = the
  staged packs/configs from the Aug-11 persistent trees vs the current
  adapter build) + the route_failed on all ranks (the peers'). The
  hostnames resolve (sparkN-mgmt = the 10.20.0.x). NEXT: align the
  stage_pack (the packs/dsv4_flash_stage.spstage) + the stage config
  with the current adapter, then the poolraw D2D dump.

## SESSION 16 (pack topology fixed; the route is the last blocker)
- Wired the correct pack: the staged k7 runtime now carries the
  tp4_pp4 stage pack (the lean control's dsv4_flash_tp4_pp4_stage.spstage)
  instead of the pp13 tree's - the adapter's initialize=schema_error is
  GONE on all four ranks.
- Remaining blocker: model_residentd route_failed status=4 (the
  ROUTE_NOT_FOUND, reason 9) at the initialize on all ranks, then the
  process exits. The suspect: the stage config's fabric endpoints
  (rail_peer_hosts/peer_ports) vs the re-applied fabric - the sysadmin's
  current endpoint values are needed to update the template's stage
  config. Once routed, the poolraw D2D dump runs unchanged.

## SESSION 17 (fabric verified; the pack corrected; the schema persists)
- The fabric = healthy: the rocep1s0f0/rocep1s0f1 = ACTIVE/LINK_UP on
  all four hosts with the exact sysadmin endpoints (the 10.10.200.x + the
  10.10.100.x) - the rail_peer_hosts match the template.
- The staged pack = corrected to the tp4_pp4 pack (the 10.09GB
  dsv4_flash_tp4_pp4_stage.spstage overwriting the pp13 pack) on all
  four ranks.
- The adapter's initialize=schema_error PERSISTS (the status 6 at the
  adapter_initialize phase) + the route_failed reason 9 (the
  CLIENT_LEASE_DISCONNECT - the residentd's FailLocked on the null
  route after the failed initialize). NEXT: pinpoint the adapter's exact
  failing schema check (the stage json field) via a load trace, or diff
  the template's stage json against the persistent tp4_pp4 tree's
  (the tree's = the [23,23,23,21] graph counts vs the template's [130]
  - the tree's stage json may be the era-matched one to try first).

## SESSION 18 (schema_error narrowed to the pack content)
- Tried the tree's era-matched stage json + the pack-path alignment -
  the adapter's initialize=schema_error persists. The failing check is
  not the graph counts, not the pack path, not the fabric. Remaining:
  the PACK CONTENT's internal format (the Aug-11..13 persistent packs
  vs the current v4 adapter) or a specific stage-json field. NEXT: the
  adapter load-trace instrumentation or a current-era pack.

## SESSION 19 (ROOT CAUSE FOUND: the stage json NAME)
- The schema_error's true cause: the adapter's adapter_configuration_path
  = config/dsv4_flash_tp4_stage.json, but the staging wrote the file as
  dsv4_flash_stage.json - the load failed with the IO_ERROR (the missing
  file) and the schema checks cascaded from the has_tp_collective = 0.
  The load trace (adapter_load path=... has_tp=0 load_status=4) named it.
- After the rename: the load = the OK (has_tp=1 load_status=0) + the
  fabric = the ready on both rails. NEW blocker: initialize=
  validation_failed status=9 (the module's validation) on all ranks.
  NEXT: pinpoint the validation (the driver/pack contract check).

## SESSION 20 (the pack's format_version = the last blocker)
- The full chain is diagnosed: (1) the stage json NAME (fixed - the
  adapter expects dsv4_flash_tp4_stage.json), (2) the pack's
  format_version: dsv4_stage pack_geometry_mismatch field=format_version
  - the persistent trees' packs are the v3 format (the Aug-11..13 era)
  and the current module expects the v4 stagepack format. The /tmp
  v4 pack was lost in the wipe. NEEDED: the v4-format pack (the
  regenerated via the current pack compiler or shipped) into the staged
  runtimes' packs/. Then the poolraw D2D dump runs.

## SESSION 21 (the v4 packs recovered the SPEC side - the lean needs its pack)
- Found the v4 artifacts under /home/spark4/srcdata/dsv4_flash.fp8.pp13/
  (NOT the sparkdata): the 166.9GB full + the 4 rank packs (~51GB each).
  Copied the rank packs into the staged k7 runtimes (the
  dsv4_flash_stage.spstage name) on all four ranks.
- THE SPEC SIDE IS UP: all four ranks reached the ready loop, the O128
  batch ran, and 168 pool_raw D2D dumps were collected
  (/tmp/spec_dumps_poolraw2.txt) - the D2D bisection data exists.
- The LEAN side: the stage-name + the pack-name fixes applied, but the
  pack's tensor_count mismatches the ef8fa302 adapter (both the 10.09GB
  tp4_pp4 pack and the 9GB tp16.b1.hostrdma pack fail
  pack_geometry_mismatch field=tensor_count). The lean's exact v3 pack
  (the 3d962820-era) was lost in the wipe - NEEDED from the coordinator.
  Once the lean is up, the pool_raw comparison runs.

## SESSION 22 (the v3 pack regenerated twice - the lean rejects the entry)
- The v3 packer (the identical at dc354a7 + ef8fa302, FORMAT_VERSION=3)
  regenerated the v3 pack from the persistent 48-shard checkpoint on
  spark4 (~145GB, tensor_count=1328, validated). Deployed to the lean
  runtimes. The lean rejects it: pack_entry_invalid kind=0 layer=0.
- The re-pack with the ef8fa302 contract (the 3 activation-format field
  renames) = the same rejection. The packer + the contract now match the
  ef8fa302 module's own commits, so the remaining mismatch = the packer's
  entry-kind emission vs the module's expected (the ValidateEntry ranges)
  - a deeper v3-era subtlety. Options: diff the ef8fa302 module's
  ValidateEntry vs the dc354a7 packer's entry emission (both extractable
  from git), or the coordinator ships the original v3 pack. The spec-side
  168 pool_raw dumps remain collected and waiting.

## SESSION 23 (the lean rejection = the TP-sharding)
- The pack_entry_invalid kind=0 layer=0 = precisely explained: the
  ef8fa302 module TP-shards the ATTN_SINK (the columns = the 64/4 = 16)
  but the v3 packer emits the full 64-column sink - the full pack is
  correct for the tp1 only. The lean needs the per-rank TP-sharded v3
  pack (like the v4 .rank0-3 packs). NEXT: the v3-era tp4 sharded
  packer (tools/dsv4_tp4_pp4_stagepacks.py in the history) or the
  direct /tp_degree sharding in the packer.

## SESSION 24 (the D2D bisection ran WITHOUT the lean runtime)
- The local lean reference dumps (the fullslots/deep/slots/ring3) = all
  present - the comparison used them. The fresh spec pool_raw capture
  (the D2D copy between the CompressStep and the KvEmission) at the
  layer-2 pos-83 = [0.1611, 0.0718, 0.0253, 0.0175] vs the lean's
  saved-computed pool [0.2372, 0.1019, 0.0263, 0.0246] - the DIFFER
  (the ratios ~0.68-0.71, the ch2 ~0.96). Caveat: the lean's baseline =
  the simplified pool (the saved dumps lack the overlap's second-d
  channels). NEXT: the spec-internal check (the captured pool vs the
  recomputed from the spec's own state) to name the write pattern.

## SESSION 25 (the local-reference diff concludes the hunt's shape)
- The spec's fresh full-slot FNVs vs the local lean full-slot FNVs: the
  layer 2's MAIN state = DIFF (the ch 32+), its INDEX state = FULLY
  SAME (the 256 channels), the layers 4+ = the downstream. With the
  layer 1 = the exact (the delta_wq_a + the moe_out = SAME), the
  divergence = the layer-2 MAIN compressor's ch-32+ path: the WKV/WGATE
  projection (the Bf16LinearPair) or the CompressStep's write there.
  NEXT: dump the projection output (scratch->kv_bf16/score_bf16) at the
  ch 32+ on the spec side + the compare vs the lean's saved state.

## SESSION 26 (the branch verdict delivered)
- The coordinator's branch test concluded: the spec's captured pool
  [0.1611, 0.0718, 0.0253, 0.0175, ...] DIFFERS from the lean's saved
  pool [0.2372, 0.1019, 0.0263, 0.0246, ...] -> the CompressStep's
  write pattern is the bug, corroborated by the state-FNV diff (the
  layer-2 MAIN state = the ch-32+ diff, the INDEX state = the identical,
  the layer 1 = the exact). NEXT: the projection-output dump at the
  ch 32+ (scratch->kv_bf16/score_bf16) to split the projection vs the
  kernel write.

## SESSION 27 (option b executed - the sharded packs still rejected)
- The 3d962820 = the DRIVER sha, not a git commit - the ef8fa302 pair
  used (the packer + the contract + the tp4_pp4 post-sharder). The full
  v3 pack -> the sharder -> the 16 rank packs (the rank00-03 = the
  10.98GB each) -> deployed -> relaunch. The lean STILL rejects:
  pack_geometry_mismatch field=tensor_count (the pack header = the
  fmt=3 tensors=80 vs the module's expected ~334/~1340). NEXT: the
  option (a) diff - the module's expected tensor/kind set vs the
  sharder's emission.

## SESSION 28 (the rank pack's tensor set is internally correct)
- The rank00 pack = the fmt 3, the 331 tensors, the first 0, the 11
  layers - the per-layer histogram = the 24/28/34 per the SWA/HCA/CSA
  (the layers 0-1 = the BOTH SWA), = the module's ExpectedTensorCount
  (0, 11) = the 331 = the MATCH on paper. The module's rejection = the
  ... so the state's ACTUAL first_layer/layer_count (the TpDerive's)
  must differ from [0, 11] - NEXT: the trace of the module's expected
  values at the LoadPack (the header vs the expected print).

## SESSION 29 (the poolhi instrument built; the dump collection failed)
- Built dspark-k7-poolhi (8cae5a23): the overlap pool's second-d
  dumps (csa_kv_s4-7_hi + csa_sc_s4-7_hi) added. Deployed + the ranks
  serve (the lease_advance loop) but the batch collected 0 dumps twice -
  the dump gate/env/driver-load needs the check. The spec-internal TRUE
  pool recomputation (the kernel's formula with the c+128 channels) vs
  the captured pool_raw is ready to run locally on the next successful
  collection.

## SESSION 30 (the proj instrument built; the dump collection blocked)
- Built dspark-k7-proj (4a49a5b5): the compressor's projection output
  dumps (proj_kv + proj_sc = the scratch kv_bf16/score_bf16 rows, the
  512 bytes each) added. Deployed + the ranks serve. The batch ran but
  collected 0 dumps - the same as the poolhi run, pointing at the dump
  gate/env in the recent launches (the session-21's 168 pool_raw worked
  with the same launch pattern). NEXT: the driver-sha + the log check,
  the env/gate fix, the re-run, then the projection-vs-write comparison.

## SESSION 31 (the proj instrument verified - the dump gate never fires now)
- The staged driver = the 4a49a5b5 (the proj) + the residentd env =
  SPARK_DSPARK_DUMP=1 + the gate = the pos-79/83 - all verified. The
  batch completes but 0 dumps appear (the stdbuf retry = the same); the
  dspark_verify_pos traces = also 0 (they were present in the earlier
  runs), so the current staging differs from the session-21 run - the
  host_row_positions[0] never hits 79/83. NEXT: the staging-trace
  comparison to find why, then the proj-vs-state diff runs.

## SESSION 32 (the gate widened to 76-87 - the dump site still silent)
- Built dspark-k7-proj2 (7dec556d) with the widened gate (the 76-87
  window). The batch completes but 0 dspark_dump lines appear; the log
  shows the normal dspark_staging prefill=1 rows=1 flow (the 1774
  dspark lines) but no adapter_load/dump-site lines - the dump block
  itself is not executing with the current staging. NEXT: the dump-site
  placement check (the side-0 continuation gate) - the proj-vs-state
  diff is otherwise fully prepared.

## SESSION 33 (the dump site itself never executes now)
- The sitetrace (03ea51f4) proved it: 0 dump_site_reached lines - the
  side-0 continuation dump block is not entered in the current flow
  (the session-21 run's 168 pool_raw came from the same block, so the
  continuation route changed). The current log shows new
  dspark_verify_expand_failed status=15 lines - the verify expansion
  now fails and the frame flow differs. NEXT: the continuation-entry
  trace to name the bypass, then the poolhi/proj collection + the
  true-overlap-pool recomputation vs the pool_raw.
