# Qwen 3.8 Flash pack lane report — 2026-08-28 (P1–P4)

Worktree /tmp/lane-qwenflash, branch lane/qwen-flash. Nodes: spark4 (build +
deploy hub), spark5/6/7 (deploy targets + smoke ranks 1-3). Model:
/mnt/model-warm/qwen3.8-flash-next rev f5d08274bafd880402bd16f5e3e6c514136ec06c,
text-only (vision skipped per brief). Contract already frozen (M1); packer +
verifier already exist (M4, commit fef7335). TP4 rank plan: 4 × 30.80 GiB.

Deployment layout (all four nodes):

```
/home/<host>/sparkdata/qwen4_flash.tp4/packs/
  qwen4_flash_full.tp4-rank{0,1,2,3}.qwen4_flashsp          33,069,091,328 B each
  qwen4_flash_full.tp4-rank{0,1,2,3}.qwen4_flashsp.receipt.json
```

4 ranks × 30.80 GiB = 123.2 GiB per node. spark6 disk after deploy: 1.3T free
(2.2T used) — inside the budget the brief set for that node.

## Milestones

### P1 Existing rank0 pack still passes — DONE

The rank0 pack from the M4 session (spark5 local, built 19:27 KST) re-verified
on spark4 against the warm checkpoint, plus sha256 identity across the
spark5→spark4 relay:

```
sha256 (spark5 and spark4, identical):
ef8bd0c95c9a0ac87c2edbee77684a792925018842a4e08d689450438da1b8be  qwen4_flash_full.tp4-rank0.qwen4_flashsp

python3 tools/qwen4_flash_pack_verify.py --pack .../tp4-rank0.qwen4_flashsp \
  --checkpoint /mnt/model-warm/qwen3.8-flash-next --tp-degree 4 --tp-rank 0 --sample 12
  trace kind=6 layer=16 fp8 relative_l2=0.02646
  trace kind=7 layer=20 fp8 relative_l2=0.02646
  trace kind=8 layer=24 fp8 relative_l2=0.02646
PASS qwen4_flash_full.tp4-rank0.qwen4_flashsp: header geometry, 899 directory
     entries (tp 4/0), 12 byte-traced samples receipt=verified
```

Same numbers as the original M4 verify (fp8 dequant traces 0.02645-0.02646
relative L2 vs the 0.2 gate).

### P2 Ranks 1-3 packs — DONE, with one real verifier bug found and fixed

Built on spark4 directly into the deploy directory (streaming reads from warm,
~600-750 MB/s; ~14-16 min per rank pack, verify <1 min):

```
qwen4_flash_stagepack slice=0+48 tp=4/2 tensors=899 file_gib=30.80 wrote ...tp4-rank2.qwen4_flashsp
qwen4_flash_stagepack slice=0+48 tp=4/3 tensors=899 file_gib=30.80 wrote ...tp4-rank3.qwen4_flashsp
ALL_RANKS_DONE Fri Aug 28 05:28:37 AM KST 2026
```

Verifies (each: header geometry, 899 directory entries, 10 byte-traced samples,
receipt verified):

```
PASS qwen4_flash_full.tp4-rank1.qwen4_flashsp: ... (tp 4/1), 10 byte-traced samples receipt=verified
PASS qwen4_flash_full.tp4-rank2.qwen4_flashsp: ... (tp 4/2), 10 byte-traced samples receipt=verified
PASS qwen4_flash_full.tp4-rank3.qwen4_flashsp: ... (tp 4/3), 10 byte-traced samples receipt=verified
```

**Verifier bug (fixed, commit 5651926).** The first rank1 verify crashed:

```
ValueError: operands could not be broadcast together with shapes (12,) (0,)
  File tools/qwen4_flash_pack_verify.py, line 263 in sample_trace (f32-widen allclose)
```

Root cause: for 1-D sharded vectors (kind 19 A_log, kind 20 dt_bias — 48 value
heads / tp4 = 12 per rank) the expected-slice code reshaped the sliced vector
to (1, N) and then fell through to the generic 2-D row rule, which re-sliced
rows 12:24 of the now single-row matrix → empty. Rank 0's slice (0:N) masked
the bug; every rank past 0 produced an empty expected vector. Never hit before
because rank 0 was the only rank ever verified. Instrumented repro (exact
failing command --tp-rank 1 --sample 10):

```
MISMATCH got (12,) want (0,)
kind 19 layer 38  entry rows=1 columns=12 weight_format=1(payload 48 B f32)
ref model.language_model.layers.38.linear_attn.A_log row_slice=(12, 12) tp 4/1
```

Fix: 1-D case now consumes the row slice and returns; 2-D row/column rules are
elif-chained. Also added a short-read guard on checkpoint reads (today's warm
mount threw one transient; it must fail loudly, not become a misleading byte
mismatch). After the fix the exact failing command passes, and rank0
--sample 10 re-verified PASS (no regression). Note: `--sample 12` happens to
stride over the entry list without ever sampling kind 19 — a `--sample 10`
(or other strides) run is the regression test for this bug.

### P3 Deployed to spark4-7 — DONE, sha-verified fleet-wide

Distribution: built once on spark4 (hub), rsynced over the 100G rail
(~575 MB/s per stream). All 16 artifacts (4 ranks × 4 nodes) hash-identical:

```
rank0  ef8bd0c95c9a0ac87c2edbee77684a792925018842a4e08d689450438da1b8be
rank1  a90252fe4f5542c13f9b70616bb838aa29a7cf5986b83ee3d344d337da558f94
rank2  70c7cda77a83e254101d18633d792102b457cc8215f80fa48a9eb2253fe38617
rank3  b3cc76c8ca37ee810f702abd8921b114ca43a7a2a0f0da8c6a6cd6bcb12f19be
(sha256sum identical on spark4, spark5, spark6, spark7)
```

Cleanup per the warm-storage discipline: no warm scratch was ever created
(builds streamed warm → local NVMe); spark5's duplicate rank0 pack
(33,069,091,328 B) and its aborted 12.8 GB rank1 tmp were deleted after
deploy; the original build logs are kept at spark5:/tmp/q4f_build_packs.log
(incident evidence) and spark4:/tmp/q4f_rank23.log.

### P4 Smoke — rank-0 loader acceptance PASS up to the documented M5 gate

No qwen4_flash residentd binary exists yet (driver compile is M5+; the module
fails initialize for tp>1 before any serving stack is useful). The smoke
therefore ran the family's own whole-stack GPU validation harness — the only
qwen4_flash-capable binary — on the rank-0 node against the DEPLOYED pack, with
the full TP collective env the module requires (SPARK_QWEN4_FLASH_STAGE_TP_{
BACKEND_PATH,IDENTIFIER,PORT_BASE,HOSTS,LOCAL_HOST}; backend =
build/libhidden_transport_spark_host_rdma_verbs.so, rail IPs 10.10.100.14-17).
Launcher: tools/qwen4_flash_tp4_smoke.sh (commit c11cea1, fully
parameterized — host/rank/hosts/port/identifier/packs-dir).

Result on spark4 (rank 0, deployed pack):

```
qwen4_flash_stage tp_whole_stack_pending degree=4 (embedding/head shards need the collective port)
qwen4_flash_stage initialize_failed status=19
qwen4_flash_validation failure=module_initialize status=19
qwen4_flash_validation check=decay_gate       elements=192     relative_l2=7.56e-08 cosine=1
qwen4_flash_validation check=write_gate       elements=192     relative_l2=4.09e-08 cosine=1
qwen4_flash_validation check=conv_update      elements=40960   relative_l2=1.653e-3 cosine=0.999998634
qwen4_flash_validation check=gdn_step_output  elements=12288   relative_l2=1.687e-3 cosine=0.999998578
qwen4_flash_validation check=gdn_step_state   elements=1572864 relative_l2=6.06e-08 cosine=1
qwen4_flash_validation check=gated_norm       elements=24576   relative_l2=1.664e-3 cosine=0.999998615
qwen4_flash_validation check=attn_decode      elements=6144    relative_l2=1.680e-3 cosine=0.999998589
qwen4_flash_validation check=gdn_chunk_output elements=786432  relative_l2=1.655e-3 cosine=0.99999863
qwen4_flash_validation check=gdn_chunk_state  elements=786432  relative_l2=1.12e-07 cosine=1
```

Reading: no status=1 (collective-env rejection — the five env vars are now
supplied and accepted), and NO pack errors (no pack_geometry_mismatch, no
pack_entry_invalid — the Q4SP/narrowed-shape acceptance from ccf9140 holds on
the deployed bytes). Initialize walks env parse → pack load → and stops
cleanly at the fail-closed M5 guard (`tp_whole_stack_pending`, UNSUPPORTED=19)
that the previous lane planted for exactly this boundary. All nine kernel
oracle checks pass at the M3 thresholds on the same run. This is the maximum
signal the current module can produce: the embedding/head collective port
(driver lane M5) is the one remaining gate in front of a live rank-0 daemon.

Ranks 1-3 harness runs: the retained GPU recipe pins the whole-stack tier to
TP_RANK=0 by design (validate_qwen4_flash_resident_decode_stage_cuda.sh:82,
"cross-rank numerics gate at the band E2E run"), so ranks 1-3 exit at the
config check with `requires SPARK_QWEN4_FLASH_TP_RANK=0, got 'N'`. Their packs
are covered by the byte-trace verifies (P2) and fleet-wide sha identity (P3).

## INCIDENT / blocker for the coordinator

**spark5's Ceph client on /mnt/model-warm is wedged.** Evidence gathered
05:00-05:20 KST, all from spark5 while spark4/6/7 read the same pool fine:

```
spark4/6/7: dd 64 MB from model-00001-of-00131.safetensors → 596-757 MB/s
spark5:     dd 128 MB same file → 439 kB/s (305 s)
spark5:     packer pid stuck 12+ min in D state on folio_wait_bit_common
            (166.4 GB read over 8.5 h ≈ 5.4 MB/s average before the stall)
spark5:     small reads OK (config.json in 4.7 ms) — bulk throughput dead
```

Casualty: the M4 session's rank1-3 build loop on spark5 — rank1 sat at 46 %
for 8.5 hours. I killed it (clean TERM; "Terminated / PACK FAIL rank 1" in
/tmp/q4f_build_packs.log), deleted the partial tmp, and moved all builds to
spark4. spark5 is otherwise healthy (ssh, NVMe, network, GPU all fine) — its
DSV4 residentd workload was untouched and is NOT mine to touch. The node did
NOT need a reboot and none was issued. The ceph client state (or mount) needs
coordinator attention; a healthy spark5 matters for future warm-side work.

## Honest negatives

  * The 4-node TP4 collective could not be exercised live: every rank owns an
    embedding shard at tp>1, so the module's fail-closed guard fires before
    any collective connect. A live 4-way initialize is possible ONLY after the
    M5 collective/embedding/argmax port — the packs and collective env are
    ready for it.
  * The pack size (30.80 GiB/rank ≈ 123 GiB total) covers only the mapped
    tensor classes. The unmapped checkpoint classes (hyper-connection mixers,
    attention indexer, layer-1 PLE) still need module support (previous
    report's integration requests 2-4 stand); when they land, packs must be
    rebuilt and redeployed.
  * Pack build determinism: receipts pin source shas; I did not verify
    bit-identical rebuilds across nodes (single-node build + byte-identical
    distribution makes it moot for this deployment).

## INTEGRATION REQUEST

  1. (driver lane / M5) TP collective + replicated-embedding + sharded-argmax
     port — unchanged from the previous report. Everything on the pack side
     (4 verified deployed ranks, collective env shape, backend .so built on
     all 4 nodes) is now in place and proven up to that gate.
  2. (driver lane) Per-rank retained GPU validation recipes, or the band E2E,
     to put module-level coverage on ranks 1-3 (the current retained recipe
     pins TP_RANK=0 by design).
  3. (coordinator) spark5 ceph client wedge — see incident above.
  4. (coordinator) tests wiring for tests/test_qwen4_flash_model_header.py —
     unchanged from the previous report.

## Commits (this lane session)

  * c11cea1 tools/qwen4_flash_tp4_smoke.sh — parameterized TP4 rank-pack smoke
    launcher (collective env + whole-stack harness), exec bit included.
  * 5651926 tools/qwen4_flash_pack_verify.py — rank>0 f32-widen trace fix +
    checkpoint short-read guard.
  * (this commit) lane report + regenerated PACKAGE_MANIFEST.json/SHA256SUMS.

## Next

  1. Driver lane M5 lands the collective/embedding/argmax port → rerun the
     smoke (same launcher) expecting initialize to pass and the full check
     ladder to run, then the residentd deployment proper.
  2. When the unmapped tensor classes (hc mixers, indexer, PLE) gain module
     support: rebuild all 4 ranks (packer already enumerates them as unmapped)
     and redeploy; budget grows past 30.80 GiB/rank.
  3. spark5 rejoin: once its ceph client is healthy, it needs no repack — the
     packs are already deployed locally; just re-verify sha if the mount work
     touched the node.
