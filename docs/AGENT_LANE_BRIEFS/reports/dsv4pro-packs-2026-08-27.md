# DSV4 Pro TP4xPP4 pack lane report — 2026-08-27

Worktree `/tmp/lane-dsv4pro`, branch `lane/dsv4pro`. Build node spark6.
Source `/mnt/model-warm/deepseek-v4-pro-0813-ga` (GA 0813, 66 shards, 832G).
Deploy targets: spark0, spark3..sparkf (14 nodes; spark1 restarting, spark2
prod — rank 1/2 packs built, verified and stashed on warm for later deploy).

Context: all 16 prior Pro TP4PP4 packs (built Aug 17; only rank1/rank3 still
on disk) were marked FAILED after the PR 721 contract change. Per the audit
decision this lane REGENERATED every rank pack from the warm checkpoint
against the current `model_contracts/dsv4_pro.json` and redeployed.

## Milestones

### P1 Source identity — DONE

The Pro contract carries no pinned source manifest (by design — the packer
records provenance instead), so identity was verified three ways:

1. Archive provenance: warm `ARCHIVE-RECEIPT.json` pins
   `source_manifest_sha256`
   `2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a`;
   the live index hashes to exactly that:

   ```
   2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a  model.safetensors.index.json
   9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0  config.json
   ```

2. Live config.json geometry == contract, every field checked:
   `DeepseekV4ForCausalLM`, hidden 7168, 61 layers, vocab 129280, 128x512
   attention, kv_heads 1, q_lora 1536, o_groups 16 / o_lora 1024, 384+1
   experts top-6, moe_intermediate 3072, fp8_e4m3 + ue8m0 block 128x128,
   expert_dtype fp4, dspark targets [58,59,60] / block 5 / noise 128799 /
   markov 512, 64-entry ratio table, rope/yarn/swa/index params all match.
   `DOWNLOAD_STATUS.json`: complete, 93 files, 892762507005 bytes. Index
   cross-check: 149782 tensors, 66 shards, index total_size 892727580904 vs
   shard bytes 892744322880 (delta = per-shard safetensors header overhead).
   Note: config declares `num_nextn_predict_layers: 1` while the checkpoint
   ships three `mtp.*` draft namespaces; the GA reconciliation (a837880) and
   the packer's PRO_MTP_PACKED=3 pack from checkpoint reality.

3. Contract pinning: `python3 tools/generate_dsv4_contracts.py --check`
   exit 0 (regenerated contract byte-exact with the authoritative source).

Packer pre-flight (`--inspect`, safetensors headers only): all 149,782
source tensor metadatas validate against the 1,975 Pro records in 1.6s.

### P2 Full pack + 16 rank packs — DONE

Chain (both packers are needed for Pro): `tools/dsv4_pro_stagepack.py`
builds the full 61-layer pack from the checkpoint;
`tools/dsv4_tp16_stagepack.py --model pro --tp-degree 4 --pp-stages 4
--pp-stage s --rank r` (the engine under `tools/dsv4_tp4_pp4_stagepacks.py`)
shards it per physical rank. world_rank = pp_stage*4 + tp_rank; rank r lives
on spark{r}. Layer slices 0+16 / 16+15 / 31+15 / 46+15.

Full pack on spark6 local NVMe (streamed read from warm at ~480-500 MB/s,
~24 min write + hash):

```
bytes 892904019728  tensors 1975  layers 0+61
codecs fp8_e4m3 / mxfp4_e2m1 / bf16
source_index_sha256 2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a
sha256 0b1692a3ebd589572db154b2b28e0101acefc5c14e9b530cd53a2022bbab759c
python3 tools/dsv4_pro_stagepack.py --verify-pack dsv4_pro_full.spstage
  -> validated: true (0.08s)
```

Each rank pack: shard -> `--verify-output` (geometry + directory + bounds
against the full source pack) must PASS -> rsync -> sha256 on target must
equal the receipt sha. One rank at a time on spark6 (~10 min each). Rank
packs are ~88 GiB (93-100 GB), not the ~52G the brief estimated: every rank
replicates the full DSpark draft block (3 draft layers + heads).

Fleet result (all 16 receipts; per-target sha equality verified at deploy
time on every node):

```
rank  0 layers 0+16  tensors 572 bytes 99603890892 sha 490c5cdc9cb2863ca1c1a576bc33b3550bb613de1351b2c057e1d23a4d73d90b  -> spark0
rank  1 layers 0+16  tensors 572 bytes 99603890892 sha e9de954be40628dcc4f0d2d88fe1b2391b50ba7910ca9a8e5ba249b6a53b9879  -> warm stash (spark1)
rank  2 layers 0+16  tensors 572 bytes 99603890892 sha acc6d761cb79174abe2ec8e8b53e8bf38ca98dde87a2ad939d3beccd95ad5863  -> warm stash (spark2)
rank  3 layers 0+16  tensors 572 bytes 99603890892 sha 1071e9ad32b8915a3a6bd19f6b6baa0524ce45dbf4aa844b7fb7d8eb2db985c9  -> spark3
rank  4 layers 16+15 tensors 549 bytes 94283835100 sha db58f92a991f170bcc3eea899479982f67708114602e1d4620ddb1870f9fc163  -> spark4
rank  5 layers 16+15 tensors 549 bytes 94283835100 sha 06487bad7933d0723b66da622d27865f97c1fc60ce1e6883e72263dd20bdee5a  -> spark5
rank  6 layers 16+15 tensors 549 bytes 94283835100 sha a5df643077165bf6f51a3ab7eee316306c3d3a2c2f85ed45e1e3b94d6e4063c1  -> spark6
rank  7 layers 16+15 tensors 549 bytes 94283835100 sha ff056efeba3646ad381f4df557792d6ddb229786db7f64741e52a5f3acbdf699  -> spark7
rank  8 layers 31+15 tensors 543 bytes 94264972524 sha c0d33b6c69b4267fddee472197ed821a1c8ae9af4a825af880d9647291f6ba8c  -> spark8
rank  9 layers 31+15 tensors 543 bytes 94264972524 sha 93dfa14e286d5e4a7e12124df65155be9a77df7a6ce3029ea318d5cd5712c34f  -> spark9
rank 10 layers 31+15 tensors 543 bytes 94264972524 sha 95e336005ae791f653ffbc47f78ec7601f04373d078329a2e3a596f2ca45615d  -> sparka
rank 11 layers 31+15 tensors 543 bytes 94264972524 sha b48802b527b857d53896b8fbc8b54953b6f28ea8717c788940a8768a47b11765  -> sparkb
rank 12 layers 46+15 tensors 554 bytes 94748565432 sha 161664ca08c97716a6ca7cb3b825133454891c8eb27c233e3ca303faaadc4444  -> sparkc
rank 13 layers 46+15 tensors 554 bytes 94748565432 sha 7652b4ef0604f49f7990136beb203f4b3ea279ac9fbfd2c4e127d8ff9c4cf2b5  -> sparkd
rank 14 layers 46+15 tensors 554 bytes 94746730424 sha 31e81d7b9f651c2bf67440e837e2cdc5246dafa3e3637b4273019de401a0797a  -> sparke
rank 15 layers 46+15 tensors 554 bytes 94746730424 sha 889276e4b12fadf070e8fecf3f4d77185e836cb475d2eae227013bfbebd8a962  -> sparkf
```

**Old-pack drift finding.** The regenerated rank3 pack is BYTE-IDENTICAL to
the Aug-17 pack it replaces (both sha256 `1071e9ad...859c9`) — the pack
bytes were never wrong, and the rebuild is deterministic end-to-end. The
post-PR721 "verifier failure" reproduces as
`dsv4_stagepack: stage-pack header does not match the model contract` when
the FULL-pack contract verifier is pointed at a TP-sharded rank pack: it
re-derives full-width records for the slice, so it cannot pass on any rank
pack by construction. The real rank verifier is `--verify-output` against
the full source pack (PASS on all 16 here). What PR 721 actually changed
was deployment-side (`max_sequence_positions` 4096 -> 33024); see P4 for
the config bug that is the likely true cause of the Aug-era failures.

### P3 Deploy — DONE (14 of 16 nodes)

Per target: `bin/{sparkpipe_model_residentd,sparkpipe_model_batch}` +
`lib/{model_driver.so,libdsv4_pro_tp4_pp4_serving_adapter.so,
libhidden_transport_spark_host_rdma_verbs.so}` (relayed from the Aug-17
spark3 deployment — the only existing Pro binaries), the repo's
`config/dsv4_pro_tp4_pp4_stage.json` (@ PR 721, 33024 ceiling, plus the
model_revision correction described under P4), a freshly generated
`config/model_resident.json` (16 nodes, runtime_root
`/home/<host>/sparkdata/dsv4_pro.tp4pp4`; the stale one on spark3 pointed
at `dsv4_pro.fp8.tp4_pp4.b1024`), kvcache dirs pre-created, then pack +
receipt with on-target sha equality. Tools (committed, parameterized):
`tools/dsv4pro_scaffold_deploy.sh`, `tools/dsv4pro_rank_deploy.sh`,
`tools/dsv4pro_smoke_fleet.sh` + `tools/dsv4pro_smoke_launch_rank.sh`.

spark1 (restarting) and spark2 (prod) untouched. Ranks 1/2 stashed at
`/mnt/model-warm/packbuild/dsv4pro/rank{1,2}.spstage` (+receipts).

One transient failure, recovered: rank 3's deploy died on an ssh broken
pipe mid-rsync (`client_loop: send disconnect: Broken pipe`,
`rsync: connection unexpectedly closed`); the pack was already built and
verified, so it was re-rsynced manually with the same sha check
(`RANK3_MANUAL_DEPLOY_OK`, sha equality confirmed).

spark6 disk note: the brief said 3.2T free; actual was 1.3T at lane start.
The 893G full pack + one-rank-at-a-time discipline fit (peak 98%, 71G
free); scratch deleted after verification (985G free at lane end).

### P4 Smoke — PARTIAL: two real bugs found and one fixed; pack load blocked by the Aug-17 driver binaries

Target: rank-0 daemon loads the deployed pack ("model_residentd ready").
What actually happened, layer by layer:

1. **Init order (source-proven).** residentd initializes
   transport_open -> host_storage -> cuda_storage -> wake_pipe ->
   adapter_initialize (pack load) -> ready line
   (node/model_residentd.c, SparkModelResidentdInitializeResources). The
   module connects the TP collective BEFORE SparkDsv4ModuleLoadPack
   (module.c: Configure -> InitializeTpCollective -> PrepareState ->
   LoadPack). So no daemon can load a pack until its boundary routes
   (rank r <-> r+4 chain) and its 4-rank TP group are up: the fleet must
   start together. A lone rank-0 daemon dies after 120s:

   ```
   hidden_spark_rdma_fabric_ready local_host=spark0-fabric device=rocep1s0f1 port=1 gid_index=3
   hidden_spark_rdma_open_timeout route=rank0_to_rank4_hidden role=sender host=spark4-fabric port=61704 waited_ms=120000
   model_residentd initialize=busy status=15 phase=transport_open rank=0 stage=0
   ```

2. **Fleet attempt 1 (all 14 nodes together): adapter schema_error.**
   Boundary transports connected (all peers present), then every rank
   failed `initialize=schema_error status=6 phase=adapter_initialize`.
   Root cause: the adapter requires config `model_revision` to equal its
   compile-time `SPARK_DSV4_PRO_SOURCE_REVISION`
   ("GA release deepseek-ai/DeepSeek-V4-Pro-0813 (HF, 2026-08-13)"), but
   `examples/deployments/dsv4_pro_tp4_pp4_stage.json` has carried the
   placeholder "official Hugging Face config observed 2026-07-31" since
   its creation (c9ff8c5, Aug 15) — and PR 721's test
   (tests/test_dsv4_pro_exact32k_stage.py) pins that wrong value. The Pro
   TP4PP4 deployment could NEVER have passed adapter config validation
   with this file; this is very likely the real "all 16 packs failed"
   mechanism. Fixed on all 14 deployed configs (sed model_revision ->
   the Pro GA revision string); repo fix is an INTEGRATION REQUEST below.

3. **Fleet attempt 2: validation_failed in the Aug-17 binaries.** After
   the config fix the failure moved to
   `model_residentd initialize=validation_failed status=9
   phase=adapter_initialize rank=N stage=N` on every rank. Exhaustive
   static analysis of every VALIDATION_FAILED site reachable in that
   phase (adapter SparkDsv4TpDeriveNodeConfig / collective topology
   slice; module shape-derivation/hash check) shows all inputs are
   consistent for TP4xPP4 Pro, and every relevant source file
   (adapter, parallel_shape, tp_device_collective, firmware header, pro
   model header) is byte-identical between the binaries' vintage
   (d30725f, Aug 17 13:56) and current main — the only pro-header delta
   since is additive (5add90b). The deployed .so's strings confirm a Pro
   TP4PP4 build ("spark.dsv4.pro.serving-adapter.tp4-pp4.v1"). Conclusion:
   the Aug-17 adapter/driver pair cannot complete initialize against the
   current deployment; the driver lane must rebuild bin/lib from current
   main (module publish + sparkpipe_model_compile with
   examples/model_descriptions/dsv4_pro_resident_decode_stage_firmware.json)
   — the same boundary the qwen-flash lane documented (their M5 guard).

   Maximum signal achieved: the deployed packs were never implicated at
   any layer (config parses, geometry validates, hashes match); the
   remaining blocker is binary-side and owned by the driver lane.

4. **Operational collision (flagged).** During smoke cleanup my TERM/KILL
   used the process NAME `sparkpipe_model_residentd`, which also matched
   an unrelated concurrent deployment `/tmp/dsv4bisect-main` running ranks
   0-3 on spark4-7 (another actor; cwd/exe /tmp/dsv4bisect-main). Their
   harness relaunched them (observed live). Lesson recorded: kill by cwd
   match, not by name. No lane daemons of mine remain (verified by
   /proc/*/cwd across all 14 nodes).

## Cleanup

spark6 scratch deleted after all verification (full pack 893G + rank
scratch; receipts and logs kept under /home/spark6/sparkbuild/dsv4pro/);
spark6 back to 985G free. Warm footprint: 2 x 93G stashed rank packs +
receipts at /mnt/model-warm/packbuild/dsv4pro/ (delete after ranks 1-2
deploy). No daemons left running by this lane.

## Blockers / integration requests

1. **INTEGRATION REQUEST (one-line repo fix + test fix).**
   `examples/deployments/dsv4_pro_tp4_pp4_stage.json`:
   `"model_revision"` must be
   `"GA release deepseek-ai/DeepSeek-V4-Pro-0813 (HF, 2026-08-13)"`
   (currently "official Hugging Face config observed 2026-07-31"), and
   tests/test_dsv4_pro_exact32k_stage.py's assertion of that string must
   be updated to match. Without it no Pro TP4PP4 daemon can init.
2. **Driver lane: rebuild the Pro TP4PP4 runtime from current main and
   redeploy bin/lib** (deployed pair is Aug-17 vintage; fails
   validation_failed at adapter_initialize after the config fix). Then
   rerun `tools/dsv4pro_smoke_fleet.sh --hosts "...14 hosts..."` — the
   ready line is the acceptance gate.
3. **spark1/spark2 availability gates the full TP4PP4 fleet.** Ranks 1/2
   packs are stashed on warm; rank 0's TP group is {spark0..spark3}, so
   group 0 cannot load packs until both nodes are serviceable (spark2
   needs a coordinator-approved window; it is prod).
4. Coordinator awareness: a concurrent `/tmp/dsv4bisect-main` deployment
   (DSV4 daemons, ranks 0-3) is active on spark4-7.

## Next experiment

After the driver rebuild lands: fleet smoke (all available ranks
simultaneously), then the PR 721 capacity claim becomes testable
(33024-position prompt+decode on group-complete ranks).
