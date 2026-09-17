# DSV4 Pro single-spark GPU validation receipts

Both runs use the pro module archive (bucket 1, MXFP4-E2M1 experts, FP8-E4M3
non-expert weights, BF16 non-expert activations, BF16 KV, sm_121a) against
packs sliced from the authoritative checkpoint at
/home/spark3/extnvme/models/hf/deepseek-ai/DeepSeek-V4-Pro.

| Field | Val4 run (round 4) | Valtail run |
| --- | --- | --- |
| Host | sparkb | sparkb |
| Validator | /tmp/dsv4pro-validator (pro, b1) | /tmp/dsv4pro-validator-tail |
| Module archive sha256 | (same archive) | 9285c8c95c6a9cb845af5af720b4b4782613d7fac41a4e88f25b4a080c09c8cc |
| Pack | dsv4_pro.val4.spstage | dsv4_pro.valtail.spstage |
| Pack slice | 0+4 | 57+4 |
| Pack sha256 | (val4, 57382440264 B) | 520de31e50ba9b268cc77289cd8dec29ebc9933028c164e39cdb4de3ece6c0bb |
| Pack size | 57382440264 | 73222567936 |
| stage | 0/16 | 1/2 |
| rows | 1 (bucket 1) | 1 (bucket 1) |
| max_seq | 4096 | 4096 |
| logical/physical pages | 1024/1024 | 1024/1024 |
| device memory | 56.2 GiB | 72.1 GiB |
| Result | PASS | PASS |
| config sha256 | 05e488c5e5405416714273808464125aac7788cf074adadb7672d2ce26e1358d | 68a63375bc7033bbc726b3241e7e7ed99c69a9d69e01da6ef411b6be0bc6b65e |
| nonzero_hidden | 28672 | 0 |
| output_token | 4294967295 (no head in slice) | **48774** (first real Pro token) |
| val4 pack sha256 (measured) | 0ac4d053b2bb8cee300225c2add88c4bac66a79e1d11acf73a48a21b3d2481d4 | (n/a) |

Config hash scheme (deterministic, reproduced with shasum -a 256 over the
exact string): for the valtail run the string was
`dsv4_pro validation stage=1/2 slice=57+4 rows=1 max_seq=4096 logical_pages=1024 physical_pages=1024 mtp=0 graphs=0`.

## Coverage meaning

- **Val4 (0+4)**: HcMix HCA layers (0-1), CSA layers (2-3), hash-routed
  gates (tid2eid), first bias gate, indexer, embedding, router sort.
- **Valtail (57+4)**: final norm, LM head, HC head params, the complete
  packed MTP record set (E/H projections, norms, HC head, full MTP layer
  records) plus the MTP-only embedding copy; deep CSA layers. First token
  ever produced by the Pro module through the real final head: 48774.

## TP4xPP4 staged-artifact identities (round 6, all verified)

Full pack (/home/sparkb/sparkdata/dsv4_pro.full.spstage):
- sha256 a15455ad83b1e5f3846d31dfcc349847e84bb2275907a570388407763ddba592 —
  matches the pack-build log /tmp/dsv4pro-pack.log exactly (unchanged since build).
- --verify-pack PASS: 1926 tensors, 61 layers, codecs fp8_e4m3/mxfp4(7)/bf16,
  864875157944 bytes.

Shipped rank packs (one per host, /home/{host}/sparkdata/dsv4_pro.tp4pp4/packs/
dsv4_pro_tp4_pp4_stage.spstage) — sha256 first-16 match vs split-time values in
/tmp/dsv4pro-split2.log, 16/16:

| rank | host | sha256 (first 16) |
| --- | --- | --- |
| 0 | spark0 | 2be8aa0a9258be00 |
| 1 | spark1 | 0e9f015877bde140 |
| 2 | spark2 | 24821d5736da788c |
| 3 | spark3 | 197348a90316e1b3 |
| 4 | spark4 | 6d39fad9fe6dab2f |
| 5 | spark5 | 81c7468c171cd63a |
| 6 | spark6 | a50eb96b3a40decb |
| 7 | spark7 | f2aeb49d9c7cafd3 |
| 8 | spark8 | 608f9bb5a6e838e5 |
| 9 | spark9 | 298243fe3a0d8a11 |
| 10 | sparka | d47dff6de53c8e40 |
| 11 | sparkb | 84986831237272be |
| 12 | sparkc | bfe6f618061bcb58 |
| 13 | sparkd | 89b54c62f5fbf721 |
| 14 | sparke | a78333103ca57efe |
| 15 | sparkf | d53de7845573d789 |

Deployment configs on all 16 hosts are byte-identical to each other and (for
the stage topology) to the repo file:
- config/dsv4_pro_tp4_pp4_stage.json sha256 7bdc343786885f1a5b3f2d78acfe6791105ca97335e0662ab95ce1084a34e57a
  (== examples/deployments/dsv4_pro_tp4_pp4_stage.json)
- config/model_resident.json sha256 23f9729a669456125e619dadf9108e27db6f20ce1428332ed7c8765c7d93c54e

Regeneration proof (tools/devcycle/verify_rank_packs_pro.sh on sparkb,
log /tmp/dsv4pro-regen-verify.log): every rank pack is re-sharded from the
current full pack and sha256-compared with the shipped bytes. COMPLETE:
16/16 MATCH, exit=0 (round 10). Full shipped sha256s:

| rank | sha256 |
| --- | --- |
| 0 | 2be8aa0a9258be00a5e255752d870395037804d7f1de5073a61c03d5bce18ed7 |
| 1 | 0e9f015877bde1400c2e36b352e646b0792c645e09f3014597f9cbbaefb75512 |
| 2 | 24821d5736da788cc1177a9265f9463f86a6e434491c90ba3b3b512138122418 |
| 3 | 197348a90316e1b3652e07a333c0d818c3440204d93e2fa57855bd3ff787e0f7 |
| 4 | 6d39fad9fe6dab2fd03ab72e93047412e18ee33a774dd13245d6a13f119519f7 |
| 5 | 81c7468c171cd63a9e5bcb8fa510e3ed2f6cb1bbca417c827edcf45c7e16b96a |
| 6 | a50eb96b3a40decb24454fceab9414492da78c01bf1ab7fd5cd6e4e82bbc4332 |
| 7 | f2aeb49d9c7cafd32542f045ae40a73cf9a0a96956b155203939921a639bfa73 |
| 8 | 608f9bb5a6e838e53d8110f96d0aceaa59ea3b37569d5c48ca8eb74d54787472 |
| 9 | 298243fe3a0d8a110fdc27391848fcba867021d23dbafbea4b5526df8f97827b |
| 10 | d47dff6de53c8e40c6cb7b7aa3412f79e9f0677db95e3a40d9dd688d2c375c69 |
| 11 | 84986831237272be59b3c9003f2d39bcb04bd12d510523a5e29978544d732c84 |
| 12 | bfe6f618061bcb58f3aa2e69aeeab4ba9f11e911836e4876ee6c0735ec7738e6 |
| 13 | 89b54c62f5fbf7213c7e366b5b903e68165f3dff2d9dfa1a6797a3fbaa4e4ce3 |
| 14 | a78333103ca57efeb52a7267d1400492b016965f0e3bd4c1b9899d749134460d |
| 15 | d53de7845573d7891622016d53486f5dbdbd5ab62fbe1962a544106aae03e64e |

The sharder on sparkb is byte-identical to the worktree version
(md5 2f9517ad4f8531f85cb1712922b47df9).

## Single-rank residentd boot test (round 7, sparkb rank 11)

The real residentd path (not the validator) was booted solo on sparkb against
the production deployment. Three blockers were found and fixed; the boot now
reaches the peer-connect stage and exits cleanly (busy) only because the
other 15 ranks are absent:

1. **Lib names**: config expected lib/libdsv4_pro_tp4_pp4_serving_adapter.so
   and lib/libhidden_transport_spark_host_rdma_verbs.so; deploy had shipped
   generic names. Renamed on all 16 hosts + deploy_pro.sh fixed.
2. **Stale adapter stage-layer table**: SPARK_DSV4_SERVING_STAGE_LAYERS was
   {11x12, 10x4} (172-layer leftovers); the runtime descriptor check
   (per-TP-group counts must sum to layer_count=61) rejected the adapter with
   invalid_argument. Fixed to {16x4, 15x12} — matches TpDeriveLayerSlice
   (16/15/15/15) and the stage JSON graph counts [49,46,46,46] (3L+1).
   Adapter rebuilt + shipped to all 16 hosts.
3. **RDMA local identity**: the host-rdma transport resolves transport_host
   for local GID discovery; "sparkN" resolves to the mgmt IP first (no GID on
   any 100G port). Added topology.transport_hosts to the spec + generator;
   nodes now use sparkN-200g (10.10.100.x, the 100G RoCEv2 port) while the
   control endpoint stays on the mgmt name. Config regenerated + deployed.

Boot log milestones (all passed):
adapter_load -> deployment_validation -> runtime_limits -> transport_contract
-> rank_plan -> transport_load -> transport_open ->
`hidden_spark_rdma_fabric_ready local_host=sparkb-200g device=rocep1s0f1
port=1 gid_index=3` -> boundary connect to rank 7 (absent) -> busy timeout
(120 s) -> clean exit.

KV backing dirs (/home/{host}/kvcache/dsv4_pro/tp4pp4.bf16) created on all 16
hosts (were missing); preflight now checks adapter/transport libs + KV dirs.
Preflight after fixes: 16/16 ready.

## 16-host fabric sweep (round 8)

tools/devcycle/fabric_probe_pro.sh (run with the spark alias) on all 16 hosts:

- Every host has exactly 2 active verbs ports with the expected GIDs:
  ::ffff:10.10.200.{rank} (200G rail) and ::ffff:10.10.100.{10+rank}
  (100G ring). The second 200G NIC pair (roceP2p1*) is DOWN fleet-wide —
  the rails in the stage JSON only reference the UP ports, so this matches
  the topology.
- Control/collective ports (20480, 64620-64635) free on all 16 hosts.
- Alias gap found and fixed: sparkN-200g was missing on sparkd/e/f (their
  hosts files only carry sparkN-fabric). transport_hosts switched to
  sparkN-fabric, which resolves to exactly one 10.10.100.x address on all
  16 hosts. Config regenerated (sha256
  0c7103acc7ab59b872bef10db450711c4ca4fb94d81d6017efcccab36c38780b) and
  redeployed; boot-tested on sparkb (0-c host) and sparkf (d-f host) - both
  reach hidden_spark_rdma_fabric_ready.
- Rails in the stage JSON: rail 0 = 10.10.200.0-15 (paired 200G links, used
  by split-ring step 0), rail 1 = 10.10.100.10-25 (all-to-all 100G /24,
  steps 1-2); step_rail_indices [0,1,1]; rail_count=2 satisfies the
  counter-rotating split-ring validation.
- Rail/edge device selection verified by code reading (round 9):
  SparkTpDeviceCollectiveRankHost() resolves each collective step's edge
  endpoints from rail_rank_hosts[step_rail_indices[step]] (tp_device_collective.c
  endpoint fill at SparkTpDeviceCollectiveConfigureRoute), and
  SparkHiddenSparkHostRdmaOpenVerbsDevice() discovers the local verbs device
  from the edge's own endpoint host. So step-0 edges open the 200G port
  (10.10.200.{self} GID on rocep1s0f0, rate 200G) and steps 1-2 open the
  100G port (10.10.100.{self} GID on rocep1s0f1, rate 100G) - one matching
  device per edge, no routing gap.
- Preflight now pins both config sha256s (resident 0c7103ac..., stage
  7bdc3437...) and checks adapter/transport libs + KV dirs. 16/16 ready.

## What is still untested until the ring reservation

- 16-rank TP4xPP4 live decode (fleet_swap dsv4-pro + model_stream_decode_benchmark).
- The measured token stream and its hash (baseline gate).
