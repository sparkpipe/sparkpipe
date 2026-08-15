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
current full pack and sha256-compared with the shipped bytes. Rank 0
regeneration == shipped bytes (2be8aa0a... full-hash match); the loop runs
through ranks 1-15 in the background. The sharder on sparkb is byte-identical
to the worktree version (md5 2f9517ad4f8531f85cb1712922b47df9).

## What is still untested until the ring reservation

- 16-rank TP4xPP4 live decode (fleet_swap dsv4pro + model_stream_decode_benchmark).
- The measured token stream and its hash (baseline gate).
