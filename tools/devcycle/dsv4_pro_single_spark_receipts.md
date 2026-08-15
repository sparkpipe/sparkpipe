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

## What is still untested until the ring reservation

- 16-rank TP4xPP4 live decode (fleet_swap dsv4pro + model_stream_decode_benchmark).
- The measured token stream and its hash (baseline gate).
