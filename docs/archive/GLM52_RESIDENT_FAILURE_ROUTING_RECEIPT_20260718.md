# GLM-5.2 Resident Failure Routing Receipt

## Artifact

- Merged main: `3525cb47beb9556fd13683384f79257603110b92`
- PR: `#479`
- Release: `glm52-fp8-3525cb4-b64-failure-routing-r1`
- CUDA payload lineage: `c2f45ad`
- Quantization: FP8
- Maximum active sequences: 64
- KV pool tokens: 16,384
- MTP graph replay: disabled
- Resident executable SHA-256 on all 13 ranks:
  `f038cd8043a8d95c9b974a082ed21969b6385cee6033c3d782ec42ab1fdec4de`
- Service backend SHA-256 on all 13 ranks:
  `31653545e48d6725ad97687031dbb1c8ab8b8008b0b63659decd3c9e1ab767a9`

## Multi-Wave Prefill

The exact former 65-token reproducer completed through all 13 ranks.

```text
prompt_tokens=65
prefill_dispatches=2
decoded_tokens=8
tokens=We need to analyze the given C function
done_status=0
curl_http=202
total_seconds=4.040569
live_requests_after=0
queued_requests_after=0
event_backlog_after=0
```

No current-request `cuda_residentd_prefill_rejected`,
`cuda_residentd_work_rejected`, or failed resident state was observed.

## Correctness Receipts

The one-token deterministic smoke returned the committed expected token:

```text
prompt=Say OK. OK.
token_id=10397
text= OK
total_seconds=0.515055
```

Plain decode and graph-disabled MTP were then run with the same 22-token
technical prompt, greedy decoding, and a 64-token output budget. Both produced
64 token events. Their token-id files were byte-identical:

```text
plain_token_id_sha256=9a75ab164284854a2f35d4e24b9ad117727d7755fcbe26b6bb709d480a559990
mtp_token_id_sha256=9a75ab164284854a2f35d4e24b9ad117727d7755fcbe26b6bb709d480a559990
token_parity=EXACT
```

This is a deterministic parity receipt, not a corpus accuracy measurement.
General model accuracy remains `NOT_MEASURED`.

## Performance

Workload for both rows:

```text
prompt=Explain why unsigned integer overflow can corrupt a C array-length calculation, and show a safe checked-add helper.
prompt_tokens=22
output_tokens=64
temperature=0
active_sequences=1
```

| Mode | Total seconds | Output tok/s |
|---|---:|---:|
| Plain FP8 | 15.493657 | 4.131 |
| FP8 MTP, graph disabled | 19.484496 | 3.285 |

MTP increased elapsed time by 25.76% and reduced output throughput by 20.48%.
The MTP request used 25 decode dispatches for 64 committed tokens:

```text
mtp_draft_tokens=110
mtp_verify_dispatches=22
mtp_accepted_draft_tokens=39
mtp_committed_tokens=61
mtp_rejected_tokens=71
```

Recent final-rank cycle lines put the draft-chain epilogue at approximately
27.8-29.0 ms. The measured request averages approximately 779 ms per decode
dispatch, compared with approximately 242 ms per plain decode dispatch. The
remaining performance blocker is therefore outside the draft-chain epilogue;
verification/ring dispatch work dominates.

## Status

- Multi-wave 65-token prefill: `OBSERVED`
- End-to-end B1 inference through 13 ranks: `OBSERVED`
- MTP/plain token parity for this 64-token workload: `MEASURED`
- General model accuracy: `NOT_MEASURED`
- MTP performance improvement: `NOT_WORKING`
- MTP graph replay: `NOT_WORKING`
- JIT KV: `NOT_WORKING`
- DSpark: `NOT_WORKING`
