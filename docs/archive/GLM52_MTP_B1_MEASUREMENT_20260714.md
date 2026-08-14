# GLM-5.2 B1 MTP Measurement

This receipt compares plain decode and one-draft MTP on the same merged commit,
FP8 artifacts, prompt, diagnostic profile, and 13-rank topology.

This is the existing serialized one-draft implementation in merged main. It is
not the advisor's batched DSpark/MTP commit `e10c0b0`; that commit was not
available in the repository, compiled, deployed, or measured for this receipt.

```text
commit:      f5364187449c45e14d5013a2e5b8243a2beabf4d
topology:    13 Spark ranks
live lanes:  1
prompt:      Reply with exactly one short sentence about batching.
prompt size: 9 tokens
output:      32 tokens
requests:    2 per mode, sequential
sampling:    greedy
transport:   host-staged TCP
```

| Metric | Plain | MTP |
| --- | ---: | ---: |
| Successful requests | 2/2 | 2/2 |
| Token events | 64 | 64 |
| Elapsed wall time | 17.236 s | 15.835 s |
| Average request latency | 8.617 s | 7.916 s |
| Average TTFT | 0.454 s | 0.689 s |
| Token events/s | 3.713 | 4.042 |

MTP improved matched end-to-end token throughput by `1.089x` and reduced
average request latency by 8.135%. It increased average TTFT by 51.647%.

All four requests emitted the same 32 token IDs and text. The final MTP health
snapshot reported 22 draft tokens, 22 verify dispatches, 16 accepted drafts,
6 rejected drafts, 38 committed tokens, 48 decode dispatches, zero event
backlog, zero dropped events, and no live request after completion.

This proves deterministic parity for this prompt and a modest B1 throughput
gain. It is not a model-accuracy score, a production-speed result, a wider
batch claim, a long-context claim, evidence that DSpark works, or a measurement
of batched MTP.

Raw JSONL, summaries, release manifests, installed artifact hashes, and exact
commands are retained under `diagnostics/glm52_mtp_b1_f536418_20260714/`.
