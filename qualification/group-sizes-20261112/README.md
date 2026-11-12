# Group sizes: first fleet runs, 2026-11-12

Qwen 3.8 27B, the publisher's FP8 release (`Qwen/Qwen3.8-27B-FP8` at
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`), served by four-rank lanes on
aligned groups of the sixteen Sparks, beside the 16-rank lane of the same
weights. The plan these runs belong to is in `TECHDEBT.md`, "Group sizes and
placement by usage".

## What ran

- Firmware: `sparkpipe_model_residentd`, `weightd_warm`, the transport and
  the device library of commit `9c7e9bd1e`; the adapter
  `libqwen38_27b_tp4_serving_adapter.so` built on the build host from
  `7d9d37944` (`make build/libqwen38_27b_tp4_serving_adapter.so`). The stage
  code is the 16-rank stage; the build names 4 ranks.
- Pack: `tools/qwen38_27b_tp16_pack.py --codec fp8 --tp-degree 4`, four rank
  packs of 7,818,290,432 and 7,299,245,312 bytes:

  | Rank | SHA-256 |
  | --- | --- |
  | 0 | `85e11a17c8b749d8812fc3855243acc17bfb8103e711beb2308e8381710cd356` |
  | 1 | `b3d6f241a606a5ea58705960b950e9daed607ead687c17863ad2d56a48591642` |
  | 2 | `95e7d1373988eaf2b7439b1767ad6a059d1130d034aee5cf2842819030f571a5` |
  | 3 | `ab628b8929e2683161513e66d54205a4092a74c488a06c24f3c6792a1725c5ae` |

  On spark0, every one of the 915 tensors of its 16-rank pack is in the
  4-rank pack unchanged: 209 whole, 130 as a block of rows, 256 as a block of
  columns, and the fused tensors segment by segment (40 of 320 examined, none
  different).
- Lanes: `tools/stage_lane.sh` with `STAGE_MODEL=qwen38_27b_fp8_tp4`,
  `STAGE_FIRST_NODE` 0, 4, 8 and 12 on mesh lanes 6, 7, 8 and 11; 8
  sequences, 1,024 rows, a 262,144-token window, at most 8,192 device KV
  pages. One hub API a lane, ports 8491 to 8494.
- The production lane (Qwen 3.8 Max) stayed loaded and idle on every node. A
  four-rank lane took about 15 GiB a node, which left 6.4 to 8.3 GiB.

## Speed

Decode is the time between a request's first and last token over the tokens
between them, from the API's token timestamps, 256 tokens, no speculation.
Prefill is the wall time of one request of that many uncached tokens with one
output token.

| | Sixteen ranks, one lane | One four-rank lane (nodes 0-3) | Four lanes at once |
| --- | ---: | ---: | ---: |
| Decode, one stream a lane | 16.64 ms a token (60.1 tok/s) | 38.9 ms a token (25.7 tok/s) | 37.8-42.6 ms each, about 100 tok/s in all |
| Decode, eight streams a lane | | 56.5 ms a token a stream (about 142 tok/s) | about 566 tok/s over 32 streams |
| Prefill, 8,192 tokens | 2,580 tok/s | 1,764 tok/s | |
| Prefill, 32,768 tokens | 2,612 tok/s | 1,649 tok/s | 1,593, 1,598 and 1,646 tok/s on the three cold lanes, about 6.4K in all |

The sixteen-rank column is the FP8 lane of the variant gate
(`qualification/qwen38_27b/fp8-variant-20261112/`), the same weights, window
and method, measured after its corpus and suite runs.

- A rank of four reads 7.82 GB of weights a token; at 243 GB/s that is 32.2
  ms, so one stream runs at 83% of the group's memory limit.
- A 1,024-row prefill wave took 530 to 560 ms on four ranks
  (`STAGE-WAVE-HOST submit_ms`); 2,612 tok/s on sixteen ranks is about 390
  ms a wave. One group prefills at 63% of the sixteen-rank rate, four groups
  at 2.5 times it.
- The fourth lane's first prefill request reused 8,191 cached tokens and is
  left out of the prefill row.

## Numbers against the 16-rank lane

The 4-rank lane and the 16-rank lane hold the same weights and scored the
same 1,024 positions (16 documents, `tools/ab_api_score.py`):

| | 4 ranks against 16 ranks |
| --- | ---: |
| Positions with the same tokens and log-probs | 0 of 1,024 |
| Top-1 agreement | 99.02% [98.73, 99.32] |
| KL lower bound, mean | 0.0090 (median 6.5e-5; one position 8.4) |
| Corpus token's NLL, 4 ranks relative to 16 | -0.57% [-1.73, -0.06] |

Every round adds BF16-rounded partials, sixteen of them on 16 ranks and four
on 4, so the two lanes round differently at every layer. On these 16
documents the difference is the size of a precision change, and the 4-rank
lane's NLL is not worse.

## What the runs found

- The stage needed no change for four ranks.
- weightd keeps the topology a mesh lane was first given until it restarts:
  `SparkWeightdMeshLaneConfigure` answers a different one with
  `SPARK_STATUS_UNSUPPORTED`. Lanes 9 and 10, which last ran 16 ranks, refused
  the group on nodes c-f; lanes 6, 7, 8 and 11 had no topology yet.
- The lane tool sized a group's recurrent state for a sixteenth; it now
  follows the entry's rank count.
- Four hub APIs given one log file interleave their lines.
