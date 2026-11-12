# Group sizes: first fleet runs, 2026-11-12

Qwen 3.8 27B, the publisher's FP8 release (`Qwen/Qwen3.8-27B-FP8` at
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`), served by lanes of eight, four,
two and one rank on aligned groups of the sixteen Sparks, beside the 16-rank
lane of the same weights. The plan these runs belong to is in `TECHDEBT.md`,
"Group sizes and placement by usage".

| Nodes a lane | 16 | 8 | 4 | 2 | 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode, one stream | 60.1 tok/s | 43.7 tok/s | 25.7 tok/s | 14.3 tok/s | 7.9 tok/s |
| Prefill, one 32,768-token prompt | 2,612 tok/s | 2,532 tok/s | 1,649 tok/s | 1,061 tok/s | 785 tok/s |
| Lanes that fit sixteen nodes | 1 | 2 | 4 | 8 | 16 |
| Decode a node, one stream a lane | 3.8 tok/s | 5.5 tok/s | 6.4 tok/s | 7.2 tok/s | 7.9 tok/s |
| Prefill a node | 163 tok/s | 316 tok/s | 412 tok/s | 530 tok/s | 785 tok/s |

Doubling a group from one node to two, four and eight makes one stream 1.81,
1.80 and 1.70 times faster and one prompt's prefill 1.35, 1.55 and 1.54
times faster; from eight to sixteen the stream gains 1.37 times and the
prefill 1.03. No size is best at everything: a request is fastest on the
largest group, and the fleet does most on the smallest that is fast enough.

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
- The production lane (Qwen 3.8 Max) stayed loaded and idle on every node
  for the four-rank speed runs. A four-rank lane took about 15 GiB a node,
  which left 6.4 to 8.3 GiB.
- The one-rank lane: `STAGE_MODEL=qwen38_27b_fp8_tp1` on spark0, the pack
  `--tp-degree 1` writes (30,226,700,928 bytes,
  `4189311b8b842af0c983f56dabac2fe1fa83d5910f5e64e663f0021461f48ff3`), the
  adapter built from `c59889ffd`. The swap controller had unloaded the
  production lane (`/deactivate`); the one-rank lane took about 57 GiB.
- Two two-rank lanes: `STAGE_MODEL=qwen38_27b_fp8_tp2` on nodes 0-1 and 2-3,
  mesh lanes 13 and 14, the packs `--tp-degree 2` writes (15,633,786,752
  and 14,595,696,512 bytes,
  `fab35bec6a72fbd9c9c760504e1bc77a7abb4bf71944d8289f17d027af0fc118` and
  `6e59885ee681bdee7189c7c6831fd203b7f2613110df0d1c91cb44ef03d53314`), the
  adapter built from `1f8d8f063`
  (`96cf8fad935cfef078153578b9a7259053dc96ba124c666e50ac239221511859`).
- One eight-rank lane: `STAGE_MODEL=qwen38_27b_fp8_tp8` on nodes 0-7, mesh
  lane 8, the packs `--tp-degree 8` writes (3,910,542,080 bytes on ranks 0
  to 3 and 3,651,019,520 on ranks 4 to 7), the adapter built from
  `c59889ffd`
  (`42fd0f80e8fe1f812ad96a30bde07409c90fd07a75ce0c51b5d587d58b57cacb`):

  | Rank | SHA-256 |
  | --- | --- |
  | 0 | `2725e22a668bb92b68d6d25857f503de9bc14393e39cfa5bb7111dc302095f6d` |
  | 1 | `e7e73a6ff7fe74b765e8238494fda11f81dc716a37bed2cc40b0f214d83f2562` |
  | 2 | `bce40218d74ba98d1374ba9d7a0129f23e8dbbcdf434385587bd24bc3defaded` |
  | 3 | `61502c3cd2bb130beb5394524ce5780b099c24c61439906029c051bd55e5d427` |
  | 4 | `65790483aa940ed49e8a8cf786c56c81613d1e76bfdb63d8ff733481e99120e9` |
  | 5 | `bac45caa79a1de8731b937082985e7bc7497b0eb1758b2a6e9a6e25400ee881c` |
  | 6 | `18377e482c0be6e915577b60b561a54264f3e2b23ca611126b654dc8c663029f` |
  | 7 | `2ffa85aa70d2086d08c243695b1dc56e1f36806f741cbe5f3e6f363d330ff9a7` |

  The production lane was unloaded for the two-rank and eight-rank runs as
  well, and reloaded through the swap controller after them.
- The adapters were built as the work went, from `7d9d37944` (four ranks),
  `c59889ffd` (one and eight) and `1f8d8f063` (two). Built again from
  `0f1d80cfc`, each has the GPU code and read-only data of the build that
  was measured (`.nv_fatbin` and `.rodata` equal); the host code differs
  only in the line numbers its failure sites record, 27 `mov` immediates
  in the 16-rank and one-rank adapters. The same holds for the 16-rank
  adapters of Qwen 3.8 Max and Flash, which share the stage. The Qwen
  verify-frames GPU test passes on the build host at `0f1d80cfc`.

## Speed

Every figure is from the API log's request measurements. Decode is the time
between a request's first and last token over the tokens between them, 256
tokens, no speculation. Prefill is the time from a request's dispatch to its
one output token, over that many uncached prompt tokens.

| Nodes a lane | 16 | 8 | 4 | 2 | 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode, one stream, ms a token | 16.64 | 22.87 | 38.9 | 69.7 | 126.5 |
| Decode, one stream, tok/s | 60.1 | 43.7 | 25.7 | 14.3 | 7.9 |
| Weight bytes the heaviest rank reads a token | 1.96 GB | 3.91 GB | 7.82 GB | 15.63 GB | 30.2 GB |
| That read at 243 GB/s | 8.05 ms | 16.1 ms | 32.2 ms | 64.3 ms | 124.4 ms |
| Share of the token that is the read | 48% | 70% | 83% | 92% | 98% |
| Prefill, 8,192 tokens, tok/s | 2,580 | 2,587 | 1,764 | 1,126 | 861 |
| Prefill, 32,768 tokens, tok/s | 2,612 | 2,532 | 1,649 | 1,061 | 785 |
| Time for 1,024 prefill rows at 32,768 | 0.392 s | 0.404 s | 0.621 s | 0.965 s | 1.304 s |
| of which a rank's share of the one-node wave | 0.082 s | 0.163 s | 0.326 s | 0.652 s | 1.304 s |
| the rest | 0.310 s | 0.241 s | 0.295 s | 0.313 s | none |
| Eight requests at once, all eight generating, tok/s | | 226 | 122 | 79 | 43 |
| The same, ms a token a stream | | 35.4 | 65.4 | 101.8 | 184.5 |
| The batch from dispatch to last token, tok/s | | 144 | 96 | 52 | 30 |

The sixteen-rank column is the FP8 lane of the variant measurement
(`qualification/qwen38_27b/fp8-variant-20261112/`), the same weights, window
and method, measured after its corpus and suite runs; its task suite, eight
requests in flight for 1,164 s, came to 270.8 tok/s with prefill.

- **Decode is the weight read.** A token costs the bytes its rank reads at
  the node's 243 GB/s plus 2 to 9 ms: 2.1 ms on one node, 5.4 on two, 6.7 on
  four, 6.8 on eight and 8.6 on sixteen. Halving the group doubles the read.
- **Prefill pays about 0.3 s a 1,024-row wave beyond compute at every size
  from two to sixteen.** One rank has no rounds, so its 1.30 s a wave is the
  whole model's compute: about 55 TFLOP, 42 to 46 TFLOPS, a fifth of the 210
  TFLOPS the node's FP8 peak measures. A lane of n ranks does a share of
  1/n of it, and what is left of its wave, 0.24 to 0.31 s, does not shrink
  with the group: a round moves the same rows whatever the number of ranks.
  That is why two nodes prefill only 1.35 times faster than one and sixteen
  no faster than eight. Two things hold prefill down, then: the FP8 prefill
  kernels at every size, and the rounds at every size but one.
- **Several lanes at once.** Four four-rank lanes served together: one
  stream a lane at 37.8 to 42.6 ms a token, about 100 tok/s in all; a
  32,768-token prefill at 1,593, 1,598 and 1,646 tok/s on the three cold
  lanes, about 6.4K tok/s in all (the fourth lane's request reused 8,191
  cached tokens and is left out). Thirty-two requests over the four lanes:
  the 31 whose log lines are whole made 6,285 tokens in 20.3 s from
  dispatch, 310 tok/s, and 448 tok/s while 23 long answers were generating
  (one lane had its prompts cached).
  Two two-rank lanes served together: a 32,768-token prefill at 1,074 tok/s
  on one while the other prefilled 24,577 uncached tokens at 1,025, and one
  stream at 14.3 to 14.4 tok/s on each. Lanes on different nodes do not slow
  each other.
- **Sixteen nodes, by layout.** One stream a lane and one prompt a lane; a
  product of lanes is marked where the lanes did not run together:

  | Layout | One request decodes at | One prompt prefills at | All lanes decode | All lanes prefill |
  | --- | ---: | ---: | ---: | ---: |
  | 1 lane of 16 | 60.1 tok/s | 2,612 tok/s | 60 tok/s | 2.6K tok/s |
  | 2 lanes of 8 | 43.7 tok/s | 2,532 tok/s | 87 tok/s (2 times one lane) | 5.1K tok/s (2 times one lane) |
  | 4 lanes of 4 | 25.7 tok/s | 1,649 tok/s | about 100 tok/s (run together) | about 6.4K tok/s (run together) |
  | 8 lanes of 2 | 14.3 tok/s | 1,061 tok/s | 114 tok/s (8 times one lane; two ran together) | 8.5K tok/s (8 times one lane; two ran together) |
  | 16 lanes of 1 | 7.9 tok/s | 785 tok/s | 126 tok/s (16 times one lane) | 12.6K tok/s (16 times one lane) |

- In every eight-request batch two answers end after 49 or 50 tokens. The
  table's rate is over the stretch in which all eight generate (1.0 s on
  eight ranks, 2.0 on four, 2.9 on two, 6.4 on one); after the short answers
  end, the six long ones (seven on four ranks) come to 187, 134, 63 and 35
  tok/s. The first version of this page multiplied the long answers' mean
  time a token by eight streams (about 142 tok/s for four ranks, 566 for
  four lanes and 49 for one rank): those were too high.
- The device's own wave on four ranks is 530 to 560 ms
  (`STAGE-WAVE-HOST submit_ms`) of the 621 ms a wave takes end to end.

## Numbers against the 16-rank lane

The lanes hold the same weights and scored the same positions
(`tools/ab_api_score.py`): the four-rank lanes all 12,800 corpus positions of
the variant measurement, the one-rank lane the first 16 documents. The
two-rank and eight-rank lanes were not scored.

| | 4 ranks against 16 | 1 rank against 16 | 1 rank against 4 |
| --- | ---: | ---: | ---: |
| Positions | 12,800 | 1,024 | 1,024 |
| Same tokens and log-probs | none | none | none |
| Same best token | 99.37% [99.24, 99.48] | 99.22% [98.92, 99.51] | 99.32% [98.83, 99.71] |
| KL lower bound | 0.0012 [0.0003, 0.0037] | 0.0018 [0.0002, 0.0057] | 0.0029 [0.0002, 0.0126] |
| NLL of the smaller group, relative | -0.082% [-0.199, -0.019] | -0.384% [-0.950, -0.039] | +0.224% [-0.110, +0.943] |
| Next token right | 68.438% against 68.383% | 64.746% against 64.746% | 64.746% against 64.941% |

Every round adds BF16-rounded partials, sixteen of them on 16 ranks, four on
4 and none on 1, so the lanes round differently at every layer and no
position comes out bit-identical. The difference is about half the size of
the FP8-against-BF16 one on the same corpus (same best token 98.12%, KL
0.0027, NLL +0.283%), and the smaller group is not worse: its NLL is equal
or slightly lower. A group size is a variant like a precision: it is
offered with these figures (`docs/VARIANT_GATE.md`).

## What the runs found

- The stage needed no change for eight ranks, four or one. On two ranks a
  rank's attention rows are 7,168 wide, more than a collective row may be
  (the hidden width, 5,120;
  `SparkTpDeviceCollectiveValidateSubmission`), and every request failed.
  The gather now splits a rank's rows into collective rows of at most the
  hidden width (`kGatherParts` in
  `model-families/qwen38/source/qwen38_hybrid_stage.cuh`): two parts on two
  ranks, one elsewhere, and the 16-rank adapter's GPU code is unchanged.
- The runner refused every round but the BF16 sum when a lane had no
  collective, so a one-rank lane failed its first request; a gather or an
  all-to-all on one rank is now a local copy (`StageRunnerLoneRound` in
  `inference/runner/stage_runner.cu`). The 16-rank adapter's GPU code is
  unchanged by it.
- weightd keeps the topology a mesh lane was first given until it restarts:
  `SparkWeightdMeshLaneConfigure` answers a different one with
  `SPARK_STATUS_UNSUPPORTED`. Lanes 9 and 10, which last ran 16 ranks, refused
  the group on nodes c-f; lanes 6, 7, 8 and 11 had no topology yet, nor had
  lanes 13 and 14 for the pairs, nor lane 8 on nodes 0-7.
- The lane tool sized a group's recurrent state for a sixteenth; it now
  follows the entry's rank count.
- Four hub APIs given one log file interleave their lines.
