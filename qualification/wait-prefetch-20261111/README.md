# Weight prefetch during a mesh wait, 2026-11-11

Model: Qwen3.8-27B, BF16 pack, TP16, 8 sequences, 1,024-row waves, on a test
lane on mesh lane 5 beside production (qwen3.8-max stayed active on lane 12),
started with `tools/stage_lane.sh` in model mode from its own root, with its
own API instance on the hub (port 8490). The API binary is the gated one
(`3a1cff2da`) for every run; only the node firmware changes.

## Speed

Load: four requests of 256 tokens after prompts of 914 to 975 tokens, one at
a time, speculation off, then two requests of 96 tokens with top-4 logprobs.
Times are from the API log's `batch_step` lines and the sixteen rank logs,
read with `tools/roofline_meter.py`; the reply text of the four requests is
identical in every run (sha256 `375d03f61e90121e...`).

| Node firmware | Prefetch during the wait | Chain, ms a token | tok/s | Logprobs step, ms |
| --- | --- | ---: | ---: | ---: |
| variant of `f1cbdaf02` | none | 22.05 | 45.3 | 28.70 |
| `b01e3c1ab` | on the polling warp, eight pieces a poll | 21.03 | 47.5 | 27.66 |
| `f1cbdaf02` | own warp, whole group regardless of the peers | 20.39 | 49.0 | 27.45 |
| variant of `f1cbdaf02` | own warp, until the peers land | 20.15 | 49.6 | 26.68 |
| `d619f5e61` sources, 12 MiB budget | own warp, until the peers land | 20.19 | 49.5 | 26.89 |
| the same, 16 MiB budget | | 20.15 | 49.6 | 27.35 |
| the same, 20 MiB budget | | 20.37 | 49.1 | 26.69 |

Rank 0, one of the eight ranks with the larger FFN share, in the windows of
pure chain decode (ms a step: own work, peer wait, publish, combine):

| Prefetch | own | wait | publish | combine | step |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 16.81 | 4.13 | 0.43 | 0.44 | 21.80 |
| on the polling warp | 15.74 | 4.04 | 0.43 | 0.44 | 20.66 |
| own warp, whole group | 10.81 | 7.09 | 0.43 | 1.83 | 20.15 |
| own warp, until the peers land | 13.45 | 5.41 | 0.43 | 0.58 | 19.87 |

The decode graph has 1,601 nodes and registers 1,544 MiB in 145 groups for
the prefetch, of the 3,329 MiB a heavy rank reads a token.

## The FP8 pack on the same lane

Not a served lane: one run with `tools/stage_lane.py` pointed by hand at the
FP8 pack that is on every node (`qwen38_27b.fp8.tp16`, the publisher's
`Qwen3.8-27B-FP8` release, 1.83 to 1.96 GB a rank against 3.24 to 3.49) and
at the FP8 adapter build, with the firmware of `d619f5e61` and the same load.
Its outputs were not compared with a reference in this run.

| Pack | Chain, ms a token | tok/s | Logprobs step, ms | Rank 0: own, wait, publish, combine |
| --- | ---: | ---: | ---: | --- |
| BF16 | 20.19 | 49.5 | 26.89 | 13.78, 5.12, 0.42, 0.59 |
| FP8 | 15.65 | 63.9 | 22.50 | 8.99, 5.37, 0.43, 0.61 |

At 1.96 GB a pass the memory floor is 8.05 ms, so the FP8 chain is at 51% of
its roofline and 43% of it is collectives.

## Outputs

With the firmware built from `d619f5e61`, the release kit's seven prompts
(`tools/release_capture.py`) against the gated capture of `3a1cff2da`,
compared with `tools/release_compare.py`: tokens and logprobs equal on seven
of seven, on every path (one step at a time with top-4 logprobs, decode
chains, speculation on, seeded sampled, all at once, sampled all at once);
the largest logprob difference is 0.0000. The ranks logged no failed round,
no adapter error and no failed graph. The API logged one `batch_rejected`,
a resume point the ranks no longer held, which the engine cleared and
prefilled again; the gated run has one too.

The last commit to touch code, `77ec4690f`, renames one kernel-local
variable after `d619f5e61`. Its adapter library, built on the same host, has the same host
code, device code and read-only data sections as the gated one (sha256 of
`.text` `5f05b2baf52d250b...`, of `.nv_fatbin` `ca3bdaedfd6056f1...`, of
`.rodata` `ac5d73ecca67aae5...`).

## The round alone

`build/mesh_round_bench` on the sixteen Sparks on mesh lane 5, a 5,120-value
BF16 all-reduce, 128 rounds a launch, 32 to 64 launches, us a round:

| Run | Round | Reading | Waiting | Publish and combine |
| --- | ---: | ---: | ---: | ---: |
| the round alone | 25.8 | | 19.6 | 5.5 |
| a cold 26 MiB read before every round, no prefetch | 155.8 to 156.6 | 122 | 28 | 6.3 |
| prefetch on the polling warp | 146.3 to 150.6 | 109 | 31 to 35 | 6.3 |
| prefetch on its own warp, until the peers land | 143.5 to 144.2 | 77 | 60 | 6.6 |
| prefetch on its own warp, whole group | 138.9 to 140.6 | 72 | 54 to 56 | 12.5 |

The round alone by size: 16.6 us for 1,024 values, 17.9 for 2,048, 21.8 for
4,096, 34.4 for 8,192, and 43.9 for 10,240 and for 16,384, where it takes
the two-hop form.

## Tests on the build host's GPU

`tests/test_tp_mesh_hardware_probe.py`: all ten modes pass, 733 cases, among
them the two of `--prefetch`. `tests/test_skinny_dependent_cuda.py`,
`tests/test_stage_runner_sum_rule_cuda.py`,
`tests/test_mesh_copy_down_cuda.py` and `tests/test_weight_prefetch_host.py`
pass.
