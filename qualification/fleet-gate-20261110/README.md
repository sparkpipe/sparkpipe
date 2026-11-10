# Fleet gate of the mega-PR, 2026-11-10

Model: Qwen3.8-27B, BF16 pack, TP16, 8 sequences, 1,024-row waves, 32,768
positions per sequence. A test lane on mesh lane 5 beside production
(qwen3.8-max stayed active on lane 12), started with `tools/stage_lane.sh`
in model mode from its own root, with its own API instance on the hub
(port 8490). Three builds ran on the same lane within one hour:

| | Commit | What it is |
| --- | --- | --- |
| A | `fded9e032` | `main`, the pull request's base |
| B | `1dc695de0` | this branch before the Metal line: audit fixes, the runner-owned layer loop, tree verification, the folded perf changes |
| C | `f5a80228f` | this branch's head: B plus the Metal line and the audit's deletions |

Prompts: the release kit's six (`tools/release_capture.py` with the kit's
`prompt_source.c`: four short, one of 1,980 tokens, one of 7,693), greedy,
48 tokens, top-4 logprobs. Compared with `tools/release_compare.py`.

## Outputs

| Comparison | Tokens | Logprobs |
| --- | --- | --- |
| B against A | equal, 6 of 6 | bit-identical, 6 of 6 |
| C against A | equal, 6 of 6 | served-token deltas 0.08 to 0.53 |
| C repeated | equal, 6 of 6 | bit-identical; the repeat resumed 1,979 of 1,980 and 7,692 of 7,693 tokens from the prefix cache |
| A against the production capture of `a85cf215a` (11-04) | equal, 6 of 6 | four short prompts bit-identical; the two long prompts differ by 0.49 and 0.35, which is `main`'s own change since that release |

B against A is the gate for everything in this pull request before the
Metal line: nothing in it changes a bit on this model. C's deltas are the
publisher numerics of the Metal line (#1739, #1865), with no token flip on
these prompts. The reference golden (three prompts of
`qualification/qwen38_27b/reference/reference3_fp8_numpy.json`) keeps top-1
agreement 71 of 71 on B and C against the production golden.

Consistency on C and on B alike: eight concurrent greedy requests return the
tokens of their single-stream runs, with served logprobs up to 0.04 apart
(multi-row steps); a seeded sampled request repeats exactly and returns the
same tokens inside a batch of eight.

## Speed, speculation off

| | A | B | C |
| --- | ---: | ---: | ---: |
| One stream, 256 tokens, tok/s (median of 3) | 46.7 | 45.0 | 47.0 |
| Eight streams of 256 tokens, aggregate tok/s, prefill included | 218 | 237 | 243 |
| 5,947-token prefill, seconds (3 runs) | 1.85 to 1.88 | 2.12 to 2.23 | 2.09 to 2.10 |

Eight streams gain 9 to 12%. **Prefill lost about 12% on this branch.** B
and C share it, so it comes from the runner changes, not the Metal line; the
suspect is the two prefill lanes interleaved on one host thread in place of
the lane worker thread. It must be fixed or reverted before a release.

## Not covered

No verify frame ran on A or C for a copy workload (600 tokens after six
copies of a C file): lookup speculation did not engage on this lane, so
verify and tree frames are unproven on the fleet. One model of twenty ran;
MoE families, 16-sequence lanes and the 1M window did not.

## Found on the way

- Every adapter of this branch needs `runtime_libs/libsparkpipe_device_cuda.so`
  beside it; the first start failed on all sixteen ranks until the lane tool
  shipped it (`fdd2d9598` fixes the release kit and the lane tools).
- Production cannot activate qwen3.8-27b: its hub API binary of 11-08 11:49
  refuses the `model_resident.json` pushed at 20:35 that day (the
  `draft_server` block), so the swap controller's activation failed and it
  restored qwen3.8-max. A release that installs the API with the records
  fixes it.

Raw captures, comparison outputs and the gate scripts are on the hub under
`/srv/workspace/sparkpipe/audit-lane/` (`gate1` for C, `gateA`, `gateB`,
`gate2.py`, `gate3.py`, `gate4.py`).
