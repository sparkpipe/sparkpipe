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
| D | C plus the fix | C with the second prefill lane back on its worker thread |

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

| | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| One stream, 256 tokens, tok/s (median of 3) | 46.7 | 45.0 | 47.0 | 47.2 to 47.6 |
| Eight streams of 256 tokens, aggregate tok/s, prefill included | 218 | 237 | 243 | 246 to 248 |
| 5,947-token prefill, seconds (3 runs) | 1.85 to 1.88 | 2.12 to 2.23 | 2.09 to 2.10 | 1.83 to 1.92 |

Eight streams gain 13% over `main`. B and C lost about 12% of prefill: the
two prefill lanes interleaved on one host thread serialise each lane's
collective rounds with the other lane's enqueues, where the worker thread
overlapped them. D restores the worker thread, returns prefill to `main`'s
time and keeps C's outputs bit for bit (6 of 6 exact against C). The
interleave on one thread can come back only with rounds that do not block
the host.

## Not covered

No verify frame ran on A or C for a copy workload (600 tokens after six
copies of a C file): lookup speculation did not engage on this lane, so
verify and tree frames are unproven on the fleet. Two models of twenty ran
(the second is below); the 1M window did not.

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

## Gemma 4 26B-A4B: the MoE and 16-sequence run

Gemma 4 26B-A4B at TP16, 16 sequences, 2,048-row waves, on the same lane with
its own API instance. A and B are the builds above; H is `669b261ce`, this
branch's head with the worker thread back; C is `f5a80228f` where named.

| Comparison | Tokens | Logprobs |
| --- | --- | --- |
| B against A | equal, 6 of 6 | bit-identical, 6 of 6 |
| H repeated | equal, 6 of 6 | bit-identical |
| H against A | four prompts differ (first difference at token 0, 3, 14 and 4), two are equal | served-token deltas up to 11.5 |
| A against the production capture of `a85cf215a` (11-04) | all six differ | A's short prompt decays into one repeated token after seven |
| H against that capture | the short prompt's 11 tokens are equal; the other five differ (first difference at 3, 47, 2, 26 and 2) | |

Everything in this pull request before the Metal line is bit-identical to
`main` on an MoE model as well. `main`'s Gemma output is no longer the last
release's. The Metal line's numerics (for this family: rope with rounded
terms and the rounded residual sum; its norms keep their order) change the
output materially and return the short prompt to the release's tokens. No
Gemma reference golden was compared here, so which of the three is the
publisher's output is not settled by this run.

| Speed, speculation off | A | H |
| --- | ---: | ---: |
| One stream, tok/s | 92.0 | 92.8 |
| Sixteen streams, aggregate tok/s, prefill included | 438 | 543 |
| About 6,000-token prefill, seconds | 1.14 to 1.49 | 1.06 to 1.21 |

### Open defect: a short prompt's first token depends on its batch

Seen on H, on C, on `b57f1e375` (the numerics merge alone on top of B) and
on `main` with the numerics branch merged and none of this pull request's
work (33 of 96 in `gate7.py`); not on A or B. A 6-token prompt that is prefilled while other short prompts
are being prefilled takes a second value for about one prompt in five.

- The second value is one per prompt: the same top-4 candidates reweighted
  (the served logprob moves by 0.1 to 1.3), and the greedy token flips where
  the top two were close. It is the same on C and H and does not depend on
  which other prompts share the batch.
- It follows the request's place in the wave, not its text: fifteen prompts
  queued behind a long prefill turn at the same places in forward and in
  reverse order. In the run that was timed, the requests that turned
  completed about 5 ms after the ones that did not.
- Never at two concurrent requests, never for prompts of 12, 18, 25 or 27
  tokens, never when a single request is repeated. Prompts this short are
  not cached (`cached_tokens` is 0), so the prefix cache is not involved.
- B is exact in every cell of the same matrix, bar one 0.0045 logprob
  difference among sixteen identical copies.
- Ruled out by lane runs: the 2 to 16 row frame kernel of #1891 (the head
  without it behaves the same), the second prefill lane's worker thread (C
  has both lanes on one thread and behaves the same) and the lane split
  (it starts at 256 rows).
- What the numerics merge changes for this family is row-local (rope with
  rounded terms, the rounded residual sum), and Gemma takes the tiled GQA
  shard kernel, which did not get the value-weight rounding of the scalar
  one. Taking them out leaves the defect: rope with plain terms 21 of 96,
  residual sum unrounded 13 of 96, both together 9 of 96. Something else the
  numerics line changes for this family takes part.
- So the defect does not come from this pull request's audit and runner
  work. It belongs to the numerics line (#1739, #1865), or to a batch
  dependence already on `main` that `main`'s Gemma output hides. The cause is
  open.

To reproduce: the lane above, then on the hub under
`/srv/workspace/sparkpipe/audit-lane/`: `gate7.py <port>` (three rounds of
sixteen short prompts, batch against single; the numerics merge gives 19 of
96), `gate5.py <port>` (the cold and warm, single and batch matrix; exact on
B) and `gate8.py <port>` (the place-in-wave pattern). Captures are in
`gemma-A-bos`, `gemma-B`, `gemma-H-bos` and `gemma-X` beside them.

Until this is closed the pull request is not ready for a release that serves
MoE models or short prompts in batches.
