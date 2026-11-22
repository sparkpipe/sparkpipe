# Gate GX on one rank (2026-11-22)

The first runs of gate GX (`docs/EXACTNESS_GATE.md`) against a real engine:
Qwen 3.8 27B FP8, one rank on one GB10, engine API and rank on the same
machine, requests as token ids through the engine API with no chat layer.
Nothing here stands for a four-rank or a sixteen-rank lane, and the gate has
not run on the fleet.

| Run | Plan | Verdict |
| --- | --- | --- |
| Main block (this folder) | cases `prefix`, `kept`, `opening`, `session`, `committed`; contexts 8,000 and 32,000; tails 1 and 500 | PASS, 14 of 14 units, `alone=yes` |
| Extension (`tails-40-5000/`) | the same cases and contexts; tails 40 and 5,000 | PASS, 14 of 14 units, `alone=yes` |
| The last reply's blocks (`item4/`) | one three-turn session from an empty cache, twice | the same 152 page cache entries after a short and a long last reply |
| First block (`first-block-07021acd5/`) | as the main block, before the page cache change | INCOMPLETE: 10 units PASS, 4 stopped by a fault |

The drafted case did not run: the time on the machine ended.

## The tree

The node files of the three passing runs were built from this branch at
`c8b8286cf` (the rules merged with stage 0 at `a794ef3d7`) with one commit on
top: `5ef8180ce` of the stacked node KV root PR, which lets a lane keep its
KV ledger in its own folder. The receipts name that build by a local merge
commit, `b18810e38`. Its tree is `c4bb124ea16eed5965064bed5b3208c589bd500f`,
and `git merge 5ef8180ce` on `c8b8286cf` gives the same tree.

## Main block

One GB10, one rank, no other process on the GPU.

| Unit | Resume cut | Cached, reusing / after restore / cold | First token, reusing / after restore / cold |
| --- | ---: | ---: | ---: |
| `prefix/8000/1` | 8,021 | 8,021 / 8,021 / 0 | 262 / 959 / 8,686 ms |
| `prefix/8000/500` | 8,021 | 8,021 / 8,021 / 0 | 973 / 987 / 9,314 ms |
| `prefix/32000/1` | 32,021 | 32,021 / 32,021 / 0 | 296 / 2,045 / 37,664 ms |
| `prefix/32000/500` | 32,021 | 32,021 / 32,021 / 0 | 1,141 / 1,125 / 38,511 ms |
| `kept/8000/1` | 6,913 | 6,913 / 6,913 / 0 | 265 / 832 / 7,618 ms |
| `kept/8000/500` | 6,913 | 6,913 / 6,913 / 0 | 969 / 967 / 8,355 ms |
| `opening/8000/1` | 5,952 | 5,952 / 5,952 / 0 | 2,764 / 3,245 / 9,052 ms |
| `opening/8000/500` | 5,952 | 5,952 / 5,952 / 0 | 3,481 / 3,468 / 9,780 ms |
| `session/8000/1` | 8,054 | 8,054 / 8,054 / 0 | 716 / 1,185 / 9,107 ms |
| `session/8000/500` | 9,052 | 9,052 / 9,052 / 0 | 1,166 / 1,444 / 10,882 ms |
| `session/32000/1` | 32,054 | 32,054 / 32,054 / 0 | 645 / 2,331 / 38,293 ms |
| `session/32000/500` | 33,022 | 33,022 / 33,022 / 0 | 1,320 / 1,591 / 41,068 ms |
| `committed/8000/1` | 8,056 | 8,056 / 8,056 / 0 | 262 / 861 / 9,084 ms |
| `committed/8000/500` | 9,085 | 9,085 / 9,085 / 0 | 992 / 1,277 / 11,167 ms |

In every unit the reusing run and the run after eviction and restore from
disk gave the cold run's tokens and logprobs: the largest logprob difference
is 0 in all fourteen. The cached tokens equal the resume cut in both reusing
runs, and the cold run cached nothing. No wave was refused for capacity and
no request waited out the in-flight budget.

The preflight unit ran first and passed (an 8,000-token prefix, 40 new
tokens, a reply that crosses a block end).

## Extension: tails 40 and 5,000

One GB10, one rank, no other process on the GPU.

| Unit | Resume cut | Cached, reusing / after restore / cold | First token, reusing / after restore / cold |
| --- | ---: | ---: | ---: |
| `prefix/8000/40` | 8,021 | 8,021 / 8,021 / 0 | 582 / 1,264 / 8,891 ms |
| `prefix/8000/5000` | 8,021 | 8,021 / 8,021 / 0 | 6,236 / 6,009 / 14,468 ms |
| `prefix/32000/40` | 32,021 | 32,021 / 32,021 / 0 | 656 / 2,349 / 38,071 ms |
| `prefix/32000/5000` | 32,021 | 32,021 / 32,021 / 0 | 7,471 / 7,433 / 45,102 ms |
| `kept/8000/40` | 6,913 | 6,913 / 6,913 / 0 | 650 / 1,132 / 7,944 ms |
| `kept/8000/5000` | 6,913 | 6,913 / 6,913 / 0 | 6,000 / 5,995 / 13,460 ms |
| `opening/8000/40` | 5,952 | 5,952 / 5,952 / 0 | 3,116 / 3,589 / 9,396 ms |
| `opening/8000/5000` | 5,952 | 5,952 / 5,952 / 0 | 8,625 / 8,593 / 14,939 ms |
| `session/8000/40` | 8,132 | 8,132 / 8,132 / 0 | 756 / 1,361 / 9,677 ms |
| `session/8000/5000` | 18,052 | 18,052 / 18,052 / 0 | 6,740 / 7,403 / 27,274 ms |
| `session/32000/40` | 32,132 | 32,132 / 32,132 / 0 | 854 / 2,567 / 39,020 ms |
| `session/32000/5000` | 42,052 | 42,052 / 42,052 / 0 | 7,932 / 8,597 / 60,024 ms |
| `committed/8000/40` | 8,165 | 8,165 / 8,165 / 0 | 585 / 1,218 / 9,665 ms |
| `committed/8000/5000` | 18,085 | 18,085 / 18,085 / 0 | 6,644 / 7,241 / 27,254 ms |

The same holds as in the main block: equal tokens and logprobs with a
largest logprob difference of 0, and cached tokens equal to the resume cut.
In `prefix/8000/5000` the 5,000 tokens after the last cut are six waves that
publish nothing: four full waves that each end on a block's last token (at
9,024, 10,048, 11,072 and 12,096 tokens), one that ends one token before the
prompt's end, and the last token's own.

## The last reply's blocks leave the cache

`item4/`: one `session` unit (8,000-token prefix, 500 new tokens a turn,
three turns) from an empty cache, with the same seed, once with a last reply
of 16 tokens and once with 144, which crosses two block ends. At the engine
stop after each, the rank's page cache held 152 entries, none of them
private, and the pool held 152 pages right after the session in both runs.
What a request computes after its last cut leaves the cache when the request
is released.

## The first block and the fault it found

`first-block-07021acd5/`: the same plan on the tree before the page cache
change, with the GPU shared with another service. Ten units passed with
equal tokens and logprobs (`prefix/8000/1`, `prefix/8000/500`, `prefix/32000/1`, `prefix/32000/500`, `kept/8000/1`, `opening/8000/1`, `opening/8000/500`, `session/8000/500`, `session/32000/500`, `committed/8000/500`).
Four stopped with `model status 1`, in the cold run as well as the reusing
one, so not as a difference between them
(`kept/8000/500`, `session/8000/1`, `session/32000/1`, `committed/8000/1`).

After its last cut a request publishes nothing. A lane of that private
stream that ended on a block's last token left the block open, and the
rank's page cache refused the next lane. Every reply that crossed a block
end after the last cut failed; the eleven failed requests of the block are
exactly those. The engine test's adapter does not hold the page cache's
rules, so no host test had seen it. `cache/kv_page_cache.c` now seals a full
block that a sequence still holds when a lane with rows begins right after
it; the test is in `tests/test_kv_cache.c`.

## How it ran

- One lane of one rank with its own KV node root, so its ledger and its
  snapshot files were its own. Cache changes between steps were an engine
  stop, a drop of the lane's KV and a start; `evict` kept the files on disk.
- Steps of a block: `chain` (every unit's reusing run), `park:32000` and
  `restore:32000` (the same after eviction to disk), then one cold step per
  tail. Requests asked for 16 tokens with four logprobs; a session's earlier
  replies were 32 tokens.
- The GPU's process list was read every 5 s through every step (174 samples
  in the main block, 232 in the extension) and held only the lane's
  processes. The kernel log had no NVRM or Xid line in any step's window.
  Each step's note in the receipt (`block`) carries both.
- Kernel 7.0.0-1025-nvidia, driver 580.178.04.
- The driver stops a block when its preflight check fails.
  `preflight-stop-proof.txt` is that stop, shown with a stub tool: with the
  check forced to fail the driver exits, makes no plan and starts no step;
  with the check passing it runs the block.

## Files

For each run: `qwen3.8-27b-fp8-one-gb10.json` (the receipt), `.txt` (the
verdict lines), `.log` (each request), `.block.log` (the driver's steps and
cache changes) and `.gpu.log` (the process list samples). `hold.log` is the
order of the three passing runs. The engine API's log of each step is not
here: the repository holds no archives, and the receipt carries what the
gate read from it.

## What this does not show

One model, one tokenizer, one rank. The chat layer was not in the path: the
gate names cuts as the chat layer does and sends token ids. The first runs
of `docs/EXACTNESS_GATE.md` on four-rank lanes, then every lane, are still
to come.
