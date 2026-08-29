# r3-flashdecode lane report — 2026-08-29

Lane: `lane/r3-flashdecode` at `7ceb7cc`. Base: origin/main `d07be2b`.
Branch pushed; **integration requested, never self-merged**.

## The rock, located

PERF_PROGRAM2's R3 line — "the shared attention family: 2-byte scalar
loads, block reduction per position, KV read twice, no split-K, 24-64
CTA grids. The DSA scorer shares the disease; its sibling can't launch
past 64K positions" — maps clause for clause onto
`inference/kernels/attn.cuh`, not onto the dsv4 module's sparse
attention (that kernel has been split across SMs since 16fca11, and its
`SparkDsv4SparseAttnKernel` + `SparkDsv4SparseAttnMergeKernel` pair is
the in-repo REFERENCE the fix follows):

- `LmLatentAttentionDecodeKernel`: grid `(rows, heads)` — at decode
  rows==1 that is exactly the 24-64 CTA grid (glm52: 64 heads; mimo25's
  16-CTA case is named in the header's own grouping comment);
- one CTA walks the WHOLE position range serially, an
  `LmBlockSum` (block reduction) per position;
- each KV slot is read twice — once for the score, once for the value
  accumulate (attn.cuh lines ~308-328);
- `LmSparseScoreKernel` (the DSA-style scorer) puts position on
  `blockIdx.y`: the 65,535 ceiling — the same ceiling its sibling
  `LmWeightedSparseScoreKernel` documented at line 401 ("Position is
  blockIdx.x so million-token contexts do not hit CUDA's 65,535
  blockIdx.y ceiling").

The consumers are glm52 and glm5_next (`modules/*/source/cuda/
layer.cuh`). The dsv4 module (my briefing's pointer) is untouched —
R2c's bulk-prefill and W2b's arena-attach contracts stay exactly as
merged; the module-header comment in the shared file had already
prescribed the shape: "Splitting the context across CTAs as well (an
exact online-softmax merge over partials) would add parallelism beyond
this, but needs a workspace and a second pass."

## Design (the R2c discipline, adapted)

`LmLatentAttentionDecodeSplitKernel` partitions the position range
CONTIGUOUSLY across `blockIdx.z` and runs the IDENTICAL per-position
body (same block-reduction width, same online-softmax update — a
partition is the base kernel restricted to a range). Partitions derive
from the DEVICE-side position count; the host only supplies an upper
bound for policy. `LmLatentAttentionDecodeSplitCombineKernel` merges the
per-partition (max, sum, accumulator) states in ASCENDING partition
order — one fixed evaluation order, dividing per element exactly like
the base epilogue.

On bit-identity, stated plainly: the single-pass kernel's running
rescale chains through every position, so a DIFFERENT partitioning
cannot reproduce its bits in the general case (FP multiplication order).
The mission anticipated this ("partition boundaries must respect the
accumulation order or use a deterministic combine"). What IS exact:

- one live partition (splits == 1, or all others empty tails): the
  combine multiplies by `exp(0) = 1` and adds zeros — bit-identical;
- empty context: `-INFINITY` partial maxes scale to zero and
  `0 / fmaxf(0, 1e-20)` reproduces the base kernel's 0;
- the general case is the same softmax up to where the rescale
  multiplies round — oracle-bounded, and DETERMINISTIC (fixed order),
  which is the property the GPU cell's kill-switch needs.

Engagement is deployment policy: `decode_split_context_threshold`
(NodeContext, ABI 4→5 both families; serving adapters parse the key;
the 16 committed glm5_next stage configs carry `0` = disabled =
byte-for-byte shipped behavior; no committed deployment changed
behavior). Below the threshold the launcher performs the same single-
pass launch byte for byte; with no workspace, or when the grid already
fills the machine (`SMs * 4 / (rows*heads)` partitions < 2), it also
falls back. Partitions cap at 16; the workspace is
`rows * heads * 16 * (latent + 2)` floats per execution slot (~1 MB at
glm52 TP1, ~0.25 MB/rank at TP8), pinned to the kernel's cap by a
`static_assert` in each layer.cuh.

## Receipts (offline qualification)

- **Host proofs** (run on every host in offline gates,
  `tests/test_glm52_layer_host.py`): empty-tail two-partition split
  BIT-EXACT vs the single-pass kernel; threshold-0 launcher BIT-EXACT
  vs the direct launch; 16-partition run deterministic across runs and
  within the 5e-2 oracle tolerance. All six codec variants agree.
- **nvcc sm_121a clean**: `make archive` (b1 bucket, fp8) for BOTH
  modules on spark8 (GB10, CUDA 13.0) — exit 0, no new warnings
  (`glm52` + `glm5_next` archives, host objects `-Wall -Wextra
  -Werror`).
- **Module validator EXECUTED on spark8** (fp8, retained rank00 pack):
  all tiers PASS — including the new split leg:
  `split_forward_hidden relative_l2=0.0061 cosine=0.99998`,
  `split_forward_residual 0.0035/0.99999` (the base walk's own numbers
  are 0.0060/0.0035 — the split path sits in the same oracle band),
  and `split_determinism elements=12288 bit_exact=1`.
- **Architecture contract test**: no re-pin needed — the pinned
  inventories (dsv4 wavefront/bulk markers) are untouched; the full
  test suite passes inside the gates.
- **Code-size ratchet 227318 → 227933 exact**, justification in-commit
  (`tests/test_code_size.py`): the launcher wiring, two call sites, two
  adapters, two firmware headers, the validator split leg, and the
  deployment keys. The validator leg was factored into
  `SparkGlm52ValRunSplitLeg` to hold the validation CCN budget at 90
  (max CCN 90, held).
- **make offline-gates fully green** (`OFFLINE GATE PASS: build-all
  run-tests package-manifest`), SHA256SUMS + PACKAGE_MANIFEST resynced
  including the queue entry below.

## Measurement: staged + QUEUED, not run

`tools/devcycle/r3flash_exact_cell.sh` (glm52.tp8.fp8 fleet, spark8
driving ranks on spark0-7) is committed and queued at priority 0
(`r3flash-glm52-exact-cell` in `runs/queue.jsonl`). Variants differ in
ONE deployment key; the published b1 driver serves both:

1. phase A — threshold 0, O128 decode gate (the byte-for-byte control);
2. phase B — threshold 2048, SAME gate, **bit-exact token-stream
   kill-switch**: a mismatch exits 6 with a RED LIGHT in
   `summary.json` and no timing may be trusted;
3. phase C — 32K-context decode timing, split vs the re-run control.

**NO PERF CLAIM is made.** The mechanism is qualified; the number waits
for the cell. Note for the runner: the script fails loudly (exit 9) if
the lane binaries or the O128 batch json are missing — stage
`/tmp/r3flash-o128-batch.json` from the retained glm52 cell files
before running.

## Observations (not my rocks, recorded for owners)

- `LmWeightedSparseScoreKernel` still scores every position (linear in
  context); the hierarchical summary/refine kernels sit unused in the
  same file — the header's own table prices that at 3.8-57x for the
  selection pass. That is a separate rock from split-K attention and
  would compound with it.
- The validator's CCN budget (90) is now exactly at the ceiling; the
  next validator leg needs its own function from the start.
- The `cache/` source directory vs runtime-cache naming bit this lane's
  first rsync (`--exclude cache` dropped real sources); the devcycle
  sync scripts that exclude `cache/` as a runtime dir are a footgun.

## Integration request

`lane/r3-flashdecode` @ `7ceb7cc` (+ this report's commit): split-K
flash-decode for the shared latent decode attention, threshold-gated
and default-off, with host + GPU-validator exactness receipts, ratchet
and CCN held, and the measurement cell queued at priority 0. The cell's
kill-switch hashes gate the landing, not the code freeze.
