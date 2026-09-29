# Teacher-forced score dump

Experiment-only instrumentation for the quantization A/B campaign (design:
`lanes/quant-ab-design.md` §3.2 and §5 H1). Every tensor-parallel rank writes
statistics of its own lm_head vocabulary shard for every prompt row it scores.
The rank files are merged offline in rank order. Nothing is exchanged between
ranks, and the served token is still the certified head's argmax.

## Build and enablement

- The hook exists only in `SCORE_DUMP=1` builds of the glm5_next module:
  `make -C modules/glm5_next_resident_decode_stage SCORE_DUMP=1 EXPERT_CODEC=... archive adapter`.
  - The module identifier gains the suffix `.scoredump`.
  - Objects go to `build/modules/glm5_next_resident_decode_stage/<codec>-scoredump`.
  - The CUDA launcher (`source/cuda/score_launch.cuh`) is force-included only in this build, so
    the production CUDA unit is byte-identical.
- The dump is enabled per root by three optional stage-config members. Each is a normalized path
  relative to the runtime root:
  - `score_dump_directory` (required when either of the others is present);
  - `score_probe_path`;
  - `score_tier2_rows_path`.
- The production adapter keeps its exact member list, so it rejects these members. It returns
  SCHEMA_ERROR, and `tests/test_glm5_next_adapter_config_load.py` gates this.
- `tools/glm5_next_gen_deployment.py` refuses them for the production root and for the committed
  tree. `tests/test_deployment_config_drift.py` checks both.
- Prefix reuse must be off: every request must report `cached_prompt_tokens == 0`. A row whose
  sequence history the module has not seen from position 0 is written without a key and counted
  as keyless.

## What a rank writes

`<score_dump_directory>/score.rNN.bin` is opened with O_EXCL, so a run never overwrites
an earlier dump. It holds:

- A 144-byte header: rank, tp, shard range, vocabulary, hidden size, top-k (64), and
  whether Tier-2 is on. It also carries the arm digest and the SHA-256 of the probe and
  Tier-2 files.
- One 568-byte row record per scored row, followed by that row's probe entries:
  - key, position, input token and served token;
  - flags: key valid, probed, Tier-2, non-finite;
  - local max, and the local sum of exp(logit − local max) in float64;
  - the local top-64 ids and logits, ordered by value descending, with ties broken by lowest id;
  - the (id, logit) pairs for every id in the row's probe entry that this rank owns.
- An end record with counts: rows, waves, skipped waves (speculative verify or graph replay),
  keyless rows, probed rows, Tier-2 rows and non-finite rows. A file without the end record is
  incomplete, and the merge refuses it.

`tier2.rNN.bin` holds the raw fp32 shard logits of the rows in the Tier-2 list.

The logits are the BF16 lm_head applied to the final-normed hidden row, accumulated in fp32. Lane
`l` of a warp sums elements `l, l+32, ...`, then a fixed xor-butterfly reduces across lanes. The
result depends only on that row, never on how the rows were batched. `tests/test_score_dump_cuda.py`
checks it bit for bit against a numpy emulation of the same order at the Flash head geometry.

## Row identity

- key(p) = FNV-1a-64 over the little-endian bytes of `tokens[0..p]`.
  - Seed `0xcbf29ce484222325`, prime `0x100000001b3`.
  - Python: `score_merge.row_keys(tokens)`.
- A row is identified by `(key, position)`. Documents that share a prefix share rows.
- The probe file stores the union of their next-token targets.

## Two-pass flow

1. `tools/score_merge.py targets corpus.jsonl --out targets.bin` writes the next-token targets.
   The corpus is JSON lines: `{"doc", "tokens"}`.
2. `tools/score_merge.py tier2 corpus.jsonl --count N --seed S --out tier2.bin` writes the Tier-2 rows.
3. The reference run uses `score_probe_path=targets.bin`.
4. `tools/score_merge.py merge score.r*.bin --probe targets.bin --out ref.merged --probe-out probe2.bin`
   - Log Z is computed as M + log Σ_r s_r·exp(m_r − M), summed in rank order.
   - The global top-64 is exact.
   - The output is byte-reproducible.
   - The merge refuses on any header, shard, row-identity or probe mismatch.
5. Every arm run uses `score_probe_path=probe2.bin` (the reference top-64 plus the targets).
6. `tools/score_kl_partial.py compare ref.merged arm.merged corpus.jsonl --out arm.json` gives,
   per document:
   - KL_b: the reference top-64 buckets plus one tail bucket, a lower bound on the true KL;
   - the reference tail mass;
   - top-1 agreement, with reference near ties (top-2 gap < 1e-3 nats) excluded and counted;
   - decisive flips (reference margin > 1 nat);
   - `dlogp = logp_Q(y) − logp_P(y)`, where dNLL = −dlogp.
7. Tier-2 exact KL:
   - On each node: `score_kl_partial.py partial --ref-tier2 --arm-tier2 --ref-merged --arm-merged --out partial.rNN.bin`.
   - Then: `score_kl_partial.py combine partial.r*.bin --out exact.json`, which sums in rank order
     and checks that the reference mass is 1 within 1e-6.
