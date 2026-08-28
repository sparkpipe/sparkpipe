# Qwen 3.8 Flash M5.2/M5.4/M5.5 semantics port — 2026-08-28 (session 1)

Worktree /tmp/lane-qwenflash, branch lane/qwen-flash (rebased to origin/main
f58a846). Build/validation node spark4 (reserved lane-qwen-flash after the
stale lane-dsv4bisect reservation expired: acquired 00:16 + 480min TTL =
08:16, no dsv4 processes remained at 09:32; released via `release --node
spark4` then re-reserved — coordinator please note the queue exposes no
`sweep` subcommand and `reserve` refuses while a stale entry lives).

Both blockers CLEARED and consumed:

  * The modeling reference (model_contracts/references/modeling_qwen4_exp.py,
    sha256 77fec77d..., fetched from huggingface/transformers main 2026-08-28)
    supplied the exact hc/indexer/PLE semantics. Ported from it, not assumptions.
  * PLE precision DECIDED: vocab-sharded bf16, 23.84 GiB/rank at TP4, NO
    quantization loss (operator decision honored; fp8 fallback not needed).

## Milestones this session (task numbering; the M5 report used swapped labels)

### S1 = M5.4 hc/GatedResidual — DONE (kernel oracle + module ladder)

Exact semantics ported (reference Qwen4ExpTextGatedResidual + TextModel):

  * Residual stream = 4 x 2560 (hc_count 4); embedding replicated into all
    streams (reference `hidden_states.repeat(1,1,hc_count)`).
  * Per sublayer: hc_norm = GROUP RMSNorm (per-stream, weight [4H]);
    sublayer input = sigmoid(up(silu(down(normed)/4))) weighted stream MEAN;
    output re-enters every stream as raw_stream += 2*sigmoid(inject(normed)/4)
    * sublayer_out. NO other input norms exist (SparseMoe/attention/GDN take
    the mixed vector directly - confirmed against the reference forward).
  * Final readout = hyper_connection_mixer use_combine=False (group norm +
    low-rank mean-mix, NO final norm). The old stream-0-section FINAL_NORM
    approximation is retired; the checkpoint has NO model.norm.
  * MTP: mtp.* keys are IGNORED by the publisher reference
    (_keys_to_ignore_on_load_unexpected), so the MTP wiring is the in-house
    EAGLE convention, hc-adapted: fc input = [norm(embed) | mtp_mixer(
    pre_fc_norm_hidden(streams))], the MTP decoder layer runs the same hc
    machinery with its own mixers, drafts read out through the mtp mixer.

Five new kernels (replicate/group-norm/silu-quarter/mix/inject); decode,
prefill, MTP and the head all rewired onto the stream vector; the hidden
transport packet width is now 4H.

### S2 = M5.2 QSA indexer — DONE (kernel oracle + module wiring)

  * Per full-attn layer (+MTP): 640-row qk projection, per-head RMSNorm(128),
    partial RoPE on the first 64 dims (the shared 32-pair frequency grid).
  * Per-token keys land in a raw key cache (paged layout, 128/token); each
    completing 4-token block pools mean->bf16->RMSNorm->RoPE-at-block-start
    into a pooled cache indexed by COMPRESSED block (slot/4).
  * Per query: relu(per-head dot).sum(heads)/sqrt(128) per block, top-512
    blocks by ordered-key binary search (ties -> smaller block), plus the
    incomplete tail tokens; the u8 mask feeds the decode kernel (null mask
    = pre-indexer behavior; contexts <= 2051 tokens select everything, so
    short-context behavior is bit-identical).

TWO real kernel bugs found by the oracle and fixed:

  1. The pooled cache was indexed per 64-token KV BLOCK: every 4-token pool
     overwrote the previous 15 in the block - only each KV block's LAST
     compressed group survived (pooled cosine vs oracle 0.0009!). Fixed to
     per-compressed-block rows; the module's pooled plane sized x16.
  2. The top-k binary search midpoint `(low+high)>>1` WRAPPED u32 (ordered
     keys live near 0xbf......, low+high exceeds 2^32), converging to
     threshold 0 = select-all. Fixed to `low + ((high-low)>>1)`.

### S3 = M5.5 PLE n-gram block — hash DONE (bit-exact), module wired

  * ngram_size 3 (2-token carried context, EOS-segmented), 8 heads per
    order -> 16 heads x 160; heads 0-7 bigrams (2-term hash), 8-15 trigrams.
  * Hash: i64 WRAPPING multiply + XOR (u64 arithmetic; torch int64 wrap),
    torch.remainder sign (non-negative mod prime), + head offset.
  * ple_layer_ids [2] MEANS layer 1 (reference `layer_idx + 1 in
    ple_layer_ids`) - the M1 census "drift" note is a convention, not drift;
    model header comment corrected.
  * Table: 320,001,536 padded rows (16 primes above 20,000,000, sum
    320,001,446, padded /128), 128 checkpoint shards x [2,500,012, 160];
    rank r packs shards [r*32,(r+1)*32) = 23.84 GiB bf16, out-of-shard
    gathers zero, the bf16 all-reduce completes the embedding (the main
    embedding pattern). layer_multipliers/head_vocab_sizes/head_offsets
    travel as RAW I64 (never converted).
  * Gate: signed-sqrt dot of group-normed key/query streams; dilated
    (kernel 4, dilation 3) depthwise conv over [4H] with a 9-column tail.

### TWO latent live-TP bugs found by inspection (masked by the standalone
### bypass, fatal for the live collective) — FIXED

  1. The shared-expert partial was added AFTER the MoE delta all-reduce;
     the ranks' stream vectors diverged from the next layer on.
  2. The attention/GDN sublayer output (column-parallel projections) had
     NO all-reduce at all - only the MoE delta was reduced.
  Both sublayer outputs now reduce before the stream inject (2 reduces per
  layer + embedding + PLE-embedding reduces per frame).

## Evidence (spark4, commands + raw output)

Mid-pipeline tier (synth v2 pack, PLE omitted under the env gate):

```
make -C modules/qwen4_flash_resident_decode_stage validate NVCC=/usr/local/cuda/bin/nvcc \
  CUDA_ARCH=sm_121a STAGE_COUNT=2 STAGE_LAYER_COUNT=4 MTP_LAYER_COUNT=0 \
  MAX_ACTIVE_SEQUENCES=8 ALLOW_UNQUALIFIED_EXECUTION=1 \
  SPARK_QWEN4_FLASH_STAGE_ALLOW_MISSING_PLE=m STAGE_PACK_PATH=/tmp/q4f_midpipeline_v2.qwen4_flashsp

qwen4_flash_validation check=hc_residual elements=40960 relative_l2=0.00242340746 cosine=0.999997107
qwen4_flash_validation check=indexer_pooled elements=67200 relative_l2=0.00497231492 cosine=0.999987639
qwen4_flash_validation check=indexer_select tokens=2100 blocks=525 topk=512 boundary_flips=0 mask_mismatches=0
qwen4_flash_validation check=ple_hash_gather rows=4 heads=16 bit_exact=1 nonzero=10240/10240
qwen4_flash_validation check=module_decode_vs_prefill elements=10240 relative_l2=0 cosine=1 max_abs=0
qwen4_flash_validation check=module_determinism bit_exact=1
qwen4_flash_validation PASS
```

(All nine prior kernel checks unchanged at their M3 thresholds; the module
decode-vs-prefill gate now compares the full 4H stream vector - 10240
elements, bit-exact.)

Pack plan (dry run, real checkpoint):

```
python3 tools/qwen4_flash_stagepack.py --checkpoint /mnt/model-warm/qwen3.8-flash-next \
  --first-layer 0 --layer-count 48 --tp-degree 4 --tp-rank 0 --dry-run
qwen4_flash_stagepack slice=0+48 tp=4/0 tensors=1246 file_bytes=60194156288 file_gib=56.06 (dry run)
```

56.06 GiB/rank = v2's 30.89 + 1.27 hc + 0.043 indexer + 23.84 PLE ngram.

## Format v2 (breaking, fail-closed)

55 tensor kinds (23 new), u64 coverage bitfields, norm slots at [1,10240],
I64 weight format, expected-count split (real packs always carry the 10 PLE
tensors; SYNTHESIZED mid-pipeline packs may omit the whole block ONLY under
SPARK_QWEN4_FLASH_STAGE_ALLOW_MISSING_PLE - the 23.8 GiB table is not
synthesizable at true shape; real v3 packs load with the env unset).

## INTEGRATION REQUEST

  1. (coordinator) spark6 is at 96% disk (161G free). packs_v3 (~56G) fits
     but leaves ~105G; flagging before the fleet deploy.
  2. (coordinator, FYI) queue `sweep` subcommand missing; stale
     reservations block `reserve` until manually released.
  3. (driver lane) the hidden transport packet for qwen4_flash now carries
     the 4H stream vector (hidden_dimension 10240); the serving adapter's
     capture and any PP stage boundaries must match.

## Next (session 2)

  1. v3 rank0 pack verify (pack_verify) + whole-stack TP4 standalone
     validator on spark4 (real weights, PLE live; the build was launched).
  2. Ranks 1-3 build + verify + deploy packs_v3 to spark4-7.
  3. M6 driver compile -> residentd/api -> LIVE 4-node smoke -> B1 cell.

## Commits (this session)

  * 9da7a65 v2 format: hc+indexer+PLE port, kernels, module rewiring,
    packer/verifier, synth PLE gate; mid-pipeline ladder PASS.
  * 9b123aa ratchet 187879 -> 192135.
  * 17982b0 hc_residual oracle check.
  * a4e7585 indexer pooled-index + midpoint fixes; oracle PASS.
  * 0c3053a PLE hash oracle bit-exact.
