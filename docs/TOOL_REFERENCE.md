# Tool reference

This file describes command-line tools under `tools/` and the libraries they
link. For each tool it gives the purpose, the arguments as the tool's
`main()` parses them, and the meaning of each exit code. Behaviour notes
describe what the code does, including known gaps.

## hy4 host reference tools (tools/hy4_dequant)

Single-threaded CPU programs, and the loader library most of them link,
for the TP16 rank bundles that `tools/hy4_tp16_shard.py` cuts from the UD-IQ1_M
GGUF. The forward tools evaluate the hy4 model, or parts of it, in fp32 for
comparison with other implementations. All arguments are positional.
Results go to stdout and stderr as text; files a tool writes are listed in
its section. Names such as `build_hc_pre` refer to functions in the vendored
hy4 reference that `tools/hy4_dequant/hy4_layer_semantics.md` names as
ground truth.

### Terms and shared behaviour

- **Rank bundle** (`pack_dir`): a directory holding `manifest.json` (schema
  `hy4-tp16-shard-v1`), the rank GGUF named by the manifest's `gguf` field,
  and a `<gguf>.sha256` sidecar. None of these tools reads the sidecar.
- **All-ranks directory** (`allranks_dir`): a directory holding the 16
  bundles `rank-00` to `rank-15`.
- **Build**: tools that include `hy4_rank_loader.h` link
  `hy4_rank_loader.c`, for example
  `cc -O2 -o hy4_generate hy4_generate.c hy4_rank_loader.c -I . -lm` (the
  line `run_gen_anchor.sh` uses). `hy4_iq_dequant_ref.c` needs only
  `hy4_iq_dequant_vendor.h`.
- **Matrix layout**: GGML stores `dims[0]` fastest. A dequantized 2-D tensor
  with manifest dims `[cols, rows]` is a row-major `rows x cols` float
  matrix, so row `r` starts at element `r * cols`.
- **Expert slabs**: `shard_rules` in `tools/hy4_tp16_shard.py` splits
  `ffn_{gate,up,down}_exps` on dim 2, 16 experts per rank. Rank `r` owns
  experts `16r` to `16r + 15`. Expert `li` of a bundle is the contiguous
  byte range at `file_offset + li * (nbytes / 16)`, because a dim-2 slab
  holds whole quant blocks. The tools decode one expert at a time from a
  copy of the view with that offset and length.
- **Fixed rank**: `hy4_layer_forward`, `hy4_moe_forward` and
  `hy4_stack_forward` hard-code rank 2: experts 32 to 47 and, in
  `hy4_stack_forward`, global heads 8 to 11 and vocabulary ids from
  `2 * 7552`. Run them on `rank-02`. On another bundle these indices are
  wrong.
- **Decoded ggml types** (`dequant_view` in each forward tool): 0 (F32),
  8 (Q8_0), 12 (Q4_K), 13 (Q5_K), 14 (Q6_K), 16 (IQ2_XXS), 18 (IQ3_XXS) and
  29 (IQ1_M). `hy4_generate` and `hy4_stack_forward` also decode 23 as IQ4_XS.
  Any other type fails the load. See the vendored-dequantization section of
  `tools/hy4_dequant/hy4_layer_semantics.md` for block sizes and an open
  question about type 23.
- **Router** (`hy4_generate`, `hy4_layer_forward`, `hy4_moe_forward`,
  `hy4_stack_forward`): `probs = sigmoid(ffn_gate_inp . x)`. The 8 experts
  with the largest `probs + exp_probs_b` are selected, so the bias affects
  selection only. Weights are `probs[sel] / max(sum, 2^-14) * 2.827`, where
  `sum` is the sum of the 8 selected `probs` (the code writes 2^-14 as
  `6.103515625e-5`).
- **Routed SwiGLU**: the up projection is clamped to [-10, 10] and the gate
  is capped at +10 (before or after `silu` depending on the tool, see the
  table below). The shared expert and the layer-0 dense FFN (intermediate
  18432) are not clamped. The shared expert output is added with weight 1.
- **hc streams**: the residual state is 4 streams of 6144 values. `hc_fn`
  maps the flattened 4 x 6144 vector to 8 coefficients `m`;
  `pre = sigmoid(m[0:4] * scale[0] + base[0:4]) + 1e-6` and
  `post = 2 * sigmoid(m[4:8] * scale[1] + base[4:8]) + 1e-6`. The branch
  input is `sum_s pre[s] * stream[s]`, RMS-normed with `attn_norm` for
  attention or `ffn_norm` for the FFN; `x` below is that normed vector.
  After the branch, `stream[s] += post[s] * branch_out`, the same update as
  the vendor `build_hc_post`.
- **Attention** (absorbed MLA): `q = attn_q_b . rms_norm(attn_q_a . x)` (norm
  weight `attn_q_a_norm`), with the last 64 of each head's 256 dims roped.
  `attn_kv_a_mqa . x` gives a 512-dim latent (RMS-normed with
  `attn_kv_a_norm`) and 64 roped key dims. The score is
  `((attn_k_b . q_nope) . latent + q_pe . k_pe) / sqrt(256)`, the softmax
  includes a per-head sink from `attn_sinks`, and the head output is
  `attn_v_b . (sum_k p_k * latent_k)`, scaled by `sigmoid(attn_gate . x)` and
  projected by `attn_output`. RoPE base is 1e7. Apart from
  `hy4_indexer_forward`, the tools do not evaluate the DSA indexer: they
  process at most 9 positions, and a 2048-entry top-k (`index_top_k` in
  `model_contracts/hy4_ud_iq1m_authoritative.json`) selects all of them.
- **Norm epsilon**: RMS norms use 1e-5 unless a section says otherwise.

The three forward tools differ in these details:

| | `hy4_generate` | `hy4_stack_forward` | `hy4_layer_forward` |
|---|---|---|---|
| Heads | all 64, 4 from each rank's bundle | rank 2's 4 heads | 64 from one bundle (expects unsliced tensors) |
| `hc_fn` input | flattened streams RMS-normed (no weight, eps 1e-5), as in the vendor `build_hc_pre` | raw streams | raw streams |
| Sink | extra softmax term only | extra softmax term, and also added to every logit | same as `hy4_stack_forward` |
| Routed gate cap | before `silu` | after `silu` | after `silu` |
| RoPE | adjacent pairs `(2d, 2d+1)` (`rope_pairs`) | NEOX halves `(d, d+32)` (`rope_neox`) | NEOX halves |
| `matvec` accumulator | double | float | float |

`hy4_moe_forward` caps the gate after `silu`.

### hy4_rank_loader.h

Library API, implemented in `hy4_rank_loader.c`. It opens one rank bundle,
joins the manifest's tensor entries with the GGUF tensor-info table by name,
and returns tensor views. It reads local files with stdio only.

`hy4_tensor_view` fields:

- `name`: tensor name from the manifest, truncated to 159 bytes.
- `dims`, `n_dims`: this rank's dims from the manifest `dims` array (up to
  4), in GGML order.
- `type`: ggml type id from the GGUF tensor info. The manifest's own `type`
  field is not read.
- `slice_kind`: 0 when the manifest `slice` is `"replicate"`; `dim + 1` when
  it is an object `{"dim": d, ...}`, so 1 is dim 0, 2 is dim 1 and 3 is
  dim 2. The offset by one keeps a dim-0 split distinct from replicate.
- `slice_start`: the slice object's `start`; 0 for replicated tensors.
- `slice_count`: never set by the loader; always 0.
- `nbytes`: the manifest `bytes` field, this rank's byte length.
- `file_offset`: absolute offset in the rank GGUF: the aligned start of the
  data section plus the tensor-info offset.

`hy4_rank` fields: `path` (the `pack_dir` passed in), `file_bytes` (size of
the GGUF), `tensor_count`, `views` (one per manifest tensor, in manifest
order) and `file` (the open `FILE *`). `manifest_data` and `gguf_handle` are
never set. Callers read these fields directly; `hy4_reassemble` uses
`tensor_count`, `views` and `file`.

`int hy4_rank_open(const char *pack_dir, int tolerate_sha_mismatch, hy4_rank **out)`

- Reads `<pack_dir>/manifest.json`, opens `<pack_dir>/<gguf>`, parses the
  GGUF header and builds one view per manifest tensor object.
- Returns 0 and sets `*out` on success. Failures: 1, manifest unreadable;
  2, manifest has no `gguf` string; 3, the GGUF cannot be opened; 4, GGUF
  header rejected (bad magic, version below 2, a metadata value type it
  cannot skip, a tensor name of 160 bytes or more, or a short read);
  5, manifest has no `tensors` key.
- `tolerate_sha_mismatch` is ignored. Neither the loader nor any caller
  verifies a digest.
- The data section starts at the end of the tensor-info table, rounded up to
  `general.alignment` when that key is stored as uint32, otherwise to 32.
- The manifest reader is a string scanner for the fixed layout that
  `tools/hy4_tp16_shard.py` writes, not a JSON parser. It takes the first
  `}` after a tensor's `{` as the end of the object, which for a sliced
  tensor is the end of the nested `slice` object. It therefore depends on
  the key order of `json.dumps(..., sort_keys=True)`: in each tensor object
  `name` comes before `slice`. If the schema changes, regenerate the bundle
  with the sharder and update the scanner.
- Silent cases: parsing stops at the first tensor object it cannot read, and
  the call still returns 0 with `tensor_count` set to the number of objects
  read before it. At most as many views are built as the GGUF has tensor
  infos. A manifest tensor whose name is not in the GGUF keeps `type` 0 and
  `file_offset` 0.

`void hy4_rank_close(hy4_rank *rank)`: closes the GGUF and frees the views
and the handle. Accepts NULL.

`const hy4_tensor_view *hy4_tensor_lookup(const hy4_rank *rank, const char *name)`:
linear search by exact name. Returns NULL if the name is absent.

`int hy4_tensor_read(const hy4_rank *rank, const hy4_tensor_view *tv, void *dst)`:
seeks to `tv->file_offset` and reads `tv->nbytes` bytes into `dst`, which
must hold that many. Returns 0 on success, -1 on a seek failure or short
read. `tv` may be a modified copy of a view to read a sub-range. All reads
share the handle's one `FILE *`, so concurrent calls on the same `hy4_rank`
are unsafe.

### hy4_rank_loader_test.c

Checks `hy4_rank_open` and `hy4_tensor_lookup` on a real bundle against
fixed expected values.

Usage: `hy4_rank_loader_test <pack_dir>`

For each tensor below it compares `type`, `n_dims`, `dims`, `slice_kind` and
`nbytes`, and requires `file_offset > 0`:

| Tensor | `type` | `dims` | `slice_kind` | `nbytes` |
|---|---|---|---|---|
| `token_embd.weight` | 12 | 6144, 7552 | 2 | 26099712 |
| `blk.0.attn_kv_a_mqa.weight` | 8 | 6144, 576 | 0 | 3760128 |
| `blk.47.ffn_gate_exps.weight` | 16 | 6144, 2048, 16 | 3 | 51904512 |
| `blk.47.ffn_down_exps.weight` | 18 | 2048, 6144, 16 | 3 | 77070336 |
| `output.weight` | 0 | 6144, 7552 | 2 | 185597952 |

It then reads `blk.0.attn_norm.weight` with `hy4_tensor_read` if that tensor
exists with `nbytes` 24576. If it is missing or has another size, the read
check is skipped and no failure is counted. Every checked field is the same
on all ranks (the sharder cuts equal slices) and `slice_start` is not
checked, so the test does not tell which rank it was given.

Exit codes: 0, all checks passed (`LOADER TEST OK`); 1, open failed, or a
tensor was missing, mismatched or unreadable (`LOADER TEST FAIL (n)`); 2,
wrong argument count.

### hy4_iq_dequant_ref.c

Decodes `nelems` values of one quantized tensor straight from a file offset
and prints statistics and a hash, to compare against another decoder. The
IQ types go through the vendored decoders; Q8_0 is decoded inline. It does
not use the loader.

Usage: `hy4_iq_dequant_ref <gguf_file> <offset> <type> <nelems>`

- `offset`: absolute byte offset of the tensor data, such as a view's
  `file_offset`.
- `type`: 8, 16, 18 or 29.
- Reads `nelems / 256` blocks of 34, 66, 98 or 56 bytes. `nelems` should be
  a multiple of 256; values past the last whole block are left
  uninitialized but still enter the statistics.
- Prints `type= nblocks= elems= nan= sum= amax= fnv1a=` (64-bit FNV-1a over
  the decoded float bytes), then the first 8 values.
- Defect: type 8 (Q8_0) uses the same `nelems / 256` block count although a
  Q8_0 block holds 32 values. Only `nelems / 8` values are decoded, and the
  statistics and hash cover uninitialized memory after them.

Exit codes: 0, no NaN in the output; 1, at least one NaN; 2, wrong argument
count or unsupported type; 3, the file could not be opened, seeked or read.

### hy4_indexer_forward.c

Computes the DSA lightning-indexer scores of layer 1 for a fixed 4-token
input on one bundle.

Usage: `hy4_indexer_forward <pack_dir>`

- Input `x[t][i] = 0.1 * sin(i + 17t)` for t = 0..3, used as is (no
  `attn_norm`).
- `qr = rms_norm(attn_q_a . x) * attn_q_a_norm`, eps 1e-6.
- `iq = indexer.attn_q_b . qr`: 32 heads of 128.
- `ik = LayerNorm(indexer.attn_k . x)` with `indexer.k_norm` weight and
  bias, eps 1e-5.
- RoPE (NEOX halves, base 1e7, position t) on each `iq` head and on `ik`.
  The code rotates dims 0 to 63. The vendor `build_indexer_top_k` keeps the
  rope part last (dims 64 to 127).
- `iw = indexer.proj . x`: 32 values per token.
- `score[q][k] = (sum_h relu(ik[k] . iq[q][h]) * iw[q][h]) / sqrt(128 * 32)`.
- Prints the scores of the last query (q = 3) against k = 0..3. It does not
  compute a top-k selection.
- stderr: the type, byte length and a raw-byte sample of `attn_q_a`, its
  decoded maximum and nonzero count, and the largest magnitudes of `qr`,
  `iq`, `ik` and `iw`.
- Limitation: `shard_rules` splits `indexer.attn_q_b` by head (2 of 32 heads
  per rank), but the tool loops over all 32 heads (`IDX_HEADS`). On a bundle
  sharded that way, 30 heads read uninitialized memory.
- The buffers for `attn_q_a_norm`, `indexer.k_norm` and `indexer.proj` are
  sized from `nbytes`, so those tensors must be F32.

Exit codes: 0, done (`INDEXER FORWARD DONE`); 1, open failed, `attn_q_a`,
`indexer.attn_q_b` or `indexer.attn_k` missing, or a decode failed; 2, wrong
argument count. A missing `attn_q_a_norm`, `indexer.k_norm` or
`indexer.proj` is not checked before use.

### hy4_layer_forward.c

Runs layer 1 (attention and sparse MoE) for 4 positions on one bundle,
using only that bundle's routed experts.

Usage: `hy4_layer_forward <pack_dir>` (rank 02)

- One stream state is carried across positions: all 4 streams start as
  `0.1 * sin(i + 17)`, and position t starts from the streams that position
  t - 1 left. Attention covers the cached keys of positions 0..t.
- Attention, hc and SwiGLU details are in the table above. Only selected
  experts in 32 to 47 contribute; the shared expert is added.
- Prints `t= used_local= stream_sum= stream_amax= nan=` per position.
  `used_local` counts selected experts in 32 to 47. `stream_amax` prints
  `|stream[0]|`, the first element, not the maximum.
- Stops with `RED STOP: NaN` if the streams contain NaN.
- Limitation: buffers and loops assume all 64 heads in `attn_q_b`,
  `attn_k_b`, `attn_v_b`, `attn_gate` and `attn_output`. `shard_rules`
  splits these by head, so on a bundle sharded that way the buffers are only
  partly filled from the file and the attention output is not meaningful.
  `hy4_stack_forward` handles the 4-head slices.

Exit codes: 0, done (`LAYER FORWARD DONE`); 1, open failed, a tensor failed
to load (`LOAD FAIL: <name>`), a shared-expert tensor is missing or cannot be
decoded, an expert slab cannot be decoded, or NaN; 2, wrong argument count.
A missing `ffn_*_exps` tensor is not checked before use.

### hy4_moe_forward.c

Runs the layer-47 router, the bundle's routed experts and the shared expert
for one synthetic token.

Usage: `hy4_moe_forward <pack_dir> [force_expert]` (rank 02)

- Input `x[i] = 0.1 * ffn_gate_inp[40][i] + 1e-4 * sin(i)`. Building `x`
  from the router row of expert 40, which rank 2 owns, steers the selection
  toward an owned expert.
- The router buffers are sized from `nbytes`, so `ffn_gate_inp` and
  `exp_probs_b` must be F32.
- Prints a router line for expert 40 (its `selkey` and `maxlogit` fields
  always print 0) and `top8:` with the selected experts and normalized
  weights, before any forcing.
- `force_expert` (ignored when negative) replaces the first selected expert
  with the given id at weight 1.0. The other seven selections and their
  weights stay, and the run is not limited to one expert. Selections outside
  32 to 47 are skipped.
- For each owned selected expert, stderr gets `slab li=...` with NaN counts
  and maximum magnitudes of the three decoded slabs. A NaN stops the run.
- Prints `rank-local experts used: n, routed partial sum s amax a`, where
  `a` is `|y[0]|`.
- Writes `/tmp/hy4_moe_trace.bin` before the shared expert: `x` (6144 f32),
  `exp_probs_b` (256 f32), router logits (256 f32), weights (8 f32) and
  selected ids (8 int32), after any forcing. It is written first so the
  trace exists even when a shared-expert tensor has a type this tool cannot
  decode (it does not decode type 23).
- If all three `ffn_*_shexp` tensors exist, adds the unclamped shared expert
  and prints `after shared:`. Otherwise the shared expert is skipped.

Exit codes: 0, done (`MOE FORWARD DONE`); 1, open failed, `exp_probs_b` or
an expert tensor missing, a decode failed, or a slab contains NaN; 2, no
arguments. A missing `ffn_gate_inp` is dereferenced before the router check.

### hy4_stack_forward.c

Runs all 78 layers for 4 synthetic positions with one bundle's slices and
scores that rank's vocabulary slice at the last position.

Usage: `hy4_stack_forward <pack_dir>` (rank 02)

- Layer-major: the 4 positions pass through each layer in order (filling
  its KV cache) before the next layer starts. A layer's hc, norm and
  attention weights are loaded at the start of the layer and serve all 4
  positions; the router, expert slabs, shared expert and layer-0 dense FFN
  are loaded again for every position.
- Each position has its own streams, all 4 set to `0.1 * sin(i + 31t)`. No
  real embedding is used: `token_embd` is split by vocabulary across ranks.
- Attention uses the bundle's 4 heads: local indices 0 to 3 in `attn_q_b`,
  `attn_k_b`, `attn_v_b` and `attn_gate`, global heads 8 to 11 in
  `attn_sinks`. `attn_kv_a_mqa` and `attn_kv_a_norm` are replicated, so the
  bundle computes the full latent and roped key for every position.
  `attn_output` is split on its input (head) axis (`split0` in
  `shard_rules`); the tool reads it as a 6144 x 1024 matrix that yields a
  full-width partial output.
- MoE uses experts 32 to 47 plus the shared expert. Layer 0 uses the dense
  FFN.
- Final step: `output_hc_fn` on the last position's raw streams,
  `pre = sigmoid(m * output_hc_scale[0] + output_hc_base) + 1e-6`, the
  weighted stream sum, `output_norm`, then the 7552-row `output.weight`
  slice. Prints the local argmax, its global id (`local + 2 * 7552`) and the
  score.
- No all-reduce happens inside or between layers: every branch after the
  first attention branch consumes streams that lack the other 15 ranks'
  attention heads and routed experts. The result is not a partial sum of the
  full model's output, and adding the outputs of 16 such runs does not
  reproduce it. `hy4_generate` sums the rank partials inside every layer.

Exit codes: 0, done (`STACK FORWARD DONE`); 1, open failed, an attention,
hc, norm, dense-FFN or final tensor is missing or cannot be decoded, or an
expert slab cannot be decoded; 2, wrong argument count. A missing router,
shared-expert or `ffn_*_exps` tensor is not checked before use.

### hy4_generate.c

Greedy generation with the whole model on one node: it opens all 16 bundles
and computes the TP16 partial sums in-process.

Usage: `hy4_generate <allranks_dir> [gen] [promptfile] [dumpprefix]`, with
optional environment `HY4_CKPT=<path>`.

- `gen`: positions to run after the prompt (default 1).
- `promptfile`: whitespace-separated decimal token ids; at most 8 are read.
  Ids are used as given and no BOS is added. `hy4_tokenize.py encode` prints
  ids in this form and adds no BOS either. Without a prompt file, or with
  one that yields no ids, the prompt is 8 zeros.
- Prompt length plus `gen` must not exceed 9 (`TOTAL_TOK`). The code does
  not check this.
- `dumpprefix`: after every layer at every position, writes the 4 x 6144 f32
  streams to `<dumpprefix>_L<layer>_t<pos>`.
- Embedding: token `id` is row `id % 7552` of rank `id / 7552`'s
  `token_embd.weight` slice, copied into all 4 streams.
- Replicated tensors are read from rank 0: hc functions, scales and bases,
  norms, `attn_q_a`, `attn_kv_a_mqa`, `attn_sinks`, the router and its bias,
  the shared expert, the layer-0 dense FFN and `output_hc_*`. Head-sliced
  attention tensors, expert slabs and `output.weight` slices are read from
  each rank. Missing rank-0 tensors are not checked before use.
- Attention runs each rank's 4 heads through that rank's `attn_output` slice
  and sums the 16 partial outputs in rank order 0..15. The MoE runs each
  rank's owned selected experts and sums in the same order. These sums are
  fp32; a real all-reduce may add in another order and differ in the last
  bits.
- Head: the flattened streams are RMS-normed, `output_hc_fn` gives 4 mixes,
  `pre = sigmoid(m * output_hc_scale[0] + output_hc_base) + 1e-6`, and the
  streams are summed with those weights, as in the vendor `build_hc_head`.
  Then `output_norm`, the 16 vocabulary slices, and a global argmax. The
  `output.weight` buffer is sized from `nbytes`, so that tensor must be F32.
- A position at or past the prompt length takes the previous position's
  argmax as its token.
- stdout: per position `t=<t> token id <id> (logit <max>) prompt|GENERATED`,
  where `<id>` is the input token at that position and `<max>` is the
  largest logit there, then `GENERATED TOKEN: <id>` (the argmax at the last
  position) and a line starting `GENERATION DONE`. stderr: `MY_*` probe
  lines at position 0 (most for layer 0 or 1 only; `MY_ATTN_L<layer>` and
  `MY_FFN_L<layer>` for every layer), the 5 largest logits per position
  (`HY4_TOP`), and the logits of five fixed ids at the last prompt position
  (`MY_REFLOGIT`).
- Checkpointing with `HY4_CKPT`: after every layer the state is written to
  `<path>.tmp` and renamed to `<path>`. Layout, in host byte order: 8 uint32
  (magic `0x48593443`, version 1, position, next layer, last argmax, prompt
  length, total positions, 0), token ids (9 int32), streams (9 x 4 x 6144
  f32), KV latents (78 x 9 x 512 f32) and roped keys (78 x 9 x 64 f32). A run
  that finds the file resumes at the stored position and layer and takes the
  tokens, last argmax, prompt length and total from it. After a save, if
  more than 660 s have passed since start, the tool prints
  `CKPT STOP t=<t> il=<next layer>` and exits 0; run it again to continue.
  When done it creates an empty `<path>.done`.
- `run_gen_anchor.sh` runs one such slice per call (4-token prompt, `gen` 1,
  dumps under `$HOME/hy4-anchor/cpu/`) and prints `GEN DONE ALREADY` once the
  `.done` marker exists.

Exit codes: 0, finished, or stopped at the 660 s checkpoint budget; 1, a
rank bundle failed to open, the embedding read failed, a head-sliced
attention tensor is missing or cannot be decoded, an expert or
`output.weight` decode failed, the checkpoint has a bad magic or version or
is truncated (`CKPT CORRUPT`) or could not be saved, or the `.done` marker
could not be created; 2, no arguments, or the prompt file cannot be opened.

### hy4_reassemble.c

Rebuilds one GGUF from the 16 bundles, to check the sharding byte for byte
against the source file.

Usage: `hy4_reassemble <allranks_dir> <out.gguf> <orig_header>`

- Opens `rank-00` to `rank-15` before it checks the argument count.
- Header: the first 5051520 bytes of `<orig_header>`, copied unchanged. The
  source header is needed because rank headers carry sliced dims. The length
  is a constant in the code; it equals `source_precision.gguf_data_offset`
  in `model_contracts/hy4_ud_iq1m_authoritative.json`.
- Tensor order: rank 0's views sorted by `file_offset`. Views come in
  manifest order (sorted by name); the sharder writes rank data in source
  offset order, so this sort restores the source order. Other ranks' views
  are taken at the same index as rank 0's, which relies on every manifest
  listing tensors in the same order, as the sharder writes them.
- Replicated tensors (`slice_kind` 0): rank 0's bytes, once. Sliced tensors:
  the 16 slabs in rank order with no padding between them.
- After each tensor, zero padding to a 32-byte boundary computed from rank
  0's `nbytes`. For a sliced tensor this matches the padding of the
  concatenated length only when the slab length is a multiple of 32.
- Concatenation restores the source bytes for dim-1 and dim-2 slices, where
  each rank's slab is one contiguous range of the source tensor. For dim-0
  slices (`split0`, used for `attn_output`), each rank's slab gathers one
  piece of every row, so rank-order concatenation does not restore the
  source order.
- The output is not verified. The tool prints `REASSEMBLED ...` and a
  reminder that the output's sha256 must equal the Hub LFS oid of the source
  file.

Exit codes: 0, output written; 1, a rank bundle, `<orig_header>` or the
output file could not be opened; 2, wrong argument count (checked after the
ranks are opened); 3, short read of the header, or a seek failure or short
read while copying tensor data.

## hy4 GPU tests (tools/hy4_gpu)

Standalone programs for the hy4 lane. `experts_ck128_check` is a host
program that checks the checksums of a placed FP8 pack. The others are CUDA
programs that compare GPU kernels with host references on rank-pack
weights. Apart from `hy4_dequant_test` (GPU dequant kernels) and
`hy4_fp8_rung` (raw FP8 bytes), they dequantize weights on the host with
`hy4_iq_dequant_vendor.h` (Q8_0 tensors with a local decoder) and upload
float32.

Shared conventions:

- Build. The `run_*_test.sh` scripts, `run_fp8_rung.sh` and
  `build_and_run_fwd.sh` compile in `~/hy4-gpu` with
  `nvcc -O2 -arch=sm_121 -fmad=false`. Every CUDA tool except
  `hy4_fp8_rung` includes `hy4_iq_dequant_vendor.h` from
  `tools/hy4_dequant/`, so the compiler must find that header (next to the
  source or through `-I`). `hy4_layer_test` and `hy4_forward_test` also
  include `hy4_rank_loader.h` and link `tools/hy4_dequant/hy4_rank_loader.c`;
  their scripts take both from `~/hy4-cmp`. `-fmad=false` stops nvcc from
  fusing a multiply and an add into one FMA; the bitwise comparison in
  `hy4_dequant_test` depends on it.
- Rank packs. The tools expect the TP16 GGUF of rank `NN` at
  `rank-NN/model-ud-iq1m-tp16-rank-NN.gguf`. Token `t` is row `t % 7552` of
  `token_embd.weight` on rank `t / 7552`; routed experts are laid out as in
  "Expert slabs" under "Terms and shared behaviour". `hy4_attn_test` and
  `hy4_moe_test` take the rank-00 GGUF and build the path of another rank by
  replacing everything from `/rank-00/` onward with
  `/rank-NN/model-ud-iq1m-tp16-rank-NN.gguf`, so their argument must contain
  `/rank-00/`.
- RoPE rotates adjacent pairs `(2d, 2d+1)` by `pos * 1e7^(-2d/64)` over the
  64 rope dimensions, in the GPU kernels and in the host references.
- Float checks. Each stage (`check`) compares element by element and fails at
  the first element where `|gpu - ref| > tol * max(1, |ref|)`. Every `check`
  except the one in `hy4_qchain_test` also fails a NaN or infinite GPU value.
  A stage prints `<STAGE> PASS (max|d|=..., N elems)` or
  `<STAGE> FAIL elem ...`.
- Exit codes, unless a tool says otherwise: 0 when every stage passes, 1 when
  a stage fails or a checked load step or CUDA allocation fails, 2 on a wrong
  argument count. Lookups are not always checked: where a tool does not check
  one (most lookups in `hy4_attn_test`, `hy4_hc_test`, `hy4_moe_test` and
  `hy4_qchain_test`, the expert tensor lookups in `hy4_layer_test` and
  `hy4_forward_test`, and the `token_embd.weight` lookup in
  `hy4_forward_test`), a tensor missing from the pack crashes the tool
  instead of exiting 1.

### experts_ck128_check.c

Checks a placed FP8 rank pack against its `.experts` sidecar by recomputing
`SparkCk128` over every recorded range.

```sh
experts_ck128_check <pack.safetensors> <pack.safetensors.experts>
```

Sidecar layout as the tool reads it:

- 16-byte header that starts with `WEPX`; the record count is a u64 at byte 8.
  The version field at byte 4 is not read.
- `count` records of 40 bytes: reserved u32 (must be 0), ordinal u32 (must
  equal the record index), offset u64, bytes u64, 16-byte ck128 digest.

These are not the 48-byte version-2 records that `SparkWeightdManifestLoad`
requires.

Output: one `CK128 MISMATCH ordinal=... off=... bytes=... got=... want=...`
line per bad range, then `CK128_CHECK GREEN|RED chunks=N mismatches=M`.

Exit codes: 0 when every range matches, 1 on any mismatch, 2 on usage error,
missing sidecar, short header or bad magic, truncated sidecar, bad record,
missing pack, or a failed seek or short read in the pack.

The source is C++ despite the `.c` suffix (it uses `<vector>`).
`spark_ck128.h` has no `extern "C"`, so `src/spark_ck128.c` must be compiled
as C++ too:

```sh
c++ -x c++ -I include tools/hy4_gpu/experts_ck128_check.c src/spark_ck128.c \
  -o experts_ck128_check
```

### hy4_attn_test.cu

Absorbed-MLA attention for layer 0, head 0, causal over the four tokens
802 5466 19405 63357 at positions 0 to 3. `tools/hy4_dequant/hy4_tokenize.py`
lists these as the tokens of "The quick brown fox". The embedding rows come
from their owner rank packs.

```sh
hy4_attn_test <dir>/rank-00/<pack>.gguf
```

GPU path, compared against a float64 host reference:

- keys for every position: `attn_norm`, `attn_kv_a_mqa` gemv, `attn_kv_a_norm`
  on the 512 latent values, rope on the 64 trailing values;
- per query position: `attn_q_a`, `attn_q_a_norm`, `attn_q_b` (head 0: 192
  no-rope plus 64 rope values), absorption through `attn_k_b`, scores over
  latent plus rope parts scaled by 1/16 with the `attn_sinks` value in the
  softmax, then `attn_v_b`.

The query rope part enters unrotated in both the GPU path and the reference.
Only the 256-value attention output of each position is checked, as stages
`ATTN_T0` to `ATTN_T3` with tolerance 1e-3. The tool prints the first three
GPU values next to fixed llama.cpp eval-callback values for a visual check;
those values are not asserted. Verdict line: `ATTN PASS|FAIL`.

### hy4_dequant_test.cu

Bitwise check of the GPU dequant kernels against the CPU vendor dequant in
`hy4_iq_dequant_vendor.h`, for GGML types Q4_K (12), Q5_K (13), Q6_K (14),
IQ2_XXS (16), IQ3_XXS (18), IQ4_XS (23) and IQ1_M (29).

```sh
hy4_dequant_test <rank.gguf>
```

For each type the tool takes the first tensor of that type and its first
`min(512, total)` 256-element blocks, and compares every output float with
`memcmp`. Each GPU thread decodes one block with no reductions, so bitwise
equality is the expected result.

Output lines: `TYPE <id> blocks <n> BITWISE PASS`,
`TYPE <id> blocks <n> BITWISE FAIL first_diff elem ...`, or
`TYPE <id> ABSENT (not in this rank pack)`. An absent type does not fail the
run, so a pack with none of the types passes with 0 classes. Read, allocation
and kernel failures fail the type. Final line:
`DEQUANT_ALL PASS|FAIL (<k> classes)`.

### hy4_forward_test.cu

Full GPU forward of the five tokens 802 5466 19405 63357 299 through all 78
layers on the 16 rank packs. For the last token it then applies the
`output_hc_*` collapse, `output_norm` and `output.weight` over the 16
vocabulary slices of 7552 rows, and compares the argmax with an expected id.

```sh
hy4_forward_test <allranks_dir> <expected_top1>
```

`<allranks_dir>/rank-00` to `rank-15` are opened with `hy4_rank_open` (see
`hy4_rank_loader.h`). The embedding rows and expert slabs are read directly
from `<rank dir>/model-ud-iq1m-tp16-rank-NN.gguf`, so the GGUF must have that
name. Each layer runs the attention hc pre-mix (`hc_pre_gpu`), attention for
all 64 heads (four per rank) with per-head sinks and the sigmoid `attn_gate`,
`attn_output` summed over ranks, the hc distribute step, the FFN hc pre-mix,
then the FFN and its distribute step. Layer 0 has a dense FFN 18432 wide.
Later layers run the routed MoE plus the shared expert, with the router and
the routed SwiGLU clamp described under "Terms and shared behaviour" (gate
capped at 10 before `silu`).

Environment:

- `HY4_CKPT=<path>`: resume from the checkpoint when it exists; a corrupt
  checkpoint exits 1. After each layer the tool saves the stream and KV state
  (`<path>.tmp`, then rename). Once more than 660 s have passed since start it
  stops after the current layer with `CKPT STOP il=N` on stderr and exit 0.
  When the run completes it writes `<path>.done` and exits 0 whatever the
  verdict.
- `HY4_DUMP_PFX=<prefix>`: after each layer, write each token's stream state
  (4 x 6144 float32) to `<prefix>_L<layer>_t<token>`. This is the naming
  `tools/hy4_dequant/hy4_generate.c` uses for its dump-prefix argument, and
  `tools/hy4_gpu/anchor_compare.py <cpu_prefix> <gpu_prefix> <n_tokens>`
  compares the two sets.

`build_and_run_fwd.sh` sets both variables (under `~/hy4-anchor/gpu/`) and
passes `-1` as the expected id, so its `FORWARD` line is not a verdict.

Output: per-layer `post-attn` and `post-ffn` sums on stderr, `layer N done`
every 10 layers, `TOP1 <id> (expected <id>) val <logit>` and
`FORWARD PASS|FAIL`.

Exit codes: 2 when fewer than two arguments are given. Without `HY4_CKPT`,
0 when the top-1 id matches, 1 when it does not or on any error. With
`HY4_CKPT`, 0 on a checkpoint stop and on completion, so the verdict is only
in the `FORWARD` line; 1 on any error, including a corrupt checkpoint, a
failed checkpoint save or a `.done` marker that cannot be written.

### hy4_fp8_rung.cu

Runs `SparkHy4GemvFp8GroupedKernel` on sampled rows of a placed FP8 rank pack
and records how far each dot lies from a double-precision host dot. The
kernel is defined in
`modules/hy4_resident_decode_stage/source/spark_hy4_resident_decode_stage_cuda.cu`;
`run_fp8_rung.sh` compiles a copy of that file into the same binary.

```sh
python3 tools/hy4_gpu/fp8_rung_manifest.py <pack> <manifest> <rank> <ranks>
HY4_FP8_CKPT=<file> HY4_FP8_TSV=<file> HY4_FP8_CUTOFF=<seconds> \
  hy4_fp8_rung <pack.safetensors> <manifest> <workdir>
```

The three variables are optional.

Manifest: 13 whitespace-separated integers per plane
(`fp8_rung_manifest.py` writes one plane per line), read into `Entry` until a
group fails to parse: `il kind payload_off scale_off experts rows cols groups
rule stride row_off group_off spine`. Offsets are absolute file offsets.
`rule` is 0 ALIGNED, 1 REPLICATED_ROWS or 2 REPLICATED_GROUPS, as
`fp8_rung_manifest.py` emits it; the tool only prints it. `spine` is read and
not used. The tool rejects a plane with `stride < groups` or `stride <= 0`,
and two planes with the same `(il, kind)`.

Per plane:

- samples 8 rows: rows 0, `rows/3`, `2*rows/3` and `rows-1` of expert 0 and of
  the last expert (for `experts == 1` both halves are the same rows);
- builds the input from an LCG seeded by `(il, kind)`, values in [-1, 1);
- expected dot: `sum_g 2^(scale[g] - 127) * sum_{i<32} e4m3(w[32g+i]) *
  x[32g+i]`, with the scale byte for expert `ex`, row `r`, group `g` at
  `scale_off + (row_off + ex*rows + r) * stride + group_off + g`.

There is no tolerance. Results go to the TSV (`HY4_FP8_TSV`, default
`<workdir>/fp8_rung.tsv`, appended): il, kind, sample index, expected, GPU,
absolute delta, relative delta. Each plane prints
`FP8RUNG il=... kind=... rule=... maxabs=... maxrel=...`; an expected value of
exactly 0 prints `FP8RUNG ZERO` on stderr.

Resume:

- a finished plane writes `<workdir>/done/fp8_<il>_<kind>.done`, and planes
  with a marker are skipped; the `done` directory must exist;
- `HY4_FP8_CKPT` holds the next plane index, read at start and rewritten after
  each plane;
- `HY4_FP8_CUTOFF` (seconds, default 660): after a plane, if more time has
  passed, the tool prints `FP8_RUNG_PARTIAL ...` and exits 0. A full run ends
  with `FP8_RUNG_DONE entries=... maxabs=... maxrel=...`.

Exit codes: 0 on completion or cutoff; 2 on a CUDA error; negative codes
(exit status 256 plus the code): -1 usage, -2 manifest missing, -3 empty
manifest, -4 TSV open, -5 pack open, -6 pack read, -7 marker write,
-8 checkpoint write, -9 bad stride, -10 duplicate plane.

Known defect: `SparkHy4GemvFp8GroupedKernel` assigns one row to each 32-lane
warp (row = global thread index / 32) and reduces with a full-mask
`__shfl_down_sync`. The tool launches one block of 8 threads. Only row index 0
is computed, by lanes 0 to 7, which cover a quarter of the columns, and
`y[1]` to `y[7]` are never written. The reported deltas therefore do not
measure the kernel.

### hy4_hc_test.cu

Hyper-connection kernels for layer 0 and the output head on rank-00 weights,
compared against a float64 host reference.

```sh
hy4_hc_test <rank00.gguf>
```

The input state is four streams, each the Q4_K embedding row of token 802.
The pre and post gates follow "hc streams" under "Terms and shared
behaviour". Stages, tolerance 1e-3:

- `HC_MIXES`: RMS over the flattened 4 x 6144 vector (eps 1e-5, no weight),
  then the `blk.0.hc_attn_fn` gemv to 8 mixes;
- `HC_REDUCED`: the pre-gate-weighted sum of the streams;
- `HC_DISTRIBUTE`: `branch * post[s]` added to each stream, with a seeded
  synthetic branch vector;
- `HC_HEAD`: on the streams after `HC_DISTRIBUTE`, the same RMS, the
  `output_hc_fn` gemv to 4 mixes, gates
  `sigmoid(mix * output_hc_scale + output_hc_base) + 1e-6`, then the
  gate-weighted sum of the streams.

The tool prints fixed llama.cpp `hc_mixes-0` values next to the GPU mixes;
those are not asserted. Verdict line: `HC PASS|FAIL`.

### hy4_layer_test.cu

Token 802 at position 0 through layers 0 and 1 on the GPU: all 64 heads across
the 16 rank packs, the dense layer-0 FFN, the layer-1 routed MoE plus shared
expert, and both hc steps of each layer. The final stream state is compared
with a CPU dump.

```sh
hy4_layer_test <allranks_dir> <cpu_dump>
```

`<cpu_dump>` is raw float32, 4 x 6144 values: the layer-1 stream state of
token 0, as `tools/hy4_dequant/hy4_generate.c` writes it to `<prefix>_L1_t0`
when given a dump prefix and a prompt whose first token is 802.
`run_layer_test.sh` passes `~/hy4-cmp/l1state.t0`. The rank directories are
opened and the embedding row and expert slabs are read as in
`hy4_forward_test`.

One stage, `LAYER_STATE`, with tolerance 2e-3. `PROBE` and `GPUSUM` lines are
diagnostics. Verdict line: `QLAYER PASS|FAIL`.

### hy4_moe_test.cu

Layer-1 routed MoE and shared expert on the GPU, compared against a float64
host reference.

```sh
hy4_moe_test <dir>/rank-00/<pack>.gguf
```

- Input: a seeded vector of 6144 values in [-1, 1), not a real hidden state.
- Router: `blk.1.ffn_gate_inp` gemv on the GPU; selection and weights on the
  host as under "Router" in "Terms and shared behaviour". The tool prints
  `selected: ...`.
- Experts: slabs are dequantized on the host from the owner rank. Accepted
  expert types are IQ2_XXS, IQ3_XXS, IQ4_XS and IQ1_M; any other type exits 1.
- Routed SwiGLU: gate clamped above at 10, up clamped to [-10, 10], then
  `silu(gate) * up`.
- Shared expert: `ffn_*_shexp` read as Q6_K.

Stages `MOE_ROUTED` and `MOE_WITH_SHEXP`, tolerance 1e-3. Verdict line:
`MOE PASS|FAIL`.

The GPU shared-expert pass runs without a clamp, while the reference clamps
the shared-expert up values to [-10, 10]. `MOE_WITH_SHEXP` agrees only while
those values stay inside that range.

### hy4_qchain_test.cu

The query chain for layer 0, head 0, token 802 on rank-00 weights:
`attn_norm`, `attn_q_a` gemv, `attn_q_a_norm`, `attn_q_b` gemv (head 0, the
first 256 rows), then rope on the 64 rope values at positions 0 and 3.

```sh
hy4_qchain_test <rank00.gguf>
```

Stages and tolerances: `ATTN_NORM` 5e-4, `Q_A_GEMV` 1e-3, `Q_A_NORM` 5e-4,
`Q_B_GEMV` 1e-3, `ROPE_POS0` 5e-4, `ROPE_POS3` 5e-4. The tool prints three
fixed float64 numpy `q_pe` values next to the GPU values; those are not
asserted. Verdict line: `QCHAIN PASS|FAIL`.

This tool's `check` has no non-finite test. A NaN difference compares false
and the element passes.

## Other tools

### hy4_wo_fleet_apply.c

Writes a repair payload into byte regions of one deployed hy4 rank GGUF, then
refreshes that pack's digest records. The regions are meant to be the
`blk.N.attn_output.weight` tensors; the tool does not check tensor names.

```sh
hy4_wo_fleet_apply <rank.gguf> <blob.bin> <offsets.txt>
```

- `offsets.txt`: whitespace-separated decimal pairs `<file offset> <length>`,
  read until the input stops parsing as integer pairs.
- `blob.bin`: the new bytes of every region, concatenated in `offsets.txt`
  order. Its size must equal the sum of the lengths.
- A region whose current bytes already equal its slice of the blob is
  skipped. Every other region is overwritten in place.
- The tool then computes SHA-256 over the whole GGUF, writes
  `<rank.gguf>.sha256` as `<hex>  <basename>`, and overwrites the 64 hex
  characters after the first `"gguf_sha256": "` in `manifest.json` in the
  GGUF's directory. The key must match that spelling, with one space after
  the colon.
- The GGUF is fsynced. The sidecar and manifest are rewritten in place without
  fsync.

Output: `applied N bytes, skipped M bytes, sha256 <hex>`.

Exit codes: 0 on success; 2 on usage error; 1 on any other failure: no
regions, an offsets file, blob, GGUF or manifest that cannot be opened or
read, a blob size that does not match, a failed region read or write, a
digest pass that reads fewer bytes than the file size, a sidecar or manifest
that cannot be opened for writing, or a manifest key that is missing or
followed by fewer than 64 characters.

### mesh_register_attach_repro.c

Reproduces the resident's host registration of the weightd mesh region. It
connects to a running weightd, lazily attaches a pack, then calls
`cudaHostRegister` on the mesh send buffer that the attach mapped.

```sh
make build/mesh_register_attach_repro
SPARK_WEIGHTD_EXPERT_POOL_BYTES=<bytes> \
  build/mesh_register_attach_repro SOCKET PACK SHA256 REVISION TOPOLOGY
```

`tools/mesh_attach_repro_run.sh` runs it against the shared weightd socket.
The make target links the real CUDA runtime only when
`$(CUDA_HOME)/include/cuda_runtime.h` exists; otherwise it links
`tests/cuda_stub/cuda_runtime_stub.c` and the registrations never reach CUDA.

Arguments and environment:

- `SOCKET`: weightd socket path.
- `PACK`: a regular, non-empty file. `<PACK>.experts` must load with
  `SparkWeightdManifestLoad`.
- `SHA256`: 64 characters, the pack identity digest.
- `REVISION`: the identity revision.
- `TOPOLOGY`: 1 to 65535.
- `SPARK_WEIGHTD_EXPERT_POOL_BYTES`: required, 1 to 2^40-1. It becomes
  `expert_pool_bytes` in the request.
- `SPARK_WEIGHTD_SPINE_BUDGET_BYTES` is parsed but has no effect:
  `SparkWeightdLazyAttachRequest` has no spine-budget field.

The identity model is fixed to `glm5_next_stage`. `arena_bytes` is the pack
file size and `abi_version` is `SPARK_WEIGHTD_IPC_ABI_VERSION`.

Sequence:

1. `SparkWeightdClientConnect`, then `SparkWeightdClientAttachLazy` with a
   120 s timeout. `tools/weightd_warm.c` makes the same call. The client maps
   the daemon's mesh memfd at a 64 KiB-aligned fixed address
   (`SparkWeightdMapSharedFd`).
2. The tool requires a non-null `mesh_send_buffer_addr` and
   `mesh_send_buffer_bytes == SPARK_WEIGHTD_MESH_REGION_BYTES`, then creates
   the CUDA context.
3. It calls `cudaHostRegister` on the whole region three times: with
   `cudaHostRegisterPortable | cudaHostRegisterMapped` (the flags
   `SparkTpDeviceCollectivePrepareReceiveBf16` uses), with
   `cudaHostRegisterPortable`, and with no flags. After each success it prints
   the `cudaHostGetDevicePointer` result and unregisters.

Exit codes: 0 when all three registrations succeed; 1 on attach failure, an
unusable mesh region, or any failed registration; 2 on usage error, invalid
pack or identity, or manifest load failure.

How the transport handles the same call: `SparkTpDeviceCollectivePrepareReceiveBf16`
in `ring/transport/tp_device_collective.c` registers the region with
`cudaHostRegister(..., SPARK_WEIGHTD_MESH_REGION_BYTES, PORTABLE | MAPPED)`,
unless another collective in the process already prepared the same buffer.
If the call returns `cudaErrorInvalidValue` and the device reports pageable
memory access through host page tables, it logs `MESH-REGISTER-SKIP` and
continues unregistered. Otherwise a failure logs `MESH-REGISTER-FAIL` and
returns `SPARK_STATUS_IO_ERROR`. `TECHDEBT.md` (Mesh collectives) tracks the
open question of why CUDA refuses to host-register a shared weightd's
RDMA-registered pages.

### mesh_register_repro.c

Control probe without weightd. It creates a memfd (`memfd_create`, then
`ftruncate` to `REGION_BYTES`) and runs six attempts. Each attempt maps the
memfd `MAP_SHARED | MAP_FIXED` at a 64 KiB- or 2 MiB-aligned address, writes
the first 4096 bytes, and calls `cudaHostRegister` with
`cudaHostRegisterPortable | cudaHostRegisterMapped`,
`cudaHostRegisterPortable` or no flags.

```sh
bash tools/mesh_register_repro_run.sh
```

Run the script from the repository root: it compiles the tool with `cc`
against `/usr/local/cuda` and runs it. The tool takes no arguments.

Output: one line per attempt,
`<label> ptr=... aligned=... flags=... -> <cuda error> (<code>)`, then
`failures=N`.

Exit codes: 0 when every attempt succeeds, 1 when any fails, 2 when the memfd
setup fails.

`REGION_BYTES` is hard-coded to 134283264. That is not
`SPARK_WEIGHTD_MESH_REGION_BYTES`, which `include/sparkpipe/spark_weightd.h`
defines as 268632064, so the probe registers a smaller region than the
transport does.

### qwen38_27b_nvfp4_smoke.cu

Checks the shared `SparkLmHostLaunchBatchedLinear` NVFP4 path
(`SPARK_LM_WEIGHT_FORMAT_NVFP4_E2M1`) on a qwen38_27b FFN gate tensor,
against a host dequant of the same pack bytes. It runs two batch sizes:
4 rows, below `SPARK_LM_TILE` (16), which takes the scalar
`SparkLmLinearKernel`; and 20 rows, which takes the tile path
(`SparkLmExpertTileKernel`).

```sh
qwen38_27b_nvfp4_smoke <pack.q38sp> [layer]
```

The `layer` argument is accepted and ignored; the tool always uses layer 1.

Pack parsing:

- Header: tensor count u32 at byte 16, directory offset u64 at byte 104.
- Directory entries of 56 bytes: kind, layer, format (u32 at 0, 4, 8), rows
  u32 at 12, columns u32 at 16, payload offset and bytes (u64 at 24, 32),
  scale offset and bytes (u64 at 40, 48).
- The tool takes the first entry with kind 5 (FFN gate), layer 1 and format 8.

Reference: `W[n][k] = e2m1(nibble) * e4m3(plane[n][k/16]) * weight_global`,
with the low nibble holding the even column. The e4m3 plane is the first
`rows * cols / 16` bytes of the scale segment. `weight_global` is the F32
right after the plane. The input is a fixed pattern in [-1, 1], uploaded as
bf16; the reference dot is computed in double from the unrounded input.

An element is out of tolerance when its relative error exceeds 0.05 (the
denominator is `max(|ref|, 1e-4)`) and its absolute error exceeds 1e-2. Any
such element fails the leg. Lines: `SMOKE gemv rows4: ...`,
`SMOKE tile rows20: ...`, `SMOKE PASS|FAIL`. Apart from the usage line, the
tool's messages are written with a literal `\n` (the source escapes the
backslash), so they arrive without line breaks.

Exit codes: 0 when both legs pass; 1 when a leg fails or a launch fails; 2 on
usage error, unreadable header, directory or segment, or no matching entry.

Build with nvcc for `sm_121a`. `sparkpipe/spark_lm_kernels.cuh` is in
`model-families/common/include` and includes
`inference/kernels/activation.cuh` (relative to the repository root) and
`sparkpipe/spark_head_screen.h` (under `include/`), so the repository root,
`include/` and `model-families/common/include` must all be on the include path.

### qwen4_flash_nvfp4_smoke.cu

Checks the two NVFP4 MoE launchers the qwen4_flash module calls,
`SparkLmHostLaunchGroupedScalarLinear` and
`SparkLmHostLaunchGroupedExpertTileMloop`. It uses one layer's W1, W3 and
DOWN tensors from a wire-8 rank pack and compares against a host dequant of
the same bytes.

```sh
qwen4_flash_nvfp4_smoke <pack.q4fsp> [layer]
```

`layer` defaults to 1.

Pack parsing:

- Header: directory entry size u32 at byte 12, tensor count u32 at byte 16,
  directory offset u64 at byte 104.
- Entries: kind, layer, format, rows, columns, group (u32 at 0 to 20), payload
  offset and bytes, scale offset and bytes (u64 at 24 to 48).
- The layer must have format-8 entries of kind 6 (W1), 7 (W3) and 8 (DOWN).

Fixed assumptions: 64 expert groups, hidden size 2560, intermediate size 640,
8 token rows, SwiGLU limit 7. The tool treats each entry as 64 expert
segments. Each expert's scale segment is `scale_bytes / 64` bytes: an e4m3
plane of `rows/64 * cols/16` bytes followed by an 8-byte tail. The launchers
get scale stride `plane + 8`, and the host dequant reads the F32 weight
global at tail offset 4. All 8 rows are routed to expert group 5. The device
buffers hold the full entries, so the launchers stride experts as serving
does.

Legs:

1. `scalar W1`: grouped-scalar launcher, gate projection.
2. Decode check: `SparkLmTileDecodeRun<16>` decodes the start of row 0 of W1
   expert 5 on the device. The tool prints the first 16 device and host
   values; there is no verdict.
3. `tile W3`: tile Mloop launcher, up projection, with extra
   nearest-reference diagnostics.
4. `tile DOWN`: tile Mloop launcher. Its input is the reference activations:
   gate and up rounded to bf16, gate clamped above at 7, up clamped to
   [-7, 7], `silu(gate) * up` rounded to bf16.

A leg passes when, over all its outputs, rel L2 <= 5e-2 and cosine >= 0.999.
These are the bounds the module's CUDA validation passes to
`SparkQwen4FlashValReport`. The per-element count (relative error > 0.02 and
absolute error > 1e-3) is printed but does not decide the verdict, for any
leg. Lines: `SMOKE <leg>: ... outside tolerance ...`,
`SMOKE <leg> aggregate: rel_l2=... cosine=...`, then `aggregate PASS` or
`FAILS the module aggregate contract`, and finally `SMOKE PASS|FAIL`.

Exit codes: 0 when all legs pass; 1 when a leg fails or a launch fails; 2 on
usage error, unreadable header, directory or segment, or missing entries.

Build: the same include requirements as `qwen38_27b_nvfp4_smoke.cu`, for
`sm_121a`.
