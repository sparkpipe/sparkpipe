# hy4 FP8 scale-row-offset contract

Normative for every F8_E4M3 payload + U8 E8M0 scale pair in a
`hy4-fp8-tp16-v1` rank pack. Code form:
`modules/hy4_resident_decode_stage/source/spark_hy4_fp8_scale_contract.h`
(`SparkHy4Fp8ScaleContractValidate`, consumed through the module's
format-check surface). Oracle form: `tools/hy4_fp8_scale_contract.py`
(own math, no driver imports). Agreement test:
`tests/test_hy4_fp8_scale_contract.py` (72 C/python cases, 7 fail-closed
codes, CLI fail-closed with the plane name).

## Pins

| what | pin |
|---|---|
| publisher reference (head-axis layout) | `tools/hy4_dequant/vendor/hyv4_reference.cpp` = llama.cpp `src/models/hyv4.cpp` @ `0cea36222` (AngelSlim hy4-preview patch 0001), sha256 `514ef62ae147171d5675229de59e8e6d3b8e1f84f20680056df8708601fe52fd` |
| FP8 checkpoint | `tencent/Hy4-preview-FP8` @ `4215ec29de873a998e849cee902654490c7ff4d1`, warm `hy4-preview-fp8-official`; `quantization_config.quant_algo = MXFP8` (modelopt) |
| placed pack used for the census | spark2 `~/sparkdata/hy4.fp8.tp16/packs/rank-02/model-fp8-tp16-rank-02.safetensors`, file sha256 `32e87405199cc2ed3e32d323fe281356 9fe9947ee8d2b594d438455bf66d69d4` (matches manifest and the rung receipt) |
| geometry | `model_contracts/hy4_authoritative.json`: hidden 6144, heads 64, qk 256 = nope 192 + rope 64, v 256, kv_lora 512, q_lora 2048, index heads 32 x 128, experts 256 |

## Derivation

MXFP8 (modelopt) stores one E8M0 scale per 32 consecutive INPUT columns of
one OUTPUT row: scale plane `[rows, columns/32]`, byte `(i, g)` scales
payload `w[i, 32g .. 32g+31]`. The rung-2 kernel
(`SparkHy4GemvFp8GroupedKernel`) indexes exactly this: row stride
`columns/32`, group `g`.

Head axes from the reference `create_tensor` lines (GGUF `ne[0]` is the
INPUT axis, so the HF `[out, in]` matrix is the transpose):

| reference tensor (line) | GGUF ne | HF weight `[out, in]` | head axis |
|---|---|---|---|
| `wq_b` (106) `{q_lora_rank, n_head * n_embd_head_k_mla}` | in=2048, out=64x256 | `q_b_proj [16384, 2048]` | rows, head-major `h*256 + d` (view_3d lines 293-296) |
| `wk_b`/`wv_b` (109/110) `{nope, kv_lora, n_head}` / `{kv_lora, v, n_head}` | per-head `[192+256, 512]` | `kv_b_proj [28672, 512]` | rows, head-major `h*448 + d` |
| `wo` (111) `{n_head * n_embd_head_v_mla, n_embd}` | in=64x256, out=6144 | `o_proj [6144, 16384]` | COLUMNS, head-major `h*256 + d` |
| `indexer_attn_q_b` (119) `{q_lora_rank, n_indexer_head * n_embd_indexer}` | in=2048, out=32x128 | `indexer.wq_b [4096, 2048]` | rows, head-major `h*128 + d` |

TP rank `r` of `N` owns heads `[r*64/N, (r+1)*64/N)` (index heads
`[r*32/N, ...)`), a CONTIGUOUS global row range for `q_b/kv_b/wq_b` and a
contiguous global column range for `o_proj`. Head boundaries are multiples
of 32, so no scale group straddles a rank.

The placed packs carry the FULL checkpoint scale plane for these four kinds
(the packer replicated every `.weight_scale` companion of a split weight;
now declared explicitly as `SCALE_REPLICATED` in `tools/hy4_fp8_stagepack.py`,
0 rule diffs over the 2838 placed names). Every other FP8 pair is ALIGNED:
scale plane `[local_rows, local_columns/32]`.

## Rules (N ranks, rank r; TP16 values shown)

Byte index of the scale for local payload row `i`, local group `g`:

    idx(i, g) = (rank_row_offset + i) * scale_groups + rank_group_offset + g

| kind (HF suffix) | stagepack kind | rule | global `[rows, cols]` | local payload (N=16) | scale plane as placed | `rank_row_offset` | `rank_group_offset` | `scale_groups` (stride) | kernel |
|---|---|---|---|---|---|---|---|---|---|
| `self_attn.q_b_proj.weight` | ATTN_Q_B | REPLICATED_ROWS | `[16384, 2048]` | `[1024, 2048]` | `[16384, 64]` | `r*1024` | 0 | 64 | zero-copy pointer offset `r*1024*64` |
| `self_attn.kv_b_proj.weight` | ATTN_K_B | REPLICATED_ROWS | `[28672, 512]` | `[1792, 512]` | `[28672, 16]` | `r*1792` | 0 | 16 | zero-copy pointer offset `r*1792*16` |
| `self_attn.indexer.wq_b.weight` | INDEX_WQ_B | REPLICATED_ROWS | `[4096, 2048]` | `[256, 2048]` | `[4096, 64]` | `r*256` | 0 | 64 | zero-copy pointer offset `r*256*64` |
| `self_attn.o_proj.weight` | ATTN_OUTPUT | REPLICATED_GROUPS | `[6144, 16384]` | `[6144, 1024]` | `[6144, 512]` | 0 | `r*32` | 512 | stride != local groups: materialize `[6144, 32]` at attach (strided copy) or stride-aware kernel |
| `self_attn.q_a_proj.weight` | ATTN_Q_A | ALIGNED | `[2048, 6144]` replicated | same | `[2048, 192]` | 0 | 0 | 192 | zero-copy |
| `self_attn.kv_a_proj_with_mqa.weight` | ATTN_KV_A | ALIGNED | `[576, 6144]` replicated | same | `[576, 192]` | 0 | 0 | 192 | zero-copy |
| `self_attn.indexer.wk.weight` | INDEX_WK | ALIGNED | `[128, 6144]` replicated | same | `[128, 192]` | 0 | 0 | 192 | zero-copy |
| `mlp.experts.gate_up_proj` | MOE_W1 | ALIGNED | `[256, 4096, 6144]` dim0 | `[16, 4096, 6144]` | `[16, 4096, 192]` | 0 | 0 | 192 | zero-copy (3-D flattened to rows) |
| `mlp.experts.down_proj` | MOE_DOWN | ALIGNED | `[256, 6144, 2048]` dim0 | `[16, 6144, 2048]` | `[16, 6144, 64]` | 0 | 0 | 64 | zero-copy |
| `mlp.shared_experts.{gate,up}_proj.weight` | MOE_SHARED_GATE/UP | ALIGNED | `[2048, 6144]` replicated | same | `[2048, 192]` | 0 | 0 | 192 | zero-copy |
| `mlp.shared_experts.down_proj.weight` | MOE_SHARED_DOWN | ALIGNED | `[6144, 2048]` replicated | same | `[6144, 64]` | 0 | 0 | 64 | zero-copy |
| `mlp.{gate,up}_proj.weight` (dense layer 0) | MOE_SHARED_GATE/UP | ALIGNED | `[18432, 6144]` replicated | same | `[18432, 192]` | 0 | 0 | 192 | zero-copy |
| `mlp.down_proj.weight` (dense layer 0) | MOE_SHARED_DOWN | ALIGNED | `[6144, 18432]` replicated | same | `[6144, 576]` | 0 | 0 | 576 | zero-copy |

`model.mtp_layers.0.*` carries the same kinds with identical geometry and
rules. Any FP8 plane outside this table, any missing U8 companion, any
payload that does not tile the rule, or any scale plane whose shape is not
the rule's plane is a contract failure: `SparkHy4Fp8ScaleContractValidate`
returns a unique negative (-1 no rule for kind, -2 rank, -3 payload
geometry, -4 scale shape, -5 null out, -6 rows do not tile, -7 columns do
not tile) and the attach path must `SPARK_FAIL` naming the plane; the
oracle prints `SCALE-CONTRACT FAIL <name>` and exits 1.

## Validation status

Shape-level (this contract, rank-02 placed header, `tools/hy4_fp8_scale_contract.py`):

    rank 2 of 16: 832 FP8 planes, 0 contract failures
      ALIGNED            EXPERT  156
      ALIGNED            SPINE   417
      REPLICATED_GROUPS  SPINE    79
      REPLICATED_ROWS    SPINE   180

Per kind (layers + MTP): q_b 78+1, kv_b 78+1, o_proj 78+1, indexer.wq_b
21+1 (= the 259); q_a 78+1, kv_a 78+1, wk 21+1, experts gate_up 77+1,
experts down 77+1, shared gate/up/down 77+1 each, dense L0 gate/up/down 1
each (= the 573). Rank-2 offsets: q_b row 2048, kv_b row 3584, wq_b row
512, o_proj group 64 at stride 512.

Byte-level (production grouped dot vs CPU double over the 259 planes through
`idx(i, g)`): PENDING — the rung-3 cell (`tools/hy4_gpu/hy4_fp8_rung.cu`
extended with the contract's row/group offsets; queue v2 spark2 gpu ttl-15,
chunked foreground, done markers). Not run: fleet suspended before the cell
could be queued.
