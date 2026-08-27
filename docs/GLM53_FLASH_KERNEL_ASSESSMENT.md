# GLM 5.3 Flash — kernel assessment (2026-08-27)

Verdict up front: **GLM 5.3 Flash is an assembly of three mechanisms we
already ship, plus three small deltas.** The working model is
"GLM 5.2 skeleton + DSV4-0731's indexer/hyper-connections + K3's KDA."
No from-scratch kernel family is required. Estimated new code: ~800-1,200
lines (module glue + rope-dim-0 MLA variant + name mapping), against
~90% reuse from existing family modules.

All numbers below are from the warm source at
`/mnt/model-warm/glm-5.3-flash` (config.json + safetensors index; 62
shards, 306 GiB, `Glm5NextForConditionalGeneration`, FP8 e4m3 dynamic
[128,128] blocks) — not from assumptions.

## Architecture (measured from config + tensor index)

- 45 text layers: **34 KDA linear-attention + 11 deepseek-sparse-attention**
  (DSA at layers 3, 7, 11, ..., 43 — every 4th after a 3-KDA head).
- First 3 layers dense MLP (intermediate 12288); 42 MoE layers +
  MTP layer 45: **288 routed + 1 shared, top-8**, moe_int 2048, sigmoid
  scoring, `noaux_tc`, routed_scaling 2.5.
- MLA on DSA layers: kv_lora 512, q_lora 1536, qk_nope 256,
  **qk_rope 0** (rope lives in the indexer), v_head 256, 64 heads.
- Indexer: 32 heads x 128, **topk 2048**, kpool 4 with compression
  (`index_kpool_compress_{ape,gate}`, `wk`, `weights_proj`),
  rope-interleaved, shared across MTP iterations.
- KDA: 64 heads x 128, short conv 4, gate lower bound -5.0
  (K3's `kda_layers` vocabulary, no `use_full_rank_gate`).
- Hyper-connections: `hc_mult 4`, sinkhorn 20 iters, eps 1e-6, per-layer
  `hc_{attn,ffn}_{base,fn,scale}` tensors — **tensor names byte-identical
  to DSV4-0731** (`mhc: true` flag aside, same weights).
- MTP: 1 nextn layer (`eh_proj`/`enorm`/`hnorm` — DeepSeek MTP names).
- max_position 1,048,576 (1M). Vocab 154,880. VLM vision tower present —
  OUT OF SCOPE for serving lane 1 (text stack only, same call as Qwen Flash).
- BF16 keeps: router + e_score_correction_bias, all hc_*, A_log, dt_bias,
  norms, embed/lm_head, indexer `weights_proj`, vision.

Active params ≈ 11.5-12B/token (dense 3 + 9/288-of-7.25B x 42 MoE +
attn + embed) — DSV4-Flash active class, at 2x total weights.

## Component → donor mapping (tensor-name evidence)

| Mechanism | GLM 5.3 Flash tensors | In-tree donor | Delta |
|---|---|---|---|
| MoE router + experts | `mlp.gate(.e_score_correction_bias)`, `experts.N.{gate,up,down}_proj.weight{,_scale_inv}`, shared_experts | `glm52_resident_decode_stage` (sigmoid/noaux_tc/2.5/first-3-dense all identical config) | expert count 288 vs 256 — a constant, kernels are expert-generic |
| MLA projections | `q_a_proj/q_a_layernorm/q_b_proj`, `kv_a_proj_with_mqa/kv_a_layernorm/kv_b_proj` | glm52 (identical tensor names, kv_lora 512 identical) | q_lora 1536 (5.2: 2048), nope 256 (5.2: 192) — widths; **rope 0 (5.2: 64) — see deltas** |
| DSA indexer | `indexer.wq_b/wk/weights_proj/k_norm` | dsv4 (DSV4-0731: same names) | 32 heads vs dsv4 64; topk 2048 (glm52's own indexer count) |
| Indexer kpool compressor | `index_kpool_compress_{ape,gate}` | dsv4-0731 `indexer.compressor.{wkv,wgate,ape,norm}` — **already in the -0731 checkpoint** and the dsv4 lane is building that path now; cache arena already has `compressor_state_arena` | rename mapping only |
| KDA layer | `q/k/v_proj`, `q/k/v_conv1d`, `A_log`, `dt_bias`, `f_a/f_b/g_a/g_b_proj`, `b_proj`, `o_norm`, `o_proj` | k3 (`kda_qkv_beta`, `kda_decay_{down,up}`, `kda_{q,k,v}_conv`, `kda_head_log_scale`, `kda_gate`, `kda_out_norm`, ...) | heads 64 vs 96 (width); K3 sets `use_full_rank_gate=true`, glm53 omits it — verify default before wiring the gate kernel |
| Hyper-connections | `hc_{attn,ffn}_{base,fn,scale}` per layer | dsv4-0731 module (15 HC references) — identical names and params (hc_mult 4, sinkhorn 20) | none observed; confirm `mhc:true` semantics during cosine bring-up |
| MTP layer | `layers.45.{eh_proj,enorm,hnorm}` + MoE | dsv4 spec path + glm52 MTP | none — standard DeepSeek MTP |
| FP8 [128,128] block quant | `weight_scale_inv` per tensor | glm52 quant path (identical scheme) | none |

KV/geometry: KDA state = 64 heads x 128x128 fp32 ≈ 4 MB/layer, ~140 MB
total — trivial. DSA KV = kv_lora 512 x seqlen (MLA compressed, cannot
head-shard): at 1M context ≈ 512 MB/layer x 11 = 5.6 GB replicated per
rank; attention compute reads topk-2048 only. Long-context serving is the
design point of this architecture — 34/45 layers are O(1) memory.

## The three deltas (all small)

1. **MLA with `qk_rope_head_dim = 0`** (`mla_use_nope: true`). All our MLA
   kernels carry a 64-dim rope term alongside nope. With rope 0 the scoring
   is pure absorbed-MLA — strictly simpler math, but if the kernel hardcodes
   rope 64 it needs a templated variant (drop the rope qk term, fold the
   layout). Verify: grep glm52 MLA kernel for rope-dim constant; add a
   compile-time `MLA_ROPE_DIM=0` instantiation. ~50-150 lines.
   [CONFIRMED 2026-08-27 by the glm53 lane against transformers'
   modeling_glm5_next.py: the text stack is NoPE EVERYWHERE — the indexer
   carries no rope either (the config's `indexer_rope_interleave` does not
   introduce positional scoring), and the KDA gate is the low-rank g_a/g_b
   + sigmoid-safe form. So delta 1 widens slightly but simplifies: strip
   rope from BOTH the MLA scoring and the dsv4-donor indexer when porting;
   `indexer_rope_interleave` is not a porting dependency.]
2. **Checkpoint→pack name mapping.** The K3 module's packed field names
   (`kda_*`) differ from glm53 checkpoint names (`A_log`, `dt_bias`,
   `f_a_proj`, ...). This is packer mapping, not kernels: a table in the
   pack tool + geometry header. The DSV4 compressor names also differ
   (`compressor.wkv` vs `wk` + `index_kpool_compress_*`). ~200-300 lines
   of mapping + tests.
3. **Hybrid layer dispatch + geometry.** 45 layers with two attention kinds
   on a 3+42x(MoE) + MTP skeleton, per-layer HC everywhere. The k3 module
   already dispatches kda/full per `linear_attn_config.kda_layers`; port
   that dispatch table pattern. ~300-500 lines of module glue (new family
   `glm5_next`, assembled from glm52 base + k3 KDA + dsv4 indexer/HC).

Everything else is constants or width parameters.

## Topology and pack budget (all-16 policy)

306 GiB FP8 source. At the fleet standard 16-rank sharding (TP16 where
heads divide cleanly — 64 KDA heads, 32 indexer heads do; or TP4xPP4):

| Topology | Per-rank pack | Notes |
|---|---|---|
| TP16 | ~19.1 GB | KV replicated on DSA layers (no head shard for lora KV) |
| TP4xPP4 | ~19.1 GB | 12 layers/rank incl. 3 DSA; better KV story at 1M ctx |

Both fit trivially; start TP16 (simplest, 1M-context KV replication is
5.6 GB), keep TP4xPP4 as the hill-climb alternate. All-16 pack budget
for this model: ~19 GB on every node (see AGENT_LANE_BRIEFS/README.md
fleet table).

## Performance expectation (honest, pre-measurement)

Decode weight traffic = active ~12 GB/token full-model. TP16: ~0.75 GB
per rank per token → ~330 tok/s memory ceiling single-stream. Kernel/launch
overhead at 45 layers (34 cheap KDA + 11 sparse MLA) lands this in the
DSV4-Flash class: **~35-45 tok/s no-spec, 80-100 with MTP-1**. The 1M
context window is the differentiator no other fleet model has.

## Recommended execution order

1. Lane starts now on non-blocking work: contract freeze (sha256 the 62
   shards), geometry header, name-mapping table, synthesized pack.
2. Copy KDA kernels from k3 module + indexer/HC from dsv4 module into the
   new `glm5_next` family; base module skeleton from glm52.
3. The rope-0 MLA variant is the only kernel edit — do it early behind a
   cosine gate vs the CPU oracle.
4. Packs after the glm52 lane's vehicle validates (it owns the shared
   MLA/MoE base correctness); inherit its nodes when it exits.
5. Emit 16 rank packs + deploy to all 16 sparks per the fleet policy.
