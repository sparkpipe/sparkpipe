# hy4 references on the publisher MXFP8 checkpoint

These tools produce ground truth for the hy4 (Hunyuan 4 preview, `HYV4ForCausalLM`) driver. They read `tencent/Hy4-preview-FP8` @4215ec29 directly, not the rank packs.

| tool | what it is |
|---|---|
| `hy4_fp8_reference.c` | Independent CPU reference in C with OpenMP. Activations and accumulation are fp32. Weights are dequantized exactly: E4M3 x E8M0 per 32 columns, and BF16 planes widened. |
| `hy4_gpu_stream_forward.cu` | Single-GPU streaming forward. Every projection goes through the shared `SparkLmLinearKernel<32>` (`SPARK_LM_WEIGHT_FORMAT_FP8_E4M3`, or BF16 for `linear_gate`). Inputs between linears are BF16, streams are fp32. Weights are streamed from the checkpoint one layer at a time, and only the routed experts each step selects are loaded. |
| `hy4_fp8_reference.py` | `manifest` builds the flat tensor manifest both binaries read. `prompts` flattens `prompts.json`. `fixtures` turns a CPU run into T1R1 fixtures plus `MANIFEST.json`. |
| `hy4_compare_runs.py` | Compares two runs of either binary. It checks streams at the capture layers (max relative error and cosine), routed expert sets and weights, greedy tokens, and top-2 head margins. |

Both binaries batch every prompt position of every prompt into one prefill pass. Each decode step then runs one row per prompt. Output files per prompt:
- `<name>.txt`: prompt ids, generated ids, top-5 head per position
- `<name>.routes.bin`: int32 ids, then f32 weights, `[position][layer 1..77][8]`
- `<name>.streams.bin`: f32 `[capture layer][position][4 x 6144]`

## Forward semantics

Checked against the transformers `hy_v4` modeling code and the checkpoint shapes.

- **Hyper-connections.** There are 4 streams, and the embedding is copied into all four.
  - The mixes are `hc_fn · flat(streams)`, scaled by `rsqrt(mean(flat²) + rms_norm_eps)`. The epsilon is 1e-5; `hc_eps` is not used there.
  - `pre = sigmoid(mix[0:4]·scale[0] + base[0:4]) + hc_eps`
  - `post = 2·sigmoid(mix[4:8]·scale[1] + base[4:8]) + hc_eps`
  - The branch input is `Σ pre_m·stream_m`, followed by the layer RMSNorm.
  - The update is `stream_m += post_m · branch_out`.
  - The head uses `hc_head_fn` [4, 24576] and one scale, then `model.norm`, then `lm_head` (BF16 weights, computed in fp32).
- **Attention.** Gated MLA.
  - `gate = sigmoid(linear_gate · x)` is elementwise over 64 x 256 values, and is applied to the per-head attention output before `o_proj`.
  - `q = q_b(rmsnorm(q_a x))` is split per head as 192 nope + 64 rope.
  - `kv_a` gives a 512 latent, RMS-normalized, and 64 rope values.
  - RoPE is half-split (`rotate_half`), with theta 1e7 over 64 dimensions.
  - `kv_b` rows per head are 192 k-nope followed by 256 v.
  - The scale is 1/16.
  - `learnable_sink_param` holds one logit per head, per layer, and joins the softmax denominator.
- **DSA indexer.** Top-k is 2048 over causal keys. Below 2048 context tokens the selection is every causal key, so both references omit the indexer and refuse positions ≥ 2048. Layers 0, 1, 5, 9, ..., 77 own an indexer; the others reuse the previous one.
- **Router.** `sigmoid(gate · x)` in fp32. Experts are selected by `score + e_score_correction_bias`. The weights are the unbiased scores for the top 8, divided by their sum plus 1e-20, times 2.827.
- **Experts.** Routed experts compute `silu(min(g, 10)) · clamp(u, ±10)`, with `gate_up_proj` rows [0, 2048) = g and [2048, 4096) = u. The shared expert and the layer-0 dense FFN are not clamped.

## Running

```
python3 tools/hy4_reference/hy4_fp8_reference.py manifest --checkpoint /mnt/model-warm/hy4-preview-fp8-official --output run/manifest.txt
python3 tools/hy4_reference/hy4_fp8_reference.py prompts --prompts qualification/t1_reference/hy4/prompts.json --output run/prompts.txt
gcc -O3 -mcpu=native -fopenmp -o run/hy4_fp8_reference tools/hy4_reference/hy4_fp8_reference.c -lm
OMP_NUM_THREADS=24 run/hy4_fp8_reference run/manifest.txt run/prompts.txt run/out
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -gencode arch=compute_121a,code=sm_121a -Xcompiler -fopenmp -I. -Iinclude -Imodel-families/common/include -o run/hy4_gpu_stream_forward tools/hy4_reference/hy4_gpu_stream_forward.cu -lgomp
OMP_NUM_THREADS=16 run/hy4_gpu_stream_forward run/manifest.txt run/prompts.txt run/gpu 78 16
python3 tools/hy4_reference/hy4_compare_runs.py --prompts qualification/t1_reference/hy4/prompts.json --reference run/out --candidate run/gpu
```

On a GB10 with the checkpoint on Ceph:
- The CPU prefill (28 rows) takes about 7 minutes, and each decode step about 2.5 minutes.
- The GPU run is limited by reads.

These times say nothing about serving speed.
