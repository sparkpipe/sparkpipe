# Hub: drafters and draft farm

The hub (`rtx5090`, x86_64) hosts the model APIs, the draft farm and the
drafter store. It is not a Spark: it has no rank and takes no part in the TP
collectives. Its rules, roles and network are in the datacenter repository
(`datacenter:operations/HUB.md`); hub operations (publishing roots,
heartbeats) follow the fleet release runbook
(`datacenter:operations/FLEET_RELEASE_RUNBOOK.md`); speculation design is in
[SPECULATION_UNIFIED_DESIGN.md](SPECULATION_UNIFIED_DESIGN.md).

## K3 block drafter (production since 2026-10-17)

K3 serves model drafts from the RadixArk DSpark block drafter on the hub.
- **Unit:** the transient user unit `sparkpipe-draftd`, started from the
  lane record's `draftd` entry (`deployment/lanes.json`) by
  `tools/lane_config.py draftd kimi-k3`, which runs `tools/hub_draftd.sh`
  on the hub with the record's unit, source directory, drafter, target
  shard, tap and window. The source directory
  (`/srv/workspace/sparkpipe/draftd-<sha>`) holds `block_draftd.py` and
  `spec_offline/` from that commit; the weights are on the hub's NVMe under
  `/srv/workspace/sparkpipe/drafter-weights`. `hub_draftd.sh` refuses a
  path off `/srv/workspace` or `/home`, since inference never reads CephFS
  or cold storage, and `tools/lane_config.py audit` reports a drafter unit
  that is down or runs other flags. `tools/hub_draftd.sh status` shows the
  unit, its two ports and the last timing and tap lines; `stop` stops it.
- **Restart after a hub reboot:** like every hub unit, it does not survive a
  reboot. Start it again with `tools/lane_config.py draftd kimi-k3`. Until then, K3
  serves without model drafts: the tap rank retries every 5 s, and each draft
  request answers unavailable.
- **Inputs:** the K3 tap rank (sparkf, rank 15) streams the drafter's five
  target layers (`taps` in the K3 lane's `config/adapter.json`) to
  `10.10.250.2:7701`, the hub's 10 GbE link to sparkf (`eno1`). The 100G
  link it used until 2026-10-23 no longer reaches a Spark.
- **Clients:** the K3 API (`~/k3-kv-api`, `draft_server` in its
  `model_resident.json`) asks `127.0.0.1:7702` over DFT3. The API's engine
  asks without waiting on the socket: it sends the request when a decode
  frame's plan wants a model draft (one or two decoding lanes, no lookup
  match of 8 or more, the request not backed off), leaves that frame
  unbuilt, and plans it again when the answer is in. One request is
  outstanding at a time, so two lanes are asked one after the other. While
  the frame waits, the engine still takes completions, admits and cancels
  requests and dispatches prefill, release and publication work. The ask
  ends with a failure after the lane's `wait_ms` plus 100 ms (200 ms more
  when the connection has to be opened first); the frame then decodes
  without model drafts and the request backs off 8 tokens, doubling. A
  request that is cancelled or fails while it waits has its ask dropped with
  the connection, and the next ask reconnects. `batch_step` reports each
  frame's wait as `draft_wait_ms`; `batch_speculation` reports the mean as
  `wait_ms` and the dropped asks as `abandoned`.
- **Log:** `/srv/workspace/sparkpipe/logs/draftd.log`. It prints
  `DRAFTD-TIMING` every 32 requests (queue, wait, compute, apply and total
  ms) and `DRAFTD-STATS` every 256 requests.
- **Context:** each sequence keeps the last `--window` positions (4096 by
  default, `DRAFTD_WINDOW`) in a ring of 82 MB, whatever the context
  length, and a draft attends to that window. On a recorded 137K prose reply
  the 4K window keeps 2.17 tokens per round against 2.05 with the whole
  context (paced replay, 400 rounds). The windows offline (`long_drafter.py`)
  score 2.24 (4K), 2.20 (16K), 2.15 (64K) and 2.12 (all) on prose at 137K,
  and 4.97, 5.05 and 4.82 on a code edit at 73K. A follow-up turn copies the
  window's positions from the previous turn's ring when the tokens match.
- **GPU:** about 10.6 GB plus 82 MB per held sequence. `--context-bytes`
  (8 GiB, `DRAFTD_CONTEXT_BYTES`) caps the held sequences; the least
  recently used goes first. The GPU is shared with the chess tenant (about
  5.2 GB): never start a second drafter beside the production one (a
  full-context test drafter ran the card out of memory on 2026-10-17).
- **Latency:** 9-12 ms per request; the draft itself is 6-7 ms.
- **Recording:** `--record-dir DIR` writes every tap connection to an
  `.spt2` file for offline scoring (`tools/spec_offline`). Use it on hold
  corpora only, never on users' traffic: a recording holds the token ids,
  from which the text can be rebuilt. No production swap entry sets
  `DRAFTD_RECORD_DIR` (`tests/test_model_swap_hub_services.py`); run a
  recording as a separate drafter on a hold corpus and delete it once
  scored.

## Latent block drafter (lightseek DFlash2, offline)

`tools/spec_offline/latent_block_drafter.py` runs checkpoints whose
`dflash_config.attention_mode` is `mla`: lightseekorg/kimi-k3-dflash2 at
e77935fb (5 layers, 3.69B parameters, received 2026-10-18 into
`/mnt/cold-raid6/models/kimi-k3-dflash2-lightseek-e77935fb`, sha256
1f2a5376...52c2, receipt beside it; the serving copy is
`/srv/workspace/sparkpipe/drafter-weights/kimi-k3-dflash2-lightseek-e77935fb`). `open_drafter` in `block_drafter.py`
picks it by that field, so `block_drafter.py` and `long_drafter.py` score it
like the RadixArk drafter, and `block_draftd.py` serves it (DRAFTD-READY says
`drafter latent`); it has not run against live taps yet.
- **draftd context:** per sequence, one 4,096-row ring shared by the sliding
  layers and 65,536-position pages for the full layer (75.5 MB each, 1,152 B
  per position), allocated as the sequence grows. A stream that starts after
  position 0 copies the longest prefix whose tokens match another sequence's;
  token checks only cover positions past a verified watermark. A draft
  attends to what is held and counts a gap otherwise. It is eager (no CUDA
  graph yet) and caps depth at the drafter's 7 drafts.

- **Inputs:** the K3 OUTPUT taps o19, o37, o66, o78 and o90 (the AttnRes
  partial after each layer, which is TokenSpeed's default `prefix` stream).
  Production taps (o7, o23, o51, o67, o83) are the RadixArk drafter's, so
  scoring it needs a dump recorded with the union of both sets on all 16
  ranks.
- **Context:** `c = rms(fc taps)`. Each layer keeps one 576-wide row per
  position, `[rms(kv_a c)[:512] | rope(kv_a c)[512:]]`, with no input norm
  and no convolution. Layers 0-3 see the 4,088 positions before the block
  (a 4,096 window measured from the block end); layer 4 sees all of them.
- **Block:** 8 rows `[t_P, mask x 7]` at positions P..P+7, non-causal. Each
  layer: `x = rms(h)`, coefficients `d = kernel_projection x` split into
  side, tap and 448 groups of 16 channels; `x' = conv(x, base[0], d[:, 0])`
  with `conv(z)[i, c] = (b0[c] + d0[i, c/16]) z[i, c] + [i >= 1] (b1[c] +
  d1[i, c/16]) z[i-1, c]`, so row 0 never reads an earlier block;
  absorbed MLA on `x'` over the context rows plus the block's own; `h +=
  conv(attention, base[1], d[:, 1])`; the same pair wraps the gated MLP on
  `rms(h)`. RoPE is DeepSeek YaRN on interleaved pairs (64 dims, theta
  50,000, factor 32, original 32,768) with a cos/sin gain of 1; the squared
  YaRN mscale goes into the softmax scale (0.130861).
- **Drafts:** rows 1..7 of `rms(h)` through the target head, the top 16 per
  row, and edge scores `unary[c] + sum_r A[previous, r] g[r] B[c, r]` with
  `g = hidden_projection y`. Drafts walk the lattice greedily from the
  anchor; a sampled stream takes the keyed-noise choice over the same row.
  The score of each draft is the lattice probability of its candidate, with
  the non-candidate tail of the full head in the normalizer.
- **Refused:** any other architecture or attention mode, a non-prefix tap
  stream, `fc_norm`, an output gate, embedding or output scales other than
  1, logit softcapping, a causal block, and layer types that do not cover
  every layer.
- **Checks:** `tests/test_latent_block_drafter.py` holds the block to a
  float64 reference with expanded (non-absorbed) attention and half-split
  RoPE, and checks the convolution, the window edge, chunked attention, the
  walk and the keyed sample on a tiny random checkpoint.

## DSpine drafter (Tencent, Qwen3.8-27B, offline)

`tools/spec_offline/dspine_drafter.py` runs checkpoints whose `dflash_config`
has `selected_prefix_code_bridge` (tencent/Qwen3.8-27B-DSpine-b16, Apache-2.0,
NOTICE): a 5-layer DFlash backbone with adjacent causal injection, block 16,
15 drafts per round. `open_drafter` picks it before every other class, so
`block_drafter.py` and `sampled_acceptance.py` score it and `block_draftd.py`
serves it through the unchanged `DraftGraph` and `Context`.
- **Inputs:** taps `o1,o16,o31,o46,o61` (the checkpoint's `target_layer_ids`
  as 0-based layer outputs, no offset). The codebook (`[248320, 1024]`)
  ships in the checkpoint; the greedy refinement tables (0.76 GB) are built
  from the target LM head at load.
- **Block:** rows `[t_A, mask x 15]` at A..A+15, causal inside the block, a
  sliding window of 8,192 per row; the residual is BF16 at every add. After
  every layer the code bridge writes `h = u + rms(u) W_l (g * m)` from the
  predicted code of the row before (the anchor's code is its codebook row).
- **Drafts:** the 16 best head logits per row, ascending by id; the last write
  redone with each candidate's exact code as sender (`serial_table`), walked
  greedily from the anchor; a sampled stream walks the FP32 refined table with
  the verifier's keyed noise. The score is the refined choice's softmax mass.
- **draftd:** `--window 8176` (8,192 minus the block) keeps every row inside
  its own window, since draftd keeps one context per sequence. The graph path
  re-ranks the 64 best of its own last-norm logits with the bridge's normed
  rows, so it picks the offline path's candidates.
- **Checks:** `tests/test_dspine_drafter.py` holds the forward to float64
  references (backbone with the writes zeroed, full DSpine with them on), the
  greedy chain to a brute-force decode that redoes the last write per row, the
  FP32 sampled table to float64, and the graph segments to the offline path.
- **Measured (2026-11-08, lenovo8, torch target reference):** GSM8K first 100,
  xhigh, decode rounds (`--replay`), per-request mean: 6.99 at T=0 (card 7.20;
  DFlash2 5.60, card 5.69) and 6.56 with rejection verify at T=1 (card 6.31;
  DFlash2 5.39, card 5.45). Whole-file rewrites and code edits 9-15.7 tokens
  per round against 5.4-8.0 for DFlash2 and the RadixArk DSpark; prose 2.1-2.2
  against DFlash2's 2.4-2.6. Tap offset sweep peaks at 0 for both drafters.
  A draftd graph round (15 drafts) takes 31.0 ms on an idle GB10 at 8,176
  context positions; its first draft matches the offline path on 233 of 233
  anchors and the whole chain on 93%.

`tools/spec_offline/selector_block_drafter.py` runs DFlash2 checkpoints with
grouped-query attention (incoai Qwen3.8-27B DFlash2): the block convolutions
and the transition selector of the latent drafter around GQA attention,
sliding window 2,048, 7 drafts per round. `tools/spec_offline/qwen38_27b_target.py`
is a torch reference of the Qwen3.8-27B decoder (chunked GatedDeltaNet prefill,
batched decode, greedy or keyed sampling) that records these tap streams on a
GB10 and writes the `--target-shard` (embed, head, final norm with its 1 + w
folded in).

## Drafter packs

`tools/drafter_pack.py` cuts a drafter and the target tensors it reads into one
named pack (`tools/spark_named_pack.py`, magic `DRPK`, 128-byte alignment in its
format, `tp_degree` 1 and `tp_rank` 0 in its config) for the engine drafters
that replace `block_draftd.py`:

```sh
python3 tools/drafter_pack.py --drafter k31dspark \
    --checkpoint /srv/workspace/sparkpipe/drafter-weights/kimi-k3.1-pro-dspark-preview-6a2ef20a05ef \
    --target /srv/workspace/sparkpipe/drafter-weights/kimi-k3.1-pro/model-00094-of-000096.safetensors \
    --revision 6a2ef20a05ef --out DIR
```

- **Drafters:** `k31dspark` (Kimi K3.1 Pro's DSpark), `dsv41dspark` (DeepSeek
  V4.1 1017's DSpark, whose target tensors are in its own files, so no
  `--target`) and `qwen27bdspine` (Qwen3.8-27B's DSpine b16).
- **Contents:** every drafter tensor byte for byte under its checkpoint name,
  with its dtype as the kind (`bf16`, `f32`, `fp8`, `e8m0`, and `mxfp4_e2m1`
  for the DSpark's packed experts), and the target's embedding, head and final
  norm as `target.embed`, `target.head` and `target.norm`. Integer version
  scalars (the DSpine's) go into the pack's config.
- **Refusals:** a tensor that no rule packs or leaves out by name, a layer past
  the config's count, a dtype the drafter does not pack, a missing target tensor,
  a drafter tensor that the checkpoint's `model.safetensors.index.json` places
  in a file that is not there (a partial download holds only the drafter's
  files), and a source whose `.sha256` sidecar or `--expect-sha256` disagrees.
- **Sidecars:** `<pack>.sha256`; `<pack>.experts`, an empty weightd expert
  manifest, so weightd binds the whole pack as its spine; and
  `<pack>.receipt.json`, which names every source with its sha256, the
  left-out tensors and the scalars.

## Building for the 5090

The node runs driver 595.84 (CUDA 13.2 at most) on glibc 2.43, and three rules
follow from that:

- **Kernels from CUDA 13.3.** glibc 2.43 declares `rsqrt` and `rsqrtf`, which
  CUDA 13.1's headers declare differently, so 13.1's `nvcc` cannot compile any
  `.cu` there. 13.3's can. Build the 5090's code only, SASS, with
  `-gencode arch=compute_120a,code=sm_120a`; plain `-arch=sm_120a` refuses the
  FP4 conversions.
- **Every other host file and libcudart from CUDA 13.1.** The device layer asks
  the driver for entry points by the runtime's version, and a 13.3 runtime's are
  missing from a 13.2 driver, so weightd built against 13.3 does not start.
  `-cudart shared` and `-L/usr/local/cuda-13.1/lib64 -lcudart` link 13.3's
  generated host stubs to the 13.1 runtime.
- **Checked on 2026-11-22:** our `LmFusedResidualRmsNormKernel` built by 13.3,
  loaded as a cubin and launched through 13.3 stubs from a 13.1-runtime host
  (runtime 13010, driver 13020), matched a double-precision CPU reference on
  64 x 7168 rows (largest relative error 0.0039, one bf16 rounding). A cubin
  for another architecture is refused at load with
  `CUDA_ERROR_NO_BINARY_FOR_GPU`, never run.

```sh
make NVCC=/usr/local/cuda-13.3/bin/nvcc CUDA_HOME=/usr/local/cuda-13.1 \
    CUDA_ARCH=sm_120a CUDA_COMPUTE_ARCH=compute_120a drafter-modules
```

## Engine drafter modules

`sparkpipe_draftd` loads an engine lane's drafter from a module (see
[SPECULATION_UNIFIED_DESIGN.md](SPECULATION_UNIFIED_DESIGN.md), Draft server).

- **K3.1 Pro DSpark** (`modules/k31_dspark_draft_stage/`,
  `build/modules/k31_dspark_draft_stage.so`). It binds the
  `k31dspark.bf16.tp1` pack through weightd and finds every tensor by name in the
  pack's manifest.
  - **Context.** For every tap row (the five 7168-wide taps of layers
    75/79/83/87/91), it writes the drafter's context row in each of the five
    layers: `context_proj` then `context_norm`, then per layer `kv_a_proj_with_mqa`,
    `kv_a_layernorm` on the 512 compressed lanes and YaRN rope on the 64 rope
    lanes. Rows are bf16 at their positions in 4,096-position pages of the
    sequence, and whole sequences leave oldest first when the lane's
    `context_bytes` is reached.
    A frame at position 0 starts the sequence over. A frame from another
    engine session or request at a later position retires the sequence's
    context, which stays as a donor, and starts a new one, as
    `block_draftd.py` does. A frame at position `p` drops every row from `p`
    on and then holds its own rows, so a rewind after rejected drafts leaves
    no stale rows behind. The module keeps each position's token beside its
    row.
  - **Donor prefixes.** When a sequence's context lacks positions below the
    anchor (after a prefix-cache hit, the tap stream starts past the cached
    prefix), `propose` copies them from other contexts, the most recently
    used first, whose tokens match the committed ones up to that point, as
    `LatentContext.copy_prefix` does. Only rows the donor holds and the
    sequence lacks are copied, on the device.
  - **Block forward.** A draft at position `p` runs the seven-row block (the
    anchor and six mask tokens at `p..p+6`) through the five layers, as
    `K3DsparkDrafter.forward_block` does, with the same bf16 rounding points:
    - the target's embedding rows; per layer `input_layernorm`, `q_a_proj`,
      `q_a_layernorm`, `q_b_proj`, the absorbed queries (`kv_b_proj`'s key half,
      in fp32) with YaRN rope, and the block's own latent rows;
    - attention over the held context rows below `p`, then causally over the
      block. The context part is a tensor-core kernel with bf16 queries and
      bf16 weights over fp32 sums, split over up to 8 ranges of the context and
      merged with the block part in fp32, as the Python drafter's Triton kernel
      and `causal_block_attention` do. It reads the rows in place in their
      pages, and a gap in the held rows only shortens the context;
    - `kv_b_proj`'s value half, `o_proj`, `post_attention_layernorm`, the SwiGLU
      MLP, and the residual in fp32; then `final_norm`.
  - **Heads.** The block rows go through the target's head; the 2,048 best
    tokens of each row (exact top-k) are the candidates. Then seven steps, each
    from the token before it (the anchor first): a candidate's score is its
    logit plus its `markov_w2` row dotted with the previous token's `markov_w1`
    row, in fp32. Greedy takes the best score. With a sampling rule
    (`inverse_temperature > 0`) the 32 best are kept under the rule's nucleus
    and top-k, and the verifier's keyed Gumbel noise
    (`SparkMathGumbelNoise` at the slot's position) picks among them, as
    `coupled_candidates` does. The confidence head reads the block row and the
    previous token's `markov_w1` row.
  - **Proposals.** `propose` answers `AWAIT_TAPS` while the context does not
    hold the position before the anchor, checked on the host before any device
    work; `NO_CONTEXT` when a held token differs from the committed one (each
    position is checked once and remembered until a frame rewrites it);
    otherwise it runs the block and the heads and answers the root and up to
    seven drafts, the request's depth and node capacity permitting, each
    scored by its confidence.
  - **Check.** `build/k31_dspark_context_probe` writes random taps through the
    module, and `tools/spec_offline/k31_dspark_context_compare.py` recomputes
    the rows with the Python drafter and in float64. On the 5090 the module's
    worst row was 0.0024-0.0032 from float64 (relative L2), the Python
    drafter's 0.0030-0.0055, at 37 and 5 rows.
  - **Check of the block forward.** `tools/spec_offline/k31_dspark_block_compare.py
    extract` cuts the first `N` positions of a sequence out of a tap recording;
    `build/k31_dspark_block_probe` ingests them, runs the block at `N` and dumps
    the block rows, the context rows and each layer's attention inputs and
    output; `compare` checks each layer's attention against the Python
    drafter's on the same inputs and the block's top tokens under the target's
    head against the Python drafter's. It also runs `propose` under a greedy
    rule and two sampling rules and checks the drafts and confidences against
    `K3DsparkDrafter.propose` on the module's block rows. `--time` times the
    forward on synthetic context. The probe also checks the donor path: a sequence that holds only the
    second half of the positions borrows the first half and drafts what the
    whole sequence drafts.
  - **Replay gate.** `build/drafter_module_replay` drives any drafter module
    through a tap recording: before each decode, verify or chain frame at
    position `p > 0` it asks for a draft at `p` with the frame's first token
    as the anchor, then ingests the frame.
    `tools/spec_offline/drafter_replay_compare.py` replays the same schedule
    through the Python drafter with `block_draftd.py`'s context store and
    compares the answers, the drafts and each side's accepted length against
    the recording's committed stream.

## Engine draft server lane switch

The K3.1 Pro lane can be served by `sparkpipe_draftd` with the K3.1 Pro DSpark
engine module instead of `block_draftd.py`. The lane record's `draftd.server`
chooses: `block_draftd` (today) or `engine`. Its `draftd.engine` names the
engine root, the lane's name in the root's config, and the two user units.
The lane's `hub_start` and `hub_stop` in `deployment/hub/model_swap.json`
carry the same settings (`DRAFTD_SERVER`, `DRAFTD_ENGINE_*`,
`DRAFTD_WEIGHTD_UNIT`), and `tests/test_model_swap_hub_services.py` keeps
them equal to the record.

With `DRAFTD_SERVER=engine`, `tools/hub_draftd.sh`:
- `start`: starts the hub's weightd (`sparkpipe-weightd-hub`) and the engine
  draft server (`sparkpipe-draftd-engine`) if they are not running, waits
  for their sockets, then activates the lane through the control socket. The
  activation binds the pack through weightd's HOLD, checks free video memory
  against the pack and the lane's context cap, and exits 1 with the module's
  numbers when it does not fit.
- `stop`: deactivates the lane. Both units keep running; an inactive lane
  answers `UNAVAILABLE` at once, so the target decodes plainly.
- `status`: the draft server's `status` line for every lane.

The engine root, `/srv/workspace/sparkpipe/draftd-engine`, is staged from
merged main:
- `bin/sparkpipe_draftd` and `bin/sparkpipe_weightd`, built with
  `CUDA_HOME=/usr/local/cuda-13.1`; `modules/k31_dspark_draft_stage.so`, from
  `make NVCC=/usr/local/cuda-13.3/bin/nvcc CUDA_HOME=/usr/local/cuda-13.1
  CUDA_ARCH=sm_120a CUDA_COMPUTE_ARCH=compute_120a drafter-modules`;
- `packs/k31dspark.bf16.tp1.rank0.sp` with its `.sha256`, `.experts` and
  receipt, cut by `tools/drafter_pack.py --drafter k31dspark` from the
  drafter checkpoint and the target shard the lane record names;
- `config/draftd.json`, a copy of `deployment/hub/draftd_engine.json`. Its
  sockets live under `run/`, beside weightd's log. weightd keeps the default
  node KV root, the serving user's home, as the release kit never sets
  `SPARK_KV_NODE_ROOT`; it holds no KV here, and its purge walk only counts
  stores and deletes leftover temporaries.

The hooks run the `hub_draftd.sh` in the lane's `draftd.source` directory, so
that copy is replaced from merged main before the switch. The context cap is
12 GiB: with the 10.07 GB pack, it leaves room for the scratch on the 32 GB
card. To switch, set `server` to `engine` in the record and `DRAFTD_SERVER`
in the lane's two hooks, and deploy the controller config; the next
activation of the model starts the engine lane. To switch back, set both to
`block_draftd` and stop `sparkpipe-draftd-engine`, which holds the lane's tap
and draft ports.

**Soak before the switch stays.** `tools/spec_soak.py` drives the lane's API
on the hub with a fixed workload built from this repository's files: 48
rename edits of single files (1.5K to 5.7K prompt tokens), 16 after other
files (about 75K tokens) and 8 after more (about 112K). Even tasks decode
greedily; odd tasks sample with a seed fixed per task. Repeats reuse cached
prefixes, so the engine module copies donor prefixes as it would in service.
Run it once per arm, an hour each, on the same workload, never on user
traffic:

```sh
python3 tools/spec_soak.py run --engine http://127.0.0.1:8481 --chat-dir /srv/workspace/sparkpipe/k31-chat \
    --model kimi-k3.1-pro --log /srv/workspace/sparkpipe/logs/k31_pro-kv-api.log --minutes 60 --out SOAK/python \
    --unit sparkpipe-draftd-k31
python3 tools/spec_soak.py run ... --out SOAK/engine --unit sparkpipe-draftd-engine --unit sparkpipe-weightd-hub
python3 tools/spec_soak.py compare SOAK/python SOAK/engine
```

`compare` sums each arm's drafter counters from the API log's
`batch_speculation` lines (an API restart starts a new segment) and passes
when the engine arm's accepted drafts per round are within 2% of the Python
arm's, its `UNAVAILABLE` rate is no higher, and neither engine unit
restarted. Deactivate and activate the model once inside the engine hour so
the lane rebinds its pack from weightd's held copy.

## Draft farm

Each drafted lane asks its own `block_draftd.py` process, named by the lane
record's `draft_server`: Kimi K3, Kimi K3.1 Pro, Qwen3.8-27B and DeepSeek
V4.1 Flash. The earlier farm process in `~/draft_service` (n-gram and suffix
sources) drafts for no lane, and the stored `glm-5.3-flash-dflash2` drafter
serves no lane either. Report a drafter as serving only when a served model
drafts through it with measured acceptance and end-to-end tokens per second.

Hidden rows reach a drafter over the tap stream
([SPECULATION_UNIFIED_DESIGN.md](SPECULATION_UNIFIED_DESIGN.md#taps)).

Drafters in `/srv/workspace/drafters/`, copied 09-03 with the transfer record
`TRANSFER-RECEIPT.json` (source, route, destination, SHA-256 per weight file):
`deepseek-v4-flash-dflash-redhatai`, `glm-5.3-flash-dflash2`,
`kimi-k3-dflash-modal`, `kimi-k3-dflash2-lightseek`,
`qwen3.8-27b-dflash2-incoai`, `qwen3.8-max-dflash-modal`.

## Tooling

`tools/rtx5090_spec_node.py` and `deployment/rtx5090_speculation/node.example.json`
model the hub role since `7a0940e` (#1297, profile format
`ds4-auxiliary-node-v2`): the netplan render carries the route to the Spark
fabric through sparkf, the sparkf `nmcli` render names
`ds4-speculation-link`, and `plan` also renders the Spark hub-route unit.
`verify` is read-only (ssh probes and pings) and checks the hub's release
endpoint, API unit and heartbeats. Two parts still predate the host:

- `validate_profile` requires `drafters_path` to be `/srv/drafters`
  (rtx5090_spec_node.py:115-116), so the storage gate checks the empty
  directory and not the drafter LV at `/srv/workspace`.
- `runtime.transport` must be `not_implemented` (:109-110) although the DFT3
  client exists.

`deployment/rtx5090_speculation/spark_ssh_failover.example.json` is used by
[SPARK_MANAGEMENT_FAILOVER.md](https://github.com/sparkpipe/datacenter/blob/main/operations/SPARK_MANAGEMENT_FAILOVER.md) and is not
affected.

## draftd drafters (2026-09-29)

Drafters run on this host; the fleet only verifies. Measured on this host
(receipts and roofline lines in the spec-draftd lane notes):

- GLM-5.3 Flash MTP (layer 45). `tools/safetensors_subset.py` copies the MTP
  layer, `lm_head` and `embed_tokens` out of the checkpoint; running
  `tools/glm5_next_pack_verify.py --mtp-only --all-tensors --source <subset>`
  against each fleet sidecar proves the copy equals the TP16 sidecars.
  `tools/draftd_glm53flash_mtp.py` runs the layer with the C-ABI kernels in
  `tools/draftd_kernels.cu` (bf16 GEMV and expert-indexed fp8 block-128 GEMV,
  sm_120, loaded through ctypes) and one CUDA graph per chain depth.
  `tools/draftd_mtp_g8.py` compares top-1 with
  `tools/glm53flash_mtp_reference.py` on identical inputs, repeats the device
  pass with f64 accumulation as a noise-floor control, and checks run-to-run
  identity. `tools/draftd_bench.py` reports per-depth latency, bytes per draft
  token and the read peak; `tools/draftd_head_eval.py` compares the bf16 head
  with an fp8 screen and a certified fp8 screen.
- Block drafters. `tools/draftd_block_bench.py` times one round of a
  DFlash-style block drafter from its own weights with synthetic taps and a
  synthetic head of the target's shape. It measures cost only; top-1 needs the
  target's taps and a reference.

### Per-sequence MTP KV (S6b)

The context-free MTP chain starts every draft from an empty MTP cache, so the
MTP attention sees only the chain's own tokens (position-1 acceptance about
0.51 in the numpy reference). `MtpSequenceKv` in
`tools/draftd_glm53flash_mtp.py` keeps the layer-45 KV of one sequence:

- One latent row (the `kv_a` latent after `kv_a_layernorm`), one indexer key
  (`indexer.wk` then `k_norm`) and one kpool gate row
  (`index_kpool_compress_gate`) per committed position.
- `commit(kv, taps, next_tokens, start)` appends rows in order. Row p is the
  MTP input of tap row p and token p+1: prefill commits every prompt row at
  once, decode commits each verified round. The row GEMM
  (`spark_draftd_gemm_rows_bf16`) accumulates in the GEMV kernel's order, so
  a batched commit equals a row-by-row commit bit for bit.
- `draft(kv, depth)` chains from the last committed row. Speculative rows are
  written only past the committed length and never overwrite a committed
  row, so a draft is a function of the committed rows; the next commit
  replaces the speculative rows. `truncate` rolls back.
- Attention is dense while the context is at most `index_topk`; beyond it
  the MTP layer's own indexer scores kpool-compressed pools (softmax of gate
  plus ape over each pool of 4, ReLU head scores weighted by `weights_proj`)
  and attends to the top `index_topk / kpool` pools plus the tail, as the
  engine's DSA layers do. `glm53flash_mtp_reference.index_positions` is the
  numpy reference.
- The MTP input hidden is the post-final-norm row by default
  (`rmsnorm(hc_mean, model.norm)`), computed from the layer-44 hc-mean tap;
  `tap="hc_mean"` keeps the old input. The numpy reference puts the
  post-final-norm tap with sequence KV at 36/41 = 0.88 next-next hits at
  depth 1, against 21/41 context-free.

`tools/draftd_mtp_seqkv_check.py` compares the sequence-KV drafter with
`glm53flash_mtp_reference.py` in sequence context (depth 1, identical inputs,
two device passes for run-to-run identity). `tests/test_draftd_mtp_kv.py`
(CUDA) covers the row GEMM, the tap, commits, the draft against a
teacher-forced cache, speculative isolation, rollback and the indexer.

### Gate and admission

`tools/draftd_mtp_g8.py` decides G8 on the real-tap positions only: PASS
needs at least 1000 of them and top-1 agreement of at least 99%, with
identical output run to run. With fewer real positions the verdict is PENDING;
implementation positions are reported but never counted toward the gate.

draftd v0 does not check admission records itself. Load only drafters with an
admission record in `SPECULATOR-LICENSE-ADMISSION-20260828.json`; the G7 load
check (checkpoint sha256 plus admission id) belongs to the draftd daemon.
