# RTX 5090 host: hub, model APIs, draft farm

`rtx5090` (x86_64, RTX 5090 32 GB) is the fleet hub and the model API host,
and holds the draft farm and drafter store. It is not a Spark: it has no
rank and is not part of the TP collectives. State below was surveyed
read-only on 2026-09-28 about 09:00 UTC unless another source is named.
Hub operations (publishing roots, heartbeats) are governed by
[FLEET_RELEASE_RUNBOOK.md](FLEET_RELEASE_RUNBOOK.md); speculation design by
[SPECULATION_UNIFIED_DESIGN.md](SPECULATION_UNIFIED_DESIGN.md).

## Rules

- SparkPipe owns only its units (`fleet-release`, `sparkpipe-swap`,
  `sparkpipe-chat`, `sparkpipe-litellm`, `spark-dma-guard-listen`,
  `sparkpipe-draftd` and the transient `kv-<model>-api` units) and paths (`~/release`,
  `~/release-staging`, `~/current`, `~/<model>-kv-api`, `~/sparkpipe-chat`,
  `~/sparkpipe-swap`, `/srv/workspace/sparkpipe`, `~/draft_service`,
  `/srv/workspace/drafters`). Never kill or restart anything else: the host runs
  unrelated jobs (FLEET_RELEASE_RUNBOOK.md §1, and AGENTS.md).
- Review the output of `tools/rtx5090_spec_node.py plan` against the live
  host before installing any of it (see [Tooling](#tooling)).
- Do not run `tools/hardware/rtx5090_qualify.cu` on a busy host: it
  allocates all free VRAM minus 2 GiB, capped at 28 GiB (rtx5090_qualify.cu:47-49),
  then runs a 60 s cuBLAS GEMM stress (:139). Rerun only with an idle GPU and
  the operator's sign-off.

## Roles

| Role | What runs | Source |
| --- | --- | --- |
| Fleet hub | user unit `fleet-release.service` (`python3 ~/fleet_release_serve.py`) serves `~/release/<root>/` on `:8802`; nodes write `~/current/<host>.json` heartbeats | FLEET_RELEASE_RUNBOOK.md §1; `systemctl --user`, `ss -ltn` |
| Model APIs | the swap controller `sparkpipe-swap` (127.0.0.1:8440) runs one model's engine API at a time, as the transient unit `kv-<model>-api` from `~/<model>-kv-api`; the chat layer `sparkpipe-chat` (`127.0.0.1:8433`) and LiteLLM `sparkpipe-litellm` (`:4000`) front it | `deployment/hub/model_swap.json`, [LITELLM_FRONTEND.md](LITELLM_FRONTEND.md), FLEET_RELEASE_RUNBOOK.md §6 |
| Draft farm | `~/draft_service` (not in this repo): `farm_server.py`, DFT3 protocol in `draft_protocol.py`, measurements in its `README.md`; not running | `ls`, `pgrep` |
| Drafter store | LV `ubuntu--vg-drafters` (700 GiB ext4) mounted at `/srv/workspace` by fstab: `drafters/`, `.qualification/`, `.staging/` | `lsblk`, `findmnt`, `/etc/fstab` |
| Other tenants | Ceph MDS and MGR, an NFS server, chess work (`stockfish`) | `ps` |

The API binary must be an x86_64 build of the engines' source commit: the
release root ships an aarch64 API that cannot run here, so every release
needs an x86 API build step (lead-dev facts, 2026-09-28).
`scripts/release_build.sh` does it on the hub in
`/srv/workspace/sparkpipe/api-build-<sha>`.

`/srv/drafters`, the path the 09-03 setup used, is now an empty directory on
the root filesystem; the drafter LV moved to `/srv/workspace`.

## Network

| Link | Addresses | Observed |
| --- | --- | --- |
| Fleet /30 (10 GbE) | rtx5090 `eno1` 10.10.250.2/30, sparkf `enP7s7` 10.10.250.1 | MTU 9000, 10000 Mb/s on both ends; sparkf connection name `ds4-speculation-link` |
| Fleet route, hub side | `10.10.100.0/24 via 10.10.250.1 dev eno1 proto static` | `ip route` on rtx5090 |
| Fleet route, Spark side | 10.10.250.0/30 via sparkf 10.10.100.25, from `sparkpipe-hub-route.service` | active on spark0; lead-dev facts |
| Site LAN, management | `eno2` 192.168.50.4 (default route, metric 100), Wi-Fi 192.168.50.68 (metric 300), Tailscale 100.123.97.61 | `ip -4 -br addr`, `ip route` |

sparkf NATs the /30 to the fleet (lead-dev facts; sparkf's nftables ruleset
held 3 masquerade rules). The `rtx5090` host alias resolves to 192.168.50.4 on spark0
and to 10.10.250.2 on sparkf. The fleet agents do not use this link: they
fetch releases from `http://100.123.97.61:8802` and copy heartbeats and mesh
records to `spec@100.123.97.61`, both over Tailscale
(`tools/fleet_node_agent.sh`, `RELEASE_HTTP` and `HUB`;
FLEET_RELEASE_RUNBOOK.md §1). Only traffic addressed to 10.10.250.2 takes the
/30.

Measured round trip (lead-dev, 2026-09-28): busy-polled 64 B UDP between
sparkf and rtx5090 is 22.6 us p50 / 25.2 us p99; forwarded from other Sparks
it is about 300 us p50.

## K3 block drafter (production since 2026-10-17)

K3 serves model drafts from the RadixArk DSpark block drafter on the 5090.
- **Unit:** the transient user unit `sparkpipe-draftd`, started by
  `tools/hub_draftd.sh start /srv/workspace/sparkpipe/draftd-<sha>`. The
  directory holds `block_draftd.py` and `spec_offline/` from that commit.
  `tools/hub_draftd.sh status` shows the unit, its two ports and the last
  timing and tap lines; `stop` stops it.
- **Restart after a hub reboot:** like every hub unit, it does not survive a
  reboot. Start it again with `tools/hub_draftd.sh start`. Until then, K3
  serves without model drafts: the tap rank retries every 5 s, and each draft
  request answers unavailable.
- **Inputs:** the K3 tap rank (sparkf, rank 15) streams the drafter's five
  target layers (`taps` in the K3 lane's `config/adapter.json`) to
  `10.10.210.3:7701`, the hub's 100G link to sparkf (`enp4s0f1np1`).
- **Clients:** the K3 API (`~/k3-kv-api`, `draft_server` in its
  `model_resident.json`) asks `127.0.0.1:7702` over DFT3.
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
  corpora only, never on users' traffic.

## Latent block drafter (lightseek DFlash2, offline)

`tools/spec_offline/latent_block_drafter.py` runs checkpoints whose
`dflash_config.attention_mode` is `mla`: lightseekorg/kimi-k3-dflash2 at
e77935fb (5 layers, 3.69B parameters, received 2026-10-18 into
`/mnt/cold-raid6/models/kimi-k3-dflash2-lightseek-e77935fb`, sha256
1f2a5376...52c2, receipt beside it). `open_drafter` in `block_drafter.py`
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

## Draft farm

Only qwen38_27b consumes remote drafts, with the tap-free sources NGRAM,
SUFFIX and NGRAM3; glm5_next has no remote call site and no DFlash2 source,
so the stored `glm-5.3-flash-dflash2` drafter cannot serve the current GLM
engine. What GLM needs first is in
[SPECULATION_UNIFIED_DESIGN.md](SPECULATION_UNIFIED_DESIGN.md#remote-drafting-on-the-rtx-5090).
Report the farm as serving only when a served model drafts through it with
measured acceptance and end-to-end tokens per second.

The fleet side for GLM Flash (hidden-row taps, `SPT1` tap frames, the tap
dump, the shadow-mode relay and the sparkf-rtx5090 relay probe with measured
round trips) is in [SPECULATION_TAPS.md](SPECULATION_TAPS.md).

Drafters in `/srv/workspace/drafters/`, copied 09-03 with the transfer record
`TRANSFER-RECEIPT.json` (source, route, destination, SHA-256 per weight file):
`deepseek-v4-flash-dflash-redhatai`, `glm-5.3-flash-dflash2`,
`kimi-k3-dflash-modal`, `kimi-k3-dflash2-lightseek`,
`qwen3.8-27b-dflash2-incoai`, `qwen3.8-max-dflash-modal`.

## GPU qualification record (2026-09-03, historical)

From `/srv/workspace/.qualification/RTX5090-GPU-QUALIFICATION.txt` (driver
595.84 open, CUDA 13.3.73, `-arch=sm_120`): 30,064,771,072 bytes (89.3% of
VRAM) pattern-verified with no mismatches; device copy 762.87 GB/s; PCIe
49.56 GB/s host-to-device and 22.75 GB/s device-to-host; 8192-square cuBLAS
GEMM for 60 s averaged 233.89 TFLOP/s with correct output; peak 384 W and
59 C, no power or thermal violation, no PCIe, Xid or AER errors.
`nvidia-smi` reports 32,607 MiB.

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
[SPARK_MANAGEMENT_FAILOVER.md](SPARK_MANAGEMENT_FAILOVER.md) and is not
affected.

## draftd drafters (2026-09-29)

Drafters run on this host; the fleet only verifies. Measured on the RTX 5090
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
- n-gram drafter. `src/spark_speculation_ngram_draft.c` drafts the most
  frequent continuation of the longest matching context (orders min..max, the
  last `scan_limit` verified occurrences, ties to the most recent) and chains
  through its own drafts. It has the policy draft callback signature of the
  lookup drafter. `tools/draftd_ngram_eval.py` checks it against a brute-force
  reference on recorded greedy streams and prices it next to the lookup
  drafter.

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

### Recorded draft tables

Until rank 15 broadcasts relay drafts to all ranks (section 11.3 of
SPECULATION_PLAN.md), a live rtx5090 draft cannot reach the verify. To
measure B1 speculation with real draftd drafts, draftd writes a draft table
offline from a tap dump of the same greedy batch, and every rank loads it
with `SPARK_GLM5_NEXT_VERIFY_DRAFTER=recorded:PATH`
(`src/spark_speculation_recorded_draft.c`). Every rank reads the same file, so
the verify rows agree without a broadcast. Because greedy B1 output repeats
run to run, the committed prefix at each request equals the dump's, and each
entry is the chain draftd would have sent. The arm pays no draft time; a
report adds the relay and draftd cost per round to give the live-equivalent
rate.

Table layout (little endian): a 32-byte header (magic `SPRD`, version 1,
depth, vocabulary size, entry count, a zero word), then fixed entries sorted
by (sequence id, position): sequence id u64, position of the last committed
token u64, chain length u32, a zero word, `depth` u32 tokens with zeros past
the chain. `tools/spec_recorded_drafts.py` writes and reads it.

`tools/draftd_mtp_g8.py` decides G8 on the real-tap positions only: PASS
needs at least 1000 of them and top-1 agreement of at least 99%, with
identical output run to run. With fewer real positions the verdict is PENDING;
implementation positions are reported but never counted toward the gate.

draftd v0 does not check admission records itself. Load only drafters with an
admission record in `SPECULATOR-LICENSE-ADMISSION-20260828.json`; the G7 load
check (checkpoint sha256 plus admission id) belongs to the draftd daemon.
