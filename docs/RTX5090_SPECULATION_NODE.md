# RTX 5090 host: hub, GLM API, draft farm

`rtx5090` (x86_64, RTX 5090 32 GB) is the fleet hub and the GLM API host,
and holds the draft farm and drafter store. It is not a Spark: it has no
rank and is not part of the TP collectives. State below was surveyed
read-only on 2026-09-28 about 09:00 UTC unless another source is named.
Hub operations (publishing roots, heartbeats) are governed by
[FLEET_RELEASE_RUNBOOK.md](FLEET_RELEASE_RUNBOOK.md); speculation design by
[SPECULATION_UNIFIED_DESIGN.md](SPECULATION_UNIFIED_DESIGN.md).

## Rules

- SparkPipe owns only its units (`g53-api`, `fleet-release`) and paths
  (`~/release`, `~/current`, `~/g53-api-channel`, `~/draft_service`,
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
| GLM API | user unit `g53-api.service`: `~/g53-api-channel/bin/sparkpipe_model_api --port 8433`, with a tokenizer sidecar (`tokenizer/tokenizer.json` in its `model_resident.json`) | `systemctl --user show g53-api`; lead-dev facts |
| Draft farm | `~/draft_service` (not in this repo): `farm_server.py`, DFT3 protocol in `draft_protocol.py`, measurements in its `README.md`; not running | `ls`, `pgrep` |
| Drafter store | LV `ubuntu--vg-drafters` (700 GiB ext4) mounted at `/srv/workspace` by fstab: `drafters/`, `.qualification/`, `.staging/` | `lsblk`, `findmnt`, `/etc/fstab` |
| Other tenants | Ceph MDS and MGR, an NFS server, chess work (`stockfish`) | `ps` |

The API binary must be an x86_64 build of the engines' source commit: the
release root ships an aarch64 API that cannot run here, so every release
needs an x86 API build step (lead-dev facts, 2026-09-28).

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

## Draft farm

Only qwen38_27b consumes remote drafts, with the tap-free sources NGRAM,
SUFFIX and NGRAM3; glm5_next has no remote call site and no DFlash2 source,
so the stored `glm-5.3-flash-dflash2` drafter cannot serve the current GLM
engine. What GLM needs first is in
[SPECULATION_UNIFIED_DESIGN.md](SPECULATION_UNIFIED_DESIGN.md#remote-drafting-on-the-rtx-5090).
Report the farm as serving only when a served model drafts through it with
measured acceptance and end-to-end tokens per second.

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

`tools/draftd_mtp_g8.py` decides G8 on the real-tap positions only: PASS
needs at least 1000 of them and top-1 agreement of at least 99%, with
identical output run to run. With fewer real positions the verdict is PENDING;
implementation positions are reported but never counted toward the gate.

draftd v0 does not check admission records itself. Load only drafters with an
admission record in `SPECULATOR-LICENSE-ADMISSION-20260828.json`; the G7 load
check (checkpoint sha256 plus admission id) belongs to the draftd daemon.
