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
  unrelated jobs (FLEET_RELEASE_RUNBOOK.md §0 and law 6).
- Do not install the output of `tools/rtx5090_spec_node.py plan` or
  `render-netplan`: it predates the hub role (see [Tooling](#tooling)).
- Do not run `tools/hardware/rtx5090_qualify.cu` on a busy host: it
  allocates all free VRAM minus 2 GiB, capped at 28 GiB (rtx5090_qualify.cu:47-49),
  then runs a 60 s cuBLAS GEMM stress (:139). Rerun only with an idle GPU and
  the operator's sign-off.

## Roles

| Role | What runs | Source |
| --- | --- | --- |
| Fleet hub | user unit `fleet-release.service` (`python3 ~/fleet_release_serve.py`) serves `~/release/<root>/` on `:8802`; nodes write `~/current/<host>.json` heartbeats | FLEET_RELEASE_RUNBOOK.md §0; `systemctl --user`, `ss -ltn` |
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
and to 10.10.250.2 on sparkf; fleet traffic to the hub uses 10.10.250.2
either way.

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
describe the 09-03 draft-only box and no longer match the host:

- `render-netplan` emits an address with no routes (rtx5090_spec_node.py:110-128).
  Installed on the hub it would drop the static route to 10.10.100.0/24 and
  cut the hub and API off from the fleet.
- The sparkf `nmcli` render modifies connection `ds4-uplink-wired`
  (:167-186); that connection was already renamed `ds4-speculation-link`.
- `verify` is read-only (ssh probes and pings), but its storage gate checks
  the profile's `drafters_path`; the example's `/srv/drafters` now fails the
  600 GiB capacity check, and `runtime.transport` still says
  `not_implemented` although the DFT3 client exists.

`deployment/rtx5090_speculation/spark_ssh_failover.example.json` is used by
[SPARK_MANAGEMENT_FAILOVER.md](SPARK_MANAGEMENT_FAILOVER.md) and is not
affected.
