# RTX 5090 speculation node

`rtx5090` is an auxiliary GPU host attached to `sparkf`. It is not a Spark,
does not occupy a rank, and must not be added to collectives, Ceph, or the
`ds4ring0` fabric.

## Network contract

| Purpose | RTX 5090 | sparkf | Routing |
| --- | --- | --- | --- |
| Management | Site LAN DHCP on wired and Wi-Fi interfaces | Site Wi-Fi | default route allowed |
| Recovery | Tailscale and site-configured key-only SSH | existing fleet recovery paths | management only |
| Speculation data | private-link endpoint A | private-link endpoint B | isolated `/30`; no gateway; MTU 9000 |

The direct `/30` prevents this cable from impersonating the former sparkf 10
GbE management uplink. The Spark 100/200 GbE interfaces, routes, Ceph bindings,
and ranks are unchanged.

Copy the redacted example to the ignored site-local profile and fill in the
actual addresses, interfaces, users, SSID, and recovery endpoint:

```bash
cp deployment/rtx5090_speculation/node.example.json \
  deployment/rtx5090_speculation/node.local.json
```

Render the intended local configuration without changing either host:

```bash
python3 tools/rtx5090_spec_node.py plan
```

After provisioning or reboot, run the end-to-end gate from the Mac controller:

```bash
python3 tools/rtx5090_spec_node.py verify
```

The gate requires the open NVIDIA driver, a 10 Gb/s link in both directions,
jumbo pings across the private cable, active and enabled Tailscale, and an
actual key-only recovery SSH login. Site-local profiles are ignored by Git.

Every Spark has a managed `rtx5090` host alias. General fleet nodes resolve it
to the site management address; `sparkf` resolves it to the private-link RTX
endpoint, keeping speculation traffic on the dedicated 10 Gb/s cable.

The workstation has a boot-mounted ext4 LV at `/srv/drafters`, owned by
`spec:spec`. Drafter models are copied one at a time into
`/srv/drafters/.staging/<model>`, checksum-compared to the published source,
and renamed to `/srv/drafters/<model>` only after the comparison is clean.

The initial published set is:

- `deepseek-v4-flash-dflash-redhatai`
- `glm-5.3-flash-dflash2`
- `kimi-k3-dflash-modal`
- `kimi-k3-dflash2-lightseek`
- `qwen3.8-27b-dflash2-incoai`
- `qwen3.8-max-dflash-modal`

The target-side record is `/srv/drafters/TRANSFER-RECEIPT.json`. It records the
source RAID, relay route, destination filesystem UUID, all model names, and the
independently verified SHA-256 for every weight file.

## Initial GPU qualification

The RTX 5090 reports 32,607 MiB of VRAM. The 2026-09-03 onsite gate used
NVIDIA open driver 595.84 and CUDA 13.3, compiled for `sm_120`. It deliberately
left CUDA headroom while pattern-writing and verifying 30,064,771,072 bytes
(89.3% of VRAM), measured 762.94 GB/s device-to-device copy and 49.44/23.12
GB/s PCIe host-to-device/device-to-host copy, then ran 8192-square cuBLAS GEMM
for 60 seconds. The GEMM result was numerically correct and averaged 233.89
TFLOP/s on the initial run and 233.92 TFLOP/s when rerun from the checked-in
source. Telemetry recorded 100% SM use, 384 W peak power, 59 C peak GPU
temperature, no thermal or power throttling, no PCIe errors, and no kernel Xid
or AER errors.

Rebuild and run the checked-in probe with:

```bash
/usr/local/cuda-13.3/bin/nvcc -O3 -std=c++17 -arch=sm_120 \
  tools/hardware/rtx5090_qualify.cu -lcublas -o /tmp/rtx5090_qualify
/tmp/rtx5090_qualify
```

The exact source, compile/run command, hardware metadata, and captured results
from the onsite run are also retained on the node under
`/srv/drafters/.qualification/`.

## Runtime boundary

This profile makes the host infrastructure-ready only. SparkPipe's current
speculator callback is process-local; the fleet does not yet have a remote
draft-token RPC protocol between sparkf and this workstation. Do not report
the node as serving speculation until that transport, model lifecycle, health
checks, and fallback semantics are implemented and measured.
