# Node baseline

Every fleet node (`spark0` to `sparkf`) runs one OS policy. The repository is
its only source:

- [`tools/devcycle/ds4_spark_brickproof.py`](../tools/devcycle/ds4_spark_brickproof.py)
  installs it from a git ref (`apply`) and checks it (`audit`). Apply runs a
  canary first, then the rest. Audit is read-only apart from staging its
  payload in `/tmp`.
- [`tools/fleet_post_reboot.sh`](../tools/fleet_post_reboot.sh) checks the boot
  outcome after any reboot and ends with `READY` or `NOT READY`.
- The node files live in [`tools/devcycle/fleet/`](../tools/devcycle/fleet/)
  and [`tools/devcycle/boot-unblock/`](../tools/devcycle/boot-unblock/). Apply
  copies them byte for byte. Audit reports any installed copy that differs as
  `asset-drift:<path>`.

```bash
python3 tools/devcycle/ds4_spark_brickproof.py audit --nodes spark8 \
  --sparkpipe-repo "$PWD" --source-ref origin/main
python3 tools/devcycle/ds4_spark_brickproof.py apply --nodes spark8 --canary spark8 \
  --sparkpipe-repo "$PWD" --source-ref origin/main
tools/fleet_post_reboot.sh spark8
```

Apply changes boot behaviour (masks, GRUB, the EFI order, the initramfs). Run
it on one node, reboot that node, and audit it before the others.

## What the baseline holds

| Item | Installed by | Why |
|---|---|---|
| `multi-user.target` default, `plymouth-quit-wait.service` masked | apply (`set-default`, mask list) | The factory DGX OS image boots `graphical.target`. Its splash wait never ends, so `multi-user.target` is never reached and every unit ordered after it, including the DMA guard, never starts (spark7/8/9, 2026-10-20). |
| Factory boot units masked: rbdmap, nvmf-autoconnect, open-iscsi, iscsid, srp_daemon, srptools, cloud-init (4), pollinate, nvidia-spark-run-apt-upgrade-once, systemd-networkd-wait-online | apply (`OPTIONAL_BOOT_UNITS`) | Boot work the fleet does not use. srp_daemon is an RDMA consumer on every port. Apply refuses while remote storage is configured. |
| Fabric ports unmanaged by NetworkManager | `ds4-switched-fabric-apply --network-manager` (`90-ds4-switched-fabric-unmanaged.conf`) | NetworkManager reads `conf.d` in name order and the last `unmanaged-devices` wins. A leftover `ds4-200g-unmanaged.conf` sorted last and left `enp1s0f0np0`/`enp1s0f1np1` managed, which flushes their manual addresses on carrier changes. The step moves the legacy override files to `/var/backups/ds4-fleet`. |
| `95-ds4-uplink-no-assume.conf` (`keep-configuration=no` on `enP7s7`) | same step | The initramfs turns the kernel `ip=` argument into `/run/netplan/enP7s7.yaml`. When carrier is already up as NetworkManager starts, it adopts that profile (gateway 10.20.0.1, no internet) instead of `ds4-uplink-wired`. The fleet-agent then cannot reach the hub. The FE nodes lose this race on every boot. Older nodes win it only on carrier timing. On sparkf the rule activates `ds4-speculation-link` (priority 300), as today. |
| `ds4-switched-fabric` and `ds4-direct-pair-fabric` scripts and timers | apply (assets) | The boot run sets only the switched-fabric address and the Ceph MDS host route. The route is added only when `10.20.0.2` is on-link on `enP7s7`. On sparkf it is not, and the old script failed every 30 s for 16 days. The pair map skips ranks 14 and 15. |
| `sparkpipe-hub-route.service` | apply (`LINKED_RANK_ASSET_SOURCES`, ranks 0-14) | Route to the hub through sparkf. Rank 15 is the hub gateway and has no unit. |
| DMA guard installed and armed | apply (`spark-dma-guard install --notify 192.168.50.4:5514`) | sysrq-b on the first CX-7 uncorrectable error or SMMU failure line. Audit requires `event=start mode=armed` in the current boot. The 7.0 kernel prints the same formats; a dry-run injection on spark7 on 2026-10-20 matched both the AER and the CMD_SYNC lines. |
| Hardware watchdog | apply (`/etc/systemd/system.conf.d/90-ds4-recovery.conf`) | `RuntimeWatchdogSec=10min`, `RebootWatchdogSec=5min`. Without it the SBSA watchdog stays off and a hard hang never reboots. |
| Emergency SSH on 2222, recovery key in root and Dropbear, GRUB recovery `ip=`, IPv4 PXE first | apply | The out-of-band and PXE rescue paths ([INCIDENT_RECOVERY_PLAYBOOK.md](INCIDENT_RECOVERY_PLAYBOOK.md)). The emergency sshd admits `root` and the `sparkemerg` account. The owner creates that account. |
| earlyoom `-m 3 -s 100 -r 5 --avoid sshd\|systemd\|init`, `user-1000.slice` limits, sysctls, `sch_fq_codel` | apply | Owner setting since 2026-10-07; `-m 7` killed K3 twice. |
| Ceph: cephadm daemons run; host package units masked; `/var/log/ceph/<fsid>` owned 167:167; mon/mgr/osd units enabled; units without a daemon directory disabled | apply (`install_ceph_policy`) | cephadm will not start a daemon without its log directory. A copied state tree once enabled a `crash` unit with no daemon. Stopped MDS or crash daemons may stay disabled. |
| `spark-osd-guard` (udev event and 2-minute scan), `95-ceph-osd-lvm.rules`, `ds4-ceph-warm` and its mount unit | apply (assets) | Repairs stale device-mapper tables after a USB enclosure re-enumerates. Gives the OSD logical volumes the container's ceph UID 167. Mounts the CephFS warm store. `ceph-osd-retry.service` is retired: its `ExecStart` named a file that did not exist on any node, and the OSD guard covers its job. |
| timesyncd to the hub, `restart-resolved`, fsck health timer, fstrim, persistent serial getty, `ssh.service` | apply | Fleet services every node already runs. |
| Tool and build packages (`iperf3`, `fio`, `ninja-build`, `gh`, `exfatprogs`, `python3.12-dev`, `libnng-dev`, `libmbedtls-dev`, `libexpat1-dev`) | apply (`FLEET_TOOL_PACKAGES`) | Present on the older nodes; missing on the factory image. |
| Stale snap mount units removed | apply (`retire_stale_snap_units`) | Units whose snap file is gone fail at every boot and hide real failures. They arrived on the FE nodes with a copy of `/etc/systemd/system` from the old nodes. |
| SoC firmware with the full memory carve-out (MemTotal at least 127,000,000 kB) | operator, per vendor (below) | Old SoC firmware keeps about 2 GB. |
| GPU clock cap 2200 MHz | the hub `fleet-clockcap` timer, runtime only | spark5 power workaround (owner, 2026-10-12). `fleet_post_reboot.sh` allows 15 minutes after boot for the timer. |

## Not in this repository

The repository is public. These files stay in the private coordination store
(`sparkpipe-coord/node-files`), or with the owner for secrets:

- the host firewall ruleset `/etc/spark-firewall.nft` (the unit is here);
- the ordered uplink failover tool `ds4_spark_uplink.py`, its monitor units and
  `/etc/ds4-uplink/config.json`, because they name the owner's Wi-Fi networks;
- `ds4-extnvme-link`, which adapts GPL-3.0 code;
- secrets: Wi-Fi PSKs and profiles, Ceph keyrings and `ds4warm.secret`,
  Tailscale state, the `sparkemerg` account and its keys.

## Replacing or adding a node

1. **Owner:** rack and cable the node exactly like its predecessor (port 0 is
   the pair link, port 1 the switch, the USB OSD enclosure moves with it). Create
   the first-boot user `sparkN` (uid 1000) with hostname `sparkN`. Authorize the
   workstation and hub keys, and give the user passwordless sudo.
2. **Firmware:** compare the node's ESRT against a node of the same vendor. If
   MemTotal is below the baseline, stage the vendor's SoC and EC capsules with a
   pinned `fwupdmgr install <device-id> <version>` (never the PD capsule, never
   `fwupdmgr update`). Set BootNext to the disk entry, because PXE is first, and
   reboot. The capsules take about 10 silent minutes. FE target on 2026-10-20:
   SoC `0x02009b0b`, EC `0x03000508`.
3. **Identity:** carry `/var/lib/ceph`, `/etc/ceph`, the SSH host keys, root's
   `authorized_keys`, the Tailscale state and `/etc/ds4-node-rank` from the
   retired node. Retire the old node first: rename it, delete its fleet network
   profiles, restart its avahi and DMA guard, and keep it off the fabric. A
   powered-on predecessor otherwise answers as `sparkN.local` and reports guard
   events under the old name.
4. **Private files and secrets:** place the firewall ruleset, the uplink failover
   set and the Wi-Fi profiles from the coordination store, and the owner's
   secrets.
5. **Baseline:** run `apply` on the node alone, reboot, then run `audit` and
   `tools/fleet_post_reboot.sh sparkN`. Both must pass before the node serves.
6. **Ceph:** hold `ceph osd set noout` from spark0 around the swap. If the OSD
   does not start, run `ceph cephadm osd activate sparkN`.

If the fleet agent logs `SAFE MODE` after several short boots, clear it with
`cat /proc/sys/kernel/random/boot_id > ~/.fleet_agent_safe_clear` and restart
`fleet-agent`.

## Open decisions

- **apt-daily:** spark0-6 mask `apt-daily.service` and `apt-daily-upgrade.service`;
  sparka-f and the FE nodes do not. Pick one.
- **GRUB serial terminal:** spark0-6 have it; sparka-f and the FE nodes do not.
- **Kernel and driver:** the FE nodes run 7.0.0-1019 with 580.178.04; the older
  nodes run 6.17.0-1026/1029 with 580.159/173. Label FE results in any A/B.
- **Dirty-page cap:** `/etc/sysctl.d/99-dirty-cap.conf` on the nodes holds a
  literal `\n` and a `kernel` prefix, so its 4 GiB `vm.dirty_bytes` cap never
  applied (`dirty_bytes` reads 0 everywhere). It is not part of the baseline
  until the setting is decided.
- **Ceph MDS route:** `10.10.100.2` via `10.20.0.2` points at an offline MDS
  host. The active MDS runs on the hub. Remove the route with that host.
- **sparkf emergency path:** the workstation reaches sparkf over Tailscale, where
  port 2222 is not reachable, so the audit's emergency login fails there.
