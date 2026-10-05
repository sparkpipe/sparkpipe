# Spark fleet boot and OOM recovery

Owner: SYSADMIN. Scope: the 16-node Spark fleet (`spark0` through `sparkf`).

This is the canonical recovery procedure. It has two deliberately separate
stages:

1. PXE restores a root login path on a node whose installed OS cannot be
   administered.
2. The brickproof controller repairs and audits the installed OS from the exact
   merged SparkPipe commit recorded in its receipt.

PXE is not an installer. It boots the installed root filesystem with boot-only
recovery arguments, prepends the fleet public key to root's authorized keys,
masks the optional fabric services for that boot, and continues to
`multi-user.target`. It does not format disks, replace packages, or overwrite
model data.

## Source-of-truth rule

Use only `sparkpipe/sparkpipe`. Do not deploy this policy from an archived fleet
repository, a dirty checkout, or unmerged branch bytes.

The controller fails closed unless its own source and every deployed asset are
read from the requested Git ref. A successful receipt therefore binds the host
configuration to one SparkPipe commit.

The normal sequence is:

1. Merge the recovery PR.
2. Fetch merged `main` in a clean SparkPipe checkout.
3. Run the controller from that checkout with `--source-ref origin/main`.
4. Keep the JSON receipt.

## Diagnosis

TCP port state is not an SSH login result:

- `connection refused`: no listener accepted the TCP connection.
- TCP connected, then `Connection timed out during banner exchange`: the kernel
  accepted the socket but userspace never emitted an SSH protocol banner. No
  authentication is possible.
- `Permission denied (publickey)`: SSH reached authentication, but none of the
  offered keys is authorized.

Probe the normal and emergency paths separately:

```bash
ssh -o ProxyCommand=none -o BatchMode=yes -o ConnectTimeout=8 \
  sparkN@10.20.0.X true

ssh -p 2222 -i ~/.ssh/sparkpipe_fleet_root \
  -o ProxyCommand=none -o BatchMode=yes -o IdentitiesOnly=yes \
  -o ConnectTimeout=8 root@10.20.0.X true
```

If either login works, skip PXE and run the brickproof controller. If neither
works, use PXE.

## Stage 1: PXE login rescue

The PXE server is generic: it serves any ARM64 UEFI client that explicitly asks
for the image. It is intentionally not tied to a MAC allowlist.

Start and verify it immediately before the affected host is rebooted:

```bash
python3 tools/devcycle/ds4_parallel_pxe_rescue.py start --server spark0
python3 tools/devcycle/ds4_parallel_pxe_rescue.py status \
  --server spark0 --require-active
```

The status must show all of the following before a power cycle:

- service `active`
- DHCP proxy listeners on UDP 67 and 4011
- TFTP listener on UDP 69
- exactly one scoped firewall rule
- empty `failures`

The fleet policy puts IPv4 PXE before the local Linux EFI entry. A correctly
hardened node therefore requests PXE automatically while the server is active
and falls through to its local OS when the server is inactive.

For a Dell Pro Max with GB10 that has not yet received the policy, the official
one-time boot key is **F7**. Tap F7 during the Dell logo and select the wired IPv4
PXE entry. F7 opens the one-time boot-device menu; it is not Dell OS recovery.

Evidence that firmware actually entered PXE includes TFTP requests for the EFI
loader, GRUB configuration, kernel, and initrd. A DHCP line containing the Linux
hostname but no TFTP transfer means the installed OS booted instead; it is not a
PXE success.

When the installed OS hangs even with the fabric units masked, deploy with
`--hold bottom`. The rescue initramfs then stops after mounting the installed
root at `/root` and before handing over to it, with its network and the
initramfs SSH server on port 22 still up (root, fleet recovery key, at the
node's DHCP address in the PXE server log). Read the hung boots' journals with
`journalctl -D /root/var/log/journal --list-boots` copied to a working host,
repair the root, then kill the `(initramfs)` console shell to continue the
boot. Redeploy with the default `--hold none` afterwards.

After PXE, test both SSH paths. The rescue initrd places the fleet public key at
the beginning of `/root/.ssh/authorized_keys`, so port 2222 should work even if
the previous file contained malformed or concatenated records.

Keep PXE active until the installed OS is repaired and its local boot is proved.
Then stop it and verify that its listeners and firewall rule are gone:

```bash
python3 tools/devcycle/ds4_parallel_pxe_rescue.py stop --server spark0
python3 tools/devcycle/ds4_parallel_pxe_rescue.py status --server spark0
```

## Stage 2: repair and audit the installed OS

Run the merged-main controller. Use a canary before the parallel fan-out:

```bash
python3 tools/devcycle/ds4_spark_brickproof.py apply \
  --nodes spark0,spark1,spark2,spark3,spark4,spark5,spark6,spark7,\
spark8,spark9,sparka,sparkb,sparkc,sparkd,sparke,sparkf \
  --canary spark3 --jobs 8 \
  --sparkpipe-repo "$PWD" --source-ref origin/main \
  --recovery-identity ~/.ssh/sparkpipe_fleet_root
```

The apply operation is idempotent and performs a post-apply audit plus a real
controller login on emergency port 2222. It fails if any node misses a gate.

The policy repairs and verifies:

- exact running and next-boot kernel image/module packages
- `/lib/modules` trees and `dpkg -V` package integrity
- initramfs rebuild for every managed kernel
- root fleet key and initramfs Dropbear key
- emergency root-only SSH on port 2222
- canonical hostname `sparkN`, `/etc/ds4-node-rank`, and no Spark loopback alias
- one generated `/etc/hosts` fleet block: `sparkN`/`sparkN-fabric` use switched
  100G and `sparkN-mgmt` uses the management network; obsolete `-ring`, `-200g`,
  and `-10g` aliases are removed
- static management address and serial-console boot arguments
- IPv4 PXE before the one canonical Linux EFI entry
- root fsck skipped during boot and handled by the post-boot health timer
- bounded fabric, remote-storage, and firewall work
- fabric services launched by timers, never as login-blocking boot dependencies
- every Ceph service and target is stopped and persistently masked; SparkPipe's
  former optional-storage timer and marker are removed
- weekly filesystem trim has an unmasked service and an enabled timer
- logrotate configuration is parsed during every fleet audit
- 64 GiB swap, earlyoom, no-swap model cgroups, and a 108 GiB user/model ceiling
- network recovery sysctls and required qdisc modules

### Ceph quarantine

This playbook treats Ceph as unsupported. Before changing Ceph state, the
controller fails closed on remote-storage configuration: a non-comment
`/etc/ceph/rbdmap` entry, an `/etc/fstab` entry of type `nfs`, `nfs4`, `cifs`
or `ceph` or with `_netdev`, a file under `/etc/iscsi/nodes`, or an
`/etc/nvme/discovery.conf` entry (`tools/devcycle/ds4_spark_brickproof.py:676-693`).
It does not inspect live mounts. A `ceph-fuse@` mount started by its unit
passes the precheck and is then stopped and masked (`CEPH_MASK_UNITS`,
`ds4_spark_brickproof.py:44-61`; `remove_ceph_startup`, `:617-649`). With the
precheck clear, apply performs the complete quarantine:

- stop every active `ceph*.service` and `ceph*.target`
- remove all Ceph wants/requires links and Cephadm-generated unit definitions
  from `/etc/systemd/system`
- remove the legacy SparkPipe optional-storage service, timer, and selection
  marker
- mask the canonical package units and every discovered Ceph service/target
- preserve `/var/lib/ceph`, `/etc/ceph`, OSD devices, and installed packages

The audit rejects any active Ceph unit, process, or labeled container, startup
link, non-mask Ceph systemd artifact, unmasked Ceph unit, legacy marker, or
obsolete SparkPipe optional-storage file. Re-enabling Ceph requires a separate
tested design and deployment; it is not a fleet-recovery operation.

**Open policy conflict.** Other current material uses Ceph at `/mnt/model-warm`
as the pack source: the GLM 5.3 Flash contract (`source.path_on_sparks` in
`model_contracts/glm53_flash_authoritative.json`) and the fleet runbook's
warm-storage law ([FLEET_RELEASE_RUNBOOK.md](FLEET_RELEASE_RUNBOOK.md#9-laws)).
spark0 had `/mnt/model-warm` mounted on 2026-09-28 (read-only check). Until the
operator rules on Ceph, do not run stage 2 on a node that mounts
`/mnt/model-warm` unless losing that mount is intended.

## DMA guard

spark8 failed twice (2026-10-01, 2026-10-02) the same way:
1. Its ConnectX-7 logged an uncorrectable PCIe completion timeout.
2. About 1 s later, SMMU0 logged `CMD_SYNC timeout`.
3. Minutes later, pages that belonged elsewhere were written to the root disk (superblock and group descriptors).

`tools/devcycle/spark_dma_guard.py` turns that sequence into an immediate reboot instead of disk corruption.

**What it does**
- `run` reads `/dev/kmsg` from the start of the current boot. Only kernel-facility records count, so user-space writes cannot trigger it.
- It reboots on the first of these:
  - an SMMU `CMD_SYNC timeout`;
  - an SMMU `CMDQ error`, global error or service-failure line;
  - `mlx5_pci_err_detected`, or an mlx5 uncorrectable `PCIe Bus Error`;
  - an AER `Uncorrectable (Fatal)` from any device;
  - an AER `Uncorrectable (Non-Fatal)` from a Mellanox device (vendor 0x15b3);
  - a fatal firmware-first `[Hardware Error]`.
- It sends a UDP note to the hub, forwards up to 30 ms of follow-up kernel lines (they carry the error type), waits 0.1 s, then writes `b` to `/proc/sysrq-trigger`. That reboots without syncing, so nothing more reaches the disk.
- The daemon locks its memory and opens the sysrq file at start.

**Loop protection**
- `/var/lib/spark-dma-guard/boots.json` records each boot. A clean stop marks the boot clean.
- If the last 3 boots ended uncleanly within 6 hours, or the state file cannot be written, the guard runs watch-only: it reports, but does not reboot.
- Delete the file to re-arm.

**Install**
- `install --notify 192.168.50.4:5514` installs `/usr/local/sbin/spark-dma-guard`, `spark-dma-guard.service` and `spark-dma-guard.timer`.
- The timer starts the service 60 s after boot. It is ordered after `multi-user.target`, `ssh.service` and `network-online.target`, and is never in the boot-critical path. Restarts are bounded to 5 in 10 minutes.
- Run `scan` before installing: it reports any trigger already in the current boot's log, because `run` would act on it at once.
- `self-test` checks the matcher against the real spark8 lines.

**Hub listener**
- `listen --port 5514 --log ~/spark-dma-guard/events.log` runs on the rtx5090 as the transient user unit `spark-dma-guard-listen`. Start it again after a hub reboot.

**Verification (spark8, 2026-10-02)**
- A clean reboot re-armed the guard 61 s after boot.
- An injected trigger in a test instance (`--accept-user`) reached the hub in 19 ms and reset the node. The next boot's first journal entry came 94 s after the trigger; every real guarded crash since took 89-95 s.
- The killed boot's journal stops before the trigger.

**After any single node reboots**
- The other nodes' weightd keep wiring to its old queue pair: `WD-WIRE-FAIL ... RTR failed errno=22`. Engines then fail with BUSY at collective attach.
- Fix: stop the engines, then run `systemctl --user restart sparkpipe-weightd` on the surviving nodes (restarting `fleet-agent` no longer restarts weightd; it runs in its own unit). On every node, wait until `~/sparkdata/core/bin/sparkpipe_mesh_status --socket /tmp/spark_weightd.sock --wait-lane-peers 0xffff --timeout 300` exits 0, then start the engines. Exit 3 lists the ranks still missing.

**IOMMU passthrough on the Lenovo nodes**
- spark7, spark8 and spark9 boot with `iommu.passthrough=1`, from `/etc/default/grub.d/zz-spark-iommu-passthrough.cfg`. It sorts after the DGX OS `iommu.cfg`, and the kernel uses the last value.
- The kernel then reports `Default domain type: Passthrough`. The CX-7, NVMe and USB groups are `identity`, and the GPU group stays `DMA`.
- To revert, delete the file, run `update-grub` and reboot.
- Passthrough lowers the Lenovo crash rate about 2x but does not stop the completion timeout. The investigation and every crash run are in [CX7_DMA_INVESTIGATION_2026-10.md](CX7_DMA_INVESTIGATION_2026-10.md).

## Fleet-duty acceptance

A host is not fleet-ready merely because it answers ping or accepts TCP. Record
all of these live checks:

1. Normal SSH command exits zero.
2. Emergency root SSH on port 2222 exits zero with the fleet key.
3. `systemctl is-system-running` is `running`.
4. No failed units; the audit rejects every failed systemd unit by name.
5. Running kernel and `/boot/vmlinuz` target both have intact image, modules,
   initrd, and module tree.
6. GPU is visible and healthy.
7. Switched interface is 100000 Mb/s full duplex with carrier.
8. Direct-pair interface is 200000 Mb/s full duplex with carrier.
9. Required local and external filesystems are mounted read-write.
10. Tailscale reports the node online with `tag:spark`.
11. Brickproof audit has no failures and records the merged source commit.

Do not call the fleet ready while a node is omitted silently. Name unreachable
or physically blocked nodes explicitly in the final receipt.

## Secrets

The repository contains only the fleet **public** recovery key. Private keys,
Wi-Fi credentials, Tailscale state, passwords, and controller-specific SSH
configuration remain outside the repository.
