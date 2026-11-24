# Hosted node image

Rented nodes boot one minimal image. It contains:

- Ubuntu 24.04 from a pinned archive snapshot, with Canonical's `linux-nvidia` 7.0 kernel;
- Canonical's signed build of NVIDIA's open GPU kernel module at the version the device backend pins, verified by hash, and its GSP firmware;
- the in-tree `mlx5_core` and `r8127` drivers, with Ubuntu's rdma-core;
- Tailscale at a pinned version, verified by hash;
- the node-independent part of the [node baseline](NODE_BASELINE.md);
- a release root and the node agent.

The image carries no NVIDIA user-space package. The build fails unless the
[dependency audit](HOSTED_DEPENDENCY_AUDIT.md) passes on the finished root
filesystem and on its release root.

## Build

```bash
sudo tools/hosted_image/build_image.sh --out BUILD_DIR \
  --source-ref HEAD --release-root RELEASE_ROOT \
  --firewall-ruleset FIREWALL_RULESET
```

It runs as root on an arm64 Ubuntu host from a clean checkout at
`--source-ref`. No GPU is used, and no package, module or setting is
changed on the build host. All four inputs are required:

- `--release-root` must pass the root audit before anything is built.
- `--firewall-ruleset` is the private host firewall ruleset, so the firewall
  is active from the first boot. It is a build input and is never committed.

The steps:

1. Download mmdebstrap `MMDEBSTRAP_VERSION` from the snapshot, check its
   sha256, and run it from the unpacked package.
2. Download `MODULE_PACKAGE` from the snapshot and check its sha256 and
   package name. Unpack it, and take only `nvidia.ko` and `nvidia-uvm.ko`.
   Each must report `MODULE_VERSION`, the vermagic of `KERNEL_RELEASE` and
   `MODULE_SIGNER`. The package itself is not installed.
3. Build the image from `image-packages.txt` with
   [`customize_image.sh`](../tools/hosted_image/customize_image.sh), which:
   - copies [`files/`](../tools/hosted_image/files/) and the firewall
     ruleset;
   - installs the two modules under `/lib/modules/KERNEL_RELEASE/updates/`
     and checks that `depmod` resolves each to them;
   - checks the GSP firmware for the module's version;
   - copies the release root to `/opt/sparkpipe/release` and the agent to
     `/opt/sparkpipe/core/bin/fleet_node_agent.sh`;
   - stages the baseline (below) and enables the host-key unit;
   - removes every SSH and initramfs host key;
   - rebuilds the initramfs.
4. Check that no mount remains under the output directory. Every removal
   the build makes first checks this with `findmnt` and refuses otherwise.
5. Run the image audit. On a failure, the root filesystem is kept for
   inspection and nothing is packed.
6. Write `rootfs.tar.zst` (sorted names, mtimes clamped to the snapshot
   time), `MANIFEST`, `audit.json`, `baseline-report.json` and the build
   logs.

`MANIFEST` records:

- the source commit and every pin;
- each module's sha256;
- the archive's sha256 and the audit verdict;
- every release-root file with its sha256;
- every installed package with its version.

## Pins

[`pins.conf`](../tools/hosted_image/pins.conf) holds:

- the snapshot (`https://snapshot.ubuntu.com/ubuntu/<time>`, which also
  serves arm64) and `SOURCE_DATE_EPOCH`, its time;
- the kernel release;
- mmdebstrap's version and sha256;
- the module package, its pool path and sha256, the module version, the
  modules taken from it and their signer;
- the firmware package;
- the Tailscale package's version, pool path and sha256 (Tailscale's own
  repository; the image installs the verified `.deb` and adds no APT source
  for it);
- the install paths.

[`image-packages.txt`](../tools/hosted_image/image-packages.txt) names every
top-level package as `name=version`. Each version is the highest the
snapshot holds across `noble`, `noble-updates` and `noble-security`, by
`dpkg --compare-versions`; the snapshot fixes every dependency. A pin
changes only together with the snapshot. The kernel and the module package
are one pair: a new module version means a new kernel release, new pins and
requalification of the device backend.

A fleet node also stores one of the cluster's Ceph OSDs, which cephadm runs
in a container, and mounts the CephFS model store. The image carries what
that needs, all from Ubuntu: `podman`, the container engine cephadm uses on
the image (from the snapshot, without Docker's own repository), `catatonit`
for the `--init` its daemons start with, `lvm2` for the OSD's logical
volume, and `ceph-common`, whose `mount.ceph` reads the CephFS client
secret. The node's cluster state comes from its previous root at the disk
install. `ceph-common` adds the `ceph` account at the Debian-allocated
uid 64045, so the image's ownership rule allows any uid or gid of 1000 or
above that the image's own `/etc/passwd` and `/etc/group` define
([`owners.sh`](../tools/hosted_image/owners.sh)). A build user's owner, which
the image does not define, still fails the build, the disk install and the
dry run.

`psmisc` (for `fuser`) and `lsof` are there so that a disk install from the
[in-memory copy](#first-boot-and-bring-up) can show that the target disk is
not open, next to `findmnt` and `losetup`.

## GB10 settings

| File | Setting |
|---|---|
| `etc/default/grub.d/90-sparkpipe-gb10.cfg` | Appends to `GRUB_CMDLINE_LINUX`: the SoC UART as `earlycon`, `init_on_alloc=0`, `iommu.passthrough=0`, `pci=pcie_bus_safe`, `initcall_blacklist=tegra234_cbb_init`, no crash kernel, and `panic=60`, so a failed boot, the initramfs included, reboots instead of waiting at a shell. GRUB runs on the console and the serial line. `GRUB_CMDLINE_LINUX_DEFAULT` and `GRUB_DEFAULT` belong to the baseline's GRUB policy: the serial console `ttyS0` at 921600, no boot-time fsck and the node's recovery address |
| `etc/modprobe.d/sparkpipe-gb10.conf` | nouveau and nova never load; the RTL8127 management port binds `r8127`; `cppc_cpufreq auto_sel_mode=1`; `sbsa_gwdt action=1` |
| `etc/modules-load.d/sparkpipe-gpu.conf` | `nvidia` and `nvidia_uvm` load at boot |
| `etc/sysctl.d/80-sparkpipe-gb10.conf` | `arp_announce=2`, `arp_ignore=1` on all interfaces: each fabric port answers only for its own address |
| `usr/local/sbin/sparkpipe-compact`, `etc/sudoers.d/90-sparkpipe-compact`, `usr/lib/tmpfiles.d/sparkpipe-compact.conf` | the compaction window around a model's activation; see [Contiguous memory](#contiguous-memory) |
| `etc/security/limits.d/90-sparkpipe.conf` | unlimited locked memory for RDMA registration, 500,000 open files (soft and hard) |
| `etc/systemd/sleep.conf.d/90-sparkpipe-no-suspend.conf` | no suspend or hibernation |
| `etc/netplan/01-sparkpipe.yaml` | NetworkManager manages the interfaces; bring-up adds the fleet profiles |

The command line, module options, sysctls and limits are the ones a Spark
runs under the factory OS (DGX OS 7.5.0), read from its configuration and
`/proc/cmdline`.

### Contiguous memory

The nvrm backend allocates device memory in 2 MiB pages through the open
kernel module. In the measurement below, the module's allocations failed
with `NV_ERR_NO_MEMORY` as soon as the free 2 MiB blocks ran out, with more
than 70 GiB still free in smaller pieces: they did not wait for the kernel
to compact. On a GB10 Spark with 115 GiB available, a process held 40 GiB in
every other 4 KiB page of 80 GiB (anonymous, movable memory), and then
802 allocations of 64 MiB (50 GiB) in 2 MiB pages ran while it held them:

| Settings | Free 2 MiB blocks just before the allocation | Allocated |
| --- | --- | --- |
| kernel defaults (proactiveness 20, `min_free_kbytes` 45166) | 6,800 | 205 of 802, then no memory |
| `min_free_kbytes` 1 GiB | 7,569 | 229 of 802, then no memory |
| defaults and one `compact_memory` write | 13,866 | 439 of 802, then no memory |
| proactiveness 90 | 21,965 | 682 of 802, then no memory |
| proactiveness 90 and `min_free_kbytes` 1 GiB, twice | 27,182 and 26,738 | 802 of 802 in 1.1 s, both times |

Kept on while a model serves, those two settings cost decode time. Qwen3.8-27B
on one GB10, with 16 GiB held in every other 4 KiB page of 32 GiB: throughput
and the median step stay within 0.5%, but about 5.5% of steps run about 60 ms
slow (p99 +5.7%), because background compaction keeps working on what it
cannot move. In a 16-rank step the slowest rank sets the pace.

So the image boots with the kernel's defaults and raises both settings only
while a model activates, when the large allocations happen:

- [`sparkpipe-compact`](../tools/hosted_image/files/usr/local/sbin/sparkpipe-compact)
  `raise` records the boot values in `/run/sparkpipe-compact/boot`, writes
  `compact_memory`, then sets proactiveness 90 and `min_free_kbytes` 1 GiB,
  and returns. `compact_memory` goes first because raising proactiveness
  wakes the kernel's background compaction: with the holder below, a
  `compact_memory` written after the raise returned within 0.03 to 0.4 s
  with almost no new blocks in 3 of 5 trials, while written first it ran
  about 5 s and left 36,600 to 37,300 free 2 MiB blocks in 4 of 5. Nothing
  guesses when compaction is done: each nvrm allocation of 2 MiB pages
  waits until the free 2 MiB blocks cover it and the raised reserve, and
  makes missing blocks itself, for up to 60 s
  ([NVRM_BACKEND.md](NVRM_BACKEND.md)). `restore` puts the
  recorded values back, but only when no activation holds
  `/run/sparkpipe-compact/window` (each holds a shared lock on it), so
  activations that overlap share one window.
- [`90-sparkpipe-compact`](../tools/hosted_image/files/etc/sudoers.d/90-sparkpipe-compact)
  lets the `sparkpipe-compact` group run exactly those two commands as root,
  and nothing else; the fleet user joins the group at bring-up.
- The fleet agent ([`fleet_node_agent.sh`](../tools/fleet_node_agent.sh)) opens
  the window in `start_root`, the one place every residentd start passes
  through, before it launches residentd: a watcher in its own session takes
  the shared lock, raises, and restores when residentd reports ready, when it
  is no longer running, or after 900 s, from an `EXIT` trap. If the raise
  fails, the root does not start and the agent says why. Each loop of the
  agent also restores a window no watcher holds, so a killed watcher leaves
  the settings raised only until the agent's next loop. A node without the
  helper opens no window.

The activation phase, from `raise` to the last of the 802 objects, with
40 GiB held in 80 GiB:

| Holder | Phase per trial (s) | NVRM kernel-log lines |
| --- | --- | --- |
| every other 4 KiB page | 6.59, 6.58, 6.77, 6.74, 100.1 | 0 |
| a random half of the 4 KiB pages | 6.68, 6.80, 6.92, 34.90, 6.75 | 0 |

Every trial allocated all 802 objects. The two slow ones are a kernel
limit, which the window accepts rather than works around:
- In those trials `compact_memory` stopped early: after 0.02 and 1.57 s,
  leaving 16,333 and 22,624 free blocks instead of about 37,000.
- Compaction tracepoints show such a run ending with status `contended`
  once its free scanner finds no target pages, and the background
  compaction backs off for about 32 s after it. Until it runs again,
  direct compaction, the allocations' own pull included, makes nothing,
  and objects wait up to 24 s each.
- A second `compact_memory` does not help. Written 2, 4 and 6 s after a
  raise that stopped early, it returned within 0.05 s and added at most one
  block, in both such trials out of eight. So the helper has no verb for
  it.
- The holder is about 25 times harsher than a real activation. A real
  activation needs few 2 MiB blocks: Qwen3.8-27B's only object of 2 MiB
  pages is its 1.9 GB state arena, about 950 blocks, which the pull makes
  in well under a second. Its weights take 64 KiB pages and its KV chunks
  4 KiB pages.

## GPU device nodes

The open module registers its character devices but creates no `/dev`
nodes:

- the GPUs on major 195, region `nvidia`, minors 0 to 247, each GPU's minor
  listed as `Device Minor` in `/proc/driver/nvidia/gpus/<bus>/information`;
- the control device on 195:255, `nvidiactl`;
- UVM on a dynamic major named `nvidia-uvm`, minor 0.

[`71-sparkpipe-gpu.rules`](../tools/hosted_image/files/etc/udev/rules.d/71-sparkpipe-gpu.rules)
runs
[`gpu-device-nodes`](../tools/hosted_image/files/usr/lib/sparkpipe/gpu-device-nodes)
when `nvidia` or `nvidia_uvm` loads and when a GPU binds the driver. It
creates `/dev/nvidiactl`, one `/dev/nvidiaN` per listed GPU and
`/dev/nvidia-uvm`, mode 0666. It fails when a GPU has no minor, or when the
control device is registered and no GPU is listed, except on the events
that can come before the GPU is listed: the `nvidia` module's add event,
which the kernel sends before the module's init probes the GPU, and a GPU's
bind event. On those it makes the control node and succeeds, and
`nvidia_uvm`'s add event, which comes after, makes the GPU's node or fails.

## Node baseline

[`stage_baseline.py`](../tools/hosted_image/stage_baseline.py) runs the
baseline's own `apply` for a staged node inside a chroot of the image, so
the image gets the same bytes `apply` writes.

- **Placed in the image:**
  - the staged assets;
  - the recovery and earlyoom settings, sysctl files and the user slice;
  - the emergency SSH server (checked with the image's `sshd -t`) and the
    recovery key;
  - the baseline's GRUB defaults for a staged node: serial console and fsck
    flags, with no recovery address;
  - the masks of the factory boot units;
  - the enabled units (earlyoom, ssh, emergency ssh, the fsck health timer,
    the serial getty, the firewall, fstrim);
  - the DMA guard: its install step's service and timer, written by the
    guard's own `unit_text` with the baseline's notify address, timer
    enabled, so the guard arms 60 s after every boot;
  - `multi-user.target` as the default;
  - the lossless RoCE settings on `enp1s0f1np1`, which a staged baseline
    leaves out and the stager adds:
    [`sparkpipe_roce_qos.sh`](../tools/fabric/sparkpipe_roce_qos.sh) as
    `/usr/local/sbin/sparkpipe_roce_qos.sh`, its unit
    `sparkpipe-roce-qos.service`, enabled and bound to the port, and the
    DSCP trust helper
    [`sparkpipe_dscp_trust.sh`](../tools/fabric/sparkpipe_dscp_trust.sh) as
    `/usr/local/sbin/sparkpipe-dscp-trust`. They set DSCP trust (the 64-entry
    DSCP>>3 map), PFC on priority 3 and prio-buffer 3:1 with iproute2's
    `dcb` and `ethtool`. Nothing from `mlnx_qos` or other MLNX tools is in
    the image.
- **Left to the live node:** `sysctl --system`, `udevadm`, `modprobe`,
  `update-grub` (the disk install generates the GRUB menu), runtime
  `systemctl` verbs and linger.
- **Left to bring-up:** the node's packages, identity, layout roots, SSH
  password shutdown (it needs the node's login keys) and the initramfs.
  Bring-up's `apply` then adds the node's recovery address to the GRUB
  defaults.

Any other command the baseline runs fails the build, so a new baseline step
is placed by a change here. `baseline-report.json` lists what was applied and
what was left.

## An image node never updates itself

A rented node changes only by a new image, never during a rental. The image
masks the APT timers and their services (`apt-daily`, `apt-daily-upgrade`),
`motd-news`, which fetches news from Ubuntu, and podman's auto-update timer
and its root API socket and service, which cephadm does not use
([`masked-units.txt`](../tools/hosted_image/masked-units.txt),
[`mask_units.sh`](../tools/hosted_image/mask_units.sh)). It carries no
`unattended-upgrades`, and the build fails if a dependency brings it in.

## First boot and bring-up

`sparkpipe-host-keys.service` runs once, before SSH. It generates the node's
SSH and initramfs SSH host keys and rebuilds the initramfs with them. The
image has none.

The image can also boot from a file on another node's filesystem, without
repartitioning, through initramfs-tools' own loop root (`root=UUID=<host
filesystem> loop=<path to the SPARKROOT file> loopfstype=ext4`). That
command line must say `rw`, not `ro`. With `ro`, the initramfs attaches
the loop device read-only, so `/` can never be remounted read-write, and
the units that write state (logind, timesyncd, the DMA guard's boot record)
fail.

With `boot=sparkpipe-ram` added, the image runs from memory instead
([`sparkpipe-ram`](../tools/hosted_image/files/usr/share/initramfs-tools/scripts/sparkpipe-ram)).
The initramfs mounts the file system named by `root=` read-only, copies the
`loop=` file into a tmpfs, unmounts that file system and mounts the copy as
`/`. Nothing on the disk is then mounted, open or mapped, so a
[disk install](#disk-install) can run from the copy against that same disk.
The initramfs also empties the copy's `/etc/fstab`: systemd takes the root
from the kernel command line and mounts nothing else, so a disk that carries
an installed image's `SPARKROOT` and `SPARKESP` labels stays unmounted, as
a rescue boot after an install needs.
The copy takes as much memory as the file's size, and changes made while it
runs are lost at the next boot.

`/etc/fstab` names the root filesystem `SPARKROOT` (ext4) and the EFI
system partition `SPARKESP` (vfat, `nofail`, so a missing partition
does not stop the boot); the [disk install](#disk-install) creates both. The
hostname is `sparkpipe-node` until bring-up. `/etc/default/locale` sets
`LANG=C.UTF-8`, so every login and `sudo` session finds a locale.

Bring-up then follows [NODE_BASELINE.md](NODE_BASELINE.md#replacing-or-adding-a-node):
identity and keys, then `apply`, reboot and `audit`. A node whose disk install
carried no Tailscale state joins the tailnet with
[`tools/tailscale_join.sh`](TAILSCALE_JOIN.md), without a browser login. The modules load under
Secure Boot through the kernel's own signing chain, so a node needs no
enrollment step.

## Disk install

[`install_disk.sh`](../tools/hosted_image/install_disk.sh) installs a built
image onto a disk whose first partition is a vfat EFI system partition and
whose second is an ext4 root. It keeps the root filesystem and its `/home`,
so a node's packs stay where they are. It runs as root from a system that
does not boot from that disk, such as a rescue system in RAM:

```sh
install_disk.sh --disk /dev/nvme0n1 --image BUILD_DIR --identity BUNDLE_DIR --work /run/install
```

Before it changes anything it checks:

- the build directory's `MANIFEST` records a passing dependency audit and
  the archive's SHA-256;
- the identity bundle (below) is well formed and carries no
  `var/lib/tailscale`;
- both partitions have the expected filesystem types and neither is mounted;
- an active firmware boot entry in `BootOrder` loads
  `\EFI\ubuntu\shimaa64.efi` from this disk's EFI partition, matched by its
  GPT partition UUID. The install never adds or changes a boot entry, so that
  entry is what boots the installed image;
- the EFI partition holds `EFI/ubuntu/shimaa64.efi`;
- `e2fsck` finds the root clean;
- nothing else is mounted under the work directory's target;
- the root has 10 GiB free and holds no earlier set-aside directory;
- the old root's `/var/lib/tailscale`, if it has one, is a directory that
  holds only files and directories.

Then it:

1. moves every top-level entry of the old root except `home` and
   `lost+found` into `/.dgxos-<UTC stamp>/root`. These are renames on one
   filesystem, so nothing is copied and the old system can be moved back.
   It then copies the old system's `EFI/ubuntu` from the EFI partition into
   `/.dgxos-<UTC stamp>/esp/EFI/ubuntu` and compares the copy.
2. extracts the image beside `/home`. The image must carry no Tailscale
   state. If the old root has `/var/lib/tailscale`, the installer copies it
   from the set-aside directory into the new root with its owners and modes
   and compares the copy. The node then rejoins the tailnet as itself at its
   first boot, with no login. Otherwise the installer says that the node joins
   at bring-up ([TAILSCALE_JOIN.md](TAILSCALE_JOIN.md)). Then it checks that
   `/` is `0:0 755` and that no path outside `/home` belongs to a uid or gid
   of 1000 or above that the image does not define.
3. labels the partitions `SPARKROOT` and `SPARKESP`, which is what the
   image's `fstab` names.
4. creates the bundle's groups and its user, then installs its files.
5. creates `/boot/efi`, mounts the EFI partition there and records every
   path and file checksum on it outside `EFI/ubuntu`. In a chroot it checks
   `sudoers` with `visudo -c`, then runs `grub-install` for `arm64-efi` with
   Secure Boot, `--no-nvram` and `--no-extra-removable`, which writes shim
   and GRUB into `EFI/ubuntu` on the existing EFI partition. Without
   `--no-extra-removable`, Ubuntu's `grub-install` also copies shim, the
   fallback loader and MokManager into the firmware's removable-media path
   `EFI/BOOT`, replacing the loader the old system left there. Then it runs
   `update-grub` and `update-initramfs` for the pinned kernel.
6. checks that `EFI/ubuntu` holds the shim, GRUB and a `grub.cfg` that
   searches the installed root, that nothing on the EFI partition outside
   `EFI/ubuntu` changed, that `/boot/grub/grub.cfg` boots the pinned
   kernel, and that `efibootmgr -v` prints exactly what it printed at the
   start.

It never deletes anything. Every mount it makes is undone on exit, and
`/sys/firmware/efi/efivars` is never mounted in the chroot.

#### Carrying a node's own state

A fleet node keeps state that no image may hold: its Ceph daemons, private
tools and the secrets of its networks. `--carry LIST` carries it from the
previous root, the same way as the Tailscale state:

- Each line of `LIST` is a path below `/`, with `*` allowed. It never names
  `home`, `lost+found`, `var/lib/tailscale` or a set-aside directory.
- Before anything changes, the installer expands the list in the previous
  root and checks that the root has 10 GiB free beyond the matches' size.
- The extraction leaves the matches out of the image, and the installer
  copies them from `/.dgxos-<stamp>/root` with their owners, modes, links,
  extended attributes and special files: podman's layers hold whiteouts as
  character devices `0:0`, and a container's run directory holds pipes and a
  socket. It compares each copy by its listing (type, owners, mode, size,
  link target, device numbers) and the checksum of every regular file, never
  reading a pipe, and names it in its log. The copies then take the image's
  place, so nothing of the image's is deleted either.
- `--carry-accounts NAME,...` carries accounts the same way: each one's
  `passwd` and `shadow` lines and its primary group, after the bundle's user.
  A name, uid or group that the installed root already has is refused.

A fleet node's list is [`carry-fleet.txt`](../tools/hosted_image/carry-fleet.txt):

- its Ceph state: `/var/lib/ceph`, `/etc/ceph` with the CephFS client
  secret, the podman store with the pinned Ceph image, cephadm's units,
  sysctl and logrotate files, and `/var/log/ceph`;
- the private external-NVMe link that its OSD's units require, and the
  private uplink failover tool, with their units and configuration;
- its NetworkManager profiles in `/etc/netplan/90-NM-*.yaml` and its
  `/etc/hosts`, which names the fleet's nodes;
- its SSH and initramfs host keys and root's `authorized_keys`;
- the emergency account's sudo rule.

It takes `--carry-accounts sparkemerg` for the emergency account. The
private files come only from the node's own previous root, never from the
repository or a bundle.

Rollback, from the same kind of rescue system: move every top-level entry
of the installed root except `home`, `lost+found` and `.dgxos-<stamp>` into
`/.hosted-<stamp>`, move the entries of `/.dgxos-<stamp>/root` back to `/`,
move the EFI partition's `EFI/ubuntu` into `/.hosted-<stamp>/esp/EFI`,
copy `/.dgxos-<stamp>/esp/EFI/ubuntu` back onto the EFI partition and
compare the whole partition with what it held before the install, restore
the old labels, and remove `/.dgxos-<stamp>`, which by then holds only that
copy. The old loader comes back byte for byte, and `EFI/BOOT` and the
firmware boot entries were never changed, so rollback runs no
`grub-install`.

The set-aside directory is removed only after the node passes acceptance.
After that, `hosted_dependency_audit.py image --rootfs /` over the whole
installed root, with no exclusions, is the gate.

### Dry run

[`dryrun_install_disk.sh`](../tools/hosted_image/dryrun_install_disk.sh)
is the behavioural test of `install_disk.sh`, and every change to the
installer runs it:

```sh
sudo make hosted-disk-dryrun IMAGE=BUILD_DIR DIR=NEW_DIR
```

It needs root, loop device support and 40 GiB free beside `NEW_DIR`. It
touches no real disk, uses no GPU, and never writes the host's firmware
boot entries. It creates a 24 GiB sparse disk image under `NEW_DIR`,
attaches it as a loop device and partitions it like a node, then seeds:

- a previous system with Tailscale state in `/var/lib/tailscale`, and a
  fleet node's state for every line of `carry-fleet.txt` (Ceph state and
  units with their wants links and a mask, the private tools, profiles, a container layer with a
  whiteout, a container run directory with a pipe and a socket) and the
  `sparkemerg` account;
- `/home/node` with a 256 MiB pack and a release root linked to a stub
  `libcuda.so.1`;
- an EFI partition with `EFI/ubuntu`, `EFI/BOOT` and another vendor's
  directory;
- an identity bundle.

The installer reads firmware boot entries through a wrapper that prints
the host's `efibootmgr -v` plus one active entry in `BootOrder` that loads
`\EFI\ubuntu\shimaa64.efi` from the loop device's EFI partition. The
wrapper accepts only `-v`. Then the harness runs, in order:

1. the installer with the host's real boot entries, which must refuse
   because no entry loads this EFI partition;
2. an install, then the checks:
   - `/home` is unchanged (owners, modes, sizes, mtimes and the pack's
     checksum);
   - the old root and the old `EFI/ubuntu` are intact in the set-aside
     directory;
   - labels, identity, user, groups, modes and `fstab` are as expected;
   - `/var/lib/tailscale` is the previous root's, with the same entries,
     owners, modes and checksums;
   - with `--carry carry-fleet.txt --carry-accounts sparkemerg`, every
     carried path is the previous root's (types, owners, modes, sizes, link
     targets and contents) and the account's `passwd`, `shadow`, `group` and
     `gshadow` lines are its;
   - nothing outside `/home` belongs to a uid or gid of 1000 or above that
     the image does not define;
   - the initramfs was rebuilt with the dropbear key;
   - the kernel entry uses the root UUID and the GB10 arguments;
   - `EFI/ubuntu` holds the image's own signed shim and GRUB;
   - everything outside `EFI/ubuntu` on the EFI partition is byte-identical;
3. the rollback described above, checked against the old root and the
   whole old EFI partition;
4. the installer refusing while `/.hosted-<stamp>` exists;
5. a second install, after the previous root's Tailscale state is
   removed and with no carry list, with the same checks, except that the
   installed root must hold no Tailscale state, no carried state and no
   carried account, and the installer must say the node joins at bring-up;
6. the acceptance gate: after the set-aside directory is deleted, the
   image audit over the installed root fails on the stub-linked release
   root in `/home`, then passes once that root is removed;
7. a check that the host's `efibootmgr -v` is unchanged.

Before it removes anything it checks that only its own mount is under
`NEW_DIR`. It detaches the loop device on exit, prints what is still
mounted or attached, and writes its exit status to `NEW_DIR/exit`.

### Identity bundle

The identity bundle is a directory with three parts:

- `files/` mirrors absolute paths in the installed root, for example
  `files/etc/hostname`. Each file is installed owned by root with its mode
  from the bundle. A directory the image lacks is created with its mode from
  the bundle; a directory the image already has keeps the image's mode.
  `files/` may hold only regular files and directories, and never
  `var/lib/tailscale`: a node's Tailscale state comes only from its previous
  root. A bundle that
  carries `/etc/ssh/ssh_host_ed25519_key` must also carry
  `/etc/dropbear/initramfs/dropbear_ed25519_host_key`: with SSH host keys
  present, `sparkpipe-host-keys.service` does not run, so nothing else would
  create the initramfs keys.
- `user.conf` holds `USER_NAME`, `USER_UID`, `USER_GID`, `USER_GROUPS` (a
  comma list) and `USER_SHELL`, one `KEY=value` per line. On a fleet node
  `USER_GROUPS` includes `sparkpipe-compact`, so the agent can open the
  [compaction window](#contiguous-memory). It is read as data,
  never sourced. The user is created with no home of its own, because
  `/home/<user>` is already on the disk.
- `groups.conf` holds one `name:gid` line per group the user needs that
  the image does not have.
