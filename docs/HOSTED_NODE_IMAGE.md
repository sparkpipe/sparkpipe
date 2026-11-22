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

## GB10 settings

| File | Setting |
|---|---|
| `etc/default/grub.d/90-sparkpipe-gb10.cfg` | Appends to `GRUB_CMDLINE_LINUX`: the SoC UART as `earlycon`, `init_on_alloc=0`, `iommu.passthrough=0`, `pci=pcie_bus_safe`, `initcall_blacklist=tegra234_cbb_init`, no crash kernel, and `panic=60`, so a failed boot, the initramfs included, reboots instead of waiting at a shell. GRUB runs on the console and the serial line. `GRUB_CMDLINE_LINUX_DEFAULT` and `GRUB_DEFAULT` belong to the baseline's GRUB policy: the serial console `ttyS0` at 921600, no boot-time fsck and the node's recovery address |
| `etc/modprobe.d/sparkpipe-gb10.conf` | nouveau and nova never load; the RTL8127 management port binds `r8127`; `cppc_cpufreq auto_sel_mode=1`; `sbsa_gwdt action=1` |
| `etc/modules-load.d/sparkpipe-gpu.conf` | `nvidia` and `nvidia_uvm` load at boot |
| `etc/sysctl.d/80-sparkpipe-gb10.conf` | `arp_announce=2`, `arp_ignore=1` on all interfaces: each fabric port answers only for its own address |
| `etc/security/limits.d/90-sparkpipe.conf` | unlimited locked memory for RDMA registration, 500,000 open files (soft and hard) |
| `etc/systemd/sleep.conf.d/90-sparkpipe-no-suspend.conf` | no suspend or hibernation |
| `etc/netplan/01-sparkpipe.yaml` | NetworkManager manages the interfaces; bring-up adds the fleet profiles |

The command line, module options, sysctls and limits are the ones a Spark
runs under the factory OS (DGX OS 7.5.0), read from its configuration and
`/proc/cmdline`.

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
`/dev/nvidia-uvm`, mode 0666. It fails when the control device is
registered and no GPU is listed, or a GPU has no minor.

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
  - `multi-user.target` as the default.
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
masks the APT timers and their services (`apt-daily`, `apt-daily-upgrade`)
and `motd-news`, which fetches news from Ubuntu
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

`/etc/fstab` names the root filesystem `SPARKROOT` (ext4) and the EFI
system partition `SPARKESP` (vfat, `nofail`, so a missing partition
does not stop the boot); the [disk install](#disk-install) creates both. The
hostname is `sparkpipe-node` until bring-up.

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
   of 1000 or above.
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

- a previous system with Tailscale state in `/var/lib/tailscale`;
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
   - nothing outside `/home` belongs to a uid or gid of 1000 or above;
   - the initramfs was rebuilt with the dropbear key;
   - the kernel entry uses the root UUID and the GB10 arguments;
   - `EFI/ubuntu` holds the image's own signed shim and GRUB;
   - everything outside `EFI/ubuntu` on the EFI partition is byte-identical;
3. the rollback described above, checked against the old root and the
   whole old EFI partition;
4. the installer refusing while `/.hosted-<stamp>` exists;
5. a second install, after the previous root's Tailscale state is
   removed, with the same checks, except that the installed root must hold no
   Tailscale state and the installer must say the node joins at bring-up;
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
  comma list) and `USER_SHELL`, one `KEY=value` per line. It is read as data,
  never sourced. The user is created with no home of its own, because
  `/home/<user>` is already on the disk.
- `groups.conf` holds one `name:gid` line per group the user needs that
  the image does not have.
