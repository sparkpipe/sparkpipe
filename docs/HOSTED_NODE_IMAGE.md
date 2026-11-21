# Hosted node image

Rented nodes boot one minimal image. It contains:

- Ubuntu 24.04 from a pinned archive snapshot, with Canonical's `linux-nvidia` 7.0 kernel;
- NVIDIA's open GPU kernel module, built from source at the version the device backend pins, and its GSP firmware;
- the in-tree `mlx5_core` and `r8127` drivers, with Ubuntu's rdma-core;
- the node-independent part of the [node baseline](NODE_BASELINE.md);
- a release root and the node agent.

The image carries no NVIDIA user-space package. The build fails unless the
[dependency audit](HOSTED_DEPENDENCY_AUDIT.md) passes on the finished root
filesystem and on its release root.

## Build

```bash
sudo tools/hosted_image/build_image.sh --out ~/nvfree/image/build-N \
  --source-ref HEAD --release-root RELEASE_ROOT \
  --firewall-ruleset spark-firewall.nft \
  --module-key module-key.pem --module-cert module-cert.pem
```

It runs as root on an arm64 Ubuntu host from a clean checkout at
`--source-ref`. No GPU is used, and no package, module or setting is
changed on the build host. All six inputs are required:

- `--release-root` must pass the root audit before anything is built.
- `--firewall-ruleset` is the private host firewall ruleset, so the firewall is active from the first boot.
- `--module-key` (PEM) and `--module-cert` (PEM) sign the kernel modules.

The steps:

1. Download mmdebstrap `MMDEBSTRAP_VERSION` from the snapshot, check its
   sha256, and run it from the unpacked package.
2. Clone the open module at `OPEN_MODULE_TAG` and check that it is commit
   `OPEN_MODULE_COMMIT`.
3. Build a chroot from `build-packages.txt`. In it, build the modules
   against `linux-headers-KERNEL_RELEASE`, then sign `nvidia.ko` and
   `nvidia-uvm.ko` with the kernel's `sign-file`. The chroot is then
   deleted, and the key with it.
4. Build the image from `image-packages.txt` with
   [`customize_image.sh`](../tools/hosted_image/customize_image.sh), which:
   - copies [`files/`](../tools/hosted_image/files/) and the firewall
     ruleset;
   - installs the signed modules under
     `/lib/modules/KERNEL_RELEASE/updates/sparkpipe/` and checks that
     `depmod` resolves `nvidia` to them;
   - checks the GSP firmware for the module's version;
   - copies the release root to `/opt/sparkpipe/release` and the agent to
     `/opt/sparkpipe/core/bin/fleet_node_agent.sh`;
   - stages the baseline (below) and enables the host-key unit;
   - removes every SSH and initramfs host key;
   - rebuilds the initramfs.
5. Run the image audit. On a failure, the root filesystem is kept for
   inspection and nothing is packed.
6. Write `rootfs.tar.zst` (sorted names, mtimes clamped to the snapshot
   time), `MANIFEST`, `audit.json`, `baseline-report.json`,
   `module-cert.der` and the build logs.

`MANIFEST` records:

- the source commit and every pin;
- the module signer's SHA-256 fingerprint;
- the archive's sha256 and the audit verdict;
- every release-root file with its sha256;
- every installed package with its version.

## Pins

[`pins.conf`](../tools/hosted_image/pins.conf) holds:

- the snapshot (`https://snapshot.ubuntu.com/ubuntu/<time>`, which also
  serves arm64) and `SOURCE_DATE_EPOCH`, its time;
- the kernel release;
- mmdebstrap's version and sha256;
- the open module's repository, tag and commit;
- the firmware package;
- the install paths.

[`image-packages.txt`](../tools/hosted_image/image-packages.txt) and
[`build-packages.txt`](../tools/hosted_image/build-packages.txt) name every
top-level package as `name=version`. Each version is the highest the
snapshot holds across `noble`, `noble-updates` and `noble-security`, by
`dpkg --compare-versions`; the snapshot fixes every dependency. A pin
changes only together with the snapshot.

## GB10 settings

| File | Setting |
|---|---|
| `etc/default/grub.d/90-sparkpipe-gb10.cfg` | Serial console `ttyS0` at 921600 (the baseline's getty line) with the SoC UART as `earlycon`, `init_on_alloc=0`, `iommu.passthrough=1`, `pci=pcie_bus_safe`, `initcall_blacklist=tegra234_cbb_init,power_wrap_init`, `kho=off`, no crash kernel, no boot-time fsck (the baseline's fsck health timer checks the disks); GRUB on the console and the serial line |
| `etc/modprobe.d/sparkpipe-gb10.conf` | nouveau and nova never load; the RTL8127 management port binds `r8127`; `cppc_cpufreq auto_sel_mode=1`; `sbsa_gwdt action=1` |
| `etc/modules-load.d/sparkpipe-gpu.conf` | `nvidia` and `nvidia_uvm` load at boot |
| `etc/sysctl.d/80-sparkpipe-gb10.conf` | `arp_announce=2`, `arp_ignore=1` on all interfaces: each fabric port answers only for its own address |
| `etc/security/limits.d/90-sparkpipe.conf` | unlimited locked memory for RDMA registration, 500,000 open files |
| `etc/systemd/sleep.conf.d/90-sparkpipe-no-suspend.conf` | no suspend or hibernation |
| `etc/netplan/01-sparkpipe.yaml` | NetworkManager manages the interfaces; bring-up adds the fleet profiles |

The command line and module options are the ones GB10 nodes run under the
factory OS, read from a GB10 workstation. Compare them with a Spark's
`/proc/cmdline` when the image is first proven on one.

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
  - the masks of the factory boot units;
  - the enabled units (earlyoom, ssh, emergency ssh, the fsck health timer,
    the serial getty, the firewall, fstrim);
  - `multi-user.target` as the default.
- **Left to the live node:** `sysctl --system`, `udevadm`, `modprobe`,
  runtime `systemctl` verbs, linger and the DMA guard install, which
  reports to the hub.
- **Left to bring-up:** the node's packages, identity, layout roots, SSH
  password shutdown (it needs the node's login keys), the GRUB recovery
  address and the initramfs.

Any other command the baseline runs fails the build, so a new baseline step
is placed by a change here. `baseline-report.json` lists what was applied and
what was left.

## First boot and bring-up

`sparkpipe-host-keys.service` runs once, before SSH. It generates the node's
SSH and initramfs SSH host keys and rebuilds the initramfs with them. The
image has none.

`/etc/fstab` names the root filesystem `SPARKROOT` (ext4) and the EFI
system partition `SPARKESP` (vfat); the disk install creates both. The
hostname is `sparkpipe-node` until bring-up.

Bring-up then follows [NODE_BASELINE.md](NODE_BASELINE.md#replacing-or-adding-a-node):

1. identity and keys;
2. `apply`, reboot and `audit`;
3. with Secure Boot on, enroll the module certificate once:
   `mokutil --import module-cert.der`, then confirm in MokManager on the
   console at the next boot.
