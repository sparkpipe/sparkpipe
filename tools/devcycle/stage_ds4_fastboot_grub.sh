#!/usr/bin/env bash
#
# stage_ds4_fastboot_grub.sh - pre-stage a "skip fsck" GRUB entry.
#
# Run on a host AFTER it is reachable (playbook stage 4 = full login). Idempotent
# and safe to re-run. DEFAULT mode makes the skip-fsck entry the GRUB default:
# every boot skips ALL fsck (fsck.mode=skip fsck.repair=no) including root, so a
# dirty-root wedge can never block boot again - no per-boot re-arming needed.
# Manual fsck moves to scheduled maintenance (run it while the host is up).
#
# Optional ONE-SHOT mode keeps the normal default and only arms the NEXT boot
# via grub-reboot (grubenv next_entry); the boot after reverts to normal. Use it
# only where skipping fsck on every boot is undesirable.
#
# Usage:
#     sudo ./stage_ds4_fastboot_grub.sh               # install entry AS DEFAULT + update-grub
#     sudo ./stage_ds4_fastboot_grub.sh --one-shot    # install entry, arm next boot only
#     sudo ./stage_ds4_fastboot_grub.sh --check       # report status, change nothing
set -euo pipefail

SUDO=""
if [[ $(id -u) -ne 0 ]]; then SUDO="sudo"; fi

SNIPPET=/etc/grub.d/40_ds4_fastboot
ENTRY_ID=ds4-fastboot
KVER="$(uname -r)"
KERNEL="/vmlinuz-$KVER"
INITRD="/initrd.img-$KVER"

die() { echo "stage_ds4_fastboot_grub: ERROR: $*" >&2; exit 1; }

# ---- preflight -------------------------------------------------------------
command -v grub-probe >/dev/null 2>&1 || die "grub-probe not found (not a GRUB host?)"
command -v update-grub >/dev/null 2>&1 || command -v grub-mkconfig >/dev/null 2>&1 \
    || die "update-grub / grub-mkconfig not found"

BOOT_UUID="$($SUDO grub-probe --target=fs_uuid /boot)"
ROOT_UUID="$($SUDO grub-probe --target=fs_uuid /)"
BOOT_FS="$($SUDO grub-probe --target=fs /boot)"

FSMOD=ext2
case "$BOOT_FS" in
    xfs)   FSMOD=xfs ;;
    btrfs) FSMOD=btrfs ;;
esac

if [[ ! -r "/boot/vmlinuz-$KVER" && ! -r "/vmlinuz-$KVER" ]]; then
    die "kernel $KVER image not found under /boot or /"
fi
[[ -r "/boot/initrd.img-$KVER" || -r "/initrd.img-$KVER" ]] \
    || echo "stage_ds4_fastboot_grub: WARNING: initrd for $KVER missing; entry may not boot" >&2

# GRUB linux/initrd paths are relative to the boot filesystem. If /boot is
# not a separate filesystem (same UUID as root), the /boot prefix is needed.
if [[ "$BOOT_UUID" == "$ROOT_UUID" ]]; then
    KERNEL="/boot/vmlinuz-$KVER"
    INITRD="/boot/initrd.img-$KVER"
fi

case "${1:-}" in
    --check)
        if [[ -x "$SNIPPET" ]]; then
            echo "present: $SNIPPET"
            grep -n -e "menuentry" -e "fsck.mode" "$SNIPPET" || true
        else
            echo "absent:  $SNIPPET"
        fi
        if grep -qs "ds4-fastboot" /boot/grub/grub.cfg; then
            echo "grub.cfg: entry installed"
        else
            echo "grub.cfg: entry NOT installed (run without --check)"
        fi
        if grep -qs 'set default="ds4-fastboot"' /boot/grub/grub.cfg; then
            echo "grub.cfg: ds4-fastboot is the DEFAULT (no re-arm needed)"
        else
            echo "grub.cfg: ds4-fastboot is NOT the default"
        fi
        exit 0
        ;;
    --one-shot) MODE=one-shot ;;
    --default)  MODE=default ;;
    "")         MODE=default ;;
    *)
        die "unknown argument '$1' (use --check | --one-shot | --default)"
        ;;
esac

# ---- install ---------------------------------------------------------------
umask 022
$SUDO tee "$SNIPPET" >/dev/null <<SNIP
#!/bin/sh
cat <<'EOF'
menuentry 'ds4-fastboot (skip fsck)' --id $ENTRY_ID {
    load_video
    insmod gzio
    insmod part_gpt
    insmod $FSMOD
    search --no-floppy --fs-uuid --set=root $BOOT_UUID
    linux   $KERNEL root=UUID=$ROOT_UUID ro fsck.mode=skip fsck.repair=no
    initrd  $INITRD
}
EOF
SNIP
$SUDO chmod 0755 "$SNIPPET"

if [[ "$MODE" == "default" ]]; then
    if grep -qs '^GRUB_DEFAULT=' /etc/default/grub; then
        $SUDO sed -i "s|^GRUB_DEFAULT=.*|GRUB_DEFAULT=\"ds4-fastboot\"|" /etc/default/grub
    else
        echo 'GRUB_DEFAULT="ds4-fastboot"' | $SUDO tee -a /etc/default/grub >/dev/null
    fi
fi

if command -v update-grub >/dev/null 2>&1; then
    $SUDO update-grub
else
    $SUDO grub-mkconfig -o /boot/grub/grub.cfg
fi

# ---- verify ----------------------------------------------------------------
if ! grep -qs "ds4-fastboot" /boot/grub/grub.cfg; then
    die "update-grub completed but the ds4-fastboot entry is missing from /boot/grub/grub.cfg"
fi
if [[ "$MODE" == "default" ]]; then
    if ! grep -qs 'set default="ds4-fastboot"' /boot/grub/grub.cfg; then
        die "GRUB_DEFAULT set but grub.cfg does not boot ds4-fastboot by default"
    fi
else
    $SUDO grub-reboot "$ENTRY_ID"
    if ! $SUDO grub-editenv list | grep -qs "next_entry=$ENTRY_ID"; then
        die "grub-reboot failed: next_entry is not $ENTRY_ID"
    fi
fi

if [[ "$MODE" == "default" ]]; then
cat <<MSG

stage_ds4_fastboot_grub: OK (DEFAULT mode).
  entry:   $ENTRY_ID  (in $SNIPPET, baked into /boot/grub/grub.cfg)
  default: ds4-fastboot - EVERY boot skips fsck (no re-arming needed)
  kernel:  $KERNEL  root=UUID=$ROOT_UUID
  boot fs: $BOOT_FS  uuid=$BOOT_UUID

Run manual fsck during maintenance windows:
    sudo fsck -f /dev/nvme0n1p2   (root; do it from a rescue/live env or when
                                   the filesystem can be unmounted - never on a
                                   mounted rw root)
MSG
else
cat <<MSG

stage_ds4_fastboot_grub: OK (ONE-SHOT mode).
  entry:   $ENTRY_ID  (in $SNIPPET, baked into /boot/grub/grub.cfg)
  armed:   next_entry=$ENTRY_ID - the NEXT boot skips fsck, then reverts
  kernel:  $KERNEL  root=UUID=$ROOT_UUID
  boot fs: $BOOT_FS  uuid=$BOOT_UUID

Re-arm after every boot: sudo grub-reboot $ENTRY_ID
Verify arming with: sudo grub-editenv list   (expect next_entry=$ENTRY_ID)
MSG
fi
