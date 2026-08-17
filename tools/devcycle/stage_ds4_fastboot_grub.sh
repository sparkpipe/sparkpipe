#!/usr/bin/env bash
#
# stage_ds4_fastboot_grub.sh - pre-stage a ONE-SHOT "skip fsck" GRUB entry.
#
# Run on a host AFTER it is reachable (playbook stage 4 = full login). Idempotent
# and safe to re-run; it only ADDS a menu entry, it never touches the default.
#
# Why: after the permanent fix (nofail + fs_passno=0 via tools/devcycle/
# ds4_fastboot_fix.sh) lands, a dirty data NVMe can no longer block boot. This
# entry is belt-and-suspenders for the cases the permanent fix does not cover
# (e.g. a dirty root or a mount that still fsck's): it lets a WEDGED host be
# broken by smart plug alone.
#
# Mechanism: GRUB boots the default entry normally. To make the NEXT boot skip
# fsck, pre-arm while the host is still healthy:
#
#     sudo grub-reboot ds4-fastboot      # sets next_entry in /boot/grub/grubenv
#
# Then the next power cycle (smart plug) boots the ds4-fastboot entry, which
# carries fsck.mode=skip fsck.repair=no, so the host reaches login; the boot
# after that automatically reverts to the normal default. Run a manual fsck on
# the skipped volume while it is up, then re-arm / clear as needed.
#
# Usage:
#     sudo ./stage_ds4_fastboot_grub.sh            # install the entry + update-grub
#     sudo ./stage_ds4_fastboot_grub.sh --check    # report status, change nothing
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

case "$1" in
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
        exit 0
        ;;
    "")
        ;;
    *)
        die "unknown argument '$1' (use --check)"
        ;;
esac

# ---- install ---------------------------------------------------------------
umask 022
$SUDO tee "$SNIPPET" >/dev/null <<SNIP
#!/bin/sh
cat <<'EOF'
menuentry 'ds4-fastboot (skip fsck, one-shot)' --id $ENTRY_ID {
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

if command -v update-grub >/dev/null 2>&1; then
    $SUDO update-grub
else
    $SUDO grub-mkconfig -o /boot/grub/grub.cfg
fi

# ---- verify ----------------------------------------------------------------
if ! grep -qs "ds4-fastboot" /boot/grub/grub.cfg; then
    die "update-grub completed but the ds4-fastboot entry is missing from /boot/grub/grub.cfg"
fi

cat <<MSG

stage_ds4_fastboot_grub: OK.
  entry:   $ENTRY_ID  (in $SNIPPET, baked into /boot/grub/grub.cfg)
  kernel:  $KERNEL  root=UUID=$ROOT_UUID
  boot fs: $BOOT_FS  uuid=$BOOT_UUID

To break a FUTURE wedge by smart plug alone, pre-arm while the host is healthy:
    sudo grub-reboot $ENTRY_ID
then power-cycle. The next boot skips fsck; the boot after reverts to normal.
Verify arming with: sudo grub-editenv list   (expect next_entry=$ENTRY_ID)
MSG
