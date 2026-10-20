#!/bin/bash
set -euo pipefail

readonly FSID="cd8715a1-47b9-4e4a-8d3c-2cddb02077d1"
readonly TARGET="ceph-${FSID}.target"
readonly TEMPLATE="ceph-${FSID}@.service"
readonly TARGET_DROPIN="/etc/systemd/system/${TARGET}.d/20-ds4-start-after-ssh.conf"
readonly TEMPLATE_DROPIN="/etc/systemd/system/${TEMPLATE}.d/20-ds4-start-after-ssh.conf"
readonly OLD_TARGET_DROPIN="/etc/systemd/system/${TARGET}.d/10-ds4-warm-guard.conf"
readonly OLD_TEMPLATE_DROPIN="/etc/systemd/system/${TEMPLATE}.d/10-ds4-warm-guard.conf"
readonly MOUNTPOINT="/mnt/model-warm"
readonly MOUNT_SOURCE="10.10.100.10:6789,10.10.100.17:6789,10.10.100.25:6789:/models"
readonly CLIENT_CONFIG="/etc/ceph/ceph.conf"
readonly CLIENT_SECRET="/etc/ceph/ds4warm.secret"

install_startup_policy()
{
    local target_tmp template_tmp
    target_tmp="$(mktemp)"
    template_tmp="$(mktemp)"
    trap 'rm -f "$target_tmp" "$template_tmp"' RETURN
    printf '%s\n' \
        '[Unit]' \
        'After=network-online.target ssh.service' \
        'Wants=network-online.target ssh.service' \
        'JobTimeoutSec=180' >"$target_tmp"
    printf '%s\n' \
        '[Unit]' \
        'After=network-online.target local-fs.target time-sync.target docker.service ssh.service' \
        'Wants=network-online.target local-fs.target time-sync.target docker.service ssh.service' \
        '' \
        '[Service]' \
        'TimeoutStartSec=180' \
        'TimeoutStopSec=180' >"$template_tmp"
    sudo -n install -d -m 0755 "$(dirname "$TARGET_DROPIN")" "$(dirname "$TEMPLATE_DROPIN")"
    sudo -n install -m 0644 "$target_tmp" "$TARGET_DROPIN"
    sudo -n install -m 0644 "$template_tmp" "$TEMPLATE_DROPIN"
    sudo -n rm -f "$OLD_TARGET_DROPIN" "$OLD_TEMPLATE_DROPIN"
    sudo -n systemctl daemon-reload
    sudo -n systemctl enable ssh.service "$TARGET" >/dev/null
}

start_cluster()
{
    install_startup_policy
    sudo -n systemctl start "$TARGET"
}

stop_cluster()
{
    unmount_storage
    sudo -n systemctl stop "$TARGET" >/dev/null 2>&1 || true
}

mount_storage()
{
    local options source type
    if mountpoint -q "$MOUNTPOINT"; then
        source="$(findmnt -rn -M "$MOUNTPOINT" -o SOURCE)"
        type="$(findmnt -rn -M "$MOUNTPOINT" -o FSTYPE)"
        options="$(findmnt -rn -M "$MOUNTPOINT" -o OPTIONS)"
        if [ "$source" != "$MOUNT_SOURCE" ] || [ "$type" != "ceph" ]; then
            printf 'unexpected mount at %s: source=%s type=%s\n' "$MOUNTPOINT" "$source" "$type" >&2
            exit 1
        fi
        case ",$options," in
            *",fsid=${FSID},"*)
                ;;
            *)
                printf 'unexpected Ceph FSID at %s: options=%s\n' "$MOUNTPOINT" "$options" >&2
                exit 1
                ;;
        esac
        return
    fi
    if [ ! -s "$CLIENT_CONFIG" ]; then
        printf 'missing Ceph client config: %s\n' "$CLIENT_CONFIG" >&2
        exit 1
    fi
    if [ ! -s "$CLIENT_SECRET" ]; then
        printf 'missing CephFS client secret: %s\n' "$CLIENT_SECRET" >&2
        exit 1
    fi
    sudo -n install -d -m 0755 "$MOUNTPOINT"
    sudo -n mount -t ceph "$MOUNT_SOURCE" "$MOUNTPOINT" \
        -o name=ds4warm,secretfile="$CLIENT_SECRET",fs=ds4warm,noatime
}

unmount_storage()
{
    if mountpoint -q "$MOUNTPOINT"; then
        sudo -n umount "$MOUNTPOINT"
    fi
}

show_status()
{
    printf 'fsid=%s target_enabled=%s target_active=%s\n' \
        "$FSID" \
        "$(systemctl is-enabled "$TARGET" 2>/dev/null || true)" \
        "$(systemctl is-active "$TARGET" 2>/dev/null || true)"
    printf 'mount=%s\n' "$(findmnt -rn -M "$MOUNTPOINT" -o SOURCE,FSTYPE 2>/dev/null || printf absent)"
    systemctl list-units --no-legend "ceph-${FSID}@*.service" || true
}

case "${1:-}" in
    guard|arm)
        install_startup_policy
        ;;
    start)
        start_cluster
        ;;
    stop)
        stop_cluster
        ;;
    mount)
        mount_storage
        ;;
    unmount)
        unmount_storage
        ;;
    status)
        show_status
        ;;
    *)
        printf 'usage: %s {guard|arm|start|stop|mount|unmount|status}\n' "$0" >&2
        exit 2
        ;;
esac
