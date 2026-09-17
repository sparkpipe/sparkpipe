#!/usr/bin/env bash
set -euo pipefail

FABRIC_DEVICE="${DS4_SWITCHED_FABRIC_DEVICE:-enp1s0f1np1}"
FABRIC_PREFIX="${DS4_SWITCHED_FABRIC_PREFIX:-10.10.100}"
FABRIC_MTU="${DS4_SWITCHED_FABRIC_MTU:-9000}"
CX7_HOTPLUG_MARKER="${DS4_CX7_HOTPLUG_MARKER:-/etc/nvidia/cx7-hotplug-enabled}"
CX7_HOTPLUG_HANDLER="${DS4_CX7_HOTPLUG_HANDLER:-/opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh}"
FLEET_SYSCTL_PATH="${DS4_FLEET_SYSCTL_PATH:-/etc/sysctl.d/99-ds4-fleet.conf}"

fabric_devices()
{
    printf '%s\n' \
        enp1s0f0np0 \
        enp1s0f1np1 \
        enP2p1s0f0np0 \
        enP2p1s0f1np1 \
        enP2p1 \
        enP2p2
}

fabric_primary_devices()
{
    printf '%s\n' \
        enp1s0f0np0 \
        enp1s0f1np1
}

disable_cx7_hotplug_power_saving()
{
    rm -f "${CX7_HOTPLUG_MARKER}"
    if [ -x "${CX7_HOTPLUG_HANDLER}" ]; then
        "${CX7_HOTPLUG_HANDLER}" boot >/dev/null
    fi
}

select_fabric_device()
{
    local attempt carrier device
    local linked=()
    if [ -n "${FABRIC_DEVICE}" ]; then
        if [ ! -e "/sys/class/net/${FABRIC_DEVICE}" ]; then
            printf 'configured fabric device is missing: %s\n' "${FABRIC_DEVICE}" >&2
            return 3
        fi
        printf '%s\n' "${FABRIC_DEVICE}"
        return 0
    fi
    while read -r device; do
        [ -e "/sys/class/net/${device}" ] || continue
        ip link set dev "${device}" up 2>/dev/null || true
    done < <(fabric_primary_devices)
    for attempt in $(seq 1 30); do
        linked=()
        while read -r device; do
            [ -r "/sys/class/net/${device}/carrier" ] || continue
            carrier="$(cat "/sys/class/net/${device}/carrier" 2>/dev/null || true)"
            if [ "${carrier}" = "1" ]; then
                linked+=("${device}")
            fi
        done < <(fabric_primary_devices)
        if [ "${#linked[@]}" -eq 1 ]; then
            printf '%s\n' "${linked[0]}"
            return 0
        fi
        if [ "${#linked[@]}" -gt 1 ]; then
            printf 'multiple fabric devices have carrier: %s\n' "${linked[*]}" >&2
            return 4
        fi
        sleep 0.5
    done
    printf 'no Spark fabric device acquired carrier\n' >&2
    return 5
}

node_name()
{
    hostname -s
}

node_rank()
{
    if [ ! -r /etc/ds4-node-rank ]; then
        printf 'missing canonical rank file: /etc/ds4-node-rank\n' >&2
        return 2
    fi
    tr -cd '0-9' < /etc/ds4-node-rank
    printf '\n'
}

install_service()
{
    install -m 0755 "$0" /usr/local/sbin/ds4-switched-fabric-apply
    printf '%s\n' '--install only installs the runtime; use ds4_spark_brickproof.py to install the service and timer' >&2
}

backup_and_remove()
{
    local path backup_root
    path="$1"
    [ -e "${path}" ] || return 0
    backup_root="/var/backups/ds4-fleet/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "${backup_root}$(dirname "${path}")"
    cp -a "${path}" "${backup_root}${path}"
    rm -rf "${path}"
}

retire_legacy_units()
{
    local unit path
    for unit in \
        ds4-ring-200g.service \
        ds4-ring-control-iface.service \
        ds4-10g-client-gateway.service \
        ds4-10g-nat-gateway.service \
        ds4-internet-client.service \
        ds4-internet-client.timer \
        ds4-internet-gateway.service \
        ds4-internet-gateway.timer \
        ds4-mac-fast-internet-gateway.service \
        ds4-mac-fast-internet-gateway.timer \
        spark-fabric-tune.service; do
        systemctl disable --now "${unit}" 2>/dev/null || true
        systemctl reset-failed "${unit}" 2>/dev/null || true
        backup_and_remove "/etc/systemd/system/${unit}"
        backup_and_remove "/etc/systemd/system/${unit}.d"
        backup_and_remove "/etc/systemd/system/multi-user.target.wants/${unit}"
        backup_and_remove "/etc/systemd/system/timers.target.wants/${unit}"
    done
    for path in \
        /usr/local/sbin/ds4-ring-200g \
        /usr/local/sbin/ds4-ring-200g-apply \
        /usr/local/sbin/ds4-ring-200g-extend13 \
        /usr/local/sbin/ds4-ring-control-iface \
        /usr/local/sbin/ds4-10g-client-gateway-apply \
        /usr/local/sbin/ds4-10g-nat-gateway-apply \
        /usr/local/sbin/ds4-mac-fast-internet-gateway \
        /usr/local/sbin/spark-fabric-tune.sh; do
        backup_and_remove "${path}"
    done
    backup_and_remove /etc/ssh/sshd_config.d/99-ds4-rescue.conf
    if command -v sshd >/dev/null 2>&1; then
        install -d -m 0755 -o root -g root /run/sshd
        sshd -t
        systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || true
    fi
    systemctl daemon-reload
    ip link delete ds4ring0 2>/dev/null || true
}

write_fleet_sysctl()
{
    local security_path
    cat > "${FLEET_SYSCTL_PATH}" <<'EOF'
net.core.rmem_max = 536870912
net.core.wmem_max = 536870912
net.core.rmem_default = 134217728
net.core.wmem_default = 134217728
net.core.netdev_max_backlog = 250000
net.ipv4.tcp_rmem = 4096 87380 536870912
net.ipv4.tcp_wmem = 4096 65536 536870912
net.ipv4.tcp_mtu_probing = 1
net.ipv4.ip_forward = 1
net.ipv4.conf.all.rp_filter = 0
net.ipv4.conf.default.rp_filter = 0
EOF
    backup_and_remove /etc/sysctl.d/90-ds4-ring-200g.conf
    backup_and_remove /etc/sysctl.d/99-spark-fabric.conf
    security_path=/etc/sysctl.d/10-network-security.conf
    if [ -f "${security_path}" ]; then
        local security_backup
        security_backup="/var/backups/ds4-fleet/$(date -u +%Y%m%dT%H%M%SZ)${security_path}"
        mkdir -p "$(dirname "${security_backup}")"
        cp -a "${security_path}" "${security_backup}"
        sed -i -E \
            '/^[[:space:]]*net\.ipv4\.conf\.(all|default)\.rp_filter[[:space:]]*=/d' \
            "${security_path}"
    fi
    sysctl -p "${FLEET_SYSCTL_PATH}" >/dev/null
    while read -r device; do
        [ -e "/proc/sys/net/ipv4/conf/${device}/rp_filter" ] || continue
        sysctl -w "net.ipv4.conf.${device}.rp_filter=0" >/dev/null
    done < <(find /sys/class/net -mindepth 1 -maxdepth 1 -printf '%f\n')
}

configure_fabric_link()
{
    ethtool -G "${FABRIC_DEVICE}" rx 8192 tx 8192
    ethtool -K "${FABRIC_DEVICE}" tx-tcp-mangleid-segmentation off
    ip link set dev "${FABRIC_DEVICE}" mtu "${FABRIC_MTU}"
}

configure_management_link()
{
    local rank management_ip lan_ip
    command -v nmcli >/dev/null 2>&1 || return 0
    rank="$(node_rank)"
    management_ip="10.20.0.$((10 + rank))/24"
    lan_ip="192.168.50.$((128 + rank))/24"
    if ! nmcli -t -f NAME con show | grep -Fxq ds4-uplink-wired; then
        nmcli con add type ethernet ifname enP7s7 con-name ds4-uplink-wired >/dev/null
    fi
    nmcli con mod ds4-uplink-wired \
        connection.interface-name enP7s7 \
        connection.autoconnect yes \
        connection.autoconnect-priority 300 \
        ipv4.method manual \
        ipv4.addresses "${management_ip},${lan_ip}" \
        ipv4.gateway 192.168.50.1 \
        ipv4.dns "192.168.50.1,1.1.1.1" \
        ipv4.never-default no \
        ipv4.ignore-auto-routes yes \
        ipv4.ignore-auto-dns yes \
        ipv4.route-metric 10 \
        ipv6.method disabled >/dev/null
    nmcli con up ds4-uplink-wired >/dev/null
    nmcli con mod ds4-uplink-asus connection.autoconnect yes connection.autoconnect-priority 200 ipv4.route-metric 100 >/dev/null 2>&1 || true
    nmcli con mod ds4-uplink-tplink connection.autoconnect yes connection.autoconnect-priority 100 ipv4.route-metric 200 >/dev/null 2>&1 || true
    while IFS=: read -r name uuid device; do
        [ "${device}" = enP7s7 ] || continue
        [ "${name}" = ds4-uplink-wired ] && continue
        nmcli con delete uuid "${uuid}" >/dev/null 2>&1 || true
    done < <(nmcli -t -f NAME,UUID,DEVICE con show)
}

retire_legacy_mac_mounts()
{
    local fstab_path fstab_backup fstab_temp mount_path unit automount
    fstab_path=/etc/fstab
    if [ -f "${fstab_path}" ] && grep -Eq '^[^#].*[[:space:]]/(mnt/mac|home/mac-volumes)/' "${fstab_path}"; then
        fstab_backup="/var/backups/ds4-fleet/$(date -u +%Y%m%dT%H%M%SZ)${fstab_path}"
        mkdir -p "$(dirname "${fstab_backup}")"
        cp -a "${fstab_path}" "${fstab_backup}"
        fstab_temp="$(mktemp)"
        awk '$2 !~ /^\/(mnt\/mac|home\/mac-volumes)\// {print}' "${fstab_path}" > "${fstab_temp}"
        install -m 0644 -o root -g root "${fstab_temp}" "${fstab_path}"
        rm -f "${fstab_temp}"
    fi
    for mount_path in /mnt/mac/16tb0 /mnt/mac/16tb1 /mnt/mac/16tb2 \
        /mnt/mac/22tb0 /mnt/mac/22tb1 /mnt/mac/22tb2 \
        /home/mac-volumes/16tb0 /home/mac-volumes/16tb1 /home/mac-volumes/16tb2 \
        /home/mac-volumes/22tb0 /home/mac-volumes/22tb1 /home/mac-volumes/22tb2; do
        unit="$(systemd-escape --path --suffix=mount "${mount_path}")"
        automount="$(systemd-escape --path --suffix=automount "${mount_path}")"
        systemctl stop "${unit}" 2>/dev/null || true
        systemctl reset-failed "${unit}" 2>/dev/null || true
        systemctl stop "${automount}" 2>/dev/null || true
        systemctl reset-failed "${automount}" 2>/dev/null || true
    done
}

retire_legacy_nat()
{
    while iptables -C FORWARD -s 10.20.0.0/24 -i enP7s7 -o enP7s7 -j ACCEPT 2>/dev/null; do
        iptables -D FORWARD -s 10.20.0.0/24 -i enP7s7 -o enP7s7 -j ACCEPT
    done
    while iptables -C FORWARD -d 10.20.0.0/24 -i enP7s7 -o enP7s7 -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT 2>/dev/null; do
        iptables -D FORWARD -d 10.20.0.0/24 -i enP7s7 -o enP7s7 -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
    done
    while iptables -t nat -C POSTROUTING -s 10.20.0.0/24 -o enP7s7 -j MASQUERADE 2>/dev/null; do
        iptables -t nat -D POSTROUTING -s 10.20.0.0/24 -o enP7s7 -j MASQUERADE
    done
    iptables -P FORWARD ACCEPT
}

remove_legacy_addresses()
{
    local device
    while read -r device; do
        ip -4 addr flush dev "${device}" 2>/dev/null || true
        ip -4 route flush dev "${device}" 2>/dev/null || true
        if [ "${device}" != "${FABRIC_DEVICE}" ]; then
            ip link set dev "${device}" down 2>/dev/null || true
        fi
    done < <(fabric_devices)
}

configure_unmanaged_fabric()
{
    local device
    mkdir -p /etc/NetworkManager/conf.d
    cat > /etc/NetworkManager/conf.d/90-ds4-switched-fabric-unmanaged.conf <<'EOF'
[keyfile]
unmanaged-devices=interface-name:enp1s0f0np0;interface-name:enp1s0f1np1;interface-name:enP2p1s0f0np0;interface-name:enP2p1s0f1np1;interface-name:enP2p1;interface-name:enP2p2
EOF
    command -v nmcli >/dev/null 2>&1 || return 0
    while read -r device; do
        nmcli device disconnect "${device}" >/dev/null 2>&1 || true
        nmcli device set "${device}" managed no >/dev/null 2>&1 || true
    done < <(fabric_devices)
    nmcli general reload >/dev/null 2>&1 || true
}

remove_legacy_profiles()
{
    local row uuid
    command -v nmcli >/dev/null 2>&1 || return 0
    while IFS=: read -r row uuid; do
        [ -n "${uuid}" ] || continue
        nmcli connection delete uuid "${uuid}" >/dev/null 2>&1 || true
    done < <(nmcli -t -f NAME,UUID connection show | awk -F: '$1 ~ /^ds4-ring-/ || $1 == "ds4ring0" { print $1 ":" $2 }')
}

apply_switched_fabric()
{
    local rank address
    disable_cx7_hotplug_power_saving
    rank="$(node_rank)"
    if [ -z "${rank}" ] || ! [[ "${rank}" =~ ^[0-9]+$ ]] || [ "${rank}" -gt 15 ]; then
        printf 'unable to resolve node rank for %s\n' "${DS4_NODE_ID:-$(node_name)}" >&2
        return 2
    fi
    FABRIC_DEVICE="$(select_fabric_device)"
    address="${FABRIC_PREFIX}.$((10 + rank))/24"
    ip link set dev "${FABRIC_DEVICE}" up
    configure_fabric_link
    ip address replace "${address}" dev "${FABRIC_DEVICE}"
    printf 'switched_fabric node=%s rank=%s device=%s address=%s mtu=%s cx7_hotplug=disabled\n' \
        "${DS4_NODE_ID:-$(node_name)}" "${rank}" "${FABRIC_DEVICE}" "${address}" "${FABRIC_MTU}"
}

if [ "$(id -u)" -ne 0 ]; then
    exec sudo -- "$0" "$@"
fi

if [ "${1:-}" = "--install" ]; then
    install_service
    retire_legacy_units
    retire_legacy_mac_mounts
    retire_legacy_nat
    write_fleet_sysctl
    remove_legacy_profiles
    configure_unmanaged_fabric
    remove_legacy_addresses
    configure_management_link
fi
apply_switched_fabric
