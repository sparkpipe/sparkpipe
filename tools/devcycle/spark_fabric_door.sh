#!/usr/bin/env bash
set -eu

SWITCH_B=enP2p1s0f1np1
SWITCH_B_PREFIX=10.10.101
DOORS="enp1s0f1np1 enP2p1s0f1np1 enp1s0f0np0 enP2p1s0f0np0"

rank() {
    tr -cd 0-9 < /etc/ds4-node-rank
}

need_root() {
    [ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 2; }
}

rdma_of() {
    ibdev2netdev 2>/dev/null | awk -v n="$1" '!f && $5==n {print $1; f=1}'
}

status() {
    local door rdma
    for door in $DOORS; do
        rdma=$(rdma_of "$door")
        printf '%s pci=%s rdma=%s state=%s mtu=%s rings=%s %s ip=%s gid3=%s\n' \
            "$door" \
            "$(basename "$(readlink -f /sys/class/net/$door/device)")" \
            "${rdma:-none}" \
            "$(cat /sys/class/net/$door/operstate)" \
            "$(cat /sys/class/net/$door/mtu)" \
            "$(ethtool -g "$door" 2>/dev/null | awk '/^Current/{c=1} c && !f && /^RX:/{print $2; f=1}')" \
            "$(mlnx_qos -i "$door" 2>/dev/null | grep -o 'trust state: [a-z]*' | tr ' ' '_')" \
            "$(ip -4 -br addr show "$door" | awk '{print $3}')" \
            "$( [ -n "$rdma" ] && cat /sys/class/infiniband/$rdma/ports/1/gids/3 2>/dev/null)"
    done
}

current_rings() {
    ethtool -g "$1" 2>/dev/null | awk -v want="$2" '/^Current/{c=1} c && $1==want":" {print $2; exit}'
}

set_rings() {
    [ "$(current_rings "$1" RX)" = "$2" ] && [ "$(current_rings "$1" TX)" = "$2" ] && return 0
    ethtool -G "$1" rx "$2" tx "$2"
}

set_mtu() {
    [ "$(cat /sys/class/net/$1/mtu)" = "$2" ] || ip link set dev "$1" mtu "$2"
}

set_trust() {
    mlnx_qos -i "$1" | grep -q "Priority trust state: $2" || mlnx_qos -i "$1" --trust "$2" > /dev/null
}

set_only_address() {
    local current
    current=$(ip -4 -o addr show dev "$1" scope global | awk '{print $4}')
    if [ -n "$2" ] && [ "$current" = "$2" ]; then
        return 0
    fi
    [ -z "$current" ] || ip -4 addr flush dev "$1" scope global
    [ -z "$2" ] || ip address replace "$2" dev "$1"
}

up_switch_b() {
    need_root
    local n address rdma gid rings
    rings="${1:-8192}"
    case "$rings" in ''|*[!0-9]*) echo "ring count must be a number: $rings" >&2; exit 2 ;; esac
    n=$((10 + $(rank)))
    address="$SWITCH_B_PREFIX.$n/24"
    set_rings "$SWITCH_B" "$rings"
    ethtool -k "$SWITCH_B" | grep -q '^tx-tcp-mangleid-segmentation: off' || ethtool -K "$SWITCH_B" tx-tcp-mangleid-segmentation off
    set_mtu "$SWITCH_B" 9000
    ip link set dev "$SWITCH_B" up
    set_trust "$SWITCH_B" dscp
    set_only_address "$SWITCH_B" "$address"
    sysctl -q -w "net.ipv4.conf.$SWITCH_B.rp_filter=0"
    for _ in $(seq 1 30); do
        rdma=$(rdma_of "$SWITCH_B")
        gid=$(cat /sys/class/infiniband/$rdma/ports/1/gids/3 2>/dev/null || true)
        [ "$(cat /sys/class/infiniband/$rdma/ports/1/state 2>/dev/null | cut -d: -f1)" = 4 ] && [ "${gid#0000:0000:0000:0000:0000:ffff:}" != "$gid" ] && break
        sleep 1
    done
    mlnx_qos -i "$SWITCH_B" | grep -q 'Priority trust state: dscp' || { echo "trust not dscp on $SWITCH_B" >&2; exit 1; }
    [ "$(cat /sys/class/infiniband/$rdma/ports/1/state | cut -d: -f1)" = 4 ] || { echo "$rdma not ACTIVE" >&2; exit 1; }
    [ "${gid#0000:0000:0000:0000:0000:ffff:}" != "$gid" ] || { echo "$rdma gid 3 is not IPv4 RoCE: $gid" >&2; exit 1; }
    [ "$(current_rings "$SWITCH_B" RX)" = "$rings" ] || { echo "$SWITCH_B rx rings are not $rings" >&2; exit 1; }
    echo "switch door B up: $SWITCH_B $address rings=$rings rdma=$rdma gid3=$gid type=$(cat /sys/class/infiniband/$rdma/ports/1/gid_attrs/types/3)"
}

down_switch_b() {
    need_root
    set_only_address "$SWITCH_B" ""
    set_trust "$SWITCH_B" pcp
    ip link set dev "$SWITCH_B" down
    set_mtu "$SWITCH_B" 1500
    set_rings "$SWITCH_B" 1024
    echo "switch door B down: $SWITCH_B"
}

case "${1:-status}" in
    status) status ;;
    up-switch-b) up_switch_b "${2:-}" ;;
    down-switch-b) down_switch_b ;;
    *) echo "usage: $0 status|up-switch-b [rings]|down-switch-b" >&2; exit 2 ;;
esac
