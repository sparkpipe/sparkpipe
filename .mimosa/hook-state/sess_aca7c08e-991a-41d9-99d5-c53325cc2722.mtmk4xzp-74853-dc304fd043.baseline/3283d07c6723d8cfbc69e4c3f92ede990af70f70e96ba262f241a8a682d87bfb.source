#!/usr/bin/env bash
set -euo pipefail

DIRECT_DEVICE="${DS4_DIRECT_FABRIC_DEVICE:-enp1s0f0np0}"
DIRECT_PREFIX="${DS4_DIRECT_FABRIC_PREFIX:-10.10.200}"
DIRECT_MTU="${DS4_DIRECT_FABRIC_MTU:-9000}"
RUNTIME_PATH="${DS4_DIRECT_FABRIC_RUNTIME_PATH:-/usr/local/sbin/ds4-direct-pair-fabric-apply}"

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
    install -m 0755 "$0" "${RUNTIME_PATH}"
    printf '%s\n' '--install only installs the runtime; use ds4_spark_brickproof.py to install the service and timer' >&2
}

configure_direct_pair()
{
    local rank address
    rank="$(node_rank)"
    if [ -z "${rank}" ] || ! [[ "${rank}" =~ ^[0-9]+$ ]] || [ "${rank}" -gt 15 ]; then
        printf 'unable to resolve direct-pair rank\n' >&2
        return 2
    fi
    if [ ! -e "/sys/class/net/${DIRECT_DEVICE}" ]; then
        printf 'direct-pair device is missing: %s\n' "${DIRECT_DEVICE}" >&2
        return 3
    fi
    ip link set dev "${DIRECT_DEVICE}" up
    ethtool -G "${DIRECT_DEVICE}" rx 8192 tx 8192
    ethtool -K "${DIRECT_DEVICE}" tx-tcp-mangleid-segmentation off
    ip link set dev "${DIRECT_DEVICE}" mtu "${DIRECT_MTU}"
    ip -4 addr flush dev "${DIRECT_DEVICE}" scope global
    ip -4 route flush dev "${DIRECT_DEVICE}"
    address="${DIRECT_PREFIX}.${rank}/31"
    ip address replace "${address}" dev "${DIRECT_DEVICE}"
    sysctl -w "net.ipv4.conf.${DIRECT_DEVICE}.rp_filter=0" >/dev/null
    printf 'direct_pair_fabric rank=%s device=%s address=%s mtu=%s\n' \
        "${rank}" "${DIRECT_DEVICE}" "${address}" "${DIRECT_MTU}"
}

if [ "$(id -u)" -ne 0 ]; then
    exec sudo -- "$0" "$@"
fi

if [ "${1:-}" = "--install" ]; then
    install_service
fi
configure_direct_pair
