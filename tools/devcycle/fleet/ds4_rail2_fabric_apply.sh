#!/usr/bin/env bash
set -euo pipefail

RAIL_DEVICE="${DS4_RAIL2_DEVICE:-enP2p1s0f0np0}"
IDLE_DEVICE="${DS4_RAIL2_IDLE_DEVICE:-enp1s0f0np0}"
RAIL_PREFIX="${DS4_RAIL2_PREFIX:-10.10.201}"
RAIL_PREFIX_LENGTH="${DS4_RAIL2_PREFIX_LENGTH:-24}"
RAIL_MTU="${DS4_RAIL2_MTU:-9000}"
FORCED_LINK_RANKS="${DS4_RAIL2_FORCED_LINK_RANKS:-2}"
NODE_RANK_PATH="${DS4_NODE_RANK_PATH:-/etc/ds4-node-rank}"
SYSFS_NET="${DS4_SYSFS_NET:-/sys/class/net}"
RUNTIME_PATH="${DS4_RAIL2_RUNTIME_PATH:-/usr/local/sbin/ds4-rail2-fabric-apply}"

node_rank()
{
    if [ ! -r "${NODE_RANK_PATH}" ]; then
        printf 'missing canonical rank file: %s\n' "${NODE_RANK_PATH}" >&2
        return 2
    fi
    tr -cd '0-9' < "${NODE_RANK_PATH}"
    printf '\n'
}

rdma_device()
{
    local path
    for path in "${SYSFS_NET}/$1/device/infiniband/"*; do
        [ -e "${path}" ] || continue
        basename "${path}"
        return 0
    done
    return 1
}

live_queue_pairs()
{
    local device
    device="$(rdma_device "$1")" || { printf '0\n'; return 0; }
    if ! command -v rdma >/dev/null 2>&1; then
        printf 'unknown\n'
        return 0
    fi
    rdma resource show qp link "${device}/1" 2>/dev/null | grep -vc '\[ib_core\]' || true
}

retire_idle_function()
{
    [ -e "${SYSFS_NET}/${IDLE_DEVICE}" ] || return 0
    if [ "$(live_queue_pairs "${IDLE_DEVICE}")" != "0" ]; then
        printf 'rail2: %s has live queue pairs; left up\n' "${IDLE_DEVICE}" >&2
        return 0
    fi
    ip -4 addr flush dev "${IDLE_DEVICE}" scope global
    ip link set dev "${IDLE_DEVICE}" down
}

rank_forces_link()
{
    local rank="$1" forced
    for forced in ${FORCED_LINK_RANKS//,/ }; do
        [ "${forced}" = "${rank}" ] && return 0
    done
    return 1
}

force_link()
{
    local settings fec
    settings="$(ethtool "${RAIL_DEVICE}")"
    if [[ "${settings}" == *"Auto-negotiation: on"* ]]; then
        ethtool -s "${RAIL_DEVICE}" autoneg off speed 100000 duplex full
    fi
    fec="$(ethtool --show-fec "${RAIL_DEVICE}")"
    if [[ "${fec}" != *"Configured FEC encodings: RS"* ]]; then
        ethtool --set-fec "${RAIL_DEVICE}" encoding rs
    fi
}

configure_link()
{
    ip link set dev "${RAIL_DEVICE}" up
    ethtool -G "${RAIL_DEVICE}" rx 1024 tx 1024
    ethtool -K "${RAIL_DEVICE}" tx-tcp-mangleid-segmentation off
    ip link set dev "${RAIL_DEVICE}" mtu "${RAIL_MTU}"
    sysctl -w "net.ipv4.conf.${RAIL_DEVICE}.rp_filter=0" >/dev/null
}

configure_address()
{
    local address="$1" stale live
    ip address replace "${address}" dev "${RAIL_DEVICE}"
    live="$(live_queue_pairs "${RAIL_DEVICE}")"
    while read -r stale; do
        [ -n "${stale}" ] && [ "${stale}" != "${address}" ] || continue
        if [ "${live}" = "0" ]; then
            ip address del "${stale}" dev "${RAIL_DEVICE}"
        else
            printf 'rail2: kept %s on %s; %s live queue pairs hold its RoCE GID\n' "${stale}" "${RAIL_DEVICE}" "${live}" >&2
        fi
    done < <(ip -4 -o addr show dev "${RAIL_DEVICE}" scope global | awk '{print $4}')
}

configure_lossless()
{
    ethtool -A "${RAIL_DEVICE}" rx off tx off
    mlnx_qos -i "${RAIL_DEVICE}" --trust dscp >/dev/null
    mlnx_qos -i "${RAIL_DEVICE}" --pfc 0,0,0,1,0,0,1,0 >/dev/null
    mlnx_qos -i "${RAIL_DEVICE}" --prio2buffer 1,1,1,1,1,1,1,1 >/dev/null
    mlnx_qos -i "${RAIL_DEVICE}" --buffer_size 0,459360,0,0,0,0,0,0 >/dev/null
    ethtool --set-tunable "${RAIL_DEVICE}" pfc-prevention-tout 100
    local trust pfc buffers
    trust="$(mlnx_qos -i "${RAIL_DEVICE}")"
    pfc="$(dcb pfc show dev "${RAIL_DEVICE}" prio-pfc)"
    buffers="$(dcb buffer show dev "${RAIL_DEVICE}" prio-buffer)"
    if [[ "${trust}" != *"Priority trust state: dscp"* || "${pfc}" != *"3:on"*"6:on"* || "${buffers}" != *"0:1 1:1 2:1 3:1 4:1 5:1 6:1 7:1"* ]]; then
        printf 'rail2: lossless state did not take on %s
' "${RAIL_DEVICE}" >&2
        return 4
    fi
}

apply_rail2()
{
    local rank address forced=no
    rank="$(node_rank)"
    if [ -z "${rank}" ] || ! [[ "${rank}" =~ ^[0-9]+$ ]] || [ "${rank}" -gt 15 ]; then
        printf 'unable to resolve rail2 rank\n' >&2
        return 2
    fi
    rank=$((10#${rank}))
    if [ ! -e "${SYSFS_NET}/${RAIL_DEVICE}" ]; then
        printf 'rail2 device is missing: %s\n' "${RAIL_DEVICE}" >&2
        return 3
    fi
    retire_idle_function
    configure_link
    if rank_forces_link "${rank}"; then
        force_link
        forced=yes
    fi
    address="${RAIL_PREFIX}.${rank}/${RAIL_PREFIX_LENGTH}"
    configure_address "${address}"
    configure_lossless
    printf 'rail2_fabric rank=%s device=%s address=%s mtu=%s idle=%s forced_link=%s\n' \
        "${rank}" "${RAIL_DEVICE}" "${address}" "${RAIL_MTU}" "${IDLE_DEVICE}" "${forced}"
}

if [ "$(id -u)" -ne 0 ]; then
    exec sudo -- "$0" "$@"
fi

if [ "${1:-}" = "--install" ]; then
    install -m 0755 "$0" "${RUNTIME_PATH}"
    printf '%s\n' '--install only installs the runtime; use ds4_spark_brickproof.py to install the service and timer' >&2
fi
apply_rail2
