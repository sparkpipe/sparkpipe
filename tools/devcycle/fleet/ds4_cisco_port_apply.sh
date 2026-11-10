#!/usr/bin/env bash
set -euo pipefail

CISCO_PREFIX="${DS4_CISCO_PREFIX:-10.10.201}"
CISCO_PREFIX_LENGTH="${DS4_CISCO_PREFIX_LENGTH:-24}"
CISCO_MTU="${DS4_CISCO_MTU:-9000}"
SYSFS_NET="${DS4_SYSFS_NET:-/sys/class/net}"

port_of_host()
{
    case "$1" in
        lenovo7) printf 'enp1s0f1np1 247\n' ;;
        lenovo8) printf 'enp1s0f1np1 248\n' ;;
        lenovo9) printf 'enp1s0f0np0 249\n' ;;
        *) return 1 ;;
    esac
}

apply_port()
{
    local host device octet address
    host="$(hostname -s)"
    if ! read -r device octet < <(port_of_host "${host}"); then
        printf 'cisco_port: no port known for %s\n' "${host}" >&2
        return 2
    fi
    if [ ! -e "${SYSFS_NET}/${device}" ]; then
        printf 'cisco_port: device is missing: %s\n' "${device}" >&2
        return 3
    fi
    address="${CISCO_PREFIX}.${octet}/${CISCO_PREFIX_LENGTH}"
    ip link set dev "${device}" up
    ethtool -G "${device}" rx 1024 tx 1024
    ip link set dev "${device}" mtu "${CISCO_MTU}"
    sysctl -w "net.ipv4.conf.${device}.rp_filter=0" >/dev/null
    ip address replace "${address}" dev "${device}"
    printf 'cisco_port host=%s device=%s address=%s mtu=%s\n' "${host}" "${device}" "${address}" "${CISCO_MTU}"
}

if [ "$(id -u)" -ne 0 ] && [ -z "${DS4_CISCO_NO_SUDO:-}" ]; then
    exec sudo -- "$0" "$@"
fi

apply_port
