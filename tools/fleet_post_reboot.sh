#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: $0 [--fix-ceph] [--timeout SECONDS] HOST [HOST...]" >&2
  echo "       HOST is a rebooted Spark (spark0..sparkf); the other fifteen are checked as its mesh peers" >&2
  exit 2
}

HEX=0123456789abcdef
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
CEPH_HOST="${FLEET_CEPH_HOST:-spark0}"
FIX_CEPH=0
TIMEOUT=900
REBOOTED=()
while [ $# -gt 0 ]; do
  case "$1" in
    --fix-ceph) FIX_CEPH=1; shift ;;
    --timeout) [ $# -ge 2 ] || usage; TIMEOUT="$2"; shift 2 ;;
    spark[0-9a-f]) REBOOTED+=("$1"); shift ;;
    *) usage ;;
  esac
done
[ "${#REBOOTED[@]}" -gt 0 ] || usage
case "$TIMEOUT" in ''|*[!0-9]*) usage ;; esac

FAILED=0
note() { printf '%s %-7s %-14s %s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" "$3"; }
fail() { note "$1" "$2" "FAIL $3"; FAILED=1; }
pass() { note "$1" "$2" "ok   $3"; }
is_rebooted() { local h; for h in "${REBOOTED[@]}"; do [ "$h" = "$1" ] && return 0; done; return 1; }

deadline=$(( $(date +%s) + TIMEOUT ))
for host in "${REBOOTED[@]}"; do
  until $SSH "$host" true 2>/dev/null; do
    if [ "$(date +%s)" -ge "$deadline" ]; then fail "$host" reachable "no ssh within ${TIMEOUT}s"; continue 2; fi
    sleep 10
  done
  pass "$host" reachable "up $($SSH "$host" 'cut -d. -f1 /proc/uptime')s"

  sb=$($SSH "$host" 'd=$(findmnt -n -o SOURCE /); c=$(sudo -n dd if=$d bs=1024 skip=1 count=1 2>/dev/null | dd bs=1 skip=56 count=2 2>/dev/null | od -An -tx1 | tr -d " \n"); o=$(sudo -n dd if=$d bs=4096 count=1 iflag=direct 2>/dev/null | dd bs=1 skip=1080 count=2 2>/dev/null | od -An -tx1 | tr -d " \n"); echo "$c $o"')
  if [ "$sb" = "53ef 53ef" ]; then pass "$host" superblock "ext4 magic in cache and on disk"; else fail "$host" superblock "root superblock magic cache/disk = '$sb', want '53ef 53ef'"; fi

  errors=$($SSH "$host" 'sudo -n dmesg 2>/dev/null | grep -i -E "EXT4-fs (error|warning)|arm-smmu.*event|smmu.*(fault|CMD_SYNC)|iommu.*\bfault|nvme.*(timeout|reset)|mlx5.*(syndrome|fatal)|mlx5_pci_err_detected|CmpltTO|AER:.*(Uncorrect|Fatal)" | tail -3')
  if [ -z "$errors" ]; then pass "$host" kernel "no ext4, SMMU, NVMe, mlx5 or AER error lines since boot"; else fail "$host" kernel "$(echo "$errors" | tr '\n' '|' | cut -c1-240)"; fi

  until [ "$($SSH "$host" 'u=$(cut -d. -f1 /proc/uptime); s=$(systemctl is-system-running 2>/dev/null); [ "$u" -ge 90 ] && [ "$s" != starting ] && [ "$s" != initializing ] && echo settled')" = settled ]; do
    if [ "$(date +%s)" -ge "$deadline" ]; then break; fi
    sleep 10
  done
  guard=$($SSH "$host" 'echo "$(systemctl is-active spark-dma-guard.service 2>/dev/null) $(sudo -n journalctl -b -u spark-dma-guard.service -o cat --no-pager 2>/dev/null | grep -o "event=start mode=[a-z-]*" | tail -1 | cut -d= -f3)"')
  if [ "$guard" = "active armed" ]; then pass "$host" dma-guard "spark-dma-guard active and armed"; else fail "$host" dma-guard "spark-dma-guard is '$guard' (want 'active armed'; watch-only means three unclean boots within six hours)"; fi

  target=$($SSH "$host" 'echo "$(systemctl get-default) $(systemctl is-system-running 2>/dev/null)"')
  case "$target" in
    "multi-user.target running"|"multi-user.target degraded") pass "$host" boot-target "$target" ;;
    *) fail "$host" boot-target "'$target' (want multi-user.target reached; a graphical target with a waiting splash never starts the DMA guard)" ;;
  esac

  rank=$($SSH "$host" 'tr -cd 0-9 < /etc/ds4-node-rank')
  if [ "$rank" = 15 ]; then
    hublink=$($SSH "$host" 'nmcli -t -f NAME,DEVICE connection show --active | grep -c "^ds4-speculation-link:enP7s7$"; ip -4 -o addr show dev enP7s7 | grep -c " inet 10\.10\.250\.1/30 "' | tr '\n' ' ')
    set -- $hublink
    if [ "${1:-0}" = 1 ] && [ "${2:-0}" = 1 ]; then pass "$host" uplink "rank 15: ds4-speculation-link active with 10.10.250.1/30 to the hub"; else fail "$host" uplink "rank 15 hub link: profile ${1:-?}, address ${2:-?} (want ds4-speculation-link with 10.10.250.1/30 on enP7s7)"; fi
  else
    uplink=$($SSH "$host" 'nmcli -t -f NAME,DEVICE connection show --active | grep -c "^ds4-uplink-wired:enP7s7$"; ip -4 route show default | grep -Ec "^default via 192\.168\.50\.1 dev enP7s7 (.* )?metric 10( |$)"' | tr '\n' ' ')
    set -- $uplink
    if [ "${1:-0}" = 1 ] && [ "${2:-0}" -ge 1 ] 2>/dev/null; then pass "$host" uplink "ds4-uplink-wired active, $2 default route(s) via 192.168.50.1 metric 10"; else fail "$host" uplink "active ds4-uplink-wired profiles ${1:-?}, metric-10 default routes ${2:-?} (want 1 and at least 1; NetworkManager may have adopted the initramfs ip= profile, see docs/NODE_BASELINE.md)"; fi
  fi

  emergency=$($SSH "$host" 'ss -ltn | grep -c ":2222 "')
  if [ "$emergency" -ge 1 ] 2>/dev/null; then pass "$host" emergency-ssh "port 2222 listening"; else fail "$host" emergency-ssh "nothing listens on 2222"; fi

  earlyoom=$($SSH "$host" 'ps -o args= -C earlyoom | head -1')
  case "$earlyoom" in
    *"-m 3 -s 100 -r 5 --avoid sshd|systemd|init") pass "$host" earlyoom "$earlyoom" ;;
    *) fail "$host" earlyoom "'$earlyoom' (want -m 3 -s 100 -r 5)" ;;
  esac

  linger=$($SSH "$host" 'loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null')
  if [ "$linger" = yes ]; then pass "$host" linger "user services start without a login"; else fail "$host" linger "Linger=$linger"; fi

  owner_query='p=$(pgrep -o -f "sparkdata/weightd/sparkpipe_[w]eightd"); [ -n "$p" ] && echo "$p $(sed -n "s|.*/||p" /proc/$p/cgroup)"'
  until [ "$($SSH "$host" 'systemctl --user is-active fleet-agent 2>/dev/null')" = active ] && [ -n "$($SSH "$host" "$owner_query")" ]; do
    if [ "$(date +%s)" -ge "$deadline" ]; then break; fi
    sleep 10
  done
  agent=$($SSH "$host" 'systemctl --user is-active fleet-agent 2>/dev/null')
  owner=$($SSH "$host" "$owner_query")
  safe=$($SSH "$host" 'journalctl --user -u fleet-agent -b -o cat --no-pager 2>/dev/null | grep -E "^agent: (rank=|SAFE MODE)" | tail -1 | grep -c "SAFE MODE"')
  if [ "${safe:-0}" = 0 ]; then pass "$host" agent-mode "fleet-agent not in safe mode"; else fail "$host" agent-mode "fleet-agent SAFE MODE after short boots; clear with: cat /proc/sys/kernel/random/boot_id > ~/.fleet_agent_safe_clear, then systemctl --user restart fleet-agent"; fi

  clocks=$($SSH "$host" 'c=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " "); echo "${c:-none} $(cut -d. -f1 /proc/uptime)"')
  clock=${clocks%% *}; up=${clocks##* }
  case "$clock:$up" in
    *[!0-9:]*|:*|*:) fail "$host" gpu-clock "cannot read the graphics clock ('$clocks')" ;;
    *) if [ "$clock" -le 2200 ]; then pass "$host" gpu-clock "$clock MHz (cap 2200)"; elif [ "$up" -lt 900 ]; then pass "$host" gpu-clock "$clock MHz, uptime ${up}s: the hub fleet-clockcap timer re-caps it within 5 min"; else fail "$host" gpu-clock "$clock MHz above the 2200 MHz cap (sudo nvidia-smi -lgc 0,2200)"; fi ;;
  esac
  if [ "$agent" != active ] || [ -z "$owner" ]; then
    fail "$host" units "fleet-agent=$agent weightd owner='${owner:-none}'"
  elif [ "${owner#* }" = sparkpipe-weightd.service ]; then
    pass "$host" units "fleet-agent active, weightd pid ${owner%% *} in sparkpipe-weightd.service"
  else
    pass "$host" units "fleet-agent active, weightd pid ${owner%% *} in ${owner#* } (an older agent started it; it moves to sparkpipe-weightd at its next restart)"
  fi
done

status_tool='~/sparkdata/core/bin/sparkpipe_mesh_status'
for rank in $(seq 0 15); do
  host="spark${HEX:$rank:1}"
  $SSH "$host" true 2>/dev/null || { fail "$host" mesh "unreachable"; continue; }
  remaining=$(( deadline - $(date +%s) )); [ "$remaining" -lt 5 ] && remaining=5
  if $SSH "$host" "test -x $status_tool"; then
    out=$($SSH "$host" "$status_tool --socket /tmp/spark_weightd.sock --wait-lane-peers 0xffff --timeout $remaining" 2>&1)
    code=$?
    if [ "$code" = 0 ]; then pass "$host" mesh "$(echo "$out" | head -1 | cut -c1-160)"; else fail "$host" mesh "sparkpipe_mesh_status exit $code: $(echo "$out" | head -2 | tr '\n' ' ' | cut -c1-200)"; fi
  else
    ready=$($SSH "$host" 'test -e /tmp/weightd-mesh/.ready && echo marker=yes || echo marker=no; grep -a WD-MESH-STATS ~/weightd.log 2>/dev/null | tail -1 | grep -o " ready=[01]" | tr -d " "' | tr '\n' ' ')
    case "$ready" in "marker=yes ready=1 ") pass "$host" mesh ".ready present, last WD-MESH-STATS ready=1 (no sparkpipe_mesh_status on this node)" ;; *) fail "$host" mesh "readiness '$ready' (no sparkpipe_mesh_status on this node)" ;; esac
  fi
done

for host in "${REBOOTED[@]}"; do
  $SSH "$host" "test -x $status_tool" 2>/dev/null || { note "$host" peers "skip sparkpipe_mesh_status not installed; peer rewiring not checked"; continue; }
  rank=$($SSH "$host" 'tr -cd 0-9 < /etc/ds4-node-rank')
  boot=$($SSH "$host" "$status_tool --socket /tmp/spark_weightd.sock --timeout 10" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["boot_ns"])' 2>/dev/null)
  [ -n "$boot" ] || { fail "$host" peers "cannot read its mesh boot identity"; continue; }
  stale=""
  for peer in $(seq 0 15); do
    peer_host="spark${HEX:$peer:1}"
    [ "$peer_host" = "$host" ] && continue
    wired=$($SSH "$peer_host" "$status_tool --socket /tmp/spark_weightd.sock --timeout 10" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['peers'][$rank]['wired_boot_ns'])" 2>/dev/null)
    [ "$wired" = "$boot" ] || stale="$stale $peer_host"
  done
  if [ -z "$stale" ]; then pass "$host" peers "all 15 peers wired to boot_ns $boot"; else fail "$host" peers "still wired to an old boot of $host:$stale"; fi
done

for host in "${REBOOTED[@]}"; do
  osds=$($SSH "$CEPH_HOST" "sudo -n ceph osd tree down -f json 2>/dev/null" | python3 -c "
import json,sys
tree=json.load(sys.stdin)['nodes']
hosts={n['name']:n.get('children',[]) for n in tree if n.get('type')=='host'}
names={n['id']:n['name'] for n in tree if n.get('type')=='osd'}
print(' '.join(names[i] for i in hosts.get('$host',[]) if i in names))" 2>/dev/null)
  if [ -z "$osds" ]; then pass "$host" ceph "no down OSD on this host"; continue; fi
  if [ "$FIX_CEPH" = 1 ]; then
    $SSH "$host" 'for lv in $(sudo -n lvs --noheadings -o vg_name,lv_name 2>/dev/null | awk "\$1 ~ /^ceph-/ {print \$1\"/\"\$2}"); do sudo -n lvchange --refresh "$lv"; done; for unit in $(systemctl list-units --failed --plain --no-legend "ceph-*@osd.*" | awk "{print \$1}"); do sudo -n systemctl reset-failed "$unit"; sudo -n systemctl start "$unit"; done' >/dev/null 2>&1
    sleep 20
    osds=$($SSH "$CEPH_HOST" "sudo -n ceph osd tree down -f json 2>/dev/null" | python3 -c "
import json,sys
tree=json.load(sys.stdin)['nodes']
hosts={n['name']:n.get('children',[]) for n in tree if n.get('type')=='host'}
names={n['id']:n['name'] for n in tree if n.get('type')=='osd'}
print(' '.join(names[i] for i in hosts.get('$host',[]) if i in names))" 2>/dev/null)
    if [ -z "$osds" ]; then pass "$host" ceph "OSDs back after lvchange --refresh and restart"; else fail "$host" ceph "still down after the refresh: $osds"; fi
  else
    fail "$host" ceph "down: $osds; after a USB re-enumeration rerun with --fix-ceph (lvchange --refresh on the ceph LVs, then restart the failed OSD units)"
  fi
done
health=$($SSH "$CEPH_HOST" "sudo -n ceph health 2>/dev/null")
note "$CEPH_HOST" ceph-health "$health"

if [ "$FAILED" = 0 ]; then
  note fleet post-reboot "READY ${REBOOTED[*]}"
  exit 0
fi
note fleet post-reboot "NOT READY ${REBOOTED[*]}"
exit 1
