#!/bin/bash
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
export LC_ALL=C
MODE=${1:-scan}
ARG=${2:-}
DRY=${SPARK_OSD_GUARD_DRY:-0}
LOG=/var/log/spark-osd-guard.log
STATE=/var/lib/spark-osd-guard
LOCK=/run/spark-osd-guard.lock
MAX_ACTIONS_PER_HOUR=3
STABLE_SECS=${SPARK_OSD_GUARD_STABLE_SECS:-20}
BOOT_QUIET_SECS=${SPARK_OSD_GUARD_BOOT_QUIET_SECS:-300}
BOOT_RETRY_UNTIL_SECS=1500
SCAN_CONFIRM_SECS=60

log() {
  printf '%s [%s] %s\n' "$(date -u +%FT%TZ)" "$MODE" "$*" >> "$LOG" 2>/dev/null
  logger -t spark-osd-guard -- "[$MODE] $*" 2>/dev/null
}

uptime_s() { cut -d. -f1 /proc/uptime; }

stale_list() {
  dmsetup table 2>/dev/null | while IFS= read -r line; do
    name=${line%%: *}
    case $name in ceph--*) ;; *) continue ;; esac
    rest=${line#*: }
    dmdev=$(dmsetup info -c --noheadings -o blkdevname "$name" 2>/dev/null)
    [ -n "$dmdev" ] || continue
    bad=0
    for mm in $(printf '%s\n' "$rest" | grep -oE ' [0-9]+:[0-9]+' | tr -d ' ' | sort -u); do
      [ -e "/sys/dev/block/$mm/holders/$dmdev" ] || bad=1
    done
    [ "$bad" = 1 ] && echo "$name"
  done | sort -u
}

action_budget_ok() {
  local now cut n
  now=$(date +%s)
  cut=$((now - 3600))
  mkdir -p "$STATE"
  touch "$STATE/actions"
  awk -v c="$cut" '$1 >= c' "$STATE/actions" > "$STATE/actions.new" 2>/dev/null && mv "$STATE/actions.new" "$STATE/actions"
  n=$(wc -l < "$STATE/actions")
  [ "$n" -lt "$MAX_ACTIONS_PER_HOUR" ]
}

record_action() { date +%s >> "$STATE/actions"; }

osd_unit_for() {
  local vg=$1 lv=$2 tags id fsid
  tags=$(lvs --noheadings -o lv_tags "$vg/$lv" 2>/dev/null 9>&- | tr ',' '\n')
  id=$(printf '%s\n' "$tags" | sed -n 's/^ *ceph\.osd_id=//p' | head -1)
  fsid=$(printf '%s\n' "$tags" | sed -n 's/^ *ceph\.cluster_fsid=//p' | head -1)
  [ -n "$id" ] && [ -n "$fsid" ] && echo "ceph-${fsid}@osd.${id}.service"
}

eio_evidence() {
  local unit=$1
  journalctl -u "$unit" --since "-30min" -o cat --no-pager 2>/dev/null | grep -ciE 'Input/output error|aio_submit|_aio_thread|bdev.*(error|failed)|caught signal|ceph_abort|unable to read'
}

handle_osd() {
  local unit=$1 st load ev
  load=$(systemctl show -p LoadState --value "$unit" 2>/dev/null)
  if [ "$load" != "loaded" ]; then log "no loaded unit $unit"; return 0; fi
  st=$(systemctl is-active "$unit" 2>/dev/null)
  case $st in
    failed|inactive|deactivating)
      log "$unit is $st, reset-failed + start"
      [ "$DRY" = 1 ] && { log "dry run: skipped start"; return 0; }
      systemctl reset-failed "$unit" 2>/dev/null
      systemctl start --no-block "$unit" 2>/dev/null
      ;;
    activating)
      log "$unit is activating (auto-restart), reset-failed + start"
      [ "$DRY" = 1 ] && { log "dry run: skipped start"; return 0; }
      systemctl reset-failed "$unit" 2>/dev/null
      systemctl start --no-block "$unit" 2>/dev/null
      ;;
    active)
      ev=$(eio_evidence "$unit")
      if [ "${ev:-0}" -gt 0 ]; then
        log "$unit active but logged $ev I/O error lines in 30 min, restarting"
        [ "$DRY" = 1 ] && { log "dry run: skipped restart"; return 0; }
        systemctl restart --no-block "$unit" 2>/dev/null
      else
        log "$unit active with no I/O error evidence, leaving it running"
      fi
      ;;
    *) log "$unit state '$st' left alone" ;;
  esac
}

fix_dm() {
  local dmname=$1 vg lv unit
  read -r vg lv < <(dmsetup splitname "$dmname" LVM --noheadings -o vg_name,lv_name --separator ' ' 2>/dev/null | head -1)
  if [ -z "${vg:-}" ] || [ -z "${lv:-}" ]; then log "cannot split dm name $dmname"; return 0; fi
  if ! action_budget_ok; then log "action budget exhausted (>= $MAX_ACTIONS_PER_HOUR in 1 h), leaving $vg/$lv stale for the operator"; return 0; fi
  log "stale dm table on $vg/$lv: lvchange --refresh"
  if [ "$DRY" = 1 ]; then log "dry run: skipped refresh"; return 0; fi
  record_action
  out=$(timeout 90 lvchange --refresh "$vg/$lv" 2>&1 9>&-)
  rc=$?
  log "lvchange --refresh rc=$rc: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-300)"
  if stale_list | grep -qx "$dmname"; then
    log "still stale after refresh: $vg/$lv (device not present or refresh failed)"
    return 0
  fi
  log "dm table for $vg/$lv now consistent"
  unit=$(osd_unit_for "$vg" "$lv")
  if [ -z "$unit" ]; then log "no ceph osd tags on $vg/$lv"; return 0; fi
  handle_osd "$unit"
}

failed_osd_units() {
  local up=$1 unit st lvdev
  [ "$up" -ge "$BOOT_QUIET_SECS" ] || return 0
  systemctl list-units --all --no-legend --plain 'ceph-*@osd.*.service' 2>/dev/null | awk '{print $1, $3}' | while read -r unit st; do
    case $st in
      failed) echo "$unit" ;;
      inactive) [ "$up" -lt "$BOOT_RETRY_UNTIL_SECS" ] && echo "$unit" ;;
    esac
  done
}

main_scan() {
  local up stale first now prev
  up=$(uptime_s)
  if [ "$up" -lt "$BOOT_QUIET_SECS" ]; then exit 0; fi
  stale=$(stale_list)
  now=$(date +%s)
  mkdir -p "$STATE"
  if [ -n "$stale" ]; then
    for dmname in $stale; do
      if [ "$MODE" = scan ]; then
        first=$(grep -F "$dmname " "$STATE/stale_seen" 2>/dev/null | awk '{print $2}' | head -1)
        if [ -z "$first" ]; then
          echo "$dmname $now" >> "$STATE/stale_seen"
          log "stale dm table first seen: $dmname (waiting $SCAN_CONFIRM_SECS s to confirm)"
          continue
        fi
        if [ $((now - first)) -lt "$SCAN_CONFIRM_SECS" ]; then continue; fi
      fi
      fix_dm "$dmname"
      grep -vF "$dmname " "$STATE/stale_seen" 2>/dev/null > "$STATE/stale_seen.new"; mv "$STATE/stale_seen.new" "$STATE/stale_seen" 2>/dev/null
    done
  else
    : > "$STATE/stale_seen"
  fi
  if [ "$MODE" = scan ] && [ -z "$stale" ]; then
    for unit in $(failed_osd_units "$up"); do
      if action_budget_ok; then
        log "$unit not running with consistent dm tables, reset-failed + start"
        if [ "$DRY" != 1 ]; then record_action; systemctl reset-failed "$unit" 2>/dev/null; systemctl start --no-block "$unit" 2>/dev/null; fi
      fi
    done
  fi
}

mkdir -p "$STATE" 2>/dev/null
exec 9>"$LOCK"
flock -w 120 9 || { log "lock timeout"; exit 0; }

case $MODE in
  event)
    udevadm settle --timeout=30 2>/dev/null
    t=0
    while [ "$t" -lt "$STABLE_SECS" ]; do
      sleep 2
      t=$((t + 2))
      if [ -n "$ARG" ] && [ ! -b "/dev/$ARG" ]; then log "device $ARG vanished within ${STABLE_SECS}s, flapping; not acting"; exit 0; fi
    done
    log "device ${ARG:-?} stable for ${STABLE_SECS}s, checking dm tables"
    main_scan
    ;;
  scan) main_scan ;;
  *) log "unknown mode $MODE" ;;
esac
exit 0
