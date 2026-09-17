#!/usr/bin/env bash
#
# sparkpipe_fsck_health.sh - post-boot filesystem health check (oneshot, root).
#
# SAFETY: NEVER runs write-mode fsck on the mounted ROOT filesystem. Root is
# checked READ-ONLY only: tune2fs -l (superblock state + error fields), a dmesg
# scan for ext4/I-O error lines from this boot, and e2fsck -n (no changes).
# NOTE: e2fsck -n on a MOUNTED rw root can report FALSE POSITIVES, so a
# NEEDS-REPAIR classification means "schedule a maintenance boot for a real
# (unmounted) root fsck" - NOT "the disk is corrupt", NOT an automatic repair,
# NOT a panic.
#
# DATA volumes (non-root real-fs mounts in /etc/fstab, e.g. extnvme) ARE checked
# write-mode: unmount -> fsck -f -y -> remount. Busy/unmountable volumes are
# recorded "skipped: busy" and left alone. This is the fleet's fsck mechanism now
# that the GRUB skip-fsck DEFAULT disables boot-time fsck.
#
# Exit: 0 unless THIS script itself fails. Health findings are DATA (written to
# /var/lib/sparkpipe/fsck-health/last.json + syslog), never exit codes.
#
# Install: cp to /usr/local/bin/sparkpipe_fsck_health.sh (chmod 0755) and enable
# the companion unit tools/devcycle/sparkpipe-fsck-health.service.
#
# Bash 3.2 compatible. Token-free.

set -u

ROOT_DEV="${SPARKPIPE_ROOT_DEV:-/dev/nvme0n1p2}"
OUT_DIR="${SPARKPIPE_FSCK_OUT_DIR:-/var/lib/sparkpipe/fsck-health}"
OUT_JSON="$OUT_DIR/last.json"
E2FSCK_TIMEOUT="${SPARKPIPE_E2FSCK_TIMEOUT:-180}"
FSTAB_PATH="${SPARKPIPE_FSTAB_PATH:-/etc/fstab}"
TAG="sparkpipe_fsck_health"
HOST="$(hostname -s 2>/dev/null || echo unknown)"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date +%Y-%m-%dT%H:%M:%SZ)"

ROOT_CLASS="HEALTHY"
TUNE_STATE=""
TUNE_ERRORS=""
E2FSCK_EXIT=""
E2FSCK_TIMED_OUT="false"

RAW="/tmp/sparkpipe-fsck-raw.$$"
rm -f "$RAW"
emit() { echo "$*" >> "$RAW"; }
log() { logger -t "$TAG" "$*"; }

# ---- root: read-only --------------------------------------------------------

if command -v tune2fs >/dev/null 2>&1; then
  TUNE_STATE="$(tune2fs -l "$ROOT_DEV" 2>/dev/null | awk -F: '/^Filesystem state:/{gsub(/[[:space:]]/,"",$2); print $2}')"
  TUNE_ERRORS="$(tune2fs -l "$ROOT_DEV" 2>/dev/null | awk -F: '/^Filesystem errors:/{sub(/^[[:space:]]+/,"",$2); print $2}')"
  if [ "$TUNE_STATE" != "clean" ]; then
    ROOT_CLASS="DEGRADED"
    emit "reason|tune2fs state=$TUNE_STATE"
  fi
else
  ROOT_CLASS="DEGRADED"
  emit "reason|tune2fs not available"
fi

# dmesg scan (this boot only - tail 2000 lines)
DMESG_HITS="$(dmesg 2>/dev/null | tail -2000 | grep -iE 'EXT4-fs error|ext4.*corrupt|I/O error|blk_update_request.*(I/O error|critical)' || true)"
if [ -n "$DMESG_HITS" ]; then
  echo "$DMESG_HITS" | tr '|' ' ' | while IFS= read -r l; do
    [ -n "$l" ] && emit "dmesg|$l"
  done
  ROOT_CLASS="NEEDS-REPAIR"
  emit "reason|dmesg shows ext4/I-O error lines"
fi

# e2fsck -n (read-only) with hard timeout
if command -v e2fsck >/dev/null 2>&1; then
  if command -v timeout >/dev/null 2>&1; then
    E2FSCK_OUT="$(timeout "$E2FSCK_TIMEOUT" e2fsck -n "$ROOT_DEV" 2>&1)"
    ec=$?
    if [ "$ec" -eq 124 ] || [ "$ec" -eq 137 ]; then
      E2FSCK_TIMED_OUT="true"
      E2FSCK_EXIT="$ec"
      [ "$ROOT_CLASS" = "HEALTHY" ] && ROOT_CLASS="DEGRADED"
      emit "reason|e2fsck -n timed out after $E2FSCK_TIMEOUT s"
    else
      E2FSCK_EXIT="$ec"
      case "$ec" in
        0) ;;
        1|2|4) ROOT_CLASS="NEEDS-REPAIR"; emit "reason|e2fsck -n exit $ec (errors found; may be a mounted-rw false positive)";;
        *) [ "$ROOT_CLASS" = "HEALTHY" ] && ROOT_CLASS="DEGRADED"; emit "reason|e2fsck -n exit $ec";;
      esac
    fi
  else
    [ "$ROOT_CLASS" = "HEALTHY" ] && ROOT_CLASS="DEGRADED"
    emit "reason|timeout command missing; e2fsck -n skipped"
  fi
else
  [ "$ROOT_CLASS" = "HEALTHY" ] && ROOT_CLASS="DEGRADED"
  emit "reason|e2fsck not available"
fi

# ---- data volumes: write-mode (unmount -> fsck -f -y -> remount) ------------

while read -r dev mp fstype opts; do
  [ -n "$dev" ] || continue
  case "$fstype" in ext2|ext3|ext4|xfs|btrfs) ;; *) continue ;; esac
  [ "$mp" = "/" ] && continue
  case "$mp" in /boot*) continue ;; esac
  device="$dev"
  case "$dev" in UUID=*) device="/dev/disk/by-uuid/${dev#UUID=}" ;; esac
  if [ ! -e "$device" ]; then
    emit "data|$mp|$device|$fstype|skipped: device missing|"
    continue
  fi
  automount_unit=""
  mount_unit=""
  restore_error=""
  was_mounted="false"
  case ",$opts," in
    *,x-systemd.automount,*) automount_configured="true" ;;
    *) automount_configured="false" ;;
  esac
  if [ "$automount_configured" = "true" ]; then
    if ! command -v systemd-escape >/dev/null 2>&1; then
      emit "data|$mp|$device|$fstype|skipped: automount control unavailable|"
      log "data $mp: skipped: automount control unavailable"
      continue
    fi
    escaped_unit="$(systemd-escape --path "$mp" 2>/dev/null)"
    if [ -z "$escaped_unit" ]; then
      emit "data|$mp|$device|$fstype|skipped: automount unit unresolved|"
      log "data $mp: skipped: automount unit unresolved"
      continue
    fi
    automount_unit="$escaped_unit.automount"
    mount_unit="$escaped_unit.mount"
    if ! systemctl is-active --quiet "$automount_unit" 2>/dev/null; then
      emit "data|$mp|$device|$fstype|skipped: automount inactive|"
      log "data $mp: skipped: automount inactive"
      continue
    fi
    if ! systemctl stop "$automount_unit" 2>/dev/null; then
      emit "data|$mp|$device|$fstype|skipped: automount busy|"
      log "data $mp: skipped: automount busy"
      continue
    fi
    if systemctl is-active --quiet "$mount_unit" 2>/dev/null; then
      if ! systemctl stop "$mount_unit" 2>/dev/null; then
        systemctl start "$automount_unit" 2>/dev/null
        emit "data|$mp|$device|$fstype|skipped: automount busy|"
        log "data $mp: skipped: automount busy"
        continue
      fi
    fi
    if mountpoint -q "$mp" 2>/dev/null; then
      systemctl start "$automount_unit" 2>/dev/null
      emit "data|$mp|$device|$fstype|skipped: automount still mounted|"
      log "data $mp: skipped: automount still mounted"
      continue
    fi
  else
    if mountpoint -q "$mp" 2>/dev/null; then
      was_mounted="true"
    fi
    if [ "$was_mounted" = "true" ] && ! umount "$mp" 2>/dev/null; then
      emit "data|$mp|$device|$fstype|skipped: busy|"
      log "data $mp: skipped: busy"
      continue
    fi
  fi
  fsck -f -y "$device" >/dev/null 2>&1
  ec=$?
  if [ -n "$automount_unit" ]; then
    if ! systemctl start "$automount_unit" 2>/dev/null; then
      restore_error="automount-restart-failed"
      log "data $mp: WARNING automount restart failed"
    fi
  elif [ "$was_mounted" = "true" ]; then
    if ! mount "$mp" 2>/dev/null; then
      restore_error="remount-failed"
      log "data $mp: WARNING remount failed"
    fi
  fi
  if [ -n "$restore_error" ]; then
    res="$restore_error"
  else
    case "$ec" in
      0) res="ok";;
      1) res="errors-corrected";;
      4) res="errors-left-uncorrected";;
      8) res="operational-error";;
      *) res="exit-$ec";;
    esac
  fi
  emit "data|$mp|$device|$fstype|$res|$ec"
  log "data $mp: $res (fsck exit $ec)"
done < <(awk '!/^[[:space:]]*#/ && NF>=4 {print $1, $2, $3, $4}' "$FSTAB_PATH" 2>/dev/null)

# ---- write result -----------------------------------------------------------

emit "host|$HOST"
emit "ts|$NOW"
emit "root_dev|$ROOT_DEV"
emit "class|$ROOT_CLASS"
emit "state|$TUNE_STATE"
emit "errors|$TUNE_ERRORS"
emit "e2fsck_exit|$E2FSCK_EXIT"
emit "e2fsck_timeout|$E2FSCK_TIMED_OUT"

mkdir -p "$OUT_DIR"
python3 - "$OUT_JSON" "$RAW" <<'PYEOF'
import json, sys
out = sys.argv[1]
scalars = {}
reasons = []
dmesg = []
data = []
for line in open(sys.argv[2]):
    line = line.rstrip()
    if not line:
        continue
    parts = line.split("|")
    k = parts[0]
    if k == "reason":
        reasons.append(parts[1])
    elif k == "dmesg":
        dmesg.append(parts[1])
    elif k == "data":
        data.append({
            "mount": parts[1],
            "device": parts[2],
            "fstype": parts[3],
            "result": parts[4],
            "fsck_exit": int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None,
        })
    else:
        scalars[k] = parts[1] if len(parts) > 1 else ""
doc = {
    "schema_version": 1,
    "host": scalars.get("host", ""),
    "timestamp": scalars.get("ts", ""),
    "classification": scalars.get("class", "DEGRADED"),
    "root": {
        "device": scalars.get("root_dev", ""),
        "classification": scalars.get("class", "DEGRADED"),
        "tune2fs_state": scalars.get("state", ""),
        "tune2fs_errors": scalars.get("errors", ""),
        "e2fsck_exit": int(scalars["e2fsck_exit"]) if scalars.get("e2fsck_exit", "").isdigit() else None,
        "e2fsck_timed_out": scalars.get("e2fsck_timeout", "false") == "true",
        "reasons": reasons,
        "dmesg_errors": dmesg,
    },
    "data_volumes": data,
}
json.dump(doc, open(out, "w"), indent=2)
PYEOF

log "root $ROOT_CLASS; $(wc -l < "$RAW") raw findings -> $OUT_JSON"
rm -f "$RAW"
exit 0
