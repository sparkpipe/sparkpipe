#!/usr/bin/env bash
# usage: qwenmax_placement_ship.sh tp16 RANK | stripped STAGE HOLDER | verify FILE | purge FILE ARM KIND
# runs on the target node; heavy ops under sparkcap; drop_caches after each placement
set -euo pipefail
MODE="${1:?mode required: tp16 RANK | stripped STAGE HOLDER | verify FILE | purge FILE ARM KIND}"
shift

fail() { echo "SHIP-FAIL:$*" >&2; exit 1; }
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cap_sha() { sudo -n sparkcap --mem 8192 sha256sum "$1" | awk '{print $1}'; }
cap_cp() { sudo -n sparkcap --mem 8192 cp "$@"; }
cap_scp() { sudo -n sparkcap --mem 8192 scp -q -o BatchMode=yes "$@"; }
purge_caches() { sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
receipt_fields() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["bytes"], d["output_sha256"])' "$1"; }

arm_of() {
  case "$1" in
  *qwenmax.pp16-stripped*) echo qwenmax.pp16-stripped ;;
  *qwenmax.pp16*) echo qwenmax.pp16 ;;
  *) fail "unknown-arm $1" ;;
  esac
}

tp16() {
  local rank="$1" warm=/mnt/model-warm/qwenmax.nvfp4.tp16/packs dst="$HOME/sparkdata/qwenmax.nvfp4.tp16/packs"
  local fields name bytes exp act=PLACED dsha=none esha_src esha_dst
  [ -d "$warm" ] || fail "no-warm-mount"
  fields=$(python3 -c 'import json,sys,os; idx=json.load(open(sys.argv[1])); e=next(x for x in idx["ranks"] if x["rank"]==int(sys.argv[2])); print(os.path.basename(e["pack"]), e["bytes"], e["sha256"])' "$warm/qwenmax.nvfp4.tp16.INDEX.json" "$rank") || fail "index-lookup rank$rank"
  read -r name bytes exp <<<"$fields"
  [ -s "$warm/$name" ] || fail "no-master $name"
  [ "$(stat -c %s "$warm/$name")" = "$bytes" ] || fail "master-size $name"
  [ -s "$warm/$name.experts" ] || fail "no-experts $name"
  [ -s "$warm/$name.receipt.json" ] || fail "no-receipt $name"
  mkdir -p "$dst"
  if [ -s "$dst/$name" ]
  then
    dsha=$(cap_sha "$dst/$name") || dsha=none
    if [ "$dsha" = "$exp" ]
    then act=ALREADY-PLACED
    fi
  fi
  if [ "$act" = PLACED ]
  then
    cap_cp "$warm/$name" "$dst/$name" || fail "copy $name"
    dsha=$(cap_sha "$dst/$name")
    if [ "$dsha" != "$exp" ]
    then
      rm -f "$dst/$name"
      fail "sha $name exp=$exp got=$dsha"
    fi
  fi
  cap_cp "$warm/$name.experts" "$dst/" || sudo -n cp "$warm/$name.experts" "$dst/"
  cap_cp "$warm/$name.receipt.json" "$dst/" || sudo -n cp "$warm/$name.receipt.json" "$dst/"
  chown "$(id -un):$(id -gn)" "$dst/$name.experts" "$dst/$name.receipt.json" 2>/dev/null || sudo -n chown "$(id -un):$(id -gn)" "$dst/$name.experts" "$dst/$name.receipt.json"
  esha_src=$(sudo -n sha256sum "$warm/$name.experts" | awk '{print $1}')
  esha_dst=$(cap_sha "$dst/$name.experts")
  [ "$esha_src" = "$esha_dst" ] || fail "experts $name src=$esha_src dst=$esha_dst"
  purge_caches
  printf '{"node":"%s","arm":"qwenmax.nvfp4.tp16","rank":%s,"file":"%s","bytes":%s,"sha256":"%s","experts_sha256":"%s","action":"%s","ts":"%s"}\n' "$(hostname -s)" "$rank" "$name" "$bytes" "$dsha" "$esha_dst" "$act" "$(ts)"
}

stripped() {
  local stage="$1" holder="$2" dst="$HOME/sparkdata/qwenmax.pp16-stripped/packs" hdst="/home/$2/sparkdata/qwenmax.pp16-stripped/packs" f="stage$1.qwen38sp"
  local fields bytes exp act=PLACED dsha=none
  fields=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$holder" "cat '$hdst/$f.receipt.json'" | receipt_fields /dev/stdin) || fail "receipt-lookup $f on $holder"
  read -r bytes exp <<<"$fields"
  [ -n "${exp:-}" ] || fail "empty-receipt $f on $holder"
  mkdir -p "$dst"
  if [ -s "$dst/$f" ]
  then
    dsha=$(cap_sha "$dst/$f") || dsha=none
    if [ "$dsha" = "$exp" ]
    then act=ALREADY-PLACED
    fi
  fi
  if [ "$act" = PLACED ]
  then
    cap_scp "$holder:$hdst/$f" "$dst/$f" || fail "scp $f from $holder"
    [ "$(stat -c %s "$dst/$f")" = "$bytes" ] || { rm -f "$dst/$f"; fail "size $f exp=$bytes"; }
    dsha=$(cap_sha "$dst/$f")
    if [ "$dsha" != "$exp" ]
    then
      rm -f "$dst/$f"
      fail "sha $f exp=$exp got=$dsha"
    fi
  fi
  cap_scp "$holder:$hdst/$f.receipt.json" "$dst/$f.receipt.json"
  purge_caches
  printf '{"node":"%s","arm":"qwenmax.pp16-stripped","stage":%s,"file":"%s","bytes":%s,"sha256":"%s","holder":"%s","action":"%s","ts":"%s"}\n' "$(hostname -s)" "$stage" "$f" "$bytes" "$dsha" "$holder" "$act" "$(ts)"
}

verify_file() {
  local f="$1" arm bytes exp dsha
  [ -s "$f" ] || fail "missing $f"
  [ -s "$f.receipt.json" ] || fail "no-receipt $f"
  arm=$(arm_of "$f")
  read -r bytes exp <<<"$(receipt_fields "$f.receipt.json")"
  [ "$(stat -c %s "$f")" = "$bytes" ] || fail "size $f exp=$bytes"
  dsha=$(cap_sha "$f")
  [ "$dsha" = "$exp" ] || fail "sha $f exp=$exp got=$dsha"
  purge_caches
  printf '{"node":"%s","arm":"%s","file":"%s","bytes":%s,"sha256":"%s","action":"VERIFIED","ts":"%s"}\n' "$(hostname -s)" "$arm" "$f" "$bytes" "$dsha" "$(ts)"
}

purge_file() {
  local f="$1" arm="$2" kind="$3" bytes exp dsha
  arm=$(arm_of "$arm")
  if [ ! -s "$f" ]
  then
    printf '{"node":"%s","arm":"%s","file":"%s","bytes":0,"sha256":"","action":"ALREADY-%s","ts":"%s"}\n' "$(hostname -s)" "$arm" "$f" "$kind" "$(ts)"
    return 0
  fi
  [ -s "$f.receipt.json" ] || fail "purge-guard no-receipt $f"
  read -r bytes exp <<<"$(receipt_fields "$f.receipt.json")"
  [ "$(stat -c %s "$f")" = "$bytes" ] || fail "purge-guard size $f exp=$bytes"
  dsha=$(cap_sha "$f")
  [ "$dsha" = "$exp" ] || fail "purge-guard sha $f exp=$exp got=$dsha"
  rm -f "$f" "$f.receipt.json"
  if [ -e "$f.experts" ]
  then rm -f "$f.experts"
  fi
  purge_caches
  printf '{"node":"%s","arm":"%s","file":"%s","bytes":%s,"sha256":"%s","action":"PURGED-%s","ts":"%s"}\n' "$(hostname -s)" "$arm" "$f" "$bytes" "$dsha" "$kind" "$(ts)"
}

case "$MODE" in
tp16) tp16 "$1" ;;
stripped) stripped "$1" "$2" ;;
verify) verify_file "$1" ;;
purge) purge_file "$1" "$2" "$3" ;;
*) fail "bad-mode $MODE" ;;
esac
