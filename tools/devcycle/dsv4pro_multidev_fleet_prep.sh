#!/bin/bash
# dsv4_pro M1 fleet prep (lane 5): uniform pack rename + .sha256 + .experts
# sidecars in every runtime root, run as a per-node queue CPU job from the
# synced lane checkout. Idempotent; writes only inside the runtime root.
#
# IMPORTANT (queue contract, learned the hard way): submit with
#   --cmd 'bash tools/devcycle/dsv4pro_multidev_fleet_prep.sh'
# and NOT --cmd-file: the dispatcher pipes the command text through
# systemd-run, which pre-expands every $VAR/${VAR} in it before bash runs
# (verified: probe dsv4pro-dollar-probe, 2026-09-22). Repo-resident scripts
# with a bare invocation line keep the variables inside real bash.
# The same rule applies to run-dsv4-pro-family-job.sh.
set -euo pipefail
RANK="${SPARK_QUEUE_RANK:?}"
CHECKOUT="$(pwd)"
# The node knows its own rank (HOST is authoritative); SPARK_QUEUE_RANK is
# only the INDEX IN --nodes, which differs from the world rank whenever the
# node list is a subset (receipt dsv4pro-m1-prep10: a 12-node list remapped
# spark5's job onto spark2's root). When the full 16-node list is used the
# two must agree and are cross-checked.
HOST="$(hostname)"
case "$HOST" in
spark*) RANK_HEX="${HOST#spark}"; WORLD_RANK=$((16#$RANK_HEX)) ;;
*) echo "unexpected host name: $HOST" >&2; exit 2 ;;
esac
if [ "$SPARK_QUEUE_SIZE" = "16" ] && [ "$WORLD_RANK" != "$RANK" ]; then
  echo "rank mismatch: queue index $RANK vs host rank $WORLD_RANK" >&2; exit 2
fi
RANK="$WORLD_RANK"
ROOT="/home/${HOST}/sparkdata/dsv4_pro.tp4pp4"
PACKS="$ROOT/packs"
OLD="$(printf 'dsv4_pro.tp4_pp4.rank%02d.spstage' "$RANK")"
NEW="dsv4_pro_tp4_pp4_stage.spstage"

case "$RANK" in
0) SHA=2be8aa0a9258be00a5e255752d870395037804d7f1de5073a61c03d5bce18ed7 ;;
1) SHA=0e9f015877bde1400c2e36b352e646b0792c645e09f3014597f9cbbaefb75512 ;;
2) SHA=24821d5736da788cc1177a9265f9463f86a6e434491c90ba3b3b512138122418 ;;
3) SHA=197348a90316e1b3652e07a3330d818c3440204d93e2fa57855bd3ff787e0f7 ;;
4) SHA=6d39fad9fe6dab2fd03ab72e93047412e18ee33a774dd13245d6a13f119519f7 ;;
5) SHA=81c7468c171cd63a9e5bcb8fa510e3ed2f6cb1bbca417c827edcf45c7e16b96a ;;
6) SHA=a50eb96b3a40decb24454fceab9414492da78c01bf1ab7fd5cd6e4e82bbc4332 ;;
7) SHA=f2aeb49d9c7cafd32542f045ae40a73cf9a0a96956b155203939921a639bfa73 ;;
8) SHA=608f9bb5a6e838e53d8110f96d0aceaa59ea3b37569d5c48ca8eb74d54787472 ;;
9) SHA=298243fe3a0d8a110fdc27391848fcba867021d23dbafbea4b5526df8f97827b ;;
10) SHA=d47dff6de53c8e40c6cb7b7aa3412f79e9f0677db95e3a40d9dd688d2c375c69 ;;
11) SHA=84986831237272be59b3c9003f2d39bcb04bd12d510523a5e29978544d732c84 ;;
12) SHA=bfe6f618061bcb58f3aa2e69aeeab4ba9f11e911836e4876ee6c0735ec7738e6 ;;
13) SHA=89b54c62f5fbf7213c7e366b5b903e68165f3dff2d9dfa1a6797a3fbaa4e4ce3 ;;
14) SHA=a78333103ca57efeb52a7267d1400492b016965f0e3bd4c1b9899d749134460d ;;
15) SHA=d53de7845573d7891622016d53486f5dbdbd5ab62fbe1962a544106aae03e64e ;;
*) echo "bad rank $RANK" >&2; exit 2 ;;
esac

cd "$PACKS"
# The placed rank packs carry the standard immutable flag (chattr +i) and
# the queue job is non-root: do NOT rename or clear the flag. The uniform
# stage_pack_path name resolves through a symlink instead; pack bytes and
# their placement stay untouched.
if [ -L "$NEW" ] && [ -f "$OLD" ]; then
  echo "link: already present ($NEW -> $OLD)"
elif [ ! -e "$NEW" ] && [ -f "$OLD" ]; then
  ln -s "$OLD" "$NEW"
  echo "link: $NEW -> $OLD"
else
  echo "link: unexpected packs state" >&2; ls -la >&2; exit 3
fi

# Header size must equal the receipt's file size before the sidecar goes in.
SIZE_HEADER="$(python3 - "$NEW" <<'PY'
import struct, sys
with open(sys.argv[1], 'rb') as fh:
    h = struct.unpack('<16I2Q', fh.read(80))
print(h[17])
PY
)"
SIZE_FILE="$(stat -Lc %s "$NEW")"
[ "$SIZE_HEADER" = "$SIZE_FILE" ] || { echo "size mismatch: header $SIZE_HEADER file $SIZE_FILE" >&2; exit 4; }
printf '%s  %s\n' "$SHA" "$NEW" > "$NEW.sha256"

cc -O2 -I"$CHECKOUT/include" -o /tmp/dsv4pro_em_"$$" \
  "$CHECKOUT/tools/dsv4_pro_experts_manifest.c" \
  "$CHECKOUT/runtime/spark_weightd_manifest.c" \
  "$CHECKOUT/src/spark_ck128.c"
/tmp/dsv4pro_em_"$$" "$NEW" 2> "$NEW.experts.log"
rm -f /tmp/dsv4pro_em_"$$"

echo "rank=$RANK host=$HOST pack=$NEW bytes=$SIZE_FILE"
echo "sha256_sidecar=$(cat "$NEW.sha256")"
echo "experts_bytes=$(stat -c %s "$NEW.experts") experts_sha256=$(sha256sum "$NEW.experts" | cut -d' ' -f1)"
cat "$NEW.experts.log"
