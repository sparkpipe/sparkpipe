#!/usr/bin/env bash
# Emit one mimo26 rank pack on the local node and place it on its home spark.
# Idempotent across queue TTL rounds: staged payload/scale files resume at
# exact size, assemble/verify/ship run only when the stage is complete.
# Env: MIMO26_ARM (pro|flash) MIMO26_TP MIMO26_RANK MIMO26_HOME (sparkX)
#      MIMO26_CHECKPOINT (default /mnt/model-wimo... set below)
#      MIMO26_DATA (default ~/sparkdata)
set -u
ARM="${MIMO26_ARM:?}"
TP="${MIMO26_TP:?}"
RANK="${MIMO26_RANK:?}"
HOME_NODE="${MIMO26_HOME:?}"
CKPT="${MIMO26_CHECKPOINT:-/mnt/model-warm/mimo-v2.6-pro-rl}"
DATA="${MIMO26_DATA:-$HOME/sparkdata}"
[ "$ARM" = flash ] && CKPT="${MIMO26_CHECKPOINT:-/mnt/model-warm/mimo-v2.6-flash-rl}"
ARM_NAME="mimo26pro.mxfp4.tp8"
LAYER_COUNT=70
WINDOW=10
[ "$ARM" = flash ] && { ARM_NAME="mimo26flash.mxfp4.tp4"; LAYER_COUNT=48; }

EMIT_ROOT="$DATA/$ARM_NAME/emit/rank$RANK"
PACKS="$DATA/$ARM_NAME/packs"
STAGE="$EMIT_ROOT/stage"
PACK="$EMIT_ROOT/rank$RANK.sp"
mkdir -p "$STAGE" "$PACKS"

log() { echo "mimo26-emit[$ARM r$RANK @$(hostname -s)] $*"; }

first=0
while [ "$first" -lt "$LAYER_COUNT" ]; do
  count=$(( WINDOW < LAYER_COUNT - first ? WINDOW : LAYER_COUNT - first ))
  if python3 -u tools/mimo26_stagepack.py --arm "$ARM" --checkpoint "$CKPT" \
      --tp "$TP" --rank "$RANK" --out "$PACK" --stage-dir "$STAGE" \
      --layer-window "$first:$count" --emit >>"$EMIT_ROOT/emit.log" 2>&1; then
    log "window $first:$count staged"
  else
    log "window $first:$count FAILED (see emit.log; resubmit resumes)"
    exit 1
  fi
  first=$(( first + count ))
done

if [ ! -f "$PACK" ]; then
  python3 -u tools/mimo26_stagepack.py --arm "$ARM" --checkpoint "$CKPT" \
    --tp "$TP" --rank "$RANK" --out "$PACK" --stage-dir "$STAGE" --assemble \
    >>"$EMIT_ROOT/emit.log" 2>&1 || { log "assemble FAILED"; exit 1; }
  log "assembled"
fi
python3 -u tools/mimo26_stagepack.py --arm "$ARM" --checkpoint "$CKPT" \
  --tp "$TP" --rank "$RANK" --out "$PACK" --verify >>"$EMIT_ROOT/emit.log" 2>&1 \
  || { log "verify FAILED"; exit 1; }
log "verified byte-exact against $CKPT"

# Shipping runs OUTSIDE the queue job (driver-side): queue participants must
# belong to the job, and the multi-source rsync to a fresh remote needs the
# trailing-slash dir form. MIMO26_NO_SHIP=1 keeps the job to emit/verify.
if [ "${MIMO26_NO_SHIP:-0}" = 1 ]; then
  log "verified; shipping deferred to the driver (MIMO26_NO_SHIP)"
  exit 0
fi

place() {
  local node="$1" dest_base="$2"
  if [ "$(hostname -s)" = "$node" ]; then
    mkdir -p "$dest_base/packs"
    cp "$PACK" "$dest_base/packs/rank$RANK.sp"
    cp "$PACK.sha256" "$dest_base/packs/rank$RANK.sp.sha256"
    cp "$PACK.receipt.json" "$dest_base/packs/rank$RANK.sp.receipt.json"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" "mkdir -p '$dest_base/packs'"
    rsync -a "$PACK" "$PACK.sha256" "$PACK.receipt.json" \
      "$node:$dest_base/packs/" 2>>"$EMIT_ROOT/ship.log" \
      || { log "ship to $node FAILED"; exit 1; }

  fi
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
    "cd '$dest_base/packs' && sha256sum -c rank$RANK.sp.sha256" \
    >>"$EMIT_ROOT/ship.log" 2>&1 || { log "home-node sha check FAILED"; exit 1; }
}

case "$(hostname -s)" in
  "$HOME_NODE") place "$HOME_NODE" "$DATA/$ARM_NAME" ;;
  *) place "$HOME_NODE" "~/sparkdata/$ARM_NAME" ;;
esac
log "PLACED rank$RANK on $HOME_NODE ($DATA/$ARM_NAME/packs/rank$RANK.sp)"
rm -rf "$STAGE"
log "stage cleaned"
