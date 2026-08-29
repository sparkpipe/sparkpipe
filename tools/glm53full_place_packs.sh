#!/usr/bin/env bash
# glm53full pack placement — rank r goes to spark r (hex: ranks 10-15 =
# sparka..sparkf) per the fleet-table all-16 policy. Disk-only placement:
# NO daemon contact, NO launches (waves are a later coordinated event).
#
# Usage (from the controller or the holding node):
#   tools/glm53full_place_packs.sh --from <holding-spark> [--packs-rel <dir>] [--ranks "0 1 2 ..."]
# The holding node reads its LOCAL copy and pushes over ssh; every target
# path is ~/sparkdata/glm53full.nvfp4.tp16/packs/. Ranks already present
# with the right byte size are skipped (resumable).
set -euo pipefail

FROM=""
PACKS_REL="sparkdata/glm53full.nvfp4.tp16/packs"
RANKS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
PACK_NAME="glm53full.nvfp4.tp16-rank{rank}.glm52sp"
BYTES=32903038976

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --packs-rel) PACKS_REL="$2"; shift 2 ;;
        --ranks) RANKS="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done
[[ -n "$FROM" ]] || { echo "--from <holding-spark> required" >&2; exit 2; }

hex() { printf '%x' "$1"; }

ssh -o BatchMode=yes "$FROM" bash -s "$PACKS_REL" "$RANKS" "$BYTES" <<'REMOTE'
set -euo pipefail
packs_rel="$1"; ranks="$2"; bytes="$3"
for rank in $ranks; do
    target="spark$(printf '%x' "$rank")"
    name="glm53full.nvfp4.tp16-rank${rank}.glm52sp"
    local_path="$HOME/$packs_rel/$name"
    if [[ ! -s "$local_path" ]]; then
        echo "rank $rank: LOCAL MISSING $local_path, skipped"
        continue
    fi
    remote_dir="\$HOME/$packs_rel"
    # existing + right size = already placed (resumable)
    if ssh -o BatchMode=yes "$target" "test -s $remote_dir/$name && [[ \$(stat -c%s $remote_dir/$name) -eq $bytes ]]" 2>/dev/null; then
        echo "rank $rank -> $target: already placed"
        continue
    fi
    ssh -o BatchMode=yes "$target" "mkdir -p $remote_dir"
    if rsync -q --inplace -e "ssh -o BatchMode=yes" "$local_path" "$target:$remote_dir/$name"; then
        echo "rank $rank -> $target: placed"
    else
        echo "rank $rank -> $target: RSYNC FAILED"
    fi
done
REMOTE
