#!/usr/bin/env bash
# glm5_next closeout C2: swap the fixed packs into the deployment, per rank.
#
# Layout (per LAUNCH-STATE.md): packs/<name>.g5nsp is a SYMLINK to the rank's
# pack file in the node home (old: ~/glm53_packs/, new: ~/glm53_packs_fixed/).
# Rank r deploys to node spark<r-hex> — the deployment is rank-addressed, so
# ranks are the single parameter. The swap renames symlinks only:
# rename(2) is atomic, nothing is copied, and the old pack FILE stays in
# ~/glm53_packs/ for rollback. The first run renames the old symlink to
# .pre-closeout-bak (guarded, so re-runs never overwrite the original backup).
# Serving is unaffected: residentds hold the old file open until the C3 wave.
#
# usage: glm5_next_pack_swap.sh [ranks...]   # default: all 16
set -euo pipefail
RANKS=("$@")
[[ ${#RANKS[@]} -gt 0 ]] || RANKS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

fail=0
for r in "${RANKS[@]}"; do
    h=$(printf "spark%x" "$r")
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "
set -e
rr=\$HOME/sparkdata/glm5_next.tp16
old=\$rr/packs/glm5_next_stage.tp16.rank$r.g5nsp
new=\$HOME/glm53_packs_fixed/glm5_next_stage.tp16.rank$r.g5nsp
newsz=\$(stat -c%s \"\$new\" 2>/dev/null || echo 0)
[ \"\$newsz\" = \"21706046976\" ] || { echo \"FIXED-PACK-BAD size=\$newsz\"; exit 1; }
if [ ! -e \"\$old\" ] && [ ! -L \"\$old\" ]; then echo \"NO-CANONICAL\"; exit 1; fi
if [ ! -e \"\$old.pre-closeout-bak\" ]; then
  mv -T \"\$old\" \"\$old.pre-closeout-bak\"
fi
ln -s \"\$new\" \"\$rr/packs/.swap-in-r$r\"
mv -T \"\$rr/packs/.swap-in-r$r\" \"\$old\"
cursz=\$(stat -c%s \"\$old\")
echo \"SWAPPED r$r -> \$(readlink \"\$old\") (\$cursz B)\"
" 2>&1) || fail=1
    echo "$h: $out"
done
exit $fail
