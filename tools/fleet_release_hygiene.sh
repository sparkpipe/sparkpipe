#!/usr/bin/env bash
set -euo pipefail
APPLY=0
AGE_DAYS=14
for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=1 ;;
        *[!0-9]*) echo "usage: tools/fleet_release_hygiene.sh [--apply] [AGE_DAYS]" >&2
                  exit 2 ;;
        *) AGE_DAYS="$arg" ;;
    esac
done
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"
HUB="${FLEET_HUB:-sparkf}"

collect_subscribed() {
    local h roots sub
    sub=""
    for h in "${HOSTS[@]}"; do
        roots=$($SSH "$h" "sed -n 's|^ExecStart=.*fleet_node_agent.sh \([^ ]*\).*|\1|p' ~/.config/systemd/user/fleet-agent.service 2>/dev/null" 2>/dev/null || true)
        sub="$sub $roots"
    done
    echo "$sub" | tr ' ' '\n' | awk 'NF' | sort -u | tr '\n' ' '
}

node_scan() {
    local h="$1" sub="$2"
    $SSH "$h" "SUB='$sub' AGE=$((AGE_DAYS * 86400)) bash -s" <<'REMOTE' || true
for d in "$HOME"/sparkdata/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    case " core weightd current $SUB " in *" $n "*) continue ;; esac
    a=$(( $(date +%s) - $(stat -c %Y "$d") ))
    [ "$a" -lt "$AGE" ] && continue
    printf 'ROOT\t%s\t%dd\t%s\t%s\n' "$n" "$((a / 86400))" "$(du -sh "$d" | cut -f1)" "$HOSTNAME"
done
REMOTE
}

node_debris() {
    local h="$1"
    $SSH "$h" "bash -s" <<'REMOTE' || true
cd "$HOME/sparkdata" 2>/dev/null || exit 0
for d in */packs/; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 1 -type f \( -name '*.old' -o -name '*.experts.old' -o -name '*.partial-*' -o -name '*.premtp-old' \) -printf "DEBRIS\t%p\t%s\t$HOSTNAME\n" 2>/dev/null
done
REMOTE
}

hub_scan() {
    local sub="$1"
    $SSH "$HUB" "SUB='$sub' AGE=$((AGE_DAYS * 86400)) bash -s" <<'REMOTE' || true
for d in "$HOME"/release/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    case " core qpn $SUB " in *" $n "*) continue ;; esac
    a=$(( $(date +%s) - $(stat -c %Y "$d") ))
    [ "$a" -lt "$AGE" ] && continue
    printf 'HUBROOT\t%s\t%dd\t%s\t%s\n' "$n" "$((a / 86400))" "$(du -sh "$d" | cut -f1)" "$HOSTNAME"
done
REMOTE
}

packs_immutable() {
    local h="$1" root="$2"
    $SSH "$h" "ROOT='$root' bash -s" <<'REMOTE'
d="$HOME/sparkdata/$ROOT/packs"
[ -d "$d" ] || exit 0
shopt -s nullglob
for f in "$d"/*.sp; do
    a=$(lsattr "$f" 2>/dev/null | awk '{print $1}')
    case "$a" in *i*) ;; *) echo "mutable pack: $f" >&2
        exit 1 ;; esac
done
exit 0
REMOTE
}

SUB=$(collect_subscribed)
echo "subscribed roots (fleet union):$SUB"
echo "age gate: ${AGE_DAYS}d  mode: $([ "$APPLY" = 1 ] && echo APPLY || echo DRY-RUN)"
mapfile -t FINDINGS < <({ for h in "${HOSTS[@]}"; do node_scan "$h" "$SUB"; node_debris "$h"; done; hub_scan "$SUB"; } | sort -u)
if [ "${#FINDINGS[@]}" = 0 ]; then
    echo "nothing stale"
    exit 0
fi
printf '%s\n' "${FINDINGS[@]}"
[ "$APPLY" = 1 ] || { echo "dry run only; rerun with --apply to delete"; exit 0; }
read -r -p "delete the above from the fleet? type DELETE: " reply
[ "$reply" = "DELETE" ] || { echo "aborted"; exit 1; }
for f in "${FINDINGS[@]}"; do
    case "$f" in
        ROOT*|HUBROOT*)
            IFS=$'\t' read -r kind name age size host <<< "$f"
            base=sparkdata
            [ "$kind" = HUBROOT ] && base=release
            echo "deleting $host:~/$base/$name ($size)"
            $SSH "$host" "rm -rf '$HOME/$base/$name'" ;;
        DEBRIS*)
            IFS=$'\t' read -r kind rel size host <<< "$f"
            root=${rel%%/*}
            if packs_immutable "$host" "$root"; then
                echo "deleting debris $host:~/sparkdata/$rel ($size)"
                $SSH "$host" "rm -f '$HOME/sparkdata/$rel'"
            else
                echo "SKIP $host:$rel (pack immutability verify failed)" >&2
            fi ;;
    esac
done
echo "hygiene pass complete"
