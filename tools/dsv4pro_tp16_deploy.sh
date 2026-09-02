#!/usr/bin/env bash
# DSV4 Pro TP16 pack deploy: place the verified rank pack on its node and
# stage the TP16 config. Deploy-side only - ranks must already be verified
# (rankN.receipt.json content OK) on warm. Two-pass placement proof: a
# re-run reports "already placed" (sha equality) instead of re-shipping.
# usage: dsv4pro_tp16_deploy.sh --rank N [--target HOST]
set -euo pipefail

rank=""
target=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank) rank="$2"; shift 2 ;;
        --target) target="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "${rank}" || ! "${rank}" =~ ^(1[0-5]|[0-9])$ ]]; then
    echo "--rank N (0..15) is required" >&2
    exit 2
fi
[[ -n "${target}" ]] || target="$(printf 'spark%x' "${rank}")"

warm="/mnt/model-warm/packbuild/dsv4pro-tp16"
pack_name="dsv4_pro_tp16_stage.spstage"
remote_dir="/home/${target}/sparkdata/dsv4_pro.tp16"
receipt="${warm}/rank${rank}.receipt.json"

python3 -c "import json,sys;d=json.load(open('$receipt'));sys.exit(0 if d.get('content')=='OK' and d.get('directory')=='OK' else 1)" \
    || { echo "RANK${rank}-NOT-VERIFIED ($receipt)"; exit 1; }
want=$(awk '{print $1}' "${warm}/rank${rank}.sha")

ssh -o BatchMode=yes "${target}" "mkdir -p '${remote_dir}/packs' '${remote_dir}/config' '${remote_dir}/logs/kvcache'"
remote_pack="${remote_dir}/packs/${pack_name}"
if ssh -o BatchMode=yes "${target}" "test -s '${remote_pack}'"; then
    have=$(ssh -o BatchMode=yes "${target}" "sha256sum '${remote_pack}'" | awk '{print $1}')
    if [[ "$have" == "$want" ]]; then
        echo "RANK${rank}-ALREADY-PLACED ${target} sha=${have}"
    else
        echo "RANK${rank}-SHA-DRIFT on ${target}: have=${have} want=${want}"
        exit 1
    fi
else
    rsync -a --info=progress2 "${warm}/rank${rank}.spstage" "${target}:${remote_pack}"
    have=$(ssh -o BatchMode=yes "${target}" "sha256sum '${remote_pack}'" | awk '{print $1}')
    [[ "$have" == "$want" ]] || { echo "RANK${rank}-SHA-MISMATCH after copy"; exit 1; }
    echo "RANK${rank}-PLACED ${target} sha=${have}"
fi

# Stage the TP16 deployment config on every visited node (idempotent).
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -q "${repo}/examples/deployments/dsv4_pro_tp16_stage.json" \
    "${target}:${remote_dir}/config/dsv4_pro_tp16_stage.json"
echo "RANK${rank}-CONFIG-STAGED ${target}"
