#!/usr/bin/env bash
# DSV4 Pro TP4xPP4 rank pipeline: shard -> verify -> deploy -> sha-check.
# Runs on the build node (default spark6). Fully parameterized: no hardcoded
# spark host; --target selects the deploy destination for this rank.
#
# usage: dsv4pro_rank_deploy.sh --rank N [--target HOST|--no-deploy|--stash-warm]
#   --rank N          world rank 0..15 (pp_stage=N/4, tp_rank=N%4)
#   --target HOST     deploy destination (default: spark<hex N>)
#   --no-deploy       build+verify only, keep the rank pack in scratch
#   --stash-warm      copy the verified pack to the warm packbuild scratch
#                     instead of a spark deploy (for ranks whose node is out)
set -euo pipefail

rank=""
target=""
mode="deploy"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank) rank="$2"; shift 2 ;;
        --target) target="$2"; mode="deploy"; shift 2 ;;
        --no-deploy) mode="none"; shift ;;
        --stash-warm) mode="warm"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "${rank}" || ! "${rank}" =~ ^(1[0-5]|[0-9])$ ]]; then
    echo "--rank N (0..15) is required" >&2
    exit 2
fi
if [[ "${mode}" == "deploy" && -z "${target}" ]]; then
    target="$(printf 'spark%x' "${rank}")"
fi

build_root="${SPARK_DSV4PRO_BUILD_ROOT:-/home/$(hostname)/sparkbuild/dsv4pro}"
full_pack="${build_root}/dsv4_pro_full.spstage"
pack_name="dsv4_pro_tp4_pp4_stage.spstage"
remote_dir="/home/${target}/sparkdata/dsv4_pro.tp4pp4/packs"
warm_scratch="/mnt/model-warm/packbuild/dsv4pro"
pp_stage=$((rank / 4))
tp_rank=$((rank % 4))
rank_out="${build_root}/ranks/rank${rank}.spstage"
mkdir -p "${build_root}/ranks" "${build_root}/logs"
receipt="${build_root}/ranks/rank${rank}.receipt.json"

[[ -s "${full_pack}" ]] || { echo "missing full pack: ${full_pack}" >&2; exit 1; }

echo "== rank ${rank}: pp_stage=${pp_stage} tp_rank=${tp_rank} target=${target:-none} mode=${mode}"
started=$(date +%s)

python3 "${build_root}/tools/dsv4_tp16_stagepack.py" \
    --input-pack "${full_pack}" \
    --output "${rank_out}" \
    --rank "${tp_rank}" --tp-degree 4 \
    --pp-stages 4 --pp-stage "${pp_stage}" \
    --model pro | tee "${receipt}"

python3 "${build_root}/tools/dsv4_tp16_stagepack.py" \
    --input-pack "${full_pack}" \
    --output "${rank_out}" \
    --rank "${tp_rank}" --tp-degree 4 \
    --pp-stages 4 --pp-stage "${pp_stage}" \
    --model pro --verify-output > /dev/null
echo "verify_output PASS rank=${rank}"

expected_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "${receipt}")

case "${mode}" in
deploy)
    ssh -o BatchMode=yes "${target}" "mkdir -p '${remote_dir}'"
    rsync -a "${rank_out}" "${target}:${remote_dir}/${pack_name}"
    rsync -a "${receipt}" "${target}:${remote_dir}/${pack_name}.receipt.json"
    actual_sha=$(ssh -o BatchMode=yes "${target}" "sha256sum '${remote_dir}/${pack_name}'" | cut -d' ' -f1)
    [[ "${actual_sha}" == "${expected_sha}" ]] || { echo "sha mismatch on ${target}" >&2; exit 1; }
    echo "deployed rank=${rank} -> ${target}:${remote_dir}/${pack_name} sha=${actual_sha}"
    rm -f "${rank_out}"
    ;;
warm)
    mkdir -p "${warm_scratch}"
    rsync -a "${rank_out}" "${warm_scratch}/rank${rank}.spstage"
    rsync -a "${receipt}" "${warm_scratch}/rank${rank}.receipt.json"
    actual_sha=$(sha256sum "${warm_scratch}/rank${rank}.spstage" | cut -d' ' -f1)
    [[ "${actual_sha}" == "${expected_sha}" ]] || { echo "sha mismatch in warm stash" >&2; exit 1; }
    echo "stashed rank=${rank} -> ${warm_scratch}/rank${rank}.spstage sha=${actual_sha}"
    rm -f "${rank_out}"
    ;;
none)
    echo "built+verified rank=${rank} kept at ${rank_out} sha=${expected_sha}"
    ;;
esac

echo "== rank ${rank} done in $(( $(date +%s) - started ))s"
