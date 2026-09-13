#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 MODEL_DIR BATCH_JSON OUTPUT_DIRECTORY" >&2
    exit 2
fi

model_directory="$1"
batch_json="$2"
output_directory="$3"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
generator="${script_directory}/dsv4_ga_reference_vector.py"
verifier="${script_directory}/verify_dsv4_ga_reference_fixture.py"
reference_python="${PYTHON:-python3}"
scratch_directory="$(mktemp -d)"
staging_directory=""
cleanup() {
    rm -rf "${scratch_directory}"
    if [[ -n "${staging_directory}" ]]; then
        rm -rf "${staging_directory}"
    fi
}
trap cleanup EXIT

if [[ -e "${output_directory}" || -L "${output_directory}" ]]; then
    echo "output directory already exists: ${output_directory}" >&2
    exit 2
fi

for run in first second; do
    "${reference_python}" "${generator}" \
        --model-dir "${model_directory}" \
        --batch-json "${batch_json}" \
        --output-dir "${scratch_directory}/${run}"
done

for artifact in manifest.json prompt_tokens.u32le after_layer_2.bf16le; do
    cmp "${scratch_directory}/first/${artifact}" "${scratch_directory}/second/${artifact}"
done
read -r manifest_sha256 remainder < <(sha256sum "${scratch_directory}/first/manifest.json")
"${reference_python}" "${verifier}" "${scratch_directory}/first" "${manifest_sha256}"

mkdir -p "$(dirname "${output_directory}")"
staging_directory="$(mktemp -d "${output_directory}.staging.XXXXXX")"
cp "${scratch_directory}/first/manifest.json" "${staging_directory}/manifest.json"
cp "${scratch_directory}/first/prompt_tokens.u32le" "${staging_directory}/prompt_tokens.u32le"
cp "${scratch_directory}/first/after_layer_2.bf16le" "${staging_directory}/after_layer_2.bf16le"
"${reference_python}" "${verifier}" "${staging_directory}" "${manifest_sha256}"
if [[ -e "${output_directory}" || -L "${output_directory}" ]]; then
    echo "output directory appeared during generation: ${output_directory}" >&2
    exit 2
fi
mv "${staging_directory}" "${output_directory}"
staging_directory=""
echo "PASS DSV4 GA reference reproducibility vector=$(sha256sum "${output_directory}/after_layer_2.bf16le")"
