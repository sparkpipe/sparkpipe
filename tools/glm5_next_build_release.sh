#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 0 ]; then
    printf '%s\n' 'glm5_next_build_release.sh takes no arguments' >&2
    exit 2
fi
exec bash "$(dirname "$0")/module_build_release.sh" glm5_next_resident_decode_stage fp8 glm53_release 84c6a6aa9497188e15a635ba793b0f95a79b1033 model_contracts/glm53_flash_authoritative.json
