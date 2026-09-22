#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/../tools/module_build_release.sh" "$@"
