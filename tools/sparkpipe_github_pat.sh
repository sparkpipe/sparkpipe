#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
pat_file=$repo_root/.env
if [ ! -r "$pat_file" ]; then
    printf '%s\n' "missing SparkPipe PAT file: $pat_file" >&2
    exit 2
fi

if [ "${SPARKPIPE_GIT_ASKPASS:-0}" = 1 ]; then
    case "${1:-}" in
    *Username*)
        printf '%s\n' x-access-token
        ;;
    *Password*)
        awk -F= '$1 == "GITHUB_PAT" {sub(/^[^=]*=/, ""); print; exit}' "$pat_file"
        ;;
    *)
        exit 1
        ;;
    esac
    exit 0
fi

github_pat=$(awk -F= '$1 == "GITHUB_PAT" {sub(/^[^=]*=/, ""); print; exit}' "$pat_file")
if [ -z "$github_pat" ]; then
    printf '%s\n' "missing GITHUB_PAT in $pat_file" >&2
    exit 3
fi

if [ "$#" -lt 2 ]; then
    printf '%s\n' "usage: $0 git|gh command [arguments...]" >&2
    exit 4
fi

client=$1
shift
case "$client" in
gh)
    GH_TOKEN=$github_pat
    export GH_TOKEN
    exec gh "$@"
    ;;
git)
    script_path=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
    SPARKPIPE_GIT_ASKPASS=1
    GIT_ASKPASS=$script_path
    GIT_TERMINAL_PROMPT=0
    export SPARKPIPE_GIT_ASKPASS GIT_ASKPASS GIT_TERMINAL_PROMPT
    exec git -c credential.helper= -c core.askPass="$script_path" "$@"
    ;;
*)
    printf '%s\n' "unsupported GitHub client: $client" >&2
    exit 5
    ;;
esac
