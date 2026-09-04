#!/usr/bin/env bash
#
# migrate_dev_system.sh TARGET_HOST — bootstrap the DSV4 dev system on a
# Spark node (the MacBook decommission). Clones the repo, checks out the
# DSpark worktree structure, installs the PAT wrapper credential, and
# verifies the pinned devcycle artifacts are reachable.
#
# usage (run FROM the MacBook while it is alive):
#   bash tools/devcycle/migrate_dev_system.sh spark4
set -euo pipefail

TARGET="${1:?usage: migrate_dev_system.sh TARGET_HOST}"
REPO_URL="https://github.com/sparkpipe/sparkpipe.git"
PAT_ENV="/Users/mac/sparkpipe/.env"

if [[ ! -f "$PAT_ENV" ]]; then
    PAT_ENV="/Users/cem/sparkpipe/.env"
fi
[[ -f "$PAT_ENV" ]] || { echo "PAT env missing" >&2; exit 2; }

REMOTE_HOME="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$TARGET" 'echo $HOME')" || { echo "target unreachable" >&2; exit 3; }
DEV_ROOT="$REMOTE_HOME/dsh"

ssh -o BatchMode=yes "$TARGET" "mkdir -p '$DEV_ROOT'" || { echo "mkdir failed" >&2; exit 3; }

# Clone the canonical repo (branch dsv4-dspark-speculative = the continuing work)
ssh -o BatchMode=yes "$TARGET" "
    set -e
    cd '$DEV_ROOT'
    if [[ ! -d sparkpipe ]]; then
        git clone -q --no-tags --single-branch --branch main '$REPO_URL' sparkpipe
    fi
    cd sparkpipe
    git fetch -q origin dsv4-dspark-speculative:refs/remotes/origin/dsv4-dspark-speculative
    if [[ ! -d ../sparkpipe-dsv4 ]]; then
        git worktree add -q ../sparkpipe-dsv4 origin/dsv4-dspark-speculative
    fi
"

# Credential: the PAT wrapper reads GITHUB_PAT from SPARKPIPE_PAT_FILE or the
# session .env. Copy the .env verbatim (never echo its contents).
ssh -o BatchMode=yes "$TARGET" "mkdir -p '$REMOTE_HOME/sparkpipe'"
scp -q -o BatchMode=yes "$PAT_ENV" "$TARGET:$REMOTE_HOME/sparkpipe/.env"

# Verify
ssh -o BatchMode=yes "$TARGET" "
    cd '$DEV_ROOT'/sparkpipe-dsv4
    git log --oneline -1
    tools/sparkpipe_github_pat.sh gh api user --jq .login
    test -x tools/devcycle.sh && echo devcycle-present
    test -f tools/devcycle/drivers/lean-dspark/model_driver.so && echo lean-dspark-driver-present || echo 'NOTE: lean-dspark driver artifact must be rebuilt (fetch from sparkf /tmp/devcycle-build-lean-dspark)'
"

echo "migrate_dev_system: done — dev root $DEV_ROOT on $TARGET"
