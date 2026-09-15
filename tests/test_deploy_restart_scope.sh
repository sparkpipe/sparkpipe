#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
AGENT=tools/fleet_node_agent.sh
ROOT=example.fp8.tp16
SB=$(mktemp -d /tmp/deploy_scope.XXXXXX)
trap 'rm -rf "$SB"' EXIT
PASS=0
FAIL=0

ok() {
    PASS=$((PASS + 1))
    echo "PASS: $1"
}
bad() {
    FAIL=$((FAIL + 1))
    echo "FAIL: $1"
}
check() {
    if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1 (got '$2' want '$3')"; fi
}

mkdir -p "$SB/bin" "$SB/release/$ROOT" "$SB/home/sparkdata"
printf '#!/bin/sh\nexec shasum -a 256 "$@"\n' > "$SB/bin/sha256sum"
printf '#!/bin/sh\nexit 0\n' > "$SB/bin/flock"
chmod +x "$SB/bin/sha256sum" "$SB/bin/flock"
export PATH="$SB/bin:$PATH"
export HOME="$SB/home"

hostname() { echo spark0; }
ssh() { :; }
ssh-keyscan() { :; }
scp() { :; }
date() {
    if [ "${1:-}" = "-Is" ]; then command date -u +%Y-%m-%dT%H:%M:%S+0000
    else command date "$@"; fi
}
FETCHLOG="$SB/fetch.log"
: > "$FETCHLOG"
curl() {
    printf '%s\n' "$*" >> "$FETCHLOG"
    command curl "$@"
}

awk '/^echo "\$\$" > "\$PID_FILE"/ {exit} {print}' "$AGENT" > "$SB/extract.sh"
export FLEET_HTTP_RELEASE="file://$SB/release"
source "$SB/extract.sh" "$ROOT" sparkf

UNLOADED=0
STARTED=0
API_DRAINS=0
unload_root() { UNLOADED=$((UNLOADED + 1)); return 0; }
start_root() { STARTED=$((STARTED + 1)); return 0; }
drain_api() { API_DRAINS=$((API_DRAINS + 1)); return 0; }

RELEASE="$SB/release/$ROOT"
NODE="$HOME/sparkdata/$ROOT"

write_manifest() {
    (
        cd "$RELEASE"
        find . -type f ! -name MANIFEST | sed 's|^\./||' | sort |
            while read -r f; do
                printf '%s  %s\n' "$(sha256sum < "$f" | cut -d' ' -f1)" "$f"
            done > MANIFEST
    )
}

seed_node() {
    rm -rf "$NODE"
    cp -R "$RELEASE" "$NODE"
    cp "$RELEASE/MANIFEST" "$NODE/.applied_manifest"
}

fetch_count() { wc -l < "$FETCHLOG" | tr -d ' '; }
fetch_has() { grep -q -- "$1" "$FETCHLOG"; }
reset_fetchlog() { : > "$FETCHLOG"; }

mkdir -p "$RELEASE/bin" "$RELEASE/lib" "$RELEASE/stages/stage_000" "$RELEASE/config"
echo rd-base > "$RELEASE/bin/sparkpipe_model_residentd"
echo api-base > "$RELEASE/bin/sparkpipe_model_api"
echo wght-base > "$RELEASE/bin/sparkpipe_weightd"
echo lib-base > "$RELEASE/lib/hidden_transport.so"
echo drv-base > "$RELEASE/stages/stage_000/model_driver.so"
echo ranks > "$RELEASE/config/model_resident.json"
echo stagecfg-v1 > "$RELEASE/config/stage_00.json"
write_manifest
seed_node

echo "== A. no-op cycle"
sync_root "$ROOT"
check "no-op: zero unload" "$UNLOADED" "0"
check "no-op: zero start" "$STARTED" "0"
check "no-op: only MANIFEST fetched" "$(fetch_count)" "1"

echo "== B. config-only publish -> zero restarts"
reset_fetchlog
echo stagecfg-v2 > "$RELEASE/config/stage_00.json"
write_manifest
sync_root "$ROOT"
check "config: zero unload" "$UNLOADED" "0"
check "config: zero start" "$STARTED" "0"
check "config: zero api drains" "$API_DRAINS" "0"
check "config: fetched exactly 2 urls" "$(fetch_count)" "2"
fetch_has "config/stage_00.json" && ok "config: stage fetched" || bad "config: stage fetched"
fetch_has "sparkpipe_model_residentd" && bad "config: residentd refetched" || ok "config: residentd not refetched"
[ "$(readlink "$NODE/config/stage.json")" = "stage_00.json" ] && ok "config: stage.json symlink relinked" || bad "config: stage.json symlink relinked"
grep -q stagecfg-v2 "$NODE/config/stage_00.json" && ok "config: node file updated" || bad "config: node file updated"

echo "== C. driver .so publish -> exactly one residentd restart"
reset_fetchlog
echo drv-v2 > "$RELEASE/stages/stage_000/model_driver.so"
write_manifest
sync_root "$ROOT"
check "driver: one unload" "$UNLOADED" "1"
check "driver: one start" "$STARTED" "1"
check "driver: zero api drains" "$API_DRAINS" "0"
check "driver: exactly 2 urls fetched" "$(fetch_count)" "2"

echo "== D. api-only publish -> api drain, no residentd restart"
reset_fetchlog
echo api-v2 > "$RELEASE/bin/sparkpipe_model_api"
write_manifest
sync_root "$ROOT"
check "api: zero unload" "$UNLOADED" "1"
check "api: zero start" "$STARTED" "1"
check "api: one api drain" "$API_DRAINS" "1"

echo "== E. weightd-only change in driver root -> nothing restarts"
reset_fetchlog
echo wght-cand > "$RELEASE/bin/sparkpipe_weightd"
write_manifest
sync_root "$ROOT"
check "weightd-root: zero unload" "$UNLOADED" "1"
check "weightd-root: zero start" "$STARTED" "1"
check "weightd-root: zero api drain" "$API_DRAINS" "1"

echo "== F. restart scope precedence (residentd + api + stage)"
reset_fetchlog
echo rd-v2 > "$RELEASE/bin/sparkpipe_model_residentd"
echo api-v3 > "$RELEASE/bin/sparkpipe_model_api"
echo stagecfg-v3 > "$RELEASE/config/stage_00.json"
write_manifest
sync_root "$ROOT"
check "scope: one unload total" "$UNLOADED" "2"
check "scope: one start total" "$STARTED" "2"
check "scope: api drain not incremented" "$API_DRAINS" "1"

echo "== G. restart_scope classifier table"
echo "bin/sparkpipe_model_residentd" > "$SB/scopeA"
check "scope(residentd)" "$(restart_scope "$SB/scopeA")" "root"
echo "bin/sparkpipe_model_api" > "$SB/scopeA"
check "scope(api)" "$(restart_scope "$SB/scopeA")" "api"
echo "config/stage_07.json" > "$SB/scopeA"
check "scope(stage)" "$(restart_scope "$SB/scopeA")" "stage"
echo "bin/sparkpipe_weightd" > "$SB/scopeA"
check "scope(weightd-only)" "$(restart_scope "$SB/scopeA")" "none"
{ echo "config/stage_07.json"; echo "lib/hidden_transport.so"; } > "$SB/scopeA"
check "scope(lib beats stage)" "$(restart_scope "$SB/scopeA")" "root"
check "scope(missing file)" "$(restart_scope "$SB/scopeMissing")" "none"

echo "== H. install_core: weightsd lifecycle contract"
CORE="$SB/release/core"
INSTALLED="$HOME/sparkdata/weightd/sparkpipe_weightd"
mkdir -p "$CORE/bin" "$HOME/sparkdata/weightd" "$HOME/sparkdata/core/bin"
echo wght-installed > "$INSTALLED"
echo wght-cand2 > "$CORE/bin/sparkpipe_weightd"
install -m 755 "$CORE/bin/sparkpipe_weightd" "$HOME/sparkdata/core/bin/sparkpipe_weightd"
cand_sha=$(sha256sum < "$CORE/bin/sparkpipe_weightd" | cut -c1-16)
installed_sha=$(sha256sum < "$INSTALLED")
daemon_ops() { echo $((UNLOADED + STARTED + API_DRAINS)); }
ops_before=$(daemon_ops)
install_core
[ "$(sha256sum < "$INSTALLED")" = "$installed_sha" ] && ok "weightd: no install without WEIGHTSD_BIN" || bad "weightd: no install without WEIGHTSD_BIN"
printf 'deadbeef00000000\n' > "$CORE/WEIGHTSD_BIN"
install_core
[ "$(sha256sum < "$INSTALLED")" = "$installed_sha" ] && ok "weightd: candidate != announced -> no install" || bad "weightd: candidate != announced -> no install"
printf '%s\n' "$cand_sha" > "$CORE/WEIGHTSD_BIN"
install_core
[ "$(sha256sum < "$INSTALLED")" = "$(sha256sum < "$CORE/bin/sparkpipe_weightd")" ] && ok "weightd: announced candidate installed" || bad "weightd: announced candidate installed"
check "weightd: install touched zero daemons" "$(daemon_ops)" "$ops_before"
install_core
check "weightd: already-announced -> no reinstall churn" "$(daemon_ops)" "$ops_before"

echo "== I. report() view schema"
root_pid() { echo 4242; }
root_rss_mb() { echo 512; }
root_log_age_s() { echo 3; }
if report && python3 - "$VIEW/$HOST.json" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
assert data["epoch"] > 0, "epoch missing"
assert "load" in data and "mem_avail_gb" in data, "host fields missing"
root = data["roots"]["example.fp8.tp16"]
assert root["pid"] == 4242, "pid missing"
assert root["rss_mb"] == 512, "rss_mb missing"
assert root["log_age_s"] == 3, "log_age_s missing"
assert root["residentd"] and root["driver"], "shas missing"
print("report JSON schema OK")
PYEOF
then ok "report: fleet_view_serve schema emitted" || bad "report: fleet_view_serve schema emitted"
else bad "report: fleet_view_serve schema emitted"; fi

echo "== J. publish_local.sh: staging + no staging left + MANIFEST consistent"
TREE="$HOME/sparkpipe-build"
PUBREL="$HOME/release/$ROOT"
mkdir -p "$TREE/build/modules/fam/codec" "$TREE/build" "$HOME/sparkdata/out/stages/stage_000" "$PUBREL/config"
echo residentd-bin > "$TREE/build/sparkpipe_model_residentd"
echo api-bin > "$TREE/build/sparkpipe_model_api"
echo transport > "$TREE/build/libhidden_transport_spark_host_rdma_verbs.so"
echo adapter > "$TREE/build/modules/fam/codec/libglm5_next_serving_adapter_codec.so"
echo driver > "$HOME/sparkdata/out/stages/stage_000/model_driver.so"
echo stagecfg-live > "$PUBREL/config/stage_00.json"
echo '{"runtime_root":"sparkdata/example.fp8.tp16"}' > "$PUBREL/model_resident.json"
if bash tools/publish_local.sh fam codec "$ROOT"; then
    ok "publish_local: exited clean"
else
    bad "publish_local: exited clean"
fi
[ -d "$PUBREL/.staging" ] && bad "publish: .staging removed" || ok "publish: .staging removed"
[ -x "$PUBREL/bin/sparkpipe_model_residentd" ] && ok "publish: residentd installed" || bad "publish: residentd installed"
[ -f "$PUBREL/lib/model_serving_adapter.so" ] && ok "publish: adapter installed" || bad "publish: adapter installed"
grep -q "  bin/sparkpipe_model_residentd" "$PUBREL/MANIFEST" && ok "publish: MANIFEST lists artifacts" || bad "publish: MANIFEST lists artifacts"
grep -q "  config/stage_00.json" "$PUBREL/MANIFEST" && ok "publish: MANIFEST covers pre-existing configs" || bad "publish: MANIFEST covers pre-existing configs"
disk=$(sha256sum < "$PUBREL/bin/sparkpipe_model_api" | cut -d' ' -f1)
listed=$(awk '$2=="bin/sparkpipe_model_api" {print $1}' "$PUBREL/MANIFEST")
check "publish: MANIFEST hash matches file" "$listed" "$disk"

echo "== K. publish_core.sh modes + weightsd_announce.sh"
COREHUB="$HOME/release/core"
mkdir -p "$TREE/build" "$TREE/tools"
cp "$AGENT" "$TREE/tools/fleet_node_agent.sh"
echo weightd-new > "$TREE/build/sparkpipe_weightd"
if bash tools/publish_core.sh agent > /dev/null; then ok "core: agent mode ran"; else bad "core: agent mode ran"; fi
[ -x "$COREHUB/bin/fleet_node_agent.sh" ] && ok "core: agent mode publishes agent" || bad "core: agent mode publishes agent"
[ ! -f "$COREHUB/bin/sparkpipe_weightd" ] && ok "core: agent mode never touches weightd" || bad "core: agent mode never touches weightd"
if bash tools/publish_core.sh weightd > /dev/null; then ok "core: weightd mode ran"; else bad "core: weightd mode ran"; fi
[ -f "$COREHUB/bin/sparkpipe_weightd" ] && ok "core: weightd mode publishes candidate" || bad "core: weightd mode publishes candidate"
if bash tools/publish_core.sh 2> /dev/null; then bad "core: missing mode rejected"; else ok "core: missing mode rejected"; fi
if bash tools/weightsd_announce.sh "$COREHUB" > /dev/null; then ok "announce: ran"; else bad "announce: ran"; fi
announced=$(cat "$COREHUB/WEIGHTSD_BIN")
want=$(sha256sum < "$COREHUB/bin/sparkpipe_weightd" | cut -c1-16)
check "announce: WEIGHTSD_BIN = candidate sha16" "$announced" "$want"
bash tools/weightsd_announce.sh "$COREHUB" > /dev/null
check "announce: idempotent" "$(cat "$COREHUB/WEIGHTSD_BIN")" "$want"

echo
echo "harness: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
