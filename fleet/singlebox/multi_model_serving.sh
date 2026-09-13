#!/usr/bin/env bash
# single-box multi-model serving supervisor.
#
# Runs SEVERAL residentd deployments side by side on ONE box, each with
# its own ports, runtime root, KV backing directory, log file, and PID
# file. Enforces the COORDINATION.md isolation contract at box scale:
#
# - port disjointness is CHECKED before anything starts (preflight);
# - tier=always-on models coexist;
# - at most one tier=big model runs; starting a big one stops the
#   running big model first and leaves always-on models alone;
# - every start stamps SPARKPIPE_RELEASE_GENERATION / _GIT_COMMIT so
#   measurements stay attributable (the repo's measurement-cleanliness
#   rule).
#
# Usage:
#   multi_model_serving.sh init                  render + preflight configs
#   multi_model_serving.sh start  [MODEL...]     start all or some models
#   multi_model_serving.sh stop   [MODEL...]
#   multi_model_serving.sh restart [MODEL...]
#   multi_model_serving.sh status
#   multi_model_serving.sh health MODEL
#
# Everything reads fleet/singlebox/models.conf unless
# SINGLEBOX_MODELS_CONF overrides it.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CONF=${SINGLEBOX_MODELS_CONF:-$HERE/models.conf}
REPO_ROOT=$(cd "$HERE/../.." && pwd)
RESIDENTD=${SINGLEBOX_RESIDENTD:-$REPO_ROOT/build/sparkpipe_model_residentd}
RUN_DIR=${SINGLEBOX_RUN_DIR:-/tmp/singlebox-serving}
GENERATION=$(date -u +%Y%m%d%H%M%S)

log()  { printf '[singlebox] %s\n' "$*"; }
die()  { printf '[singlebox] FATAL: %s\n' "$*" >&2; exit 1; }

# ---- conf parsing: MODEL blocks -> $RUN_DIR/models/<name>.env ----------

parse_conf() {
	[[ -r "$CONF" ]] || die "unreadable conf: $CONF"
	# Idempotent: every parse re-renders from scratch so repeated calls
	# (model_names -> field chains) never trip on their own output.
	rm -rf "$RUN_DIR/models"
	mkdir -p "$RUN_DIR/models"
	local name="" line
	while IFS= read -r line || [[ -n "$line" ]]; do
		case "$line" in
			''|'#'*) continue ;;
			MODEL_NAME=*)
				name=${line#MODEL_NAME=}
				[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] ||
					die "bad MODEL_NAME: $name"
				[[ ! -e "$RUN_DIR/models/$name.env" ]] ||
					die "duplicate MODEL block: $name"
				: > "$RUN_DIR/models/$name.env"
				;;
			*=*)
				[[ -n "$name" ]] || die "key outside a block: $line"
				printf '%s=%s\n' "${line%%=*}" "${line#*=}" >> "$RUN_DIR/models/$name.env"
				;;
		esac
	done < "$CONF"
	[[ -n "$name" ]] || die "no MODEL blocks in $CONF"
}

model_names() {
	parse_conf
	local f
	for f in "$RUN_DIR"/models/*.env; do
		basename "$f" .env
	done
}

require_model() {
	[[ -f "$RUN_DIR/models/$1.env" ]] ||
		die "unknown model: $1 (known: $(model_names | tr '\n' ' '))"
}

field() {
	require_model "$1"
	local value
	value=$(sed -n "s/^$2=//p" "$RUN_DIR/models/$1.env" | head -1)
	[[ -n "$value" ]] || die "model $1 missing $2"
	printf '%s' "$value"
}

pid_file() { printf '%s/%s.pid' "$RUN_DIR" "$1"; }
log_file() { printf '%s/%s.log' "$RUN_DIR" "$1"; }

tier_of() { sed -n 's/^TIER=//p' "$RUN_DIR/models/$1.env"; }

# ---- preflight: ports free, pairwise disjoint, artifacts present ------

port_in_use() {
	bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null
}

preflight() {
	local seen="" m p base i artifact root kvdir
	for m in $(model_names); do
		base=$(field "$m" TRANSPORT_BASE)
		for p in "$(field "$m" CONTROL_PORT)" \
			$((base)) $((base+1)) $((base+2)) $((base+3)) \
			$((base+4)) $((base+5)) $((base+6)) $((base+7)); do
			[[ " $seen " == *" $p "* ]] &&
				die "port $p claimed twice (model $m)"
			port_in_use "$p" &&
				die "port $p already listening on this box (model $m)"
			seen="$seen $p"
		done
		root=$(field "$m" RUNTIME_ROOT)
		kvdir=$(field "$m" KV_BACKING_DIR)
		[[ "$kvdir" == "$root"* ]] ||
			die "model $m: KV_BACKING_DIR must live under RUNTIME_ROOT"
		case "$(tier_of "$m")" in
			always-on|big) ;;
			*) die "model $m: TIER must be always-on or big" ;;
		esac
		for artifact in "$(field "$m" ADAPTER_SO)" \
			"$(field "$m" DRIVER_SO)" "$(field "$m" TRANSPORT_SO)" \
			"$(field "$m" ADAPTER_CONFIG)"; do
			[[ -e "$REPO_ROOT/$artifact" ]] ||
				die "model $m missing artifact: $artifact"
		done
	done
	[[ -x "$RESIDENTD" ]] || die "residentd not built: $RESIDENTD"
	log "preflight OK: ports disjoint and free, artifacts present"
}

# ---- big-model exclusivity -------------------------------------------

is_running() {
	local pf pid
	pf=$(pid_file "$1")
	[[ -f "$pf" ]] || return 1
	pid=$(cat "$pf")
	kill -0 "$pid" 2>/dev/null
}

stop_big_running() {
	local m
	for m in $(model_names); do
		[[ "$(tier_of "$m")" == "big" ]] || continue
		if is_running "$m"; then
			log "big-model exclusivity: stopping $m first"
			stop_one "$m"
		fi
	done
}

# ---- config rendering (schema-2 deployment JSON) ----------------------

render_config() {
	local m=$1 root control base
	root=$(field "$m" RUNTIME_ROOT)
	control=$(field "$m" CONTROL_PORT)
	base=$(field "$m" TRANSPORT_BASE)
	mkdir -p "$root/config" "$(field "$m" KV_BACKING_DIR)"
	cat > "$root/config/model_resident.json" <<JSON
{
  "schema_version": 2,
  "coordinator_rank_index": 0,
  "adapter": {
    "shared_object_path": "$(field "$m" ADAPTER_SO)"
  },
  "driver": {
    "shared_object_path": "$(field "$m" DRIVER_SO)",
    "program_name": "resident_decode"
  },
  "transport": {
    "shared_object_path": "$(field "$m" TRANSPORT_SO)",
    "mode": "host-rdma",
    "control_port_base": $base
  },
  "runtime_limits": {
    "max_inflight_submissions": $(field "$m" MAX_INFLIGHT_SUBMISSIONS),
    "max_active_sequences": $(field "$m" MAX_ACTIVE_SEQUENCES),
    "max_input_rows": $(field "$m" MAX_INPUT_ROWS),
    "resident_sequence_capacity": $(field "$m" RESIDENT_SEQUENCE_CAPACITY),
    "kv_logical_page_capacity": $(field "$m" KV_LOGICAL_PAGE_CAPACITY),
    "kv_physical_page_capacity": $(field "$m" KV_PHYSICAL_PAGE_CAPACITY)
  },
  "nodes": [
    {
      "rank_index": 0,
      "stage_index": 0,
      "runtime_root": "$root",
      "node_target": "singlebox.$m",
      "transport_host": "127.0.0.1",
      "adapter_configuration_path": "$(field "$m" ADAPTER_CONFIG)",
      "kv_backing_directory": "$(field "$m" KV_BACKING_DIR)",
      "kv_backing_maximum_bytes": $(field "$m" KV_BACKING_MAX_BYTES),
      "control_endpoint": {"kind": "tcp", "host": "127.0.0.1", "port": $control}
    }
  ]
}
JSON
}

# ---- lifecycle --------------------------------------------------------

start_one() {
	local m=$1 root env_extra commit pid
	root=$(field "$m" RUNTIME_ROOT)
	if is_running "$m"; then
		log "$m already running (pid $(cat "$(pid_file "$m")"))"
		return 0
	fi
	render_config "$m"
	[[ "$(tier_of "$m")" == "big" ]] && stop_big_running
	env_extra=$(sed -n 's/^EXTRA_ENV=//p' "$RUN_DIR/models/$m.env")
	commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
	(
		cd "$root"
		export LD_LIBRARY_PATH="$root/lib:${LD_LIBRARY_PATH:-}"
		export SPARKPIPE_RELEASE_GENERATION="$GENERATION"
		export SPARKPIPE_RELEASE_GIT_COMMIT="$commit"
		export SPARKPIPE_RELEASE_ID="singlebox-$m"
		# EXTRA_ENV is a shell word list, optionally wrapped in one
		# pair of double quotes in the conf; strip that wrapper and skip
		# the empty case so `EXTRA_ENV=""` stays legal.
		env_extra=${env_extra%\"}
		env_extra=${env_extra#\"}
		if [[ -n "$env_extra" ]]; then
			# shellcheck disable=SC2086 - deliberately word-split
			eval "export $env_extra"
		fi
		exec "$RESIDENTD" --deployment config/model_resident.json \
			--rank-index 0 >> "$(log_file "$m")" 2>&1
	) &
	pid=$!
	echo "$pid" > "$(pid_file "$m")"
	sleep 1
	if ! kill -0 "$pid" 2>/dev/null; then
		rm -f "$(pid_file "$m")"
		die "$m failed to stay up; tail of $(log_file "$m"):
$(tail -5 "$(log_file "$m")" 2>/dev/null || true)"
	fi
	log "started $m (pid $pid, control :$(field "$m" CONTROL_PORT))"
}

stop_one() {
	local m=$1 pf pid i
	pf=$(pid_file "$m")
	if ! is_running "$m"; then
		rm -f "$pf"
		log "$m not running"
		return 0
	fi
	pid=$(cat "$pf")
	kill "$pid" 2>/dev/null || true
	for i in 1 2 3 4 5 6 7 8 9 10; do
		kill -0 "$pid" 2>/dev/null || break
		sleep 1
	done
	kill -9 "$pid" 2>/dev/null || true
	rm -f "$pf"
	log "stopped $m"
}

health_one() {
	local m=$1 control
	control=$(field "$m" CONTROL_PORT)
	if port_in_use "$control"; then
		log "$m HEALTHY: control endpoint :$control accepting"
		return 0
	fi
	log "$m UNHEALTHY: nothing listening on :$control"
	return 1
}

status_all() {
	local m state
	printf '%-24s %-10s %-8s %s\n' MODEL TIER STATE CONTROL
	for m in $(model_names); do
		if is_running "$m"; then state=running; else state=stopped; fi
		printf '%-24s %-10s %-8s :%s\n' "$m" "$(tier_of "$m")" \
			"$state" "$(field "$m" CONTROL_PORT)"
	done
}

usage() { sed -n '2,25p' "$0"; exit 2; }

main() {
	[[ $# -ge 1 ]] || usage
	local action=$1 m
	shift
	parse_conf
	case "$action" in
		init)
			preflight
			for m in $(model_names); do render_config "$m"; done
			log "configs rendered under each RUNTIME_ROOT/config/"
			;;
		start)
			preflight
			if [[ $# -gt 0 ]]; then
				for m in "$@"; do start_one "$m"; done
			else
				for m in $(model_names); do start_one "$m"; done
			fi
			;;
		stop)
			if [[ $# -gt 0 ]]; then
				for m in "$@"; do stop_one "$m"; done
			else
				for m in $(model_names); do stop_one "$m"; done
			fi
			;;
		restart)
			preflight
			for m in "$@"; do stop_one "$m"; start_one "$m"; done
			;;
		status)
			status_all
			;;
		health)
			[[ $# -eq 1 ]] || die "health needs exactly one model"
			health_one "$1"
			;;
		*)
			die "unknown action: $action"
			;;
	esac
}

main "$@"
