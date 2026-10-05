lane_run_id() {
  local run_id="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
  case "$run_id" in
    ''|*[!A-Za-z0-9._-]*) echo "$LANE_TOOL: a run id is one or more of [A-Za-z0-9._-], got '$run_id'" >&2; return 2 ;;
  esac
  printf '%s\n' "$run_id"
}

lane_log_file() {
  printf 'logs/residentd-%s.log\n' "$1"
}

lane_log_prelude() {
  local host="$1" log
  log="$(lane_log_file "$2")"
  printf "mkdir -p logs && if [ -e %s ]; then echo '%s: %s already holds %s; refusing to overwrite it' >&2; exit 2; fi && if [ -f residentd.log ] && [ ! -L residentd.log ]; then mv residentd.log logs/residentd-before-%s.log; fi && ln -sfn %s residentd.log" "$log" "$LANE_TOOL" "$host" "$log" "$2" "$log"
}

lane_archive() {
  local run_id destination="$2" rank host root log failed=0
  run_id="$(lane_run_id "$1")" || return 2
  log="$(lane_log_file "$run_id")"
  mkdir -p "$destination"
  for rank in $(seq 0 $((LANE_RANKS - 1))); do
    host="$(host_of "$rank")"
    root="$(root_of "$host")"
    if scp -q "$host:$root/$log" "$destination/rank$(printf %02d "$rank").log"; then
      $SSH "$host" "rm -f $root/$log"
    else
      echo "$LANE_TOOL: rank $rank log of run $run_id was not copied; it stays on $host" >&2
      failed=1
    fi
  done
  return "$failed"
}
