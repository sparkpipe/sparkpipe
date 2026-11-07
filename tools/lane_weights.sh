lane_weight_check() {
  local host="$1"
  shift
  printf 'for f in %s; do if [ -L "$f" ] || [ ! -f "$f" ]; then echo "%s: %s: $f is missing, not a regular file or a symlink; a lane names its real weight files directly" >&2; exit 3; fi; done' "$*" "$LANE_TOOL" "$host"
}

lane_config_path() {
  printf 'python3 -c "import json, sys; print(next((node[\\"adapter_configuration_path\\"] for node in json.load(open(sys.argv[1]))[\\"nodes\\"] if node[\\"rank_index\\"] == int(sys.argv[2])), \\"(no node entry)\\"))" %s %s' "$1" "$2"
}

lane_config_check() {
  printf 'f="$(%s)" || exit 3; if [ -L "$f" ] || [ ! -f "$f" ]; then echo "%s: %s: rank %s adapter configuration $f is missing, not a regular file or a symlink; the deployment names the real configuration of every rank" >&2; exit 3; fi' "$(lane_config_path "$2" "$3")" "$LANE_TOOL" "$1" "$3"
}
