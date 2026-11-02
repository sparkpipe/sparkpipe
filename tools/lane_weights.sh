lane_weight_check() {
  local host="$1"
  shift
  printf 'for f in %s; do if [ -L "$f" ] || [ ! -f "$f" ]; then echo "%s: %s: $f is missing, not a regular file or a symlink; a lane names its real weight files directly" >&2; exit 3; fi; done' "$*" "$LANE_TOOL" "$host"
}
