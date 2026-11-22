foreign_owner() {
	local root="$1" accounts="$2"
	shift 2
	find "$root" -xdev "$@" \( -uid +999 -o -gid +999 \) -printf '%U %G %p\n' | awk -v passwd="$accounts/etc/passwd" -v group="$accounts/etc/group" '
		BEGIN {
			while ((getline line < passwd) > 0) { split(line, field, ":"); users[field[3]] = 1 }
			while ((getline line < group) > 0) { split(line, field, ":"); groups[field[3]] = 1 }
		}
		!found && (($1 > 999 && !($1 in users)) || ($2 > 999 && !($2 in groups))) { print substr($0, length($1) + length($2) + 3); found = 1 }'
}
