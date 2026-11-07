import json
import re
import sys
from pathlib import Path

RETENTION_SECONDS = 86400
OWN_ACCOUNT = "sparkpipe"
ACCOUNT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
DIRECTIVE = re.compile(r"sparkpipe-working-state 2\nwiped (0|[1-9][0-9]{0,12})\nheld_until (0|[1-9][0-9]{0,12})\nstorage_wiped (0|[1-9][0-9]{0,12})\n")
SECONDS_MAX = 1099511627776


def expired(used, wiped, held_until, now):
    return used < wiped or now >= max(used, held_until) + RETENTION_SECONDS


def storage_expired(used, storage_wiped, held_until, now):
    return expired(used, storage_wiped, held_until, now)


def directive_text(wiped, held_until, storage_wiped):
    return f"sparkpipe-working-state 2\nwiped {int(wiped)}\nheld_until {int(held_until)}\nstorage_wiped {int(storage_wiped)}\n"


def parse_directive(text):
    match = DIRECTIVE.fullmatch(text)
    if match is None:
        raise ValueError("not four lines: sparkpipe-working-state 2, wiped <unix seconds>, held_until <unix seconds>, storage_wiped <unix seconds>")
    wiped, held_until, storage_wiped = (int(group) for group in match.groups())
    if max(wiped, held_until, storage_wiped) > SECONDS_MAX:
        raise ValueError(f"a time is past {SECONDS_MAX}")
    if storage_wiped > wiped:
        raise ValueError("storage_wiped is later than wiped; a change of user or a deletion of storage moves both")
    return wiped, held_until, storage_wiped


def read_directive(path):
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return 0, 0, 0
    except (OSError, UnicodeDecodeError) as error:
        print(json.dumps({"event": "working_state_directive_malformed", "path": str(path), "error": str(error)}), file=sys.stderr, flush=True)
        return 0, 0, 0
    try:
        return parse_directive(text)
    except ValueError as error:
        print(json.dumps({"event": "working_state_directive_malformed", "path": str(path), "error": str(error)}), file=sys.stderr, flush=True)
        return 0, 0, 0


def valid_account(value):
    return isinstance(value, str) and ACCOUNT.fullmatch(value) is not None
