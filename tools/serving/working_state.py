import json
import re
import sys
from pathlib import Path

RETENTION_SECONDS = 86400
OWN_ACCOUNT = "sparkpipe"
ACCOUNT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
DIRECTIVE = re.compile(r"sparkpipe-working-state 1\nwiped ([0-9]{1,19})\nheld_until ([0-9]{1,19})\n")


def expired(used, wiped, held_until, now):
    return used < wiped or now >= max(used, held_until) + RETENTION_SECONDS


def parse_directive(text):
    match = DIRECTIVE.fullmatch(text)
    if match is None:
        raise ValueError("not three lines: sparkpipe-working-state 1, wiped <unix seconds>, held_until <unix seconds>")
    return int(match.group(1)), int(match.group(2))


def read_directive(path):
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return 0, 0
    except (OSError, UnicodeDecodeError) as error:
        print(json.dumps({"event": "working_state_directive_malformed", "path": str(path), "error": str(error)}), file=sys.stderr, flush=True)
        return 0, 0
    try:
        return parse_directive(text)
    except ValueError as error:
        print(json.dumps({"event": "working_state_directive_malformed", "path": str(path), "error": str(error)}), file=sys.stderr, flush=True)
        return 0, 0


def valid_account(value):
    return isinstance(value, str) and ACCOUNT.fullmatch(value) is not None
