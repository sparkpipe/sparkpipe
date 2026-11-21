import json
import re
import sys
from pathlib import Path

OWN_ACCOUNT = "sparkpipe"
ACCOUNT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
DIRECTIVE = re.compile(r"sparkpipe-working-state 2\nwiped ([0-9]{1,19})\n")


def expired(used, wiped):
    return used < wiped


def parse_directive(text):
    match = DIRECTIVE.fullmatch(text)
    if match is None:
        raise ValueError("not two lines: sparkpipe-working-state 2, wiped <unix seconds>")
    return int(match.group(1))


def read_directive(path):
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return 0
    except (OSError, UnicodeDecodeError) as error:
        print(json.dumps({"event": "working_state_directive_malformed", "path": str(path), "error": str(error)}), file=sys.stderr, flush=True)
        return 0
    try:
        return parse_directive(text)
    except ValueError as error:
        print(json.dumps({"event": "working_state_directive_malformed", "path": str(path), "error": str(error)}), file=sys.stderr, flush=True)
        return 0


def valid_account(value):
    return isinstance(value, str) and ACCOUNT.fullmatch(value) is not None
