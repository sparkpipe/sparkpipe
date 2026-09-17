#!/usr/bin/env python3
"""Render the SCOREBOARD section of docs/PERFORMANCE_LEDGER.md.

The ledger is the single source of truth (human-edited, versioned,
receipt-backed). This tool only reads the table between the
scoreboard:start/end markers and prints it compactly — the coordinator
sweep emits the result every 30 minutes so the best measured decode and
prefill per model (and the gaps) are always visible.

Usage:
  perf_scoreboard.py            print the scoreboard
  perf_scoreboard.py --check    exit 1 if markers/table are malformed
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "PERFORMANCE_LEDGER.md")
START = "<!-- scoreboard:start -->"
END = "<!-- scoreboard:end -->"


def load_rows():
    with open(LEDGER) as fh:
        text = fh.read()
    if START not in text or END not in text:
        return None, "markers missing"
    section = text.split(START, 1)[1].split(END, 1)[0]
    rows = [l.strip() for l in section.splitlines() if l.strip().startswith("|")]
    if len(rows) < 3:
        return None, f"table too small ({len(rows)} lines)"
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    if header[:2] != ["Model", "Best decode (tok/s)"]:
        return None, f"unexpected header: {header}"
    body = []
    for line in rows[2:]:
        body.append([c.strip() for c in line.strip("|").split("|")])
    return (header, body), None


def main():
    parsed, err = load_rows()
    if err:
        if "--check" in sys.argv:
            sys.exit(f"SCOREBOARD MALFORMED: {err}")
        print(f"SCOREBOARD MALFORMED: {err}")
        return
    header, body = parsed
    widths = [max(len(r[i]) if i < len(r) else 0 for r in [header] + body)
              for i in range(len(header))]
    for row in [header] + [["-" * w for w in widths]] + body:
        print("  ".join((row[i] if i < len(row) else "").ljust(widths[i])
                        for i in range(len(header))))
    missing = [r[0] for r in body if r[1:2] == ["—"] or (len(r) > 1 and r[1] == "—")]
    if missing:
        print(f"[gaps] no measured decode: {', '.join(missing)}")


if __name__ == "__main__":
    main()
