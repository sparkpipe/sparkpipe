import argparse
import ipaddress
import sys


def prefixes(lines, country="KR"):
    keys = set()
    for line in lines:
        fields = line.strip().split("|")
        if len(fields) < 7 or line.startswith("#") or fields[1] != country or fields[2] != "ipv4":
            continue
        if fields[6] not in ("allocated", "assigned"):
            continue
        start = int(ipaddress.IPv4Address(fields[3]))
        count = int(fields[4])
        for block in range(start >> 8, (start + count - 1 >> 8) + 1):
            keys.add(block)
    return [str(ipaddress.IPv4Address(block << 8)).rsplit(".", 1)[0] for block in sorted(keys)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("delegated")
    parser.add_argument("output")
    parser.add_argument("--minimum", type=int, default=100000)
    arguments = parser.parse_args()
    with open(arguments.delegated, encoding="ascii", errors="replace") as source:
        keys = prefixes(source)
    if len(keys) < arguments.minimum:
        print(f"site_kr_networks: only {len(keys)} Korean /24 networks in {arguments.delegated}, expected at least {arguments.minimum}", file=sys.stderr)
        return 1
    with open(arguments.output, "w", encoding="ascii") as out:
        out.writelines(f"{key} kr\n" for key in keys)
    print(f"site_kr_networks: {len(keys)} Korean /24 networks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
