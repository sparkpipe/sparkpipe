"""Every device property is its own bit, and the device layer's guard names them all.

include/sparkpipe/spark_device.h defines the SPARK_DEVICE_PROPERTY_* bits that
backends report. Two pull requests that each add a property can pick the same
bit and merge without a textual conflict. runtime/device/device.c asserts at
compile time that the properties it lists are single, distinct bits; this test
checks the header's values the same way and that the list in device.c names
every property the header defines, so a new property cannot skip the guard.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "include/sparkpipe/spark_device.h"
DEVICE = ROOT / "runtime/device/device.c"


def main():
    failures = []
    defined = {name: int(value) for name, value in re.findall(r"^#define SPARK_DEVICE_PROPERTY_(\w+) (\d+)u$", HEADER.read_text(), re.M)}
    if not defined:
        failures.append("no SPARK_DEVICE_PROPERTY_ defines found in the header")
    owners = {}
    for name, value in sorted(defined.items(), key=lambda item: item[1]):
        if value == 0 or value & (value - 1):
            failures.append(f"SPARK_DEVICE_PROPERTY_{name} is {value}, not a single bit")
        elif value in owners:
            failures.append(f"SPARK_DEVICE_PROPERTY_{name} and SPARK_DEVICE_PROPERTY_{owners[value]} share bit {value}")
        else:
            owners[value] = name
    match = re.search(r"^#define DEVICE_PROPERTIES\(APPLY\)((?:.*\\\n)*.*)$", DEVICE.read_text(), re.M)
    listed = set(re.findall(r"APPLY\((\w+)\)", match.group(1))) if match else set()
    if not match:
        failures.append("runtime/device/device.c has no DEVICE_PROPERTIES list")
    for name in sorted(set(defined) - listed):
        failures.append(f"SPARK_DEVICE_PROPERTY_{name} is missing from DEVICE_PROPERTIES in runtime/device/device.c")
    for name in sorted(listed - set(defined)):
        failures.append(f"DEVICE_PROPERTIES lists {name}, which the header does not define")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print(f"PASS {len(defined)} device properties are distinct single bits, and the device layer's compile-time guard names every one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
