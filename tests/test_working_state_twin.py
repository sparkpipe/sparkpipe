#!/usr/bin/env python3
import importlib.util
import itertools
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("working_state", ROOT / "tools" / "serving" / "working_state.py")
working_state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(working_state)

PROBE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sparkpipe/spark_working_state.h"
int main(int argc, char **argv) {
    char line[512];
    if (argc != 2) return 2;
    if (strcmp(argv[1], "rule") == 0) {
        unsigned long long wiped, held, storage, used, now;
        while (scanf("%llu %llu %llu %llu %llu", &wiped, &held, &storage, &used, &now) == 5) {
            SparkWorkingStateDirective directive = {wiped, held, storage};
            printf("%u %u\n", SparkWorkingStateExpired(&directive, used, now), SparkWorkingStateStorageExpired(&directive, used, now));
        }
        return 0;
    }
    while (fgets(line, sizeof(line), stdin) != 0) {
        size_t length = strlen(line);
        char text[512];
        size_t out = 0, index;
        SparkWorkingStateDirective directive;
        SparkStatus status;
        for (index = 0; index < length && line[index] != '\n'; index++) {
            if (line[index] == '|') text[out++] = '\n';
            else text[out++] = line[index];
        }
        status = SparkWorkingStateDirectiveParse(text, (uint32_t)out, &directive);
        printf("%d %llu %llu %llu\n", status == SPARK_STATUS_OK ? 1 : 0, (unsigned long long)directive.wiped_seconds,
               (unsigned long long)directive.held_until_seconds, (unsigned long long)directive.storage_wiped_seconds);
    }
    return 0;
}
"""

TEXTS = [
    "sparkpipe-working-state 2|wiped 1764550800|held_until 1764554400|storage_wiped 1764550800|",
    "sparkpipe-working-state 2|wiped 0|held_until 0|storage_wiped 0|",
    "sparkpipe-working-state 2|wiped 100|held_until 0|storage_wiped 50|",
    "sparkpipe-working-state 2|wiped 50|held_until 0|storage_wiped 100|",
    "sparkpipe-working-state 1|wiped 1|held_until 2|",
    "sparkpipe-working-state 2|wiped 1|held_until 2|",
    "sparkpipe-working-state 2|wiped 01|held_until 2|storage_wiped 1|",
    "sparkpipe-working-state 2|wiped 1|held_until 2|storage_wiped 1",
    "sparkpipe-working-state 2|wiped 1|held_until 2|storage_wiped 1|extra 3|",
    "sparkpipe-working-state 2|wiped 1099511627776|held_until 2|storage_wiped 1|",
    "sparkpipe-working-state 2|wiped 1099511627777|held_until 2|storage_wiped 1|",
    "sparkpipe-working-state 2|held_until 2|wiped 1|storage_wiped 1|",
    "sparkpipe-working-state 2|wiped  1|held_until 2|storage_wiped 1|",
]


class WorkingStateTwin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="working-state-twin-")
        probe = Path(cls.build.name) / "probe.c"
        probe.write_text(PROBE)
        cls.binary = Path(cls.build.name) / "probe"
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-D_DARWIN_C_SOURCE", "-Wall", "-Wextra", "-Werror", "-Iinclude",
                        "cache/working_state.c", "src/spark_status.c", str(probe), "-o", str(cls.binary)], cwd=ROOT, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def test_rules_agree(self):
        day = working_state.RETENTION_SECONDS
        base = 2_000_000_000
        cases = []
        for wiped, held, storage in [(0, 0, 0), (base - 600, 0, base - 600), (base - 600, 0, base - 7200), (0, base + 1800, 0),
                                     (base - 600, base - 3600, base - 600), (base + 1, 0, 0)]:
            for used, now in itertools.product((0, base - day - 1, base - day, base - 7201, base - 7200, base - 601, base - 600, base - 1, base),
                                               (base, base + day - 7200, base + day - 1, base + day, base + day + 1800, base + 3 * day)):
                cases.append((wiped, held, storage, used, now))
        text = "".join(" ".join(str(value) for value in case) + "\n" for case in cases)
        output = subprocess.run([str(self.binary), "rule"], input=text, capture_output=True, text=True, check=True).stdout.split("\n")
        for case, line in zip(cases, output):
            wiped, held, storage, used, now = case
            expected = f"{int(working_state.expired(used, wiped, held, now))} {int(working_state.storage_expired(used, storage, held, now))}"
            self.assertEqual(line, expected, case)
        self.assertTrue(any(line == "1 0" for line in output), "a cache-only deletion must expire cache and keep storage in some case")

    def test_parsers_agree(self):
        output = subprocess.run([str(self.binary), "parse"], input="".join(text + "\n" for text in TEXTS), capture_output=True, text=True,
                                check=True).stdout.split("\n")
        for text, line in zip(TEXTS, output):
            try:
                parsed = working_state.parse_directive(text.replace("|", "\n"))
                expected = "1 " + " ".join(str(value) for value in parsed)
            except ValueError:
                expected = "0 0 0 0"
            self.assertEqual(line, expected, text)
        self.assertEqual(working_state.parse_directive(working_state.directive_text(9, 8, 7)), (9, 8, 7))

    def test_read_directive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "directive"
            self.assertEqual(working_state.read_directive(path), (0, 0, 0))
            path.write_text(working_state.directive_text(100, 200, 50))
            self.assertEqual(working_state.read_directive(path), (100, 200, 50))
            path.write_text("sparkpipe-working-state 2\nwiped 100\n")
            self.assertEqual(working_state.read_directive(path), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
