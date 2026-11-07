#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CXX = os.environ.get("CXX", "c++")

PROGRAM = r"""
#include <cstdio>
#include <string>
#include <vector>
#include "sparkpipe/spark_motif_family.h"

static std::vector<std::string> trace;
static uint32_t fail_at = UINT32_MAX;

struct Recorder
{
	struct State {};
	struct Frame { uint32_t calls; };
	static SparkStatus Note(const char *name, int layer = -1)
	{
		trace.push_back(layer < 0 ? std::string(name) : std::string(name) + " " + std::to_string(layer));
		return trace.size() - 1u == fail_at ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK;
	}
	static SparkStatus Begin(State *, const SparkStageRunnerStep *, void *, Frame *frame) { frame->calls = 0u; return Note("begin"); }
	static SparkStatus HyperBegin(State *, Frame &) { return Note("hyper-begin"); }
	static void BindLayer(State *, Frame &, uint32_t layer) { Note("bind", (int)layer); }
	static SparkStatus HyperNorm(State *, Frame &, uint32_t layer, uint32_t part) { return Note(part == MOTIF_ATTENTION_PART ? "norm-attn" : "norm-ffn", (int)layer); }
	static SparkStatus WindowAttention(State *, Frame &, uint32_t layer) { return Note("window", (int)layer); }
	static SparkStatus FullAttention(State *, Frame &, uint32_t layer) { return Note("full", (int)layer); }
	static SparkStatus AllReduce(State *, Frame &) { return Note("all-reduce"); }
	static SparkStatus HyperPost(State *, Frame &, uint32_t layer, uint32_t part) { return Note(part == MOTIF_ATTENTION_PART ? "post-attn" : "post-ffn", (int)layer); }
	static SparkStatus RoutedMlp(State *, Frame &, uint32_t layer) { return Note("routed", (int)layer); }
	static SparkStatus DenseMlp(State *, Frame &, uint32_t layer) { return Note("dense", (int)layer); }
	static SparkStatus HyperEnd(State *, Frame &) { return Note("hyper-end"); }
};

int main(int argc, char **argv)
{
	if ( argc > 1 )
		fail_at = (uint32_t)std::stoul(argv[1]);
	Recorder::State state;
	SparkStageRunnerStep step = {};
	const SparkStatus status = MotifSchedule<void, Recorder>::Step(&state, &step, nullptr);
	std::printf("status %d\n", (int)status);
	for ( const std::string &line : trace )
		std::printf("%s\n", line.c_str());
	return 0;
}
"""


class MotifScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        source = Path(cls.directory.name) / "schedule.cpp"
        source.write_text(PROGRAM)
        cls.binary = Path(cls.directory.name) / "schedule"
        build = subprocess.run([CXX, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-Iinclude", "-Imodel-families/motif/include", str(source), "-o",
                                str(cls.binary)], cwd=ROOT, capture_output=True, text=True)
        if build.returncode != 0:
            raise AssertionError(build.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_schedule(self, fail_at=None):
        arguments = [str(self.binary)] + ([str(fail_at)] if fail_at is not None else [])
        lines = subprocess.run(arguments, capture_output=True, text=True, check=True).stdout.splitlines()
        return int(lines[0].split()[1]), lines[1:]

    def test_layer_order_and_kinds(self):
        status, trace = self.run_schedule()
        self.assertEqual(status, 0)
        self.assertEqual(trace[:2], ["begin", "hyper-begin"])
        self.assertEqual(trace[-1], "hyper-end")
        body = trace[2:-1]
        self.assertEqual(len(body), 53 * 9)
        for layer in range(53):
            ops = body[layer * 9:(layer + 1) * 9]
            attention = "full" if layer % 4 == 0 else "window"
            mlp = "dense" if layer < 2 else "routed"
            self.assertEqual(ops, [f"bind {layer}", f"norm-attn {layer}", f"{attention} {layer}", "all-reduce", f"post-attn {layer}",
                                   f"norm-ffn {layer}", f"{mlp} {layer}", "all-reduce", f"post-ffn {layer}"], layer)

    def test_full_layers_are_every_fourth(self):
        _, trace = self.run_schedule()
        self.assertEqual([int(line.split()[1]) for line in trace if line.startswith("full ")], list(range(0, 53, 4)))

    def test_a_failure_stops_the_step(self):
        _, healthy = self.run_schedule()
        failing = healthy.index("window 1")
        status, trace = self.run_schedule(failing)
        self.assertNotEqual(status, 0)
        self.assertEqual(trace[-1], "window 1")
        self.assertNotIn("hyper-end", trace)


if __name__ == "__main__":
    sys.exit(unittest.main())
