"""Run the quantisation calibration self-test under the standard runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import sparkpipe_quant_calibrate as calibrate

raise SystemExit(calibrate.self_test())
