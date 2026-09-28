import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import mimo26_model_inputs as inputs  # noqa: E402
from t1_reference_common import read_fixture  # noqa: E402

FIXTURES = os.path.join(ROOT, "qualification", "t1_reference", "mimo26")
LAYERS = 48
MOE = [0] + [1] * (LAYERS - 1)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    for prompt in ("capital", "code", "science"):
        _, arrays = read_fixture(os.path.join(FIXTURES, prompt + ".t1r"))
        arrays = dict(arrays)
        positions = len(arrays["prompt_token_ids"]) + len(arrays["generated_token_ids"]) - 1
        count = inputs.require_route_sets(arrays, MOE)
        expect(count == positions * (LAYERS - 1), f"{prompt}: {count} route sets for {positions} positions")
        dropped = f"pos{positions - 1:04d}_layer0047_route_ids"
        del arrays[dropped]
        try:
            inputs.require_route_sets(arrays, MOE)
        except SystemExit as error:
            expect(dropped in str(error), f"{prompt}: the missing route set must be named: {error}")
        else:
            raise AssertionError(f"{prompt}: a fixture without {dropped} must be refused")
    print("PASS mimo26 model inputs: every decoded position and MoE layer carries a reference route set; "
          "a missing set is refused by name")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
