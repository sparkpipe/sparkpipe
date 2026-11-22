"""Host checks of the fused kernel generator, tools/fused_kernels.py.

Every committed fusion plan (model-families/<model>/fusion/*.json) must
regenerate its header, exactness test and manifest byte for byte; --check must
fail when a generated file differs from its plan and pass once it is
regenerated; and the generator must refuse plans that break its rules: an
unknown op, an op missing a parameter or given one it does not take, a binding
to a name that is not an argument, a name that is not a C identifier of at most
64 characters, a generated file in shared code or outside the model families
and the tests, a perturbed input or an output that is not an argument, a tail
without zeroed uint32_t arrival counters or one that runs neither per row nor
once per launch, more than 16 rows, a block size that is not a whole number of
warps, an argument given twice, a model op that shadows a shared one or whose
body is in shared code, a scalar argument given a fill or both a value and a
launch-derived value, an index fill without its expression, a constant or
scalar output, a perturbed scalar, a bump that is not a numeric literal, a
read-only scratch buffer and a row-0 region that is neither a list of regions
nor none.
"""
import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "fused_kernels.py"
SPEC = importlib.util.spec_from_file_location("fused_kernels", TOOL)
fused_kernels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fused_kernels)
PLANS = sorted(path for path in ROOT.glob("model-families/*/fusion/*.json") if not path.name.endswith(".manifest.json"))


def tail_role(kernel):
    return next(role for role in kernel["roles"] if "tail" in role)


def drop_counters(kernel):
    kernel["arguments"] = [argument for argument in kernel["arguments"] if argument["name"] != "counters"]


def set_counter_fill(kernel, fill):
    next(argument for argument in kernel["arguments"] if argument["name"] == "counters")["fill"] = fill


def argument_of(plan, kind):
    for kernel in plan["kernels"]:
        for argument in kernel["arguments"]:
            if kind(argument):
                return kernel, argument
    raise AssertionError("no committed plan has the argument this refusal needs")


def model_op(plan):
    return next(iter(plan["ops"].values()))


def scalar_fill(plan):
    argument_of(plan, lambda argument: "value" in argument)[1]["fill"] = "zero"


def index_without_expression(plan):
    argument_of(plan, lambda argument: argument.get("fill") == "index")[1].pop("expression")


def constant_output(plan):
    kernel, argument = argument_of(plan, lambda argument: argument.get("type", "").startswith("const "))
    kernel["outputs"] = [argument["name"]]


def scalar_perturb(plan):
    kernel, argument = argument_of(plan, lambda argument: "value" in argument)
    kernel["perturb"] = argument["name"]


def derived_and_value(plan):
    argument_of(plan, lambda argument: "derived" in argument)[1]["value"] = "0u"


def bad_row_zero(plan):
    argument_of(plan, lambda argument: argument.get("row_zero") == "none")[1]["row_zero"] = "all"


def constant_scratch(plan):
    next(kernel for kernel in plan["kernels"] if kernel.get("scratch"))["scratch"][0]["type"] = "const uint16_t *"


REFUSALS = (
    ("unknown op", lambda plan, kernel: kernel["roles"][0].update(op="no_such_op")),
    ("needs fn", lambda plan, kernel: kernel["roles"][0]["bind"].pop("fn")),
    ("takes no extra", lambda plan, kernel: kernel["roles"][0]["bind"].update(extra="1u")),
    ("missing is not an argument", lambda plan, kernel: kernel["roles"][0]["bind"].update(fn="arguments.missing")),
    ("not a C identifier of at most 64 characters", lambda plan, kernel: kernel.update(name="K" * 65)),
    ("is in shared code", lambda plan, kernel: plan.update(header="inference/kernels/fused/generated.cuh")),
    ("outside the model families and the tests", lambda plan, kernel: plan.update(header="tools/generated.cuh")),
    ("perturb names an input argument", lambda plan, kernel: kernel.update(perturb="missing")),
    ("bump is a numeric literal", lambda plan, kernel: kernel.update(bump="scores[0]")),
    ("outputs name arguments", lambda plan, kernel: kernel.update(outputs=["missing"])),
    ("needs the counters argument", lambda plan, kernel: drop_counters(kernel)),
    ("counters are uint32_t * and start at zero", lambda plan, kernel: set_counter_fill(kernel, "f32")),
    ("a tail runs for its row", lambda plan, kernel: tail_role(kernel)["tail"].update(scope="grid")),
    ("rows_max is 1 to 16", lambda plan, kernel: kernel.update(rows_max=17)),
    ("threads is a multiple of 32", lambda plan, kernel: kernel.update(threads=100)),
    ("twice", lambda plan, kernel: kernel["arguments"].append(dict(kernel["arguments"][0]))),
    ("is in the shared catalogue already", lambda plan, kernel: plan["ops"].update(hc_sinkhorn_row=model_op(plan))),
    ("declares only its model's own ops", lambda plan, kernel: model_op(plan).update(source="inference/kernels/hc.cuh")),
    ("takes a value expression only", lambda plan, kernel: scalar_fill(plan)),
    ("is filled from an expression of its index", lambda plan, kernel: index_without_expression(plan)),
    ("outputs name arguments", lambda plan, kernel: constant_output(plan)),
    ("perturb names an input argument", lambda plan, kernel: scalar_perturb(plan)),
    ("is a writable pointer with an element count", lambda plan, kernel: constant_scratch(plan)),
    ("or a launch-derived one", lambda plan, kernel: derived_and_value(plan)),
    ("or is none for an output of the whole launch", lambda plan, kernel: bad_row_zero(plan)),
)


def check_committed(failures):
    if not PLANS:
        failures.append("no fusion plan is committed")
    for plan in PLANS:
        result = subprocess.run([sys.executable, "-I", str(TOOL), str(plan), "--check"], capture_output=True, text=True)
        if result.returncode != 0:
            failures.append("%s does not regenerate its committed files: %s" % (plan.relative_to(ROOT), result.stderr.strip()))


def check_refusals(failures):
    plan = json.loads(PLANS[0].read_text())
    with tempfile.TemporaryDirectory() as directory:
        for expected, mutate in REFUSALS:
            broken = copy.deepcopy(plan)
            mutate(broken, broken["kernels"][0])
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(broken))
            try:
                fused_kernels.load(path)
                failures.append("a plan was accepted that should be refused for %r" % expected)
            except fused_kernels.PlanError as error:
                if expected not in str(error):
                    failures.append("a plan refused for %r was refused for another reason: %s" % (expected, error))


def check_stale(failures):
    plan = json.loads(PLANS[0].read_text())
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
        root = Path(directory)
        path = root / "plan.json"
        path.write_text(json.dumps(plan))
        saved, argv = fused_kernels.ROOT, sys.argv
        try:
            fused_kernels.ROOT = root
            sys.argv = ["fused_kernels.py", str(path)]
            if fused_kernels.main() != 0:
                failures.append("generation into an empty tree failed")
            sys.argv = ["fused_kernels.py", str(path), "--check"]
            if fused_kernels.main() != 0:
                failures.append("--check failed right after generation")
            header = root / plan["header"]
            header.write_text(header.read_text() + "\n")
            if fused_kernels.main() != 1:
                failures.append("--check passed with a generated header that differs from its plan")
            sys.argv = ["fused_kernels.py", str(path)]
            fused_kernels.main()
            sys.argv = ["fused_kernels.py", str(path), "--check"]
            if fused_kernels.main() != 0:
                failures.append("--check failed after regeneration")
        finally:
            fused_kernels.ROOT, sys.argv = saved, argv


def main():
    failures = []
    check_committed(failures)
    if PLANS:
        check_refusals(failures)
        check_stale(failures)
    for failure in failures:
        print("FAIL " + failure)
    if not failures:
        print("PASS fused kernel generator")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
