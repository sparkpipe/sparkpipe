import math
import subprocess
import tempfile
from pathlib import Path


class MacroProbeError(Exception):
    pass


def c_macro_values(include_dirs, header, expressions, defines=()):
    expressions = list(dict.fromkeys(expressions))
    body = "".join(f'\tprintf("%.17g\\n",(double)({expression}));\n'
                   for expression in expressions)
    source = (f'#include <math.h>\n#include <stdint.h>\n#include <stdio.h>\n'
              f'#include "{header}"\n\nint main(void)\n'
              f'{{\n{body}\treturn(0);\n}}\n')
    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / "probe.c"
        binary = Path(directory) / "probe"
        probe.write_text(source)
        command = ["cc", "-std=c11", "-Wall", "-Werror",
                   *[f"-I{path}" for path in include_dirs],
                   *[f"-D{define}" for define in defines],
                   str(probe), "-lm", "-o", str(binary)]
        built = subprocess.run(command, capture_output=True, text=True)
        if built.returncode != 0:
            raise MacroProbeError(f"macro probe does not compile:\n{built.stderr}")
        ran = subprocess.run([str(binary)], capture_output=True, text=True)
        if ran.returncode != 0:
            raise MacroProbeError(f"macro probe failed: {ran.stderr}")
    values = [float(line) for line in ran.stdout.split()]
    if len(values) != len(expressions):
        raise MacroProbeError(f"macro probe printed {len(values)} values "
                              f"for {len(expressions)} expressions")
    return dict(zip(expressions, values))


def value_matches(expected, actual):
    if isinstance(expected, bool):
        expected = int(expected)
    if isinstance(expected, int):
        return actual == float(expected)
    return math.isclose(float(expected), actual, rel_tol=1e-7, abs_tol=0.0)
