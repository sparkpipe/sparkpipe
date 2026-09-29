#!/usr/bin/env python3
"""tools/glm52_validate_pack.py mirrors the C stage pack policy exactly.

The packer tests check tools/glm52_resident_stagepack.py against the Python
mirror, so the mirror itself must agree with the C policy the module loads
with. This compiles spark_glm52_stagepack_format.h and compares, for every
tensor kind, every layer (plus the global slot and one past the last layer),
the bf16, fp8 and nvfp4 expert codecs and TP degrees 1 through 16:
acceptance, all six shape fields and the payload and scale byte counts.
It also runs the validator on pack headers to check its expert codec gate.
"""
import importlib.util
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "glm52_validate_pack.py"
_spec = importlib.util.spec_from_file_location("glm52_validate_pack", str(TOOL))
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

CODECS = (V.BF16, V.FP8, V.NVFP4)
TP_DEGREES = (1, 2, 4, 8, 16)
LAYERS = (V.GLOBAL_LAYER, *range(79))

PROGRAM = r"""
#include <stdio.h>
#include "spark_glm52_stagepack_format.h"

int main(void)
{
	static const uint32_t codecs[] = {%(codecs)s},tp_degrees[] = {%(tp_degrees)s},layers[] = {%(layers)s};
	SparkGlm52StagePackTensorShape shape;
	uint32_t codec,tp,layer,kind;
	int32_t rc;
	for (codec=0u; codec<sizeof(codecs)/sizeof(codecs[0]); codec++)
		for (tp=0u; tp<sizeof(tp_degrees)/sizeof(tp_degrees[0]); tp++)
			for (layer=0u; layer<sizeof(layers)/sizeof(layers[0]); layer++)
				for (kind=0u; kind<SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT; kind++)
				{
					rc = SparkGlm52StagePackExpectedShape(kind,layers[layer],codecs[codec],tp_degrees[tp],&shape);
					if ( rc != 0 )
						printf("%%u %%u %%u %%u reject\n",codecs[codec],tp_degrees[tp],layers[layer],kind);
					else
						printf("%%u %%u %%u %%u %%u %%u %%u %%u %%u %%u %%llu %%llu\n",codecs[codec],tp_degrees[tp],layers[layer],kind,
							shape.payload_type,shape.weight_codec,shape.scale_encoding,shape.group_count,shape.rows,shape.columns,
							(unsigned long long)SparkGlm52StagePackExpectedPayloadBytes(&shape),
							(unsigned long long)SparkGlm52StagePackExpectedScaleBytes(&shape));
				}
	return(0);
}
"""


def c_policy():
    source = PROGRAM % {
        "codecs": ",".join("%du" % codec for codec in CODECS),
        "tp_degrees": ",".join("%du" % tp for tp in TP_DEGREES),
        "layers": ",".join("%du" % layer for layer in LAYERS),
    }
    with tempfile.TemporaryDirectory() as directory:
        path, binary = Path(directory) / "mirror.c", Path(directory) / "mirror"
        path.write_text(source)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I.", "-Iinclude",
                        "-Imodel-families/common/include", "-Imodel-families/glm52/include",
                        "-Imodules/glm52_resident_decode_stage/source", str(path), "-o", str(binary)],
                       cwd=ROOT, check=True)
        output = subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout
    policy = {}
    for line in output.splitlines():
        fields = line.split()
        key = tuple(int(value) for value in fields[:4])
        policy[key] = None if fields[4] == "reject" else tuple(int(value) for value in fields[4:])
    return policy


def check_policy():
    policy = c_policy()
    expected_count = len(CODECS) * len(TP_DEGREES) * len(LAYERS) * V.KIND_COUNT
    assert len(policy) == expected_count, (len(policy), expected_count)
    accepted = 0
    for (codec, tp, layer, kind), c_shape in policy.items():
        shape, _error = V.expected_shape(kind, layer, tp, codec)
        if shape is None or c_shape is None:
            assert shape is None and c_shape is None, \
                f"codec {codec} tp{tp} layer {layer} kind {kind}: C {c_shape} python {shape}"
            continue
        mirrored = (*shape, V.expected_payload_bytes(shape), V.expected_scale_bytes(shape))
        assert mirrored == c_shape, \
            f"codec {codec} tp{tp} layer {layer} kind {kind}: C {c_shape} python {mirrored}"
        accepted += 1
    assert accepted > 0
    print(f"PASS the Python mirror matches SparkGlm52StagePackExpectedShape and its byte counts "
          f"on {len(policy)} cases ({accepted} accepted)")


def check_codec_gate():
    with tempfile.TemporaryDirectory() as directory:
        pack = Path(directory) / "header.glm52sp"
        for codec, supported in ((V.BF16, True), (V.FP8, True), (V.NVFP4, True), (2, False), (7, False)):
            header = struct.pack("<20I2Q", 0x32534C47, 3, 264, 64, 1, 0, 0, 1, 0, 0, 78, 78,
                                 6144, 154880, 256, 1, codec, 1, 4, 0, 512, 512)
            pack.write_bytes(header + b"\0" * (512 - len(header)))
            result = subprocess.run([sys.executable, str(TOOL), str(pack), "4"],
                                    capture_output=True, text=True)
            rejected = "UNSUPPORTED expert codec %d " % codec in result.stdout
            assert result.returncode == 1 and rejected != supported, (codec, result.returncode, result.stdout)
    print("PASS the validator accepts bf16, fp8 and nvfp4 expert codecs and rejects the others")


def main():
    check_policy()
    check_codec_gate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
