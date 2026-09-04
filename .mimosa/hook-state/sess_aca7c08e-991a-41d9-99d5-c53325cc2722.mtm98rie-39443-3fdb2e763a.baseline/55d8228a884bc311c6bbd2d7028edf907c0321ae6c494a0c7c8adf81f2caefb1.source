#!/usr/bin/env python3
"""Apply the Pro gate-route weight expectation to the validation source."""
from pathlib import Path

p = Path("modules/dsv4_resident_decode_stage/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu")
t = p.read_text()
old1 = '\tuint32_t expert,rank;\n\tif ( SparkDsv4ValidationRequire(isfinite(scores[0]) && scores[0] > 0.0f,"gate_route_score_finite") != 0 ) return(1);'
new1 = '\tuint32_t expert,rank;\n\tfloat expected_weight;\n\tif ( SparkDsv4ValidationRequire(isfinite(scores[0]) && scores[0] > 0.0f,"gate_route_score_finite") != 0 ) return(1);'
old2 = '\t\tif ( SparkDsv4ValidationRequire(fabsf(weights[rank] - 0.25f) <= 1.0e-6f,"gate_route_weight") != 0 ) return(1);'
new2 = ('\t\texpected_weight = SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR /\n'
        '\t\t\t(float)SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN;\n'
        '\t\tif ( SparkDsv4ValidationRequire(fabsf(weights[rank] - expected_weight) <= 1.0e-6f,"gate_route_weight") != 0 ) return(1);')
assert old1 in t, "old1 missing"
assert old2 in t, "old2 missing"
t = t.replace(old1, new1, 1)
t = t.replace(old2, new2, 1)
p.write_text(t)
print("pro change applied; expected_weight count =", t.count("expected_weight"))
