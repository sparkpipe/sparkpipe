#include <metal_stdlib>

using namespace metal;

kernel void spark_test_exp(constant float &step [[buffer(0)]], device float *out [[buffer(1)]], uint gid [[thread_position_in_grid]])
{
	out[gid] = exp(-float(gid) * step);
}
