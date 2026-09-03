import re

p = '/Users/mac/lane-glm53/ring/transport/tp_device_collective.c'
s = open(p).read()

inner_old = "static SparkStatus SparkTpDeviceCollectiveSubmitHidden(\n    SparkTpDeviceCollective *collective,\n    const SparkTpDeviceCollectiveSubmission *submission,\n    uint32_t operation_kind)\n{"
inner_new = "static SparkStatus SparkTpDeviceCollectiveSubmitHiddenInner(\n    SparkTpDeviceCollective *collective,\n    const SparkTpDeviceCollectiveSubmission *submission,\n    uint32_t operation_kind)\n{"
assert s.count(inner_old) == 1, "inner anchor"
s = s.replace(inner_old, inner_new)

wrapper = (
    "static SparkStatus SparkTpDeviceCollectiveSubmitHidden(\n"
    "    SparkTpDeviceCollective *collective,\n"
    "    const SparkTpDeviceCollectiveSubmission *submission,\n"
    "    uint32_t operation_kind)\n"
    "{\n"
    "    SparkStatus status = SparkTpDeviceCollectiveSubmitHiddenInner(\n"
    "        collective,submission,operation_kind);\n"
    "    if (status != SPARK_STATUS_OK)\n"
    '        fprintf(stderr,"LR-SUBRET rank=%u ordinal=%llu status=%u\\n",\n'
    "            collective != 0 ? collective->tp_rank : 0u,\n"
    "            (unsigned long long)(submission != 0 ? submission->ordinal : 0u),\n"
    "            (uint32_t)status);\n"
    "    return status;\n"
    "}\n\n"
)

tail = "SparkStatus SparkTpDeviceCollectiveSubmitBf16("
idx = s.find(tail)
assert idx > 0, "public entry anchor"
s = s[:idx] + wrapper + s[idx:]
open(p, 'w').write(s)
print("submit wrapper installed")
