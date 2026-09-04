from pathlib import Path

base = Path("/Users/mac/lane-glm53/staging")
p = base / "mb_doorbell.cu"
resolved = p.resolve()
if str(resolved)[: len(str(base))] != str(base):
    raise SystemExit("unexpected path")
s = resolved.open().read()

old = (
    "        {\n"
    "            uint64_t retry_started = bench_now_ns();\n"
    "            for (;;)\n"
    "            {\n"
    "                status = SparkTpDeviceCollectiveSubmitBf16(&collective, &submission);\n"
    "                if (status == SPARK_STATUS_OK)\n"
    "                    break;\n"
    "                if (status != SPARK_STATUS_BUSY ||\n"
    "                    bench_now_ns() - retry_started > 600000000000ull)\n"
    "                {\n"
    '                    printf("warmup submit %llu -> %u t=%.1fs\\n", (unsigned long long)ordinal, (unsigned)status,(bench_now_ns()-retry_started)/1e9);\n'
    "                    return 1;\n"
    "                }\n"
    "                usleep(50u);\n"
    "            }\n"
    "        }\n"
    "        if (cudaStreamSynchronize(stream) != cudaSuccess)"
)
assert s.count(old) == 1, "anchor %d" % s.count(old)
new = (
    "        {\n"
    "            uint64_t retry_started = bench_now_ns();\n"
    "            uint64_t target = ordinal | 1u;\n"
    "            uint64_t sub;\n"
    "            for (sub = ordinal; sub <= target; sub++)\n"
    "            {\n"
    "                submission.ordinal = sub;\n"
    "                submission.slot_index = (uint32_t)(sub % BENCH_CREDITS);\n"
    "                submission.local_device = payload[sub % 64u];\n"
    "                submission.full_device = payload[sub % 64u];\n"
    "                for (;;)\n"
    "                {\n"
    "                    status = SparkTpDeviceCollectiveSubmitBf16(&collective, &submission);\n"
    "                    if (status == SPARK_STATUS_OK)\n"
    "                        break;\n"
    "                    if (status != SPARK_STATUS_BUSY ||\n"
    "                        bench_now_ns() - retry_started > 600000000000ull)\n"
    "                    {\n"
    '                        printf("warmup submit %llu -> %u t=%.1fs\\n", (unsigned long long)sub, (unsigned)status,(bench_now_ns()-retry_started)/1e9);\n'
    "                        return 1;\n"
    "                    }\n"
    "                    usleep(50u);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        if (cudaStreamSynchronize(stream) != cudaSuccess)"
)
s = s.replace(old, new)
resolved.open("w").write(s)
print("warmup pairs installed")
