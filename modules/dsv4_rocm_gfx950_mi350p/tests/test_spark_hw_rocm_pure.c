// test_spark_hw_rocm_pure.c - host-runnable unit tests for the pure logic of
// the rocm.gfx950.mi350p archive: the frozen status-mapping table
// (hwiface_v1.md section 4.0) and SparkHwStatusToString. No HIP calls are
// made - hipError_t values are exercised as integers, so this runs on any
// machine via tools/selftest.sh with the vendored headers.
//
// The alias cases matter most: several hipError_t spellings share numeric
// values (OutOfMemory==MemoryAllocation==2, NotInitialized==InitializationError==3),
// which is why spark_hw_rocm_map_status is an if-chain rather than a switch.

#include "spark_hw_rocm_internal.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK_EQ(expr, expected)                                             \
    do                                                                       \
    {                                                                        \
        long long got_ = (long long)(expr);                                  \
        long long want_ = (long long)(expected);                             \
        if (got_ != want_)                                                   \
        {                                                                    \
            printf("FAIL %s:%d %s: got %lld want %lld\n", __FILE__,        \
                   __LINE__, #expr, got_, want_);                            \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static void test_status_mapping(void)
{
    /* OK + query-only NOT_READY */
    CHECK_EQ(spark_hw_rocm_map_status(hipSuccess), SPARK_HW_OK);
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorNotReady), SPARK_HW_NOT_READY);
    /* EXHAUSTED bucket, including both spellings of value 2 */
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorOutOfMemory), SPARK_HW_EXHAUSTED);
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorMemoryAllocation), SPARK_HW_EXHAUSTED);
    /* LOST bucket: device fault, context teardown, ambiguous fatal */
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorIllegalAddress), SPARK_HW_LOST);
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorContextIsDestroyed), SPARK_HW_LOST);
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorDeinitialized), SPARK_HW_LOST);
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorUnknown), SPARK_HW_LOST);
    /* UNSUPPORTED: capability/descriptor refusals */
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorNotSupported), SPARK_HW_UNSUPPORTED);
    /* INVALID bucket: everything else lands here by default */
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorInvalidValue), SPARK_HW_INVALID);
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorInvalidResourceHandle), SPARK_HW_INVALID);
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorInvalidDevice), SPARK_HW_INVALID);
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorInitializationError), SPARK_HW_INVALID);
    CHECK_EQ(spark_hw_rocm_map_status(hipErrorTbd), SPARK_HW_INVALID);
    /* Out-of-range code from a future ROCm: must stay inside the frozen six */
    CHECK_EQ(spark_hw_rocm_map_status((hipError_t)9999), SPARK_HW_INVALID);
    CHECK_EQ(spark_hw_rocm_map_status((hipError_t)-1), SPARK_HW_INVALID);
}

static void test_status_strings(void)
{
    /* Every enum value stringifies non-NULL; exact wire names for two; the
     * out-of-range slot falls back instead of dereferencing garbage. */
    CHECK_EQ(strcmp(SparkHwStatusToString(SPARK_HW_OK), "SPARK_HW_OK"), 0);
    CHECK_EQ(strcmp(SparkHwStatusToString(SPARK_HW_NOT_READY), "SPARK_HW_NOT_READY"), 0);
    CHECK_EQ(strcmp(SparkHwStatusToString(SPARK_HW_LOST), "SPARK_HW_LOST"), 0);
    for (int status = 0; status <= 5; status++)
    {
        const char *text = SparkHwStatusToString((SparkHwStatus)status);
        if (text == NULL || text[0] == '\0' || strncmp(text, "SPARK_HW_", 9) != 0)
        {
            printf("FAIL status_string: value %d bad text\n", status);
            failures++;
        }
    }
    if (strstr(SparkHwStatusToString((SparkHwStatus)42), "UNKNOWN") == NULL)
    {
        printf("FAIL status_string: out-of-range lacks UNKNOWN fallback\n");
        failures++;
    }
}

int main(void)
{
    test_status_mapping();
    test_status_strings();
    if (failures != 0)
    {
        printf("test_spark_hw_rocm_pure FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_spark_hw_rocm_pure PASS\n");
    return 0;
}

