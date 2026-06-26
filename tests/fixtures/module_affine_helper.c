#include <stdint.h>

uint64_t SparkTestAffineApply(uint64_t value, uint64_t multiplier, uint64_t increment);

uint64_t SparkTestAffineApply(uint64_t value, uint64_t multiplier, uint64_t increment)
{
    return (value * multiplier) + increment;
}
