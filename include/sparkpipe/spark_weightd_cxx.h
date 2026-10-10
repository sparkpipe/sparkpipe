#pragma once

#include <stdint.h>

#ifdef __cplusplus
#define _Static_assert(expr, msg) static_assert(expr, msg)
#endif
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_map.h"
#include "sparkpipe/spark_weightd_lease.h"
#include "sparkpipe/spark_weightd_attach.h"
#ifdef __cplusplus
#undef _Static_assert
#endif
