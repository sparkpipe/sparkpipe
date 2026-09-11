#pragma once
#include <stdint.h>

/* Require an externally supervised daemon and one valid pack digest sidecar.
 * Does not spawn, retry, or switch loading modes. Attach validates IPC later. */
int32_t SparkModelResidentdPrepareWeightd(const char *runtime_root,const char *socket_path);
