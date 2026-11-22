#pragma once

#include <string.h>

#include "uvm_linux_ioctl.h"

#define SPARK_NVRM_MODULE_VERSION "580.178.04"
#define SPARK_NVRM_UVM_FREE_LENGTH(params, bytes) ((params).length = (bytes))
#define SPARK_NVRM_UVM_UNREGISTER_GPU(params, gpu_uuid) memcpy((params).gpuUuid.uuid,(gpu_uuid),sizeof((params).gpuUuid.uuid))
