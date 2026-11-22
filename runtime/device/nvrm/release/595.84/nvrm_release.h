#pragma once

#include "uvm_linux_ioctl.h"

#define SPARK_NVRM_MODULE_VERSION "595.84"
#define SPARK_NVRM_UVM_FREE_LENGTH(params, bytes) ((void)(params),(void)(bytes))
#define SPARK_NVRM_UVM_UNREGISTER_GPU(params, gpu_uuid) ((void)(params),(void)(gpu_uuid))
