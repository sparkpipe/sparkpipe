#ifndef SPARKPIPE_SPARK_GPU_H
#define SPARKPIPE_SPARK_GPU_H

#ifdef SPARKPIPE_DEVICE_NVRM
#include "runtime/device/nvrm/spark_gpu_nvrm.h"
#else
#include "runtime/device/cuda/spark_gpu_cuda.h"
#endif

#endif
