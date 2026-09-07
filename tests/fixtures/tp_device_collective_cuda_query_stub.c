#include "cuda_runtime_api.h"

#if defined(__GNUC__)
__attribute__((weak))
#endif
cudaError_t cudaEventQuery(cudaEvent_t event)
{
    return event != 0 ? cudaSuccess : cudaErrorInvalidValue;
}
