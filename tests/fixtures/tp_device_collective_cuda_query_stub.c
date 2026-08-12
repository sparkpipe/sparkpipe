#include "cuda_runtime_api.h"

/* The generic host CUDA shim predates cudaEventQuery. Keep this weak test-only
 * definition beside the TP fixture so a real cudart definition wins. */
#if defined(__GNUC__)
__attribute__((weak))
#endif
cudaError_t cudaEventQuery(cudaEvent_t event)
{
    return event != 0 ? cudaSuccess : cudaErrorInvalidValue;
}
