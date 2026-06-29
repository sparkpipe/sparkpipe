#include "spark_glm52_sota_decode_common.cuh"

extern "C" cudaError_t SparkGlm52SotaDecodeGraphLaunchOrCaptureSm121(
    SparkGlm52SotaDecodeGraphPlan *plan,
    cudaStream_t stream,
    cudaError_t (*capture_body)(void *context, cudaStream_t stream),
    void *context)
{
    cudaError_t error;

    if (plan == 0 || capture_body == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->graph_ready != 0u && plan->graph_exec != 0)
    {
        error = cudaGraphLaunch(plan->graph_exec, stream);
        if (error == cudaSuccess)
        {
            plan->replay_count += 1u;
        }
        return error;
    }

    if (plan->capture_requested == 0u)
    {
        return capture_body(context, stream);
    }

    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (error != cudaSuccess)
    {
        return error;
    }
    error = capture_body(context, stream);
    if (error != cudaSuccess)
    {
        cudaStreamEndCapture(stream, &plan->graph);
        return error;
    }
    error = cudaStreamEndCapture(stream, &plan->graph);
    if (error != cudaSuccess)
    {
        return error;
    }
    error = cudaGraphInstantiate(&plan->graph_exec, plan->graph, 0, 0, 0);
    if (error != cudaSuccess)
    {
        return error;
    }
    plan->graph_ready = 1u;
    plan->capture_count += 1u;
    error = cudaGraphLaunch(plan->graph_exec, stream);
    if (error == cudaSuccess)
    {
        plan->replay_count += 1u;
    }
    return error;
}
