// spark_hw_rocm_graph.c - Graph primitive family for rocm.gfx950.mi350p.
//
// Frozen contract: hwiface_v1.md section 4.2. HIP graphs map 1:1 onto the
// graph family (section 6 hardware facts). Capture topology is out of the ABI
// (section 3.3): the core only opens a capture window, closes it, instantiates,
// and replays - what the target enqueues between begin and end is target-internal.
//
//   capture_begin  -> hipStreamBeginCapture with hipStreamCaptureModeRelaxed;
//                     v1 accepts only SPARK_HW_CAPTURE_RELAXED.
//   capture_end    -> hipStreamEndCapture; an invalidated capture surfaces as
//                     the mapped error with *graph_out == NULL.
//   instantiate    -> hipGraphInstantiate (5-argument ROCm form).
//   launch         -> hipGraphLaunch on the given queue.
//   destroy / exec_destroy -> hipGraphDestroy / hipGraphExecDestroy. Both may
//                     internally synchronize during teardown drain (section 4.2
//                     synchronization note).

#include "spark_hw_rocm_internal.h"

SparkHwStatus spark_hw_graph_capture_begin(SparkHwQueue q, uint32_t mode)
{
    if (q == NULL)
    {
        return SPARK_HW_INVALID;
    }
    if (mode != SPARK_HW_CAPTURE_RELAXED)
    {
        return SPARK_HW_UNSUPPORTED; /* v1: relaxed capture only */
    }
    return spark_hw_rocm_map_status(
        hipStreamBeginCapture((hipStream_t)q, hipStreamCaptureModeRelaxed));
}

SparkHwStatus spark_hw_graph_capture_end(SparkHwQueue q, void *graph_out)
{
    if (q == NULL || graph_out == NULL)
    {
        return SPARK_HW_INVALID;
    }
    hipGraph_t graph = NULL;
    hipError_t err = hipStreamEndCapture((hipStream_t)q, &graph);
    *(void **)graph_out = (void *)graph;
    return spark_hw_rocm_map_status(err);
}

SparkHwStatus spark_hw_graph_instantiate(void *graph, SparkHwGraphExec *exec)
{
    if (graph == NULL || exec == NULL)
    {
        return SPARK_HW_INVALID;
    }
    hipGraphExec_t exec_handle = NULL;
    /* ROCm keeps the CUDA-style 5-argument instantiate form. */
    hipError_t err = hipGraphInstantiate(&exec_handle, (hipGraph_t)graph,
                                         NULL, NULL, 0);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err); /* resource refusal -> EXHAUSTED */
    }
    *exec = (SparkHwGraphExec)exec_handle;
    return SPARK_HW_OK;
}

SparkHwStatus spark_hw_graph_launch(SparkHwGraphExec exec, SparkHwQueue q)
{
    if (exec == NULL || q == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return spark_hw_rocm_map_status(
        hipGraphLaunch((hipGraphExec_t)exec, (hipStream_t)q));
}

SparkHwStatus spark_hw_graph_destroy(void *graph)
{
    if (graph == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return spark_hw_rocm_map_status(hipGraphDestroy((hipGraph_t)graph));
}

SparkHwStatus spark_hw_graph_exec_destroy(SparkHwGraphExec exec)
{
    if (exec == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return spark_hw_rocm_map_status(hipGraphExecDestroy((hipGraphExec_t)exec));
}
