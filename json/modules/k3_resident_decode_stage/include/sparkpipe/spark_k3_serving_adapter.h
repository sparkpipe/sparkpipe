#ifndef SPARKPIPE_SPARK_K3_SERVING_ADAPTER_H
#define SPARKPIPE_SPARK_K3_SERVING_ADAPTER_H

#include "sparkpipe/spark_model_serving_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * K3 serving adapter: the residentd-facing SparkModelServingAdapterInterface
 * over the K3 stage runner. Configuration JSON members (under the runtime
 * root's adapter config path):
 *   stage_pack_path      - the rank pack for this rank/stage
 *   tp_degree, tp_rank   - the TP4 group placement
 *   max_sequences, max_rows, resident_capacity, kv_page_bytes
 *   context_tokens       - the deployment's context axis; the MLA page
 *                          count derives from it over K3_KV_PAGE_SLOTS
 *                          (64-token pages). An explicit "kv_pages"
 *                          overrides; with neither key a 2048-token
 *                          context is assumed - never a fixed two pages.
 *   tp_collective: { listen_port, connect_timeout_milli,
 *     operation_timeout_milli, collective_identifier,
 *     peers: ["host:port", ...] } - peers are the step topology
 */
const SparkModelServingAdapterInterface *SparkK3ServingAdapterGetInterface(void);

#ifdef __cplusplus
}
#endif

#endif
