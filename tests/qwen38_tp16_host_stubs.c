/* Host stubs for loader-delta testing of spark_qwen38_resident_decode_stage_module.
 * Launchers record invocations so the harness can prove the rank-local
 * fail-closed guard fires BEFORE any grouped-expert launch. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_qwen38_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_tp_device_collective.h"

#define STUB_COUNT 24
static const char *stub_names[STUB_COUNT] = {
    "ConfigureCudaKernels", "EmbeddingGather", "RmsNorm", "FusedResidualRmsNorm",
    "Linear", "ConvUpdate", "DecayBeta", "GdnStep", "GatedNorm", "AttnPrepare",
    "AttnDecode", "ResidualAdd", "GateScores", "GateSelect", "MoeRoute",
    "SwiGlu", "MoePairReduceOverwrite", "GroupedExpertLinear",
    "GroupedExpertTileLinear", "SharedGate", "HeadShadowQuantize",
    "HeadScreenedArgmax", "HeadArgmax", "TpCombineAdd"
};
unsigned long long q38_stub_counts[STUB_COUNT];
static int stub_index(const char *name)
{
    int i;
    for (i = 0; i < STUB_COUNT; i++)
        if (strcmp(stub_names[i], name) == 0)
            return i;
    return -1;
}
#define STUB_RET(name) do { int _i = stub_index(#name); if (_i >= 0) q38_stub_counts[_i]++; return cudaSuccess; } while (0)

void q38_stub_dump(void)
{
    int i;
    for (i = 0; i < STUB_COUNT; i++)
        if (q38_stub_counts[i] != 0ull)
            fprintf(stderr, "stub_calls %s %llu\n", stub_names[i], q38_stub_counts[i]);
}

cudaError_t SparkQwen38ConfigureCudaKernels(void) { STUB_RET(ConfigureCudaKernels); }
cudaError_t SparkQwen38LaunchEmbeddingGather(cudaStream_t s, const uint32_t *t, const void *e, void *h, uint32_t rows) { (void)s;(void)t;(void)e;(void)h;(void)rows; STUB_RET(EmbeddingGather); }
cudaError_t SparkQwen38LaunchRmsNorm(cudaStream_t s, const void *in, const void *gain, void *out, uint32_t rows, uint32_t dim, float eps) { (void)s;(void)in;(void)gain;(void)out;(void)rows;(void)dim;(void)eps; STUB_RET(RmsNorm); }
cudaError_t SparkQwen38LaunchFusedResidualRmsNorm(cudaStream_t s, void *h, const void *d, const void *g, void *o, uint32_t rows, uint32_t dim, float eps) { (void)s;(void)h;(void)d;(void)g;(void)o;(void)rows;(void)dim;(void)eps; STUB_RET(FusedResidualRmsNorm); }
cudaError_t SparkQwen38LaunchLinear(cudaStream_t s, const SparkQwen38LinearView *v, const void *in, void *out, uint32_t rows) { (void)s;(void)v;(void)in;(void)out;(void)rows; STUB_RET(Linear); }
cudaError_t SparkQwen38LaunchConvUpdate(cudaStream_t s, const void *qkv, const SparkQwen38GdnLayerWeights *w, void *co, const SparkQwen38GdnStatePool *p, const uint32_t *lanes, uint32_t rows, uint32_t ord) { (void)s;(void)qkv;(void)w;(void)co;(void)p;(void)lanes;(void)rows;(void)ord; STUB_RET(ConvUpdate); }
cudaError_t SparkQwen38LaunchDecayBeta(cudaStream_t s, const void *dp, const void *bp, const SparkQwen38GdnLayerWeights *w, float *ld, float *beta, uint32_t rows) { (void)s;(void)dp;(void)bp;(void)w;(void)ld;(void)beta;(void)rows; STUB_RET(DecayBeta); }
cudaError_t SparkQwen38LaunchGdnStep(cudaStream_t s, const void *conv, const float *ld, const float *beta, const SparkQwen38GdnStatePool *p, void *core, const uint32_t *lanes, uint32_t rows, uint32_t ord) { (void)s;(void)conv;(void)ld;(void)beta;(void)p;(void)core;(void)lanes;(void)rows;(void)ord; STUB_RET(GdnStep); }
cudaError_t SparkQwen38LaunchGatedNorm(cudaStream_t s, const void *core, const void *z, const SparkQwen38GdnLayerWeights *w, void *out, uint32_t rows, float eps) { (void)s;(void)core;(void)z;(void)w;(void)out;(void)rows;(void)eps; STUB_RET(GatedNorm); }
cudaError_t SparkQwen38LaunchAttnPrepare(cudaStream_t s, void *q, const void *k, const void *v, const SparkQwen38AttnLayerWeights *w, void *kvc, const uint32_t *sm, const uint64_t *pos, uint32_t rows, uint32_t ord, unsigned long long cls, unsigned long long cbs, float eps, uint32_t tpd, uint32_t tpr) { (void)s;(void)q;(void)k;(void)v;(void)w;(void)kvc;(void)sm;(void)pos;(void)rows;(void)ord;(void)cls;(void)cbs;(void)eps;(void)tpd;(void)tpr; STUB_RET(AttnPrepare); }
cudaError_t SparkQwen38LaunchAttnDecode(cudaStream_t s, const void *q, const void *kvc, const SparkQwen38KvBlockTableView *t, const uint32_t *lanes, const uint32_t *cl, void *ho, uint32_t rows, uint32_t ord, unsigned long long cls, unsigned long long cbs, uint32_t tpd, uint32_t tpr) { (void)s;(void)q;(void)kvc;(void)t;(void)lanes;(void)cl;(void)ho;(void)rows;(void)ord;(void)cls;(void)cbs;(void)tpd;(void)tpr; STUB_RET(AttnDecode); }
cudaError_t SparkQwen38LaunchResidualAdd(cudaStream_t s, void *h, const void *d, uint32_t rows, uint32_t dim) { (void)s;(void)h;(void)d;(void)rows;(void)dim; STUB_RET(ResidualAdd); }
cudaError_t SparkQwen38LaunchGateScores(cudaStream_t s, const SparkQwen38LinearView *g, const void *in, float *sc, uint32_t rows) { (void)s;(void)g;(void)in;(void)sc;(void)rows; STUB_RET(GateScores); }
cudaError_t SparkQwen38LaunchGateSelect(cudaStream_t s, const float *sc, const float *b, uint32_t rows, uint32_t ec, uint32_t topk, float rs, uint32_t *ix, float *w) { (void)s;(void)sc;(void)b;(void)rows;(void)ec;(void)topk;(void)rs;(void)ix;(void)w; STUB_RET(GateSelect); }
cudaError_t SparkQwen38LaunchMoeRoute(cudaStream_t s, const uint32_t *re, uint32_t rows, uint32_t ew, uint32_t *go, uint32_t *pr, uint32_t *st, uint32_t *p1, uint32_t *p2) { (void)s;(void)re;(void)rows;(void)ew;(void)go;(void)pr;(void)st;(void)p1;(void)p2; STUB_RET(MoeRoute); }
cudaError_t SparkQwen38LaunchSwiGlu(cudaStream_t s, const void *g, void *u, uint32_t rows, uint32_t dim) { (void)s;(void)g;(void)u;(void)rows;(void)dim; STUB_RET(SwiGlu); }
cudaError_t SparkQwen38LaunchMoePairReduceOverwrite(cudaStream_t s, const void *so, const uint32_t *im, const float *pw, void *o, uint32_t rows, uint32_t dim) { (void)s;(void)so;(void)im;(void)pw;(void)o;(void)rows;(void)dim; STUB_RET(MoePairReduceOverwrite); }
unsigned long long q38_grouped_linear_calls;
unsigned long long q38_grouped_tile_calls;
uint32_t q38_last_linear_rpe, q38_last_linear_tpd, q38_last_linear_tpr;
cudaError_t SparkQwen38LaunchGroupedExpertLinear(cudaStream_t s, const SparkQwen38LinearView *v, const void *in, const uint32_t *src, const uint32_t *off, const uint32_t *pfx, void *o, uint32_t src_rows, uint32_t mp, uint32_t tpd, uint32_t tpr, uint32_t rpe) { (void)s;(void)v;(void)in;(void)src;(void)off;(void)pfx;(void)o;(void)src_rows;(void)mp; int _i = stub_index("GroupedExpertLinear"); if (_i >= 0) q38_stub_counts[_i]++; q38_grouped_linear_calls++; q38_last_linear_rpe = rpe; q38_last_linear_tpd = tpd; q38_last_linear_tpr = tpr; return cudaSuccess; }
cudaError_t SparkQwen38LaunchGroupedExpertTileLinear(cudaStream_t s, const SparkQwen38LinearView *v, const void *in, const uint32_t *src, const uint32_t *off, void *o, uint32_t src_rows, uint32_t tpd, uint32_t tpr, uint32_t rpe) { (void)s;(void)v;(void)in;(void)src;(void)off;(void)o;(void)src_rows;(void)tpd;(void)tpr;(void)rpe; int _i = stub_index("GroupedExpertTileLinear"); if (_i >= 0) q38_stub_counts[_i]++; q38_grouped_tile_calls++; return cudaSuccess; }
cudaError_t SparkQwen38LaunchSharedGate(cudaStream_t s, void *a, const void *gw, const void *gi, uint32_t rows, uint32_t dim) { (void)s;(void)a;(void)gw;(void)gi;(void)rows;(void)dim; STUB_RET(SharedGate); }
cudaError_t SparkQwen38LaunchHeadShadowQuantize(cudaStream_t s, const void *lh, void *sp, void *ss, float *en, uint32_t vocab, uint32_t hidden) { (void)s;(void)lh;(void)sp;(void)ss;(void)en;(void)vocab;(void)hidden; STUB_RET(HeadShadowQuantize); }
cudaError_t SparkQwen38LaunchHeadScreenedArgmax(cudaStream_t s, const void *n, const void *lh, const void *sp, const void *ss, const float *en, uint32_t *ci, uint32_t *cc, uint32_t *oi, uint32_t rows, uint32_t vocab) { (void)s;(void)n;(void)lh;(void)sp;(void)ss;(void)en;(void)ci;(void)cc;(void)oi;(void)rows;(void)vocab; STUB_RET(HeadScreenedArgmax); }
cudaError_t SparkQwen38LaunchHeadArgmax(cudaStream_t s, const void *n, const void *lh, const uint32_t *ii, uint32_t *oi, uint32_t rows, uint32_t vocab) { (void)s;(void)n;(void)lh;(void)ii;(void)oi;(void)rows;(void)vocab; STUB_RET(HeadArgmax); }
cudaError_t SparkQwen38LaunchTpCombineAdd(cudaStream_t s, void *dst, const void *src, uint32_t rows, uint32_t width) { (void)s;(void)dst;(void)src;(void)rows;(void)width; STUB_RET(TpCombineAdd); }

/* TP device collective: deterministic local success. Submit completes the
 * caller's flag immediately so TpAllReduceHidden returns OK. */
SparkStatus SparkTpDeviceCollectiveApplyTopology(const SparkTpDeviceCollectiveTopology *topology, SparkTpDeviceCollectiveConfig *configuration)
{
    (void)topology;
    (void)configuration;
    return SPARK_STATUS_OK;
}
SparkStatus SparkTpDeviceCollectiveCreditBindingRouteCount(const SparkTpDeviceCollectiveConfig *configuration, uint32_t *route_count)
{
    (void)configuration;
    *route_count = 1u;
    return SPARK_STATUS_OK;
}
SparkStatus SparkTpDeviceCollectiveProbeMemoryMode(uint32_t backend_kind, const char *backend_module_path, uint32_t *memory_mode_out)
{
    (void)backend_kind;
    (void)backend_module_path;
    *memory_mode_out = SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE;
    return SPARK_STATUS_OK;
}
SparkStatus SparkTpDeviceCollectiveCreate(const SparkTpDeviceCollectiveConfig *configuration, SparkTpDeviceCollective *collective)
{
    (void)configuration;
    memset(collective, 0, sizeof(*collective));
    return SPARK_STATUS_OK;
}
void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
    (void)collective;
}
SparkStatus SparkTpDeviceCollectiveSubmitBf16(SparkTpDeviceCollective *collective, const SparkTpDeviceCollectiveSubmission *submission)
{
    SparkTpDeviceCollectiveCompletion completion;
    (void)collective;
    memset(&completion, 0, sizeof(completion));
    completion.status = SPARK_STATUS_OK;
    submission->completion_function(submission->completion_context, &completion);
    return SPARK_STATUS_OK;
}

/* KV client: provider none never opens; Destroy calls Close unconditionally. */
#include "sparkpipe/spark_stage_kv_client.h"
SparkStatus SparkStageKvClientOpen(SparkStageKvClient *client, const char *module_tag, const char *provider, uint32_t rank_index, uint32_t first_layer_index, uint32_t layer_count, unsigned long long model_fingerprint, unsigned long long layout_fingerprint, const char *service_address, const char *ipc_socket_path, unsigned long long pool_bytes, uint32_t worker_count)
{
    (void)client;(void)module_tag;(void)provider;(void)rank_index;(void)first_layer_index;(void)layer_count;(void)model_fingerprint;(void)layout_fingerprint;(void)service_address;(void)ipc_socket_path;(void)pool_bytes;(void)worker_count;
    return SPARK_STATUS_OK;
}
void SparkStageKvClientClose(SparkStageKvClient *client) { (void)client; }

/* Missing from the repo CUDA stub: SM count probe used by Initialize. */
cudaError_t cudaDeviceGetAttribute(int *value, int attribute, int device)
{
    (void)attribute;
    (void)device;
    *value = 108;
    return cudaSuccess;
}

/* KV store/client internals referenced by work control. With
 * SPARK_QWEN38_STAGE_KV_STORE=none nothing calls these at runtime; they only
 * need to link (and abort loudly if ever reached). */
uint32_t SparkKvStoreSelectPressureLimitedLookaheadPacketCount(
    uint32_t lookahead_packet_count, uint32_t queue_depth,
    uint32_t physical_block_capacity, uint32_t allocated_physical_block_count,
    uint32_t staging_block_capacity, const uint32_t *cumulative_nonresident_block_counts)
{
    (void)lookahead_packet_count;(void)queue_depth;(void)physical_block_capacity;
    (void)allocated_physical_block_count;(void)staging_block_capacity;
    (void)cumulative_nonresident_block_counts;
    fprintf(stderr, "STUB BUG: pressure lookahead reached\n");
    abort();
}
int32_t SparkStageKvClientFormatKey(char *key, uint32_t key_capacity, unsigned long long model_fingerprint, unsigned long long cache_layout_fingerprint, uint32_t rank_index, unsigned long long sequence_id, uint32_t logical_block)
{
    (void)key;(void)key_capacity;(void)model_fingerprint;(void)cache_layout_fingerprint;
    (void)rank_index;(void)sequence_id;(void)logical_block;
    fprintf(stderr, "STUB BUG: FormatKey reached\n");
    abort();
}
SparkStatus SparkStageKvClientSubmit(SparkStageKvClient *client, uint32_t operation, const SparkKvStoreBlock *blocks, uint32_t block_count, uint32_t priority, uint64_t *batch_id)
{
    (void)client;(void)operation;(void)blocks;(void)block_count;(void)priority;(void)batch_id;
    fprintf(stderr, "STUB BUG: KvClientSubmit reached\n");
    abort();
}
SparkStatus SparkStageKvClientPoll(SparkStageKvClient *client, uint64_t batch_id, SparkKvStoreCompletion *completion)
{
    (void)client;(void)batch_id;(void)completion;
    fprintf(stderr, "STUB BUG: KvClientPoll reached\n");
    abort();
}
