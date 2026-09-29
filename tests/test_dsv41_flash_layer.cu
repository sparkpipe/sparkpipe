#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_kernels.cuh"

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "FAIL line=%d cuda=%s call=%s\n", __LINE__, cudaGetErrorString(e), #call); \
    exit(1); } } while (0)

static const uint32_t kHidden = 5120u, kHeads = 64u, kHeadDim = 512u, kRope = 64u, kQLora = 1280u;
static const uint32_t kGroups = 8u, kOLora = 1024u, kWindow = 128u, kHc = 4u, kSinkhorn = 20u;
static const uint32_t kExperts = 384u, kRouted = 6u, kInter = 2304u;
static const float kEpsilon = 1e-20f, kHcEpsilon = 1e-6f, kRouteScale = 1.5f, kSwigluLimit = 10.0f;

static std::string directory;

static std::vector<uint8_t> ReadFile(const std::string &name)
{
    std::string path = directory + "/" + name;
    FILE *file = fopen(path.c_str(), "rb");
    if (file == 0)
    {
        fprintf(stderr, "FAIL cannot open %s\n", path.c_str());
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    std::vector<uint8_t> bytes((size_t)size);
    if (fread(bytes.data(), 1u, bytes.size(), file) != bytes.size())
    {
        fprintf(stderr, "FAIL short read %s\n", path.c_str());
        exit(1);
    }
    fclose(file);
    return bytes;
}

template <class T>
static T *Device(const std::vector<uint8_t> &bytes)
{
    void *pointer = 0;
    CUDA(cudaMalloc(&pointer, bytes.size()));
    CUDA(cudaMemcpy(pointer, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    return (T *)pointer;
}

template <class T>
static T *Scratch(size_t count)
{
    void *pointer = 0;
    CUDA(cudaMalloc(&pointer, count * sizeof(T)));
    CUDA(cudaMemset(pointer, 0, count * sizeof(T)));
    return (T *)pointer;
}

static float Bf16(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16u;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static std::string Name(uint32_t layer, const std::string &suffix)
{
    return "layers." + std::to_string(layer) + "." + suffix + ".bin";
}

typedef struct Attention
{
    uint8_t *wq_a, *wq_a_scale, *wq_b, *wq_b_scale, *wkv, *wkv_scale, *wo_a, *wo_a_scale, *wo_b, *wo_b_scale;
    uint16_t *q_norm, *kv_norm;
    float *sink, *rope, *activation;
    uint16_t *q_latent, *q_normed, *query, *kv, *ring, *attention, *grouped;
    int32_t *indices;
    std::vector<int32_t> slots;
    uint32_t ratio, compressed;
    uint16_t *compress_wkv, *compress_wgate, *compress_norm, *pooled;
    float *kv_state, *score_state;
} Attention;

static const uint32_t kMaxCompressed = 512u;

typedef struct Moe
{
    uint32_t layer;
    uint16_t *gate;
    float *bias, *activation, *accumulator, *weights;
    uint8_t *shared[3], *shared_scale[3];
    uint16_t *gate_out, *up_out, *hidden, *down;
    uint32_t *indices;
    std::vector<uint8_t *> payload, scale;
} Moe;

static void AttentionLoad(Attention *a, uint32_t layer)
{
    a->wq_a = Device<uint8_t>(ReadFile(Name(layer, "attn.wq_a.weight")));
    a->wq_a_scale = Device<uint8_t>(ReadFile(Name(layer, "attn.wq_a.scale")));
    a->q_norm = Device<uint16_t>(ReadFile(Name(layer, "attn.q_norm.weight")));
    a->wq_b = Device<uint8_t>(ReadFile(Name(layer, "attn.wq_b.weight")));
    a->wq_b_scale = Device<uint8_t>(ReadFile(Name(layer, "attn.wq_b.scale")));
    a->wkv = Device<uint8_t>(ReadFile(Name(layer, "attn.wkv.weight")));
    a->wkv_scale = Device<uint8_t>(ReadFile(Name(layer, "attn.wkv.scale")));
    a->kv_norm = Device<uint16_t>(ReadFile(Name(layer, "attn.kv_norm.weight")));
    a->sink = Device<float>(ReadFile(Name(layer, "attn.attn_sink")));
    a->wo_a = Device<uint8_t>(ReadFile(Name(layer, "attn.wo_a.weight")));
    a->wo_a_scale = Device<uint8_t>(ReadFile(Name(layer, "attn.wo_a.scale")));
    a->wo_b = Device<uint8_t>(ReadFile(Name(layer, "attn.wo_b.weight")));
    a->wo_b_scale = Device<uint8_t>(ReadFile(Name(layer, "attn.wo_b.scale")));
    a->rope = Device<float>(ReadFile("rope.bin"));
    a->activation = Scratch<float>(kHeads * kHeadDim);
    a->q_latent = Scratch<uint16_t>(kQLora);
    a->q_normed = Scratch<uint16_t>(kQLora);
    a->query = Scratch<uint16_t>(kHeads * kHeadDim);
    a->kv = Scratch<uint16_t>(kHeadDim);
    a->ring = Scratch<uint16_t>((kWindow + kMaxCompressed) * kHeadDim);
    std::vector<uint8_t> ratio = ReadFile("ratio.bin");
    a->ratio = *(const uint32_t *)ratio.data();
    a->compressed = 0u;
    if (a->ratio > 1u)
    {
        a->compress_wkv = Device<uint16_t>(ReadFile(Name(layer, "attn.compressor.wkv.weight")));
        a->compress_wgate = Device<uint16_t>(ReadFile(Name(layer, "attn.compressor.wgate.weight")));
        a->compress_norm = Device<uint16_t>(ReadFile(Name(layer, "attn.compressor.norm.weight")));
        a->kv_state = Scratch<float>(a->ratio * kHeadDim);
        a->score_state = Scratch<float>(a->ratio * kHeadDim);
        a->pooled = Scratch<uint16_t>(kHeadDim);
    }
    a->attention = Scratch<uint16_t>(kHeads * kHeadDim);
    a->grouped = Scratch<uint16_t>(kGroups * kOLora);
    a->indices = Scratch<int32_t>(kWindow + kMaxCompressed);
    a->slots.resize(kWindow + kMaxCompressed);
}

static void AttentionStep(Attention *a, uint32_t position, const uint16_t *x, uint16_t *out)
{
    const float *cos_sin = a->rope + (size_t)position * kRope;
    CUDA(SparkDsv41FlashLaunchFp8BlockLinear(0, a->wq_a, a->wq_a_scale, x, a->activation, a->q_latent, 1u, kHidden, kQLora));
    CUDA(SparkDsv41FlashLaunchRmsNorm(0, a->q_latent, a->q_norm, a->q_normed, 1u, kQLora, kEpsilon));
    CUDA(SparkDsv41FlashLaunchFp8BlockLinear(0, a->wq_b, a->wq_b_scale, a->q_normed, a->activation, a->query, 1u, kQLora, kHeads * kHeadDim));
    CUDA(SparkDsv41FlashLaunchRope(0, a->query, cos_sin, kHeads, kHeadDim, kRope, 0u));
    CUDA(SparkDsv41FlashLaunchFp8BlockLinear(0, a->wkv, a->wkv_scale, x, a->activation, a->kv, 1u, kHidden, kHeadDim));
    CUDA(SparkDsv41FlashLaunchRmsNorm(0, a->kv, a->kv_norm, a->kv, 1u, kHeadDim, kEpsilon));
    CUDA(SparkDsv41FlashLaunchRope(0, a->kv, cos_sin, 1u, kHeadDim, kRope, 0u));
    CUDA(SparkDsv41FlashLaunchFp8Qdq(0, a->kv, kHeadDim));
    CUDA(cudaMemcpy(a->ring + (size_t)(position % kWindow) * kHeadDim, a->kv, kHeadDim * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
    uint32_t oldest = position % kWindow + 1u;
    for (uint32_t column = 0u; column < kWindow; column++)
    {
        uint32_t slot = oldest + column < kWindow ? oldest + column : oldest + column - kWindow;
        a->slots[column] = slot > position ? -1 : (int32_t)slot;
    }
    if (a->ratio > 1u)
    {
        uint32_t slot = position % a->ratio;
        CUDA(SparkDsv41FlashLaunchBf16Linear(0, a->compress_wkv, x, a->kv_state + (size_t)slot * kHeadDim, 0, kHidden, kHeadDim));
        CUDA(SparkDsv41FlashLaunchBf16Linear(0, a->compress_wgate, x, a->score_state + (size_t)slot * kHeadDim, 0, kHidden, kHeadDim));
        if ((position + 1u) % a->ratio == 0u)
        {
            uint16_t *row = a->ring + (size_t)(kWindow + a->compressed) * kHeadDim;
            SparkDsv41FlashCompressPoolKernel<<<(kHeadDim + 255u) / 256u, 256u>>>(a->kv_state, a->score_state, a->pooled, a->ratio, kHeadDim);
            CUDA(cudaGetLastError());
            CUDA(SparkDsv41FlashLaunchRmsNorm(0, a->pooled, a->compress_norm, row, 1u, kHeadDim, kEpsilon));
            CUDA(SparkDsv41FlashLaunchRope(0, row, a->rope + (size_t)(position + 1u - a->ratio) * kRope, 1u, kHeadDim, kRope, 0u));
            CUDA(SparkDsv41FlashLaunchKvFp4Qdq(0, row, kHeadDim, 1u, kHeadDim));
            a->compressed += 1u;
        }
    }
    uint32_t count = kWindow + a->compressed;
    for (uint32_t column = 0u; column < a->compressed; column++)
        a->slots[kWindow + column] = (int32_t)(kWindow + column);
    CUDA(cudaMemcpy(a->indices, a->slots.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA(SparkDsv41FlashLaunchSinkAttention(0, a->query, a->ring, a->indices, count, a->sink, a->attention, kHeads, kHeadDim, 1.0f / sqrtf((float)kHeadDim)));
    CUDA(SparkDsv41FlashLaunchRope(0, a->attention, cos_sin, kHeads, kHeadDim, kRope, 1u));
    CUDA(SparkDsv41FlashLaunchGroupedFp8Linear(0, a->wo_a, a->wo_a_scale, a->attention, a->grouped, kHeads * kHeadDim / kGroups, kGroups * kOLora, kOLora));
    CUDA(SparkDsv41FlashLaunchFp8BlockLinear(0, a->wo_b, a->wo_b_scale, a->grouped, a->activation, out, 1u, kGroups * kOLora, kHidden));
}

static const char *kParts[3] = {"w1", "w2", "w3"};

static void MoeLoad(Moe *m, uint32_t layer)
{
    m->layer = layer;
    m->gate = Device<uint16_t>(ReadFile(Name(layer, "ffn.gate.weight")));
    m->bias = Device<float>(ReadFile(Name(layer, "ffn.gate.bias")));
    for (uint32_t part = 0u; part < 3u; part++)
    {
        m->shared[part] = Device<uint8_t>(ReadFile(Name(layer, std::string("ffn.shared_experts.") + kParts[part] + ".weight")));
        m->shared_scale[part] = Device<uint8_t>(ReadFile(Name(layer, std::string("ffn.shared_experts.") + kParts[part] + ".scale")));
    }
    m->activation = Scratch<float>(kHidden);
    m->accumulator = Scratch<float>(kHidden);
    m->weights = Scratch<float>(kRouted);
    m->indices = Scratch<uint32_t>(kRouted);
    m->gate_out = Scratch<uint16_t>(kInter);
    m->up_out = Scratch<uint16_t>(kInter);
    m->hidden = Scratch<uint16_t>(kInter);
    m->down = Scratch<uint16_t>(kHidden);
    m->payload.assign(kExperts * 3u, (uint8_t *)0);
    m->scale.assign(kExperts * 3u, (uint8_t *)0);
}

static void MoeStep(Moe *m, const uint16_t *x, uint16_t *out)
{
    uint32_t chosen[kRouted];
    float chosen_weight[kRouted];
    CUDA(SparkDsv41FlashLaunchGate(0, x, m->gate, m->bias, m->indices, m->weights, kExperts, kHidden, kRouted, kRouteScale));
    CUDA(cudaMemcpy(chosen, m->indices, sizeof(chosen), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(chosen_weight, m->weights, sizeof(chosen_weight), cudaMemcpyDeviceToHost));
    for (uint32_t a = 0u; a < kRouted; a++)
        for (uint32_t b = a + 1u; b < kRouted; b++)
            if (chosen[b] < chosen[a])
            {
                uint32_t index = chosen[a]; chosen[a] = chosen[b]; chosen[b] = index;
                float weight = chosen_weight[a]; chosen_weight[a] = chosen_weight[b]; chosen_weight[b] = weight;
            }
    CUDA(cudaMemset(m->accumulator, 0, kHidden * sizeof(float)));
    for (uint32_t slot = 0u; slot < kRouted; slot++)
    {
        uint32_t expert = chosen[slot];
        for (uint32_t part = 0u; part < 3u; part++)
            if (m->payload[expert * 3u + part] == 0)
            {
                std::string base = "ffn.experts." + std::to_string(expert) + "." + kParts[part];
                m->payload[expert * 3u + part] = Device<uint8_t>(ReadFile(Name(m->layer, base + ".weight")));
                m->scale[expert * 3u + part] = Device<uint8_t>(ReadFile(Name(m->layer, base + ".scale")));
            }
        CUDA(SparkDsv41FlashLaunchMxfp4Linear(0, m->payload[expert * 3u], m->scale[expert * 3u], x, m->activation, m->gate_out, kHidden, kInter));
        CUDA(SparkDsv41FlashLaunchMxfp4Linear(0, m->payload[expert * 3u + 2u], m->scale[expert * 3u + 2u], x, m->activation, m->up_out, kHidden, kInter));
        CUDA(SparkDsv41FlashLaunchSwiglu(0, m->gate_out, m->up_out, m->hidden, kInter, kSwigluLimit, chosen_weight[slot]));
        CUDA(SparkDsv41FlashLaunchMxfp4Linear(0, m->payload[expert * 3u + 1u], m->scale[expert * 3u + 1u], m->hidden, m->activation, m->down, kInter, kHidden));
        SparkDsv41FlashAccumulateKernel<<<(kHidden + 255u) / 256u, 256u>>>(m->accumulator, m->down, kHidden);
        CUDA(cudaGetLastError());
    }
    CUDA(SparkDsv41FlashLaunchFp8BlockLinear(0, m->shared[0], m->shared_scale[0], x, m->activation, m->gate_out, 1u, kHidden, kInter));
    CUDA(SparkDsv41FlashLaunchFp8BlockLinear(0, m->shared[2], m->shared_scale[2], x, m->activation, m->up_out, 1u, kHidden, kInter));
    CUDA(SparkDsv41FlashLaunchSwiglu(0, m->gate_out, m->up_out, m->hidden, kInter, kSwigluLimit, 1.0f));
    CUDA(SparkDsv41FlashLaunchFp8BlockLinear(0, m->shared[1], m->shared_scale[1], m->hidden, m->activation, m->down, 1u, kInter, kHidden));
    SparkDsv41FlashAccumulateKernel<<<(kHidden + 255u) / 256u, 256u>>>(m->accumulator, m->down, kHidden);
    SparkDsv41FlashNarrowKernel<<<(kHidden + 255u) / 256u, 256u>>>(m->accumulator, out, kHidden);
    CUDA(cudaGetLastError());
}

static double Compare(const std::vector<uint16_t> &got, const uint16_t *expected, uint32_t positions, uint32_t width)
{
    double worst_cosine = 1.0;
    for (uint32_t position = 0u; position < positions; position++)
    {
        double dot = 0.0, a2 = 0.0, b2 = 0.0;
        for (uint32_t element = 0u; element < width; element++)
        {
            double a = Bf16(got[(size_t)position * width + element]);
            double b = Bf16(expected[(size_t)position * width + element]);
            dot += a * b;
            a2 += a * a;
            b2 += b * b;
        }
        double cosine = dot / sqrt(a2 * b2 + 1e-300);
        if (cosine < worst_cosine)
            worst_cosine = cosine;
        printf("{\"position\": %u, \"cosine\": %.8f, \"norm_ratio\": %.6f}\n", position, cosine, sqrt(a2 / (b2 + 1e-300)));
    }
    printf("{\"worst_cosine\": %.8f}\n", worst_cosine);
    return worst_cosine;
}

static void Finish(const char *path, const uint16_t *device, uint32_t positions, uint32_t width, const char *expected_name)
{
    std::vector<uint16_t> got((size_t)positions * width);
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(got.data(), device, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    FILE *file = fopen(path, "wb");
    if (file == 0 || fwrite(got.data(), sizeof(uint16_t), got.size(), file) != got.size())
    {
        fprintf(stderr, "FAIL cannot write %s\n", path);
        exit(1);
    }
    fclose(file);
    std::vector<uint8_t> expected = ReadFile(expected_name);
    Compare(got, (const uint16_t *)expected.data(), positions, width);
}

int main(int argc, char **argv)
{
    if (argc != 6 || (strcmp(argv[1], "attn") != 0 && strcmp(argv[1], "ffn") != 0 && strcmp(argv[1], "layer") != 0))
    {
        fprintf(stderr, "usage: %s attn|ffn|layer FIXTURE_DIR LAYER POSITIONS OUT\n", argv[0]);
        return 2;
    }
    directory = argv[2];
    uint32_t layer = (uint32_t)strtoul(argv[3], 0, 10);
    uint32_t positions = (uint32_t)strtoul(argv[4], 0, 10);
    if (strcmp(argv[1], "attn") == 0)
    {
        Attention attention;
        AttentionLoad(&attention, layer);
        uint16_t *inputs = Device<uint16_t>(ReadFile("attn_in.bin"));
        uint16_t *outputs = Scratch<uint16_t>((size_t)positions * kHidden);
        for (uint32_t position = 0u; position < positions; position++)
            AttentionStep(&attention, position, inputs + (size_t)position * kHidden, outputs + (size_t)position * kHidden);
        Finish(argv[5], outputs, positions, kHidden, "attn_out.bin");
        return 0;
    }
    if (strcmp(argv[1], "ffn") == 0)
    {
        Moe moe;
        MoeLoad(&moe, layer);
        uint16_t *inputs = Device<uint16_t>(ReadFile("ffn_in.bin"));
        uint16_t *outputs = Scratch<uint16_t>((size_t)positions * kHidden);
        for (uint32_t position = 0u; position < positions; position++)
            MoeStep(&moe, inputs + (size_t)position * kHidden, outputs + (size_t)position * kHidden);
        Finish(argv[5], outputs, positions, kHidden, "ffn_out.bin");
        return 0;
    }
    Attention attention;
    Moe moe;
    AttentionLoad(&attention, layer);
    MoeLoad(&moe, layer);
    float *attn_fn = Device<float>(ReadFile(Name(layer, "hc_attn_fn")));
    float *attn_base = Device<float>(ReadFile(Name(layer, "hc_attn_base")));
    float *attn_scale = Device<float>(ReadFile(Name(layer, "hc_attn_scale")));
    float *ffn_fn = Device<float>(ReadFile(Name(layer, "hc_ffn_fn")));
    float *ffn_base = Device<float>(ReadFile(Name(layer, "hc_ffn_base")));
    float *ffn_scale = Device<float>(ReadFile(Name(layer, "hc_ffn_scale")));
    uint16_t *attn_norm = Device<uint16_t>(ReadFile(Name(layer, "attn_norm.weight")));
    uint16_t *ffn_norm = Device<uint16_t>(ReadFile(Name(layer, "ffn_norm.weight")));
    std::vector<uint8_t> pre_mix_host = ReadFile("pre_mix_in.bin");
    float *pre_mix_all = Device<float>(pre_mix_host);
    uint16_t *inputs = Device<uint16_t>(ReadFile("layer_in.bin"));
    uint16_t *outputs = Scratch<uint16_t>((size_t)positions * kHc * kHidden);
    uint16_t *middle = Scratch<uint16_t>(kHc * kHidden);
    uint16_t *collapsed = Scratch<uint16_t>(kHidden);
    uint16_t *normed = Scratch<uint16_t>(kHidden);
    uint16_t *sublayer = Scratch<uint16_t>(kHidden);
    float *mixes = Scratch<float>(SPARK_DSV41_FLASH_HC_MIX_MAX);
    float *attn_pre = Scratch<float>(kHc), *attn_post = Scratch<float>(kHc), *attn_comb = Scratch<float>(kHc * kHc);
    float *ffn_pre = Scratch<float>(kHc), *ffn_post = Scratch<float>(kHc), *ffn_comb = Scratch<float>(kHc * kHc);
    for (uint32_t position = 0u; position < positions; position++)
    {
        const uint16_t *streams = inputs + (size_t)position * kHc * kHidden;
        CUDA(SparkDsv41FlashLaunchHcMixes(0, streams, attn_fn, attn_scale, attn_base, mixes, attn_pre, attn_post, attn_comb, kHc, kHidden, kSinkhorn, kEpsilon, kHcEpsilon));
        SparkDsv41FlashHcPreKernel<<<(kHidden + 255u) / 256u, 256u>>>(streams, pre_mix_all + (size_t)position * kHc, collapsed, kHc, kHidden);
        CUDA(SparkDsv41FlashLaunchRmsNorm(0, collapsed, attn_norm, normed, 1u, kHidden, kEpsilon));
        AttentionStep(&attention, position, normed, sublayer);
        SparkDsv41FlashHcPostKernel<<<(kHidden + 255u) / 256u, 256u>>>(sublayer, streams, attn_post, attn_comb, middle, kHc, kHidden);
        CUDA(SparkDsv41FlashLaunchHcMixes(0, middle, ffn_fn, ffn_scale, ffn_base, mixes, ffn_pre, ffn_post, ffn_comb, kHc, kHidden, kSinkhorn, kEpsilon, kHcEpsilon));
        SparkDsv41FlashHcPreKernel<<<(kHidden + 255u) / 256u, 256u>>>(middle, attn_pre, collapsed, kHc, kHidden);
        CUDA(SparkDsv41FlashLaunchRmsNorm(0, collapsed, ffn_norm, normed, 1u, kHidden, kEpsilon));
        MoeStep(&moe, normed, sublayer);
        SparkDsv41FlashHcPostKernel<<<(kHidden + 255u) / 256u, 256u>>>(sublayer, middle, ffn_post, ffn_comb, outputs + (size_t)position * kHc * kHidden, kHc, kHidden);
        CUDA(cudaGetLastError());
    }
    Finish(argv[5], outputs, positions, kHc * kHidden, "layer_out.bin");
    return 0;
}
