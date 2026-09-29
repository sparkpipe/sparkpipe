#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__)
#define SPARK_KV_QUANT_SIM_FN static inline __host__ __device__
#else
#define SPARK_KV_QUANT_SIM_FN static inline
#endif

#define SPARK_KV_QUANT_SIM_BF16 0u
#define SPARK_KV_QUANT_SIM_FP8_E4M3 1u
#define SPARK_KV_QUANT_SIM_MXFP4 2u
#define SPARK_KV_QUANT_SIM_CODEC_COUNT 3u

#define SPARK_KV_STATE_SIM_FP32 0u
#define SPARK_KV_STATE_SIM_BF16 1u
#define SPARK_KV_STATE_SIM_CODEC_COUNT 2u

#define SPARK_KV_QUANT_SIM_SCALE_EXPONENT_MIN (-127)
#define SPARK_KV_QUANT_SIM_SCALE_EXPONENT_MAX 127
#define SPARK_KV_QUANT_SIM_TOKEN_BYTES 64u

typedef struct SparkKvQuantSim
{
	uint32_t latent_codec;
	uint32_t latent_group;
	uint32_t index_codec;
	uint32_t index_group;
	uint32_t state_codec;
}
SparkKvQuantSim;

SPARK_KV_QUANT_SIM_FN uint32_t SparkKvQuantSimGroupValid(uint32_t codec,uint32_t group)
{
	if ( codec == SPARK_KV_QUANT_SIM_BF16 )
		return(group == 0u ? 1u : 0u);
	if ( codec == SPARK_KV_QUANT_SIM_FP8_E4M3 )
		return(group == 64u || group == 128u ? 1u : 0u);
	if ( codec == SPARK_KV_QUANT_SIM_MXFP4 )
		return(group == 32u ? 1u : 0u);
	return(0u);
}

SPARK_KV_QUANT_SIM_FN uint32_t SparkKvQuantSimRowValid(uint32_t codec,uint32_t group,uint32_t width)
{
	if ( SparkKvQuantSimGroupValid(codec,group) == 0u || width == 0u )
		return(0u);
	return(group == 0u || width % group == 0u ? 1u : 0u);
}

SPARK_KV_QUANT_SIM_FN uint32_t SparkKvQuantSimValid(const SparkKvQuantSim *sim)
{
	if ( sim == 0 || sim->state_codec >= SPARK_KV_STATE_SIM_CODEC_COUNT )
		return(0u);
	return(SparkKvQuantSimGroupValid(sim->latent_codec,sim->latent_group) != 0u &&
		SparkKvQuantSimGroupValid(sim->index_codec,sim->index_group) != 0u ? 1u : 0u);
}

SPARK_KV_QUANT_SIM_FN uint32_t SparkKvQuantSimIdentity(const SparkKvQuantSim *sim)
{
	return(sim != 0 && sim->latent_codec == SPARK_KV_QUANT_SIM_BF16 &&
		sim->index_codec == SPARK_KV_QUANT_SIM_BF16 &&
		sim->state_codec == SPARK_KV_STATE_SIM_FP32 ? 1u : 0u);
}

#define SPARK_KV_QUANT_SIM_PACKED_MASK 0x7fu

SPARK_KV_QUANT_SIM_FN uint32_t SparkKvQuantSimPackStream(uint32_t codec,uint32_t group)
{
	return(codec | (codec == SPARK_KV_QUANT_SIM_FP8_E4M3 && group == 64u ? 4u : 0u));
}

SPARK_KV_QUANT_SIM_FN uint32_t SparkKvQuantSimStreamGroup(uint32_t codec,uint32_t narrow)
{
	if ( codec == SPARK_KV_QUANT_SIM_FP8_E4M3 )
		return(narrow != 0u ? 64u : 128u);
	if ( codec == SPARK_KV_QUANT_SIM_MXFP4 )
		return(narrow != 0u ? 0u : 32u);
	return(narrow != 0u ? 1u : 0u);
}

SPARK_KV_QUANT_SIM_FN int32_t SparkKvQuantSimPack(const SparkKvQuantSim *sim,uint32_t *packed)
{
	if ( packed == 0 || SparkKvQuantSimValid(sim) == 0u )
		return(-1);
	*packed = SparkKvQuantSimPackStream(sim->latent_codec,sim->latent_group) |
		(SparkKvQuantSimPackStream(sim->index_codec,sim->index_group) << 3u) |
		(sim->state_codec << 6u);
	return(0);
}

SPARK_KV_QUANT_SIM_FN int32_t SparkKvQuantSimUnpack(uint32_t packed,SparkKvQuantSim *sim)
{
	SparkKvQuantSim unpacked;
	if ( sim == 0 || (packed & ~SPARK_KV_QUANT_SIM_PACKED_MASK) != 0u )
		return(-1);
	unpacked.latent_codec = packed & 3u;
	unpacked.latent_group = SparkKvQuantSimStreamGroup(unpacked.latent_codec,(packed >> 2u) & 1u);
	unpacked.index_codec = (packed >> 3u) & 3u;
	unpacked.index_group = SparkKvQuantSimStreamGroup(unpacked.index_codec,(packed >> 5u) & 1u);
	unpacked.state_codec = (packed >> 6u) & 1u;
	if ( SparkKvQuantSimValid(&unpacked) == 0u )
		return(-2);
	*sim = unpacked;
	return(0);
}

static inline const char *SparkKvQuantSimCodecName(uint32_t codec)
{
	if ( codec == SPARK_KV_QUANT_SIM_BF16 )
		return("bf16");
	if ( codec == SPARK_KV_QUANT_SIM_FP8_E4M3 )
		return("fp8_e4m3");
	if ( codec == SPARK_KV_QUANT_SIM_MXFP4 )
		return("mxfp4");
	return(0);
}

static inline const char *SparkKvStateSimCodecName(uint32_t codec)
{
	if ( codec == SPARK_KV_STATE_SIM_FP32 )
		return("fp32");
	if ( codec == SPARK_KV_STATE_SIM_BF16 )
		return("bf16");
	return(0);
}

static inline int32_t SparkKvQuantSimParseCodec(const char *name,uint32_t *codec)
{
	uint32_t candidate;
	if ( name == 0 || codec == 0 )
		return(-1);
	for (candidate = 0u; candidate < SPARK_KV_QUANT_SIM_CODEC_COUNT; candidate++)
		if ( strcmp(name,SparkKvQuantSimCodecName(candidate)) == 0 )
		{
			*codec = candidate;
			return(0);
		}
	return(-2);
}

static inline int32_t SparkKvStateSimParseCodec(const char *name,uint32_t *codec)
{
	uint32_t candidate;
	if ( name == 0 || codec == 0 )
		return(-1);
	for (candidate = 0u; candidate < SPARK_KV_STATE_SIM_CODEC_COUNT; candidate++)
		if ( strcmp(name,SparkKvStateSimCodecName(candidate)) == 0 )
		{
			*codec = candidate;
			return(0);
		}
	return(-2);
}

static inline size_t SparkKvQuantSimAppend(char *out,size_t cap,size_t used,const char *text)
{
	size_t length = strlen(text);
	if ( used == (size_t)-1 || used + length >= cap )
		return((size_t)-1);
	memcpy(out + used,text,length + 1u);
	return(used + length);
}

static inline size_t SparkKvQuantSimAppendCodec(char *out,size_t cap,size_t used,uint32_t codec,uint32_t group)
{
	char digits[4];
	used = SparkKvQuantSimAppend(out,cap,used,SparkKvQuantSimCodecName(codec));
	if ( group == 0u )
		return(used);
	digits[0] = (char)('0' + (group / 100u) % 10u);
	digits[1] = (char)('0' + (group / 10u) % 10u);
	digits[2] = (char)('0' + group % 10u);
	digits[3] = 0;
	used = SparkKvQuantSimAppend(out,cap,used,".g");
	return(SparkKvQuantSimAppend(out,cap,used,digits + (group < 100u ? 1 : 0)));
}

static inline int32_t SparkKvQuantSimToken(const SparkKvQuantSim *sim,char *out,size_t cap)
{
	size_t used = 0u;
	if ( out == 0 || cap == 0u )
		return(-1);
	out[0] = 0;
	if ( SparkKvQuantSimValid(sim) == 0u )
		return(-2);
	used = SparkKvQuantSimAppend(out,cap,used,"kv=");
	used = SparkKvQuantSimAppendCodec(out,cap,used,sim->latent_codec,sim->latent_group);
	used = SparkKvQuantSimAppend(out,cap,used,"/");
	used = SparkKvQuantSimAppendCodec(out,cap,used,sim->index_codec,sim->index_group);
	used = SparkKvQuantSimAppend(out,cap,used,"/");
	used = SparkKvQuantSimAppend(out,cap,used,SparkKvStateSimCodecName(sim->state_codec));
	used = SparkKvQuantSimAppend(out,cap,used,SparkKvQuantSimIdentity(sim) != 0u ? "/store" : "/sim");
	if ( used == (size_t)-1 )
	{
		out[0] = 0;
		return(-3);
	}
	return(0);
}
