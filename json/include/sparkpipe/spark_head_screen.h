#pragma once

#include <stdint.h>

#define SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE 32u
#define SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT 128u

static inline uint64_t SparkHeadCertifiedFp8PayloadBytes(
	uint64_t vocabulary_rows,uint64_t hidden_dimension)
{
	return(vocabulary_rows * hidden_dimension);
}

static inline uint64_t SparkHeadCertifiedFp8NormBytes(
	uint64_t vocabulary_rows,uint64_t hidden_dimension)
{
	return(vocabulary_rows *
		(hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE) *
		sizeof(float));
}

static inline uint64_t SparkHeadCertifiedFp8ScratchBytes(
	uint64_t vocabulary_rows,uint64_t hidden_dimension)
{
	uint64_t groups = hidden_dimension /
		SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	return(((2u * vocabulary_rows) + groups +
		(2u * SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT)) * sizeof(float));
}

static inline uint64_t SparkHeadCertifiedFp8CandidateBytes(
	uint64_t vocabulary_rows)
{
	return(vocabulary_rows * sizeof(uint32_t));
}
