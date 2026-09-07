
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define VERIFY_LANES 32u
#define VERIFY_TILE_K 128u
#define VERIFY_SWIZZLE_CHUNKS 8u

typedef struct layout_mode
{
	uint32_t shape[4],stride[4],rank;
}
layout_mode_t;

static uint32_t layout_mode_index(const layout_mode_t *mode, uint32_t coordinate)
{
	uint32_t index = 0,rank,digit;
	for (rank = 0; rank < mode->rank; ++rank)
	{
		digit = coordinate % mode->shape[rank];
		coordinate = coordinate / mode->shape[rank];
		index += digit * mode->stride[rank];
	}
	return(index);
}

static uint32_t layout_mode_size(const layout_mode_t *mode)
{
	uint32_t size = 1,rank;
	for (rank = 0; rank < mode->rank; ++rank)
		size *= mode->shape[rank];
	return(size);
}

static uint32_t cute_index(const layout_mode_t *thread_mode, const layout_mode_t *value_mode, uint32_t lane, uint32_t value)
{
	return(layout_mode_index(thread_mode,lane) + layout_mode_index(value_mode,value));
}

static void kernel_a_coordinate(uint32_t lane, uint32_t reg, uint32_t byte, uint32_t *m, uint32_t *k)
{
	*m = (lane / 4u) + (8u * (reg % 2u));
	*k = (4u * (lane % 4u)) + byte + (16u * (reg / 2u));
}

static void kernel_b_coordinate(uint32_t lane, uint32_t reg, uint32_t byte, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = (4u * (lane % 4u)) + byte + (16u * reg);
}

static void kernel_c_coordinate(uint32_t lane, uint32_t accumulator, uint32_t *m, uint32_t *n)
{
	*m = (lane / 4u) + (8u * (accumulator / 2u));
	*n = (2u * (lane % 4u)) + (accumulator % 2u);
}

static int32_t verify_operand_a(void)
{
	layout_mode_t thread_mode = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t value_mode = { { 4u, 2u, 2u, 0u }, { 16u, 8u, 256u, 0u }, 3u };
	uint8_t seen[512];
	uint32_t lane,value,reg,byte,expected,m,k,actual,mismatches = 0,uncovered = 0;
	memset(seen,0,sizeof(seen));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
		for (value = 0; value < layout_mode_size(&value_mode); ++value)
		{
			expected = cute_index(&thread_mode,&value_mode,lane,value);
			reg = value / 4u;
			byte = value % 4u;
			kernel_a_coordinate(lane,reg,byte,&m,&k);
			actual = m + (16u * k);
			if ( actual != expected )
				mismatches++;
			seen[expected & 511u]++;
		}
	for (value = 0; value < 512u; ++value)
		if ( seen[value] != 1 )
			uncovered++;
	printf("  A m16n8k32  mismatches=%-4u  elements covered exactly once=%u/512\n",
		mismatches,512u - uncovered);
	return(mismatches == 0 && uncovered == 0 ? 0 : -1);
}

static int32_t verify_operand_b(void)
{
	layout_mode_t thread_mode = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t value_mode = { { 4u, 2u, 0u, 0u }, { 8u, 128u, 0u, 0u }, 2u };
	uint8_t seen[256];
	uint32_t lane,value,reg,byte,expected,n,k,actual,mismatches = 0,uncovered = 0;
	memset(seen,0,sizeof(seen));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
		for (value = 0; value < layout_mode_size(&value_mode); ++value)
		{
			expected = cute_index(&thread_mode,&value_mode,lane,value);
			reg = value / 4u;
			byte = value % 4u;
			kernel_b_coordinate(lane,reg,byte,&n,&k);
			actual = n + (8u * k);
			if ( actual != expected )
				mismatches++;
			seen[expected & 255u]++;
		}
	for (value = 0; value < 256u; ++value)
		if ( seen[value] != 1 )
			uncovered++;
	printf("  B m16n8k32  mismatches=%-4u  elements covered exactly once=%u/256\n",
		mismatches,256u - uncovered);
	return(mismatches == 0 && uncovered == 0 ? 0 : -1);
}

static int32_t verify_accumulator_c(void)
{
	layout_mode_t thread_mode = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t value_mode = { { 2u, 2u, 0u, 0u }, { 16u, 8u, 0u, 0u }, 2u };
	uint8_t seen[128];
	uint32_t lane,value,expected,m,n,actual,mismatches = 0,uncovered = 0;
	memset(seen,0,sizeof(seen));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
		for (value = 0; value < layout_mode_size(&value_mode); ++value)
		{
			expected = cute_index(&thread_mode,&value_mode,lane,value);
			kernel_c_coordinate(lane,value,&m,&n);
			actual = m + (16u * n);
			if ( actual != expected )
				mismatches++;
			seen[expected & 127u]++;
		}
	for (value = 0; value < 128u; ++value)
		if ( seen[value] != 1 )
			uncovered++;
	printf("  C 16x8 row  mismatches=%-4u  elements covered exactly once=%u/128\n",
		mismatches,128u - uncovered);
	return(mismatches == 0 && uncovered == 0 ? 0 : -1);
}

static int32_t verify_known_bad_c_is_rejected(void)
{
	layout_mode_t thread_mode = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t value_mode = { { 2u, 2u, 0u, 0u }, { 16u, 8u, 0u, 0u }, 2u };
	uint32_t lane,value,expected,m,n,actual,mismatches = 0;
	for (lane = 0; lane < VERIFY_LANES; ++lane)
		for (value = 0; value < 4u; ++value)
		{
			expected = cute_index(&thread_mode,&value_mode,lane,value);
			m = ((value & 2u) << 2u) + (lane >> 2u) + ((value & 1u) << 3u);
			n = 2u * (lane % 4u);
			actual = m + (16u * n);
			if ( actual != expected )
				mismatches++;
		}
	printf("  negative control (first-draft C mapping) mismatches=%u/128 %s\n",
		mismatches,mismatches > 0 ? "- correctly rejected" : "- VERIFIER IS BROKEN");
	return(mismatches > 0 ? 0 : -1);
}

static uint32_t ldmatrix_row_for_lane(uint32_t lane)
{
	return((lane % 8u) + (8u * ((lane / 8u) % 2u)));
}

static uint32_t ldmatrix_chunk_for_lane(uint32_t lane)
{
	return(lane / 16u);
}

static uint32_t swizzle_chunk(uint32_t chunk, uint32_t row)
{
	return(chunk ^ (row % VERIFY_SWIZZLE_CHUNKS));
}

static uint32_t count_bank_conflicts(int32_t apply_swizzle, uint32_t k_base)
{
	uint32_t bank_population[32],lane,row,chunk,byte_offset,bank,worst = 0;
	memset(bank_population,0,sizeof(bank_population));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
	{
		row = ldmatrix_row_for_lane(lane);
		chunk = (k_base / 16u) + ldmatrix_chunk_for_lane(lane);
		if ( apply_swizzle != 0 )
			chunk = swizzle_chunk(chunk,row);
		byte_offset = (row * VERIFY_TILE_K) + (chunk * 16u);
		bank = (byte_offset / 4u) % 32u;
		bank_population[bank]++;
	}
	for (bank = 0; bank < 32u; ++bank)
		if ( bank_population[bank] > worst )
			worst = bank_population[bank];
	return(worst);
}

#define PIPE_MAX_STAGES 8
#define PIPE_MAX_K_TILES 64

static int32_t verify_pipeline_schedule(uint32_t stages, uint32_t k_tiles)
{
	uint32_t produced_round[PIPE_MAX_STAGES],consumed_round[PIPE_MAX_STAGES];
	uint32_t k_tile,stage,ahead,round,expected_phase,fill,errors = 0;
	for (stage = 0; stage < stages; ++stage)
	{
		produced_round[stage] = 0xffffffffu;
		consumed_round[stage] = 0xffffffffu;
	}
	for (fill = 0; fill + 1u < stages && fill < k_tiles; ++fill)
		produced_round[fill % stages] = fill / stages;
	for (k_tile = 0; k_tile < k_tiles; ++k_tile)
	{
		stage = k_tile % stages;
		round = k_tile / stages;
		ahead = k_tile + stages - 1u;
		if ( ahead < k_tiles )
		{
			if ( produced_round[ahead % stages] != 0xffffffffu
				&& consumed_round[ahead % stages] != produced_round[ahead % stages] )
				errors++;
			produced_round[ahead % stages] = ahead / stages;
		}
		if ( produced_round[stage] != round )
			errors++;
		expected_phase = round & 1u;
		if ( expected_phase != ((k_tile / stages) & 1u) )
			errors++;
		consumed_round[stage] = round;
	}
	return((int32_t)errors);
}

static int32_t verify_pipeline_persistent(uint32_t stages, uint32_t k_tiles, uint32_t output_tiles)
{
	uint32_t produced[PIPE_MAX_STAGES],consumed[PIPE_MAX_STAGES];
	uint32_t k_tile,stage,ahead,fill,tile,errors = 0,stale = 0;
	uint32_t phase = 0;
	for (stage = 0; stage < stages; ++stage)
	{
		produced[stage] = 0;
		consumed[stage] = 0;
	}
	for (tile = 0; tile < output_tiles; ++tile)
	{
		for (fill = 0; fill + 1u < stages && fill < k_tiles; ++fill)
		{
			if ( produced[fill % stages] != consumed[fill % stages] )
				errors++;
			produced[fill % stages]++;
		}
		for (k_tile = 0; k_tile < k_tiles; ++k_tile)
		{
			stage = k_tile % stages;
			ahead = k_tile + stages - 1u;
			if ( ahead < k_tiles )
			{
				if ( produced[ahead % stages] != consumed[ahead % stages] )
					errors++;
				produced[ahead % stages]++;
			}
			if ( ((phase >> stage) & 1u) != (consumed[stage] & 1u) )
				errors++;
			if ( produced[stage] != consumed[stage] + 1u )
				errors++;
			if ( ((k_tile / stages) & 1u) != (consumed[stage] & 1u) )
				stale++;
			consumed[stage]++;
			phase ^= 1u << stage;
		}
		for (stage = 0; stage < stages; ++stage)
			if ( produced[stage] != consumed[stage] )
				errors++;
	}
	if ( output_tiles > 1u && (k_tiles % (2u * stages)) != 0u && stale == 0u )
		errors++;
	if ( output_tiles == 1u && stale != 0u )
		errors++;
	return((int32_t)errors);
}

static int32_t verify_pipeline_persistent_matrix(void)
{
	uint32_t stages,k_tiles,tiles;
	int32_t errors,total = 0;
	printf("\npipeline schedule, persistent multi-tile: errors per (stages, k_tiles, tiles) cell\n");
	for (stages = 2u; stages <= 6u; ++stages)
	{
		printf("  stages=%u :",stages);
		for (k_tiles = 1u; k_tiles <= 21u; k_tiles += 5u)
			for (tiles = 1u; tiles <= 3u; ++tiles)
			{
				errors = verify_pipeline_persistent(stages,k_tiles,tiles);
				printf(" k=%-2u,t=%u:%d",k_tiles,tiles,errors);
				total += errors;
			}
		printf("\n");
	}
	return(total);
}

static int32_t verify_pipeline_matrix(void)
{
	uint32_t stages,k_tiles;
	int32_t errors,total = 0;
	printf("\npipeline schedule: stages x k_tiles, errors per cell\n");
	for (stages = 2u; stages <= 6u; ++stages)
	{
		printf("  stages=%u :",stages);
		for (k_tiles = 1u; k_tiles <= 48u; k_tiles += 8u)
		{
			errors = verify_pipeline_schedule(stages,k_tiles);
			printf(" k=%-2u:%d",k_tiles,errors);
			total += errors;
		}
		printf("\n");
	}
	return(total);
}


static int32_t verify_sm120_equals_sm89(void)
{
	layout_mode_t sm89_a_t = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t sm89_a_v = { { 4u, 2u, 2u, 0u }, { 16u, 8u, 256u, 0u }, 3u };
	layout_mode_t sm80_a_t = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t sm80_a_v = { { 4u, 2u, 2u, 0u }, { 16u, 8u, 256u, 0u }, 3u };
	layout_mode_t sm89_b_t = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t sm89_b_v = { { 4u, 2u, 0u, 0u }, { 8u, 128u, 0u, 0u }, 2u };
	layout_mode_t sm80_b_t = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t sm80_b_v = { { 4u, 2u, 0u, 0u }, { 8u, 128u, 0u, 0u }, 2u };
	uint32_t lane,value,differences = 0;
	for (lane = 0; lane < VERIFY_LANES; ++lane)
	{
		for (value = 0; value < 16u; ++value)
			if ( cute_index(&sm89_a_t,&sm89_a_v,lane,value) != cute_index(&sm80_a_t,&sm80_a_v,lane,value) )
				differences++;
		for (value = 0; value < 8u; ++value)
			if ( cute_index(&sm89_b_t,&sm89_b_v,lane,value) != cute_index(&sm80_b_t,&sm80_b_v,lane,value) )
				differences++;
	}
	printf("  SM120_16x8x32_TN vs SM89 e4m3: differing (lane,value) pairs=%u/768\n",differences);
	return(differences == 0 ? 0 : -1);
}

static void kernel_nvfp4_a_coordinate(uint32_t lane, uint32_t reg, uint32_t nibble, uint32_t *m, uint32_t *k)
{
	*m = (lane / 4u) + (8u * (reg % 2u));
	*k = (8u * (lane % 4u)) + nibble + (32u * (reg / 2u));
}

static void kernel_nvfp4_b_coordinate(uint32_t lane, uint32_t reg, uint32_t nibble, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = (8u * (lane % 4u)) + nibble + (32u * reg);
}

static int32_t verify_nvfp4_operands(void)
{
	layout_mode_t a_thread = { { 4u, 8u, 0u, 0u }, { 128u, 1u, 0u, 0u }, 2u };
	layout_mode_t a_value = { { 8u, 2u, 2u, 0u }, { 16u, 8u, 512u, 0u }, 3u };
	layout_mode_t b_thread = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t b_value = { { 8u, 2u, 0u, 0u }, { 8u, 256u, 0u, 0u }, 2u };
	uint8_t seen_a[1024],seen_b[512];
	uint32_t lane,value,expected,m,n,k,actual,a_bad = 0,b_bad = 0,a_gap = 0,b_gap = 0;
	memset(seen_a,0,sizeof(seen_a));
	memset(seen_b,0,sizeof(seen_b));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
	{
		for (value = 0; value < 32u; ++value)
		{
			expected = cute_index(&a_thread,&a_value,lane,value);
			kernel_nvfp4_a_coordinate(lane,value / 8u,value % 8u,&m,&k);
			actual = m + (16u * k);
			if ( actual != expected )
				a_bad++;
			seen_a[expected & 1023u]++;
		}
		for (value = 0; value < 16u; ++value)
		{
			expected = cute_index(&b_thread,&b_value,lane,value);
			kernel_nvfp4_b_coordinate(lane,value / 8u,value % 8u,&n,&k);
			actual = n + (8u * k);
			if ( actual != expected )
				b_bad++;
			seen_b[expected & 511u]++;
		}
	}
	for (value = 0; value < 1024u; ++value)
		if ( seen_a[value] != 1 )
			a_gap++;
	for (value = 0; value < 512u; ++value)
		if ( seen_b[value] != 1 )
			b_gap++;
	printf("  A m16n8k64 nvf4  mismatches=%-4u covered exactly once=%u/1024\n",a_bad,1024u - a_gap);
	printf("  B m16n8k64 nvf4  mismatches=%-4u covered exactly once=%u/512\n",b_bad,512u - b_gap);
	return((a_bad == 0 && b_bad == 0 && a_gap == 0 && b_gap == 0) ? 0 : -1);
}


static void kernel_sfa_coordinate(uint32_t lane, uint32_t value, uint32_t *m, uint32_t *k)
{
	*m = (8u * (lane % 2u)) + (lane / 4u);
	*k = value;
}

static void kernel_sfb_coordinate(uint32_t lane, uint32_t value, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = value;
}

static int32_t verify_nvfp4_scale_layouts(void)
{
	layout_mode_t sfa_thread = { { 2u, 2u, 8u, 0u }, { 8u, 0u, 1u, 0u }, 3u };
	layout_mode_t sfa_value = { { 64u, 0u, 0u, 0u }, { 16u, 0u, 0u, 0u }, 1u };
	layout_mode_t sfb_thread = { { 4u, 8u, 0u, 0u }, { 0u, 1u, 0u, 0u }, 2u };
	layout_mode_t sfb_value = { { 64u, 0u, 0u, 0u }, { 8u, 0u, 0u, 0u }, 1u };
	uint32_t lane,value,expected,m,n,k,actual,a_bad = 0,b_bad = 0;
	uint32_t a_multiplicity[1024],b_multiplicity[512],a_max = 0,b_max = 0,index;
	memset(a_multiplicity,0,sizeof(a_multiplicity));
	memset(b_multiplicity,0,sizeof(b_multiplicity));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
		for (value = 0; value < 64u; ++value)
		{
			expected = cute_index(&sfa_thread,&sfa_value,lane,value);
			kernel_sfa_coordinate(lane,value,&m,&k);
			actual = m + (16u * k);
			if ( actual != expected )
				a_bad++;
			a_multiplicity[expected & 1023u]++;
			expected = cute_index(&sfb_thread,&sfb_value,lane,value);
			kernel_sfb_coordinate(lane,value,&n,&k);
			actual = n + (8u * k);
			if ( actual != expected )
				b_bad++;
			b_multiplicity[expected & 511u]++;
		}
	for (index = 0; index < 1024u; ++index)
		if ( a_multiplicity[index] > a_max )
			a_max = a_multiplicity[index];
	for (index = 0; index < 512u; ++index)
		if ( b_multiplicity[index] > b_max )
			b_max = b_multiplicity[index];
	printf("  SFA mismatches=%-4u lanes sharing each element=%u (CUTLASS: effectively 16 threads)\n",a_bad,a_max);
	printf("  SFB mismatches=%-4u lanes sharing each element=%u (CUTLASS: effectively 8 threads)\n",b_bad,b_max);
	if ( a_max != 2u || b_max != 4u )
	{
		printf("  SHARING FACTOR DISAGREES WITH THE STRIDE-0 MODE - decode is wrong\n");
		return(-1);
	}
	return((a_bad == 0 && b_bad == 0) ? 0 : -1);
}


static void lm_mma8_a(uint32_t lane, uint32_t reg, uint32_t byte, uint32_t *m, uint32_t *k)
{
	*m = (lane / 4u) + (8u * (reg % 2u));
	*k = (4u * (lane % 4u)) + (16u * (reg / 2u)) + byte;
}
static void lm_mma8_b(uint32_t lane, uint32_t reg, uint32_t byte, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = (4u * (lane % 4u)) + (16u * reg) + byte;
}
static void lm_mma4_a(uint32_t lane, uint32_t reg, uint32_t nibble, uint32_t *m, uint32_t *k)
{
	*m = (lane / 4u) + (8u * (reg % 2u));
	*k = (8u * (lane % 4u)) + (32u * (reg / 2u)) + nibble;
}
static void lm_mma4_b(uint32_t lane, uint32_t reg, uint32_t nibble, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = (8u * (lane % 4u)) + (32u * reg) + nibble;
}
static int32_t verify_lm_mma_formulas(void)
{
	layout_mode_t a8t = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t a8v = { { 4u, 2u, 2u, 0u }, { 16u, 8u, 256u, 0u }, 3u };
	layout_mode_t b8t = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t b8v = { { 4u, 2u, 0u, 0u }, { 8u, 128u, 0u, 0u }, 2u };
	layout_mode_t a4t = { { 4u, 8u, 0u, 0u }, { 128u, 1u, 0u, 0u }, 2u };
	layout_mode_t a4v = { { 8u, 2u, 2u, 0u }, { 16u, 8u, 512u, 0u }, 3u };
	layout_mode_t b4t = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t b4v = { { 8u, 2u, 0u, 0u }, { 8u, 256u, 0u, 0u }, 2u };
	uint32_t lane,value,m,n,k,bad = 0;
	for (lane = 0; lane < VERIFY_LANES; ++lane)
	{
		for (value = 0; value < 16u; ++value)
		{
			lm_mma8_a(lane,value / 4u,value % 4u,&m,&k);
			if ( m + (16u * k) != cute_index(&a8t,&a8v,lane,value) )
				bad++;
		}
		for (value = 0; value < 8u; ++value)
		{
			lm_mma8_b(lane,value / 4u,value % 4u,&n,&k);
			if ( n + (8u * k) != cute_index(&b8t,&b8v,lane,value) )
				bad++;
		}
		for (value = 0; value < 32u; ++value)
		{
			lm_mma4_a(lane,value / 8u,value % 8u,&m,&k);
			if ( m + (16u * k) != cute_index(&a4t,&a4v,lane,value) )
				bad++;
		}
		for (value = 0; value < 16u; ++value)
		{
			lm_mma4_b(lane,value / 8u,value % 8u,&n,&k);
			if ( n + (8u * k) != cute_index(&b4t,&b4v,lane,value) )
				bad++;
		}
	}
	printf("  lm_mma.cuh operand formulas vs CUTLASS layouts: mismatches=%u/2304\n",bad);
	return(bad == 0 ? 0 : -1);
}


static void lm_mma16_a(uint32_t lane, uint32_t reg, uint32_t half, uint32_t *m, uint32_t *k)
{
	*m = (lane / 4u) + (8u * (reg % 2u));
	*k = (2u * (lane % 4u)) + half + (8u * (reg / 2u));
}
static void lm_mma16_b(uint32_t lane, uint32_t reg, uint32_t half, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = (2u * (lane % 4u)) + half + (8u * reg);
}
static int32_t verify_bf16_atom(void)
{
	layout_mode_t at = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t av = { { 2u, 2u, 2u, 0u }, { 16u, 8u, 128u, 0u }, 3u };
	layout_mode_t bt = { { 4u, 8u, 0u, 0u }, { 16u, 1u, 0u, 0u }, 2u };
	layout_mode_t bv = { { 2u, 2u, 0u, 0u }, { 8u, 64u, 0u, 0u }, 2u };
	uint8_t seen_a[256],seen_b[128];
	uint32_t lane,value,m,n,k,bad = 0,gap = 0;
	memset(seen_a,0,sizeof(seen_a));
	memset(seen_b,0,sizeof(seen_b));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
	{
		for (value = 0; value < 8u; ++value)
		{
			lm_mma16_a(lane,value / 2u,value % 2u,&m,&k);
			if ( m + (16u * k) != cute_index(&at,&av,lane,value) )
				bad++;
			seen_a[cute_index(&at,&av,lane,value) & 255u]++;
		}
		for (value = 0; value < 4u; ++value)
		{
			lm_mma16_b(lane,value / 2u,value % 2u,&n,&k);
			if ( n + (8u * k) != cute_index(&bt,&bv,lane,value) )
				bad++;
			seen_b[cute_index(&bt,&bv,lane,value) & 127u]++;
		}
	}
	for (value = 0; value < 256u; ++value)
		if ( seen_a[value] != 1 )
			gap++;
	for (value = 0; value < 128u; ++value)
		if ( seen_b[value] != 1 )
			gap++;
	printf("  BF16 m16n8k16 A/B: mismatches=%u/384  covered exactly once=%u/384\n",
		bad,384u - gap);
	for (lane = 0; lane < VERIFY_LANES; ++lane)
	{
		uint32_t reg,k0,k1,dummy;
		for (reg = 0; reg < 4u; ++reg)
		{
			lm_mma16_a(lane,reg,0u,&dummy,&k0);
			lm_mma16_a(lane,reg,1u,&dummy,&k1);
			if ( k1 != k0 + 1u )
				bad++;
		}
	}
	printf("  register halves adjacent in k: %s\n",bad == 0 ? "yes" : "NO");
	return((bad == 0 && gap == 0) ? 0 : -1);
}

#define IND_SOURCE_ROWS 40u
#define IND_ROW_PITCH 128u
#define IND_CHUNK_BYTES 16u
#define IND_K_TILES 3u
#define IND_SOURCE_PITCH (IND_ROW_PITCH * IND_K_TILES)
#define IND_MAX_TILE_M 64u
#define IND_MAX_PACKED (2u * IND_MAX_TILE_M)
#define IND_POISON 0xffffffffu

static uint8_t ind_gathered[IND_MAX_TILE_M * IND_ROW_PITCH];
static uint8_t ind_staged_box[IND_MAX_TILE_M * IND_ROW_PITCH];
static uint8_t ind_staged_ind[IND_MAX_TILE_M * IND_ROW_PITCH];
static uint8_t ind_writes[IND_MAX_TILE_M * IND_ROW_PITCH];
static uint32_t ind_row_index[IND_MAX_PACKED];

static uint8_t ind_tag(uint32_t source_row, uint32_t byte_in_row)
{
	return((uint8_t)(source_row * 61u + byte_in_row * 7u + 1u));
}

static uint32_t indirect_clamped_row(uint32_t row_base, uint32_t row_limit, uint32_t local_row)
{
	uint32_t packed = row_base + local_row;
	return(packed >= row_limit ? row_base : packed);
}

static int32_t indirect_stage_model(uint32_t tile_m, uint32_t row_base, uint32_t row_limit, uint32_t k_tile, int32_t apply_clamp)
{
	uint32_t r,c,x,packed,source_row,chunks,dst,errors = 0;
	chunks = IND_ROW_PITCH / IND_CHUNK_BYTES;
	memset(ind_staged_ind,0,tile_m * IND_ROW_PITCH);
	memset(ind_writes,0,tile_m * IND_ROW_PITCH);
	for (r = 0u; r < tile_m; ++r)
		for (c = 0u; c < chunks; ++c)
		{
			packed = row_base + r;
			if ( apply_clamp != 0 )
				packed = indirect_clamped_row(row_base,row_limit,r);
			source_row = ind_row_index[packed];
			if ( packed >= row_limit && packed != row_base )
				errors++;
			if ( source_row >= IND_SOURCE_ROWS )
				errors++;
			source_row %= IND_SOURCE_ROWS;
			dst = (r * IND_ROW_PITCH)
				+ (swizzle_chunk(c,r) * IND_CHUNK_BYTES);
			for (x = 0u; x < IND_CHUNK_BYTES; ++x)
			{
				ind_staged_ind[dst + x] =
					ind_tag(source_row,(k_tile * IND_ROW_PITCH) + (c * IND_CHUNK_BYTES) + x);
				ind_writes[dst + x]++;
			}
		}
	for (r = 0u; r < tile_m * IND_ROW_PITCH; ++r)
		if ( ind_writes[r] != 1u )
			errors++;
	return((int32_t)errors);
}

static int32_t verify_indirect_staging(void)
{
	uint32_t tile_m,valid,row_base,row_limit,k_tile,r,x,src_row;
	uint32_t tile_ms[3] = { 16u, 32u, 64u };
	int32_t errors,total = 0,unclamped_caught = 0,control_misses = 0;
	for (tile_m = 0u; tile_m < 3u; ++tile_m)
		for (valid = 1u; valid <= tile_ms[tile_m]; valid = valid * 2u + 1u)
			for (row_base = 0u; row_base <= 5u; row_base += 5u)
				for (k_tile = 0u; k_tile < IND_K_TILES; ++k_tile)
				{
					row_limit = row_base + valid;
					if ( row_limit > IND_MAX_PACKED )
						continue;
					for (x = 0u; x < IND_MAX_PACKED; ++x)
						ind_row_index[x] = x < row_limit
							? ((x * 17u + 5u) % IND_SOURCE_ROWS)
							: IND_POISON;
					memset(ind_gathered,0,sizeof(ind_gathered));
					memset(ind_staged_box,0,sizeof(ind_staged_box));
					for (r = 0u; r < valid; ++r)
						for (x = 0u; x < IND_ROW_PITCH; ++x)
						{
							src_row = ind_row_index[row_base + r];
							ind_gathered[(r * IND_ROW_PITCH) + x] =
								ind_tag(src_row,(k_tile * IND_ROW_PITCH) + x);
						}
					for (r = 0u; r < valid; ++r)
						for (x = 0u; x < IND_ROW_PITCH; ++x)
							ind_staged_box[(r * IND_ROW_PITCH)
								+ (swizzle_chunk(x / IND_CHUNK_BYTES,r) * IND_CHUNK_BYTES)
								+ (x % IND_CHUNK_BYTES)] =
								ind_gathered[(r * IND_ROW_PITCH) + x];
					errors = indirect_stage_model(tile_ms[tile_m],row_base,row_limit,k_tile,1);
					if ( errors != 0 )
						printf("  tile_m=%u valid=%u base=%u k=%u: %d staging defects\n",
							tile_ms[tile_m],valid,row_base,k_tile,errors);
					total += errors;
					for (r = 0u; r < valid; ++r)
						for (x = 0u; x < IND_ROW_PITCH; ++x)
							if ( ind_staged_ind[(r * IND_ROW_PITCH) + x]
								!= ind_staged_box[(r * IND_ROW_PITCH) + x] )
								total++;
					for (r = valid; r < tile_ms[tile_m]; ++r)
						for (x = 0u; x < IND_ROW_PITCH; ++x)
							if ( ind_staged_ind[(r * IND_ROW_PITCH)
								+ (swizzle_chunk(x / IND_CHUNK_BYTES,r) * IND_CHUNK_BYTES)
								+ (x % IND_CHUNK_BYTES)]
								!= ind_tag(ind_row_index[row_base],(k_tile * IND_ROW_PITCH) + x) )
								total++;
					if ( valid < tile_ms[tile_m] )
					{
						if ( indirect_stage_model(tile_ms[tile_m],row_base,row_limit,k_tile,0) != 0 )
							unclamped_caught++;
						else
							control_misses++;
					}
				}
	printf("  indirect staging vs gather+box: tile_m {16,32,64} x ragged tails x k tiles, defects=%d\n",total);
	printf("  unclamped-tail poison control: %d ragged configurations caught\n",unclamped_caught);
	if ( unclamped_caught == 0 || control_misses != 0 )
	{
		printf("  POISON CONTROL FAILED (%d misses) - the clamp is verified by nothing\n",control_misses);
		total++;
	}
	return(total);
}

int32_t main(void)
{
	int32_t failures = 0;
	printf("MMA fragment mappings, checked against CUTLASS CuTe layouts\n");
	printf("target sm_121a: FP8 via SM120_16x8x32_TN, NVFP4 via SM120_16x8x64_TN_VS\n\n");
	failures += verify_operand_a() != 0;
	failures += verify_operand_b() != 0;
	failures += verify_accumulator_c() != 0;
	failures += verify_known_bad_c_is_rejected() != 0;
	printf("\nldmatrix.x4 shared-memory bank spread over a %u-byte row\n",VERIFY_TILE_K);
	printf("  without swizzle: worst bank holds %u of 32 lanes\n",count_bank_conflicts(0,0u));
	printf("  with 128B swizzle: worst bank holds %u of 32 lanes\n",count_bank_conflicts(1,0u));
	if ( count_bank_conflicts(1,0u) >= count_bank_conflicts(0,0u) )
	{
		printf("  SWIZZLE DOES NOT HELP - layout is wrong\n");
		failures++;
	}
	failures += verify_sm120_equals_sm89() != 0;
	printf("\nNVFP4 atom SM120::BLOCKSCALED::SM120_16x8x64_TN_VS\n");
	failures += verify_nvfp4_operands() != 0;
	failures += verify_nvfp4_scale_layouts() != 0;
	printf("\nrewrite: lm/ kernel library\n");
	failures += verify_lm_mma_formulas() != 0;
	failures += verify_bf16_atom() != 0;
	failures += verify_pipeline_matrix() != 0;
	failures += verify_pipeline_persistent_matrix() != 0;
	printf("\nindirect A staging (route row-indirection consumer contract)\n");
	failures += verify_indirect_staging() != 0;
	printf("\n%s (%d failing checks)\n",failures == 0 ? "PASS" : "FAIL",failures);
	return(failures == 0 ? 0 : 1);
}

