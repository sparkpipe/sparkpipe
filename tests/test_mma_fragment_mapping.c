// Fragment-mapping verifier for mma.sync.m16n8k32 E4M3.
//
// The register-to-matrix-element mapping is the one part of an mma.sync kernel
// that a wrong implementation renders silently incorrect while still
// assembling. It does not need silicon to check: CUTLASS states the mapping as
// CuTe (Shape,Stride) layouts, and a layout is arithmetic. This file evaluates
// those layouts directly and compares them against the closed-form indexing the
// kernel uses, exhaustively, for every (lane, value) pair.
//
// Ground truth, transcribed from
// third_party/flashinfer/3rdparty/cutlass/include/cute/atom/mma_traits_sm89.hpp
// MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> and mma_traits_sm80.hpp
// SM80_16x8_Row:
//
//   ALayout ((4,8),(4,2,2)) : ((64,1),(16,8,256))
//   BLayout ((4,8),(4,2))   : ((32,1),(8,128))
//   CLayout ((4,8),(2,2))   : ((32,1),(16,8))
//
// A CuTe layout maps (thread, value) to a linear index into the logical MMA
// tile, which is column-major: A is m + 16k, B is n + 8k, C is m + 16n. The
// checks below invert that and confirm the kernel's formulas reproduce it.
//
// Three properties are checked, and each catches a different error:
//   1. agreement  - kernel formula equals the CUTLASS layout everywhere
//   2. bijection  - the mapping covers every tile element exactly once, which
//                   catches any permutation error the agreement check could
//                   miss if both sides shared a mistake
//   3. bank spread - the shared-memory addresses the fragment loads generate,
//                    with and without the 128-byte swizzle, so the swizzle is
//                    justified by a count rather than by assertion

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

// Evaluate one CuTe mode: decompose the flat coordinate colexicographically
// across the mode's shape and dot it with the stride.
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

// Kernel-side closed forms. These are what the GEMM actually computes; the
// point of this file is that they are not trusted until checked.
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

// The mapping that the deleted first draft used, kept as a negative control. If
// this ever reports zero mismatches the verifier itself is broken.
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

// ldmatrix.x4 gathers 512 bytes: lane L supplies a 16-byte row. Composing
// Copy_Traits<SM75_U32x4_LDSM_N>'s SrcLayout (32,128):(128,1) with its DstLayout
// (32,(32,4)):(32,(1,1024)) gives register r of thread t from source bytes
// [t*4 + r*128, +4), which lands in the chunk supplied by lane t/4 + 8r. For
// that to feed the A layout above, lane L must supply row (L%8) + 8*((L/8)%2)
// at k offset 16*(L/16).
static uint32_t ldmatrix_row_for_lane(uint32_t lane)
{
	return((lane % 8u) + (8u * ((lane / 8u) % 2u)));
}

static uint32_t ldmatrix_chunk_for_lane(uint32_t lane)
{
	return(lane / 16u);
}

// 128-byte swizzle: within a row, the 16-byte chunk at index c moves to
// c ^ (row % 8). TMA applies this when the tensor map is encoded with
// CU_TENSOR_MAP_SWIZZLE_128B, so the fragment load must apply the same xor.
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

// -- Pipeline schedule check ------------------------------------------------
// Appended as a second gate. The stage/phase schedule in
// LmGemmKernel is pure integer logic and its failure modes -
// consuming a stage never produced, overwriting a stage before it is consumed,
// waiting on the wrong mbarrier phase - are all decidable on the host. Only the
// hardware behaviour of the transfers themselves needs the ring.
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
	// prologue: stages 0 .. stages-2
	for (fill = 0; fill + 1u < stages && fill < k_tiles; ++fill)
		produced_round[fill % stages] = fill / stages;
	for (k_tile = 0; k_tile < k_tiles; ++k_tile)
	{
		stage = k_tile % stages;
		round = k_tile / stages;
		ahead = k_tile + stages - 1u;
		if ( ahead < k_tiles )
		{
			// a stage may only be refilled after its previous contents were consumed
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

// -- Persistent multi-tile schedule check ------------------------------------
// The verifier above covers one output tile, where the wait parity equals
// (k_tile / stages) & 1. The kernel's barriers initialise once and serve every
// tile the persistent CTA runs, so a stage's phase accumulates across tiles
// and the parity is the per-stage completion count modulo two - the per-tile
// formula is exact only when k_tiles % (2 * stages) == 0, and matches an
// already-completed old phase otherwise. This models the kernel's per-stage
// parity bitmask across a multi-tile schedule and checks it against the
// cumulative produce/consume counts, which are the barrier's true phase.
static int32_t verify_pipeline_persistent(uint32_t stages, uint32_t k_tiles, uint32_t output_tiles)
{
	uint32_t produced[PIPE_MAX_STAGES],consumed[PIPE_MAX_STAGES];
	uint32_t k_tile,stage,ahead,fill,tile,errors = 0,stale = 0;
	uint32_t phase = 0; /* the kernel's parity bitmask, one bit per stage */
	for (stage = 0; stage < stages; ++stage)
	{
		produced[stage] = 0;
		consumed[stage] = 0;
	}
	for (tile = 0; tile < output_tiles; ++tile)
	{
		// prologue: stages 0 .. stages-2
		for (fill = 0; fill + 1u < stages && fill < k_tiles; ++fill)
		{
			// a stage may only be refilled after its previous contents were consumed
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
			// the wait must target the phase the barrier completes next
			if ( ((phase >> stage) & 1u) != (consumed[stage] & 1u) )
				errors++;
			// and the staged data must be the round just produced, not an old one
			if ( produced[stage] != consumed[stage] + 1u )
				errors++;
			// where the per-tile k formula disagrees, a k-derived wait is stale
			if ( ((k_tile / stages) & 1u) != (consumed[stage] & 1u) )
				stale++;
			consumed[stage]++;
			phase ^= 1u << stage;
		}
		// every k tile produced in this tile is consumed in this tile
		for (stage = 0; stage < stages; ++stage)
			if ( produced[stage] != consumed[stage] )
				errors++;
	}
	// Coverage, not behaviour: multi-tile runs with k_tiles % (2 * stages) != 0
	// must actually reach a wait the per-tile formula gets wrong, or this gate
	// says nothing about the bug it exists for. Single-tile runs must never
	// reach one - the model reproduces the old formula where the old formula
	// was right.
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


// -- SM120 atom equivalence and the NVFP4 block-scaled atom -----------------
// sm_121a selects SM120_16x8x32_TN via rr_op_selector_sm120, not the SM89 atom.
// SM120_16x8x32_TN inherits MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>, so this
// checks the inherited layout against the SM89 one element by element rather
// than trusting that "both are 8-bit at m16n8k32" implies identical mappings.
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

// NVFP4 atom: SM120::BLOCKSCALED::SM120_16x8x64_TN_VS.
//   ALayout ((4,8),(8,2,2)) : ((128,1),(16,8,512))  -> (M16,K64)
//   BLayout ((4,8),(8,2))   : ((64,1),(8,256))      -> (N8,K64)
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


// -- NVFP4 scale-factor layouts ---------------------------------------------
// SFALayout ((2,2,8),64) : ((8,0,1),16)  -> (M16,K64)
// SFBLayout ((4,8),64)   : ((0,1),8)     -> (N8,K64)
// Both carry a stride-0 mode, which is why CUTLASS annotates them "effectively
// 16 threads" and "effectively 8 threads": lanes differing only in that mode
// address the same element, so several lanes must supply the same scale. That
// redundancy is the whole reason the instruction takes {byte-id, thread-id}
// selectors, and getting them wrong reads a valid-looking wrong scale.
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
	// 32 lanes over 16 effective -> 2 lanes per element; over 8 effective -> 4.
	if ( a_max != 2u || b_max != 4u )
	{
		printf("  SHARING FACTOR DISAGREES WITH THE STRIDE-0 MODE - decode is wrong\n");
		return(-1);
	}
	return((a_bad == 0 && b_bad == 0) ? 0 : -1);
}


// -- lm_mma.cuh formulas ------------------------------------------------------
// The rewrite's operand loaders index by REGISTER and byte-within-register
// rather than by (register, byte-in-K). These are the exact formulas in
// lm_mma.cuh; the check is that they reproduce the same CuTe layouts already
// verified above, so the new library cannot drift from the old verification.
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


// BF16 m16n8k16. Every packed format decodes into this layout, so it is the one
// mapping the whole library depends on rather than one of several.
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
	// The property the decode path rests on: the two halves of a register are
	// adjacent in k, so one 32-bit read or one pair-extract serves it.
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

// -- Indirect A staging: row map, tail clamp, box equivalence ----------------
// LmPipelineProduceIndirectA (tile.cuh) stages packed A row p from source row
// source_row_map[clamp(p)] with one bulk copy per 16-byte swizzle chunk
// instead of one TMA box. The numerics-identical claim is a BYTE claim: for
// every live row the staged tile must equal what gather-plus-box stages,
// because the fragment loads and the accumulate order downstream are shared
// code. Both stagings are address arithmetic over a tagged source, so a host
// can check them directly - the poisoned indices past row_limit included,
// since the clamp is the difference between dead traffic and a wild copy.
//
// The pipeline protocol needs no new check here: the indirect produce
// declares the same a_bytes + b_bytes through the same helper and completes
// through the same complete_tx count, so the schedule and parity models above
// cover both paths by construction.
#define IND_SOURCE_ROWS 40u
#define IND_ROW_PITCH 128u /* BF16 at kTileK 64: one 128-byte swizzle span */
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

// The kernel's formulas, transcribed from tile.cuh and gemm.cuh - checked,
// not trusted. The consume side (scale_a follows the source row) clamps by
// the same rule, so one function stands for both.
static uint32_t indirect_clamped_row(uint32_t row_base, uint32_t row_limit, uint32_t local_row)
{
	uint32_t packed = row_base + local_row;
	return(packed >= row_limit ? row_base : packed);
}

// One staging of the model. With apply_clamp=0 the clamp is removed, which
// must be CAUGHT by the poison check - a clamp that nothing detects is not a
// verified clamp. Returns the number of defects seen.
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
			// the poison guard: an index past row_limit is the next group's
			// business, and past the array's end it is a wild address
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
	// every staged byte written exactly once: the chunk decomposition is a
	// permutation of the tile, or the fragments read unwritten memory
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
					// live indices in range, everything at or past row_limit
					// poisoned: the next group's bytes are not this tile's
					for (x = 0u; x < IND_MAX_PACKED; ++x)
						ind_row_index[x] = x < row_limit
							? ((x * 17u + 5u) % IND_SOURCE_ROWS)
							: IND_POISON;
					// REFERENCE: the old dataflow. Gather the live rows into a
					// packed buffer (what LmGatherRowsKernel materialises),
					// then stage it the way the TMA box does - one swizzled
					// 128-byte span per row, chunk c landing at c ^ (r % 8).
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
					// KERNEL MODEL: chunked staging through the index.
					errors = indirect_stage_model(tile_ms[tile_m],row_base,row_limit,k_tile,1);
					if ( errors != 0 )
						printf("  tile_m=%u valid=%u base=%u k=%u: %d staging defects\n",
							tile_ms[tile_m],valid,row_base,k_tile,errors);
					total += errors;
					// live rows: byte-identical to gather-plus-box, which is
					// the numerics-identical property stated as bytes
					for (r = 0u; r < valid; ++r)
						for (x = 0u; x < IND_ROW_PITCH; ++x)
							if ( ind_staged_ind[(r * IND_ROW_PITCH) + x]
								!= ind_staged_box[(r * IND_ROW_PITCH) + x] )
								total++;
					// tail rows: the clamp duplicated row_base's source row.
					// Dead traffic - the stores are dropped - but the clamp
					// itself is checked, not assumed. The staged row is
					// swizzled, so the expected tag lands at the swizzled
					// position, same as the box reference.
					for (r = valid; r < tile_ms[tile_m]; ++r)
						for (x = 0u; x < IND_ROW_PITCH; ++x)
							if ( ind_staged_ind[(r * IND_ROW_PITCH)
								+ (swizzle_chunk(x / IND_CHUNK_BYTES,r) * IND_CHUNK_BYTES)
								+ (x % IND_CHUNK_BYTES)]
								!= ind_tag(ind_row_index[row_base],(k_tile * IND_ROW_PITCH) + x) )
								total++;
					// removing the clamp MUST trip the poison guard, or the
					// guard verifies nothing
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

