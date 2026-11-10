#pragma once


#include <stdint.h>

#include "model-families/k3/source/config.h"
#include "model-families/k3/source/dspark.h"

#define K3_VERIFY_MAX_DRAFT 7u
#define K3_VERIFY_MAX_ROWS (K3_VERIFY_MAX_DRAFT + 1u)

#define K3_ADAPTIVE_DEPTH_WINDOW 8u
#define K3_ADAPTIVE_DEPTH_MINIMUM 4u

struct K3AdaptiveDepth
{
	uint32_t window;
	uint32_t minimum_depth;
	uint32_t ceiling_depth;
	uint32_t history[K3_ADAPTIVE_DEPTH_WINDOW];
	uint32_t observed;
	uint32_t cursor;
	uint32_t depth;
};

static void K3AdaptiveDepthInit(struct K3AdaptiveDepth *controller,
	uint32_t window, uint32_t minimum_depth, uint32_t ceiling_depth,
	uint32_t initial_depth)
{
	if ( controller == 0 )
		return;
	if ( window == 0u || window > K3_ADAPTIVE_DEPTH_WINDOW )
		window = K3_ADAPTIVE_DEPTH_WINDOW;
	if ( minimum_depth == 0u )
		minimum_depth = 1u;
	if ( ceiling_depth < minimum_depth )
		ceiling_depth = minimum_depth;
	if ( initial_depth < minimum_depth )
		initial_depth = minimum_depth;
	if ( initial_depth > ceiling_depth )
		initial_depth = ceiling_depth;
	controller->window = window;
	controller->minimum_depth = minimum_depth;
	controller->ceiling_depth = ceiling_depth;
	controller->observed = 0u;
	controller->cursor = 0u;
	controller->depth = initial_depth;
}

static uint32_t K3AdaptiveDepthDepth(const struct K3AdaptiveDepth *controller)
{
	if ( controller == 0 )
		return(K3_VERIFY_MAX_DRAFT);
	return(controller->depth);
}

static uint32_t K3AdaptiveDepthObserve(struct K3AdaptiveDepth *controller,
	uint32_t drafted, uint32_t accepted)
{
	uint32_t window,index,target;
	uint64_t total;
	float mean;
	if ( controller == 0 )
		return(K3_VERIFY_MAX_DRAFT);
	if ( drafted > controller->ceiling_depth )
		drafted = controller->ceiling_depth;
	if ( accepted > drafted )
		accepted = drafted;
	controller->history[controller->cursor] = accepted;
	controller->cursor = (controller->cursor + 1u) % controller->window;
	if ( controller->observed < controller->window )
		controller->observed++;
	window = controller->window;
	if ( controller->observed < window )
		window = controller->observed;
	total = 0u;
	for ( index = 0u; index < window; ++index )
		total += controller->history[index];
	mean = (float)total / (float)window;
	target = (uint32_t)(mean + 1.5f);
	if ( target < controller->minimum_depth )
		target = controller->minimum_depth;
	if ( target > controller->ceiling_depth )
		target = controller->ceiling_depth;
	if ( target < controller->depth )
		controller->depth = target;
	else if ( controller->depth < target )
		controller->depth = controller->depth + 1u;
	return(controller->depth);
}

static uint32_t K3VerifyResolve(const uint32_t *row_tokens, uint32_t begin,
	const uint32_t *draft_tokens, uint32_t draft_count, uint32_t *bonus_out)
{
	uint32_t accepted = 0u;
	if ( row_tokens == 0 || draft_tokens == 0 || bonus_out == 0 )
		return(0u);
	if ( draft_count > K3_VERIFY_MAX_DRAFT )
		draft_count = K3_VERIFY_MAX_DRAFT;
	while ( accepted < draft_count
		&& row_tokens[begin + accepted] == draft_tokens[accepted] )
		accepted++;
	*bonus_out = row_tokens[begin + accepted];
	return(accepted);
}
