#!/usr/bin/env python3
"""A replayed glm5_next decode graph makes the same context-dependent attention choices as an eager wave at the same context, and draws tokens with the same head (greedy or sampled), whatever the capture history."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <stdio.h>
#include <stdlib.h>
#include "sparkpipe/spark_glm5_next_graph_regime.h"

static uint32_t state = 2463534242u;

static uint32_t next_random(uint32_t bound)
{
	state ^= state << 13u;
	state ^= state >> 17u;
	state ^= state << 5u;
	return(state % bound);
}

static void check_replay(uint32_t context,uint32_t bound,uint32_t threshold,uint32_t max_positions)
{
	if ( context > bound || bound > max_positions || SparkGlm5NextAttentionSplit(bound,threshold) != SparkGlm5NextAttentionSplit(context,threshold) || (bound > SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K) != (context > SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K) )
	{
		fprintf(stderr,"FAIL threshold=%u max=%u: a graph bounded at %u replays context %u with different attention choices\n",threshold,max_positions,bound,context);
		exit(1);
	}
}

static void run_wave(uint32_t *bounds,uint32_t *heads,uint32_t context,uint32_t sampled,uint32_t threshold,uint32_t max_positions)
{
	uint32_t key;
	key = SparkGlm5NextGraphKey(context,threshold,sampled);
	if ( key >= SPARK_GLM5_NEXT_GRAPH_KEY_COUNT || key % SPARK_GLM5_NEXT_GRAPH_REGIME_COUNT != SparkGlm5NextGraphRegime(context,threshold) )
	{
		fprintf(stderr,"FAIL threshold=%u: context %u sampled %u maps to graph key %u\n",threshold,context,sampled,key);
		exit(1);
	}
	if ( bounds[key] != 0u && context > bounds[key] )
		bounds[key] = 0u;
	if ( bounds[key] == 0u )
	{
		bounds[key] = SparkGlm5NextGraphBound(context,threshold,max_positions);
		heads[key] = sampled;
	}
	if ( heads[key] != sampled )
	{
		fprintf(stderr,"FAIL threshold=%u: a %s wave at context %u replays a graph captured for a %s head\n",threshold,sampled != 0u ? "sampled" : "greedy",context,heads[key] != 0u ? "sampled" : "greedy");
		exit(1);
	}
	check_replay(context,bounds[key],threshold,max_positions);
}

static void check_every_context(uint32_t threshold,uint32_t max_positions)
{
	uint32_t bounds[SPARK_GLM5_NEXT_GRAPH_KEY_COUNT] = {0u},heads[SPARK_GLM5_NEXT_GRAPH_KEY_COUNT] = {0u},context;
	for (context=1u; context<=max_positions; context++)
	{
		run_wave(bounds,heads,context,0u,threshold,max_positions);
		run_wave(bounds,heads,context,1u,threshold,max_positions);
	}
}

static void check_histories(uint32_t threshold,uint32_t max_positions)
{
	uint32_t bounds[SPARK_GLM5_NEXT_GRAPH_KEY_COUNT] = {0u},heads[SPARK_GLM5_NEXT_GRAPH_KEY_COUNT] = {0u},request,prompt,length,position,sampled;
	for (request=0u; request<400u; request++)
	{
		prompt = 1u + next_random(next_random(4u) == 0u ? max_positions : 600u);
		length = prompt + next_random(64u);
		sampled = next_random(2u);
		for (position=prompt; position<=length && position<=max_positions; position++)
			run_wave(bounds,heads,position,sampled,threshold,max_positions);
	}
}

int main(void)
{
	static const uint32_t thresholds[] = {0u,1u,2u,64u,1000u,2048u,2049u,4096u};
	static const uint32_t max_positions[] = {4096u,32768u};
	uint32_t t,m;
	for (m=0u; m<2u; m++)
		for (t=0u; t<sizeof(thresholds)/sizeof(thresholds[0]); t++)
		{
			if ( thresholds[t] > max_positions[m] )
				continue;
			check_every_context(thresholds[t],max_positions[m]);
			check_histories(thresholds[t],max_positions[m]);
		}
	return(0);
}
'''


def main():
    with tempfile.TemporaryDirectory() as directory:
        source, binary = Path(directory) / "regime.c", Path(directory) / "regime"
        source.write_text(HARNESS)
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Werror", "-I.", "-Iinclude",
                        "-Imodel-families/glm5_next/include", str(source), "-o", str(binary)], cwd=ROOT, check=True)
        subprocess.run([str(binary)], check=True, timeout=600)
    print("PASS replayed decode graphs choose split-KV, DSA selection and the greedy or sampled head exactly as an eager wave at the same context")


if __name__ == "__main__":
    main()
