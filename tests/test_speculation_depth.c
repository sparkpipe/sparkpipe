#include <stdio.h>
#include <stdlib.h>

#include "sparkpipe/spark_speculation_depth.h"

static void Require(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	exit(1);
}

int main(void)
{
	uint32_t cap,proposed,accepted,max,next,round,cost,base;
	Require(SparkSpeculationDepthCapNext(3u,3u,3u,0u) == 0u,"no speculation keeps the cap at zero");
	Require(SparkSpeculationDepthCapNext(7u,7u,0u,7u) == 1u,"a rejection at depth one drops the cap to one");
	Require(SparkSpeculationDepthCapNext(7u,7u,3u,7u) == 4u,"a partial accept caps at the accepted length plus one");
	Require(SparkSpeculationDepthCapNext(1u,1u,1u,7u) == 2u && SparkSpeculationDepthCapNext(2u,2u,2u,7u) == 4u && SparkSpeculationDepthCapNext(4u,4u,4u,7u) == 7u,"full accepts double the cap up to the maximum");
	Require(SparkSpeculationDepthCapNext(7u,3u,3u,7u) == 7u,"a short full accept keeps a large cap");
	Require(SparkSpeculationDepthCapNext(0u,0u,0u,7u) == 1u && SparkSpeculationDepthCapNext(9u,9u,9u,7u) == 7u,"out-of-range caps are clamped to the maximum before the update");
	for (max=1u; max<=31u; max++)
		for (cap=0u; cap<=max + 1u; cap++)
			for (proposed=0u; proposed<=max; proposed++)
				for (accepted=0u; accepted<=proposed; accepted++)
				{
					next = SparkSpeculationDepthCapNext(cap,proposed,accepted,max);
					Require(next >= 1u && next <= max,"the cap stays in 1..max");
					Require(accepted == proposed && proposed != 0u ? next >= (cap == 0u || cap > max ? max : cap) : next == (accepted + 1u > max ? max : accepted + 1u),"growth on full accepts, accepted plus one otherwise");
				}
	cap = 7u;
	cost = base = 0u;
	for (round=0u; round<100u; round++)
	{
		accepted = round % 10u == 9u ? 7u : 0u;
		proposed = cap;
		accepted = accepted < proposed ? accepted : proposed;
		cost += proposed;
		base += 7u;
		cap = SparkSpeculationDepthCapNext(cap,proposed,accepted,7u);
	}
	Require(cost * 3u < base,"a mostly rejecting drafter proposes far fewer rows than a fixed depth");
	puts("PASS the speculation depth cap doubles on full accepts, falls to accepted+1 on a rejection and stays within 1..max");
	return(0);
}
