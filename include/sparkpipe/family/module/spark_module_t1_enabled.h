#pragma once

static int SPARK_FAMILY(T1Enabled)(void)
{
	static int t1_enabled = -1;
	if ( t1_enabled < 0 )
		t1_enabled = getenv("SPARK_" SPARK_FAMILY_STRING(SPARK_FAMILY_UPPER) "_T1") != 0 ? 1 : 0;
	return(t1_enabled);
}
