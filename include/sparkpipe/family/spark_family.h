#pragma once

#define SPARK_FAMILY_CAT_(prefix,family,suffix) prefix##family##suffix
#define SPARK_FAMILY_CAT(prefix,family,suffix) SPARK_FAMILY_CAT_(prefix,family,suffix)
#define SPARK_FAMILY(name) SPARK_FAMILY_CAT(Spark,SPARK_FAMILY_CAMEL,name)
#define SPARK_FAMILY_CONST(name) SPARK_FAMILY_CAT(SPARK_,SPARK_FAMILY_UPPER,_##name)
#define SPARK_FAMILY_STRING_(family) #family
#define SPARK_FAMILY_STRING(family) SPARK_FAMILY_STRING_(family)
