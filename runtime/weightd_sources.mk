# Shared weight daemon/client sources for runtime and model module linkage.
SPARKPIPE_WEIGHTD_SOURCES := \
	runtime/spark_weightd.c \
	runtime/spark_weightd_manifest.c \
	runtime/spark_weightd_lease.c \
	runtime/spark_weightd_attach.c \
	runtime/spark_weightd_map.c \
	runtime/spark_weightd_spine.c \
	runtime/spark_weightd_worker.c
