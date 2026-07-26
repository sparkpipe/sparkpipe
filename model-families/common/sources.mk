SPARKPIPE_MODEL_COMMON_SOURCES := \
    model-families/common/src/spark_hidden_transport.c \
    model-families/common/src/spark_memlink.c \
    model-families/common/src/spark_kv_store.c \
    model-families/common/src/spark_tp_collective.c \
    model-families/common/src/spark_tokenizer.c \
    model-families/common/src/spark_stage_kv_client.c

SPARKPIPE_MODEL_COMMON_CUDA_STAGE_SUPPORT_SOURCE := \
    model-families/common/src/spark_stage_module_common.c
