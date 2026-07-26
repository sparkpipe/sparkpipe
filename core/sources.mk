SPARKPIPE_CORE_SUPPORT_SOURCES := \
    src/spark_status.c \
    src/spark_filesystem.c \
    src/spark_json.c \
    src/spark_sha256.c

SPARKPIPE_CORE_COMPILER_SOURCES := \
    src/spark_model_description.c \
    src/spark_module_library.c \
    src/spark_driver_compiler.c

SPARKPIPE_CORE_RUNTIME_SOURCES := \
    src/spark_driver_loader.c \
    src/spark_orchestrator.c
