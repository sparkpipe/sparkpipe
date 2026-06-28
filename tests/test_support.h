#ifndef SPARKPIPE_TEST_SUPPORT_H
#define SPARKPIPE_TEST_SUPPORT_H

#include <stdint.h>

#include "sparkpipe/spark_driver_compiler.h"
#include "sparkpipe/spark_module_library.h"

uint64_t SparkTestReadCounter(const char *path);
SparkStatus SparkTestPublishAddOneModule(const char *library_root, const char *target, const char *counter_path, SparkModulePublishReport *report);
SparkStatus SparkTestPublishAddTwoAsAddOneModule(const char *library_root, const char *target, const char *counter_path, SparkModulePublishReport *report);
SparkStatus SparkTestPublishDoubleModule(const char *library_root, const char *target, const char *counter_path, SparkModulePublishReport *report);
SparkStatus SparkTestPublishAffineArchiveModule(const char *library_root, const char *target, const char *counter_path, SparkModulePublishReport *report);
SparkStatus SparkTestCompileDemoStage(const char *library_root, const char *stage_name, const char *output_directory, SparkDriverCompileReport *report);
SparkStatus SparkTestCompileDemoPackage(const char *library_root, const char *output_directory, SparkModelPackageCompileReport *report);
SparkStatus SparkTestCompileLinkUnitDemoStage(const char *library_root, const char *output_directory, SparkDriverCompileReport *report);

#endif
