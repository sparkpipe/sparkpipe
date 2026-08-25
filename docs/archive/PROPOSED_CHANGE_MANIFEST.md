# Proposed Change Manifest

This is a hash-based inventory, not a textual diff. Generated build products,
transient audit logs, object files, archives, Git metadata, and Python bytecode are excluded.
Retained validation receipts under `docs/` are included.

## Summary

- Added source/documentation files: **2**
- Modified source/documentation files: **29**
- Deleted source/documentation files: **0**

## Added

| Path | Bytes | SHA-256 |
|---|---:|---|
| `docs/validation-logs/host_test_suite_without_loopback.log` | 9772 | `848c7bbd9539feeb8978400e971fa11001629f1b524a13b80f305c066f36b8d6` |
| `tests/test_cuda_performance_contracts.py` | 63600 | `89d37ffff94662ee1b7b9e958e21303eb640d0c77f0e890e9ec4e0ec800c660d` |

## Modified

| Path | Bytes | SHA-256 |
|---|---:|---|
| `Makefile` | 58341 | `1b4c0db51b46fef0562488514e7ee6d854c2d3dda08fe9ef6c5da87df860336c` |
| `docs/SPARK_HOST_RDMA_DOORBELL.md` | 6180 | `3899c88a6054222ca7e4fd1148687766cc489084890e7820f4f1d978d07a1a6a` |
| `docs/VALIDATION_STATUS.json` | 4795 | `d58c37ff29ac6284e0be48641c758898c8a5de9feb3ecaec14d20acc6d74a071` |
| `docs/VALIDATION_STATUS.md` | 3915 | `418ea9243b5bee8ecc33d6e266f04e511e2b320ef5127184963111e9018c71f7` |
| `docs/validation-logs/host_test_suite.log` | 24052 | `d8261c5e88b34860fd30cd1b93638bac1ade2d6be28057ffbece049047306243` |
| `docs/validation-logs/source_tree_stability.log` | 178 | `c6a9b743611729e9e8859a5b768bfe5715d50d68616352d2730cc93a394fe2d8` |
| `model-families/common/include/sparkpipe/spark_hidden_transport.h` | 14503 | `072d254c33e3bf59f87acbdbb02e6af7d08411fda71ea23319565cee172e45b8` |
| `model-families/common/include/sparkpipe/spark_lm_fp8_tile.cuh` | 23061 | `9484547053f71a3ac2dbe7fe8ef788307252c58266e3ec46c286ac0cf22cfff7` |
| `model-families/common/include/sparkpipe/spark_lm_kernels.cuh` | 97821 | `55217cb4e4b79ec2167e39c7f3b2a6a72f28dd6e1724c1b073036086b4457a9d` |
| `model-families/common/src/spark_hidden_transport.c` | 36945 | `5024f9eb4b3416bc8e9b52ec3ca3a16d85d1c2ee3e56c7fb85c3c0174fc241af` |
| `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu` | 68543 | `ce2e61b9f248f876cbcef0761ec4d77681ce810eefc3455cf6db50707645f9c5` |
| `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c` | 105605 | `9e15a0bff545c37ca1ca540382a981477dcd76e39f10aeaf6e9f4356d624f50e` |
| `modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu` | 419942 | `3cb9a0b31a7ffed1b5d90f353e10c90836f36d29b1d67c94b0825f38152a061e` |
| `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_cuda.cu` | 14903 | `3d4bd00a42d1838f9d7530bfa67a0b3252c30d946ab92aeba528456d160b467e` |
| `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_linear_plan.cu` | 54009 | `2ce9f0c77a760006c1ad10a86b3ff9f02c4f9e3324ed8b3b29b2271a8d3132e4` |
| `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c` | 180968 | `45a2a883f2007ccdb889127721cf86bcaa7fd12dde592ee161a2456c81ac116e` |
| `modules/glm52_resident_decode_stage/source/spark_glm52_sm121_required_decode_stage.cu` | 1036525 | `b71bac9e2f1db1d4168a504275eddd46f8199e4339979330baebe165b4df0fae` |
| `modules/hidden_transport_spark_host_rdma_verbs.cu` | 132033 | `bf41527976d2aad224b51cd53e5de14f5329c6a0dbe468974af0cbdd99aedb6a` |
| `modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_resident_decode_stage_firmware.h` | 19411 | `e6608987fb65404b0809cc0bd597ca1ed1ae07721e74d6cd9d9801f50dc24e28` |
| `modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu` | 92526 | `ba037e54fa82662d351c742233679032313abbef9936be100abd26092742fb58` |
| `modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c` | 103110 | `d3f8c8b10988acefbd60ce249533fe2c81b3ef58f5292be630a4ebe051656e1d` |
| `modules/mimo25_resident_decode_stage/source/spark_mimo25_resident_decode_stage_cuda.cu` | 24569 | `1b6a34656a5a4fcc06a515d9a1bdcb520d160a0159381356fdad4c23cc71fd9b` |
| `modules/mimo25_resident_decode_stage/source/spark_mimo25_resident_decode_stage_module.c` | 90980 | `16c9f4101e6981ce0d7b680325e90be3549ab3d7dd241f384b4a841d09c27c93` |
| `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_cuda.cu` | 62243 | `bce272fd27c2ed85780c1411c6c4c2096a8511c8ed70ef4a25f0c5d0ff0c808e` |
| `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c` | 107819 | `71b4d3fcc25ba82fe933bd72aba88288adf495b3ec4e9c2e892f499f5ea51b60` |
| `tests/fixtures/glm52_resident_decode_stage_fake_backend.c` | 16244 | `74a7532459edf90f565b71064d02b6db37a25dbe26037cc776209937acf29d72` |
| `tests/test_glm52_exact_pp13_prefill_hidden.py` | 76596 | `288b7a9007c5ef29bf5eb39073401ea92056e7d095d95ad5a53dcc19d3ec785a` |
| `tests/test_glm52_quantized_cuda_contract.py` | 10246 | `31062815fd8968a9f010d473588723d2796128f26ffa9ff2bf5f371b432b5827` |
| `tests/test_glm52_resident_decode_stage_firmware.c` | 191228 | `16c2688c9bee10e9e1efc2d48fe5151964407b5151190d9fb5c7c07ea612ef50` |

## Deleted from Proposed Tree

None.
