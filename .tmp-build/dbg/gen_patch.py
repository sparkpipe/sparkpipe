import subprocess
head = subprocess.run(['git','show','HEAD:modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu'],capture_output=True,text=True).stdout
assert len(head) > 100000, len(head)
new = open('.tmp-build/dbg/new/spark_glm52_resident_decode_stage_cuda_validation.cu').read()
import difflib
d = list(difflib.unified_diff(head.splitlines(True), new.splitlines(True), fromfile='a/modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu', tofile='b/modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu'))
open('.agents/coord/glm52_tp8_validator_route_owner_pairing.patch','w').writelines(d)
print('patch lines:', len(d))
