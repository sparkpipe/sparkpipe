#!/bin/bash
set -u
cd "$(dirname "$0")/.."
status=0
for fam in ling gemma4 laguna; do
  for side in a b; do
    dir=qualification/t1_reference/$fam
    if [ ! -d ~/refs2_verify/$side/$fam ]; then
      echo "SKIP $fam/$side (absent)"
      continue
    fi
    python3 tools/t1_reference_compare.py verify-manifest \
      --fixture-dir ~/refs2_verify/$side/$fam || status=1
  done
  for fix in ~/refs2_verify/a/$fam/*.t1r; do
    name=$(basename "$fix")
    sha_a=$(shasum -a 256 "$fix" | cut -d' ' -f1)
    sha_b=$(shasum -a 256 ~/refs2_verify/b/$fam/"$name" | cut -d' ' -f1)
    if [ "$sha_a" = "$sha_b" ]; then
      echo "DETERMINISM PASS $fam/$name $sha_a"
    else
      echo "DETERMINISM FAIL $fam/$name"
      status=1
    fi
    python3 tools/t1_reference_compare.py compare \
      --reference "$fix" --candidate ~/refs2_verify/b/$fam/"$name" | tail -1
    neg=~/refs2_verify/neg_$fam.t1r
    first_array=$(python3 -c "
from t1_reference_common import read_fixture
import sys
sys.path.insert(0, 'tools')
meta, _ = read_fixture('$fix')
print([e['name'] for e in meta['arrays']
       if e['name'].endswith('_streams')][0])")
    python3 tools/t1_reference_compare.py corrupt-fixture \
      --source "$fix" --target "$neg" --array "$first_array" --offset 9 > /dev/null
    out=$(python3 tools/t1_reference_compare.py compare \
      --reference "$fix" --candidate "$neg" | tail -1)
    named=$(python3 tools/t1_reference_compare.py compare \
      --reference "$fix" --candidate "$neg" | grep -c "$first_array" || true)
    echo "NEGATIVE-CONTROL $fam/$name: $out (names array: $named)"
    [ "$named" -ge 1 ] || status=1
    echo "$out" | grep -q "FAIL" || status=1
    rm -f "$neg"
  done
done
exit $status
