#!/usr/bin/env bash
set -u
SHA=${1:?sha}
HOST=${TEST_HOST:-sparke}
S=${RELEASE_WORK:-${TMPDIR:-/tmp}/sparkpipe-release}; mkdir -p "$S"
scp -q $S/src-$SHA.tar $HOST:build-rp/src-$SHA.tar || exit 1
/usr/local/bin/flock -s -w 10800 /Users/mac/sparkpipe-coord/locks/tp16.lock ssh -o BatchMode=yes $HOST "cd ~/build-rp && rm -rf test-$SHA && mkdir test-$SHA && tar -C test-$SHA -xf src-$SHA.tar && cd test-$SHA && export PATH=/usr/local/cuda/bin:\$PATH && nice -n 10 make -j8 test TEST_EXCLUDE=\"${TEST_EXCLUDE:-}\" > ../test-$SHA.out 2>&1; echo TEST-RC=\$?; grep -c '^PASS' ../test-$SHA.out; grep -E '^FAIL|FAILED|Error [0-9]' ../test-$SHA.out | head -20; tail -3 ../test-$SHA.out" < /dev/null
