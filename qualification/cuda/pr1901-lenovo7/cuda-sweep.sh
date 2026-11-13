#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
cd ~/build-rp
deadline=$(date -d "22:08" +%s)
out=~/build-rp/cuda-sweep-pr1901.tsv
: > $out
for path in pr1901-ce30c377b-src/tests/*_cuda.py; do
    test=$(basename $path .py)
    [ $(date +%s) -ge $deadline ] && { echo "deadline reached before $test" >> $out; break; }
    for tree in main-fded9e032-src pr1901-ce30c377b-src; do
        [ -f $tree/tests/$test.py ] || { printf '%s\t%s\tabsent\t0\t-\n' $test $tree >> $out; continue; }
        start=$(date +%s)
        (cd $tree && timeout 90 python3 tests/$test.py > ../sweep.$tree.$test.out 2>&1)
        rc=$?
        printf '%s\t%s\t%s\t%s\t%s\n' $test $tree $rc $(( $(date +%s) - start )) "$(tail -1 sweep.$tree.$test.out | cut -c1-120 | tr '\t' ' ')" >> $out
    done
done
echo "sweep finished $(date +%T)" >> $out
