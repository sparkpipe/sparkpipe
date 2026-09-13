#!/usr/bin/env bash
cd /Users/mac/dsh.sparkpipe/modules/k3_resident_decode_stage/source
for f in *.c; do
  before=$(wc -l < "/tmp/k3-strip-backup/$f.orig")
  after=$(wc -l < "$f")
  echo "$f: $before -> $after"
done
