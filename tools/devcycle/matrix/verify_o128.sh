#!/usr/bin/env bash
h=$(jq -r '[.events[] | select(.event.event == "token") | .event.token_id] | join(",") + "\n"' /tmp/dsv4-matrix-receipts/restore-final-o128.json | shasum -a 256 | cut -d' ' -f1)
echo "hash=$h"
echo "pinned=a9385d0b296ca083e577e715d2f6335067691dce0e0dd5ab1394a102a3d3631f"
cp /tmp/dsv4-matrix-receipts/restore-final-o128.json /Users/mac/dsh.sparkpipe/qualification/dsv4/performance/tp4_b1_matrix_20260825/receipts/
echo archived
