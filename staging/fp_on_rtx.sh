#!/bin/sh
# fp.sh — run ON sparkf: list authorized_keys fingerprints on rtx5090.
set -eu
ssh -o BatchMode=yes spec@10.10.250.2 'ssh-keygen -lf ~/.ssh/authorized_keys' | awk '{print $2}'
