#!/bin/sh
# install_keys_on_rtx.sh — run ON sparkf: append the 16 spark pubkeys to
# spec@rtx5090 authorized_keys (idempotent), verify count.
set -eu
ssh -o BatchMode=yes spec@10.10.250.2 'cat >> ~/.ssh/authorized_keys && sort -u ~/.ssh/authorized_keys > ~/.ssh/ak.new && mv ~/.ssh/ak.new ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && wc -l ~/.ssh/authorized_keys' < /tmp/all16.pub
