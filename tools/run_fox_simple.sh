#!/bin/sh
exec /home/spark2/llama-cpp-ref/build-cpu/bin/llama-simple \
  -m /home/spark2/hy4-full.gguf \
  -n 4 \
  "The quick brown fox"
