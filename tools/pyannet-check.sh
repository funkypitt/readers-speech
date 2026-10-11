#!/bin/sh
# Builds pyannet.c for the host with gcc and compares it with onnxruntime on the test fixture
# (and on any wav given as argument: a few ten-second windows of each).
#     tools/pyannet-check.sh [more.wav ...]
# Needs the interview env (onnxruntime, onnx): ~/miniconda3/envs/interview/bin/python.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/interview/bin/python}
# the ONNX export the weights come from (not in the repository any more): sherpa-onnx's
# https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
ONNX=${ONNX:?set ONNX to the path of sherpa-onnx-pyannote-segmentation-3-0/model.onnx}
BIN=src/main/assets/pyannote-segmentation-3.0.bin
OUT=${OUT:-build/pyannet-check}
CC=${CC:-gcc}
mkdir -p "$OUT"
# gnu99 rather than c99: in strict mode gcc stops fusing multiply and add, which clang (the NDK) always does
$CC -std=gnu99 -O3 -march=native -Wall -Wextra -o "$OUT/pyannet-check" tools/pyannet-check.c src/main/cpp/pyannet.c -lm
[ -f "$BIN" ] || "$PY" tools/export-pyannet.py "$ONNX" "$BIN"
"$PY" tools/pyannet-check.py "$ONNX" "$BIN" "$OUT/pyannet-check" src/test/resources/fixture.wav "$@"
