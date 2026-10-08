#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
BUILD=${BUILD:-$REPO_ROOT/cmake-build-release}
LLVM_ROOT=${LLVM_ROOT:?Set LLVM_ROOT to an LLVM 14 installation built with libc++}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-11.3}
NVCOMP_ROOT=${NVCOMP_ROOT:?Set NVCOMP_ROOT to the nvCOMP CUDA 11 package root}
GDS_ROOT=${GDS_ROOT:-$CUDA_ROOT}
CMAKE_BIN=${CMAKE_BIN:-cmake}
JOBS=${JOBS:-$(nproc)}

export CC=${CC:-$LLVM_ROOT/bin/clang}
export CXX=${CXX:-$LLVM_ROOT/bin/clang++}
export PATH="$LLVM_ROOT/bin:$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake_prefix_path=$NVCOMP_ROOT
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  cmake_prefix_path="$cmake_prefix_path;$CMAKE_PREFIX_PATH"
fi

"$CMAKE_BIN" -S "$REPO_ROOT" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX="$BUILD/opt" \
  -DLLVM_DIR="$LLVM_ROOT/lib/cmake/llvm" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
  -DCMAKE_PREFIX_PATH="$cmake_prefix_path" \
  -DGDS_ROOT="$GDS_ROOT" \
  -DPROTEUS_CB2T_ARTIFACT=ON

"$CMAKE_BIN" --build "$BUILD" -- -j"$JOBS"

printf 'Complete CB2T artifact built under %s\n' "$BUILD"
