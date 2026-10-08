#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPECTED_REVISION=da23f70ea060f308bb9536ec8515a4ef1d07a4bd

: "${HEAVYDB_ROOT:?Set HEAVYDB_ROOT to a HeavyDB v9.0.0 checkout}"
HEAVYDB_ROOT="$(cd "$HEAVYDB_ROOT" && pwd)"
BUILD_DIR="${BUILD_DIR:-$HEAVYDB_ROOT/build-release}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-11.8}"
LLVM_DIR="${LLVM_DIR:-/usr/lib/llvm-14/cmake}"
CLANG_LIB="${CLANG_LIB:-/usr/lib/llvm-14/lib/libclang-cpp.so.14}"
EXTRA_PREFIX="${EXTRA_PREFIX:-}"
JOBS="${JOBS:-$(nproc)}"

for tool in git pixi mvn; do
  command -v "$tool" >/dev/null || { echo "Missing required command: $tool" >&2; exit 1; }
done

actual_revision="$(git -C "$HEAVYDB_ROOT" rev-parse HEAD)"
if [[ "$actual_revision" != "$EXPECTED_REVISION" ]]; then
  echo "Expected HeavyDB v9.0.0 at $EXPECTED_REVISION, found $actual_revision" >&2
  exit 1
fi

patch_file="$ARTIFACT_DIR/patches/heavydb-v9.0.0-benchmark.patch"
if git -C "$HEAVYDB_ROOT" apply --check "$patch_file"; then
  git -C "$HEAVYDB_ROOT" apply "$patch_file"
elif git -C "$HEAVYDB_ROOT" apply --reverse --check "$patch_file"; then
  echo "HeavyDB benchmark patch is already applied."
else
  echo "HeavyDB checkout has changes that conflict with $patch_file" >&2
  exit 1
fi

cp "$ARTIFACT_DIR/pixi.toml" "$ARTIFACT_DIR/pixi.lock" "$HEAVYDB_ROOT/"
pixi install --manifest-path "$HEAVYDB_ROOT/pixi.toml" --frozen

PIXI_ENV="${PIXI_ENV:-$HEAVYDB_ROOT/.pixi/envs/default}"
export PATH="$PIXI_ENV/bin:$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="${EXTRA_PREFIX:+$EXTRA_PREFIX/lib:}$PIXI_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$BUILD_DIR" "$TMPDIR"

prefix_path="$PIXI_ENV"
[[ -z "$EXTRA_PREFIX" ]] || prefix_path="$EXTRA_PREFIX:$prefix_path"

cmake_args=(
  -S "$HEAVYDB_ROOT"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER="$PIXI_ENV/bin/x86_64-conda-linux-gnu-gcc"
  -DCMAKE_CXX_COMPILER="$PIXI_ENV/bin/x86_64-conda-linux-gnu-g++"
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc"
  -DCUDACXX="$CUDA_ROOT/bin/nvcc"
  -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_COMPILER:-$PIXI_ENV/bin/x86_64-conda-linux-gnu-gcc}"
  -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler
  -DCMAKE_PREFIX_PATH="$prefix_path"
  -DLLVM_DIR="$LLVM_DIR"
  -DClang_DIR="${CLANG_DIR:-$ARTIFACT_DIR/cmake/clang}"
  -DCLANG_LIB="$CLANG_LIB"
  -DCPR_DIR="$PIXI_ENV/lib/cmake/cpr"
  -DBOOST_ROOT="$PIXI_ENV"
  -DBOOST_LIBRARYDIR="$PIXI_ENV/lib"
  -DENABLE_IMPORT_PARQUET=OFF
  -DENABLE_AWS_S3=OFF
  -DENABLE_CUDA=ON
  -DENABLE_WARNINGS_AS_ERRORS=OFF
  -Dllvm_clangpp_cmd="$SCRIPT_DIR/clangpp-system-clean.sh"
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)

if [[ -n "${EXTRA_CMAKE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($EXTRA_CMAKE_ARGS)
  cmake_args+=("${extra_args[@]}")
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --parallel "$JOBS" --target heavydb heavysql initheavy

printf 'HeavyDB runtime built in %s\n' "$BUILD_DIR"
