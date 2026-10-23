#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPECTED_REVISION=8d8353a31ec2b1a46a90981fd064d592e3e5afe2
EXPECTED_DUCKDB_REVISION=8a5851971fae891f292c2714d86046ee018e9737

: "${SIRIUS_ROOT:?Set SIRIUS_ROOT to the Sirius checkout}"
SIRIUS_ROOT="$(cd "$SIRIUS_ROOT" && pwd)"
SIRIUS_BUILD_DIR="${SIRIUS_BUILD_DIR:-${TMPDIR:-/tmp}/sirius-legacy-release}"
TMPDIR="${TMPDIR:-/tmp}"
JOBS="${JOBS:-$(nproc)}"

for tool in git pixi cmake; do
  command -v "$tool" >/dev/null || { echo "Missing required command: $tool" >&2; exit 1; }
done

actual_revision="$(git -C "$SIRIUS_ROOT" rev-parse HEAD)"
if [[ "$actual_revision" != "$EXPECTED_REVISION" ]]; then
  echo "Expected Sirius at $EXPECTED_REVISION, found $actual_revision" >&2
  exit 1
fi

patch_file="$ARTIFACT_DIR/patches/sirius-8d8353a3-legacy.patch"
if git -C "$SIRIUS_ROOT" apply --check "$patch_file"; then
  git -C "$SIRIUS_ROOT" apply "$patch_file"
elif git -C "$SIRIUS_ROOT" apply --reverse --check "$patch_file"; then
  echo "Sirius legacy benchmark patch is already applied."
else
  echo "Sirius checkout has changes that conflict with $patch_file" >&2
  exit 1
fi

git -C "$SIRIUS_ROOT" submodule sync --recursive
git -C "$SIRIUS_ROOT" submodule update --init --recursive
actual_duckdb_revision="$(git -C "$SIRIUS_ROOT/duckdb" rev-parse HEAD)"
if [[ "$actual_duckdb_revision" != "$EXPECTED_DUCKDB_REVISION" ]]; then
  echo "Expected DuckDB v1.5.2 at $EXPECTED_DUCKDB_REVISION, found $actual_duckdb_revision" >&2
  exit 1
fi
pixi install --manifest-path "$SIRIUS_ROOT/pixi.toml" -e cuda12 --frozen

mkdir -p "$SIRIUS_BUILD_DIR" "$TMPDIR"
pixi run --manifest-path "$SIRIUS_ROOT/pixi.toml" -e cuda12 cmake \
  -S "$SIRIUS_ROOT/duckdb" \
  -B "$SIRIUS_BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DDUCKDB_EXTENSION_CONFIGS="$SIRIUS_ROOT/extension_config.cmake" \
  -DEXTENSION_STATIC_BUILD=ON \
  -DEXPORT_DYNAMIC_SYMBOLS=ON \
  -DENABLE_LEGACY_SIRIUS=ON \
  -DSIRIUS_LEGACY_ONLY=ON \
  -DCMAKE_C_COMPILER_LAUNCHER= \
  -DCMAKE_CXX_COMPILER_LAUNCHER= \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=

pixi run --manifest-path "$SIRIUS_ROOT/pixi.toml" -e cuda12 cmake \
  --build "$SIRIUS_BUILD_DIR" --parallel "$JOBS" \
  --target duckdb duckdb_local_extension_repo

printf 'Sirius benchmark binary: %s/duckdb\n' "$SIRIUS_BUILD_DIR"
