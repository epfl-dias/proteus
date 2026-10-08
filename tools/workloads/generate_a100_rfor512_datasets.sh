#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKLOADS="${WORKLOADS:-tpch stack}" CODECS=RFOR RFOR_TILES=512 \
  exec bash "$SCRIPT_DIR/generate_cb2t_datasets.sh"
