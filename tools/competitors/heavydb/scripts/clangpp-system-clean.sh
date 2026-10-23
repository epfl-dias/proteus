#!/usr/bin/env bash
unset LD_LIBRARY_PATH
exec "${CLANGXX:-$(command -v clang++)}" "$@"
