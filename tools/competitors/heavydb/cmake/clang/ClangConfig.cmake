# Minimal local Clang package shim for this server build.
#
# The system ClangConfig.cmake references missing static component archives,
# but HeavyDB only needs Clang include definitions here and links through the
# top-level clang-cpp library selection.
set(Clang_FOUND TRUE)
set(CLANG_INCLUDE_DIRS "/usr/include")
set(CLANG_DEFINITIONS "")
