find_path("/usr/bin" "llvm-config")

set(LLVM_FOUND TRUE)
execute_process(COMMAND llvm-config --cppflags OUTPUT_VARIABLE LLVM_CPPFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND llvm-config --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND llvm-config --ldflags --libs OUTPUT_VARIABLE llvm_libs OUTPUT_STRIP_TRAILING_WHITESPACE)
set(LLVM_DEFINITIONS "${LLVM_CPPFLAGS}")
