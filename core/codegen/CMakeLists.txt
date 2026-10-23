project(codegen VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(codegen_cxx
  lib/context/context.cpp
  lib/context/parallel-context.cpp
  lib/expressions/binary-operators.cpp
  lib/expressions/expressions.cpp
  lib/expressions/expressionTypes.cpp
  lib/expressions/indexed-seq.cpp
  lib/expressions/ref-expressions.cpp
  lib/jit/control-flow/if-statement.cpp
  lib/jit/cpu-module.cpp
  lib/jit/cpu-pipeline.cpp
  lib/jit/gpu-module.cpp
  lib/jit/gpu-pipeline.cpp
  lib/jit/jit-module.cpp
  lib/jit/pipeline.cpp
  lib/jit/DebugIR.cpp
  lib/util/functions.cpp
  lib/util/gpu/gpu-intrinsics.cpp
)

add_library(${PROJECT_NAME} SHARED ${codegen_cxx})

# If we have compiler requirements for this library, list them
# here
target_compile_features(${PROJECT_NAME}
  PUBLIC
  cxx_std_20
)

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  LLVM::LLVM
  proteus::platform
  jsmn::jsmn
)

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
