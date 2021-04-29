project(oltp_bench VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
  tpcc/tpcc_64.cpp
  tpcc/txn.cpp
  tpcc/utils.cpp
  tpcc/consistency_verifiers.cpp
  )

# If we have compiler requirements for this library, list them
# here
target_compile_features(oltp_bench
  PUBLIC cxx_std_17)

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
  PRIVATE
  oltp::engine
  PUBLIC
  oltp::adaptor
  )

include(proteus-lib)
add_library(oltp::bench ALIAS ${PROJECT_NAME})
