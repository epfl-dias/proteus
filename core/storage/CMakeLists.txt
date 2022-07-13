project(storage VERSION 0.1 LANGUAGES CXX)


# Pure, regular C++ files
set(storage_cxx
  lib/mmap-file.cpp
  lib/storage-load-policy-registry.cpp
  lib/storage-manager.cpp
  )


#-----------------------------------------------------------------------------

install(DIRECTORY DESTINATION ${CMAKE_INSTALL_BINDIR}/generated_code
  #COMPONENT ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_SUFFIX_BINARIES}
  COMPONENT ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_BINARIES}
  )

add_library(storage SHARED
  ${storage_cxx}
  )

# If we have compiler requirements for this library, list them
# here
target_compile_features(storage PUBLIC cxx_std_20)

# Depend on a library that we defined in the top-level file
target_link_libraries(storage
  PUBLIC
  proteus::platform
  PRIVATE
  magic_enum::magic_enum
  )

if (VTUNE AND VTUNE_ENABLE)
  target_link_libraries(storage PUBLIC vtune::vtune)
endif ()

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

#-----------------------------------------------------------------------------
# Unit tests

add_proteus_unit_test(unit-tests-storage proteus::storage
  test/storage-test-main.cpp
  test/test-mmap-file.cpp
  )

target_link_libraries(unit-tests-storage
  PUBLIC
  proteus::storage
  )

# Unit-tests
if (UNIT_TESTS)
  add_subdirectory(tests)
  add_subdirectory(perftests)
endif ()
