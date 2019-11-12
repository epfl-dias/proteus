project(aeolus_engine VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(txn_engine_cxx
    glo.cpp
    scheduler/worker.cpp
    scheduler/topology.cpp
    scheduler/affinity_manager.cpp
    scheduler/comm_manager.cpp
    storage/memory_manager.cpp
    storage/table.cpp
    storage/row_store.cpp
    storage/column_store.cpp
    storage/delta_storage.cpp
    transactions/transaction_manager.cpp
    #transactions/cc.cpp
    snapshot/cor_arena.cpp
    snapshot/cow_arena.cpp
    snapshot/cor_const_arena.cpp
    indexes/hash_array.cpp
    #oltp-cli-flags.cpp
    )

add_library(aeolus_engine SHARED
        ${txn_engine_cxx}
        )

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(aeolus_engine
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})

# If we have compiler requirements for this library, list them
# here
target_compile_features(aeolus_engine
        PUBLIC cxx_std_17)

# Depend on a library that we defined in the top-level file
target_link_libraries(aeolus_engine
        PUBLIC
            cuckoo codegen
        PRIVATE
            gflags::gflags)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS aeolus_engine EXPORT ${PROJECT_NAME}Config
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

# This makes the project importable from the build directory
export(TARGETS aeolus_engine FILE ${PROJECT_NAME}Config.cmake)

#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?
