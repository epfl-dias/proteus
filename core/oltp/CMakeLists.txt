project(oltp VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(oltp_cxx

        #-------- Common
        lib/common/constants.cpp
        lib/common/oltp-cli-flags.cpp

        #-------- Execution
        lib/execution/worker.cpp

        #-------- Index
        lib/index/hash_array.cpp
        # lib/index/ART/adaptive_radix_tree_index.cpp

        #-------- Snapshot
        lib/snapshot/cor_arena.cpp
        lib/snapshot/cow_arena.cpp
        lib/snapshot/cor_const_arena.cpp

        #-------- Storage
        lib/storage/table.cpp
        lib/storage/layout/column_store.cpp
        # lib/storage/layout/row_store.cpp
        lib/storage/multi-version/delta_storage.cpp
        lib/storage/multi-version/mv.cpp
        lib/storage/multi-version/mv-record-list.cpp
        lib/storage/multi-version/mv-attribute-list.cpp

        #-------- Transaction
        lib/transactions/transaction_manager.cpp
        #lib/transactions/concurrency-control/cc.cpp
        )


foreach (target ${txn_engine_cxx})
    set_source_files_properties(${target} PROPERTIES LANGUAGE CXX)
endforeach ()

foreach (target ${txn_engine_cxx})
    set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
endforeach ()

add_library(oltp SHARED
        ${oltp_cxx}
        )

add_subdirectory(lib/external/libcuckoo)

set_target_properties(oltp PROPERTIES
        LINK_FLAGS "-Wl,--no-allow-shlib-undefined -Wl,-z,defs")

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(oltp
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

# If we have compiler requirements for this library, list them
# here
target_compile_features(oltp
        PUBLIC cxx_std_20)

# Depend on a library that we defined in the top-level file
target_link_libraries(oltp
        PUBLIC
        ${llvm_libs}
        proteus::olap
        cuckoo::cuckoo
        cli-flags)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS oltp EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows


# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME})

# This makes the project importable from the build directory
export(TARGETS oltp FILE ${PROJECT_NAME}Config.cmake)

add_library(oltp::engine ALIAS oltp)


add_subdirectory(adaptor)

# Unit-tests
if (UNIT_TESTS)
    add_subdirectory(tests)
endif ()


