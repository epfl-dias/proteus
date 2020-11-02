project(htap VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(htap_cxx
	# utils/utils.cpp
        include/queries/query-interface.cpp
    )

foreach(target ${htap_cxx})
    set_source_files_properties(${target} PROPERTIES LANGUAGE CXX)
endforeach()

# foreach(target ${htap_cxx})
#     set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
# endforeach()

add_library(${PROJECT_NAME} SHARED
        ${htap_cxx}
        )

# If we have compiler requirements for this library, list them
# here
target_compile_features(${PROJECT_NAME}
        PUBLIC cxx_std_20)

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
        PUBLIC
            proteus::olap
            oltp::engine
            oltp::adaptor
            )

include(proteus-lib)
add_library(htap::engine ALIAS ${PROJECT_NAME})
