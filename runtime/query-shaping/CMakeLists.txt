get_filename_component(STANDALONE_DIR ${CMAKE_CURRENT_SOURCE_DIR} NAME)

project(${STANDALONE_DIR} VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
    lib/experimental-shapers.cpp
    lib/input-prefix-query-shaper.cpp
    lib/query-shaper.cpp
    lib/scale-out-query-shaper.cpp
    )

target_link_libraries(${PROJECT_NAME}
    PUBLIC proteus::olap
    )

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
