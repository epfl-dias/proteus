get_filename_component(TS_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_proteus_olap_perftest(perftest-${TS_NAME} proteus::olap queue-perf.cpp)
