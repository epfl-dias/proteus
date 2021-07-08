# Add google test here, so that we do not propagate to it the WARNING_FLAGS
include(external/CMakeLists.txt.gtest.in)

set(GTEST gtest)
set(GTEST_MAIN gtest_main)

export(TARGETS ${GTEST} ${GTEST_MAIN} FILE GtestConfig.cmake)
