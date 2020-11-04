project(oltp-bench-runner VERSION 0.1 LANGUAGES CXX)

add_subdirectory(bench)

add_executable(${PROJECT_NAME}
	bench-runner.cpp
	bench-cli-flags.cpp
)

target_link_libraries(${PROJECT_NAME}
	oltp::engine
	oltp::bench
	gflags::gflags
	codegen::olap_flags
	Threads::Threads
)

find_package(Threads REQUIRED)

include(proteus-benchmark)
