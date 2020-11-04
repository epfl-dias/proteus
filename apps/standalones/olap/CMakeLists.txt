foreach(target
		scaleout-brdcst
		scaleout-pingpong
		scaleout-ssb
	)
	project(${target} VERSION 0.1 LANGUAGES CXX)

	add_executable(${target} main-${target}.cpp)

	target_link_libraries(${target}
		codegen::olap_flags
		proteus::olap
		proteus::command_provider
	)

	include(proteus-experiment)
endforeach(target)

