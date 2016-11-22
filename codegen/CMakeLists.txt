file(GLOB codegen_src
	common/*.cpp
	util/*.cpp
	util/radix/*.cpp
	util/radix/joins/*.cpp
	util/radix/aggregations/*.cpp
	values/*.cpp
	expressions/*.cpp
	plugins/*.cpp
	plugins/output/*.cpp
	operators/*.cpp
	memory/*.cpp
	plan/*.cpp
	postgres/*.cpp
    )
add_library(codegen ${codegen_src})
set_target_properties(codegen PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
