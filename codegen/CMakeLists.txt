file(GLOB codegen_src
	common/*.cpp
	util/*.cpp
	util/gpu/*.cpp
	util/radix/*.cpp
	util/radix/joins/*.cpp
	util/radix/aggregations/*.cpp
	values/*.cpp
	expressions/*.cpp
	plugins/*.cpp
	plugins/output/*.cpp
	operators/*.cpp
	operators/gpu/*.cpp
	memory/*.cpp
	plan/*.cpp
	postgres/*.cpp
    )
add_library(codegen SHARED ${codegen_src})
set_target_properties(codegen PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})
target_link_libraries(codegen jsmn)

install(TARGETS codegen
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
    )
