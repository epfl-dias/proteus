file(GLOB jsmn_src
	*.c
    )
add_library(jsmn SHARED ${jsmn_src})
set_target_properties(jsmn PROPERTIES COMPILE_FLAGS "${LLVM_CFLAGS}")

install(TARGETS jsmn
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
    )
