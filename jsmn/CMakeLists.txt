set(jsmn_src
	jsmn.c
	)
add_library(jsmn SHARED ${jsmn_src})

install(TARGETS jsmn
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
