project(cuckoo VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} INTERFACE)

# If we have compiler requirements for this library, list them
# here
target_compile_features(${PROJECT_NAME}
	INTERFACE
		cxx_std_17
)

include(proteus-headers-lib)
add_library(cuckoo::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
