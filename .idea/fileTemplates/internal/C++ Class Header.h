#parse("C File Header.h")
#[[#ifndef]]# ${INCLUDE_GUARD}
#[[#define]]# ${INCLUDE_GUARD}

#if (${NAMESPACES_OPEN} == "")
namespace ${PROJECT_NAME} {
#else
${NAMESPACES_OPEN}
#end

class ${NAME} {

};

#if (${NAMESPACES_OPEN} == "")
}  // namespace ${PROJECT_NAME}
#else
${NAMESPACES_CLOSE}
#end

#[[#endif]]# /* ${INCLUDE_GUARD} */
