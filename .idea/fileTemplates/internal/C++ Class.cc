#parse("C File Header.h")

#[[#include]]# "${HEADER_FILENAME}"

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

