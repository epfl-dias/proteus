set(_proteus_install_target ${PROJECT_NAME})

#set(_proteus_install_dev ${PROJECT_NAME}_Development)
#set(_proteus_install_bin ${PROJECT_NAME}_RunTime)
#set(_proteus_install_dev ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_SUFFIX_DEV})
#set(_proteus_install_bin ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_SUFFIX_BINARIES})

target_enable_default_warnings(${PROJECT_NAME})

set(_proteus_install_dev ${PROTEUS_CPACK_COMP_DEV})
set(_proteus_install_bin ${PROTEUS_CPACK_COMP_BINARIES})
include(_proteus-install)

#set(CPACK_DEBIAN_PACKAGE_DEPENDS )
