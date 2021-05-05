function(target_enable_default_warnings target)
  set(scope PRIVATE)

  if (${ARGC} GREATER 1)
    set(scope ${vargc} ${ARGV1})
  endif ()

  target_compile_options(${target} BEFORE ${scope}
    # Turn on maximum code compliance and all the warnings
    -pedantic
    -Weverything
    # Set as error returning a C++ object from a C-linkage function
    -Wreturn-type-c-linkage
    -Werror=return-type-c-linkage
    # Turn warnings into errors
    #    -Werror

    # Our Code generates the following warnings:
    -Wno-assign-enum
    -Wno-c++98-compat
    -Wno-c++98-compat-pedantic
    -Wno-cast-align
    -Wno-cast-qual
    -Wno-conversion
    -Wno-covered-switch-default

    # Disable warning-as-error for deprecated calls
    -Wno-deprecated
    -Wno-double-promotion
    -Wno-exit-time-destructors
    -Wno-global-constructors
    -Wno-ignored-qualifiers
    -Wno-missing-prototypes
    -Wno-old-style-cast
    -Wno-padded
    -Wno-reorder
    -Wno-reserved-id-macro
    -Wno-shadow
    -Wno-shadow-field
    -Wno-shadow-field-in-constructor
    -Wno-shorten-64-to-32
    -Wno-sign-compare
    -Wno-sign-conversion
    -Wno-switch-enum
    -Wno-undefined-func-template
    -Wno-unused-command-line-argument
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-vla
    -Wno-vla-extension
    -Wno-weak-vtables
    -Wno-atomic-implicit-seq-cst
    -Wno-unknown-cuda-version

    -Wno-zero-length-array
    -Wno-packed

    # LLVM & RapidJSON generated following.
    -Wno-ambiguous-reversed-operator

    # Unit-tests:
    -Wno-shift-sign-overflow
    )
endfunction()