# preprocess files with m4 if needed
set(FILES1
    "ndarray/ArrayBaseN.h"
    )

set(FILES2
    "ndarray/ArrayRef.h"
    "ndarray/operators.h"
    "ndarray/Vector.h"
    "ndarray/fft/FFTWTraits.h"
    )

set(PREPROCESS_FILES1 "")
foreach(FILENAME ${FILES1})
	list(APPEND PREPROCESS_FILES1 "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
endforeach(FILENAME)

set(PREPROCESS_FILES2 "")
foreach(FILENAME ${FILES2})
    list(APPEND PREPROCESS_FILES2 "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
endforeach(FILENAME)

foreach(FILENAME ${PREPROCESS_FILES1})
    add_custom_command (
        OUTPUT ${FILENAME}
        DEPENDS ${FILENAME}.m4
        COMMAND m4 ${FILENAME}.m4 > ${FILENAME}
        )
endforeach(FILENAME)

foreach(FILENAME ${PREPROCESS_FILES2})
    add_custom_command (
        OUTPUT ${FILENAME}
        DEPENDS ${FILENAME}.m4
        COMMAND m4 ${FILENAME}.m4 > ${FILENAME}
        )
endforeach(FILENAME)

# record certain build options
configure_file(ndarray/buildOptions.h.in "${CMAKE_CURRENT_SOURCE_DIR}/ndarray/buildOptions.h")

add_custom_target(preprocess_includes1 ALL DEPENDS ${PREPROCESS_FILES1})
add_custom_target(preprocess_includes2 ALL DEPENDS ${PREPROCESS_FILES2})
add_dependencies(preprocess_includes2 preprocess_includes1)
