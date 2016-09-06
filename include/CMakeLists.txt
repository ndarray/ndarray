# preprocess files with m4 if needed
set(FILES
	"ndarray/ArrayRef.h"
	"ndarray/ArrayBaseN.h"
	"ndarray/operators.h"
	"ndarray/Vector.h"
	"ndarray/fft/FFTWTraits.h"
	)

set(PREPROCESS_FILES "")
foreach(FILENAME ${FILES})
	list(APPEND PREPROCESS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
endforeach(FILENAME)

foreach(FILENAME ${PREPROCESS_FILES})
	add_custom_command (
		OUTPUT ${FILENAME}
		DEPENDS ${FILENAME}.m4
		COMMAND m4 ${FILENAME}.m4 > ${FILENAME}
		)
endforeach(FILENAME)

add_custom_target(preprocess_includes ALL DEPENDS ${PREPROCESS_FILES})