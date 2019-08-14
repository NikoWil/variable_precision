ADD_LIBRARY(segmentation bitwise_helper.cpp)

set_target_properties(segmentation PROPERTIES LINKER_LANGUAGE CXX)

TARGET_SOURCES(segmentation PUBLIC segmentation.h segmentation.cpp bitwise_helper.h)

IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    TARGET_COMPILE_OPTIONS(segmentation PRIVATE /W4 /WX)
ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    TARGET_COMPILE_OPTIONS(segmentation PRIVATE -Wall -Wextra -pedantic -Werror)
ELSEIF()
    MESSAGE(FATAL_ERROR "Neither GCC nor MSVC detected!")
ENDIF()