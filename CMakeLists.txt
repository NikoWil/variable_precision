CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(code)

SET(CMAKE_CXX_STANDARD 17)

IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    ADD_COMPILE_OPTIONS(-fopenmp -lpthread)
ELSEIF()
    MESSAGE(FATAL_ERROR "Neither GCC nor MSVC detected!")
ENDIF()

FIND_PACKAGE(OpenMP)
IF (OPENMP_FOUND)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

FIND_PACKAGE(MPI REQUIRED)
INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
LINK_LIBRARIES(${MPI_LIBRARIES})

ADD_EXECUTABLE(code spmv_benchmark.cpp spmv_benchmark.h spmv/pr_spmv.cpp spmv/pr_spmv.h power_iteration/pagerank.cpp
        power_iteration/pagerank.h power_iteration/pi_util.h pagerank_test.cpp pagerank_test.h main.cpp util/util.hpp
        matrix_formats/csr.hpp matrix_formats/csr.cpp communication.cpp communication.h segmentation/seg_uint.h
        spmv/spmv_fixed.cpp spmv/spmv_fixed.h power_iteration/poweriteration.cpp power_iteration/poweriteration.h
        pi_benchmarks.cpp pi_benchmarks.h)

IF(MPI_COMPILE_FLAGS)
    SET_TARGET_PROPERTIES(code PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
ENDIF()

IF(MPI_LINK_FLAGS)
    SET_TARGET_PROPERTIES(code PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
ENDIF()

IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    MESSAGE("MSCV detected")
    TARGET_COMPILE_OPTIONS(code PRIVATE /W3 /WX /OpenMP)
ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    MESSAGE("GCC detected")
    TARGET_COMPILE_OPTIONS(code PRIVATE -ggdb -Wall -Wextra -pedantic -Werror)
ELSEIF()
    MESSAGE(FATAL_ERROR "Neither GCC nor MSVC detected!")
ENDIF()
