ADD_LIBRARY(linalg)

TARGET_SOURCES(linalg PUBLIC spmv/spmv_fixed.h spmv/spmv_fixed.cpp power_iteration/poweriteration.h power_iteration/poweriteration.cpp)

TARGET_LINK_LIBRARIES(linalg PUBLIC formats seg_uint)