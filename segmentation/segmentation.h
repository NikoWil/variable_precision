//
// Created by niko on 8/9/19.
//

#ifndef CODE_SEGMENTATION_H
#define CODE_SEGMENTATION_H

#include <cstdint>

uint64_t to_uint64_t(double d);

double fill_head(uint32_t head, double d = 0);

double fill_tail(uint32_t tail, double d = 0);

uint32_t get_head(double d);

uint32_t get_tail(double d);

#endif // CODE_SEGMENTATION_H
