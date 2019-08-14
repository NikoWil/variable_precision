//
// Created by niko on 8/14/19.
//

#include "bitwise_helper.h"

double to_double_bitwise(uint64_t u) {
  return *(double*)&u;
}

uint64_t to_uint64_bitwise(double d) {
  return *(uint64_t*)&d;
}
