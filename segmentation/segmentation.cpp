//
// Created by niko on 8/14/19.
//

#include <array>
#include <cassert>
#include <limits>

#include "bitwise_helper.h"
#include "segmentation.h"

union Seg_lazy {
  static_assert(sizeof(double) == sizeof(uint64_t), "Double segmentation only works if double and uint64_t are of same size");
  double d;
  uint64_t u;
};

union Segmentation {
  double d;
  std::array<char, sizeof(double)> arr;
};

double fill_head(uint32_t head, double d) {
  Seg_lazy s{};
  s.u = head;
  s.u = s.u << 32;

  uint64_t from_double = to_uint64_bitwise(d);
  from_double &= 0x00000000FFFFFFFF;
  s.u |= from_double;

  return s.d;
}

double fill_tail(uint32_t tail, double d) {
  Seg_lazy s{};
  s.u = tail;

  auto from_double = to_uint64_bitwise(d);
  from_double &= 0xFFFFFFFF00000000;
  s.u |= from_double;

  return s.d;
}

uint32_t get_head(double d) {
  Seg_lazy s{d};
  auto intermediate = s.u >> 32;

  assert(intermediate <= std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(intermediate);
}

uint32_t get_tail(double d) {
  Seg_lazy s{d};

  uint32_t mask = (uint64_t(1) << 32) - 1;

  s.u = s.u & mask;

  assert(s.u <= std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(s.u);
}