//
// Created by niko on 8/14/19.
//

#include <cassert>
#include <limits>

#include "segmentation.h"

union Seg_lazy {
  static_assert(sizeof(double) == sizeof(uint64_t), "Double segmentation only works if double and uint64_t are of same size");
  double d;
  uint64_t u;
};

uint64_t to_uint64_t(double d) {
  Seg_lazy s{d};
  return s.u;
}

double to_double(uint64_t u) {
  Seg_lazy s{};
  s.u = u;

  return s.d;
}

double fill_head(uint32_t head, double d) {
  Seg_lazy s{};
  s.u = head;
  s.u = s.u << 32u;

  auto from_double = get_tail(d);
  s.u |= from_double;

  return s.d;
}

double fill_tail(uint32_t tail, double d) {
  Seg_lazy s{};
  s.u = tail;

  auto from_double = get_head(d);
  s.u |= static_cast<uint64_t>(from_double) << 32u;

  return s.d;
}

uint32_t get_head(double d) {
  Seg_lazy s{d};
  auto intermediate = s.u >> 32u;

  assert(intermediate <= std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(intermediate);
}

uint32_t get_tail(double d) {
  Seg_lazy s{d};

  // TODO: use uint64_t for mask?
  uint32_t mask = (static_cast<uint64_t>(1) << 32u) - 1;

  s.u = s.u & mask;

  assert(s.u <= std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(s.u);
}