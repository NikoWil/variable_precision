//
// Created by niko on 8/14/19.
//

#include <cassert>
#include <limits>

#include "seg_uint.h"

union Seg_lazy {
  static_assert(sizeof(double) == sizeof(uint64_t), "Double segmentation only works if double and uint64_t are of same size");
  double d;
  uint64_t u;
};

union Seg_16 {
  static_assert(sizeof(double) == 4 * sizeof(uint16_t), "Double segmentation needs 4 uint16_t to be exactly 1 double size-wise");

  double d;
  uint64_t u64;
  uint16_t u16[4];
};

union Seg_32 {
  static_assert(sizeof(double) == 2 * sizeof(uint32_t), "Double segmentation needs 2 uint32_t to be exactly 1 double size-wise");
  double d;
  uint64_t u64;
  uint32_t u32[2];
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

namespace seg_uint {
void read_2(const uint16_t *const u, double *d) {
  Seg_16 s;
  s.u64 = 0;
  s.u16[3] = *u;
  *d = s.d;
}

void read_4(const uint32_t *const u, double *d) {
  Seg_32 s;
  s.u64 = 0;
  s.u32[1] = *u;
  *d = s.d;
}

void read_6(const uint16_t *const u, double *d) {
  Seg_16 s;
  s.u64 = 0;
  s.u16[3] = u[0];
  s.u16[2] = u[1];
  s.u16[1] = u[2];
  *d = s.d;
}

void write_2(uint16_t *u, const double *const d) {
  Seg_16 s;
  s.d = *d;
  u[0] = s.u16[3];
}

void write_4(uint32_t *u, const double *const d) {
  Seg_32 s;
  s.d = *d;
  u[0] = s.u32[1];
}

void write_6(uint16_t *u, const double *const d) {
  Seg_16 s;
  s.d = *d;
  u[0] = s.u16[3];
  u[1] = s.u16[2];
  u[2] = s.u16[1];
}
}
