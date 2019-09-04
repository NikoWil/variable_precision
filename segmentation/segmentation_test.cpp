//
// Created by niko on 8/13/19.
//

#include <cassert>
#include <cstdint>
#include <limits>

#include "segmentation.h"

void test_all_one() {
  uint64_t all_one = std::numeric_limits<uint64_t>::max();
  double d = to_double(all_one);

  auto head = get_head(d);
  auto tail = get_tail(d);
  assert(head == 0xFFFFFFFF);
  assert(tail == 0xFFFFFFFF);

  auto head_only = fill_head(head);
  auto tail_only = fill_tail(tail);
  assert(to_uint64_t(head_only) == 0xFFFFFFFF00000000);
  assert(to_uint64_t(tail_only) == 0x00000000FFFFFFFF);

  assert(to_uint64_t(fill_head(head, fill_tail(tail))) == all_one);
  assert(to_uint64_t(fill_tail(tail, fill_head(head))) == all_one);
}

void test_front_one() {
  uint64_t front_one = 0xFFFFFFFF00000000;
  double d = to_double(front_one);

  auto head = get_head(d);
  auto tail = get_tail(d);

  assert(head == 0xFFFFFFFF);
  assert(tail == 0x00000000);

  auto head_only = fill_head(head);
  auto tail_only = fill_tail(tail);
  assert(to_uint64_t(head_only) == 0xFFFFFFFF00000000);
  assert(to_uint64_t(tail_only) == 0x0000000000000000);

  assert(to_uint64_t(fill_head(head, fill_tail(tail))) == front_one);
  assert(to_uint64_t(fill_tail(tail, fill_head(head))) == front_one);
}

void test_back_one() {
  uint64_t back_one = 0x00000000FFFFFFFF;
  double d = to_double(back_one);

  auto head = get_head(d);
  auto tail = get_tail(d);
  assert(head == 0x00000000);
  assert(tail == 0xFFFFFFFF);

  auto head_only = fill_head(head);
  auto tail_only = fill_tail(tail);
  assert(to_uint64_t(head_only) == 0x0000000000000000);
  assert(to_uint64_t(tail_only) == 0x00000000FFFFFFFF);

  assert(to_uint64_t(fill_head(head, fill_tail(tail))) == back_one);
  assert(to_uint64_t(fill_tail(tail, fill_head(head))) == back_one);
}

void segmentation_test() {
  test_all_one();
  test_front_one();
  test_back_one();
}