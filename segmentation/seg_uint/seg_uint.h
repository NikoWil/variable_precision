//
// Created by niko on 8/9/19.
//

#ifndef CODE_SEG_UINT_H
#define CODE_SEG_UINT_H

#include <cstdint>

uint64_t to_uint64_t(double d);

double to_double(uint64_t u);

double fill_head(uint32_t head, double d = 0);

double fill_tail(uint32_t tail, double d = 0);

uint32_t get_head(double d);

uint32_t get_tail(double d);

namespace seg_uint {
/*
 * read_N(...): read from variable precision storage uintXX_t into usable double
 * write_N(...): write double into variable precision storage uintXX_t
 * TODO: is there a nice way to pack this thing in templates and have it called
 *  by going 'read<N>(...)'?
 *  Side goal: have it fail when handing in the wrong types
 */
void read_2(const uint16_t *, double *);

void read_4(const uint32_t *, double *);

void read_6(const uint16_t *, double *);

void write_2(uint16_t *, const double *);

void write_4(uint32_t *, const double *);

void write_6(uint16_t *, const double *);
}

#endif // CODE_SEG_UINT_H
