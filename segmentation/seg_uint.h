//
// Created by niko on 8/9/19.
//

#ifndef CODE_SEG_UINT_H
#define CODE_SEG_UINT_H

#include <cstdint>

namespace {
    static_assert(std::numeric_limits<double>::is_iec559,
                  "IEEE 754 floating point format is required for segmentation to work");

    union Seg_16 {
        static_assert(sizeof(double) == 4 * sizeof(uint16_t),
                      "Double segmentation needs 4 uint16_t to be exactly 1 double size-wise");

        double d;
        uint64_t u64;
        uint16_t u16[4];
    };

    union Seg_32 {
        static_assert(sizeof(double) == 2 * sizeof(uint32_t),
                      "Double segmentation needs 2 uint32_t to be exactly 1 double size-wise");
        double d;
        uint64_t u64;
        uint32_t u32[2];
    };
}

namespace seg_uint {
    inline void read_2(const uint16_t *const u, double *d) {
        Seg_16 s{};
        s.u64 = 0;
        s.u16[3] = *u;
        *d = s.d;
    }

    inline void read_4(const uint32_t *const u, double *d) {
        Seg_32 s{};
        s.u64 = 0;
        s.u32[1] = *u;
        *d = s.d;
    }

    inline void read_6(const uint16_t *const u, double *d) {
        Seg_16 s{};
        s.u64 = 0;
        s.u16[3] = u[2];
        s.u16[2] = u[1];
        s.u16[1] = u[0];
        *d = s.d;
    }

    inline void write_2(uint16_t *u, const double *const d) {
        Seg_16 s{};
        s.d = *d;
        u[0] = s.u16[3];
    }

    inline void write_4(uint32_t *u, const double *const d) {
        Seg_32 s{};
        s.d = *d;
        u[0] = s.u32[1];
    }

    inline void write_6(uint16_t *u, const double *const d) {
        Seg_16 s{};
        s.d = *d;
        u[0] = s.u16[1];
        u[1] = s.u16[2];
        u[2] = s.u16[3];
    }
}

#endif // CODE_SEG_UINT_H
