//
// Created by niko on 8/9/19.
//

#ifndef CODE_SEG_UINT_H
#define CODE_SEG_UINT_H

#include <array>
#include <cstdint>

namespace {
    static_assert(std::numeric_limits<double>::is_iec559,
                  "IEEE 754 floating point format is required for segmentation to work");

    union Seg_16 {
        static_assert(sizeof(double) == 4 * sizeof(std::uint16_t),
                      "Double segmentation needs 4 uint16_t to be exactly 1 double size-wise");

        double d;
        std::uint64_t u64;
        std::uint16_t u16[4];
    };

    union Seg_32 {
        static_assert(sizeof(double) == 2 * sizeof(std::uint32_t),
                      "Double segmentation needs 2 uint32_t to be exactly 1 double size-wise");
        double d;
        std::uint64_t u64;
        std::uint32_t u32[2];
    };
}

namespace seg_uint {
    inline double read_2(const std::uint16_t *const u) {
        Seg_16 s{};
        s.u64 = 0;
        s.u16[3] = *u;
        return s.d;
    }

    inline double read_4(const std::uint32_t *const u) {
        Seg_32 s{};
        s.u64 = 0;
        s.u32[1] = *u;
        return s.d;
    }

    inline double read_6(std::array<std::uint16_t, 3> u) {
        Seg_16 s{};
        s.u64 = 0;
        s.u16[1] = u[0];
        s.u16[2] = u[1];
        s.u16[3] = u[2];
        return s.d;
    }

    inline std::uint16_t write_2(const double *const d) {
        Seg_16 s{};
        s.d = *d;
        return s.u16[3];
    }

    inline std::uint32_t write_4(const double *const d) {
        Seg_32 s{};
        s.d = *d;
        return s.u32[1];
    }

    inline std::array<std::uint16_t, 3> write_6(const double *const d) {
        Seg_16 s{};
        s.d = *d;
        return {s.u16[1], s.u16[2], s.u16[3]};
    }
}

#endif // CODE_SEG_UINT_H
