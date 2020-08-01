//
// Created by khondar on 18.03.20.
//

#ifndef CODE_PI_UTIL_H
#define CODE_PI_UTIL_H

#include <array>
#include <vector>
#include <cstdint>
#include "../segmentation/seg_uint.h"

namespace {
    template <int N>
    inline double norm(const std::vector<double> &v) {
        double sum{0.};
#pragma omp parallel for reduction(+ : sum) default(none) shared(v)
        for (size_t i = 0; i < v.size(); ++i) {
            sum += std::pow(std::abs(v.at(i)), N);
        }

        return std::pow(sum, 1. / static_cast<double>(N));
    }

    template <int N>
    inline double norm_2(const std::vector<std::uint16_t> &v) {
        double sum{0.};
#pragma omp parallel for reduction(+ : sum) default(none) shared(v)
        for (size_t i = 0; i < v.size(); ++i) {
            const double val = seg_uint::read_2(&v.at(i));
            sum += std::pow(std::abs(val), N);
        }

        return std::pow(sum, 1./ static_cast<double>(N));
    }

    template <int N>
    inline double norm_4(const std::vector<std::uint32_t> &v) {
        double sum{0.};
#pragma omp parallel for reduction(+ : sum) default(none) shared(v)
        for (size_t i = 0; i < v.size(); ++i) {
            const double val = seg_uint::read_4(&v.at(i));
            sum += std::pow(std::abs(val), N);
        }

        return std::pow(sum, 1./ static_cast<double>(N));
    }

    template <int N>
    inline double norm_6(const std::vector<std::array<std::uint16_t, 3>> &v) {
        double sum{0.};
#pragma omp parallel for reduction(+ : sum) default(none) shared(v)
        for (size_t i = 0; i < v.size(); i += 3) {
            const double val = seg_uint::read_6(v.at(i));
            sum += std::pow(std::abs(val), N);
        }
        return std::pow(sum, 1./ static_cast<double>(N));
    }
}

template <int N>
inline bool normalize(std::vector<double> &v) {
    const auto norm_fac = norm<N>(v);
    if (norm_fac == 0) {
        return false;
    }

#pragma omp parallel for default(none) shared(v)
    for (size_t i = 0; i < v.size(); ++i) {
        double new_val = v.at(i) / norm_fac;
        v.at(i) = new_val;
    }
    return true;
}

template <int N>
inline bool normalize_2(std::vector<std::uint16_t> &v) {
    const auto norm_fac = norm_2<N>(v);
    if (norm_fac == 0) {
        return false;
    }

#pragma omp parallel for default(none) shared(v)
    for (size_t i = 0; i < v.size(); ++i) {
        double num = seg_uint::read_2(&v.at(i));
        num /= norm_fac;
        v.at(i) = seg_uint::write_2(&num);
    }
    return true;
}

template <int N>
inline bool normalize_4(std::vector<std::uint32_t> &v) {
    const auto norm_fac = norm_4<N>(v);
    if (norm_fac == 0) {
        return false;
    }

#pragma omp parallel for default(none) shared(v)
    for (size_t i = 0; i < v.size(); ++i) {
        double num = seg_uint::read_4(&v.at(i));
        num /= norm_fac;
        v.at(i) = seg_uint::write_4(&num);
    }
    return true;
}

template <int N>
inline bool normalize_6(std::vector<std::array<std::uint16_t, 3>> &v) {
    const auto norm_fac = norm_6<N>(v);
    if (norm_fac == 0) {
        return false;
    }

#pragma omp parallel for default(none) shared(v)
    for (size_t i = 0; i < v.size(); i += 3) {
        double num = seg_uint::read_6(v.at(i));
        num /= norm_fac;
        v.at(i) = seg_uint::write_6(&num);
    }

    return true;
}

#endif //CODE_PI_UTIL_H
