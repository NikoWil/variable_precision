//
// Created by khondar on 18.03.20.
//

#ifndef CODE_PI_UTIL_H
#define CODE_PI_UTIL_H

#include <vector>
#include <cstdint>
#include "../segmentation/seg_uint.h"

namespace {
    template <int N>
    inline double norm(const std::vector<double> &v) {
        double sum{0.};
        for (const auto e : v) {
            sum += std::pow(std::abs(e), N);
        }

        return std::pow(sum, 1. / static_cast<double>(N));
    }

    template <int N>
    inline double norm_2(const std::vector<std::uint16_t> &v) {
        double sum{0.};
        for (size_t i{0}; i < v.size(); ++i) {
            double val;
            seg_uint::read_2(&v.at(i), &val);
            sum += std::pow(std::abs(val), N);
        }

        return std::pow(sum, 1./ static_cast<double>(N));
    }

    template <int N>
    inline double norm_4(const std::vector<std::uint32_t> &v) {
        double sum{0.};
        for (size_t i{0}; i < v.size(); ++i) {
            double val;
            seg_uint::read_4(&v.at(i), &val);
            sum += std::pow(std::abs(val), N);
        }

        return std::pow(sum, 1./ static_cast<double>(N));
    }

    template <int N>
    inline double norm_6(const std::vector<std::uint16_t> &v) {
        assert(v.size() % 3 == 0 && "norm_6 vector size has to be multiple of 3");

        double sum{0.};
        for (size_t i{0}; i < v.size(); i += 3) {
            double val;
            seg_uint::read_6(&v.at(i), &val);
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

    for (auto &e : v) {
        e = e / norm_fac;
    }
    return true;
}

template <int N>
inline bool normalize_2(std::vector<std::uint16_t> &v) {
    const auto norm_fac = norm_2<N>(v);
    if (norm_fac == 0) {
        return false;
    }

    for (size_t i{0}; i < v.size(); ++i) {
        double num;
        seg_uint::read_2(&v.at(i), &num);
        num /= norm_fac;
        seg_uint::write_2(&v.at(i), &num);
    }
    return true;
}

template <int N>
inline bool normalize_4(std::vector<std::uint32_t> &v) {
    const auto norm_fac = norm_4<N>(v);
    if (norm_fac == 0) {
        return false;
    }

    for (size_t i{0}; i < v.size(); ++i) {
        double num;
        seg_uint::read_4(&v.at(i), &num);
        num /= norm_fac;
        seg_uint::write_4(&v.at(i), &num);
    }
    return true;
}

template <int N>
inline bool normalize_6(std::vector<std::uint16_t> &v) {
    const auto norm_fac = norm_6<N>(v);
    if (norm_fac == 0) {
        return false;
    }

    for (size_t i{0}; i < v.size(); i += 3) {
        double num;
        seg_uint::read_6(&v.at(i), &num);
        num /= norm_fac;
        seg_uint::write_6(&v.at(i), &num);
    }

    return true;
}

#endif //CODE_PI_UTIL_H
