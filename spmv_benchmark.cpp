//
// Created by khondar on 23.02.20.
//

#include "spmv_benchmark.h"
#include "matrix_formats/csr.hpp"
#include "segmentation/seg_uint.h"
#include "spmv/spmv_fixed.h"

#include <array>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>


void benchmark_spmv(unsigned size, double density, unsigned iterations, unsigned warmup, std::mt19937 &rng) {
    // Generate matrix & (segmented) vector
    std::uniform_real_distribution<> value_distrib(0, 100'000);
    std::uniform_int_distribution<> index_distrib(0, size);

    const CSR matrix = CSR::diagonally_dominant(size, density, rng);

    std::vector<double> x(size);
    std::vector<uint32_t> x_halves(size);
    for (size_t i{0}; i < size; ++i) {
        const double val = value_distrib(rng);
        x[i] = val;
        x_halves[i] = seg_uint::write_4(&val);
    }

    std::vector<double> y(size);
    std::vector<uint32_t> y_halves(size);
    double d_sum{0};
    uint64_t u_sum{0};

    using namespace std::chrono;

    std::vector<uint64_t> timings_fixed;
    // test normal spmv timing
    {
        for (unsigned i{0}; i < warmup; ++i) {
            fixed::spmv(matrix, x, y);

            d_sum += y[0];
            x[index_distrib(rng)] = value_distrib(rng);
        }

        for (unsigned i{0}; i < iterations; ++i) {
            const auto start = high_resolution_clock::now();
            fixed::spmv(matrix, x, y);
            const auto end = high_resolution_clock::now();
            timings_fixed.push_back(duration_cast<nanoseconds>(end - start).count());

            d_sum += y[0];
            x[index_distrib(rng)] = value_distrib(rng);
        }
    }

    std::vector<uint64_t> timings_calc_convert;
    // test segmented spmv
    // calc_convert
    {
        for (unsigned i{0}; i < warmup; ++i) {
            seg_uint::calc_convert::spmv_4(matrix, x_halves, y_halves);

            u_sum += y_halves[0];
            const double val = value_distrib(rng);
            const unsigned index = index_distrib(rng);
            x_halves[index] = seg_uint::write_4(&val);
        }

        for (unsigned i{0}; i < iterations; ++i) {
            const auto start = high_resolution_clock::now();
            seg_uint::calc_convert::spmv_4(matrix, x_halves, y_halves);
            const auto end = high_resolution_clock::now();
            timings_calc_convert.push_back(duration_cast<nanoseconds>(end - start).count());

            u_sum += y_halves[0];
            const double val = value_distrib(rng);
            const unsigned index = index_distrib(rng);
            x_halves[index] = seg_uint::write_4(&val);
        }
    }

    std::vector<uint64_t> timings_pre_convert_conversion;
    std::vector<uint64_t> timings_pre_convert_spmv;
    // pre_convert
    {
        std::vector<double> x_double(x.size());
        for (unsigned i{0}; i < warmup; ++i) {
            for (size_t k{0}; k < x.size(); ++k) {
                x_double[k] = seg_uint::read_4(&x_halves[k]);
            }
            fixed::spmv(matrix, x_double, y);
            for (size_t k{0}; k < y.size(); ++k) {
                y_halves.at(k) = seg_uint::write_4(&y.at(k));
            }

            u_sum += y_halves[0];
            const double val = value_distrib(rng);
            const unsigned index = index_distrib(rng);
            x_halves.at(index) = seg_uint::write_4(&val);
        }

        for (unsigned i{0}; i < iterations; ++i) {
            const auto conv_to_start = high_resolution_clock::now();
            for (size_t k{0}; k < x.size(); ++k) {
                x_double.at(k) = seg_uint::read_4(&x_halves[k]);
            }
            const auto conv_to_end = high_resolution_clock::now();

            const auto spmv_start = high_resolution_clock::now();
            fixed::spmv(matrix, x_double, y);
            const auto spmv_end = high_resolution_clock::now();

            const auto conv_from_start = high_resolution_clock::now();
            for (size_t k{0}; k < y.size(); ++k) {
                y_halves.at(k) = seg_uint::write_4(&y.at(k));
            }
            const auto conv_from_end = high_resolution_clock::now();

            timings_pre_convert_spmv.push_back(duration_cast<nanoseconds>(spmv_end - spmv_start).count());
            timings_pre_convert_conversion.push_back(
                    duration_cast<nanoseconds>(conv_to_end - conv_to_start + conv_from_end - conv_from_start).count());

            u_sum += y_halves[0];
            const double val = value_distrib(rng);
            const unsigned index = index_distrib(rng);
            x_halves.at(index) = seg_uint::write_4(&val);
        }
    }

    std::vector<uint64_t> timings_pre_convert_total(iterations);
    std::transform(timings_pre_convert_conversion.begin(), timings_pre_convert_conversion.end(),
                   timings_pre_convert_spmv.begin(), timings_pre_convert_total.begin(), std::plus<uint64_t >());

    // calculate median/ average timings
    const auto median_fixed = median(timings_fixed);
    const auto median_calc_convert = median(timings_calc_convert);

    const auto median_pre_convert_conv = median(timings_pre_convert_conversion);
    const auto median_pre_convert_spmv = median(timings_pre_convert_spmv);
    const auto median_pre_convert_total = median(timings_pre_convert_total);

    const auto average_fixed = average(timings_fixed);
    const auto average_calc_convert = average(timings_calc_convert);

    const auto average_pre_convert_conv = average(timings_pre_convert_conversion);
    const auto average_pre_convert_spmv = average(timings_pre_convert_spmv);
    const auto average_pre_convert_total = average(timings_pre_convert_total);


    // output results
    std::cout << "SpMV Timing benchmark.\nParameters:\nsize: " << size
              << ", density: " << density << ", iterations: " << iterations
              << ", warmup: " << warmup << "\n";

    std::cout << "Fixed SpMV\n";
    std::cout << "median: total\n";
    std::cout << median_fixed << "\n";
    std::cout << "average: total\n";
    std::cout << average_fixed << "\n";
    print_vector(timings_fixed, "timings_fixed");

    std::cout << "Segmented SpMV, conversion both ways during SpMV\n";
    std::cout << "median: total\n";
    std::cout << median_calc_convert << "\n";
    std::cout << "average: total\n";
    std::cout << average_calc_convert << "\n";
    print_vector(timings_calc_convert, "timings_fixed");

    std::cout << "Segmented SpMV, conversion before/ after SpMV\n";
    std::cout << "median: total conversion spmv\n";
    std::cout << median_pre_convert_total << " " << median_pre_convert_conv << " " << median_pre_convert_spmv << "\n";
    std::cout << "average: total conversion spmv\n";
    std::cout << average_pre_convert_total << " " << average_pre_convert_conv << " " << average_pre_convert_spmv
              << "\n";
    print_vector(timings_pre_convert_total, "timings_pre-conv-total");
    print_vector(timings_pre_convert_conversion, "timings_pre-conv-conversion");
    print_vector(timings_pre_convert_spmv, "timings_pre-conv-spmv");
}


void benchmark_spmv(unsigned num_instances) {
    const auto seed = std::random_device{}();
    std::cout << "Seed: " << seed << "\n";
    std::mt19937 rng(seed);

    constexpr unsigned min_size{1u << 10u};
    constexpr unsigned max_size{1u << 18u};
    const std::array<double, 5> densities{1. / 32., 1. / 64., 1. / 128., 1. / 256., 1. / 512.};
    constexpr unsigned iterations{100};
    constexpr unsigned warmup{20};

    for (unsigned size{min_size}; size <= max_size; size <<= 1u) {
        for (const auto density: densities) {
            for (unsigned i{0}; i < num_instances; ++i) {
                benchmark_spmv(size, density, iterations, warmup, rng);
            }
        }
    }
}
