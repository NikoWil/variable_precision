#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <random>
#include <array>
#include <vector>

#include "communication.h"
#include "linalg/power_iteration/poweriteration.h"
#include "matrix_formats/csr.hpp"
#include "performance_tests.h"
#include "power_iteration.h"
#include "seg_char.h"
#include "seg_uint.h"
#include "util/util.hpp"
#include "pi_benchmarks.h"

void benchmark_spmv();
void benchmark_spmv(unsigned size, double density, unsigned iterations, unsigned warmup, std::mt19937& rng);

void get_rowcnt_start_row(MPI_Comm comm, int num_rows, std::vector<int> &rowcnt, std::vector<int> &start_row) {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);

    rowcnt.clear();
    start_row.clear();
    start_row.push_back(0);

    for (int i{0}; i < comm_size; ++i) {
        unsigned start = (num_rows * i) / comm_size;
        unsigned end = (num_rows * (i + 1)) / comm_size;
        rowcnt.push_back(end - start);

        const auto last_start = start_row.back();
        start_row.push_back(last_start + rowcnt.back());
    }
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    const auto requested = MPI_THREAD_FUNNELED;
    int provided;
    MPI_Init_thread(&argc, &argv, requested, &provided);
    if (provided < requested) {
        std::cout << "No sufficient MPI multithreading support found\n";
        return 0;
    }

    std::cout << std::setprecision(20);

    //benchmark_spmv();
    std::mt19937 rng{std::random_device{}()};

    const int size = 1024 * 8;
    const double density = 1. / 256.;
    const double eta = 0.37;
    const unsigned max_iterations = 200;
    const int num_tests = 3;

    std::vector<int> rowcnt;
    std::vector<int> start_row;
    get_rowcnt_start_row(MPI_COMM_WORLD, size, rowcnt, start_row);

    std::vector<unsigned> seg_timings;
    std::vector<unsigned> fixed_timings;
    speedup_test(size, density, eta, max_iterations, num_tests, rowcnt, seg_timings, fixed_timings, MPI_COMM_WORLD);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    for (int i{0}; i < comm_size; ++i) {
        if (rank == i) {
            std::cout << "results rank " << i << "\n";
            print_vector(seg_timings, "seg_timings");
            print_vector(fixed_timings, "fixed_timings");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

template<typename T>
T median(std::vector<T> &v) {
    const auto size = v.size();
    const auto index_1 = size / 2;
    const auto index_2 = (size - 1) / 2;

    std::nth_element(v.begin(), v.begin() + index_1, v.end());
    const T e1 = v[index_1];
    std::nth_element(v.begin(), v.begin() + index_2, v.end());
    const T e2 = v[index_2];

    return (e1 + e2) / 2;
}

template<typename T>
T average(const std::vector<T> &v) {
    T sum{0};
    for (const auto e : v) {
        sum += e;
    }
    return sum / v.size();
}

void benchmark_spmv() {
    const auto seed = std::random_device{}();
    std::cout << "Seed: " << seed << "\n";
    std::mt19937 rng(seed);

    constexpr unsigned min_size{1u << 9u};
    constexpr unsigned max_size{1u << 18u};
    const std::array<double, 5> densities{1./ 32., 1. / 64., 1. / 128., 1. / 256., 1. / 512.};
    constexpr unsigned iterations{100};
    constexpr unsigned warmup{50};

    for (unsigned size{min_size}; size <= max_size; size <<= 1u) {
        for (const auto density: densities) {
            benchmark_spmv(size, density, iterations, warmup, rng);
        }
    }
}

void benchmark_spmv(unsigned size, double density, unsigned iterations, unsigned warmup, std::mt19937& rng) {
    // Generate matrix & (segmented) vector
    std::uniform_real_distribution<> value_distrib(0, 100'000);
    std::uniform_int_distribution<> index_distrib(0, size);

    const CSR matrix = CSR::diagonally_dominant(size, density, rng);

    std::vector<double> x(size);
    std::vector<uint32_t> x_halves(size);
    for (size_t i{0}; i < size; ++i) {
        const double val = value_distrib(rng);
        x[i] = val;
        seg_uint::write_4(&x_halves[i], &val);
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
            seg_uint::write_4(&x_halves[index], &val);
        }

        for (unsigned i{0}; i < iterations; ++i) {
            const auto start = high_resolution_clock::now();
            seg_uint::calc_convert::spmv_4(matrix, x_halves, y_halves);
            const auto end = high_resolution_clock::now();
            timings_calc_convert.push_back(duration_cast<nanoseconds>(end - start).count());

            u_sum += y_halves[0];
            const double val = value_distrib(rng);
            const unsigned index = index_distrib(rng);
            seg_uint::write_4(&x_halves[index], &val);
        }
    }

    std::vector<uint64_t> timings_pre_convert_conversion;
    std::vector<uint64_t> timings_pre_convert_spmv;
    // pre_convert
    {
        std::vector<double> x_double(x.size());
        for (unsigned i{0}; i < warmup; ++i) {
            for (size_t k{0}; k < x.size(); ++k) {
                seg_uint::read_4(&x_halves[k], &x_double[k]);
            }
            fixed::spmv(matrix, x_double, y);
            for (size_t k{0}; k < y.size(); ++k) {
                seg_uint::write_4(&y_halves.at(k), &y.at(k));
            }

            u_sum += y_halves[0];
            const double val = value_distrib(rng);
            const unsigned index = index_distrib(rng);
            seg_uint::write_4(&x_halves[index], &val);
        }

        for (unsigned i{0}; i < iterations; ++i) {
            const auto conv_to_start = high_resolution_clock::now();
            for (size_t k{0}; k < x.size(); ++k) {
                seg_uint::read_4(&x_halves[k], &x_double[k]);
            }
            const auto conv_to_end = high_resolution_clock::now();

            const auto spmv_start = high_resolution_clock::now();
            fixed::spmv(matrix, x_double, y);
            const auto spmv_end = high_resolution_clock::now();

            const auto conv_from_start = high_resolution_clock::now();
            for (size_t k{0}; k < y.size(); ++k) {
                seg_uint::write_4(&y_halves.at(k), &y.at(k));
            }
            const auto conv_from_end = high_resolution_clock::now();

            timings_pre_convert_spmv.push_back(duration_cast<nanoseconds>(spmv_end - spmv_start).count());
            timings_pre_convert_conversion.push_back(
                    duration_cast<nanoseconds>(conv_to_end - conv_to_start + conv_from_end - conv_from_start).count());

            u_sum += y_halves[0];
            const double val = value_distrib(rng);
            const unsigned index = index_distrib(rng);
            seg_uint::write_4(&x_halves[index], &val);
        }
    }

    // out_convert
    std::vector<uint64_t> timings_out_convert_conversion;
    std::vector<uint64_t> timings_out_convert_spmv;
    {
        std::vector<double> x_double(x.size());
        for (unsigned i{0}; i < warmup; ++i) {
            for (size_t k{0}; k < x.size(); ++k) {
                seg_uint::read_4(&x_halves.at(k), &x_double.at(k));
            }
            seg_uint::out_convert::spmv_4(matrix, x_double, y_halves);

            u_sum += y_halves[0];
            const double val = value_distrib(rng);
            const unsigned index = index_distrib(rng);
            seg_uint::write_4(&x_halves[index], &val);
        }


        for (unsigned i{0}; i < iterations; ++i) {
            const auto start_conv = high_resolution_clock::now();
            for (size_t k{0}; k < x.size(); ++k) {
                seg_uint::read_4(&x_halves.at(k), &x_double.at(k));
            }
            const auto end_conv = high_resolution_clock::now();

            const auto start_spmv = high_resolution_clock::now();
            seg_uint::out_convert::spmv_4(matrix, x_double, y_halves);
            const auto end_spmv = high_resolution_clock::now();

            timings_out_convert_conversion.push_back(duration_cast<nanoseconds>(end_conv - start_conv).count());
            timings_out_convert_spmv.push_back(duration_cast<nanoseconds>(end_spmv - start_spmv).count());

            u_sum += y_halves[0];
            const double val = value_distrib(rng);
            const unsigned index = index_distrib(rng);
            seg_uint::write_4(&x_halves[index], &val);
        }
    }

    std::vector<uint64_t> timings_pre_convert_total(iterations);
    std::transform(timings_pre_convert_conversion.begin(), timings_pre_convert_conversion.end(),
                   timings_pre_convert_spmv.begin(), timings_pre_convert_total.begin(), std::plus<uint64_t>());

    std::vector<uint64_t> timings_out_convert_total(iterations);
    std::transform(timings_out_convert_conversion.begin(), timings_out_convert_conversion.end(),
                   timings_out_convert_spmv.begin(), timings_out_convert_total.begin(), std::plus<uint64_t>());

    // calculate median/ average timings
    const auto median_fixed = median(timings_fixed);
    const auto median_calc_convert = median(timings_calc_convert);

    const auto median_pre_convert_conv = median(timings_pre_convert_conversion);
    const auto median_pre_convert_spmv = median(timings_pre_convert_spmv);
    const auto median_pre_convert_total = median(timings_pre_convert_total);

    const auto median_out_convert_conv = median(timings_out_convert_conversion);
    const auto median_out_convert_spmv = median(timings_out_convert_spmv);
    const auto median_out_convert_total = median(timings_out_convert_total);

    const auto average_fixed = average(timings_fixed);
    const auto average_calc_convert = average(timings_calc_convert);

    const auto average_pre_convert_conv = average(timings_pre_convert_conversion);
    const auto average_pre_convert_spmv = average(timings_pre_convert_spmv);
    const auto average_pre_convert_total = average(timings_pre_convert_total);

    const auto average_out_convert_conv = average(timings_out_convert_conversion);
    const auto average_out_convert_spmv = average(timings_out_convert_spmv);
    const auto average_out_convert_total = average(timings_out_convert_total);


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

    std::cout << "Segmented SpMV, conversion from segments before, conversion to segments during SpMV\n";
    std::cout << "median: total conversion spmv\n";
    std::cout << median_out_convert_total << " " << median_out_convert_conv << " " << median_out_convert_spmv << "\n";
    std::cout << "average: total conversion spmv\n";
    std::cout << average_out_convert_total << " " << average_out_convert_conv << " " << average_out_convert_spmv
              << "\n";
    print_vector(timings_out_convert_total, "timings_out-conv-total");
    print_vector(timings_out_convert_conversion, "timings_out-conv-conversion");
    print_vector(timings_out_convert_spmv, "timings_out-conv-spmv");
}
