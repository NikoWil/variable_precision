#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <random>

#include "pi_benchmarks.h"
#include "spmv_benchmark.h"
#include "matrix_formats/csr.hpp"

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
    (void) argc;
    (void) argv;
    const auto requested = MPI_THREAD_FUNNELED;
    int provided;
    MPI_Init_thread(&argc, &argv, requested, &provided);
    if (provided < requested) {
        std::cout << "No sufficient MPI multithreading support found\n";
        return 0;
    }

    std::cout << std::setprecision(20);

    std::mt19937 rng(std::random_device{}());
    const CSR matrix = CSR::fixed_eta(6, 0.4, 0.5, rng);
    const CSR transpose = CSR::transpose(matrix);

    std::cout << "matrix:\n";
    matrix.print();

    std::cout << "\ntranspose:\n";
    transpose.print();

    // benchmark_spmv(20);
    /*std::mt19937 rng{std::random_device{}()};
    std::array<int, 7> sizes{1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16};

    const double density = 1. / 128.;
    const double eta = 0.5;

    for (const auto size : sizes) {
        const auto matrix = CSR::fixed_eta(size, density, eta, rng);

        std::vector<double> x(size, 1);
        std::vector<uint32_t> x_halves(size);
        for (size_t i{0}; i < x.size(); ++i) {
            seg_uint::write_4(x_halves.data() + i, x.data() + i);
        }

        std::vector<double> y(x.size());
        std::vector<uint32_t> y_halves(x_halves.size());

        std::vector<unsigned long> fixed_timings;
        std::vector<unsigned long> calc_conv_timings;
        std::vector<unsigned long> conv_both_timings;

        for (int i{0}; i < 100; ++i) {
            const auto start_fixed = std::chrono::high_resolution_clock::now();
            fixed::spmv(matrix, x, y);
            const auto end_fixed = std::chrono::high_resolution_clock::now();
            fixed_timings.push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end_fixed - start_fixed).count());
        }

        for (int i{0}; i < 100; ++i) {
            const auto start_seg = std::chrono::high_resolution_clock::now();
            seg_uint::calc_convert::spmv_4(matrix, x_halves, y_halves);
            const auto end_seg = std::chrono::high_resolution_clock::now();
            calc_conv_timings.push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end_seg - start_seg).count());
        }

        for (int i{0}; i < 100; ++i) {
            const auto start_conv_both = std::chrono::high_resolution_clock::now();
            seg_uint::out_convert::spmv_4(matrix, x, y_halves);
            const auto end_conv_both = std::chrono::high_resolution_clock::now();
            conv_both_timings.push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end_conv_both - start_conv_both).count());
        }

        std::cout << "size: " << size << "\n";
        std::cout << "medians:\n";
        std::cout << "\tfixed:\t" << median(fixed_timings) << '\n';
        std::cout << "\tseg:\t" << median(calc_conv_timings) << '\n';
        std::cout << "\tboth:\t" << median(conv_both_timings) << '\n';

        std::cout << "averages:\n";
        std::cout << "\tfixed:\t" << average(fixed_timings) << '\n';
        std::cout << "\tseg:\t" << average(calc_conv_timings) << '\n';
        std::cout << "\tboth:\t" << average(conv_both_timings) << '\n';

        print_vector(fixed_timings, "fixed_timings\t\t");
        print_vector(calc_conv_timings, "calc_conv_timings\t");
        print_vector(conv_both_timings, "conv_both_timings\t");
        std::cout << '\n';
    }
    // */

    MPI_Finalize();
    return 0;
}
