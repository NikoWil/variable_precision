#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <random>
#include <chrono>

#include "pi_benchmarks.h"
#include "spmv_benchmark.h"
#include "matrix_formats/csr.hpp"
#include "power_iteration/pagerank.h"
#include "segmentation/seg_uint.h"
#include "spmv/spmv_fixed.h"
#include "communication.h"
#include "spmv/pr_spmv.h"
#include "power_iteration/pi_util.h"

void get_rowcnt_start_row(MPI_Comm comm, unsigned num_rows, std::vector<int> &rowcnt, std::vector<int> &start_row) {
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

void compare_partial_pr_spmv() {
    const unsigned num_cols = 5;

    const std::vector<double> values{0.5, 0.5, 1, 1, 1, 1};
    const std::vector<int> colidx{1, 4, 2, 0, 2, 2};
    const std::vector<int> rowptr{0, 2, 3, 4, 5, 6};
    CSR matrix{values, colidx, rowptr, num_cols};
    matrix = CSR::transpose(matrix);

    const std::vector<double> v1{1.};
    const std::vector<int> c1{2};
    const std::vector<int> r1{0, 1};
    const CSR m1{v1, c1, r1, num_cols};

    const std::vector<double> v2{0.5};
    const std::vector<int> c2{0};
    const std::vector<int> r2{0, 1};
    const CSR m2{v2, c2, r2, num_cols};

    const std::vector<double> v3{1., 1., 1.};
    const std::vector<int> c3{1, 3, 4};
    const std::vector<int> r3{0, 3};
    const CSR m3{v3, c3, r3, num_cols};

    const std::vector<double> v4{0.5};
    const std::vector<int> c4{0};
    const std::vector<int> r4{0, 0, 1};
    const CSR m4{v4, c4, r4, num_cols};

    std::vector<double> initial(num_cols, 1.);
    std::vector<double> result(matrix.num_rows());
    std::vector<double> result1(m1.num_rows());
    std::vector<double> result2(m2.num_rows());
    std::vector<double> result3(m3.num_rows());
    std::vector<double> result4(m4.num_rows());

    const double c{0.85};
    pagerank::fixed::spmv(matrix, initial, result, c);
    pagerank::fixed::spmv(m1, initial, result1, c);
    pagerank::fixed::spmv(m2, initial, result2, c);
    pagerank::fixed::spmv(m3, initial, result3, c);
    pagerank::fixed::spmv(m4, initial, result4, c);

    print_vector(result, "result");
    print_vector(result1, "result1");
    print_vector(result2, "result2");
    print_vector(result3, "result3");
    print_vector(result4, "result4");
}

int main(int argc, char *argv[]) {
    const auto requested = MPI_THREAD_FUNNELED;
    int provided;
    MPI_Init_thread(&argc, &argv, requested, &provided);
    if (provided < requested) {
        std::cout << "No sufficient MPI multithreading support found\n";
        return 0;
    }

    const auto comm = MPI_COMM_WORLD;

    int rank, comm_size;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    std::cout << std::setprecision(7);
    std::mt19937 rng(std::random_device{}());

    unsigned n = std::stoi(argv[1]);
    double density = std::stod(argv[2]);
    std::cout << "size: " << n << "\n";
    std::cout << "density: " << density << "\n";
    std::cout << "------\n";

    std::vector<int> rowcnt;
    std::vector<int> start_row;
    get_rowcnt_start_row(comm, n , rowcnt, start_row);

    const CSR matrix = CSR::row_stochastic(n, density, rng);
    const CSR transposed = CSR::transpose(matrix);

    std::cout << "matrix:\n";
    matrix.print();
    std::cout << "------\n";
    std::cout << "transpose:\n";
    transposed.print();
    std::cout << "------\n";

    std::vector<double> initial(n, 1.);
    std::vector<double> result(n);
    const double c{0.85};
    const auto meta = pagerank::fixed::pagerank(transposed, initial, result, c, comm, rowcnt);

    std::cout << "Meta information\n\tconverged: " << meta.first << "\n\titerations: " << meta.second << "\n";
    print_vector(result, "result");
    std::cout << "------\n";
    /** #############################################################################################
      *  #############################################################################################
      */
    std::vector<std::uint16_t> initial_2;
    initial_2.reserve(n);
    for (size_t i{0}; i < initial.size(); ++i) {
        std::uint16_t val;
        seg_uint::write_2(&val, &initial[i]);
        initial_2.push_back(val);
    }
    std::vector<uint16_t> result_2_seg(n);
    const auto meta_2 = pagerank::seg::pagerank_2(transposed, initial_2, result_2_seg, c, comm, rowcnt);
    std::vector<double> result_2_dbl(n);
    for (size_t i{0}; i < result_2_seg.size(); ++i) {
        seg_uint::read_2(&result_2_seg[i], &result_2_dbl[i]);
    }

    std::cout << "Meta information (2 bytes)\n\tconverged: " << meta_2.first << "\n\titerations: " << meta_2.second << "\n";
    print_vector(result_2_dbl, "result_2");
    std::cout << "------\n";
    /** #############################################################################################
     *  #############################################################################################
     */
    std::vector<std::uint32_t > initial_4;
    initial_4.reserve(n);
    for (size_t i{0}; i < initial.size(); ++i) {
        std::uint32_t val;
        seg_uint::write_4(&val, &initial[i]);
        initial_4.push_back(val);
    }
    std::vector<std::uint32_t> result_4_seg(n);
    const auto meta_4 = pagerank::seg::pagerank_4(transposed, initial_4, result_4_seg, c, comm, rowcnt);
    std::vector<double> result_4_dbl(n);
    for (size_t i{0}; i < result_4_seg.size(); ++i) {
        seg_uint::read_4(&result_4_seg[i], &result_4_dbl[i]);
    }

    std::cout << "Meta information (4 bytes)\n\tconverged: " << meta_4.first << "\n\titerations: " << meta_4.second << "\n";
    print_vector(result_4_dbl, "result_4");
    /** #############################################################################################
     *  #############################################################################################
     */
    std::vector<std::uint16_t > initial_6;
    initial_6.resize(3 * n);
    for (size_t i{0}; i < initial.size(); ++i) {
        seg_uint::write_6(&initial_6[3 * i], &initial[i]);
    }
    std::vector<std::uint16_t> result_6_seg(3 * n);
    const auto meta_6 = pagerank::seg::pagerank_6(transposed, initial_6, result_6_seg, c, comm, rowcnt);
    std::vector<double> result_6_dbl(n);
    for (size_t i{0}; i < result_6_dbl.size(); ++i) {
        seg_uint::read_6(&result_6_seg[3 * i], &result_6_dbl[i]);
    }
    std::cout << "Meta information (6 bytes)\n\tconverged: " << meta_6.first << "\n\titerations: " << meta_6.second << "\n";
    print_vector(result_6_dbl, "result_6");//*/
    /** #############################################################################################
     *  #############################################################################################
     */

    normalize<1>(initial);
    std::cout << "\n######################################\n";
    std::cout << "##### SPMV result tests ##############\n";
    std::cout << "######################################\n";

    std::cout << "Test normal SpMV\n";
    {
        std::vector<double> one_iter(n);
        fixed::spmv(matrix, initial, one_iter);
        print_vector(one_iter, "one_iter");
        std::vector<double> two_iter(n);
        fixed::spmv(matrix, initial, two_iter);
        print_vector(two_iter, "two_iter");
    }
    std::cout << "------\n";
    std::cout << "\nTest pagerank SpMV\n";
    {
        std::vector<double> one_iter(n);
        pagerank::fixed::spmv(matrix, initial, one_iter, c);
        print_vector(one_iter, "one_iter");
        std::vector<double> two_iter(n);
        pagerank::fixed::spmv(matrix, one_iter, two_iter, c);
        print_vector(two_iter, "two_iter");
    }
    std::cout << "------\n";

    std::cout << std::hex;
    std::vector<std::uint16_t> spmv_test_2(n);
    pagerank::seg::spmv_2(matrix, initial_2, spmv_test_2, c);
    print_vector(initial_2, "initial_2");
    print_vector(spmv_test_2, "spmv_test_2");

    std::cout << "--------\n";
    std::vector<std::uint32_t> spmv_test_4(n);
    pagerank::seg::spmv_4(matrix, initial_4, spmv_test_4, c);
    print_vector(initial_4, "initial_4");
    print_vector(spmv_test_4, "spmv_test_4");

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
