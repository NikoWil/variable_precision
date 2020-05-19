//
// Created by khondar on 14.04.20.
//

#include <algorithm>
#include <array>
#include <iomanip>

#include "pagerank_test.h"
#include "matrix_formats/csr.hpp"
#include "power_iteration/pagerank.h"
#include "segmentation/seg_uint.h"
#include "communication.h"

void test_convergence() {
    // TODO: make sure memory requitements remain <= 60 GB
    //  => 64 GB = 2 ** 36 bytes => < 16 bytes per element
    //  => we can fit 2 ** 32 elements
    //  => size^2 * density <= 2 ** 16 as requirement
    /*constexpr std::array<int, 7> sizes{64 * 1014, 128 * 1014, 256 * 1014, 512 * 1014, 1024 * 1014, 2 * 1014 * 1024,
                                       4 * 1024 * 1024};
    constexpr std::array<double, 5> densities{1. / 64., 1. / 128., 1. / 256., 1. / 512., 1. / 1024};//*/

    // TODO: generate test that can run on the laptop alone (< 6 GB?)
    //  => 4 GB = 2^32 bytes
    //  => we can fit 2 ^ 28 elements
    //  => size^2 * density <= 2^14
    constexpr std::array<int, 5> sizes{64 * 1014, 128 * 1014, 256 * 1014, 512 * 1014, 1024 * 1014};
    constexpr std::array<double, 5> densities{1. / 64., 1. / 128., 1. / 256., 1. / 512., 1. / 1024};

    constexpr double c{0.85};

    constexpr unsigned num_tests{3};
    constexpr int iteration_limit{200};

    for (const auto s : sizes) {
        for (const auto d : densities) {
            std::cout << "size: " << s << ", density: " << d << "\n";
            for (unsigned k{0}; k < num_tests; ++k) {

                const unsigned seed{std::random_device{}()};
                std::mt19937 rng{seed};
                auto matrix = CSR::row_stochastic(s, d, rng);
                matrix = CSR::transpose(matrix);

                const std::vector<double> initial(s, 1.);
                std::vector<double> result(s);
                const auto meta = pagerank::local::pagerank(matrix, initial, result, c, iteration_limit);

                std::cout << "\tseed:       " << seed << "\n";
                std::cout << "\tconverged:  " << meta.first << "\n";
                std::cout << "\titerations: " << meta.second << "\n";
                std::cout << "\t------\n";
            }
            std::cout << "\n";
        }
    }
}

void test_precision_levels(unsigned n, double density, const std::vector<int> &rowcnt) {
    const auto comm = MPI_COMM_WORLD;

    int rank, comm_size;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    std::cout << std::setprecision(7);
    std::mt19937 rng(std::random_device{}());

    if (rank == 0) {
        std::cout << "size: " << n << "\n";
        std::cout << "density: " << density << "\n";
        std::cout << "------\n";
    }

    const CSR matrix = CSR::row_stochastic(n, density, rng);
    const CSR transposed = CSR::transpose(matrix);
    const CSR transposed_slice = distribute_matrix(transposed, comm, 0);

    if (rank == 0) {
        std::cout << "matrix:\n";
        matrix.print();
        std::cout << "------\n";
        std::cout << "transpose:\n";
        transposed.print();
        std::cout << "------\n";
    }

    const double c{0.85};

    if (rank == 0) {
        std::vector<double> initial_local(n, 1.);
        std::vector<double> result_local(n);

        const auto meta = pagerank::local::pagerank(transposed, initial_local, result_local, c);
        std::cout << "Meta information (local)\n\tconverged: " << meta.first << "\n\titerations: " << meta.second
                  << "\n";
        print_vector(result_local, "result_l");
        std::cout << "------\n";
    }
    MPI_Barrier(comm);

    std::vector<double> initial(n, 1.);
    std::vector<double> result(n);
    const auto meta = pagerank::fixed::pagerank(transposed_slice, initial, result, c, comm, rowcnt);

    if (rank == 0) {
        std::cout << "Meta information (fixed precision)\n\tconverged: " << meta.converged << "\n\titerations: "
                  << meta.used_iterations << "\n";
        print_vector(result, "result");
        std::cout << "------\n";
    }
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
    const auto meta_2 = pagerank::seg::pagerank_2(transposed_slice, initial_2, result_2_seg, c, comm, rowcnt);
    std::vector<double> result_2_dbl(n);
    for (size_t i{0}; i < result_2_seg.size(); ++i) {
        seg_uint::read_2(&result_2_seg[i], &result_2_dbl[i]);
    }

    if (rank == 0) {
        std::cout << "Meta information (2 bytes)\n\tconverged: " << meta_2.converged << "\n\titerations: "
                  << meta_2.used_iterations << "\n";
        print_vector(result_2_dbl, "result_2");
        std::cout << "------\n";
    }
    /** #############################################################################################
     *  #############################################################################################
     */
    std::vector<std::uint32_t> initial_4;
    initial_4.reserve(n);
    for (size_t i{0}; i < initial.size(); ++i) {
        std::uint32_t val;
        seg_uint::write_4(&val, &initial[i]);
        initial_4.push_back(val);
    }
    std::vector<std::uint32_t> result_4_seg(n);
    const auto meta_4 = pagerank::seg::pagerank_4(transposed_slice, initial_4, result_4_seg, c, comm, rowcnt);
    std::vector<double> result_4_dbl(n);
    for (size_t i{0}; i < result_4_seg.size(); ++i) {
        seg_uint::read_4(&result_4_seg[i], &result_4_dbl[i]);
    }

    if (rank == 0) {
        std::cout << "Meta information (4 bytes)\n\tconverged: " << meta_4.converged << "\n\titerations: "
                  << meta_4.used_iterations << "\n";
        print_vector(result_4_dbl, "result_4");
        std::cout << "------\n";
    }
    /** #############################################################################################
     *  #############################################################################################
     */
    std::vector<std::uint16_t> initial_6(3 * n);
    for (size_t i{0}; i < initial.size(); ++i) {
        seg_uint::write_6(&initial_6[3 * i], &initial[i]);
    }
    std::vector<std::uint16_t> result_6_seg(3 * n);
    const auto meta_6 = pagerank::seg::pagerank_6(transposed_slice, initial_6, result_6_seg, c, comm, rowcnt);
    std::vector<double> result_6_dbl(n);
    for (size_t i{0}; i < result_6_dbl.size(); ++i) {
        seg_uint::read_6(&result_6_seg[3 * i], &result_6_dbl[i]);
    }

    if (rank == 0) {
        std::cout << "Meta information (6 bytes)\n\tconverged: " << meta_6.converged << "\n\titerations: "
                  << meta_6.used_iterations << "\n";
        print_vector(result_6_dbl, "result_6");
        std::cout << "------\n";
    }
    /** #############################################################################################
     *  #############################################################################################
     */
    std::vector<double> result_variable(n);
    const auto meta_variable = pagerank::variable::pagerank_2_4_6_8(transposed_slice, initial, result_variable, c, comm,
                                                                    rowcnt);

    if (rank == 0) {
        std::cout << "meta information (2, 4, 6, 8):\n";
        for (const auto &m : meta_variable) {
            std::cout << "\tconverged: " << m.converged << "\titerations: " << m.used_iterations << "\n";
        }
        print_vector(result_variable, "result_variable");
    }
}

void pr_performance_test(MPI_Comm comm) {
    int comm_size, rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);
    constexpr int root{0};

    /*constexpr std::array<std::uint32_t, 7> sizes{1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024};
    constexpr std::array<double, 5> densities{1. / 64., 1. / 128., 1. / 256., 1. / 512., 1. / 1024.};
    constexpr double c{0.85};

    constexpr std::uint32_t num_tests{10};
    constexpr std::uint32_t num_samples{30};*/

    constexpr std::array<std::uint32_t, 2> sizes{10, 20};
    constexpr std::array<double, 2> densities{0.5, 0.6};
    constexpr double c{0.85};

    constexpr std::uint32_t num_tests{2};
    constexpr std::uint32_t num_samples{2};

    if (rank == root) {
        std::cout << "num_tests " << num_tests << "\n";
        std::cout << "num_samples " << num_samples << "\n";
    }
    MPI_Barrier(comm);

    if (rank == root) {
        std::cout << "num_tests " << num_tests << "\n";
        std::cout << "num_samples " << num_samples << "\n";
    }
    MPI_Barrier(comm);

    double sum{0.};

    for (const auto s : sizes) {
        for (const auto d : densities) {
            for (std::uint32_t test_num{0}; test_num < num_tests; ++test_num) {
                // generate a new matrix
                const std::uint32_t seed{std::random_device{}()};
                std::mt19937 rng{seed};
                const auto matrix = CSR::row_stochastic(s, d, rng);
                const auto transposed = CSR::transpose(matrix);
                const auto matrix_slice = distribute_matrix(transposed, comm, root);

                std::vector<int> rowcnt, start_row;
                get_rowcnt_start_row(comm, s, rowcnt, start_row);

                std::vector<pagerank::pr_meta> meta_fixed;
                meta_fixed.reserve(num_samples);
                std::vector<std::array<pagerank::pr_meta, 4>> meta_2_4_6_8;
                meta_2_4_6_8.reserve(num_samples);

                std::vector<double> initial(s, 1.);
                std::vector<double> result(s);
                for (std::uint32_t sample{0}; sample < num_samples; ++sample) {
                    // perform pagerank, fixed precision
                    const auto meta = pagerank::fixed::pagerank(matrix_slice, initial, result, c, comm, rowcnt);
                    meta_fixed.push_back(std::move(meta));

                    sum += result.at(0);
                    initial.at(0)++;
                }

                for (std::uint32_t sample{0}; sample < num_samples; ++sample) {
                    // perform pagerank, variable precision
                    const auto meta = pagerank::variable::pagerank_2_4_6_8(matrix_slice, initial, result, c, comm, rowcnt);
                    meta_2_4_6_8.push_back(std::move(meta));

                    sum += result.at(0);
                    initial.at(0)++;
                }

                //TODO: print meta information somehow?
                if (rank == root) {
                    std::cout << "seed " << seed << "\n";

                    for (const auto& m : meta_fixed) {
                        pagerank::print_fixed(m);
                    }
                    for (const auto &m : meta_2_4_6_8) {
                        pagerank::print_2_4_6_8(m);
                    }

                    std::cout << "------\n\n";
                }
            }
        }
    }
    if (rank == 0) {
        std::cout << "sum " << sum << "\n";
    }
}
