//
// Created by khondar on 08.02.20.
//

#include <mpi.h>
#include <chrono>

#include "pi_benchmarks.h"
#include "matrix_formats/csr.hpp"
#include "power_iteration/poweriteration.h"
#include "communication.h"

void
iteration_counter(const std::vector<int> &sizes, const std::vector<double> &densities, const std::vector<double> &etas,
                  const unsigned max_iterations, const int num_tests) {
    const auto seed = std::random_device{}();
    std::cout << "seed: " << seed << " max_iterations: " << max_iterations << " num_tests: " << num_tests << "\n";
    std::mt19937 rng(seed);

    for (const auto size : sizes) {
        for (const auto d : densities) {
            for (const auto eta : etas) {
                std::cout << "parameters: " << size << " " << d << " " << eta << "\n";

                const std::vector<double> x(size, 1.);
                const std::vector<int> rowcnt{size};

                for (int i{0}; i < num_tests; ++i) {
                    const auto matrix = CSR::fixed_eta(size, d, eta, rng);
                    // test 2 4 6 8
                    {
                        unsigned avail_iterations = max_iterations;
                        std::vector<double> result_2;
                        const auto meta_2 = distributed::seg_uint::power_iteration_2(matrix, x, result_2,
                                                                                     MPI_COMM_WORLD, rowcnt,
                                                                                     avail_iterations);
                        avail_iterations -= meta_2.second;
                        std::vector<double> result_4;
                        const auto meta_4 = distributed::seg_uint::power_iteration_4(matrix, result_2, result_4,
                                                                                     MPI_COMM_WORLD, rowcnt,
                                                                                     avail_iterations);
                        avail_iterations -= meta_4.second;
                        std::vector<double> result_6;
                        const auto meta_6 = distributed::seg_uint::power_iteration_6(matrix, result_4, result_6,
                                                                                     MPI_COMM_WORLD, rowcnt,
                                                                                     avail_iterations);
                        avail_iterations -= meta_6.second;
                        std::vector<double> result;
                        const auto meta_8 = local::power_iteration(matrix, result_6, result, avail_iterations);

                        std::cout << "2_4_6_8 ";
                        if (meta_2.first && meta_4.first && meta_6.first && meta_8.first) {
                            std::cout << "all_converged ";
                        } else {
                            std::cout << "not_all_converged ";
                        }
                        std::cout << meta_2.second << " " << meta_4.second << " " << meta_6.second << " "
                                  << meta_8.second << "\n";
                    }

                    // test 4 6 8
                    {
                        unsigned avail_iterations = max_iterations;
                        std::vector<double> result_4;
                        const auto meta_4 = distributed::seg_uint::power_iteration_4(matrix, x, result_4,
                                                                                     MPI_COMM_WORLD, rowcnt,
                                                                                     avail_iterations);
                        avail_iterations -= meta_4.second;
                        std::vector<double> result_6;
                        const auto meta_6 = distributed::seg_uint::power_iteration_6(matrix, result_4, result_6,
                                                                                     MPI_COMM_WORLD, rowcnt,
                                                                                     avail_iterations);
                        avail_iterations -= meta_6.second;
                        std::vector<double> result;
                        const auto meta_8 = local::power_iteration(matrix, result_6, result, avail_iterations);

                        std::cout << "4_6_8 ";
                        if (meta_4.first && meta_6.first && meta_8.first) {
                            std::cout << "all_converged ";
                        } else {
                            std::cout << "not_all_converged ";
                        }
                        std::cout << meta_4.second << " " << meta_6.second << " " << meta_8.second << "\n";
                    }
                    // test 4 8
                    {
                        unsigned avail_iterations = max_iterations;
                        std::vector<double> result_4;
                        const auto meta_4 = distributed::seg_uint::power_iteration_4(matrix, x, result_4,
                                                                                     MPI_COMM_WORLD, rowcnt,
                                                                                     avail_iterations);
                        avail_iterations -= meta_4.second;
                        std::vector<double> result;
                        const auto meta_8 = local::power_iteration(matrix, result_4, result, avail_iterations);

                        std::cout << "4_8 ";
                        if (meta_4.first && meta_8.first) {
                            std::cout << "all_converged ";
                        } else {
                            std::cout << "not_all_converged ";
                        }
                        std::cout << meta_4.second << " " << meta_8.second << "\n";
                    }

                    // test 6 8
                    {
                        unsigned avail_iterations = max_iterations;
                        std::vector<double> result_6;
                        const auto meta_6 = distributed::seg_uint::power_iteration_6(matrix, x, result_6,
                                                                                     MPI_COMM_WORLD, rowcnt,
                                                                                     avail_iterations);
                        avail_iterations -= meta_6.second;
                        std::vector<double> result;
                        const auto meta_8 = local::power_iteration(matrix, result_6, result, avail_iterations);

                        std::cout << "6_8 ";
                        if (meta_6.first && meta_8.first) {
                            std::cout << "all_converged ";
                        } else {
                            std::cout << "not_all_converged";
                        }
                        std::cout << meta_6.second << " " << meta_8.second << "\n";
                    }

                    // test 8
                    {
                        std::vector<double> result;
                        const auto meta_8 = local::power_iteration(matrix, x, result, max_iterations);
                        std::cout << "8 ";
                        if (meta_8.first) {
                            std::cout << "all_converged ";
                        } else {
                            std::cout << "not_all_converged ";
                        }
                        std::cout << meta_8.second << "\n";
                    }
                }
            }
        }
    }
}

void
speedup_test(const int size, const double density, const double eta, const int max_iterations, const int num_tests,
             const std::vector<int> &rowcnt, std::vector<unsigned> &seg_timings, std::vector<unsigned> &fixed_timings,
             MPI_Comm comm) {
    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    std::mt19937 rng{std::random_device{}()};
    const int root = 0;

    const auto matrix = CSR::fixed_eta(size, density, eta, rng);
    const auto matrix_slice = distribute_matrix(matrix, comm, root);
    std::vector<double> x(matrix_slice.num_cols(), 1);

    double warmup_sum{0};
    for (int i{0}; i < 10; ++i) {
        x.at(0) = i + 1;

        MPI_Barrier(comm);

        int avail_iterations = max_iterations;
        std::vector<double> result_4;
        const auto meta_4 = distributed::seg_uint::power_iteration_4(matrix_slice, x, result_4, comm, rowcnt,
                                                                     avail_iterations);
        avail_iterations -= meta_4.second;
        std::vector<double> result;
        distributed::fixed::power_iteration(matrix_slice, result_4, result, comm, avail_iterations);
        warmup_sum += result.at(0);
    }
    for (int i{0}; i < num_tests; ++i) {
        x.at(0) = i + 1;

        MPI_Barrier(comm);

        using namespace std::chrono;
        const auto start = high_resolution_clock::now();
        int avail_iterations = max_iterations;
        std::vector<double> result_4;
        const auto meta_4 = distributed::seg_uint::power_iteration_4(matrix_slice, x, result_4, comm, rowcnt,
                                                                     avail_iterations);
        avail_iterations -= meta_4.second;
        std::vector<double> result;
        const auto meta_8 = distributed::fixed::power_iteration(matrix_slice, result_4, result, comm, avail_iterations);
        const auto stop = high_resolution_clock::now();

        if (meta_8.first) {
            seg_timings.push_back(duration_cast<milliseconds>(stop - start).count());
        }
    }

    for (int i{0}; i < 10; ++i) {
        x.at(0) = i + 1;

        MPI_Barrier(comm);

        int avail_iterations = max_iterations;
        std::vector<double> result;
        distributed::fixed::power_iteration(matrix_slice, x, result, comm, avail_iterations);
        warmup_sum += result.at(0);
    }
    for (int i{0}; i < num_tests; ++i) {
        x.at(0) = i + 1;

        MPI_Barrier(comm);

        using namespace std::chrono;
        const auto start = high_resolution_clock::now();
        int avail_iterations = max_iterations;
        std::vector<double> result;
        const auto meta = distributed::fixed::power_iteration(matrix_slice, x, result, comm, avail_iterations);
        const auto stop = high_resolution_clock::now();

        if (meta.first) {
            fixed_timings.push_back(duration_cast<milliseconds>(stop - start).count());
        }
    }

    if (rank > comm_size) {
        std::cout << "warmup_sum " << warmup_sum << "\n";
    }
}
