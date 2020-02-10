//
// Created by khondar on 08.02.20.
//

#include <mpi.h>

#include "pi_benchmarks.h"
#include "matrix_formats/csr.hpp"
#include "linalg/power_iteration/poweriteration.h"

void
iteration_counter(const std::vector<int> &sizes, const std::vector<double> &densities, const std::vector<double> &etas,
                  const int num_tests) {
    const auto seed = std::random_device{}();
    std::cout << "seed: " << seed << "\n";
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
                        std::vector<double> result_2;
                        const auto meta_2 = distributed::seg_uint::power_iteration_2(matrix, x, result_2, MPI_COMM_WORLD, rowcnt);
                        std::vector<double> result_4;
                        const auto meta_4 = distributed::seg_uint::power_iteration_4(matrix, result_2, result_4, MPI_COMM_WORLD, rowcnt);
                        std::vector<double> result_6;
                        const auto meta_6 = distributed::seg_uint::power_iteration_6(matrix, result_4, result_6, MPI_COMM_WORLD, rowcnt);
                        std::vector<double> result;
                        const auto meta_8 = local::power_iteration(matrix, result_6, result);

                        std::cout << "2_4_6_8 ";
                        if (meta_2.first && meta_4.first && meta_6.first && meta_8.first) {
                            std::cout << "all_converged ";
                        } else {
                            std::cout << "not_all_converged ";
                        }
                        std::cout << meta_2.second << " " << meta_4.second << " " << meta_6.second << " " << meta_8.second << "\n";
                    }

                    // test 4 6 8
                    {
                        std::vector<double> result_4;
                        const auto meta_4 = distributed::seg_uint::power_iteration_4(matrix, x, result_4, MPI_COMM_WORLD, rowcnt);
                        std::vector<double> result_6;
                        const auto meta_6 = distributed::seg_uint::power_iteration_6(matrix, result_4, result_6, MPI_COMM_WORLD, rowcnt);
                        std::vector<double> result;
                        const auto meta_8 = local::power_iteration(matrix, result_6, result);

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
                        std::vector<double> result_4;
                        const auto meta_4 = distributed::seg_uint::power_iteration_4(matrix, x, result_4, MPI_COMM_WORLD, rowcnt);
                        std::vector<double> result;
                        const auto meta_8 = local::power_iteration(matrix, result_4, result);

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
                        std::vector<double> result_6;
                        const auto meta_6 = distributed::seg_uint::power_iteration_6(matrix, x, result_6, MPI_COMM_WORLD, rowcnt);
                        std::vector<double> result;
                        const auto meta_8 = local::power_iteration(matrix, result_6, result);

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
                        const auto meta_8 = local::power_iteration(matrix, x, result);
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