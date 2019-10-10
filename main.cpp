#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <random>
#include <vector>

#include "communication.h"
#include "matrix_formats/csr.hpp"
#include "performance_tests.h"
#include "power_iteration.h"
#include "segmentation_char/segmentation_char.h"
#include "util/util.hpp"

void power_iteration_tests(std::mt19937 rng, unsigned rank, unsigned comm_size);

void measure_performance(std::mt19937 rng, unsigned rank, unsigned comm_size);

int main(int argc, char* argv[]) {
  const auto requested = MPI_THREAD_FUNNELED;
  int provided;
  MPI_Init_thread(&argc, &argv, requested, &provided);
  if (provided < requested) {
    std::cout << "No sufficient MPI multithreading support found\n";
    return 0;
  }

  std::mt19937 rng{std::random_device{}()};

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if (rank == 0) {
    const auto matrix = CSR::random(5, 5, 0.5, rng);
    std::cout << matrix.num_values() << std::endl;
    matrix.print();
  }

  MPI_Finalize();
  return 0;
}

void power_iteration_tests(std::mt19937 rng, unsigned rank, unsigned comm_size) {
  const int matrix_rows = 4;
  const double density = 0.5;
  if (rank == 0) {
    std::cout << "matrix rows: " << matrix_rows << "\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  CSR matrix = CSR::unit(0);
  if (rank == 0) {
    matrix = CSR::diagonally_dominant(matrix_rows, density, rng);
  }
  if (rank == 0) {
    std::cout << "matrix generated" << std::endl;
  }
  auto matrix_slice = distribute_matrix(matrix, MPI_COMM_WORLD, 0);

  std::uniform_real_distribution<> distribution{1, 100};
  std::vector<double> x(matrix_slice.num_cols());
  if (rank == 0) {
    for (unsigned i = 0; i < matrix_slice.num_cols(); ++i) {
      x.at(i) = distribution(rng);
    }
  }
  auto square_sum = std::accumulate(x.begin(), x.end(), 0., [](double curr, double d){ return curr + d * d; });
  auto norm_fac = sqrt(square_sum);
  std::for_each(x.begin(), x.end(), [norm_fac](double& d) { d /= norm_fac; });

  MPI_Bcast(x.data(), static_cast<int>(matrix_slice.num_cols()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<int> rowcnt;
  for (unsigned i = 0u; i < comm_size; ++i) {
    unsigned start = (matrix_rows * i) / comm_size;
    unsigned end = (matrix_rows * (i + 1)) / comm_size;

    rowcnt.push_back(end - start);
  }

  if (rank == 0) {
    auto result_simple = power_iteration(matrix, x);
    std::cout << "Simple Power Iteration\n";
    print_vector(result_simple.first, "simple power iteration");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int end = 4;
  std::vector<seg::Double_slice<0, end>> x_slice;
  for (const auto v : x) {
    x_slice.emplace_back(v);
  }
  auto seg_result = power_iteration_segmented<end>(matrix_slice, x_slice, rowcnt, MPI_COMM_WORLD);
  if (rank == 0) {
    auto res_vector = std::get<0>(seg_result);
    auto iter_count = std::get<1>(seg_result);
    auto done = std::get<2>(seg_result);
    std::cout << "Segmented:\n";
    std::cout << "Result: \n";
    for (const auto& s : res_vector) {
      std::cout << s.to_double() << " ";
    }
    std::cout << std::endl;
    std::cout << "Iter_count: " << iter_count << ", done: " << done << "\n\n";
  }

  auto var_result = power_iteration_variable(matrix_slice, x, rowcnt, MPI_COMM_WORLD);
  if (rank == 0) {
    print_vector(var_result, "var_result");
  }
}

void measure_performance(std::mt19937 rng, unsigned rank, unsigned comm_size) {
  if (rank == 0) {
    std::cout << "<matrix_size> <density_fac> <index> <mode> <time>\n\n";
  }
  for (unsigned matrix_size = 1u << 15u; matrix_size <= 1u << 20u; matrix_size <<= 1u) {
    for (unsigned density_fac = 2u; density_fac < 3u; ++density_fac) {
      double density = density_fac * 0.1;

      for (unsigned index = 0u; index < 30u; ++index) {
        CSR matrix = CSR::unit(0);
        if (rank == 0) {
          matrix = CSR::diagonally_dominant(matrix_size, density, rng);
        }
        auto matrix_slice = distribute_matrix(matrix, MPI_COMM_WORLD, 0);

        std::uniform_real_distribution<> distribution{1, 100};
        std::vector<double> x(matrix_slice.num_cols());
        if (rank == 0) {
          for (unsigned i = 0; i < matrix_slice.num_cols(); ++i) {
            x.at(i) = distribution(rng);
            auto square_sum = std::accumulate(x.begin(), x.end(), 0., [](double curr, double d){ return curr + d * d; });
            auto norm_fac = sqrt(square_sum);
            std::for_each(x.begin(), x.end(), [norm_fac](double& d) { d /= norm_fac; });
          }
        }

        MPI_Bcast(x.data(), static_cast<int>(matrix_slice.num_cols()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<int> rowcnt;
        for (unsigned i = 0; i < comm_size; ++i) {
          unsigned start = (matrix_size * i) / comm_size;
          unsigned end = (matrix_size * (i + 1)) / comm_size;

          rowcnt.push_back(end - start);
        }

        for (unsigned k = 0u; k < 20u; ++k) {
          auto start = std::chrono::high_resolution_clock::now();
          auto result = power_iteration(matrix_slice, x, rowcnt, MPI_COMM_WORLD);
          auto end = std::chrono::high_resolution_clock::now();

          int precision_switch = std::get<1>(result);
          unsigned iterations = std::get<2>(result);
          bool done = std::get<3>(result);

          if (rank == 0) {
            std::cout << matrix_size << " "
                      << density_fac << " "
                      << index << " "
                      << "VARIABLE "
                      << precision_switch << " "
                      << iterations << " "
                      << (done ? "true " : "false ")
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                      << std::endl;
          }
        }

        for (unsigned k = 0u; k < 20u; ++k) {
          auto start = std::chrono::high_resolution_clock::now();
          auto result = power_iteration_fixed(matrix_slice, x, rowcnt, MPI_COMM_WORLD);
          auto end = std::chrono::high_resolution_clock::now();

          unsigned iterations = std::get<1>(result);
          bool done = std::get<2>(result);

          if (rank == 0) {
            std::cout << matrix_size << " "
                      << density_fac << " "
                      << index << " "
                      << "FIXED "
                      << iterations << " "
                      << (done ? "true " : "false ")
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                      << std::endl;
          }
        }
      }
    }
  }
}

/*
 * +-------+
 * | 1   2 |
 * |   3 4 |
 * | 2 4   |
 * +-------+
 */
CSR symmetric_matrix_1(std::mt19937 rng) {
  std::vector<double> values(6);
  const std::vector<int> colidx{0, 2, 1, 2, 0, 1};
  const std::vector<int> rowptr{0, 2, 4, 6};
  std::uniform_real_distribution<> distribution{-100, 100};

  const auto v1 = distribution(rng);
  const auto v2 = distribution(rng);
  const auto v3 = distribution(rng);
  const auto v4 = distribution(rng);

  values.at(0) = v1;
  values.at(1) = v2;
  values.at(2) = v3;
  values.at(3) = v4;
  values.at(4) = v2;
  values.at(5) = v4;

  return CSR{values, colidx, rowptr, 3};
}

/*
 * +-------+
 * |   1   |
 * | 1 2   |
 * |     3 |
 * +-------+
 */
CSR symmetric_matrix_2(std::mt19937 rng) {
  std::vector<double> values(4);
  std::vector<int> colidx{1, 0, 1, 2};
  std::vector<int> rowptr{0, 1, 3, 4};

  std::uniform_real_distribution<> distribution{-100, 100};

  const auto v1 = distribution(rng);
  const auto v2 = distribution(rng);
  const auto v3 = distribution(rng);

  values.at(0) = v1;
  values.at(1) = v1;
  values.at(2) = v2;
  values.at(3) = v3;

  return CSR{values, colidx, rowptr, 3};
}

/*
 * +---------+
 * | 1 2     |
 * | 2       |
 * |     3 4 |
 * |     4   |
 * +---------+
 */
CSR symmetric_matrix_3(std::mt19937 rng) {
  std::vector<double> values(6);
  std::vector<int> colidx{0, 1, 0, 2, 3, 2};
  std::vector<int> rowptr{0, 2, 3, 5, 6};

  std::uniform_real_distribution<> distribution{-100, 100};

  const auto v1 = distribution(rng);
  const auto v2 = distribution(rng);
  const auto v3 = distribution(rng);
  const auto v4 = distribution(rng);

  values.at(0) = v1;
  values.at(1) = v2;
  values.at(2) = v2;
  values.at(3) = v3;
  values.at(4) = v4;
  values.at(5) = v4;

  return CSR{values, colidx, rowptr, 4};
}

/*
 * +---------+
 * | 1     2 |
 * |   3 4   |
 * |   4 5   |
 * | 2     6 |
 * +---------+
 */
CSR symmetric_matrix_4(std::mt19937 rng) {
  std::vector<double> values(8);
  std::vector<int> colidx{0, 3, 1, 2, 1, 2, 0, 3};
  std::vector<int> rowptr{0, 2, 4, 6, 8};

  std::uniform_real_distribution<> distribution{-100, 100};

  const auto v1 = distribution(rng);
  const auto v2 = distribution(rng);
  const auto v3 = distribution(rng);
  const auto v4 = distribution(rng);
  const auto v5 = distribution(rng);
  const auto v6 = distribution(rng);

  values.at(0) = v1;
  values.at(1) = v2;
  values.at(2) = v3;
  values.at(3) = v4;
  values.at(4) = v4;
  values.at(5) = v5;
  values.at(6) = v2;
  values.at(7) = v6;

  return CSR{values, colidx, rowptr, 4};
}
