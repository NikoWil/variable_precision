//
// Created by niko on 9/24/19.
//

#ifndef CODE_PERFORMANCE_TESTS_H
#define CODE_PERFORMANCE_TESTS_H

#include <chrono>
#include <cmath>
#include <iostream>

#include "matrix_formats/csr.hpp"
#include "power_iteration.h"
#include "seg_char.h"
#include "spmv.h"

namespace {

template <int n>
std::vector<seg::Double_slice<0, n>> to_slice_vector(const std::vector<double> &v) {
  std::vector<seg::Double_slice<0, n>> new_vec;
  new_vec.reserve(v.size());

  for (const auto e : v) {
    new_vec.emplace_back(e);
  }

  return new_vec;
}

void print_result(int slice_length, unsigned width, unsigned height,
                  double density, unsigned num_values, unsigned time) {

  std::cout << slice_length << " " << width << " " << height << " " << density
            << " " << num_values << " " << time << "\n";
}

template <int end_idx>
void time_spmv_slice(const CSR &matrix, const std::vector<double> &x,
                double density, unsigned num_tests, unsigned warmup_iterations,
                std::mt19937 rng) {
  using slice_type = seg::Double_slice<0, end_idx>;
  auto slice_vec = to_slice_vector<end_idx>(x);
  std::vector<slice_type> result_vec(matrix.num_rows());
  std::uniform_int_distribution<> index_distrib(0, matrix.num_rows() - 1);

  double sum{0};
  for (unsigned i{0}; i < warmup_iterations; ++i) {
    spmv(matrix, slice_vec, result_vec);
    sum += result_vec.at(index_distrib(rng)).to_double();
  }
  for (unsigned i{0}; i < num_tests; ++i) {
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    spmv(matrix, slice_vec, result_vec);
    auto end = high_resolution_clock::now();

    sum += result_vec.at(index_distrib(rng)).to_double();
    print_result(
        end_idx + 1, matrix.num_cols(), matrix.num_rows(), density,
        matrix.num_values(), duration_cast<nanoseconds>(end - start).count());
  }
  std::cout << "IGNORE sum " << sum << std::endl;
  std::cout << "TEST RUN DONE\n";
}

}

namespace perf_test {

void spmv_single_node() {
  std::mt19937 rng{std::random_device{}()};
  std::uniform_real_distribution<> val_distrib(0., 100'000);

  constexpr unsigned min_width = 1u << 4u;
  constexpr unsigned max_width = (1u << 15u) + 1;
  constexpr unsigned min_height = 1u << 4u;
  constexpr unsigned max_height = (1u << 15u) + 1;
  constexpr unsigned min_density_fac = 1;
  constexpr unsigned max_density_fac = 5 + 1;
  constexpr unsigned num_tests = 50;
  constexpr unsigned warmup_iterations = 5;

  std::cout
      << "<slice_length> <width> <height> <density> <num_values> <time (ns)>\n";
  for (unsigned width = min_width; width < max_width; width *= 2) {
    for (unsigned height = min_height; height < max_height; height *= 2) {
      std::uniform_int_distribution<> index_distrib(0, height - 1);

      std::vector<double> x;
      for (unsigned i = 0; i < width; ++i) {
        x.push_back(val_distrib(rng));
      }

      double sum{0};
      for (unsigned density_fac = min_density_fac;
           density_fac < max_density_fac; ++density_fac) {
        const auto density = density_fac * 0.1;
        CSR matrix = CSR::random(width, height, density, rng);

        for (unsigned i{0}; i < warmup_iterations; ++i) {
          auto result = matrix.spmv(x);
          sum += result.at(index_distrib(rng));
        }
        for (unsigned i{0}; i < num_tests; ++i) {
          using namespace std::chrono;

          auto start = high_resolution_clock::now();
          auto result = matrix.spmv(x);
          auto end = high_resolution_clock::now();

          sum += result.at(index_distrib(rng));
          print_result(-1, width, height, density, matrix.num_values(),
                       duration_cast<nanoseconds>(end - start).count());
        }
        std::cout << "TEST RUN DONE\n";

        time_spmv_slice<0>(matrix, x, density, num_tests, warmup_iterations,
                           rng);
        time_spmv_slice<1>(matrix, x, density, num_tests, warmup_iterations,
                           rng);
        time_spmv_slice<2>(matrix, x, density, num_tests, warmup_iterations,
                           rng);
        time_spmv_slice<3>(matrix, x, density, num_tests, warmup_iterations,
                           rng);
        time_spmv_slice<4>(matrix, x, density, num_tests, warmup_iterations,
                           rng);
        time_spmv_slice<5>(matrix, x, density, num_tests, warmup_iterations,
                           rng);
        time_spmv_slice<6>(matrix, x, density, num_tests, warmup_iterations,
                           rng);
        time_spmv_slice<7>(matrix, x, density, num_tests, warmup_iterations,
                           rng);
      }
      std::cout << "IGNORE sum " << sum << std::endl;
    }
  }
}

void power_iteration_segmented(MPI_Comm comm) {
  std::mt19937 rng{std::random_device{}()};

  int comm_size, rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &rank);

  const auto max_size =
      (1u << 15u) * static_cast<unsigned>(sqrt(comm_size)) / 4;
  const auto min_size = static_cast<unsigned>(max_size / 64);
  constexpr unsigned min_density_fac = 1;
  constexpr unsigned max_density_fac = 5 + 1;
  constexpr unsigned num_tests = 15;
  constexpr unsigned iter_limit = 500;
  constexpr unsigned num_instances = 5;
  constexpr unsigned broken_limit{10};

  const auto print_fixed =
      [](const std::string &mode,
         const std::tuple<std::vector<double>, unsigned, bool> &result,
         const auto time) {
        std::cout << mode << " " << std::get<1>(result) << " "
                  << std::get<2>(result) << " " << time << "\n";
      };

  const auto print_variable =
      [](const std::string &mode,
         const std::tuple<std::vector<double>, std::vector<unsigned>,
                          std::vector<bool>> &result,
         const auto time) {
        std::cout << mode << " ";
        for (const auto e : std::get<1>(result)) {
          std::cout << e << " ";
        }
        for (const auto e : std::get<2>(result)) {
          std::cout << e << " ";
        }
        std::cout << time << "\n";
      };

  for (auto size{min_size}; size < max_size + 1; size *= 2) {
    std::vector<int> rowcnt;
    for (int i{0}; i < comm_size; ++i) {
      unsigned start = (size * i) / comm_size;
      unsigned end = (size * (i + 1)) / comm_size;
      rowcnt.push_back(end - start);
    }

    std::vector<int> start_row{0};
    for (size_t i{0}; i < rowcnt.size(); ++i) {
      const auto next = rowcnt.at(i) + start_row.back();
      start_row.push_back(next);
    }

    for (auto density_fac{min_density_fac}; density_fac < max_density_fac;
         ++density_fac) {

      unsigned instance_cnt{0};
      unsigned broken_cnt{0};
      while (instance_cnt < num_instances && broken_cnt < broken_limit) {
        bool broken = false;
        // Generate and distribute initial guess
        std::uniform_real_distribution<> distribution{1, 100};
        std::vector<double> x(size);
        if (rank == 0) {
          for (size_t i = 0; i < x.size(); ++i) {
            x.at(i) = distribution(rng);
          }
          auto square_sum = std::accumulate(
              x.begin(), x.end(), 0.,
              [](double curr, double d) { return curr + d * d; });
          auto norm_fac = sqrt(square_sum);
          std::for_each(x.begin(), x.end(),
                        [norm_fac](double &d) { d /= norm_fac; });
        }
        MPI_Bcast(x.data(), x.size(), MPI_DOUBLE, 0, comm);

        // Generate and distribute matrix
        const double density = 0.1 * density_fac;
        const auto &matrix = CSR::diagonally_dominant_slice(
            size, density, rng, start_row.at(rank), start_row.at(rank + 1) - 1);

        // Do the testing
        if (rank == 0) {
          std::cout << "parameters: " << size << " " << density << " "
                    << matrix.num_values() << " " << instance_cnt << "\n";
        }

        for (unsigned i{0}; i < num_tests; ++i) {
          using namespace std::chrono;

          auto start_fixed = high_resolution_clock::now();
          const auto result_fixed =
              fixed::power_iteration(matrix, x, rowcnt, comm, iter_limit);
          auto end_fixed = high_resolution_clock::now();

          auto start_1 = high_resolution_clock::now();
          const auto result_1 = variable::power_iteration_eigth(
              matrix, x, rowcnt, comm, iter_limit);
          auto end_1 = high_resolution_clock::now();

          auto start_2 = high_resolution_clock::now();
          const auto result_2 = variable::power_iteration_quarter(
              matrix, x, rowcnt, comm, iter_limit);
          auto end_2 = high_resolution_clock::now();

          auto start_3 = high_resolution_clock::now();
          const auto result_3 = variable::power_iteration_half(
              matrix, x, rowcnt, comm, iter_limit);
          auto end_3 = high_resolution_clock::now();

          if (!std::get<2>(result_fixed) || !std::get<2>(result_1).back() ||
              !std::get<2>(result_2).back() || !std::get<2>(result_3).back()) {
            if (rank == 0) {
              std::cout << "Breaking\n\n";
            }
            broken = true;
            ++broken_cnt;
            break;
          } else {
            if (rank == 0) {
              print_fixed(
                  "fixed", result_fixed,
                  duration_cast<nanoseconds>(end_fixed - start_fixed).count());
              print_variable(
                  "eigths", result_1,
                  duration_cast<nanoseconds>(end_1 - start_1).count());
              print_variable(
                  "quarter", result_2,
                  duration_cast<nanoseconds>(end_2 - start_2).count());
              print_variable(
                  "half", result_3,
                  duration_cast<nanoseconds>(end_3 - start_3).count());
              std::cout << std::endl;
            }
          }
        }
        if (!broken) {
          ++instance_cnt;
        }
      }
    }
  }
}

void power_iteration_segmented(double density, MPI_Comm comm) {
  std::mt19937 rng{std::random_device{}()};

  int comm_size, rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &rank);

  // assumes that there is 64GB of memory per node available
  const auto max_size = static_cast<unsigned>(
      0.8 * (1u << 17u) * sqrt(comm_size) * sqrt(1. / (3. * density)));
  const auto min_size = max_size / 64;
  constexpr unsigned num_tests = 15;
  constexpr unsigned iter_limit = 500;
  constexpr unsigned num_instances = 5;
  constexpr unsigned broken_limit{10};

  const auto print_fixed =
      [](const std::string &mode,
         const std::tuple<std::vector<double>, unsigned, bool> &result,
         const auto time) {
        std::cout << mode << " " << std::get<1>(result) << " "
                  << std::get<2>(result) << " " << time << "\n";
      };

  const auto print_variable =
      [](const std::string &mode,
         const std::tuple<std::vector<double>, std::vector<unsigned>,
                          std::vector<bool>> &result,
         const auto time) {
        std::cout << mode << " ";
        for (const auto e : std::get<1>(result)) {
          std::cout << e << " ";
        }
        for (const auto e : std::get<2>(result)) {
          std::cout << e << " ";
        }
        std::cout << time << "\n";
      };

  for (auto size{min_size}; size < max_size + 1; size *= 2) {
    std::vector<int> rowcnt;
    for (int i{0}; i < comm_size; ++i) {
      unsigned start = (size * i) / comm_size;
      unsigned end = (size * (i + 1)) / comm_size;
      rowcnt.push_back(end - start);
    }

    std::vector<int> start_row{0};
    for (size_t i{0}; i < rowcnt.size(); ++i) {
      const auto next = rowcnt.at(i) + start_row.back();
      start_row.push_back(next);
    }

    unsigned instance_cnt{0};
    unsigned broken_cnt{0};
    while (instance_cnt < num_instances && broken_cnt < broken_limit) {
      bool broken = false;
      // Generate and distribute initial guess
      std::uniform_real_distribution<> distribution{1, 100};
      std::vector<double> x(size);
      if (rank == 0) {
        for (size_t i = 0; i < x.size(); ++i) {
          x.at(i) = distribution(rng);
        }
        auto square_sum =
            std::accumulate(x.begin(), x.end(), 0.,
                            [](double curr, double d) { return curr + d * d; });
        auto norm_fac = sqrt(square_sum);
        std::for_each(x.begin(), x.end(),
                      [norm_fac](double &d) { d /= norm_fac; });
      }
      MPI_Bcast(x.data(), x.size(), MPI_DOUBLE, 0, comm);

      // Generate and distribute matrix
      const auto &matrix = CSR::diagonally_dominant_slice(
          size, density, rng, start_row.at(rank), start_row.at(rank + 1) - 1);

      // Do the testing
      if (rank == 0) {
        std::cout << "parameters: " << size << " " << density << " "
                  << matrix.num_values() << " " << instance_cnt << "\n";
      }

      std::array<bool, 4> converge{true, true, true, true};
      for (unsigned i{0}; i < num_tests; ++i) {
        using namespace std::chrono;

        if (converge[0]) {
          auto start_fixed = high_resolution_clock::now();
          const auto result_fixed =
              fixed::power_iteration(matrix, x, rowcnt, comm, iter_limit);
          auto end_fixed = high_resolution_clock::now();
          converge[0] = std::get<2>(result_fixed);
          if (rank == 0 && converge[0]) {
            print_fixed(
                "fixed", result_fixed,
                duration_cast<nanoseconds>(end_fixed - start_fixed).count());
          }
        }

        if (converge[1]) {
          auto start_1 = high_resolution_clock::now();
          const auto result_1 = variable::power_iteration_eigth(
              matrix, x, rowcnt, comm, iter_limit);
          auto end_1 = high_resolution_clock::now();
          converge[1] = std::get<2>(result_1).back();
          if (rank == 0 && converge[1]) {
            print_variable("eigths", result_1,
                           duration_cast<nanoseconds>(end_1 - start_1).count());
          }
        }

        if (converge[2]) {
          auto start_2 = high_resolution_clock::now();
          const auto result_2 = variable::power_iteration_quarter(
              matrix, x, rowcnt, comm, iter_limit);
          auto end_2 = high_resolution_clock::now();
          converge[2] = std::get<2>(result_2).back();
          if (rank == 0 && converge[2]) {
            print_variable("quarter", result_2,
                           duration_cast<nanoseconds>(end_2 - start_2).count());
          }
        }

        if (converge[3]) {
          auto start_3 = high_resolution_clock::now();
          const auto result_3 = variable::power_iteration_half(
              matrix, x, rowcnt, comm, iter_limit);
          auto end_3 = high_resolution_clock::now();
          converge[3] = std::get<2>(result_3).back();
          if (rank == 0 && converge[3]) {
            print_variable("half", result_3,
                           duration_cast<nanoseconds>(end_3 - start_3).count());
          }
        }

        if (!(converge[0] || converge[1] || converge[2] || converge[3])) {
          if (rank == 0) {
            std::cout << "Breaking\n\n";
          }
          broken = true;
          ++broken_cnt;
          break;
        } else {
          if (rank == 0) {
            std::cout << std::endl;
          }
        }
      }
      if (!broken) {
        ++instance_cnt;
      }
    }
  }
}

}
#endif // CODE_PERFORMANCE_TESTS_H
