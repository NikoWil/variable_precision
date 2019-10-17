//
// Created by niko on 9/24/19.
//

#ifndef CODE_PERFORMANCE_TESTS_H
#define CODE_PERFORMANCE_TESTS_H

#include <chrono>
#include <iostream>

#include "matrix_formats/csr.hpp"
#include "segmentation_char.h"
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
    spmv(matrix, slice_vec, result_vec.begin(), result_vec.end());
    sum += result_vec.at(index_distrib(rng)).to_double();
  }
  for (unsigned i{0}; i < num_tests; ++i) {
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    spmv(matrix, slice_vec, result_vec.begin(), result_vec.end());
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

  std::cout << "<slice_length> <width> <height> <density> <num_values> <time (ns)>\n";
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
          print_result(
              -1, width, height, density, matrix.num_values(),
              duration_cast<nanoseconds>(end - start).count());
        }
        std::cout << "TEST RUN DONE\n";

        time_spmv_slice<0>(matrix, x, density, num_tests, warmup_iterations, rng);
        time_spmv_slice<1>(matrix, x, density, num_tests, warmup_iterations, rng);
        time_spmv_slice<2>(matrix, x, density, num_tests, warmup_iterations, rng);
        time_spmv_slice<3>(matrix, x, density, num_tests, warmup_iterations, rng);
        time_spmv_slice<4>(matrix, x, density, num_tests, warmup_iterations, rng);
        time_spmv_slice<5>(matrix, x, density, num_tests, warmup_iterations, rng);
        time_spmv_slice<6>(matrix, x, density, num_tests, warmup_iterations, rng);
        time_spmv_slice<7>(matrix, x, density, num_tests, warmup_iterations, rng);
      }
      std::cout << "IGNORE sum " << sum << std::endl;
    }
  }
}

void power_iteration_segmented_test() {
  
}

}
#endif // CODE_PERFORMANCE_TESTS_H
