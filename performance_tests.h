//
// Created by niko on 9/24/19.
//

#ifndef CODE_PERFORMANCE_TESTS_H
#define CODE_PERFORMANCE_TESTS_H

#include <chrono>

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
                double density) {
  using slice_type = seg::Double_slice<0, end_idx>;
  auto slice_vec = to_slice_vector<end_idx>(x);
  std::vector<slice_type> result_vec(matrix.num_rows());

  double sum{0};
  for (int i{0}; i < 50; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    spmv(matrix, slice_vec, std::begin(result_vec), std::end(result_vec));
    auto end = std::chrono::high_resolution_clock::now();

    sum += result_vec.at(i % result_vec.size()).to_double();
    print_result(
        end_idx + 1, matrix.num_cols(), matrix.num_rows(), density,
        matrix.num_values(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count());
  }
  std::cout << "IGNORE sum " << sum << std::endl;
}

}

namespace perf_test {

void spmv_single_node() {
  std::mt19937 rng{std::random_device{}()};
  std::uniform_real_distribution<> val_distrib(0., 100'000);

  constexpr unsigned min_width = 1u << 4u;
  constexpr unsigned max_width = (1u << 5u) + 1;
  constexpr unsigned min_height = 1u << 4u;
  constexpr unsigned max_height = (1u << 5u) + 1;
  constexpr unsigned min_density_fac = 1;
  constexpr unsigned max_density_fac = 5 + 1;

  std::cout << "<slice_length> <width> <height> <density> <num_values> <time (ns)>\n";
  for (unsigned width = min_width; width < max_width; width *= 2) {
    for (unsigned height = min_height; height < max_height; height *= 2) {
      std::vector<double> x;
      for (unsigned i = 0; i < width; ++i) {
        x.push_back(val_distrib(rng));
      }

      double sum{0};
      for (unsigned density_fac = min_density_fac;
           density_fac < max_density_fac; ++density_fac) {
        const auto density = density_fac * 0.1;
        CSR matrix = CSR::random(width, height, density, rng);

        for (int i{0}; i < 50; ++i) {
          auto start = std::chrono::high_resolution_clock::now();
          auto result = matrix.spmv(x);
          auto end = std::chrono::high_resolution_clock::now();

          sum += result.at(i % result.size());
          print_result(
              -1, width, height, density, matrix.num_values(),
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count());
        }

        time_spmv_slice<0>(matrix, x, density);
        time_spmv_slice<1>(matrix, x, density);
        time_spmv_slice<2>(matrix, x, density);
        time_spmv_slice<3>(matrix, x, density);
        time_spmv_slice<4>(matrix, x, density);
        time_spmv_slice<5>(matrix, x, density);
        time_spmv_slice<6>(matrix, x, density);
        time_spmv_slice<7>(matrix, x, density);
      }
      std::cout << "IGNORE sum " << sum << std::endl;
    }
  }
}

void power_iteration_segmented_test() {
  // generate matrices: different size, density
  //
}

}
#endif // CODE_PERFORMANCE_TESTS_H
