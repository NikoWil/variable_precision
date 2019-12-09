#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <random>
#include <vector>

#include "communication.h"
#include "linalg/power_iteration/poweriteration.h"
#include "matrix_formats/csr.hpp"
#include "performance_tests.h"
#include "power_iteration.h"
#include "seg_char.h"
#include "seg_uint.h"
#include "util/util.hpp"

void benchmark_spmv(unsigned size, double density, unsigned iterations);

int main(int argc, char* argv[]) {
  const auto requested = MPI_THREAD_FUNNELED;
  int provided;
  MPI_Init_thread(&argc, &argv, requested, &provided);
  if (provided < requested) {
    std::cout << "No sufficient MPI multithreading support found\n";
    return 0;
  }

  std::mt19937 rng{std::random_device{}()};
  std::cout << std::setprecision(20);

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  /*
  const unsigned size{1u << 17u}; // 32768 = 2^15
  const CSR matrix = CSR::diagonally_dominant(size, 0.01, rng);
  std::uniform_real_distribution<> distribution (0, 1'000'000);
  std::vector<double> x(size);
  for (size_t i{0}; i < x.size(); ++i) {
    x[i] = distribution(rng);
  }
  std::vector<uint32_t> x_halves(x.size());
  for (size_t i{0}; i < x.size(); ++i) {
    seg_uint::write_4(&x_halves[i], &x[i]);
  }
  std::vector<uint32_t> y_halves(x_halves.size());

  std::uniform_int_distribution<> index_distribution(0, size);
  uint32_t sum{0};
  for (int i{0}; i < 100; ++i) {
    seg_uint::spmv_4(matrix, x_halves, y_halves);
    sum += y_halves[index_distribution(rng)];
  }
  std::cout << sum << "\n";
  */
  const unsigned n{30};

  std::vector<int> rowcnt;
  std::vector<int> start_row{0};
  for (int i{0}; i < comm_size; ++i) {
    unsigned start = (n * i) / comm_size;
    unsigned end = (n * (i + 1)) / comm_size;
    rowcnt.push_back(end - start);

    const auto last_start = start_row.back();
    start_row.push_back(last_start + rowcnt.back());
  }

  const CSR matrix = CSR::diagonally_dominant(n, 0.3, rng);
  const CSR matrix_slice = distribute_matrix(matrix, MPI_COMM_WORLD, 0);

  const std::vector<double> initial(n, 1.);

  if (rank == 0) {
    std::vector<double> result_local;
    const auto meta_inf_local =
        local::power_iteration(matrix, initial, result_local, 1000);
    std::cout << "Local\nDone: " << std::get<0>(meta_inf_local)
              << ", iter: " << std::get<1>(meta_inf_local) << "\n";
    print_vector(result_local, "result local");
    std::cout << "\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<double> result_distrib;
  const auto meta_inf_distrib = distributed::fixed::power_iteration(matrix_slice, initial, result_distrib, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "Distributed::fixed\nDone: " << std::get<0>(meta_inf_distrib)
              << ", iter: " << std::get<1>(meta_inf_distrib) << "\n";
    print_vector(result_distrib, "result distrib::fixed");
  }

  std::vector<double> result_distrib_seg;
  const auto meta_inf_dis_seg = distributed::seg_uint::power_iteration_4(matrix_slice, initial, result_distrib_seg, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "Distributed::seg_uint\nDone: " << std::get<0>(meta_inf_dis_seg)
              << ", iter: " << std::get<1>(meta_inf_dis_seg) << "\n";
    print_vector(result_distrib_seg, "result distrib::seg_uint");
  }

/**
  // Create x, y_1_double as vector<uint16_t>
  std::uniform_real_distribution<> distrib(0, 100);
  std::vector<double> x(n);
  for (size_t i{0}; i < x.size(); ++i) {
    x[i] = distrib(rng);
  }
  std::vector<uint16_t> x_1_uint(x.size());
  std::vector<uint32_t> x_2_uint(x.size());
  std::vector<uint16_t> x_3_uint(3 * x.size());
  for (size_t i{0}; i < x.size(); ++i) {
    seg_uint::write_2(&x_1_uint[i], &x[i]);
    seg_uint::write_4(&x_2_uint[i], &x[i]);
    seg_uint::write_6(&x_3_uint[3 * i], &x[i]);
  }


  // Create output vectors
  std::vector<uint16_t> y_1_uint(x_1_uint.size());
  std::vector<uint32_t> y_2_uint(x_2_uint.size());
  std::vector<uint16_t> y_3_uint(x_3_uint.size());
  std::vector<double> y_8_double(x.size());

  // do SpmV
  seg_uint::spmv_1(matrix, x_1_uint, y_1_uint);
  seg_uint::spmv_2(matrix, x_2_uint, y_2_uint);
  seg_uint::spmv_3(matrix, x_3_uint, y_3_uint);
  fixed::spmv(matrix, x, y_8_double);

  // Convert y_1_double to double
  std::vector<double> y_1_double(y_1_uint.size());
  std::vector<double> y_2_double(y_2_uint.size());
  std::vector<double> y_3_double(y_3_uint.size() / 3);
  for (size_t i{0}; i < y_1_double.size(); ++i) {
    seg_uint::read_2(&y_1_uint[i], &y_1_double[i]);
    seg_uint::read_4(&y_2_uint[i], &y_2_double[i]);
    seg_uint::read_6(&y_3_uint[i * 3], &y_3_double[i]);
  }

  // print result
  print_vector(y_1_double, "y_1_double 2 byte precision");
  print_vector(y_2_double, "y_2_double 4 byte precision");
  print_vector(y_3_double, "y_3_double 6 byte precision");
  print_vector(y_8_double, "y_8_double 8 byte precision");
*/
  MPI_Finalize();
  return 0;
}

template <typename T>
T median(std::vector<T>& v) {
  const auto size = v.size();
  const auto index_1 = size / 2;
  const auto index_2 = (size - 1) / 2;

  std::nth_element(v.begin(), v.begin() + index_1, v.end());
  const T e1 = v[index_1];
  std::nth_element(v.begin(), v.begin() + index_2, v.end());
  const T e2 = v[index_2];

  return (e1 + e2) / 2;
}

template <typename T>
T average(const std::vector<T>& v) {
  T sum{0};
  for (const auto e : v) {
    sum += e;
  }
  return sum / v.size();
}

void benchmark_spmv(unsigned size, double density, unsigned iterations) {
  // Generate matrix & (segmented) vector
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<> value_distrib(0, 1'000'000);
  std::uniform_int_distribution<> index_distrib(0, size);

  const CSR matrix = CSR::random(size, size, density, rng);
  std::vector<double> x(size);
  std::vector<uint32_t> x_halves(size);
  for (size_t i{0}; i < size; ++i) {
    const double val = value_distrib(rng);
    x[i]  = val;
    seg_uint::write_4(&x_halves[i], &val);
  }

  std::vector<double> y(size);
  std::vector<uint32_t> y_halves(size);
  double d_sum{0};
  uint32_t u_sum{0};

  // test normal spmv timing
  unsigned warmup = 5;
  for (unsigned i{0}; i < warmup; ++i) {
    fixed::spmv(matrix, x, y);

    d_sum += y[0];
    x[index_distrib(rng)] = value_distrib(rng);
  }

  using namespace std::chrono;
  std::vector<unsigned> timings_fixed;
  for (unsigned i{0}; i < iterations; ++i) {
    const auto start = high_resolution_clock::now();
    fixed::spmv(matrix, x, y);
    const auto end = high_resolution_clock::now();
    timings_fixed.push_back(duration_cast<nanoseconds>(end - start).count());

    d_sum += y[0];
    x[index_distrib(rng)] = value_distrib(rng);
  }

  // test segmented spmv
  for (unsigned i{0}; i < warmup; ++i) {
    seg_uint::calc_convert::spmv_4(matrix, x_halves, y_halves);

    u_sum += y_halves[0];
    const double val = value_distrib(rng);
    const unsigned index = index_distrib(rng);
    seg_uint::write_4(&x_halves[index], &val);
  }

  std::vector<unsigned> timings_seg_during;
  for (unsigned i{0}; i < warmup; ++i) {
    const auto start = high_resolution_clock::now();
    seg_uint::calc_convert::spmv_4(matrix, x_halves, y_halves);
    const auto end = high_resolution_clock::now();
    timings_seg_during.push_back(duration_cast<nanoseconds>(end - start).count());

    u_sum += y_halves[0];
    const double val = value_distrib(rng);
    const unsigned index = index_distrib(rng);
    seg_uint::write_4(&x_halves[index], &val);
  }

  // test normal spmv with conversion before & after
  std::vector<unsigned> timings_seg_convert;

  std::vector<double> x_half_doubles(size);
  std::vector<double> y_half_doubles(size);
  for (unsigned i{0}; i < warmup; ++i) {
    const auto start = high_resolution_clock::now();
    seg_uint::pre_convert::spmv_4(matrix, x_halves, y_halves);
    const auto end = high_resolution_clock::now();
    timings_seg_convert.push_back(duration_cast<nanoseconds>(end - start).count());

    u_sum += y_halves[0];
    const double val = value_distrib(rng);
    const unsigned index = index_distrib(rng);
    seg_uint::write_4(&x_halves[index], &val);
  }

  // calculate average/ median timings
  const auto median_fixed = median(timings_fixed);
  const auto median_seg_during = median(timings_seg_during);
  const auto median_seg_convert = median(timings_seg_convert);

  const auto average_fixed = average(timings_fixed);
  const auto average_seg_during = average(timings_seg_during);
  const auto average_seg_convert = average(timings_seg_convert);


  // output results
  std::cout << "SpMV Timing benchmark.\nParameters:\nsize: " << size
            << ", density: " << density << ", iterations: " << iterations
            << "\n";
  std::cout << "Fixed SpMV:\n\tmedian:\t\t" << median_fixed << "\n\taverage:\t"
            << average_fixed << "\n";
  std::cout << "Segmented SpMV (conversion during SpMV):\n\tmedian:\t\t"
            << median_seg_during << "\n\taverage:\t" << average_seg_during
            << "\n";
  std::cout << "Segmented SpMV (conversion before & after SpMV):\n\tmedian:\t\t"
            << median_seg_convert << "\n\taverage:\t" << average_seg_convert
            << "\n";
}