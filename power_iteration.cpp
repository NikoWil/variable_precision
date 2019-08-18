//
// Created by niko on 8/14/19.
//

#include <cassert>
#include <numeric>

#include "power_iteration.h"
#include "segmentation/segmentation.h"

// TODO: copy over from other code base
//CSR distribute_matrix(const CSR& matrix, const std::vector<int>& rowcnt, MPI_Comm comm, int root);

/*
   * I: CALCULATE METADATA
   *  calculate rows per process on all nodes
   *  calculate displs/ sendcnts for all nodes
   *
   * II: DISTRIBUTE DATA
   *  distribute matrix
   *  distribute x
   *
   * III: ITERATE
   *  1. calculate local result_s = matrix_s * x_s
   *  2. distribute result_s (MPI_Alltoallv ?)
   *  3. check if result has changed from last time
   *    3.1 yes:  keep iterating
   *    3.2 no:   is more precision available?
   *      3.2.1 yes:  increase precision, keep iterating
   *      3.2.2 no:   finish
   */
std::vector<double> power_iteration(const CSR& matrix, const std::vector<double>& x, const std::vector<int>& rowcnt, MPI_Comm comm) {
  for (auto e : rowcnt) {
    assert(e >= 0);
  }
  assert(x.size() == matrix.num_cols());

  // TODO: Algorithm
  int s_rank, s_comm_size;
  MPI_Comm_rank(comm, &s_rank);
  MPI_Comm_size(comm, &s_comm_size);

  if (s_rank < 0) {
    std::cout << "Erronous value for MPI_Comm_rank: " << s_rank << "\n";
    std::exit(1);
  }
  if (s_comm_size < 0) {
    std::cout << "Erronous value for MPI_Comm_size: " << s_comm_size << "\n";
    std::exit(1);
  }
  unsigned rank{static_cast<unsigned>(s_rank)};
  unsigned comm_size{static_cast<unsigned >(s_comm_size)};

  assert(matrix.num_rows() == static_cast<unsigned>(rowcnt.at(rank)));
  assert(rowcnt.size() == comm_size);

  // prefix sum of rowcnt to get displs for MPI_Alltoallv
  std::vector<int> displs(rowcnt.size());
  displs.at(0) = 0;
  std::partial_sum(rowcnt.begin(), rowcnt.end() - 1, displs.begin() + 1);

  bool half_precision = true;
  bool done = false;
  auto old_result = x;
  std::vector<double> new_result(old_result.size());

  int i = 0;
  while ((half_precision || !done) && i < 100) {
    if (rank == 0) {
      std::cout << "Iteration: " << i << "\n";
      for (const auto e : old_result) {
        std::cout << get_head(e) << get_tail(e) << " ";
      }
      std::cout << " ";
    }
    ++i;

    auto partial_result = matrix.spmv(x);

    if (half_precision) {
      // convert partial result to half precision
      std::vector<uint32_t> front_halves;
      front_halves.reserve(partial_result.size());
      for (const auto e : partial_result) {
        front_halves.push_back(get_head(e));
      }

      // distribute half precision values
      std::vector<uint32_t> result_half_precision(old_result.size());
      MPI_Alltoallv(front_halves.data(), rowcnt.data(), displs.data(),
                    MPI_UINT32_T, result_half_precision.data(), rowcnt.data(),
                    displs.data(), MPI_UINT32_T, comm);

      // convert back to double
      new_result.clear();
      for (const auto e : result_half_precision) {
        new_result.push_back(fill_head(e));
      }

      if (old_result == new_result) {
        std::cout << "Switching precision\n";
        half_precision = false;
      }
      if (rank == 0) {
        print_vector(new_result, "current");
      }

    } else {
      // distribute double
      MPI_Alltoallv(partial_result.data(), rowcnt.data(), displs.data(),
                    MPI_DOUBLE, new_result.data(), rowcnt.data(), displs.data(),
                    MPI_DOUBLE, comm);
      if (old_result == new_result) {
        done = true;
      }
    }

    // normalize vector
    double sum = std::accumulate(new_result.begin(), new_result.end(), 0.);
    std::for_each(new_result.begin(), new_result.end(), [sum](double &n){ n /= sum; });

    if (old_result == new_result) {
      if (half_precision) {
        std::cout << "Switching precision\n";
        half_precision = false;
      } else {
        done = true;
      }
    }

    old_result = new_result;
  }

  return old_result;
}