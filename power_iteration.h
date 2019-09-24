//
// Created by niko on 8/14/19.
//

#ifndef CODE_POWER_ITERATION_H
#define CODE_POWER_ITERATION_H

#include <mpi.h>
#include <vector>

#include "matrix_formats/csr.hpp"
#include "segmentation_char/segmentation_char.h"
#include "spmv.h"

std::pair<std::vector<double>, bool> power_iteration(const CSR& matrix, const std::vector<double>& x);

std::tuple<std::vector<double>, int, unsigned, bool> power_iteration(const CSR&matrix_slice, const std::vector<double>& x, const std::vector<int>& rowcnt, MPI_Comm comm);

std::tuple<std::vector<double>, unsigned, bool> power_iteration_fixed(const CSR&matrix_slice, const std::vector<double>&x, const std::vector<int>& rowcnt, MPI_Comm comm);

std::vector<double> power_iteration_variable(const CSR& matrix_slice,
    const std::vector<double>& x, const std::vector<int>& rowcnt,
    MPI_Comm comm, int iteration_limit = 1000);

template <int end>
std::tuple<std::vector<Double_slice<0, end>>, int, bool>
    power_iteration_segmented(
        const CSR& matrix_slice,
        const std::vector<Double_slice<0, end>>& x,
        const std::vector<int>& rowcnt,
        MPI_Comm comm,
        int iteration_limit = 1000) {
  static_assert(sizeof(Double_slice<0, end>[4]) == 4 * sizeof(Double_slice<0, end>),
      "No padding between Double_slice instances allowed");
  static_assert(sizeof(Double_slice<0, end>[3]) == 3 * sizeof(Double_slice<0, end>),
                "No padding between Double_slice instances allowed");

  unsigned rank, comm_size;
  {
    for (auto e : rowcnt) {
      (void)e;
      assert(e >= 0);
    }
    assert(x.size() == matrix_slice.num_cols());

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
    rank = static_cast<unsigned>(s_rank);
    comm_size = static_cast<unsigned>(s_comm_size);

    assert(matrix_slice.num_rows() == static_cast<unsigned>(rowcnt.at(rank)));
    assert(rowcnt.size() == comm_size);
  }
  // prefix sum of rowcnt to get recvdispls for MPI_Alltoallv
  std::vector<int> recvdispls(rowcnt.size());
  recvdispls.at(0) = 0;
  std::partial_sum(rowcnt.begin(), rowcnt.end() - 1, recvdispls.begin() + 1);

  std::vector<Double_slice<0, end>> old_result(x.size());
  std::vector<Double_slice<0, end>> new_result = x;
  std::vector<Double_slice<0, end>> partial_result(matrix_slice.num_rows());

  const std::vector<int> sendcounts(comm_size, rowcnt.at(rank));
  const std::vector<int> sdispls(comm_size, 0);

  std::vector<int> char_sendcnts;
  for (auto s : sendcounts) {
    char_sendcnts.push_back(s * sizeof(Double_slice<0, end>));
  }
  std::vector<int> char_sdispls;
  for (auto s : sdispls) {
    char_sdispls.push_back(s * sizeof(Double_slice<0, end>));
  }
  std::vector<int> char_recvcnt;
  for (auto r : rowcnt) {
    char_recvcnt.push_back(r * sizeof(Double_slice<0, end>));
  }
  std::vector<int> char_recvdispls;
  for (auto r : recvdispls) {
    char_recvdispls.push_back(r * sizeof(Double_slice<0, end>));
  }

  bool done{false};
  int i = 0;
  do {
    std::swap(old_result, new_result);

    spmv(matrix_slice, new_result, partial_result.begin(), partial_result.end());

    MPI_Alltoallv(partial_result.data(), char_sendcnts.data(), char_sdispls.data(),
        MPI_BYTE, new_result.data(), char_recvcnt.data(), char_recvdispls.data(),
        MPI_BYTE, comm);

    auto square_sum = std::accumulate(new_result.begin(), new_result.end(), 0.,
      [](double curr, Double_slice<0, end> ds){
      double d = ds.to_double();
      return curr + d * d;
    });
    auto norm_fac = sqrt(square_sum);
    for (size_t k = 0; k < new_result.size(); ++k) {
      double old_val = new_result.at(k).to_double();
      new_result.at(k) = Double_slice<0, end>{old_val / norm_fac};
    }

    const auto new_result_char = reinterpret_cast<unsigned char*>(new_result.data());
    const auto old_result_char = reinterpret_cast<unsigned char*>(old_result.data());
    done = std::equal(new_result_char, new_result_char + (sizeof(Double_slice<0, end>) * new_result.size()),
        old_result_char);
    ++i;
  } while(!done && i < iteration_limit);

  return std::make_tuple(new_result, i, done);
}
#endif // CODE_POWER_ITERATION_H
