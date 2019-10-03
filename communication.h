//
// Created by niko on 8/20/19.
//

#ifndef CODE_COMMUNICATION_H
#define CODE_COMMUNICATION_H

#include <bitset>
#include <cmath>
#include <mpi.h>

#include "matrix_formats/csr.hpp"
#include "segmentation_char/segmentation_char.h"

void bcast_vector(std::vector<double>& v, MPI_Comm comm, int root);

CSR distribute_matrix(const CSR& matrix, MPI_Comm comm, int root);

namespace {
constexpr bool is_power_of_two(unsigned x) {
  return (x != 0) && (x & (x - 1)) == 0;
}

constexpr unsigned floor_to(unsigned n, unsigned m) {
  return n - (n % m);
}
}

template <int end>
void custom_alltoallv(const std::vector<Double_slice<0, end>>& sendbuf, const std::vector<int>& rowcnt,
                      std::vector<Double_slice<0, end>>& recvbuf, const MPI_Comm& comm) {
  int rank, comm_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  assert(is_power_of_two(comm_size) &&
         "communication.cpp custom_alltoallv works only for communicator that is power of two in size");
  assert(rowcnt.size() == static_cast<unsigned>(rank));
  assert(static_cast<unsigned>(rowcnt.at(rank)) == sendbuf.size());

  using slice_type = Double_slice<0, end>;
  int slice_size = sizeof(slice_type);

  std::vector<int> displs;
  displs.reserve(comm_size + 1);
  displs.at(0) = 0;
  for (int i{1}; i < comm_size + 1; ++i) {
    displs.at(i) = rowcnt.at(i) + displs.at(i - 1);
  }

  std::copy(std::begin(sendbuf), std::end(sendbuf), std::begin(recvbuf) + displs.at(rank));

  // TODO: stop assumption of 8 bits per char
  auto max_level = static_cast<unsigned>(std::round(std::log2(comm_size)));
  for (unsigned i{0}; i < max_level; ++i) {
    std::bitset<8 * sizeof(int)> other_rank_bits{static_cast<unsigned>(rank)};
    other_rank_bits.flip(i);
    auto other_rank = static_cast<int>(other_rank_bits.to_ulong());
    auto other_start_rank = floor_to(other_rank, 2u << i);
    auto other_end_rank = other_start_rank + (2u << i);
    auto other_num_elemens = std::accumulate(rowcnt[other_start_rank],
                                             rowcnt[other_end_rank], 0);
    auto other_start_idx = displs[other_start_rank];

    auto start_rank = floor_to(rank, 2u << i);
    auto end_rank = start_rank + (2u << i);
    auto num_elements = std::accumulate(rowcnt[start_rank], rowcnt[end_rank], 0);
    auto start_idx = displs[start_rank];

    MPI_Sendrecv(recvbuf.data() + (start_idx * slice_size), num_elements * slice_size,
        MPI_CHAR, other_rank, i, recvbuf.data() + (other_start_idx * slice_size),
        other_num_elemens * slice_size, MPI_CHAR, other_rank, i, comm, nullptr);
  }
}
#endif // CODE_COMMUNICATION_H
