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

template <int end, bool only_changed=false>
std::tuple<std::vector<seg::Double_slice<0, end>>, int, bool>
    power_iteration_segmented(
        const CSR& matrix_slice,
        const std::vector<seg::Double_slice<0, end>>& x,
        const std::vector<int>& rowcnt,
        MPI_Comm comm,
        int iteration_limit = 1000) {
  using slice_type = seg::Double_slice<0, end>;
  constexpr int slice_size = sizeof(slice_type);

  static_assert(sizeof(slice_type[4]) == 4 * slice_size,
      "No padding between Double_slice instances allowed");
  static_assert(sizeof(slice_type[3]) == 3 * slice_size,
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

  std::vector<slice_type> old_result(x.size());
  std::vector<slice_type> new_result = x;
  std::vector<slice_type> partial_result(matrix_slice.num_rows());

  const std::vector<int> sendcounts(comm_size, rowcnt.at(rank));
  const std::vector<int> sdispls(comm_size, 0);

  std::vector<int> char_sendcnts;
  for (auto s : sendcounts) {
    char_sendcnts.push_back(s * slice_size);
  }
  std::vector<int> char_recvcnt;
  for (auto r : rowcnt) {
    char_recvcnt.push_back(r * slice_size);
  }
  std::vector<int> char_recvdispls;
  for (auto r : recvdispls) {
    char_recvdispls.push_back(r * slice_size);
  }

  bool done{false};
  int i = 0;
  do {
    std::swap(old_result, new_result);

    spmv(matrix_slice, new_result, partial_result.begin(), partial_result.end());

    if (only_changed) {
      // TODO: send only those bytes that have been changed in this iteration
      // determine changed bytes
      const unsigned row_offset = recvdispls.at(rank);
      int local_max_index{slice_size};
      for (int k{0}; k < partial_result.size(); ++k) {
        local_max_index = std::max(
            local_max_index,
            partial_result.at(k).compare_bytes(new_result.at(row_offset + k)));
      }
      if (local_max_index == slice_size) {
        // All Bytes are different, just send full doubles
        MPI_Allgatherv(partial_result.data(), char_sendcnts.at(rank), MPI_BYTE,
                       new_result.data(), char_recvcnt.data(),
                       char_recvdispls.data(), MPI_BYTE, comm);
      } else {
        // aggregate vector<char> with only those bytes
        const int local_num_bytes = partial_result.size()
            * (slice_size - local_max_index);
        std::vector<unsigned char> local_changed_bytes;
        local_changed_bytes.reserve(local_num_bytes);
        // TODO: copy of bytes as constexpr?

        for (auto ds : partial_result) {
          local_changed_bytes.insert(local_changed_bytes.end(),
              ds.get_bytes() + local_max_index, ds.get_bytes() + slice_size);
        }

        // MPI_Allgather to distribute the number of bytes
        //std::vector<int> num_bytes(comm_size);
        std::vector<int> max_index(comm_size);
        MPI_Allgather(&local_max_index, 1, MPI_INT, max_index.data(),
            1, MPI_INT, comm);

        std::vector<int> num_bytes;
        for (unsigned k{0}; k < comm_size; ++k) {
          const auto chg_bytes_cnt = slice_size - max_index.at(k);
          num_bytes.push_back(rowcnt.at(k) * chg_bytes_cnt);
        }

        const int total_bytes = std::accumulate(num_bytes.begin(), num_bytes.end(), 0);
        std::vector<unsigned char> changed_bytes(total_bytes);

        std::vector<int> changed_bytes_displs(rowcnt.size());
        changed_bytes_displs.at(0) = 0;
        // TODO: is this okay?
        std::partial_sum(num_bytes.begin(), num_bytes.end() - 1, changed_bytes_displs.begin() + 1);

        // Do MPI_Allgatherv on the bytes
        MPI_Allgatherv(local_changed_bytes.data(), num_bytes.at(rank), MPI_BYTE,
            changed_bytes.data(), num_bytes.data(), changed_bytes_displs.data(),
            MPI_BYTE, comm);

        // insert the new bytes as needed into new_result
        for (unsigned p{0}; p < comm_size; ++p) {
          const auto m = max_index.at(p);
          const auto chg_bytes_cnt = slice_size - max_index.at(p);
          const auto offset = recvdispls.at(p);

          for (int k{0}; k < rowcnt.at(p); ++k) {
            // index avoids asking for recvdispls.at(comm_size)
            const auto index = k + offset;
            const auto first = changed_bytes.data() + k * chg_bytes_cnt + changed_bytes_displs.at(p);
            const auto last = first + chg_bytes_cnt;
            const auto out = new_result.at(index).get_bytes() + m;
            std::copy(first, last, out);
          }
        }
      }
    } else {
      MPI_Allgatherv(partial_result.data(), char_sendcnts.at(rank), MPI_BYTE,
                     new_result.data(), char_recvcnt.data(),
                     char_recvdispls.data(), MPI_BYTE, comm);
    }

    auto square_sum = std::accumulate(new_result.begin(), new_result.end(), 0.,
      [](double curr, slice_type ds) {
      double d = ds.to_double();
      return curr + d * d;
    });
    auto norm_fac = sqrt(square_sum);
    for (size_t k = 0; k < new_result.size(); ++k) {
      double old_val = new_result.at(k).to_double();
      new_result.at(k) = slice_type{old_val / norm_fac};
    }

    const auto new_result_char = reinterpret_cast<unsigned char*>(new_result.data());
    const auto old_result_char = reinterpret_cast<unsigned char*>(old_result.data());
    done = std::equal(new_result_char, new_result_char + (slice_size * new_result.size()),
        old_result_char);
    ++i;
  } while(!done && i < iteration_limit);

  return std::make_tuple(new_result, i, done);
}

#endif // CODE_POWER_ITERATION_H
