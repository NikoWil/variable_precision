//
// Created by niko on 8/14/19.
//

#include <cassert>
#include <cmath>
#include <numeric>

#include "power_iteration.h"
#include "segmentation/segmentation.h"

const unsigned iteration_limit = 10000;
const double eps_scaling = 1 << 7;

std::vector<double> power_iteration(const CSR& matrix, const std::vector<double>& x) {
  auto new_result = x;
  std::vector<double> old_result;

  bool done = false;
  unsigned i = 0;
  while(!done && /*old_result != new_result &&*/ i < iteration_limit) {
    old_result = new_result;
    new_result = matrix.spmv(old_result);

    auto square_sum = std::accumulate(new_result.begin(), new_result.end(), 0., [](double curr, double d){ return curr + d * d; });
    auto norm_fac = sqrt(square_sum);
    std::for_each(new_result.begin(), new_result.end(), [norm_fac](double& d) { d /= norm_fac; });

    double diff{};
    for (size_t k = 0; k < new_result.size(); ++k) {
      diff += std::abs(new_result.at(k) - old_result.at(k));
    }
    diff /= new_result.size();
    done = diff < std::numeric_limits<double>::epsilon() / eps_scaling;

    ++i;
  }

  std::cout << "single node: " << i << " iterations\n";
  return new_result;
}

std::vector<double> power_iteration(const CSR& matrix_slice, const std::vector<double>& x, const std::vector<int>& rowcnt, MPI_Comm comm) {
  for (auto e : rowcnt) {
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
  unsigned rank{static_cast<unsigned>(s_rank)};
  unsigned comm_size{static_cast<unsigned >(s_comm_size)};

  assert(matrix_slice.num_rows() == static_cast<unsigned>(rowcnt.at(rank)));
  assert(rowcnt.size() == comm_size);

  // prefix sum of rowcnt to get displs for MPI_Alltoallv
  std::vector<int> displs(rowcnt.size());
  displs.at(0) = 0;
  std::partial_sum(rowcnt.begin(), rowcnt.end() - 1, displs.begin() + 1);

  std::vector<double> old_result;
  std::vector<double> new_result = x;

  const std::vector<int> sendcounts(comm_size, rowcnt.at(rank));
  const std::vector<int> sdispls(comm_size, 0);

  bool half_precision = true;
  bool done = false;
  unsigned i = 0;
  while(!done && i < iteration_limit) {
    old_result = new_result;

    auto partial_result = matrix_slice.spmv(old_result);

    if (half_precision) {
      std::vector<uint32_t> partial_result_heads;
      for (const auto e : partial_result) {
        partial_result_heads.push_back(get_head(e));
      }
      std::vector<uint32_t> result_heads(x.size());

      MPI_Alltoallv(partial_result_heads.data(), sendcounts.data(), sdispls.data(), MPI_UINT32_T, result_heads.data(), rowcnt.data(), displs.data(), MPI_UINT32_T, comm);

      for (size_t k = 0; k < result_heads.size(); ++k) {
        new_result.at(k) = fill_head(result_heads.at(k));
      }
    } else {
      MPI_Alltoallv(partial_result.data(), sendcounts.data(), sdispls.data(), MPI_DOUBLE, new_result.data(), rowcnt.data(), displs.data(), MPI_DOUBLE, comm);
    }

    auto square_sum = std::accumulate(new_result.begin(), new_result.end(), 0., [](double curr, double d){ return curr + d * d; });
    auto norm_fac = sqrt(square_sum);
    std::for_each(new_result.begin(), new_result.end(), [norm_fac](double& d) { d /= norm_fac; });

    double diff{};
    for (size_t k = 0; k < new_result.size(); ++k) {
      diff += std::abs(new_result.at(k) - old_result.at(k));
    }
    diff /= new_result.size();

    const double half_epsilon = fill_head(0x3ff00001) - 1;

    if (half_precision && diff < half_epsilon / eps_scaling) {
      std::cout << "Rank " << rank << ", iteration " << i << ", switching precision\n";
      MPI_Barrier(comm);
      if (rank == 0) {
        print_vector(old_result, "intermediate result");
      }
      half_precision = false;
    } else if (!half_precision && diff < std::numeric_limits<double>::epsilon() / eps_scaling) {
      std::cout << "Rank " << rank << ", iteration " << i << ", done\n";
      MPI_Barrier(comm);
      done = true;
    }
    MPI_Barrier(comm);
    ++i;
  }

  if (rank == 0) {
    std::cout << "Variable precision power iteration, " << i << " iterations" << std::endl;
  }
  return new_result;
}

std::vector<double> power_iteration_fixed(const CSR& matrix_slice, const std::vector<double>&x, const std::vector<int>& rowcnt, MPI_Comm comm) {
  for (auto e : rowcnt) {
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
  unsigned rank{static_cast<unsigned>(s_rank)};
  unsigned comm_size{static_cast<unsigned >(s_comm_size)};

  assert(matrix_slice.num_rows() == static_cast<unsigned>(rowcnt.at(rank)));
  assert(rowcnt.size() == comm_size);

  // prefix sum of rowcnt to get displs for MPI_Alltoallv
  std::vector<int> displs(rowcnt.size());
  displs.at(0) = 0;
  std::partial_sum(rowcnt.begin(), rowcnt.end() - 1, displs.begin() + 1);

  std::vector<double> old_result;
  std::vector<double> new_result = x;

  const std::vector<int> sendcounts(comm_size, rowcnt.at(rank));
  const std::vector<int> sdispls(comm_size, 0);

  bool done = false;
  unsigned i = 0;
  while(!done && /*old_result != new_result &&*/ i < iteration_limit) {
    old_result = new_result;

    auto partial_result = matrix_slice.spmv(old_result);
    MPI_Alltoallv(partial_result.data(), sendcounts.data(), sdispls.data(), MPI_DOUBLE, new_result.data(), rowcnt.data(), displs.data(), MPI_DOUBLE, comm);

    auto square_sum = std::accumulate(new_result.begin(), new_result.end(), 0., [](double curr, double d){ return curr + d * d; });
    auto norm_fac = sqrt(square_sum);
    std::for_each(new_result.begin(), new_result.end(), [norm_fac](double& d) { d /= norm_fac; });

    double diff{};
    for (size_t k = 0; k < new_result.size(); ++k) {
      diff += std::abs(new_result.at(k) - old_result.at(k));
    }
    diff /= new_result.size();
    done = diff < std::numeric_limits<double>::epsilon() / eps_scaling;
    ++i;
  }

  if (rank == 0) {
    std::cout << "Fixed precision power iteration, " << i << " iterations" << std::endl;
  }

  return new_result;
}