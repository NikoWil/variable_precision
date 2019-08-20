//
// Created by niko on 8/20/19.
//

#include "communication.h"

void bcast_vector(std::vector<double>& v, MPI_Comm comm, int root) {
  int rank, comm_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  int size;
  if (rank == root) {
    size = v.size();
  }
  MPI_Bcast(&size, 1, MPI_INT, root, comm);

  if (rank != root) {
    v.clear();
    v.resize(size);
  }

  MPI_Bcast(v.data(), size, MPI_DOUBLE, root, comm);
}

CSR distribute_matrix(const CSR& matrix, MPI_Comm comm, int root) {
  int rank, comm_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  const int metadata_size = 3;

  std::vector<int> r_num_rows;
  std::vector<int> r_num_values;
  std::vector<int> r_stretched_rowptr;
  std::vector<unsigned> r_metadata;

  std::vector<int> r_rowptr_displs;
  std::vector<int> r_values_displs;

  if (rank == root) {
    r_num_rows.resize(comm_size);
    r_num_values.resize(comm_size);
    for (int i = 0; i < comm_size; ++i) {
      unsigned start = (matrix.num_rows() * i) / comm_size;
      unsigned end = (matrix.num_rows() * (i + 1)) / comm_size;

      r_num_rows.at(i) = end - start;
      r_num_values.at(i) = matrix.rowptr().at(end) - matrix.rowptr().at(start);
    }

    // compute prefix sums
    r_rowptr_displs.resize(comm_size);
    r_rowptr_displs.at(0) = 0;
    std::partial_sum(r_num_rows.begin(), r_num_rows.end() - 1, r_rowptr_displs.begin() + 1);

    r_values_displs.resize(comm_size);
    r_values_displs.at(0) = 0;
    std::partial_sum(r_num_values.begin(), r_num_values.end() - 1, r_values_displs.begin() + 1);

    // The rowptr of a CSR matrix with n rows contains n+1 elements
    for (auto& e : r_num_rows) {
      e++;
    }

    // Duplicate indices, as MPI Scatterv does not guarantee correct behaviour
    // in case of overlapping send buffers
    r_stretched_rowptr = duplicate_indices(matrix.rowptr(), r_rowptr_displs);

    for (size_t i = 0; i < r_rowptr_displs.size(); ++i) {
      r_rowptr_displs.at(i) += i;
    }

    r_metadata.resize(comm_size * metadata_size);
    for (int i = 0; i < comm_size; ++i) {
      auto start = i * metadata_size;
      r_metadata.at(start + 0) = matrix.num_cols();
      r_metadata.at(start + 1) = r_num_rows.at(i);
      r_metadata.at(start + 2) = r_num_values.at(i);
    }
  }

  std::vector<unsigned> metadata(metadata_size);
  MPI_Scatter(r_metadata.data(), 3, MPI_UNSIGNED, metadata.data(), 3, MPI_UNSIGNED, root, comm);

  std::vector<int> rowptr(metadata.at(1));
  std::vector<int> colidx(metadata.at(2));
  std::vector<double> values(metadata.at(2));

  MPI_Scatterv(r_stretched_rowptr.data(), r_num_rows.data(), r_rowptr_displs.data(), MPI_INT, rowptr.data(), rowptr.size(), MPI_INT, root, comm);
  MPI_Scatterv(matrix.colidx().data(), r_num_values.data(), r_values_displs.data(), MPI_INT, colidx.data(), metadata.at(2), MPI_INT, root, comm);
  MPI_Scatterv(matrix.values().data(), r_num_values.data(), r_values_displs.data(), MPI_DOUBLE, values.data(), metadata.at(2), MPI_DOUBLE, root, comm);

  // A node might own no row, if #nodes > #rows
  if (!rowptr.empty()) {
    int displacement = rowptr.at(0);

    for (auto &e : rowptr) {
      e -= displacement;
    }
  }

  return CSR{values, colidx, rowptr, metadata.at(0)};
}